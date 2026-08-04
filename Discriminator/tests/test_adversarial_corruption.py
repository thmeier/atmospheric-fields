import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xarray as xr
from omegaconf import OmegaConf

from Discriminator.scripts.train_adversarial_corruption import (
    CoarseResidualUNet,
    ERA5TemperatureDataset,
    binary_auroc,
    constrain_residual,
    cosine_latitude_weights,
    differentiable_scwd,
    differentiable_zonal_log_spectrum_distance,
    discriminator_loss,
    effective_area_penalty,
    generator_loss,
    peak_penalty,
    residual_statistics,
    select_case_study_indices,
)


class TinyDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(1, 4, 3, padding=1), nn.SiLU())
        self.head = nn.Linear(4, 1)

    def forward(self, values):
        features = self.layers(values).mean(dim=(-2, -1))
        return self.head(features)


class AdversarialCorruptionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.latitudes = np.linspace(-75.0, 75.0, 9)
        self.latitude_weights = cosine_latitude_weights(self.latitudes)

    def test_generator_returns_fixed_rms_zero_mean_residual(self):
        model = CoarseResidualUNet(base_channels=4, coarsening=2)
        clean = torch.randn(3, 1, 9, 16)
        fake, residual = model(clean, self.latitude_weights, 0.1)
        self.assertEqual(fake.shape, clean.shape)
        weights = self.latitude_weights[None, None, :, None]
        weighted_mean = torch.sum(weights * residual, dim=(-2, -1))
        weighted_mean /= torch.sum(self.latitude_weights) * residual.shape[-1]
        rms, effective_fraction, peak_ratio = residual_statistics(
            residual, self.latitude_weights
        )
        self.assertTrue(torch.allclose(weighted_mean, torch.zeros_like(weighted_mean), atol=2e-6))
        self.assertTrue(torch.allclose(rms, torch.full_like(rms, 0.1), atol=2e-5))
        self.assertTrue(torch.all(torch.isfinite(effective_fraction)))
        self.assertTrue(torch.all(torch.isfinite(peak_ratio)))

    def test_constraint_penalties_are_finite_and_differentiable(self):
        raw = torch.randn(2, 1, 9, 16, requires_grad=True)
        residual = constrain_residual(raw, self.latitude_weights, 0.1)
        loss = peak_penalty(residual, 0.1, 3.0)
        loss += effective_area_penalty(residual, self.latitude_weights, 0.2)
        loss.backward()
        self.assertIsNotNone(raw.grad)
        self.assertTrue(torch.all(torch.isfinite(raw.grad)))

    def test_metric_surrogates_backpropagate(self):
        clean = torch.randn(6, 1, 9, 16)
        candidate = (clean + 0.05 * torch.randn_like(clean)).requires_grad_()
        n_pixels = clean.shape[-2] * clean.shape[-1]
        anchor_indices = torch.arange(12)
        sparse_indices = torch.stack((anchor_indices, anchor_indices))
        weights = torch.sparse_coo_tensor(
            sparse_indices,
            torch.ones(12),
            size=(12, n_pixels),
            check_invariants=False,
        ).coalesce()
        scwd = differentiable_scwd(candidate, clean, weights, n_quantiles=8)
        spectrum = differentiable_zonal_log_spectrum_distance(
            candidate, clean, self.latitude_weights
        )
        (scwd + spectrum).backward()
        self.assertGreater(float(scwd.detach()), 0.0)
        self.assertGreater(float(spectrum.detach()), 0.0)
        self.assertIsNotNone(candidate.grad)
        self.assertGreater(float(torch.linalg.vector_norm(candidate.grad)), 0.0)

    def test_one_discriminator_optimizer_step(self):
        discriminator = TinyDiscriminator()
        optimizer = torch.optim.AdamW(discriminator.parameters(), lr=1e-3)
        clean = torch.randn(4, 1, 9, 16)
        fake = clean + 0.1
        before = [parameter.detach().clone() for parameter in discriminator.parameters()]
        loss, accuracy, _, _ = discriminator_loss(discriminator, clean, fake)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(accuracy))
        self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, discriminator.parameters())))

    def test_combined_generator_loss_backpropagates(self):
        cfg = OmegaConf.create(
            {
                "adversarial_corruption": {
                    "scwd_quantiles": 8,
                    "scwd_order": 2.0,
                    "minimum_effective_area_fraction": 0.2,
                    "maximum_peak_ratio": 3.0,
                    "metric_loss_weight": 1.0,
                    "detectability_weight": 1.0,
                    "constraint_weight": 10.0,
                    "fake_logit_margin": 2.0,
                }
            }
        )
        generator = CoarseResidualUNet(base_channels=4, coarsening=2)
        discriminator = TinyDiscriminator()
        clean = torch.randn(6, 1, 9, 16)
        reference = torch.randn_like(clean)
        fake, residual = generator(clean, self.latitude_weights, 0.1)
        anchor_indices = torch.arange(12)
        weights = torch.sparse_coo_tensor(
            torch.stack((anchor_indices, anchor_indices)),
            torch.ones(12),
            size=(12, clean.shape[-2] * clean.shape[-1]),
            check_invariants=False,
        ).coalesce()
        loss, _ = generator_loss(
            cfg,
            discriminator,
            fake,
            reference,
            residual,
            self.latitude_weights,
            weights,
            {"scwd": 1.0, "spectrum": 1.0},
            0.1,
        )
        loss.backward()
        gradients = [parameter.grad for parameter in generator.parameters()]
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(value is not None for value in gradients))
        self.assertTrue(all(torch.all(torch.isfinite(value)) for value in gradients if value is not None))

    def test_era5_dataset_respects_ranges_and_normalization(self):
        values = np.arange(12 * 9 * 16, dtype=np.float32).reshape(12, 9, 16)
        dataset = xr.Dataset(
            {"2m_temperature": (("time", "latitude", "longitude"), values)},
            coords={
                "time": pd.date_range("2018-01-01", periods=12, freq="6h"),
                "latitude": self.latitudes,
                "longitude": np.arange(16) * 22.5,
            },
        )
        selected = ERA5TemperatureDataset(
            dataset,
            "2m_temperature",
            ["2018-01-02", "2018-01-02 18:00"],
            float(values.mean()),
            float(values.std()),
        )
        self.assertEqual(len(selected), 4)
        self.assertEqual(selected[0].shape, (1, 9, 16))

        with tempfile.TemporaryDirectory() as directory:
            model = CoarseResidualUNet(base_channels=4, coarsening=2)
            path = Path(directory) / "generator.pth"
            torch.save(model.state_dict(), path)
            restored = CoarseResidualUNet(base_channels=4, coarsening=2)
            restored.load_state_dict(torch.load(path, weights_only=True))
            sample = selected[0].unsqueeze(0)
            model.eval()
            restored.eval()
            with torch.no_grad():
                expected = model(sample, self.latitude_weights, 0.1)[0]
                actual = restored(sample, self.latitude_weights, 0.1)[0]
            self.assertTrue(torch.equal(expected, actual))

    def test_binary_auroc_handles_tied_scores(self):
        self.assertAlmostEqual(binary_auroc([0.0, 0.0], [0.0, 0.0]), 0.5)
        self.assertAlmostEqual(binary_auroc([1.0, 2.0], [-2.0, -1.0]), 1.0)

    def test_case_studies_span_fake_logit_quantiles(self):
        choices = select_case_study_indices(
            [-4.0, -2.0, -1.0, 0.0, 1.0, 3.0], 4, [0.0, 0.33, 0.67, 1.0]
        )
        self.assertEqual([index for index, _ in choices], [0, 2, 3, 5])
        self.assertEqual([quantile for _, quantile in choices], [0.0, 0.33, 0.67, 1.0])


if __name__ == "__main__":
    unittest.main()
