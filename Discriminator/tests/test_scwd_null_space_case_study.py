import unittest

import numpy as np
import scipy.sparse as sparse

from Discriminator.scripts.plot_scwd_null_space_case_study import (
    area_weighted_mean,
    area_weighted_rms,
    center_and_scale,
    checkerboard_seed,
    deranged_indices,
    filter_responses,
    local_response_w2,
    near_null_pattern,
    per_sample_area_weighted_rms,
    project_into_null_space,
    splice_hemispheres,
)


class SCWDNullSpaceCaseStudyTest(unittest.TestCase):
    def setUp(self):
        self.latitudes = np.linspace(-75.0, 75.0, 7)
        self.shape = (7, 12)
        rng = np.random.default_rng(4)
        self.operator = sparse.csr_matrix(rng.normal(size=(18, np.prod(self.shape))))

    def test_projection_is_mean_preserving_and_in_operator_null_space(self):
        seed = checkerboard_seed(self.shape, block_pixels=2)
        projected, diagnostics = project_into_null_space(
            self.operator, seed, self.latitudes, atol=1e-12, btol=1e-12, maxiter=500
        )
        relative_response = np.linalg.norm(self.operator @ projected.ravel()) / np.linalg.norm(seed)
        self.assertLess(relative_response, 1e-9)
        self.assertLess(abs(area_weighted_mean(projected, self.latitudes)), 1e-10)
        self.assertLess(diagnostics["relative_response_residual"], 1e-9)
        self.assertGreater(diagnostics["retained_seed_norm_fraction"], 0.1)

    def test_scaling_attains_requested_rms_without_changing_responses(self):
        seed = checkerboard_seed(self.shape, block_pixels=1)
        projected, _ = project_into_null_space(
            self.operator, seed, self.latitudes, atol=1e-12, btol=1e-12, maxiter=500
        )
        direction = center_and_scale(projected, self.latitudes, 1.0)
        clean = np.random.default_rng(8).normal(size=(6, *self.shape))
        clean_response = filter_responses(clean, self.operator)
        for scale in (0.05, 0.10, 0.20):
            perturbation = direction * scale
            perturbed_response = filter_responses(clean + perturbation, self.operator)
            self.assertAlmostEqual(area_weighted_rms(perturbation, self.latitudes), scale, places=10)
            self.assertTrue(np.allclose(perturbed_response, clean_response, atol=1e-9))

    def test_near_null_gallery_patterns_are_distinct_and_can_share_rms(self):
        names = (
            "equatorial_checker_texture",
            "meridional_scanlines",
            "checkerboard_2px",
            "zonal_scanlines",
        )
        patterns = [
            center_and_scale(
                near_null_pattern(name, self.shape, self.latitudes),
                self.latitudes,
                0.2,
            )
            for name in names
        ]
        for pattern in patterns:
            self.assertEqual(pattern.shape, self.shape)
            self.assertAlmostEqual(area_weighted_mean(pattern, self.latitudes), 0.0, places=12)
            self.assertAlmostEqual(area_weighted_rms(pattern, self.latitudes), 0.2, places=12)
        for left, right in zip(patterns, patterns[1:]):
            self.assertFalse(np.allclose(left, right))

    def test_unknown_near_null_pattern_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown near-null pattern"):
            near_null_pattern("unknown", self.shape, self.latitudes)

    def test_deranged_indices_are_a_reproducible_permutation_without_self_pairs(self):
        first = deranged_indices(12, seed=9)
        second = deranged_indices(12, seed=9)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(sorted(first.tolist()), list(range(12)))
        self.assertTrue(np.all(first != np.arange(12)))

    def test_hemisphere_splice_takes_each_half_from_the_requested_source(self):
        northern = np.ones((3, *self.shape))
        southern = np.full((3, *self.shape), -2.0)
        spliced = splice_hemispheres(northern, southern, self.latitudes, 0.0)
        self.assertTrue(np.all(spliced[:, self.latitudes >= 0] == 1.0))
        self.assertTrue(np.all(spliced[:, self.latitudes < 0] == -2.0))
        rms = per_sample_area_weighted_rms(spliced - northern, self.latitudes)
        self.assertEqual(rms.shape, (3,))
        self.assertTrue(np.all(rms > 0.0))

    def test_local_response_w2_identifies_only_changed_anchors(self):
        reference = np.arange(20, dtype=np.float64).reshape(5, 4)
        candidate = reference.copy()
        candidate[:, 2] += 3.0
        local = local_response_w2(candidate, reference, n_quantiles=9)
        self.assertTrue(np.allclose(local[[0, 1, 3]], 0.0))
        self.assertAlmostEqual(local[2], 3.0)


if __name__ == "__main__":
    unittest.main()
