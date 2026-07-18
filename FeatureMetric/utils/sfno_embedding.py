"""Adapter that exposes a colleague's pretrained SFNO autoencoder as a frozen
feature extractor, compatible with this repo's eval pipeline.

The SFNO encoder lives in a sibling repository (``SFNO-Embedding``). This adapter
targets the **4-field** checkpoints (no precipitation, trained 1975-2019 with
2020 held out of training) stored in ``SFNO-Embedding/weights_4fields/``. They map
raw, un-normalized, un-padded ERA5 surface fields ``(B, 4, 121, 240)`` — in the
fixed order ``[T2M, U10, V10, MSL]`` — to a *spatial* embedding ``(B, C, h, w)``.
To plug into the FID/MMD machinery (which expects one feature vector per sample)
we spatially pool the embedding to ``(B, feature_dim)``.

Two integration notes that differ from the MAE/I-JEPA path:
  * SFNO normalizes inputs internally with its *own* training stats, so we feed it
    raw physical units — not this repo's normalized+padded tensors.
  * The fields are the same 4 surface vars MAE/I-JEPA use, so SFNO reads the
    standard ERA5/Pangu/GraphCast NetCDF files directly via :class:`RawFourVarDataset`
    (un-normalized, un-padded). Dropping precipitation is what lets SFNO compare
    against Pangu (which forecasts no precip) as well as GraphCast.
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from netCDF4 import Dataset as NetCDFDataset

# Channel order the released 4-field SFNO checkpoints were trained on (load-bearing).
SFNO_VARS = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
]

# Default subdirectory of the SFNO repo holding the 4-field checkpoints.
DEFAULT_WEIGHTS_SUBDIR = "weights_4fields"


def _resolve_sfno_repo(repo_root=None, weights_subdir=DEFAULT_WEIGHTS_SUBDIR):
    """Locate the SFNO-Embedding repo and put its ``src/`` on ``sys.path``.

    Resolution order: explicit ``repo_root`` arg → ``SFNO_REPO`` env var →
    sibling ``../SFNO-Embedding`` next to this project. We import the vendored
    ``models.SFNO`` (which uses the repo's bundled ``th_copy`` SHT code), so the
    external ``torch_harmonics`` package is not required.
    """
    if repo_root is None:
        repo_root = os.environ.get("SFNO_REPO")
    if repo_root is None:
        here = os.path.dirname(os.path.abspath(__file__))
        # utils/ -> FeatureMetric/ -> atmospheric-fields/ -> PMLR_L/
        candidate = os.path.normpath(
            os.path.join(here, "..", "..", "..", "SFNO-Embedding")
        )
        repo_root = candidate
    src = os.path.join(repo_root, "src")
    weights = os.path.join(repo_root, weights_subdir)
    if not os.path.isdir(src):
        raise FileNotFoundError(
            f"SFNO repo src/ not found at {src!r}. Set --sfno-repo or $SFNO_REPO "
            f"to the SFNO-Embedding checkout."
        )
    if not os.path.isdir(weights):
        raise FileNotFoundError(
            f"SFNO weights not found at {weights!r}. Place the 4-field checkpoints "
            f"(model_<C>c_<HxW>_4fields.pth, static_fields.pth, "
            f"normalization_means_4fields.pt, normalization_stds_4fields.pt) there."
        )
    if src not in sys.path:
        sys.path.insert(0, src)
    return repo_root, weights


class SFNOEmbedding(nn.Module):
    """Frozen 4-field SFNO encoder + spatial pooling → per-sample feature vector.

    ``extract_features(x)`` mirrors the contract used by MAE/I-JEPA so the same
    ``extract_features_for_loader`` helper drives all three. Input ``x`` is
    ``(B, 4, 121, 240)`` in raw physical units.
    """

    # (embed_channels, embedding_resolution) combinations Marino shipped.
    VALID_CONFIGS = {
        (4, (15, 28)),
        (8, (15, 28)),
        (16, (15, 28)),
        (8, (31, 60)),
    }
    VALID_RES = {res for _, res in VALID_CONFIGS}
    VALID_CHANNELS = {c for c, _ in VALID_CONFIGS}

    # Number of input/output physical fields (T2M, U10, V10, MSL).
    IN_CHANNELS = 4

    POOLINGS = ("mean", "max", "meanstd", "grid", "flatten")

    def __init__(self, embedding_channels=8, embedding_resolution=(31, 60),
                 repo_root=None, pooling="mean", pool_grid=(7, 8),
                 weights_subdir=DEFAULT_WEIGHTS_SUBDIR):
        super().__init__()
        res = tuple(embedding_resolution)
        if (embedding_channels, res) not in self.VALID_CONFIGS:
            raise ValueError(
                f"({embedding_channels}c, {res}) is not an available 4-field config. "
                f"Valid configs: {sorted(self.VALID_CONFIGS)}"
            )
        if pooling not in self.POOLINGS:
            raise ValueError(f"unknown pooling {pooling!r}, expected one of {self.POOLINGS}")

        repo_root, weights = _resolve_sfno_repo(repo_root, weights_subdir)
        from models import SFNO  # vendored; no torchvision / external torch_harmonics

        self.embedding_channels = embedding_channels
        self.embedding_resolution = res
        self.pooling = pooling
        # Target grid for 'grid' pooling: adaptive-avg-pool the (h, w) embedding to
        # this size, then flatten → C * gh * gw dims. Keeps coarse spatial structure
        # at a dimensionality that stays well below N (so FID's covariance is
        # non-singular), unlike 'flatten' (= C * h * w, usable only with MMD).
        gh, gw = int(pool_grid[0]), int(pool_grid[1])
        self.pool_grid = (min(gh, res[0]), min(gw, res[1]))
        # Only consumed by the wind-patch-shuffle corruption in eval_distances.py.
        # SFNO is patch-free; this is a nominal value so that corruption still runs.
        self.patch_size = 16

        self.model = SFNO(
            nlat=121, nlon=240, channels=self.IN_CHANNELS,
            static_field_set="geopotential_and_mask", blocks=7,
            embed_size=[res[0], res[1]], embed_channels=embedding_channels,
        )
        ckpt = os.path.join(
            weights, f"model_{embedding_channels}c_{res[0]}x{res[1]}_4fields.pth"
        )
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self.model.eval()

        static = torch.load(
            os.path.join(weights, "static_fields.pth"), map_location="cpu"
        )["static_channels"].float()
        means = torch.load(
            os.path.join(weights, "normalization_means_4fields.pt"),
            map_location="cpu", weights_only=False,
        )
        stds = torch.load(
            os.path.join(weights, "normalization_stds_4fields.pt"),
            map_location="cpu", weights_only=False,
        )
        self.register_buffer("static_channels", static)
        self.register_buffer(
            "norm_mean",
            torch.as_tensor(np.asarray(means), dtype=torch.float32).view(1, -1, 1, 1),
        )
        self.register_buffer(
            "norm_std",
            torch.as_tensor(np.asarray(stds), dtype=torch.float32).view(1, -1, 1, 1),
        )

    @property
    def feature_dim(self):
        """Dimensionality of the feature vector produced by ``extract_features``."""
        c = self.embedding_channels
        if self.pooling == "meanstd":
            return 2 * c
        if self.pooling == "grid":
            return c * self.pool_grid[0] * self.pool_grid[1]
        if self.pooling == "flatten":
            return c * self.embedding_resolution[0] * self.embedding_resolution[1]
        return c

    def _pool(self, emb):
        """Reduce a spatial embedding ``(B, C, h, w)`` to ``(B, feature_dim)``."""
        if self.pooling == "mean":
            return emb.mean(dim=(-2, -1))
        if self.pooling == "max":
            return emb.amax(dim=(-2, -1))
        if self.pooling == "meanstd":
            # concat spatial mean and std per channel.
            return torch.cat([emb.mean(dim=(-2, -1)), emb.std(dim=(-2, -1))], dim=1)
        if self.pooling == "grid":
            # Adaptive-pool to a coarse grid, then flatten → keeps spatial structure.
            pooled = torch.nn.functional.adaptive_avg_pool2d(emb, self.pool_grid)
            return pooled.flatten(1)
        # flatten: full (C, h, w) concatenation. High-dim — use with MMD, not FID.
        return emb.flatten(1)

    @torch.no_grad()
    def encode(self, x, corruption_fn=None):
        """Raw fields ``(B, 4, 121, 240)`` → SFNO spatial embedding ``(B, C, h, w)``.

        ``corruption_fn`` (if given) is applied in SFNO's *standardized* space —
        i.e. after per-channel normalization, before the encoder. This is the
        space the corruption severities are calibrated for (unit-ish per channel),
        and it matches how the MAE/I-JEPA eval corrupts. Applying corruptions to
        the raw physical fields instead would be meaningless across the wildly
        different variable scales (e.g. K vs Pa).
        """
        if x.shape[-3:] != (self.IN_CHANNELS, 121, 240):
            raise ValueError(
                f"expected (B, {self.IN_CHANNELS}, 121, 240) raw fields, got {tuple(x.shape)}"
            )
        xn = (x - self.norm_mean) / self.norm_std
        if corruption_fn is not None:
            xn = corruption_fn(xn)
        return self.model(xn, self.static_channels)

    @torch.no_grad()
    def extract_features(self, x, corruption_fn=None):
        """Raw fields ``(B, 4, 121, 240)`` → pooled feature vector ``(B, feature_dim)``.

        ``corruption_fn`` is applied in standardized space (see :meth:`encode`).
        """
        return self._pool(self.encode(x, corruption_fn=corruption_fn))

    def forward(self, x):
        return self.extract_features(x)


class RawFourVarDataset(torch.utils.data.Dataset):
    """Reads raw (un-normalized, un-padded) 4-var ``(4, 121, 240)`` samples.

    Mirrors :class:`utils.dataset.AtmosphereDataset`'s file orientation
    (``(time, longitude, latitude)`` → ``(latitude, longitude)``) but applies no
    normalization or padding, since the SFNO encoder standardizes inputs itself.
    Works directly on the standard ERA5/Pangu/GraphCast surface NetCDF files.
    """

    def __init__(self, nc_path):
        self.nc_path = str(nc_path)
        self.ds = None
        with NetCDFDataset(self.nc_path, mode="r") as ds:
            self.n_samples = len(ds.dimensions["time"])
            missing = [v for v in SFNO_VARS if v not in ds.variables]
            if missing:
                raise ValueError(
                    f"{nc_path} is missing SFNO variables {missing}."
                )

    def _get_dataset(self):
        if self.ds is None:
            self.ds = NetCDFDataset(self.nc_path, mode="r")
        return self.ds

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        ds = self._get_dataset()
        arrays = []
        for var_name in SFNO_VARS:
            a = np.asarray(ds.variables[var_name][idx, :, :], dtype=np.float32)
            if a.shape == (240, 121):       # (lon, lat) -> (lat, lon)
                a = np.transpose(a, (1, 0))
            if a.shape != (121, 240):
                raise ValueError(
                    f"{var_name} slice has shape {a.shape}, expected (121, 240) or (240, 121)"
                )
            arrays.append(a)
        x = np.stack(arrays, axis=0)        # (4, 121, 240)
        return torch.from_numpy(x)
