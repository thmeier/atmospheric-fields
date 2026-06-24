import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

import numpy as np
import torch

from utils.dataset import AtmosphereDataset
from utils.ijepa_masking import MultiBlockMaskGenerator, target_masks_to_mae_masks
from utils.model_io import build_model


LOCAL_DATA_PATH = Path(__file__).parent.parent / "data" / "test_data_local.nc"


def main():
    """Smoke test: drive the MAE with I-JEPA-style structured (block) masking.

    Converts target blocks to MAE visible/loss masks and checks visible/masked
    partitioning, a finite loss, and a well-shaped feature vector.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available() and device.type != "cuda":
        device = torch.device("cpu")

    mean = np.load(Path("checkpoints") / "data_mean.npy")
    std = np.load(Path("checkpoints") / "data_std.npy")
    dataset = AtmosphereDataset(LOCAL_DATA_PATH, split="train", stats=(mean, std), lazy=True)
    batch = torch.stack([dataset[i] for i in range(args.batch_size)], dim=0).to(device)

    model = build_model("mae", device=device, model_size="twin")
    mask_generator = MultiBlockMaskGenerator(
        input_size=(128, 256),
        patch_size=model.patch_size,
        disjoint_targets=True,
    )

    _, target_masks = mask_generator.sample(batch.shape[0], device=device)
    visible_indices, loss_mask = target_masks_to_mae_masks(target_masks, model.patch_embed.num_patches)

    assert visible_indices.ndim == 2
    assert loss_mask.shape == (args.batch_size, model.patch_embed.num_patches)
    assert torch.all(loss_mask.sum(dim=1) == loss_mask.sum(dim=1)[0])

    visible_set = [set(indices.tolist()) for indices in visible_indices]
    for batch_idx in range(args.batch_size):
        masked = set()
        for target_mask in target_masks:
            masked.update(target_mask[batch_idx].tolist())
        assert visible_set[batch_idx].isdisjoint(masked)
        assert len(visible_set[batch_idx]) + len(masked) == model.patch_embed.num_patches

    loss, pred, returned_mask = model(batch, visible_indices=visible_indices, loss_mask=loss_mask)
    assert torch.isfinite(loss)
    assert pred.shape[1] == model.patch_embed.num_patches
    assert torch.allclose(returned_mask, loss_mask)

    features = model.extract_features(batch)
    assert features.shape == (args.batch_size, model.encoder_norm.normalized_shape[0])
    assert torch.isfinite(features).all()

    print("MAE structured masking smoke test passed")
    print(f"visible_shape={tuple(visible_indices.shape)}")
    print(f"target_mask_shapes={[tuple(mask.shape) for mask in target_masks]}")
    print(f"pred_shape={tuple(pred.shape)}")
    print(f"feature_shape={tuple(features.shape)}")


if __name__ == "__main__":
    main()
