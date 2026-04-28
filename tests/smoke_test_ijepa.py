import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

import numpy as np
import torch

from utils.dataset import AtmosphereDataset
from utils.ijepa_masking import MultiBlockMaskGenerator
from utils.model_io import build_model


LOCAL_DATA_PATH = Path(__file__).parent.parent / "data" / "test_data_local.nc"


def to_coords(indices, grid_w):
    rows = indices // grid_w
    cols = indices % grid_w
    return rows, cols


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--model-size", choices=["tiny", "small"], default="tiny")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available() and device.type != "cuda":
        device = torch.device("cpu")

    mean = np.load(Path("checkpoints") / "data_mean.npy")
    std = np.load(Path("checkpoints") / "data_std.npy")
    dataset = AtmosphereDataset(LOCAL_DATA_PATH, split="train", stats=(mean, std), lazy=True)
    batch = torch.stack([dataset[i] for i in range(args.batch_size)], dim=0).to(device)

    model = build_model("ijepa", device=device, model_size=args.model_size)
    mask_generator = MultiBlockMaskGenerator(input_size=(128, 256), patch_size=model.patch_size)

    context_masks, target_masks = mask_generator.sample(batch.shape[0], device=device)

    assert batch.shape[1:] == (4, 128, 256)
    assert model.grid_size == (8, 16)
    assert context_masks.ndim == 2
    assert len(target_masks) == 4
    assert int(context_masks.min()) >= 0 and int(context_masks.max()) < 128
    for target_mask in target_masks:
        assert target_mask.ndim == 2
        assert int(target_mask.min()) >= 0 and int(target_mask.max()) < 128

    for batch_idx in range(batch.shape[0]):
        context_set = set(context_masks[batch_idx].tolist())
        for target_mask in target_masks:
            assert context_set.isdisjoint(set(target_mask[batch_idx].tolist()))

    rows, cols = to_coords(torch.arange(128), model.grid_size[1])
    assert int(rows.max()) == 7 and int(cols.max()) == 15
    assert model.context_encoder.pos_embed.shape[1] == 128
    assert model.predictor.predictor_pos_embed.shape[1] == 128

    loss, pred_tokens, target_tokens = model(batch, context_masks, target_masks)
    assert torch.isfinite(loss)
    assert pred_tokens.shape == target_tokens.shape

    features = model.extract_features(batch)
    assert features.shape == (args.batch_size, model.embed_dim)
    assert torch.isfinite(features).all()

    print("JEPA smoke test passed")
    print(f"batch_shape={tuple(batch.shape)}")
    print(f"grid_size={model.grid_size}")
    print(f"context_mask_shape={tuple(context_masks.shape)}")
    print(f"target_mask_shapes={[tuple(mask.shape) for mask in target_masks]}")
    print(f"pred_shape={tuple(pred_tokens.shape)}")
    print(f"feature_shape={tuple(features.shape)}")


if __name__ == "__main__":
    main()
