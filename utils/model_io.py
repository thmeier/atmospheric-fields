from pathlib import Path

import torch

from .models import MaskedAutoencoderViT, build_ijepa


def build_model(model_name, device, ijepa_size="tiny"):
    if model_name == "mae":
        model = MaskedAutoencoderViT(
            embed_dim=256,
            depth=6,
            num_heads=8,
            decoder_embed_dim=128,
            decoder_depth=2,
            decoder_num_heads=4,
        )
    elif model_name == "ijepa":
        model = build_ijepa(model_size=ijepa_size)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model.to(device)


def checkpoint_path(model_name, checkpoint_dir=Path("checkpoints")):
    if model_name == "mae":
        return checkpoint_dir / "best_mae_model.pth"
    if model_name == "ijepa":
        return checkpoint_dir / "best_ijepa_model.pth"
    raise ValueError(f"Unknown model: {model_name}")


def save_ijepa_checkpoint(path, model, optimizer, epoch, val_loss, args):
    torch.save(
        {
            "context_encoder": model.context_encoder.state_dict(),
            "target_encoder": model.target_encoder.state_dict(),
            "predictor": model.predictor.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "config": vars(args),
        },
        path,
    )


def load_model_checkpoint(model_name, model, path, device):
    if model_name == "mae":
        model.load_state_dict(torch.load(path, map_location=device))
        return model

    checkpoint = torch.load(path, map_location=device)
    model.context_encoder.load_state_dict(checkpoint["context_encoder"])
    model.target_encoder.load_state_dict(checkpoint["target_encoder"])
    model.predictor.load_state_dict(checkpoint["predictor"])
    return model
