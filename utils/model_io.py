import torch
from pathlib import Path
from .models import build_mae, build_ijepa


def build_model(model_name, device, model_size):
    if model_name == "mae":
        model = build_mae(model_size=model_size)
    elif model_name == "ijepa":
        model = build_ijepa(model_size=model_size)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model.to(device)


def checkpoint_path(model_name, model_size, checkpoint_dir=Path("checkpoints")):
    if model_name in ("mae", "ijepa"):
        return checkpoint_dir / f"best_{model_name}_model_{model_size}.pth"
    raise ValueError(f"Unknown model: {model_name}")

def save_mae_checkpoint(path, model, optimizer, epoch, val_loss, args):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "config": vars(args),
        },
        path,
    )

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
    checkpoint = torch.load(path, map_location=device)
    print(f"loading path {path}")
    if model_name == "mae":
        # Support both old format (raw state dict) and new format (dict with "model" key)
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        return model

    # IJEPA
    checkpoint = torch.load(path, map_location=device)
    model.context_encoder.load_state_dict(checkpoint["context_encoder"])
    model.target_encoder.load_state_dict(checkpoint["target_encoder"])
    model.predictor.load_state_dict(checkpoint["predictor"])
    return model
