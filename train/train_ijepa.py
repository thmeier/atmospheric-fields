import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import math

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

from utils.dataset import AtmosphereDataset
from utils.ijepa_masking import MultiBlockMaskGenerator
from utils.model_io import build_model, save_ijepa_checkpoint


CLUSTER_DATA_PATH = Path("/cluster/courses/pmlr/teams/team07/data/era5_1.5deg_2004-01-01_2023-12-31.nc")
LOCAL_DATA_PATH = Path(__file__).parent.parent / "data" / "test_data_local.nc"
LARGE_LOCAL_DATA_PATH = Path(__file__).parent.parent / "data" / "test_data_local_5y.nc"


class WarmupCosineSchedule:
    def __init__(self, optimizer, warmup_steps, start_lr, ref_lr, final_lr, total_steps):
        self.optimizer = optimizer
        self.warmup_steps = max(1, warmup_steps)
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self.total_steps = max(1, total_steps)
        self.step_num = 0

    def step(self):
        self.step_num += 1
        if self.step_num <= self.warmup_steps:
            progress = self.step_num / self.warmup_steps
            lr = self.start_lr + progress * (self.ref_lr - self.start_lr)
        else:
            progress = (self.step_num - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            lr = self.final_lr + 0.5 * (self.ref_lr - self.final_lr) * (1.0 + math.cos(math.pi * progress))

        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr


class CosineWDSchedule:
    def __init__(self, optimizer, start_wd, final_wd, total_steps):
        self.optimizer = optimizer
        self.start_wd = start_wd
        self.final_wd = final_wd
        self.total_steps = max(1, total_steps)
        self.step_num = 0

    def step(self):
        self.step_num += 1
        progress = self.step_num / self.total_steps
        wd = self.final_wd + 0.5 * (self.start_wd - self.final_wd) * (1.0 + math.cos(math.pi * progress))
        for group in self.optimizer.param_groups:
            if group.get("WD_exclude", False):
                continue
            group["weight_decay"] = wd
        return wd


def make_optimizer(model, lr, weight_decay):
    param_groups = [
        {
            "params": [
                p
                for n, p in list(model.context_encoder.named_parameters()) + list(model.predictor.named_parameters())
                if p.requires_grad and ("bias" not in n) and (len(p.shape) != 1)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in list(model.context_encoder.named_parameters()) + list(model.predictor.named_parameters())
                if p.requires_grad and (("bias" in n) or (len(p.shape) == 1))
            ],
            "weight_decay": 0.0,
            "WD_exclude": True,
        },
    ]
    return AdamW(param_groups, lr=lr)


def resolve_data_path(args):
    if args.local and args.large_local:
        raise ValueError("Use only one of --local or --large-local.")
    if args.local:
        return LOCAL_DATA_PATH
    if args.large_local:
        return LARGE_LOCAL_DATA_PATH
    return CLUSTER_DATA_PATH


def load_stats(checkpoint_dir, recompute_stats):
    mean_path = checkpoint_dir / "data_mean.npy"
    std_path = checkpoint_dir / "data_std.npy"
    if not recompute_stats and mean_path.exists() and std_path.exists():
        return np.load(mean_path), np.load(std_path)
    return None


def build_datasets(data_path, stats, lazy_load, stats_chunk_size):
    train_dataset = AtmosphereDataset(
        data_path,
        split="train",
        stats=stats,
        lazy=lazy_load,
        stats_chunk_size=stats_chunk_size,
    )
    stats = train_dataset.get_stats()
    val_dataset = AtmosphereDataset(
        data_path,
        split="val",
        stats=stats,
        lazy=lazy_load,
        stats_chunk_size=stats_chunk_size,
    )
    return train_dataset, val_dataset, stats


def truncate_for_smoke(dataset, limit):
    limit = min(limit, len(dataset))
    return Subset(dataset, list(range(limit)))


def train_epoch(model, loader, optimizer, device, mask_generator, lr_schedule, wd_schedule, momentum_schedule, grad_clip):
    model.train()
    total_loss = 0.0
    pred_std_sum = 0.0
    target_std_sum = 0.0
    total_batches = 0
    current_lr = optimizer.param_groups[0]["lr"]
    current_wd = optimizer.param_groups[0].get("weight_decay", 0.0)
    pred_shape = None
    target_shape = None

    clip_params = [
        p for p in list(model.context_encoder.parameters()) + list(model.predictor.parameters())
        if p.requires_grad
    ]

    for batch in loader:
        batch = batch.to(device, non_blocking=device.type == "cuda")
        context_masks, target_masks = mask_generator.sample(batch.shape[0], device=device)

        optimizer.zero_grad()
        loss, pred_tokens, target_tokens = model(batch, context_masks, target_masks)
        if not torch.isfinite(loss):
            raise RuntimeError("Encountered non-finite JEPA loss")

        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(clip_params, max_norm=grad_clip)
        optimizer.step()
        model.update_target_encoder(next(momentum_schedule))
        current_lr = lr_schedule.step()
        current_wd = wd_schedule.step()

        total_loss += loss.item()
        pred_std_sum += pred_tokens.detach().float().std().item()
        target_std_sum += target_tokens.detach().float().std().item()
        total_batches += 1
        pred_shape = pred_tokens.shape
        target_shape = target_tokens.shape

    n = max(1, total_batches)
    return {
        "loss": total_loss / n,
        "pred_std": pred_std_sum / n,
        "target_std": target_std_sum / n,
        "lr": current_lr,
        "wd": current_wd,
        "pred_shape": pred_shape,
        "target_shape": target_shape,
    }


@torch.no_grad()
def val_epoch(model, loader, device, mask_generator):
    model.eval()
    total_loss = 0.0
    pred_std_sum = 0.0
    target_std_sum = 0.0
    total_batches = 0

    for batch in loader:
        batch = batch.to(device, non_blocking=device.type == "cuda")
        context_masks, target_masks = mask_generator.sample(batch.shape[0], device=device)
        loss, pred_tokens, target_tokens = model(batch, context_masks, target_masks)
        if not torch.isfinite(loss):
            raise RuntimeError("Encountered non-finite JEPA loss")

        total_loss += loss.item()
        pred_std_sum += pred_tokens.float().std().item()
        target_std_sum += target_tokens.float().std().item()
        total_batches += 1

    n = max(1, total_batches)
    return {
        "loss": total_loss / n,
        "pred_std": pred_std_sum / n,
        "target_std": target_std_sum / n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--large-local", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--stats-chunk-size", type=int, default=64)
    parser.add_argument("--recompute-stats", action="store_true")
    parser.add_argument("--lazy", dest="lazy", action="store_true")
    parser.add_argument("--eager", dest="lazy", action="store_false")
    parser.set_defaults(lazy=None)
    parser.add_argument("--model-size", choices=["tiny", "small"], default="tiny")
    parser.add_argument("--start-lr", type=float, default=2e-4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--final-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.04)
    parser.add_argument("--final-weight-decay", type=float, default=0.1)
    parser.add_argument("--ema-start", type=float, default=0.996)
    parser.add_argument("--ema-end", type=float, default=1.0)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Max grad norm; set to 0 to disable clipping.")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-samples", type=int, default=64)
    args = parser.parse_args()

    if args.model_size == "small":
        if parser.get_default("lr") == args.lr:
            args.lr = 1e-3
        if parser.get_default("final_weight_decay") == args.final_weight_decay:
            args.final_weight_decay = 0.2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available() and device.type != "cuda":
        device = torch.device("cpu")
    print(f"Using device: {device}")

    data_path = resolve_data_path(args)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    stats = load_stats(checkpoint_dir, args.recompute_stats)

    lazy_load = (not args.local) if args.lazy is None else args.lazy
    num_workers = 0 if args.num_workers is None else args.num_workers
    train_dataset, val_dataset, stats = build_datasets(
        data_path=data_path,
        stats=stats,
        lazy_load=lazy_load,
        stats_chunk_size=args.stats_chunk_size,
    )
    np.save(checkpoint_dir / "data_mean.npy", stats[0])
    np.save(checkpoint_dir / "data_std.npy", stats[1])

    if args.smoke_test:
        train_dataset = truncate_for_smoke(train_dataset, args.smoke_samples)
        val_dataset = truncate_for_smoke(val_dataset, max(8, args.smoke_samples // 2))
        args.epochs = min(args.epochs, 2)
        args.batch_size = min(args.batch_size, 8)

    if args.local and not args.smoke_test:
        args.epochs = min(args.epochs, 5)
        args.batch_size = min(args.batch_size, 8)

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    model = build_model("ijepa", device=device, ijepa_size=args.model_size)
    optimizer = make_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)

    mask_generator = MultiBlockMaskGenerator(
        input_size=(128, 256),
        patch_size=model.patch_size,
        enc_mask_scale=(0.85, 1.0),
        pred_mask_scale=(0.15, 0.2),
        aspect_ratio=(0.75, 1.5),
        nenc=1,
        npred=4,
        min_keep=4,
        allow_overlap=False,
    )

    total_steps = max(1, len(train_loader) * args.epochs)
    warmup_steps = min(total_steps, max(1, len(train_loader) * args.warmup_epochs))
    lr_schedule = WarmupCosineSchedule(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        start_lr=args.start_lr,
        ref_lr=args.lr,
        final_lr=args.final_lr,
        total_steps=total_steps,
    )
    wd_schedule = CosineWDSchedule(
        optimizer=optimizer,
        start_wd=args.weight_decay,
        final_wd=args.final_weight_decay,
        total_steps=total_steps,
    )
    momentum_schedule = iter(
        np.linspace(args.ema_start, args.ema_end, total_steps + 1, dtype=np.float64).tolist()
    )

    best_val_loss = float("inf")
    best_path = checkpoint_dir / "best_ijepa_model.pth"

    print(
        f"JEPA setup: grid={model.grid_size}, embed_dim={model.embed_dim}, "
        f"batch_size={args.batch_size}, epochs={args.epochs}, warmup_steps={warmup_steps}"
    )

    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        train_metrics = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            mask_generator=mask_generator,
            lr_schedule=lr_schedule,
            wd_schedule=wd_schedule,
            momentum_schedule=momentum_schedule,
            grad_clip=args.grad_clip,
        )
        val_metrics = val_epoch(
            model=model,
            loader=val_loader,
            device=device,
            mask_generator=mask_generator,
        )
        val_loss = val_metrics["loss"]
        msg = (
            f"Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_loss:.4f} | "
            f"LR: {train_metrics['lr']:.6f} | WD: {train_metrics['wd']:.4f} | "
            f"Pred std: {train_metrics['pred_std']:.3f} (val {val_metrics['pred_std']:.3f}) | "
            f"Target std: {train_metrics['target_std']:.3f}"
        )
        if epoch == 0:
            msg += (
                f" | Pred shape: {tuple(train_metrics['pred_shape'])} | "
                f"Target shape: {tuple(train_metrics['target_shape'])}"
            )
        print(msg)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_ijepa_checkpoint(best_path, model, optimizer, epoch + 1, val_loss, args)
            print(f"  -> Saved best model to {best_path}")


if __name__ == "__main__":
    main()
