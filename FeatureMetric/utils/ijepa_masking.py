import math
from logging import getLogger
from multiprocessing import Value

import torch

logger = getLogger()


class MultiBlockMaskGenerator:
    def __init__(
        self,
        input_size=(128, 256),
        patch_size=16,
        enc_mask_scale=(0.85, 1.0),
        pred_mask_scale=(0.15, 0.2),
        aspect_ratio=(0.75, 1.5),
        nenc=1,
        npred=4,
        min_keep=4,
        allow_overlap=False,
        context_mode="multiblock",
        disjoint_targets=False,
    ):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.patch_size = patch_size
        self.height = input_size[0] // patch_size
        self.width = input_size[1] // patch_size
        self.num_patches = self.height * self.width
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio = aspect_ratio
        self.nenc = nenc
        self.npred = npred
        self.min_keep = min_keep
        self.allow_overlap = allow_overlap
        self.context_mode = context_mode
        self.disjoint_targets = disjoint_targets
        self._itr_counter = Value("i", -1)

        if self.context_mode not in {"multiblock", "world-band"}:
            raise ValueError(f"Unknown context mode: {self.context_mode}")

    def step(self):
        counter = self._itr_counter
        with counter.get_lock():
            counter.value += 1
            value = counter.value
        return value

    def _make_generator(self):
        generator = torch.Generator()
        generator.manual_seed(self.step())
        return generator

    def _sample_block_size(self, generator, scale, aspect_ratio_scale):
        rand = torch.rand(1, generator=generator).item()
        min_scale, max_scale = scale
        mask_scale = min_scale + rand * (max_scale - min_scale)
        max_keep = max(1, int(self.num_patches * mask_scale))

        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + rand * (max_ar - min_ar)

        height = max(1, int(round(math.sqrt(max_keep * aspect_ratio))))
        width = max(1, int(round(math.sqrt(max_keep / aspect_ratio))))
        while height >= self.height:
            height -= 1
        while width >= self.width:
            width -= 1
        return max(1, height), max(1, width)

    def _sample_block_mask(self, generator, block_size, acceptable_regions=None):
        height, width = block_size

        def constrain_mask(mask, tries=0):
            if acceptable_regions is None:
                return
            n_regions = max(int(len(acceptable_regions) - tries), 0)
            for idx in range(n_regions):
                mask *= acceptable_regions[idx]

        tries = 0
        timeout = original_timeout = 20
        valid_mask = False
        while not valid_mask:
            top = torch.randint(0, self.height - height + 1, (1,), generator=generator).item()
            left = torch.randint(0, self.width - width + 1, (1,), generator=generator).item()
            mask = torch.zeros((self.height, self.width), dtype=torch.int32)
            mask[top:top + height, left:left + width] = 1
            constrain_mask(mask, tries=tries)
            mask = torch.nonzero(mask.flatten()).squeeze(-1)
            valid_mask = len(mask) > self.min_keep
            if not valid_mask:
                timeout -= 1
                if timeout == 0:
                    tries += 1
                    timeout = original_timeout
                    logger.warning(
                        'Mask generator says: "Valid mask not found, decreasing acceptable-regions [%s]"',
                        tries,
                    )

        mask_complement = torch.ones((self.height, self.width), dtype=torch.int32)
        mask_complement[top:top + height, left:left + width] = 0
        return mask, mask_complement

    def _sample_target_masks(self, batch_size, device, generator):
        pred_size = self._sample_block_size(
            generator=generator,
            scale=self.pred_mask_scale,
            aspect_ratio_scale=self.aspect_ratio,
        )
        collated_masks = []
        min_keep_pred = self.num_patches

        for _ in range(batch_size):
            sample_masks = []
            acceptable_regions = None
            if self.disjoint_targets:
                acceptable_regions = [torch.ones((self.height, self.width), dtype=torch.int32)]

            for _ in range(self.npred):
                mask, mask_complement = self._sample_block_mask(
                    generator=generator,
                    block_size=pred_size,
                    acceptable_regions=acceptable_regions,
                )
                sample_masks.append(mask)
                min_keep_pred = min(min_keep_pred, len(mask))
                if self.disjoint_targets:
                    acceptable_regions = [acceptable_regions[0] * mask_complement]
            collated_masks.append(sample_masks)

        pred_masks = []
        for pred_idx in range(self.npred):
            pred_masks.append(
                torch.stack(
                    [collated_masks[batch_idx][pred_idx][:min_keep_pred] for batch_idx in range(batch_size)],
                    dim=0,
                ).to(device)
            )
        return pred_masks

    def _sample_multiblock_context(self, batch_size, device, generator, target_masks):
        enc_size = self._sample_block_size(
            generator=generator,
            scale=self.enc_mask_scale,
            aspect_ratio_scale=(1.0, 1.0),
        )
        collated_masks = []
        min_keep_enc = self.num_patches

        for batch_idx in range(batch_size):
            acceptable_regions = masks_to_complements(
                [target_masks[pred_idx][batch_idx] for pred_idx in range(self.npred)],
                self.height,
                self.width,
            )
            if self.allow_overlap:
                acceptable_regions = None

            sample_contexts = []
            for _ in range(self.nenc):
                mask, _ = self._sample_block_mask(
                    generator=generator,
                    block_size=enc_size,
                    acceptable_regions=acceptable_regions,
                )
                sample_contexts.append(mask)
                min_keep_enc = min(min_keep_enc, len(mask))
            collated_masks.append(sample_contexts)

        enc_masks = [
            torch.stack(
                [collated_masks[batch_idx][enc_idx][:min_keep_enc] for batch_idx in range(batch_size)],
                dim=0,
            ).to(device)
            for enc_idx in range(self.nenc)
        ]
        return enc_masks[0] if self.nenc == 1 else enc_masks

    def _sample_world_band_context(self, batch_size, device, generator, target_masks):
        enc_scale = self.enc_mask_scale[0] + (
            self.enc_mask_scale[1] - self.enc_mask_scale[0]
        ) * torch.rand(1, generator=generator).item()
        band_width = int(round(enc_scale * self.width))
        band_width = max(1, min(self.width, band_width))

        context_masks = []
        min_keep_enc = self.num_patches
        for batch_idx in range(batch_size):
            start_col = torch.randint(0, self.width, (1,), generator=generator).item()
            cols = [(start_col + offset) % self.width for offset in range(band_width)]

            context_grid = torch.zeros((self.height, self.width), dtype=torch.int32)
            context_grid[:, cols] = 1

            if not self.allow_overlap:
                for pred_idx in range(self.npred):
                    rows, longitudes = to_coords(target_masks[pred_idx][batch_idx], self.width)
                    context_grid[rows, longitudes] = 0

            context_indices = torch.nonzero(context_grid.flatten()).squeeze(-1)
            if len(context_indices) <= self.min_keep:
                raise RuntimeError("World-band context produced too few visible patches")
            min_keep_enc = min(min_keep_enc, len(context_indices))
            context_masks.append(context_indices)

        context_masks = torch.stack(
            [context_masks[batch_idx][:min_keep_enc] for batch_idx in range(batch_size)],
            dim=0,
        ).to(device)
        return context_masks

    def sample(self, batch_size, device):
        generator = self._make_generator()
        target_masks = self._sample_target_masks(batch_size=batch_size, device=device, generator=generator)
        if self.context_mode == "multiblock":
            context_masks = self._sample_multiblock_context(
                batch_size=batch_size,
                device=device,
                generator=generator,
                target_masks=target_masks,
            )
        else:
            context_masks = self._sample_world_band_context(
                batch_size=batch_size,
                device=device,
                generator=generator,
                target_masks=target_masks,
            )
        return context_masks, target_masks


def to_coords(indices, grid_width):
    rows = indices // grid_width
    cols = indices % grid_width
    return rows, cols


def masks_to_complements(target_masks, grid_height, grid_width):
    complements = []
    for mask in target_masks:
        complement = torch.ones((grid_height, grid_width), dtype=torch.int32)
        rows, cols = to_coords(mask, grid_width)
        complement[rows, cols] = 0
        complements.append(complement)
    return complements


def target_masks_to_mae_masks(target_masks, num_patches):
    batch_size = target_masks[0].shape[0]
    mask = torch.zeros((batch_size, num_patches), dtype=torch.float32, device=target_masks[0].device)

    for target_mask in target_masks:
        mask.scatter_(1, target_mask.long(), 1.0)

    visible = (mask == 0).to(dtype=torch.int64)
    visible_counts = visible.sum(dim=1)
    if not torch.all(visible_counts == visible_counts[0]):
        raise ValueError("Structured MAE masking requires a fixed number of visible patches per batch")

    visible_indices = torch.argsort(mask, dim=1, stable=True)[:, :visible_counts[0].item()]
    return visible_indices, mask
