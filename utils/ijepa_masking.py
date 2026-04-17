import math

import torch


class MultiBlockMaskGenerator:
    def __init__(
        self,
        input_size=(128, 256),
        patch_size=16,
        context_scale=(0.85, 1.0),
        target_scale=(0.15, 0.2),
        target_aspect_ratio=(0.75, 1.5),
        num_targets=4,
        min_keep=4,
    ):
        if not isinstance(input_size, tuple):
            input_size = (input_size, input_size)
        self.grid_h = input_size[0] // patch_size
        self.grid_w = input_size[1] // patch_size
        self.context_scale = context_scale
        self.target_scale = target_scale
        self.target_aspect_ratio = target_aspect_ratio
        self.num_targets = num_targets
        self.min_keep = min_keep
        self.num_patches = self.grid_h * self.grid_w

    def _sample_block_size(self, scale, aspect_ratio_range, generator):
        rand_scale = torch.rand(1, generator=generator).item()
        rand_aspect = torch.rand(1, generator=generator).item()

        min_scale, max_scale = scale
        area = self.grid_h * self.grid_w * (min_scale + rand_scale * (max_scale - min_scale))

        min_ar, max_ar = aspect_ratio_range
        aspect_ratio = min_ar + rand_aspect * (max_ar - min_ar)

        h = max(1, int(round(math.sqrt(area * aspect_ratio))))
        w = max(1, int(round(math.sqrt(area / aspect_ratio))))
        h = min(h, self.grid_h)
        w = min(w, self.grid_w)
        return h, w

    def _sample_block_mask(self, block_size, generator):
        h, w = block_size
        top_max = max(1, self.grid_h - h + 1)
        left_max = max(1, self.grid_w - w + 1)
        top = torch.randint(0, top_max, (1,), generator=generator).item()
        left = torch.randint(0, left_max, (1,), generator=generator).item()

        mask = torch.zeros((self.grid_h, self.grid_w), dtype=torch.bool)
        mask[top:top + h, left:left + w] = True
        return mask

    def _mask_to_indices(self, mask):
        return torch.nonzero(mask.flatten(), as_tuple=False).squeeze(-1).to(dtype=torch.long)

    def sample(self, batch_size, device):
        context_masks = []
        target_masks = [[] for _ in range(self.num_targets)]

        for _ in range(batch_size):
            generator = torch.Generator()
            generator.seed()

            sampled_targets = []
            target_union = torch.zeros((self.grid_h, self.grid_w), dtype=torch.bool)
            for _target_idx in range(self.num_targets):
                target_size = self._sample_block_size(
                    self.target_scale,
                    self.target_aspect_ratio,
                    generator,
                )
                target_mask = self._sample_block_mask(target_size, generator)
                sampled_targets.append(target_mask)
                target_union |= target_mask

            context_indices = None
            context_size = self._sample_block_size(self.context_scale, (1.0, 1.0), generator)
            for _ in range(16):
                context_block = self._sample_block_mask(context_size, generator)
                visible_context = context_block & (~target_union)
                if visible_context.sum().item() >= self.min_keep:
                    context_indices = self._mask_to_indices(visible_context)
                    break

            if context_indices is None:
                visible_context = ~target_union
                context_indices = self._mask_to_indices(visible_context)

            if context_indices.numel() < self.min_keep:
                raise RuntimeError("Failed to sample a valid context mask without target overlap")

            context_masks.append(context_indices)
            for target_idx, target_mask in enumerate(sampled_targets):
                target_indices = self._mask_to_indices(target_mask)
                if target_indices.numel() == 0:
                    target_indices = context_indices[:1].clone()
                target_masks[target_idx].append(target_indices)

        min_context = min(mask.numel() for mask in context_masks)
        context_masks = torch.stack([mask[:min_context] for mask in context_masks], dim=0).to(device)

        global_min_target = min(mask.numel() for masks in target_masks for mask in masks)
        stacked_targets = []
        for masks in target_masks:
            stacked_targets.append(
                torch.stack([mask[:global_min_target] for mask in masks], dim=0).to(device)
            )

        return context_masks, stacked_targets
