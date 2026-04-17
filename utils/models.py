import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy

def get_1d_sincos_pos_embed(embed_dim, positions):
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be even for sin/cos embeddings, got {embed_dim}")

    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000 ** omega)

    positions = positions.reshape(-1).astype(np.float32)
    out = np.einsum("m,d->md", positions, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)

def get_2d_sincos_pos_embed(embed_dim, grid_size_h, grid_size_w, cls_token=False):
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be even for 2D sin/cos embeddings, got {embed_dim}")

    grid_h = np.arange(grid_size_h, dtype=np.float32)
    grid_w = np.arange(grid_size_w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h, indexing="xy")
    grid = np.stack(grid, axis=0)
    grid = grid.reshape(2, -1)

    emb_h = get_1d_sincos_pos_embed(embed_dim // 2, grid[1])
    emb_w = get_1d_sincos_pos_embed(embed_dim // 2, grid[0])
    pos_embed = np.concatenate([emb_h, emb_w], axis=1)

    if cls_token:
        cls = np.zeros((1, embed_dim), dtype=np.float32)
        pos_embed = np.concatenate([cls, pos_embed], axis=0)
    return pos_embed

class PatchEmbed(nn.Module):
    """
    Splits 2D spatial fields into patches and embeds them.
    Output sequence length is: (img_size[0] // patch_size) * (img_size[1] // patch_size)
    """
    def __init__(self, img_size=(128, 256), patch_size=16, in_chans=4, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        
        # Conv2d with stride=patch_size perfectly cuts out non-overlapping patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, embed_dim, grid_H, grid_W)
        x = self.proj(x)
        # Flatten HW: (B, embed_dim, L) -> (B, L, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        return x

class MaskedAutoencoderViT(nn.Module):
    """
    Masked Autoencoder 
    Minimal v1 implementation
    """
    def __init__(
        self, 
        img_size=(128, 256), 
        patch_size=16, 
        in_chans=4,
        embed_dim=256, 
        depth=6, 
        num_heads=8,
        decoder_embed_dim=128, 
        decoder_depth=3, 
        decoder_num_heads=4,
        mlp_ratio=4.0
    ):
        super().__init__()
        
        # 1. Patch Embed
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        
        # 2. Encoder
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=int(embed_dim * mlp_ratio),
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.encoder_norm = nn.LayerNorm(embed_dim)
        
        # 3. Decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)
        
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_embed_dim, 
            nhead=decoder_num_heads, 
            dim_feedforward=int(decoder_embed_dim * mlp_ratio),
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=decoder_depth)
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        
        # 4. Reconstruction Head
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True)
        self.initialize_weights()

    def initialize_weights(self):
        encoder_pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            self.grid_size[0],
            self.grid_size[1],
            cls_token=True,
        )
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            self.grid_size[0],
            self.grid_size[1],
            cls_token=True,
        )

        self.pos_embed.data.copy_(torch.from_numpy(encoder_pos_embed).float().unsqueeze(0))
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        x: [N, L, D]
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        
        # Noise in [0, 1]
        noise = torch.rand(N, L, device=x.device)
        
        # Sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is mask
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # Generate mask: 0 is keep, 1 is mask
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # Embed patches
        x = self.patch_embed(x)
        
        # Add pos embed without cls token
        x = x + self.pos_embed[:, 1:, :]
        
        # Masking: L -> L * (1 - mask_ratio)
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        # Append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Apply Transformer Encoder
        x = self.encoder(x)
        x = self.encoder_norm(x)
        
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # Project to decoder dim
        x = self.decoder_embed(x)
        
        # Append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        # Gather original order: remove cls token, combine with mask tokens, then add cls token back
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        
        x = torch.cat([x[:, :1, :], x_], dim=1)
        
        # Add pos embed
        x = x + self.decoder_pos_embed
        
        # Apply Transformer Decoder
        x = self.decoder(x)
        x = self.decoder_norm(x)
        
        # Predict pixels
        x = self.decoder_pred(x)
        
        # Remove cls token
        x = x[:, 1:, :]
        
        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 4, H, W]
        pred: [N, L, p*p*4]
        mask: [N, L], 1 is mask, 0 is keep
        """
        target = self.patchify(imgs)
        # Using L1 loss
        loss = (pred - target).abs().mean(dim=-1)
        
        # Mean on masked patches only
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def patchify(self, imgs):
        """
        imgs: (N, 4, H, W)
        x: (N, L, patch_size**2 * 4)
        """
        p = self.patch_embed.patch_size
        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwcpq', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * self.in_chans))
        return x

    def forward(self, imgs, mask_ratio=0.75):
        # For training
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

    def extract_features(self, imgs):
        """
        For downstream tasks (extracting representation z).
        Outputs a single mean-pooled representation of the visible sequence.
        Since we want the whole field for validation, we pass mask_ratio=0.
        """
        # Embed patches
        x = self.patch_embed(imgs)
        x = x + self.pos_embed[:, 1:, :]
        
        # No masking
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        x = self.encoder(x)
        x = self.encoder_norm(x)
        
        # Mean pool patch tokens (ignore CLS token here, or pool it, let's pool the patch tokens)
        z = x[:, 1:, :].mean(dim=1)
        return z


def _normalize_mask_groups(masks):
    if isinstance(masks, torch.Tensor):
        if masks.ndim == 2:
            return [masks]
        if masks.ndim == 3:
            return [masks[i] for i in range(masks.shape[0])]
    return list(masks)


def apply_index_masks(x, masks):
    mask_groups = _normalize_mask_groups(masks)
    gathered = []
    for mask in mask_groups:
        index = mask.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        gathered.append(torch.gather(x, dim=1, index=index))
    return torch.cat(gathered, dim=0)


def repeat_interleave_batch(x, batch_size, repeat):
    if repeat == 1:
        return x
    chunks = torch.split(x, batch_size, dim=0)
    interleaved = []
    for chunk in chunks:
        interleaved.extend([chunk] * repeat)
    return torch.cat(interleaved, dim=0)


class RectangularVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=(128, 256),
        patch_size=16,
        in_chans=4,
        embed_dim=192,
        depth=8,
        num_heads=3,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)
        pos_embed = get_2d_sincos_pos_embed(embed_dim, self.grid_size[0], self.grid_size[1], cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            batch_first=True,
            norm_first=True,
            activation="gelu",
            dropout=0.0,
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Conv2d):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x, masks=None):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        if masks is not None:
            x = apply_index_masks(x, masks)
        x = self.blocks(x)
        return self.norm(x)

    def extract_features(self, x):
        return self.forward(x).mean(dim=1)


class IJEPAPredictor(nn.Module):
    def __init__(
        self,
        grid_size,
        num_patches,
        embed_dim,
        predictor_embed_dim=192,
        depth=3,
        num_heads=3,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        self.predictor_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, predictor_embed_dim), requires_grad=False
        )
        pos_embed = get_2d_sincos_pos_embed(
            predictor_embed_dim,
            grid_size[0],
            grid_size[1],
            cls_token=False,
        )
        self.predictor_pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=predictor_embed_dim,
            nhead=num_heads,
            dim_feedforward=int(predictor_embed_dim * mlp_ratio),
            batch_first=True,
            norm_first=True,
            activation="gelu",
            dropout=0.0,
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(predictor_embed_dim)
        self.proj = nn.Linear(predictor_embed_dim, embed_dim)

        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0)

    def forward(self, context_tokens, context_masks, target_masks):
        context_masks = _normalize_mask_groups(context_masks)
        target_masks = _normalize_mask_groups(target_masks)

        batch_size = context_tokens.shape[0] // len(context_masks)
        x = self.predictor_embed(context_tokens)

        context_pos = self.predictor_pos_embed.repeat(batch_size, 1, 1)
        x = x + apply_index_masks(context_pos, context_masks)
        _, n_context, _ = x.shape

        target_pos = self.predictor_pos_embed.repeat(batch_size, 1, 1)
        target_pos = apply_index_masks(target_pos, target_masks)
        target_pos = repeat_interleave_batch(target_pos, batch_size, repeat=len(context_masks))

        pred_tokens = self.mask_token.repeat(target_pos.shape[0], target_pos.shape[1], 1) + target_pos
        x = x.repeat(len(target_masks), 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)

        x = self.blocks(x)
        x = self.norm(x)
        x = self.proj(x[:, n_context:, :])
        return x


class IJEPA(nn.Module):
    def __init__(
        self,
        img_size=(128, 256),
        patch_size=16,
        in_chans=4,
        embed_dim=192,
        depth=8,
        num_heads=3,
        predictor_embed_dim=192,
        predictor_depth=3,
        predictor_num_heads=3,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.context_encoder = RectangularVisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )
        self.target_encoder = deepcopy(self.context_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        self.predictor = IJEPAPredictor(
            grid_size=self.context_encoder.grid_size,
            num_patches=self.context_encoder.patch_embed.num_patches,
            embed_dim=embed_dim,
            predictor_embed_dim=predictor_embed_dim,
            depth=predictor_depth,
            num_heads=predictor_num_heads,
            mlp_ratio=mlp_ratio,
        )
        self.patch_size = patch_size
        self.grid_size = self.context_encoder.grid_size
        self.embed_dim = embed_dim

    @torch.no_grad()
    def update_target_encoder(self, momentum):
        for context_param, target_param in zip(
            self.context_encoder.parameters(), self.target_encoder.parameters()
        ):
            target_param.data.mul_(momentum).add_(context_param.data, alpha=1.0 - momentum)

    def forward(self, imgs, context_masks, target_masks):
        self.target_encoder.eval()
        with torch.no_grad():
            target_tokens = self.target_encoder(imgs)
            target_tokens = F.layer_norm(target_tokens, (target_tokens.size(-1),))
            target_tokens = apply_index_masks(target_tokens, target_masks)

        context_tokens = self.context_encoder(imgs, masks=context_masks)
        pred_tokens = self.predictor(context_tokens, context_masks, target_masks)
        loss = F.smooth_l1_loss(pred_tokens, target_tokens)
        return loss, pred_tokens, target_tokens

    @torch.no_grad()
    def extract_features(self, imgs):
        return self.target_encoder.extract_features(imgs)

    @property
    def encoder(self):
        return self.context_encoder


def build_ijepa(model_size="tiny", img_size=(128, 256), patch_size=16, in_chans=4):
    configs = {
        "tiny": {
            "embed_dim": 192,
            "depth": 8,
            "num_heads": 3,
            "predictor_embed_dim": 192,
            "predictor_depth": 3,
            "predictor_num_heads": 3,
        },
        "small": {
            "embed_dim": 384,
            "depth": 10,
            "num_heads": 6,
            "predictor_embed_dim": 192,
            "predictor_depth": 4,
            "predictor_num_heads": 3,
        },
    }
    if model_size not in configs:
        raise ValueError(f"Unknown I-JEPA model size: {model_size}")
    return IJEPA(img_size=img_size, patch_size=patch_size, in_chans=in_chans, **configs[model_size])

if __name__ == "__main__":
    model = MaskedAutoencoderViT()
    dummy_input = torch.randn(2, 4, 128, 256)
    loss, pred, mask = model(dummy_input)
    print("Loss:", loss.item())
    
    z = model.extract_features(dummy_input)
    print("Latent shape:", z.shape)
