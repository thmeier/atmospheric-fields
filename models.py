import torch
import torch.nn as nn
import numpy as np

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

if __name__ == "__main__":
    model = MaskedAutoencoderViT()
    dummy_input = torch.randn(2, 4, 128, 256)
    loss, pred, mask = model(dummy_input)
    print("Loss:", loss.item())
    
    z = model.extract_features(dummy_input)
    print("Latent shape:", z.shape)
