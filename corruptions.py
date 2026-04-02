import torch
import scipy.ndimage as ndimage
import numpy as np


MAX_SEVERITY = 1.0
PAD_TOP = 4
PAD_BOTTOM = 3
PAD_LEFT = 8
PAD_RIGHT = 8
PADDED_SHAPE = (128, 256)


def get_corruption_ladder(corruption_type, n_steps=5):
    """
    Returns discrete severity levels in [0, MAX_SEVERITY] for distance evaluation.
    """
    if corruption_type in ("blur", "noise", "grf", "pixel_replace"):
        return np.linspace(0.0, MAX_SEVERITY, n_steps).tolist()
    else:
        raise ValueError(f"Unknown corruption type: {corruption_type}")


def _is_model_padded_shape(x):
    return tuple(x.shape[-2:]) == PADDED_SHAPE


def _crop_interior(x):
    return x[:, :, PAD_TOP:x.shape[-2] - PAD_BOTTOM, PAD_LEFT:x.shape[-1] - PAD_RIGHT]


def _repad_like_dataset(x):
    x_np = x.detach().cpu().numpy()
    x_np = np.pad(x_np, pad_width=((0, 0), (0, 0), (0, 0), (PAD_LEFT, PAD_RIGHT)), mode="wrap")
    x_np = np.pad(x_np, pad_width=((0, 0), (0, 0), (PAD_TOP, PAD_BOTTOM), (0, 0)), mode="constant", constant_values=0)
    return torch.from_numpy(x_np).to(device=x.device, dtype=x.dtype)


def apply_gaussian_blur(x, severity):
    """
    x: torch tensor of shape (N, C, H, W)
    severity: float in [0, 1]. Maps to sigma in [0, 1.125].
    """
    if severity <= 0:
        return x
    sigma = severity * 1.125
    if _is_model_padded_shape(x):
        interior = _crop_interior(x)
        interior_np = interior.detach().cpu().numpy()
        blurred_np = ndimage.gaussian_filter(interior_np, sigma=(0, 0, sigma, sigma))
        return _repad_like_dataset(torch.from_numpy(blurred_np).to(device=x.device, dtype=x.dtype))

    x_np = x.detach().cpu().numpy()
    x_blurred_np = ndimage.gaussian_filter(x_np, sigma=(0, 0, sigma, sigma))
    return torch.from_numpy(x_blurred_np).to(device=x.device, dtype=x.dtype)


def apply_high_freq_noise(x, severity):
    """
    x: torch tensor of shape (N, C, H, W)
    severity: float in [0, 1]. Maps to std in [0, 0.25].
    """
    if severity <= 0:
        return x
    std = severity * 0.25
    if _is_model_padded_shape(x):
        interior = _crop_interior(x)
        noise = torch.randn_like(interior) * std
        return _repad_like_dataset(interior + noise)

    noise = torch.randn_like(x) * std
    return x + noise


def apply_gaussian_field_noise(x, severity, len_scale=10.0):
    """
    x: torch tensor of shape (N, C, H, W)
    severity: float in [0, 1]. Maps to std in [0, 0.375].
    Adds spatially correlated Gaussian Random Field noise using FFT.
    The power spectrum is the Fourier transform of a Gaussian covariance
    C(r) = std^2 * exp(-r^2 / len_scale^2), which is also a Gaussian in
    frequency space.
    """
    if severity <= 0:
        return x

    std = severity * 0.375
    work_x = _crop_interior(x) if _is_model_padded_shape(x) else x
    N, C, H, W = work_x.shape

    # Frequency grids
    freq_y = np.fft.fftfreq(H).astype(np.float32)
    freq_x = np.fft.fftfreq(W).astype(np.float32)
    fy, fx = np.meshgrid(freq_y, freq_x, indexing="ij")

    # Power spectrum of Gaussian covariance: S(f) = pi * l^2 * exp(-pi^2 * l^2 * |f|^2)
    # We scale so that the spatial-domain variance equals std^2.
    freq_sq = fy**2 + fx**2
    power = np.exp(-2.0 * (np.pi * len_scale) ** 2 * freq_sq)

    # For ifft2 normalization (1/(H*W)): Var(Re(x)) = sum(A^2) / (H*W)^2
    # We want Var = std^2, so sum(A^2) = std^2 * (H*W)^2
    # With A[k] = c * sqrt(power[k]): c = std * H * W / sqrt(sum(power))
    amplitude = np.sqrt(power) * std * (H * W) / np.sqrt(power.sum())
    amplitude = torch.from_numpy(amplitude)  # (H, W)

    # Sample white noise in frequency domain, weight by amplitude, IFFT back
    # Real and imaginary parts are independent normals
    noise_freq = torch.randn(N, C, H, W) + 1j * torch.randn(N, C, H, W)
    noise_freq = noise_freq * amplitude
    noise = torch.fft.ifft2(noise_freq).real

    corrupted = work_x + noise.to(work_x.device)
    if _is_model_padded_shape(x):
        return _repad_like_dataset(corrupted)
    return corrupted


def apply_random_pixel_replace(x, severity, max_replace_prob=0.3):
    """
    x: torch tensor of shape (N, C, H, W)
    severity: float in [0, 1]. Maps to replacement probability in [0, 0.3].
    Replaces a random subset of interior pixels with Gaussian samples.
    """
    if severity <= 0:
        return x

    replace_prob = severity * max_replace_prob
    work_x = _crop_interior(x) if _is_model_padded_shape(x) else x

    mask = torch.rand_like(work_x) < replace_prob
    replacement = torch.randn_like(work_x)
    corrupted = torch.where(mask, replacement, work_x)

    if _is_model_padded_shape(x):
        return _repad_like_dataset(corrupted)
    return corrupted


if __name__ == "__main__":
    dummy_input = torch.randn(2, 4, 128, 256)
    blurred = apply_gaussian_blur(dummy_input, severity=0.5)
    noised = apply_high_freq_noise(dummy_input, severity=0.5)
    grf_noised = apply_gaussian_field_noise(dummy_input, severity=0.5)
    pixel_replaced = apply_random_pixel_replace(dummy_input, severity=0.5)

    print(f"Clean std: {dummy_input.std():.3f}")
    print(f"Blurred std: {blurred.std():.3f}")
    print(f"High-freq noised std: {noised.std():.3f}")
    print(f"GRF noised std: {grf_noised.std():.3f}")
    print(f"Pixel-replaced std: {pixel_replaced.std():.3f}")
