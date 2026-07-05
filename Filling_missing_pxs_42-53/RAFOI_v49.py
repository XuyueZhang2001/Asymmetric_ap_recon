VERSION = 49
"""
2026.06.15

Modified from v41

Key changes:
  - Add Poisson noise to the padded images
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
import os

# ─────────────────────────────────────────────────────────────
#  Device selection
# ─────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ─────────────────────────────────────────────────────────────
#  Global hyper-parameters
# ─────────────────────────────────────────────────────────────

AP_L = 120                                           # Long axis of rectangular aperture
AP_W = int(AP_L / 10)                               # Short axis (10:1 ratio)
N_APERTURES = 16                                    # Number of rotation angles
ANGLES = np.linspace(0, 180, N_APERTURES, endpoint=False)  # kept as numpy for indexing
LENGTH = 1024                                        # Image height/width
WIDTH = 512                                         # Crop width for final comparison
C_SIZE = 1200                                        # Canvas size
H_NUM = 2                                           # Subplot rows for plotting
W_NUM = 8                                           # Subplot cols for plotting

IMG_PATH  = '/home/xz127/earth_project/Reconstruction/test3_1600_1200.jpg'
SAVE_PATH = f'/home/xz127/earth_project/Reconstruction/Results/v{VERSION}/v{VERSION}_{N_APERTURES}AP_combined_result.jpg'


# ═══════════════════════════════════════════════════════════════
#  Section 1 — Spatial-domain helpers
#  (Still use numpy/cv2 for polygon rasterization; result is
#   transferred to GPU where needed.)
# ═══════════════════════════════════════════════════════════════

def crop_and_stitch(image_np: np.ndarray, phi_ap_degrees: float,
                    L: int = LENGTH, W: int = WIDTH,
                    canvas_size: int = C_SIZE, sigma: float = 10) -> np.ndarray:
    """
    Crop a rotated rectangular ROI and place it on a square canvas.
    Background is filled with the mean color of the cropped region.
    Returns float32 numpy array (stays on CPU — only called during pre-processing).
    """
    h, w = image_np.shape[:2]
    phi_degrees = phi_ap_degrees + 90

    canvas_cx, canvas_cy = canvas_size // 2, canvas_size // 2

    # Rotated rectangle vertices in canvas space
    rect_points = np.array([[-L/2, -W/2], [L/2, -W/2],
                             [L/2,  W/2], [-L/2,  W/2]])
    phi_rad = np.radians(phi_degrees)
    cos_phi, sin_phi = np.cos(phi_rad), np.sin(phi_rad)
    rot_mat = np.array([[cos_phi, -sin_phi], [sin_phi, cos_phi]])
    rotated_pts = (rect_points @ rot_mat.T + [canvas_cx, canvas_cy]).astype(np.int32)

    # Place image centered on canvas
    img_cx, img_cy = w // 2, h // 2
    y_off, x_off = canvas_cy - img_cy, canvas_cx - img_cx
    temp_canvas = np.zeros((canvas_size, canvas_size, 3), dtype=image_np.dtype)
    y1, y2 = max(0, y_off), min(canvas_size, y_off + h)
    x1, x2 = max(0, x_off), min(canvas_size, x_off + w)
    temp_canvas[y1:y2, x1:x2] = image_np[max(0, -y_off): min(h, canvas_size-y_off),
                                           max(0, -x_off): min(w, canvas_size-x_off)]

    # Mean-color background via hard mask
    hard_mask = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    cv2.fillPoly(hard_mask, [rotated_pts], 255)
    mean_color = cv2.mean(temp_canvas, mask=hard_mask)[:3]

    # Feathered (Gaussian-blurred) mask for smooth blending
    feather_mask = np.zeros((canvas_size, canvas_size), dtype=np.float32)
    cv2.fillPoly(feather_mask, [rotated_pts], 1.0)
    if sigma > 0:
        feather_mask = cv2.GaussianBlur(feather_mask, (0, 0), sigmaX=sigma)
    feather_mask = feather_mask[:, :, np.newaxis]

    bg_canvas = np.full((canvas_size, canvas_size, 3), mean_color, dtype=image_np.dtype)
    return temp_canvas * feather_mask + bg_canvas * (1.0 - feather_mask)


def crop_and_stitch_const_bg(image_np: np.ndarray, phi_ap_degrees: float,
                              bg_color=(0, 0, 0),
                              L: int = LENGTH, W: int = WIDTH,
                              canvas_size: int = C_SIZE, sigma: float = 10) -> np.ndarray:
    """
    Crop a rotated rectangular ROI onto a canvas with a fixed background color.
    Returns float32 numpy array (CPU only — used in pre-processing).
    """
    h, w = image_np.shape[:2]
    phi_degrees = phi_ap_degrees + 90
    canvas_cx, canvas_cy = canvas_size // 2, canvas_size // 2

    rect_points = np.array([[-L/2, -W/2], [L/2, -W/2],
                             [L/2,  W/2], [-L/2,  W/2]])
    phi_rad = np.radians(phi_degrees)
    cos_phi, sin_phi = np.cos(phi_rad), np.sin(phi_rad)
    rot_mat = np.array([[cos_phi, -sin_phi], [sin_phi, cos_phi]])
    rotated_pts = (rect_points @ rot_mat.T + [canvas_cx, canvas_cy]).astype(np.int32)

    img_cx, img_cy = w // 2, h // 2
    y_off, x_off = canvas_cy - img_cy, canvas_cx - img_cx
    temp_canvas = np.zeros((canvas_size, canvas_size, 3), dtype=image_np.dtype)
    y1, y2 = max(0, y_off), min(canvas_size, y_off + h)
    x1, x2 = max(0, x_off), min(canvas_size, x_off + w)
    temp_canvas[y1:y2, x1:x2] = image_np[max(0, -y_off): min(h, canvas_size-y_off),
                                           max(0, -x_off): min(w, canvas_size-x_off)]

    feather_mask = np.zeros((canvas_size, canvas_size), dtype=np.float32)
    cv2.fillPoly(feather_mask, [rotated_pts], 1.0)
    if sigma > 0:
        feather_mask = cv2.GaussianBlur(feather_mask, (0, 0), sigmaX=sigma)
    feather_mask = feather_mask[:, :, np.newaxis]

    bg_canvas = np.full((canvas_size, canvas_size, 3), bg_color, dtype=image_np.dtype)
    return temp_canvas * feather_mask + bg_canvas * (1.0 - feather_mask)


def crop_center_np(img, cropx: int = WIDTH, cropy: int = WIDTH):
    """Center-crop a HxWxC numpy image."""
    y, x = img.shape[:2]
    sx = x // 2 - cropx // 2
    sy = y // 2 - cropy // 2
    return img[sy:sy+cropy, sx:sx+cropx, :]


def get_single_weight_map_np(phi_ap_degrees: float,
                              L: int = LENGTH, W: int = WIDTH,
                              canvas_size: int = C_SIZE) -> np.ndarray:
    """
    Binary spatial-domain weight map for one rotated rectangular aperture.
    Rasterization is done with cv2 (CPU); caller converts to GPU tensor as needed.
    """
    phi_degrees = phi_ap_degrees + 90
    cx, cy = canvas_size // 2, canvas_size // 2
    rect_points = np.array([[-L/2, -W/2], [L/2, -W/2],
                             [L/2,  W/2], [-L/2,  W/2]])
    phi_rad = np.radians(phi_degrees)
    rot_mat = np.array([[np.cos(phi_rad), -np.sin(phi_rad)],
                        [np.sin(phi_rad),  np.cos(phi_rad)]])
    rotated_pts = (rect_points @ rot_mat.T + [cx, cy]).astype(np.int32)
    mask = np.zeros((canvas_size, canvas_size), dtype=np.float32)
    cv2.fillPoly(mask, [rotated_pts], 1.0)
    return mask

def add_poisson_noise(img, peak=1000.0, seed=None):
    """
    img: numpy array, float, [0,1]
    """
    rng = np.random.default_rng(seed)
    img_clip = np.clip(img, 0, None)
    
    scaled = img_clip * peak
    noisy = rng.poisson(scaled).astype(np.float32)

    noisy = noisy / peak
    return noisy


# ═══════════════════════════════════════════════════════════════
#  Section 2 — GPU-based optics functions
# ═══════════════════════════════════════════════════════════════

def get_physical_aperture_gpu(phi_ap_deg: float,
                               canvas_H: int = C_SIZE, canvas_W: int = C_SIZE,
                               L: float = LENGTH, W: float = WIDTH,
                               device: torch.device = DEVICE) -> torch.Tensor:
    """
    Generate a binary rectangular aperture mask on GPU.
    Returns: float32 tensor of shape (canvas_H, canvas_W).
    """
    y_coord = torch.linspace(-canvas_H / 2, canvas_H / 2, canvas_H, device=device)
    x_coord = torch.linspace(-canvas_W / 2, canvas_W / 2, canvas_W, device=device)
    x, y = torch.meshgrid(x_coord, y_coord, indexing='xy')  # (H, W)

    phi_rad = np.radians(phi_ap_deg)
    cos_p = float(np.cos(phi_rad))
    sin_p = float(np.sin(phi_rad))

    # Rotate to aperture-local frame
    xr = x * cos_p + y * sin_p
    yr = -x * sin_p + y * cos_p

    aperture = ((xr.abs() <= L / 2) & (yr.abs() <= W / 2)).float()
    return aperture


def get_psf_gpu(aperture: torch.Tensor) -> torch.Tensor:
    """
    Compute normalised PSF from a binary aperture tensor on GPU.
    aperture: (H, W) float32 tensor.
    Returns: (H, W) float32 PSF tensor.
    """
    # fft2 on real input → complex; norm='forward' matches the original
    ft = torch.fft.fftshift(
        torch.fft.fft2(aperture.to(torch.complex64), norm='forward')
    )
    psf = ft.abs() ** 2
    total = psf.sum()
    if total > 0:
        psf = psf / total
    return psf.float()


def get_mtf_gpu(psf: torch.Tensor) -> torch.Tensor:
    """
    Compute normalised MTF from PSF on GPU.
    psf: (H, W) float32 tensor.
    Returns: (H, W) float32 MTF tensor (values in [0, 1]).
    """
    otf = torch.fft.fftshift(torch.fft.fft2(psf.to(torch.complex64)))
    mtf = otf.abs()
    mtf_max = mtf.max()
    if mtf_max > 0:
        mtf = mtf / mtf_max
    return mtf.float()


def get_combination_mask_gpu(phi_ap_deg: float,
                              canvas_H: int = C_SIZE, canvas_W: int = C_SIZE,
                              N_apertures: int = N_APERTURES,
                              device: torch.device = DEVICE) -> torch.Tensor:
    """
    Frequency-domain bow-tie (butterfly) combination mask for one aperture angle.
    Returns: float32 tensor of shape (H, W).
    """
    y_coord = torch.linspace(-canvas_H / 2, canvas_H / 2, canvas_H, device=device)
    x_coord = torch.linspace(-canvas_W / 2, canvas_W / 2, canvas_W, device=device)
    x, y = torch.meshgrid(x_coord, y_coord, indexing='xy')

    phi_rad = float(np.radians(phi_ap_deg))
    cos_p, sin_p = float(np.cos(phi_rad)), float(np.sin(phi_rad))

    xr = x * cos_p + y * sin_p
    yr = -x * sin_p + y * cos_p

    theta_local = torch.atan2(yr, xr)
    # Map to [-π/2, π/2] and threshold at π / (2n)
    d_i = ((theta_local + np.pi / 2) % np.pi - np.pi / 2).abs()
    mask = (d_i < (np.pi / (2 * N_apertures))).float()
    return mask


def get_combination_masks_not_equal_gpu(angle_list_deg,
                                         canvas_H: int = C_SIZE,
                                         canvas_W: int = C_SIZE,
                                         device: torch.device = DEVICE):
    """
    Asymmetric bow-tie masks with guaranteed full coverage correction.
    Each mask's angular extent is bounded by bisectors with neighboring angles
    (in the mod-180 sense).

    After initial mask generation, a coverage-correction pass ensures every pixel
    is covered by exactly one mask:
      - Pixels covered zero times are assigned to the angularly closest aperture.
      - Pixels covered more than once are kept only in the angularly closest aperture.

    Args:
        angle_list_deg: list or array of aperture rotation angles in degrees.
        canvas_H: height of the frequency grid.
        canvas_W: width of the frequency grid.
        device: torch device to run computation on.

    Returns:
        list of float32 tensors, one per angle, shape (H, W), every pixel covered exactly once.
    """
    # ------------------------------------------------------------------
    # 1. Angle preprocessing on CPU (tiny arrays)
    # ------------------------------------------------------------------
    angles_mod = np.array([a % 180 for a in angle_list_deg], dtype=np.float64)
    N = len(angles_mod)
    sorted_idx = np.argsort(angles_mod)
    sorted_ang = angles_mod[sorted_idx]

    hw_left  = np.zeros(N)
    hw_right = np.zeros(N)
    for i in range(N):
        prev = sorted_ang[(i - 1) % N]
        nxt  = sorted_ang[(i + 1) % N]
        cur  = sorted_ang[i]
        hw_left[i]  = ((cur - prev) % 180) / 2.0
        hw_right[i] = ((nxt  - cur)  % 180) / 2.0

    left_orig  = np.zeros(N)
    right_orig = np.zeros(N)
    for i in range(N):
        orig = sorted_idx[i]
        left_orig[orig]  = hw_left[i]
        right_orig[orig] = hw_right[i]

    # ------------------------------------------------------------------
    # 2. Build coordinate grid on GPU
    # ------------------------------------------------------------------
    y_coord = torch.linspace(-canvas_H / 2, canvas_H / 2, canvas_H, device=device)
    x_coord = torch.linspace(-canvas_W / 2, canvas_W / 2, canvas_W, device=device)
    x, y = torch.meshgrid(x_coord, y_coord, indexing='xy')   # (H, W) each

    # ------------------------------------------------------------------
    # 3. Pass 1 — generate all N masks, stacked into (N, H, W)
    #
    # Broadcasting strategy: expand angle tensors to (N, 1, 1) and pixel
    # grids to (1, H, W) so all masks are computed in a single kernel call.
    # ------------------------------------------------------------------
    phi     = torch.tensor(np.radians(angle_list_deg), dtype=torch.float32, device=device)  # (N,)
    hw_l_t  = torch.tensor(np.radians(left_orig),      dtype=torch.float32, device=device)  # (N,)
    hw_r_t  = torch.tensor(np.radians(right_orig),     dtype=torch.float32, device=device)  # (N,)

    phi_b   = phi[:, None, None]      # (N, 1, 1)
    hw_l_b  = hw_l_t[:, None, None]   # (N, 1, 1)
    hw_r_b  = hw_r_t[:, None, None]   # (N, 1, 1)

    x_b = x[None]                     # (1, H, W)
    y_b = y[None]                     # (1, H, W)

    xr = x_b * torch.cos(phi_b) + y_b * torch.sin(phi_b)    # (N, H, W)
    yr = -x_b * torch.sin(phi_b) + y_b * torch.cos(phi_b)   # (N, H, W)

    theta_local = torch.atan2(yr, xr)                        # (N, H, W)
    d_signed    = (theta_local + np.pi / 2) % np.pi - np.pi / 2

    # mask_stack: (N, H, W), float32, values in {0, 1}
    mask_stack = ((d_signed >= -hw_l_b) & (d_signed <= hw_r_b)).float()

    # ------------------------------------------------------------------
    # 4. Pass 2 — coverage correction, fully vectorised on GPU
    # ------------------------------------------------------------------
    coverage = mask_stack.sum(dim=0)   # (H, W)

    # Pixel direction angle in the mod-180 domain, shape (H, W)
    pixel_angle = torch.atan2(y, x).rad2deg() % 180

    # Angular distance from each pixel to each aperture center (mod-180 wrapped)
    # angles_mod_t: (N,) → (N, 1, 1) broadcast with pixel_angle (H, W) → (1, H, W)
    angles_mod_t = torch.tensor(angles_mod, dtype=torch.float32, device=device)
    delta        = (pixel_angle[None] - angles_mod_t[:, None, None]).abs()  # (N, H, W)
    angular_dist = torch.minimum(delta, 180.0 - delta)                      # (N, H, W)

    # For each pixel, index of the angularly closest aperture
    closest = angular_dist.argmin(dim=0)   # (H, W)

    # Boolean plane: True where this aperture is the closest one for the pixel
    aperture_idx = torch.arange(N, device=device)[:, None, None]  # (N, 1, 1)
    is_closest   = (closest[None] == aperture_idx)                 # (N, H, W)

    # Fix under-covered pixels (coverage == 0): assign to closest aperture
    under = (coverage == 0)[None]   # (1, H, W)
    mask_stack = torch.where(under & is_closest,
                             torch.ones_like(mask_stack),
                             mask_stack)

    # Fix over-covered pixels (coverage > 1): keep only closest aperture
    over = (coverage > 1)[None]     # (1, H, W)
    mask_stack = torch.where(over & ~is_closest,
                             torch.zeros_like(mask_stack),
                             mask_stack)

    # ------------------------------------------------------------------
    # 5. Sanity check — remains on GPU to avoid unnecessary transfer
    # ------------------------------------------------------------------
    final_coverage = mask_stack.sum(dim=0)
    assert final_coverage.min().item() == 1 and final_coverage.max().item() == 1, (
        f"Coverage correction failed: "
        f"min={final_coverage.min().item():.0f}, max={final_coverage.max().item():.0f}"
    )

    # ------------------------------------------------------------------
    # 6. Return as a list of (H, W) tensors, kept on GPU
    # ------------------------------------------------------------------
    return [mask_stack[j] for j in range(N)]


def get_circular_mtf_gpu(size: int = C_SIZE, L: float = AP_L,
                          wavelength: float = 550e-9,
                          focal_length: float = 33.6e-3,
                          pixel_size: float = 1.5e-6,
                          device: torch.device = DEVICE) -> torch.Tensor:
    """
    Diffraction-limited MTF for a circular aperture of diameter L.
    Returns: float32 tensor of shape (size, size).
    """
    f_c = L / (wavelength * focal_length)
    cutoff_freq_pixels = f_c * pixel_size * size

    coords = torch.linspace(-1, 1, size, device=device)
    y, x = torch.meshgrid(coords, coords, indexing='ij')
    rho = torch.sqrt(x**2 + y**2) * (size / 2)

    nu = rho / (cutoff_freq_pixels / 2)
    nu = nu.clamp(max=1.0)          # clamp so sqrt stays real

    # Circular aperture MTF: (2/π) * (arccos(ν) − ν√(1−ν²))
    mtf_dl = torch.zeros(size, size, device=device)
    valid = nu < 1.0
    nu_v = nu[valid]
    mtf_dl[valid] = (2.0 / np.pi) * (torch.acos(nu_v) - nu_v * torch.sqrt(1.0 - nu_v**2))
    return mtf_dl


def convolve_with_psf_gpu(img_t: torch.Tensor, psf_t: torch.Tensor) -> torch.Tensor:
    H, W, C = img_t.shape

    img_chw = img_t.permute(2, 0, 1).unsqueeze(1)            # (3, 1, H, W)
    psf_hw  = psf_t.unsqueeze(0).unsqueeze(0)                 # (1, 1, H, W)

    sig_f  = torch.fft.rfft2(img_chw)
    # ifftshift moves PSF center from (H//2, W//2) to (0, 0) before FFT,
    # which is required for correct linear convolution in frequency domain
    kern_f = torch.fft.rfft2(torch.fft.ifftshift(psf_hw), s=(H, W))
    out_f  = sig_f * kern_f
    out    = torch.fft.irfft2(out_f, s=(H, W))

    out = out.squeeze(1).permute(1, 2, 0)                      # (H, W, 3)
    return out

def add_poisson_noise_gpu(img, peak=1000.0):
    """
    img: torch.Tensor, float
    peak: Amp of noise, the larger, noise will be smaller
    """
    img_clip = torch.clamp(img, min=0)
    
    scaled = img_clip * peak
    noisy = torch.poisson(scaled)  # Sample at the same device
    
    noisy = noisy / peak
    return noisy.to(img.dtype)


def fftshift2_gpu(x: torch.Tensor, axes=(-2, -1)) -> torch.Tensor:
    """torch.fft.fftshift over specified axes."""
    return torch.fft.fftshift(x, dim=axes)


def ifftshift2_gpu(x: torch.Tensor, axes=(-2, -1)) -> torch.Tensor:
    """torch.fft.ifftshift over specified axes."""
    return torch.fft.ifftshift(x, dim=axes)


# ═══════════════════════════════════════════════════════════════
#  Section 3 — Utility / plotting helpers (CPU)
# ═══════════════════════════════════════════════════════════════

def fft_for_plot(fft_np: np.ndarray, epi: float = 1e-10) -> np.ndarray:
    """Log-normalise a complex FFT array for display."""
    log_abs = np.log(np.abs(fft_np) + epi)
    return log_abs / np.max(log_abs)


def plot_series_angles(images, sup_title, title, angles,
                       mode=None, H_num=H_NUM, W_num=W_NUM, save_path=None):
    """Tile a list of numpy images as a grid plot."""
    cmap_kw = {} if mode is None else {'cmap': mode}
    plt.figure(figsize=(W_num*3, H_num*3))
    for i, phi in enumerate(angles):
        plt.subplot(H_num, W_num, i+1)
        plt.imshow(images[i], **cmap_kw)
        plt.title(f'{title} (phi={phi}°)', fontsize=10)
        plt.axis('off')
    plt.suptitle(sup_title, fontsize=16)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


# ═══════════════════════════════════════════════════════════════
#  Section 4 — GPU reconstruction core
# ═══════════════════════════════════════════════════════════════

def combine_unequal_images_gpu(
    ang_id, k: float = 0.01,
    canvas_size: int = C_SIZE, ap_length: float = AP_L,
    all_angles=ANGLES,
    g_ffts_gpu=None,   # list of complex64 GPU tensors (H, W, 3)
    mtfs_gpu=None,     # list of float32  GPU tensors (H, W)
    dc_value=None,     # float or list of 3 floats for RGB
    dc_px: int = None,   # if dc_value is provided, this pixel's value in reconstructed_fft will be set to dc_value
    device: torch.device = DEVICE
):
    """
    GPU version of combine_unequal_images.

    Fuses the frequency-domain observations from the angles in ang_id,
    applies an H-filter (Wiener-like sharpening), and returns the
    reconstructed spatial image.

    Returns:
        rec_img_ori    (H, W, 3) float32 numpy array — before H-filter
        reconstructed  (H, W, 3) float32 numpy array — after  H-filter
        final_fft_gpu  (H, W, 3) complex64 GPU tensor
        masks          list of (H, W) float32 GPU tensors
        h_filter_gpu   (H, W)    float32 GPU tensor
    """
    assert g_ffts_gpu is not None and mtfs_gpu is not None

    ang_list   = [float(all_angles[i]) for i in ang_id]
    g_fft_sel  = [g_ffts_gpu[i] for i in ang_id]
    mtf_sel    = [mtfs_gpu[i]   for i in ang_id]

    masks = get_combination_masks_not_equal_gpu(ang_list,
                                                canvas_H=canvas_size,
                                                canvas_W=canvas_size,
                                                device=device)

    # Fuse frequency spectra: Σ M_i · G_i
    reconstructed_fft = torch.zeros(canvas_size, canvas_size, 3,
                                    dtype=torch.complex64, device=device)
    for i in range(len(ang_list)):
        reconstructed_fft += masks[i].unsqueeze(-1) * g_fft_sel[i]  # broadcast over RGB

    # ------NEW: Replace the DC pixel of reconstructed_fft with the mean DC value from all single FFTs------
    reconstructed_fft[dc_px, dc_px, :] = torch.tensor(dc_value, dtype=torch.complex64, device=device)
    
    # Σ M_i · MTF_i  (2-D accumulator)
    sum_m_mtf = torch.zeros(canvas_size, canvas_size, device=device)
    for i in range(len(ang_list)):
        sum_m_mtf += masks[i] * mtf_sel[i]

    # Ideal target MTF (diffraction-limited circular aperture)
    mtf_ideal = get_circular_mtf_gpu(size=canvas_size, L=ap_length, device=device)

    # H-filter (Wiener-like)
    h_filter = mtf_ideal / (sum_m_mtf + k)             # (H, W)

    # Inverse FFT → spatial domain (before H-filter)
    rec_spatial_raw = torch.fft.ifft2(
        ifftshift2_gpu(reconstructed_fft, axes=(0, 1)), dim=(0, 1)
    ).abs()                                             # (H, W, 3)

    rec_img_ori_np = rec_spatial_raw.cpu().numpy().astype(np.float32)

    # Apply H-filter and inverse FFT
    final_fft = reconstructed_fft * h_filter.unsqueeze(-1)
    rec_spatial_h = torch.fft.ifft2(
        ifftshift2_gpu(final_fft, axes=(0, 1)), dim=(0, 1)
    ).abs()

    # # Normalise only if extreme outliers are present
    # if rec_img_ori_np.max() > 1.2 or (rec_img_ori_np.max() < 0.8 and rec_img_ori_np.max() > 0):
    #     print(f"Outlier detected: group {ang_id}, max before normalisation = {rec_img_ori_np.max():.2f}")
    #     rec_img_ori_np /= rec_img_ori_np.max()
    #     rec_spatial_h = rec_spatial_h / rec_spatial_h.max()

    reconstructed_np = rec_spatial_h.cpu().numpy().astype(np.float32)
    
    return rec_img_ori_np, reconstructed_np, reconstructed_fft, final_fft, masks, sum_m_mtf, h_filter


# ═══════════════════════════════════════════════════════════════
#  Section 5 — Weight-map & group-dictionary construction
# ═══════════════════════════════════════════════════════════════

def build_weight_maps(angles=ANGLES, canvas_size: int = C_SIZE,
                      L: int = LENGTH, W: int = WIDTH):
    """
    Build the stacked binary weight map and the pixel-group lookup tables.

    Returns:
        weight_stack_np      (N, H, W) bool numpy array
        TOTAL_WEIGHT_MAP     (H, W)    float32 numpy array
        GROUP_ID_ARR         (H, W)    int32  numpy array
        GROUP_MAP            dict {group_id → tuple_of_angle_indices or None}
    """
    N = len(angles)
    weight_stack = np.zeros((N, canvas_size, canvas_size), dtype=bool)
    for i, phi in enumerate(angles):
        wm = get_single_weight_map_np(phi, L=L, W=W, canvas_size=canvas_size)
        weight_stack[i] = wm > 0

    TOTAL_WEIGHT_MAP = weight_stack.sum(axis=0).astype(np.float32)

    # Encode each pixel's combination of covering masks as a group ID
    weight_HWN  = weight_stack.transpose(1, 2, 0)   # (H, W, N)
    active_mask = TOTAL_WEIGHT_MAP > 0
    rows, cols  = np.where(active_mask)

    group_map_ori = {}
    GROUP_ID_ARR  = np.zeros((canvas_size, canvas_size), dtype=np.int32)
    counter = 1
    for r, c in zip(rows, cols):
        key = tuple(int(k) for k in np.where(weight_HWN[r, c])[0])
        if key not in group_map_ori:
            group_map_ori[key] = counter
            counter += 1
        GROUP_ID_ARR[r, c] = group_map_ori[key]

    GROUP_MAP = {v: k for k, v in group_map_ori.items()}
    GROUP_MAP[0] = None   # group 0 → no coverage

    return weight_stack, TOTAL_WEIGHT_MAP, GROUP_ID_ARR, GROUP_MAP, rows, cols


# ═══════════════════════════════════════════════════════════════
#  Section 6 — Per-group reconstruction + stitching (GPU)
# ═══════════════════════════════════════════════════════════════

def fill_corr_px_gpu(group_id: int,
                     combined_gpu:      torch.Tensor,   # (H, W, 3) float32 GPU
                     combined_no_h_gpu: torch.Tensor,   # (H, W, 3) float32 GPU
                     GROUP_ID_ARR,
                     GROUP_MAP,
                     k: float = 0.05,
                     canvas_size: int = C_SIZE,
                     aperture_length: float = AP_L,
                     angle_series=ANGLES,
                     g_img_ffts_gpu=None,
                     img_mtfs_gpu=None,
                     dc_mean=None,
                     center_px=None,
                     device: torch.device = DEVICE):
    """
    Reconstruct the pixels belonging to group_id and write them into combined_gpu
    and combined_no_h_gpu in-place (cloned copies).

    Returns:
        new_img             (H, W, 3) float32 GPU tensor — after  H-filter
        new_img_no_h        (H, W, 3) float32 GPU tensor — before H-filter
        single_img_no_h_np  (H, W, 3) float32 numpy array
        single_img_np       (H, W, 3) float32 numpy array
    """
        
    ang_idxs_tup = GROUP_MAP[group_id]
    if ang_idxs_tup is None:
        print(f"Group {group_id}: no contributing angles. Skipping.")
        return combined_gpu, combined_no_h_gpu, None, None

    ang_ids = list(ang_idxs_tup)

    rec_no_h_np, rec_np, reconstructed_fft, final_fft, masks, sum_m_mtf, h_filter = combine_unequal_images_gpu(
        ang_id=ang_ids, k=k,
        canvas_size=canvas_size, ap_length=aperture_length,
        all_angles=angle_series,
        g_ffts_gpu=g_img_ffts_gpu,
        mtfs_gpu=img_mtfs_gpu,
        dc_value=dc_mean,
        dc_px=center_px,
        device=device
    )
    
    if torch.isnan(reconstructed_fft).any() or torch.isinf(reconstructed_fft).any():
        print(f"Warning: NaN/Inf detected in group {group_id}")

    # Build boolean mask on GPU
    mask_np  = (GROUP_ID_ARR == group_id)                     # (H, W) bool numpy
    mask_gpu = torch.from_numpy(mask_np).to(device).unsqueeze(-1)  # (H, W, 1)

    rec_gpu      = torch.from_numpy(rec_np).to(device)        # (H, W, 3)
    rec_no_h_gpu = torch.from_numpy(rec_no_h_np).to(device)

    new_img      = combined_gpu.clone()
    new_img_no_h = combined_no_h_gpu.clone()

    # Write only the pixels that belong to this group
    new_img[mask_gpu.expand_as(new_img)]           = rec_gpu[mask_gpu.expand_as(rec_gpu)]
    new_img_no_h[mask_gpu.expand_as(new_img_no_h)] = rec_no_h_gpu[mask_gpu.expand_as(rec_no_h_gpu)]

    return new_img, new_img_no_h, rec_no_h_np, rec_np, reconstructed_fft, final_fft, masks, sum_m_mtf, h_filter


# ═══════════════════════════════════════════════════════════════
#  Section 7 — Main pipeline
# ═══════════════════════════════════════════════════════════════

def main():
    # ── 7.1  Load and pre-process image (CPU) ───────────────────
    img_bgr = cv2.imread(IMG_PATH)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {IMG_PATH}")
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    IMG_FULL = img_rgb.astype(np.float32) / 255.0

    # Ground-truth crop and background colour
    GT_size = max(LENGTH, WIDTH)
    GT_IMG  = crop_center_np(IMG_FULL, cropx=GT_size, cropy=GT_size)
    BG_COLOR = cv2.mean(GT_IMG)[:3]
    print(f"Background colour (RGB mean): {BG_COLOR}")

    # ── 7.2  Determine convolution kernel size  ──────────────────
    # v35: convolve on the largest square crop that fits in the image
    ker_size = min(IMG_FULL.shape[0], IMG_FULL.shape[1], C_SIZE)
    img_for_conv = crop_center_np(IMG_FULL, cropx=ker_size, cropy=ker_size)

    # ── 7.3  Upload crop to GPU ──────────────────────────────────
    img_for_conv_gpu = torch.from_numpy(img_for_conv).to(DEVICE)   # (H, W, 3)

    # ── 7.4  Compute apertures / PSFs / MTFs and convolve (GPU) ──
    # v35 workflow: convolve on ker_size canvas, THEN crop & stitch
    APERTURES_GPU = []
    PSFS_GPU      = []
    MTFS_GPU      = []
    CONV_IMGS     = []   # (H, W, 3) float32 numpy — convolved images for stitching

    for phi in ANGLES:
        aperture = get_physical_aperture_gpu(
            phi_ap_deg=phi, canvas_H=ker_size, canvas_W=ker_size,
            L=AP_L, W=AP_W, device=DEVICE
        )
        psf = get_psf_gpu(aperture)
        mtf = get_mtf_gpu(psf)

        # Convolve full (ker_size × ker_size) image with PSF on GPU
        conv_result_gpu = convolve_with_psf_gpu(img_for_conv_gpu, psf)

        APERTURES_GPU.append(aperture)
        PSFS_GPU.append(psf)
        MTFS_GPU.append(mtf)
        CONV_IMGS.append(conv_result_gpu.cpu().numpy())

    print("Convolution done on GPU.")

    # ── 7.5  Crop & stitch convolved images onto canvas (CPU) ────
    # (sigma=0 matches the v35 notebook — no Gaussian feathering here)
    IMG_CROP    = []
    G_IMG_FFTS_GPU = []   # complex64 fftshifted GPU tensors (C_SIZE, C_SIZE, 3)

    # If ker_size != C_SIZE, recompute apertures/PSFs/MTFs at canvas size
    if ker_size != C_SIZE:
        print(f"Recomputing PSFs/MTFs at canvas size {C_SIZE} (ker_size={ker_size})")
        APERTURES_GPU = []
        PSFS_GPU      = []
        MTFS_GPU      = []
        for phi in ANGLES:
            aperture = get_physical_aperture_gpu(
                phi_ap_deg=phi, canvas_H=C_SIZE, canvas_W=C_SIZE,
                L=AP_L, W=AP_W, device=DEVICE
            )
            psf = get_psf_gpu(aperture)
            mtf = get_mtf_gpu(psf)
            APERTURES_GPU.append(aperture)
            PSFS_GPU.append(psf)
            MTFS_GPU.append(mtf)

    for i, phi in enumerate(ANGLES):
        # crop_and_stitch_const_bg handles images of any size (ker_size or C_SIZE)
        padded_img = crop_and_stitch_const_bg(
            CONV_IMGS[i], phi, bg_color=BG_COLOR,
            L=LENGTH, W=WIDTH, canvas_size=C_SIZE, sigma=0
        )
        padded_img_noise = add_poisson_noise(padded_img, peak=1000)
        IMG_CROP.append(padded_img_noise)

        # Compute fftshifted FFT of the padded image on GPU
        padded_gpu = torch.from_numpy(padded_img_noise.astype(np.float32)).to(DEVICE)
        g_fft = torch.fft.fftshift(
            torch.fft.fft2(padded_gpu.to(torch.complex64), dim=(0, 1)),
            dim=(0, 1)
        )                                                         # (C_SIZE, C_SIZE, 3)
        G_IMG_FFTS_GPU.append(g_fft)

    print("Crop & stitch done; FFTs computed on GPU.")

    # ── 7.6  Build weight maps and group dictionaries (CPU) ─────
    _, TOTAL_WEIGHT_MAP, GROUP_ID_ARR, GROUP_MAP, rows, cols = build_weight_maps(
        angles=ANGLES, canvas_size=C_SIZE, L=LENGTH, W=WIDTH
    )
    num_groups = len(GROUP_MAP) - 1   # exclude group 0 (no-coverage border)
    print(f"Number of pixel groups: {num_groups}")

    # ── 7.7  Reconstruct group by group (GPU) ───────────────────
    combined_gpu      = torch.zeros(C_SIZE, C_SIZE, 3,
                                    dtype=torch.float32, device=DEVICE)
    combined_no_h_gpu = torch.zeros(C_SIZE, C_SIZE, 3,
                                    dtype=torch.float32, device=DEVICE)

    parent_dir = os.path.dirname(SAVE_PATH)
    group_save_dir = os.path.join(parent_dir, "single_results")
    os.makedirs(group_save_dir, exist_ok=True)
    data_list = []
    ids = []
    
    # -------NEW in v39: Compute mean DC value from all single-FFT images for DC replacement in reconstruction-------
    CENTER_PX = int(C_SIZE // 2)  # px of DC component
    stacked_fft = torch.stack(G_IMG_FFTS_GPU, dim=0)
    mean_channels = torch.mean(stacked_fft[:, CENTER_PX, CENTER_PX, :], dim=0)
    DC_MEAN = mean_channels.cpu().numpy()  # list: [mean_R, mean_G, mean_B]
    print(f"Position of DC component: ({CENTER_PX}, {CENTER_PX})")
    print(f"Mean DC value from single FFTs: {DC_MEAN}")
    
    for i in range(1, num_groups + 1):
        ang_tup   = GROUP_MAP[i]
        ang_list  = list(ang_tup)
        n_ang     = len(ang_list)

        # Adaptive k based on coverage count
        if   1 <= n_ang <= 4:  k = 0.05
        elif 5 <= n_ang <= 9:  k = 0.05
        elif n_ang >= 10:       k = 0.05
        else:
            print(f"Group {i}: unexpected n_ang={n_ang}, skipping.")
            continue

        combined_gpu, combined_no_h_gpu, rec_no_h_np, rec_np, reconstructed_fft, final_fft, masks, sum_m_mtf, h_filter = fill_corr_px_gpu(
            group_id=i,
            combined_gpu=combined_gpu,
            combined_no_h_gpu=combined_no_h_gpu,
            GROUP_ID_ARR=GROUP_ID_ARR,
            GROUP_MAP=GROUP_MAP,
            k=k,
            canvas_size=C_SIZE,
            aperture_length=AP_L,
            angle_series=ANGLES,
            g_img_ffts_gpu=G_IMG_FFTS_GPU,
            img_mtfs_gpu=MTFS_GPU,
            dc_mean=DC_MEAN,
            center_px=CENTER_PX,
            device=DEVICE
        )

        # Save per-group reconstruction results for debugging
        if rec_np is not None:
            cv2.imwrite(
                os.path.join(group_save_dir, f"{i}.jpg"),
                cv2.cvtColor(
                    (np.clip(rec_np, 0, 1) * 255).astype(np.uint8),
                    cv2.COLOR_RGB2BGR
                )
            )
        if rec_no_h_np is not None:
            cv2.imwrite(
                os.path.join(group_save_dir, f"{i}_no_h.jpg"),
                cv2.cvtColor(
                    (np.clip(rec_no_h_np, 0, 1) * 255).astype(np.uint8),
                    cv2.COLOR_RGB2BGR
                )
            )
            
        if masks is not None:
            np.savez_compressed(os.path.join(group_save_dir, f"{i}_masks.npz"), *[m.cpu().numpy() for m in masks])
            
        if reconstructed_fft is not None:
            np.save(os.path.join(group_save_dir, f"{i}_fft.npy"), reconstructed_fft.cpu().numpy())

        if sum_m_mtf is not None:
            np.save(os.path.join(group_save_dir, f"{i}_sum_m_mtf.npy"), sum_m_mtf.cpu().numpy())
            
        if h_filter is not None:
            np.save(os.path.join(group_save_dir, f"{i}_h_filter.npy"), h_filter.cpu().numpy())
        
        data_list.append(f"Group {i}: {np.max(rec_no_h_np)}")
        ids.append(GROUP_MAP[i])
        
    print("All groups reconstructed.")
    
    data_path = os.path.join(group_save_dir, "max_values.txt")
    with open(data_path, 'w') as f:
        for item1, item2 in zip(ids, data_list):
            f.write(f"{item1} {item2}\n")
    print(f"Reconstruction data saved to: {data_path}")
    
    # Create subdirectory for G_IMG_FFTS_GPU
    fft_save_dir = os.path.join(group_save_dir, "G_IMG_FFTS")
    os.makedirs(fft_save_dir, exist_ok=True)
    # Save each element of G_IMG_FFTS_GPU as a separate numpy file
    for i, g_fft in enumerate(G_IMG_FFTS_GPU):
        np.save(
            os.path.join(fft_save_dir, f"{i}_phi{ANGLES[i]:.0f}.npy"),
            g_fft.cpu().numpy()
        )

    print(f"G_IMG_FFTS_GPU saved to: {fft_save_dir}")

    # ── 7.8  Move results to CPU for display / saving ───────────
    combined_img      = combined_gpu.cpu().numpy()
    combined_img_no_h = combined_no_h_gpu.cpu().numpy()

    # Save final result
    # Create output directory if it does not exist
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    # Save final result (with H-filter)
    result_bgr = cv2.cvtColor(
        (np.clip(combined_img, 0, 1) * 255).astype(np.uint8),
        cv2.COLOR_RGB2BGR
    )
    cv2.imwrite(SAVE_PATH, result_bgr)
    print(f"Saved result to: {SAVE_PATH}")

    # Save result before H-filter
    save_path_no_h = SAVE_PATH.replace('.jpg', '_no_h.jpg')
    result_no_h_bgr = cv2.cvtColor(
        (np.clip(combined_img_no_h, 0, 1) * 255).astype(np.uint8),
        cv2.COLOR_RGB2BGR
    )
    cv2.imwrite(save_path_no_h, result_no_h_bgr)
    print(f"Saved no-H-filter result to: {save_path_no_h}")

    return combined_img, combined_img_no_h


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    combined, combined_no_h = main()