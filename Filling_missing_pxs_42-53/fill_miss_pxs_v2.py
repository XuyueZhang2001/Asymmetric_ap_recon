VERSION = 2
"""
2026.06.22

Modified from v0

Key changes:
  - Solve the problem in binning
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
import os
import torchvision.utils as vutils

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
BIN_FACTOR = 20
IMG_PATH  = '/home/xz127/earth_project/Reconstruction/test3_1600_1200.jpg'
SAVE_PATH = f'/home/xz127/earth_project/Reconstruction/Results/fill_miss_px_v{VERSION}/v{VERSION}_{N_APERTURES}AP_combined_result.jpg'


# ═══════════════════════════════════════════════════════════════
#  Section 1 — Spatial-domain helpers
#  (Still use numpy/cv2 for polygon rasterization; result is
#   transferred to GPU where needed.)
# ═══════════════════════════════════════════════════════════════

# def crop_and_stitch(image_np: np.ndarray, phi_ap_degrees: float,
#                     L: int = LENGTH, W: int = WIDTH,
#                     canvas_size: int = C_SIZE, sigma: float = 10) -> np.ndarray:
#     """
#     Crop a rotated rectangular ROI and place it on a square canvas.
#     Background is filled with the mean color of the cropped region.
#     Returns float32 numpy array (stays on CPU — only called during pre-processing).
#     """
#     h, w = image_np.shape[:2]
#     phi_degrees = phi_ap_degrees + 90

#     canvas_cx, canvas_cy = canvas_size // 2, canvas_size // 2

#     # Rotated rectangle vertices in canvas space
#     rect_points = np.array([[-L/2, -W/2], [L/2, -W/2],
#                              [L/2,  W/2], [-L/2,  W/2]])
#     phi_rad = np.radians(phi_degrees)
#     cos_phi, sin_phi = np.cos(phi_rad), np.sin(phi_rad)
#     rot_mat = np.array([[cos_phi, -sin_phi], [sin_phi, cos_phi]])
#     rotated_pts = (rect_points @ rot_mat.T + [canvas_cx, canvas_cy]).astype(np.int32)

#     # Place image centered on canvas
#     img_cx, img_cy = w // 2, h // 2
#     y_off, x_off = canvas_cy - img_cy, canvas_cx - img_cx
#     temp_canvas = np.zeros((canvas_size, canvas_size, 3), dtype=image_np.dtype)
#     y1, y2 = max(0, y_off), min(canvas_size, y_off + h)
#     x1, x2 = max(0, x_off), min(canvas_size, x_off + w)
#     temp_canvas[y1:y2, x1:x2] = image_np[max(0, -y_off): min(h, canvas_size-y_off),
#                                            max(0, -x_off): min(w, canvas_size-x_off)]

#     # Mean-color background via hard mask
#     hard_mask = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
#     cv2.fillPoly(hard_mask, [rotated_pts], 255)
#     mean_color = cv2.mean(temp_canvas, mask=hard_mask)[:3]

#     # Feathered (Gaussian-blurred) mask for smooth blending
#     feather_mask = np.zeros((canvas_size, canvas_size), dtype=np.float32)
#     cv2.fillPoly(feather_mask, [rotated_pts], 1.0)
#     if sigma > 0:
#         feather_mask = cv2.GaussianBlur(feather_mask, (0, 0), sigmaX=sigma)
#     feather_mask = feather_mask[:, :, np.newaxis]

#     bg_canvas = np.full((canvas_size, canvas_size, 3), mean_color, dtype=image_np.dtype)
#     return temp_canvas * feather_mask + bg_canvas * (1.0 - feather_mask)


def crop_and_stitch_const_bg(image_np: np.ndarray, phi_ap_degrees: float,
                              bg_color=(0, 0, 0),
                              L: int = LENGTH, W: int = WIDTH,
                              canvas_size: int = C_SIZE, sigma: float = 10,
                              bin_factor: int = BIN_FACTOR) -> np.ndarray:
    """
    Crop a rotated rectangular ROI onto a canvas with a fixed background color.
    Optionally bins the ROI along the L (length) direction before pasting it
    back (simulating row/column binning to reduce noise), then resizes back
    to the original ROI size via interpolation.
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

    # Source crop from the input image corresponding to the canvas region above
    src_y1, src_y2 = max(0, -y_off), min(h, canvas_size - y_off)
    src_x1, src_x2 = max(0, -x_off), min(w, canvas_size - x_off)
    roi = image_np[src_y1:src_y2, src_x1:src_x2]

    if bin_factor > 1 and roi.size > 0:
        roi_h, roi_w = roi.shape[:2]

        # Determine binning axis: the ROI's longer dimension is assumed to be
        # the "blurry" / LENGTH direction along which we bin.
        if roi_h >= roi_w:
            bin_axis = 0  # bin along rows
            orig_size = roi_h
        else:
            bin_axis = 1  # bin along columns
            orig_size = roi_w

        # Largest size along bin_axis divisible by bin_factor
        trimmed_size = (orig_size // bin_factor) * bin_factor

        if trimmed_size >= bin_factor:
            if bin_axis == 0:
                roi_trimmed = roi[:trimmed_size, :, :]
                binned_h = trimmed_size // bin_factor
                # Average neighboring rows (group rows by bin_factor and mean)
                roi_binned = roi_trimmed.reshape(
                    binned_h, bin_factor, roi_trimmed.shape[1], roi_trimmed.shape[2]
                ).mean(axis=1)
            else:
                roi_trimmed = roi[:, :trimmed_size, :]
                binned_w = trimmed_size // bin_factor
                # Average neighboring columns (group columns by bin_factor and mean)
                roi_binned = roi_trimmed.reshape(
                    roi_trimmed.shape[0], binned_w, bin_factor, roi_trimmed.shape[2]
                ).mean(axis=2)

            roi_binned = roi_binned.astype(image_np.dtype)

            # Interpolate back to the original ROI size
            roi = cv2.resize(
                roi_binned,
                (roi_w, roi_h),
                interpolation=cv2.INTER_LINEAR
            ).astype(image_np.dtype)

    temp_canvas[y1:y2, x1:x2] = roi

    feather_mask = np.zeros((canvas_size, canvas_size), dtype=np.float32)
    cv2.fillPoly(feather_mask, [rotated_pts], 1.0)
    if sigma > 0:
        feather_mask = cv2.GaussianBlur(feather_mask, (0, 0), sigmaX=sigma)
    feather_mask = feather_mask[:, :, np.newaxis]

    bg_canvas = np.full((canvas_size, canvas_size, 3), bg_color, dtype=image_np.dtype)
    return temp_canvas * feather_mask + bg_canvas * (1.0 - feather_mask), roi

def crop_and_rotate(image, phi_ap_degrees, L=LENGTH, W=WIDTH, bin_factor=BIN_FACTOR):
    """
    Extracts a region of size L x W from the original image. 
    The window is rotated phi degrees clockwise, but the IMAGE CONTENT 
    inside is NOT rotated (it maintains its original orientation).
    
    For the result, it will rotate phi degrees counterclockwise to make the window 'straight' in the output,
    
    Args:
        image (numpy.ndarray): The source image.
        phi_degrees (float): Clockwise rotation of the selection window.
        L (int): Length of the crop window.
        W (int): Width of the crop window.
    """
    target_w, target_h = int(L), int(W)
    img_h, img_w = image.shape[:2]
    center_source = (img_w / 2.0, img_h / 2.0)
    phi_degrees = phi_ap_degrees - 90

    # 1. To keep the content orientation UNCHANGED while the window is rotated,
    # we use an identity rotation matrix for the content but adjust the 
    # sampling coordinates based on the window rotation.
    
    # We essentially want the 'camera' to be rotated phi degrees, 
    # but the 'world' to stay still.
    
    # Calculate the rotation matrix M
    # We use -phi_degrees to represent a clockwise rotation in OpenCV
    M = cv2.getRotationMatrix2D(center_source, phi_degrees, 1.0)
    
    # 2. Adjust translation so the center of the rotated window 
    # becomes the center of our L x W output
    M[0, 2] += (target_w / 2.0) - center_source[0]
    M[1, 2] += (target_h / 2.0) - center_source[1]

    # 3. Perform the warp
    # This will result in the window being 'straight' in the output file,
    # but because the mapping was rotated, the building inside will 
    # appear tilted exactly like your right-hand example.
    result = cv2.warpAffine(
        image, 
        M, 
        (target_w, target_h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    rot_result = np.rot90(result, k=1)
    
    # Binning
    if bin_factor > 1 and rot_result.size > 0:
        roi_h, roi_w = rot_result.shape[:2]

        # Determine binning axis: the ROI's longer dimension is assumed to be
        # the "blurry" / LENGTH direction along which we bin.
        if roi_h >= roi_w:
            bin_axis = 0  # bin along rows
            orig_size = roi_h
        else:
            bin_axis = 1  # bin along columns
            orig_size = roi_w

        # Largest size along bin_axis divisible by bin_factor
        trimmed_size = (orig_size // bin_factor) * bin_factor

        if trimmed_size >= bin_factor:
            if bin_axis == 0:
                roi_trimmed = rot_result[:trimmed_size, :, :]
                binned_h = trimmed_size // bin_factor
                # Average neighboring rows (group rows by bin_factor and mean)
                roi_binned = roi_trimmed.reshape(
                    binned_h, bin_factor, roi_trimmed.shape[1], roi_trimmed.shape[2]
                ).mean(axis=1)
            else:
                roi_trimmed = rot_result[:, :trimmed_size, :]
                binned_w = trimmed_size // bin_factor
                # Average neighboring columns (group columns by bin_factor and mean)
                roi_binned = roi_trimmed.reshape(
                    roi_trimmed.shape[0], binned_w, bin_factor, roi_trimmed.shape[2]
                ).mean(axis=2)

            roi_binned = roi_binned.astype(image.dtype)

            # Interpolate back to the original ROI size
            roi = cv2.resize(
                roi_binned,
                (roi_w, roi_h),
                interpolation=cv2.INTER_LINEAR
            ).astype(image.dtype)
    return roi

from scipy.ndimage import rotate, gaussian_filter
def rotate_and_blend(img, phi_deg, canvas_H=C_SIZE, canvas_W=C_SIZE, 
                     bg_color=(0,0,0), sigma=15,):
    """
    Rotate an image clockwise by phi_deg degrees, paste it onto a canvas 
    with a background color similar to the image, and apply Gaussian 
    blending at the edges.
 
    Parameters
    ----------
    img : np.ndarray
        Input image, shape (H, W, 3) or (H, W), dtype uint8 or float.
    phi_deg : float
        Clockwise rotation angle in degrees.
    canvas_H : int or None
        Canvas height. If None, defaults to 1.5x the image height.
    canvas_W : int or None
        Canvas width. If None, defaults to 1.5x the image width.
    sigma : float
        Standard deviation of the Gaussian blur applied to the alpha mask 
        for edge blending. Larger = softer transition.
    bg_color : tuple/list or None
        Background color as (R, G, B). If None, automatically estimated 
        from the image border pixels.
 
    Returns
    -------
    result : np.ndarray
        Blended image on canvas, same dtype as input, shape (canvas_H, canvas_W, 3).
    """
    is_float = img.dtype in [np.float32, np.float64]
    if not is_float:
        img = img.astype(np.float64) / 255.0
 
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
 
    h, w = img.shape[:2]
 
    if canvas_H is None:
        canvas_H = int(h * 1.5)
    if canvas_W is None:
        canvas_W = int(w * 1.5)
 
    # --- Step 1: Estimate background color from image border pixels ---
    if bg_color is None:
        border_pixels = np.concatenate([
            img[0, :, :],           # top row
            img[-1, :, :],          # bottom row
            img[:, 0, :],           # left column
            img[:, -1, :],          # right column
        ], axis=0)
        bg_color = np.median(border_pixels, axis=0)
    else:
        bg_color = np.array(bg_color, dtype=np.float64)
        if bg_color.max() > 1.0:
            bg_color = bg_color / 255.0
 
    # --- Step 2: Create a binary mask (ones where image exists) ---
    mask = np.ones((h, w), dtype=np.float64)
    
    # In rotate_and_blend, before rotation, add padding with bg_color
    # --- Step 2.5: Pad image with bg_color to avoid black edge artifacts ---
    pad = 5  # a few pixels is enough
    img_padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
    mask = np.ones((img_padded.shape[0], img_padded.shape[1]), dtype=np.float64)
    # Set mask padding region to 0 so it still blends out
    mask[:pad, :] = 0
    mask[-pad:, :] = 0
    mask[:, :pad] = 0
    mask[:, -pad:] = 0

    # Then rotate img_padded and mask as before
    rotated_img = rotate(img_padded, angle=-phi_deg, reshape=True, order=3,
                        mode='constant', cval=0.0)
    rotated_mask = rotate(mask, angle=-phi_deg, reshape=True, order=1,
                        mode='constant', cval=0.0)

    # --- Step 3: Rotate image and mask ---
    # scipy.ndimage.rotate uses counter-clockwise convention,
    # so negate the angle for clockwise rotation.
    # Use order=1 (bilinear) for speed, order=3 (cubic) for quality.

 
    rh, rw = rotated_img.shape[:2]
 
    # --- Step 4: Apply Gaussian blur to the mask for soft edges ---
    blended_mask = gaussian_filter(rotated_mask, sigma=sigma)
    # Clip to [0, 1] to avoid overshoot
    blended_mask = np.clip(blended_mask, 0, 1)
 
    # --- Step 5: Create canvas filled with background color ---
    canvas = np.ones((canvas_H, canvas_W, 3), dtype=np.float64)
    canvas[:, :, 0] *= bg_color[0]
    canvas[:, :, 1] *= bg_color[1]
    canvas[:, :, 2] *= bg_color[2]
 
    # --- Step 6: Compute placement offset to center the rotated image ---
    y_offset = (canvas_H - rh) // 2
    x_offset = (canvas_W - rw) // 2
 
    # Handle cases where rotated image is larger than canvas
    # Source region (from rotated image)
    src_y_start = max(0, -y_offset)
    src_x_start = max(0, -x_offset)
    src_y_end = min(rh, canvas_H - y_offset)
    src_x_end = min(rw, canvas_W - x_offset)
 
    # Destination region (on canvas)
    dst_y_start = max(0, y_offset)
    dst_x_start = max(0, x_offset)
    dst_y_end = dst_y_start + (src_y_end - src_y_start)
    dst_x_end = dst_x_start + (src_x_end - src_x_start)
 
    # --- Step 7: Alpha-blend rotated image onto canvas ---
    region_mask = blended_mask[src_y_start:src_y_end, src_x_start:src_x_end]
    region_img = rotated_img[src_y_start:src_y_end, src_x_start:src_x_end]
    alpha = region_mask[:, :, np.newaxis]  # expand to (H, W, 1) for broadcasting
 
    canvas[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = (
        alpha * region_img + (1 - alpha) * canvas[dst_y_start:dst_y_end, dst_x_start:dst_x_end]
    )
 
    # --- Step 8: Convert back to uint8 if needed ---
    if not is_float:
        canvas = np.clip(canvas * 255, 0, 255).astype(np.uint8)
 
    return canvas
 


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

def get_contributor_map(weights: np.ndarray, #(N_APERTURE, C_SIZE, C_SIZE), the same as weight_stack
                        c_size=C_SIZE, 
                        n_apertures=N_APERTURES
                        )->np.ndarray:
    # Initialize every pixel with an empty list
    contributor_map = np.empty((c_size, c_size), dtype=object)
    for y in range(c_size):
        for x in range(c_size):
            contributor_map[y, x] = []

    # Iterate over apertures only — O(N_APERTURE) outer loops
    for aid in range(n_apertures):
        mask = weights[aid].astype(bool)          # (C_SIZE, C_SIZE) bool
        ys, xs = np.where(mask)
        for y, x in zip(ys, xs):
            contributor_map[y, x].append(aid)
    return contributor_map

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
#  Section 2 — GPU-based optics functions & fill in empty pxs
# ═══════════════════════════════════════════════════════════════

def get_physical_aperture_gpu(phi_ap_deg: float,
                               canvas_H: int = C_SIZE, canvas_W: int = C_SIZE,
                               L: float = AP_L, W: float = AP_W,
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

def get_round_aperture_gpu(
    D: int,
    canvas_H: int,
    canvas_W: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Generate a circular aperture mask on the GPU.
    """
    # Pixel coordinate grids centred at the canvas centre
    cy = (canvas_H - 1) / 2.0
    cx = (canvas_W - 1) / 2.0

    ys = torch.arange(canvas_H, dtype=torch.float32, device=device)  # (H,)
    xs = torch.arange(canvas_W, dtype=torch.float32, device=device)  # (W,)

    # Broadcast to (H, W)
    dy = ys.unsqueeze(1) - cy   # (H, 1)
    dx = xs.unsqueeze(0) - cx   # (1, W)

    dist2 = dy ** 2 + dx ** 2   # (H, W) squared distance from centre

    radius = D / 2.0
    aperture = (dist2 <= radius ** 2).to(torch.float32)   # (H, W)

    return aperture

def get_psf_gpu(aperture: torch.Tensor) -> torch.Tensor:
    """
    Compute normalised PSF from a binary aperture tensor on GPU.

    The aperture is first shifted to the DC corner (ifftshift) before the
    FFT so that the zero-frequency component is at the origin as expected
    by the FFT convention. The resulting intensity PSF is then shifted back
    to the centre (fftshift) for display and use.

    Args:
        aperture : (H, W) float32 tensor, binary aperture mask

    Returns:
        psf      : (H, W) float32 normalised PSF tensor
    """
    # Shift aperture centre to DC corner before FFT
    aperture_shifted = torch.fft.ifftshift(aperture.to(torch.complex64))

    # FFT; norm='ortho' keeps numerical scale reasonable
    ft = torch.fft.fft2(aperture_shifted, norm='ortho')

    # Intensity PSF = |FT|^2, shifted back to centre for display/use
    psf = torch.fft.fftshift(ft.abs() ** 2)

    # Normalise so that PSF sums to 1 (energy conservation)
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

# def conv_pad_2D_gpu(
#     img: torch.Tensor,
#     psf: torch.Tensor,
#     bg_color: tuple = (0,0,0),
# ) -> torch.Tensor:
#     """
#     Convolve a 3-channel image with a PSF on the GPU, using background-colour
#     padding to reduce wrap-around / edge artefacts.

#     Both img and psf are padded to twice their original spatial size before
#     the FFT-based convolution; the result is then centre-cropped back to the
#     original image size.
#     """
#     device = img.device
#     H, W = img.shape[0], img.shape[1]

#     pad_H = H * 2
#     pad_W = W * 2

#     bg = torch.tensor(bg_color, dtype=img.dtype, device=device)  # (3,)
#     # Start from a canvas filled with the background colour
#     img_padded = bg.view(1, 1, 3).expand(pad_H, pad_W, 3).clone()  # (2H, 2W, 3)
#     img_padded[:H, :W, :] = img

#     psf_padded = torch.zeros(pad_H, pad_W, dtype=psf.dtype, device=device)
#     psf_padded[:H, :W] = psf
#     # ifftshift moves the PSF centre to the DC corner
#     psf_padded = torch.fft.ifftshift(psf_padded)

#     PSF_F = torch.fft.rfft2(psf_padded)                      # (2H, W+1) complex

#     # img_padded: (2H, 2W, 3) -> process each channel
#     img_padded_chw = img_padded.permute(2, 0, 1)             # (3, 2H, 2W)
#     IMG_F = torch.fft.rfft2(img_padded_chw)                  # (3, 2H, W+1) complex

#     # Multiply in frequency domain (PSF broadcast over channel dimension)
#     OUT_F = IMG_F * PSF_F.unsqueeze(0)                       # (3, 2H, W+1)

#     out_padded = torch.fft.irfft2(OUT_F, s=(pad_H, pad_W))  # (3, 2H, 2W)

#     crop_y = H // 2
#     crop_x = W // 2
#     out_chw = out_padded[:, crop_y:crop_y + H, crop_x:crop_x + W]  # (3, H, W)

#     out = out_chw.permute(1, 2, 0)                           # (H, W, 3)

#     return out

def conv_pad_2D_gpu(
    img: torch.Tensor,
    psf: torch.Tensor,
    bg_color: tuple = (0, 0, 0),
) -> torch.Tensor:
    """
    Convolve a 3-channel image with a PSF on the GPU, using background-colour
    padding to reduce wrap-around / edge artefacts.

    Both img and psf are padded to twice their original spatial size before
    the FFT-based convolution; the result is then centre-cropped back to the
    original image size.
    """
    device = img.device
    H, W = img.shape[0], img.shape[1]

    pad_H = H * 2
    pad_W = W * 2

    # Pad image with background colour; place original in top-left corner
    bg = torch.tensor(bg_color, dtype=img.dtype, device=device)
    img_padded = bg.view(1, 1, 3).expand(pad_H, pad_W, 3).clone()  # (2H, 2W, 3)
    img_padded[:H, :W, :] = img

    # Shift PSF centre to DC corner at its ORIGINAL size first,
    # then place into the top-left corner of the padded canvas.
    # This ensures the DC corner of the padded PSF is at (0, 0) as required.
    psf_dc = torch.fft.ifftshift(psf)                        # (H, W) centre -> DC corner
    psf_padded = torch.zeros(pad_H, pad_W, dtype=psf.dtype, device=device)
    psf_padded[:H, :W] = psf_dc

    # FFT-based convolution
    PSF_F = torch.fft.rfft2(psf_padded)                      # (2H, pad_W//2+1)

    img_padded_chw = img_padded.permute(2, 0, 1)             # (3, 2H, 2W)
    IMG_F = torch.fft.rfft2(img_padded_chw)                  # (3, 2H, pad_W//2+1)

    OUT_F = IMG_F * PSF_F.unsqueeze(0)                       # (3, 2H, pad_W//2+1)

    out_padded = torch.fft.irfft2(OUT_F, s=(pad_H, pad_W))  # (3, 2H, 2W)

    # Crop the valid centre region back to original size.
    # With the PSF DC at (0,0) and the image in the top-left, the valid
    # output starts at (0, 0) — no offset needed.
    out_chw = out_padded[:, :H, :W]                          # (3, H, W)

    out = out_chw.permute(1, 2, 0)                           # (H, W, 3)

    return out


def fftshift2_gpu(x: torch.Tensor, axes=(-2, -1)) -> torch.Tensor:
    """torch.fft.fftshift over specified axes."""
    return torch.fft.fftshift(x, dim=axes)


def ifftshift2_gpu(x: torch.Tensor, axes=(-2, -1)) -> torch.Tensor:
    """torch.fft.ifftshift over specified axes."""
    return torch.fft.ifftshift(x, dim=axes)

import numpy as np
import torch


def fill_missing_pixels_gpu(
    img_list: list,
    weights: np.ndarray,
    contributor_map: np.ndarray,
) -> list:
    """
    GPU-accelerated version of fill_missing_pixels.

    For each aperture image, fill pixels that lie within some aperture's
    footprint but are NOT observed by this aperture (weights[id]==0).
    Each such pixel is replaced by the average value across ALL apertures
    that DO observe that pixel (as recorded in contributor_map).

    Args:
        img_list        : list of N_APERTURE RGB images, each (H, W, 3), float [0,1]
        weights         : binary ndarray (N_APERTURE, H, W)
        contributor_map : object ndarray (H, W), each cell is a list/array of
                          aperture IDs that observe that pixel

    Returns:
        filled_crops    : list of N_APERTURE filled images as ndarrays, same
                          shape as input
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    N_APERTURE = len(img_list)
    H, W = weights.shape[1], weights.shape[2]

    # Build a dense padded contributor tensor from the object array.
    max_k = int(max(
        len(contributor_map[y, x])
        for y in range(H) for x in range(W)
        if len(contributor_map[y, x]) > 0
    ))

    contributors_np = np.full((H, W, max_k), -1, dtype=np.int32)
    for y in range(H):
        for x in range(W):
            ids = contributor_map[y, x]
            if len(ids) > 0:
                contributors_np[y, x, :len(ids)] = ids

    # Boolean mask: True where the slot holds a real contributor ID
    valid_mask_np = contributors_np >= 0                      # (H, W, MAX_K)

    # Move everything to GPU
    contributors = torch.from_numpy(contributors_np).to(device)   # (H, W, K) int32
    valid_mask   = torch.from_numpy(valid_mask_np).to(device)     # (H, W, K) bool
    weights_t    = torch.from_numpy(weights.astype(bool)).to(device)  # (N, H, W)
    img_stack    = torch.from_numpy(
        np.stack(img_list, axis=0).astype(np.float32)
    ).to(device)                                              # (N, H, W, 3)

    # Determine which (aperture, pixel) pairs need filling
    any_contributor = valid_mask.any(dim=-1)                  # (H, W)
    need_fill = any_contributor.unsqueeze(0) & ~weights_t     # (N, H, W)

    # Compute the per-pixel average across all contributing apertures.
    # contributors_safe replaces -1 padding with 0 so it can be used as a
    # gather index without going out of bounds; the validity mask will zero
    # out the padded slots before the mean is taken.
    contributors_safe = contributors.clamp(min=0).long()      # (H, W, K)

    # Flatten spatial dims to simplify indexing
    # img_flat : (N, HW, 3)
    # contrib_flat : (HW, K)  — pixel-level contributor ID table
    # valid_flat   : (HW, K)  — corresponding validity mask
    img_flat     = img_stack.view(N_APERTURE, H * W, 3)       # (N, HW, 3)
    contrib_flat = contributors_safe.view(H * W, max_k)       # (HW, K)
    valid_flat   = valid_mask.view(H * W, max_k)              # (HW, K)

    HW = H * W

    # For every pixel location hw and every candidate slot k, fetch the
    # pixel value from the contributing aperture.
    # contrib_flat[hw, k] is the aperture ID; we need img_flat[id, hw, :].
    # Expand contrib_flat to (HW, K) and gather from img_flat (N, HW, 3).
    # Reindex: treat HW as the "batch" dimension.
    #   src_ids  : (HW, K)   aperture indices
    #   hw_idx   : (HW, K)   spatial indices (just 0..HW-1 repeated K times)
    hw_idx  = torch.arange(HW, device=device).unsqueeze(1).expand(HW, max_k)  # (HW, K)

    # candidate_pixels[hw, k, c] = img_flat[ contrib_flat[hw,k], hw, c ]
    # img_flat is (N, HW, 3); index with (aperture=contrib_flat, spatial=hw_idx)
    candidate_pixels = img_flat[
        contrib_flat,   # (HW, K) — selects along the N dimension
        hw_idx          # (HW, K) — selects along the HW dimension
    ]                                                         # (HW, K, 3)

    # Zero out padded slots so they do not contribute to the sum
    valid_for_mean = valid_flat.unsqueeze(-1).float()         # (HW, K, 1)
    candidate_pixels = candidate_pixels * valid_for_mean      # (HW, K, 3)

    # Sum over contributors and divide by the count of valid contributors
    pixel_sum   = candidate_pixels.sum(dim=1)                 # (HW, 3)
    pixel_count = valid_flat.sum(dim=1, keepdim=True).float() # (HW, 1)
    pixel_count = pixel_count.clamp(min=1)                    # avoid division by zero
    avg_pixels  = pixel_sum / pixel_count                     # (HW, 3)

    # avg_pixels is the same for every aperture (it depends only on the pixel
    # location, not on which aperture is being filled), so expand to (N, HW, 3)
    avg_pixels_exp = avg_pixels.unsqueeze(0).expand(N_APERTURE, HW, 3)  # (N, HW, 3)

    need_flat   = need_fill.view(N_APERTURE, HW)             # (N, HW)

    filled_flat = torch.where(
        need_flat.unsqueeze(-1),   # (N, HW, 1) broadcast over channels
        avg_pixels_exp,
        img_flat
    )                                                         # (N, HW, 3)

    filled_stack = filled_flat.view(N_APERTURE, H, W, 3)     # (N, H, W, 3)

    filled_np    = filled_stack.cpu().numpy()
    filled_crops = [filled_np[i] for i in range(N_APERTURE)]

    return filled_crops

def add_poisson_noise_gpu(img, peak=1000.0, device='cuda'):
    is_numpy = isinstance(img, np.ndarray)
    if is_numpy:
        img = torch.from_numpy(img).float().to(device)

    img_clip = torch.clamp(img, min=0)
    scaled = img_clip * peak
    noisy = torch.poisson(scaled) / peak

    if is_numpy:
        noisy = noisy.cpu().numpy()

    return noisy

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
    ker_size = min(IMG_FULL.shape[0], IMG_FULL.shape[1], C_SIZE)
    img_for_conv = crop_center_np(IMG_FULL, cropx=ker_size, cropy=ker_size)

    # ── 7.3  Upload crop to GPU ──────────────────────────────────
    img_for_conv_gpu = torch.from_numpy(img_for_conv).to(DEVICE)   # (H, W, 3)

    # ── 7.4  Compute apertures / PSFs / MTFs and convolve (GPU) ──
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
        conv_result_noise = add_poisson_noise_gpu(conv_result_gpu)

        APERTURES_GPU.append(aperture)
        PSFS_GPU.append(psf)
        MTFS_GPU.append(mtf)
        CONV_IMGS.append(conv_result_noise.cpu().numpy())

    print("Convolution done on GPU.")
    
    # ── 7.5  Build weight maps and group dictionaries (CPU) ─────
    w_stack, TOTAL_WEIGHT_MAP, GROUP_ID_ARR, GROUP_MAP, rows, cols = build_weight_maps(
        angles=ANGLES, canvas_size=C_SIZE, L=LENGTH, W=WIDTH
    )
    num_groups = len(GROUP_MAP) - 1   # exclude group 0 (no-coverage border)
    print(f"Number of pixel groups: {num_groups}")
    CONTR_MAP = get_contributor_map(weights = w_stack)
    print("Contributor map finished.")

    # ── 7.6  Crop & stitch convolved images onto canvas (CPU), fill empty pxs ────
    # (sigma=0 matches the v35 notebook — no Gaussian feathering here)
    IMG_CROP    = []
    G_IMG_FFTS_GPU = []   # complex64 fftshifted GPU tensors (C_SIZE, C_SIZE, 3)
    IMG_ROI = []

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
        # padded_img, roi = crop_and_stitch_const_bg(
        #     CONV_IMGS[i], phi, bg_color=BG_COLOR,
        #     L=LENGTH, W=WIDTH, canvas_size=C_SIZE, sigma=0, bin_factor=BIN_FACTOR
        # )
        roi = crop_and_rotate(image=CONV_IMGS[i], phi_ap_degrees=phi)
        padded_img = rotate_and_blend(img=roi, phi_deg=phi, canvas_H=C_SIZE, canvas_W=C_SIZE,
                                      bg_color=BG_COLOR, sigma=0)

        IMG_CROP.append(padded_img)
        IMG_ROI.append(roi)
    print("Crop and stitch done")
    
    SINGLE_FFT = []
    for i, phi in enumerate(ANGLES):
    # Compute fftshifted FFT of the padded image on GPU
        single_gpu = torch.from_numpy(IMG_CROP[i].astype(np.float32)).to(DEVICE)
        g_fft = torch.fft.fftshift(
            torch.fft.fft2(single_gpu.to(torch.complex64), dim=(0, 1)),
            dim=(0, 1)
        )                                                         # (C_SIZE, C_SIZE, 3)
        SINGLE_FFT.append(g_fft)

    IMG_CROP_FILL = fill_missing_pixels_gpu(img_list = IMG_CROP, 
                                        weights = w_stack, 
                                        contributor_map = CONTR_MAP)
    print("Fill empty pxs done.")
    
    # Create subdirectory for IMG_CROP_FILL
    parent_dir = os.path.dirname(SAVE_PATH)
    group_save_dir = os.path.join(parent_dir, "single_results")
    os.makedirs(group_save_dir, exist_ok=True)
    
    f_img_save_dir = os.path.join(group_save_dir, "IMG_CROP_FILL")
    os.makedirs(f_img_save_dir, exist_ok=True)
    # Save each element of G_IMG_FFTS_GPU as a separate numpy file
    for i, f_img in enumerate(IMG_CROP_FILL):
        f_img_bgr = cv2.cvtColor(
            (np.clip(f_img, 0, 1) * 255).astype(np.uint8),
            cv2.COLOR_RGB2BGR
        )
        f_path = os.path.join(f_img_save_dir, f"image_{i}_phi{ANGLES[i]:.0f}.jpg")
        cv2.imwrite(f_path, f_img_bgr)
    print(f"Filled images saved to {f_img_save_dir}")
    
    f_img_save_dir = os.path.join(group_save_dir, "IMG_SINGLE")
    os.makedirs(f_img_save_dir, exist_ok=True)
    for i, f_img in enumerate(IMG_CROP):
        f_img_bgr = cv2.cvtColor(
            (np.clip(f_img, 0, 1) * 255).astype(np.uint8),
            cv2.COLOR_RGB2BGR
        )
        f_path = os.path.join(f_img_save_dir, f"image_{i}_phi{ANGLES[i]:.0f}.jpg")
        cv2.imwrite(f_path, f_img_bgr)
    print(f"Single images saved to {f_img_save_dir}")
    
    f_img_save_dir = os.path.join(group_save_dir, "ROI")
    os.makedirs(f_img_save_dir, exist_ok=True)
    for i, f_img in enumerate(IMG_ROI):
        f_img_bgr = cv2.cvtColor(
            (np.clip(f_img, 0, 1) * 255).astype(np.uint8),
            cv2.COLOR_RGB2BGR
        )
        f_path = os.path.join(f_img_save_dir, f"image_{i}_phi{ANGLES[i]:.0f}.jpg")
        cv2.imwrite(f_path, f_img_bgr)
    print(f"ROI images saved to {f_img_save_dir}")
    
    
    # --------------------NEW in v45: blur the pxs out side the weighting map
    aperture_round = get_round_aperture_gpu(D = AP_L, canvas_H=C_SIZE, canvas_W=C_SIZE, device=DEVICE)
    psf_round = get_psf_gpu(aperture_round)
    # Conv the round psf
    FINAL_IMG = []
    for i, filled_img in enumerate(IMG_CROP_FILL):
        fill_img_gpu = torch.from_numpy(filled_img).to(DEVICE)
        conv_img_gpu = convolve_with_psf_gpu(fill_img_gpu, psf_round)
        # conv_img_gpu = conv_pad_2D_gpu(fill_img_gpu, psf_round, bg_color = BG_COLOR)
        weight_map_t = torch.from_numpy(w_stack[i, :, :]).to(DEVICE)
        # conv_img_gpu = torch.where(weight_map_t.unsqueeze(-1).bool(), fill_img_gpu, conv_img_gpu)
        # FINAL_IMG.append(conv_img_gpu.cpu().numpy())
        # Blur the binary weight map to create a soft transition at boundaries.
        # Gaussian blur via two separable 1D convolutions in the frequency domain.
        # Sigma controls the width of the feathering zone in pixels.
        sigma = 50.0
        ksize = int(6 * sigma + 1) | 1                          # odd kernel size
        half  = ksize // 2
        coords = torch.arange(ksize, dtype=torch.float32, device=DEVICE) - half
        kernel_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()                 # normalise

        # Apply separable Gaussian blur to the weight map using F.conv2d
        weight_map_t = torch.from_numpy(w_stack[i, :, :]).to(DEVICE).float()
        w = weight_map_t.unsqueeze(0).unsqueeze(0).float()   # ensure float32 before conv2d # (1, 1, H, W)          
        k_h = kernel_1d.view(1, 1, ksize, 1)
        k_w = kernel_1d.view(1, 1, 1, ksize)
        w = F.conv2d(w, k_h, padding=(half, 0))
        w = F.conv2d(w, k_w, padding=(0, half))
        soft_weight = w.squeeze(0).squeeze(0)                   # (H, W) in [0, 1]

        # Optional: apply a non-linear curve to the soft weight for a smoother blend.
        # smoothstep: 3t^2 - 2t^3  (S-curve, zero derivative at 0 and 1)
        soft_weight = soft_weight ** 2 * (3.0 - 2.0 * soft_weight)

        # Weighted blend: weight=1 -> original, weight=0 -> convolved
        soft_weight = soft_weight.unsqueeze(-1)                 # (H, W, 1) broadcast over RGB
        blended = soft_weight * fill_img_gpu + (1.0 - soft_weight) * conv_img_gpu

        FINAL_IMG.append(blended.cpu().numpy())
    print("Blur the outside pixels done.")
        
    f_img_save_dir = os.path.join(group_save_dir, "IMG_FINAL")
    os.makedirs(f_img_save_dir, exist_ok=True)
    # Save each element of G_IMG_FFTS_GPU as a separate numpy file
    for i, f_img in enumerate(FINAL_IMG):
        f_img_bgr = cv2.cvtColor(
            (np.clip(f_img, 0, 1) * 255).astype(np.uint8),
            cv2.COLOR_RGB2BGR
        )
        f_path = os.path.join(f_img_save_dir, f"image_{i}_phi{ANGLES[i]:.0f}.jpg")
        cv2.imwrite(f_path, f_img_bgr)
    print(f"Filled images saved to {f_img_save_dir}")
    
    psf_path = os.path.join(f_img_save_dir, f"psf.jpg")
    # Log scale to compress dynamic range
    psf_vis = torch.log1p(psf_round)
    # Normalise to [0, 1]
    psf_vis = (psf_vis - psf_vis.min()) / (psf_vis.max() - psf_vis.min())
    # Save as image
    vutils.save_image(psf_vis.unsqueeze(0), psf_path)
    
    ap_path = os.path.join(f_img_save_dir, f"aperture.jpg")
    ap_img = (aperture_round.cpu().numpy() * 255).astype(np.uint8)
    cv2.imwrite(ap_path, ap_img)
    print(ap_img.shape)
    
    for i, phi in enumerate(ANGLES):
        # Compute fftshifted FFT of the padded image on GPU
        padded_gpu = torch.from_numpy(FINAL_IMG[i].astype(np.float32)).to(DEVICE)
        g_fft = torch.fft.fftshift(
            torch.fft.fft2(padded_gpu.to(torch.complex64), dim=(0, 1)),
            dim=(0, 1)
        )                                                         # (C_SIZE, C_SIZE, 3)
        G_IMG_FFTS_GPU.append(g_fft)

    print("FFTs computed on GPU.")
    
    # Create subdirectory for SINGLE_FFT
    fft_save_dir = os.path.join(group_save_dir, "SINGLE_FFTS")
    os.makedirs(fft_save_dir, exist_ok=True)
    # Save each element of G_IMG_FFTS_GPU as a separate numpy file
    for i, g_fft in enumerate(SINGLE_FFT):
        np.save(
            os.path.join(fft_save_dir, f"{i}_phi{ANGLES[i]:.0f}.npy"),
            g_fft.cpu().numpy()
        )

    print(f"SINGLE_FFT saved to: {fft_save_dir}")
    
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
    
    # print(test)

    # ── 7.7  Reconstruct group by group (GPU) ───────────────────
    combined_gpu      = torch.zeros(C_SIZE, C_SIZE, 3,
                                    dtype=torch.float32, device=DEVICE)
    combined_no_h_gpu = torch.zeros(C_SIZE, C_SIZE, 3,
                                    dtype=torch.float32, device=DEVICE)

    
    
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