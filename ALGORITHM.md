# Multi-Aperture Frequency-Domain Fusion with DC Calibration and Soft-Weight Stitching

Simulation-and-reconstruction pipeline for a rotating slit aperture. A single
scene is observed through `N_APERTURES = 16` rectangular apertures at angles
`0°…180°`, each producing a directionally blurred, Poisson-noisy measurement.
The observations are fused in the frequency domain with per-angle bow-tie masks,
sharpened by a Wiener-like `H`-filter, and stitched back into a single canvas.

Illustration for fill_miss_pxs_v3.py

## Data flow (forward pass)

```
                          INPUT IMAGE
                    IMG_FULL ∈ R^(H×W×3), [0,1]
                              │
                              ▼
              ┌──────────────────────────────────┐
              │  BG_COLOR = mean(center crop)    │
              │  used as canvas / pad fill value │
              └──────────────┬───────────────────┘
                              │
   ══════════════════════════╪═══════════════════════════════════════
        FORWARD MODEL (per angle φ ∈ ANGLES, 16×)
                              │
                              ▼
              ┌──────────────────────────────────┐
              │ crop_and_rotate(φ)               │
              │  rotated L×W window, content     │
              │  orientation preserved           │
              └──────────────┬───────────────────┘
                              │ roi
                              ▼
              ┌──────────────────────────────────┐
              │ pad_to_size(BG_COLOR)            │
              │  → (L+200, W+200, 3)             │
              └──────────────┬───────────────────┘
                              │
                              ▼        ┌───────────────────────────┐
              ┌───────────────────────┐│ aperture(φ=0), L×W rect   │
              │ convolve_with_psf_gpu │◄┤ PSF = |FFT(ifftshift(A))|²│
              │  x*psf via FFT        ││ normalized ΣPSF = 1       │
              └──────────────┬────────┘└───────────────────────────┘
                              │
                              ▼
              ┌──────────────────────────────────┐
              │ add_poisson_noise(peak=1000)     │  (if NOISE)
              └──────────────┬───────────────────┘
                              │
                              ▼
              ┌──────────────────────────────────┐
              │ bin_blurring_direction(BIN_FACTOR)│
              │  average along long axis, resize  │
              │  back (row/column binning)        │
              └──────────────┬───────────────────┘
                              │
                              ▼
              ┌──────────────────────────────────┐
              │ rotate_and_blend(φ) → C_SIZE²    │
              │  paste back at original angle    │
              └──────────────┬───────────────────┘
                              │  IMG_CROP[i]
   ══════════════════════════╪═══════════════════════════════════════
        MISSING-PIXEL FILL
                              │
                              ▼
              ┌──────────────────────────────────┐
              │ w_stack[i] = binary footprint(φᵢ)│
              │ CONTR_MAP[y,x] = {i : covered}   │
              └──────────────┬───────────────────┘
                              │
                              ▼
              ┌──────────────────────────────────┐
              │ fill_missing_pixels_gpu          │
              │  pixels seen by SOME aperture    │
              │  but not by this one ←           │
              │  mean over all contributors      │
              └──────────────┬───────────────────┘
                              │  IMG_CROP_FILL[i]
                              ▼
              ┌──────────────────────────────────┐
              │ blur outside the footprint:      │
              │  conv with PSF(φᵢ), then blend   │
              │  w = smoothstep(G_σ=500 * w_map) │
              │  out = w·filled + (1−w)·conv     │
              └──────────────┬───────────────────┘
                              │  FINAL_IMG[i]
                              ▼
              ┌──────────────────────────────────┐
              │ G_IMG_FFTS[i] = fftshift(FFT2)   │
              └──────────────┬───────────────────┘
                              │
   ══════════════════════════╪═══════════════════════════════════════
        DC CALIBRATION
                              │
                              ▼
       ┌─────────────────────────────────────────────┐
       │ DC_MEAN = mean_i G_IMG_FFTS[i][c,c,:]       │
       │  c = C_SIZE//2, per-channel RGB             │
       └────────────────────┬────────────────────────┘
                              │
   ══════════════════════════╪═══════════════════════════════════════
        PER-GROUP RECONSTRUCTION
                              │
                              ▼
       ┌─────────────────────────────────────────────┐
       │ build_weight_maps → GROUP_ID_ARR, GROUP_MAP │
       │  pixel group = set of apertures covering it │
       └────────────────────┬────────────────────────┘
                              │
                              ▼  for each group g (angles A_g)
       ┌─────────────────────────────────────────────┐
       │ masks = bow-tie(A_g), Σ masks ≡ 1 (verified)│
       │ Ĝ    = Σ_i Mᵢ · Gᵢ                          │
       │ Ĝ[c,c] ← DC_MEAN               (calibration) │
       │ Σ_m   = Σ_i Mᵢ · MTFᵢ                       │
       │ k_r   = k·(1 + β·(r/r_max)²),  β = 10       │
       │ H     = MTF_ideal / (Σ_m + k_r)             │
       │ H[c,c] ← 1                     (calibration) │
       │ rec_no_h = |IFFT(ifftshift(Ĝ))|             │
       │ rec      = |IFFT(ifftshift(Ĝ·H))|           │
       │ rec      ← rec · (target_mean / cur_mean)   │
       └────────────────────┬────────────────────────┘
                              │
                              ▼
       ┌─────────────────────────────────────────────┐
       │ SOFT-WEIGHT ACCUMULATION  (v3)              │
       │  w_g = G_σ=15( 1[GROUP_ID == g] )           │
       │  acc_num += w_g · rec_g                     │
       │  acc_den += w_g                             │
       └────────────────────┬────────────────────────┘
                              │
                              ▼
              ┌──────────────────────────────────┐
              │ combined = acc_num / (acc_den+ε) │
              │ ε = 1e-6                         │
              └──────────────┬───────────────────┘
                              ▼
                        SAVE_PATH .jpg
                   (+ `_no_h.jpg` counterpart)
```

## Components

### Global configuration

| symbol | value | meaning |
|---|---|---|
| `AP_L`, `AP_W` | 120, 12 | rectangular aperture long / short axis (10:1) |
| `N_APERTURES` | 16 | number of rotation angles |
| `ANGLES` | `linspace(0,180,16,endpoint=False)` | aperture orientations, degrees |
| `LENGTH`, `WIDTH` | 1024, 512 | ROI crop dimensions |
| `C_SIZE` | 1200 | square canvas / FFT grid size |
| `BIN_FACTOR` | 1 | row/column binning along the blur direction |
| `NOISE` | `True` | enable Poisson noise |
| `REG_K` | 0.05 | base Wiener regularization constant |

`DEVICE` is CUDA when available; polygon rasterization stays on CPU (`cv2`),
everything downstream is `torch` on GPU.

### Section 1 — Spatial-domain helpers (CPU)

- **`crop_and_rotate(image, φ)`** — extracts an `L×W` window rotated by `φ−90°`.
  The *window* rotates; the *content* keeps its original orientation. Uses
  `cv2.warpAffine` with Lanczos4 and a final `rot90`.
- **`crop_and_stitch_const_bg`** — alternative path: rasterizes a rotated
  rectangle onto a constant-color canvas with a Gaussian-feathered alpha mask.
  Optionally bins along the ROI's longer axis before pasting.
- **`pad_to_size`** — center-pad to `(LENGTH+200, WIDTH+200)` with `BG_COLOR`,
  so the FFT convolution does not wrap dark pixels into the signal.
- **`bin_blurring_direction`** — reshape-and-mean along the longer axis by
  `bin_factor`, then `cv2.resize` back to the original size. Simulates detector
  binning: noise ↓ by `√bin_factor`, resolution ↓ along the blurred direction.
- **`rotate_and_blend`** — rotates the processed ROI back to angle `φ` and alpha-
  blends it onto a `C_SIZE²` canvas pre-filled with `BG_COLOR`. Pads the image
  edge-wise before rotating so `scipy.ndimage.rotate` never introduces black
  borders; the alpha mask is Gaussian-blurred by `sigma` (called with `sigma=0`).
- **`get_single_weight_map_np(φ)`** — binary rasterized footprint of the ROI at
  angle `φ`, on the canvas.
- **`get_contributor_map(weights)`** — object array; `CONTR_MAP[y,x]` is the list
  of aperture IDs whose footprint contains pixel `(y,x)`.
- **`add_poisson_noise(img, peak=1000)`** — `Poisson(img·peak)/peak`. Higher
  `peak` = higher SNR.

### Section 2 — Optics on GPU

```
aperture  →  PSF = fftshift( |FFT( ifftshift(A) )|² ),  ΣPSF = 1
PSF       →  MTF = |fftshift( FFT(PSF) )| / max(·)
```

- **`get_physical_aperture_gpu(φ)`** — binary `L×W` rectangle rotated by `φ`,
  built from a rotated coordinate frame `(xr, yr)`, valid for non-square canvases.
- **`get_round_aperture_gpu(D)`** — circular reference aperture of diameter `D`.
- **`get_psf_gpu`** — `ifftshift` **before** the FFT is required so the aperture
  center sits at the DC corner; the intensity PSF is `fftshift`ed back for use.
  `norm='ortho'` keeps the dynamic range manageable.
- **`get_mtf_gpu`** — modulus of the OTF, peak-normalized to `[0,1]`.
- **`get_circular_mtf_gpu(L)`** — analytic diffraction-limited MTF of a circular
  aperture, `(2/π)(arccos ν − ν√(1−ν²))` with `ν = ρ / (f_c/2)` and
  `f_c = L / (λ·f)` converted to pixels via `pixel_size · size`. Defaults:
  `λ=550 nm`, `f=33.6 mm`, `pixel=1.5 µm`. This is the **target** MTF that the
  `H`-filter tries to restore.
- **`convolve_with_psf_gpu`** — circular FFT convolution, `ifftshift` on the PSF
  before `rfft2`.
- **`conv_pad_2D_gpu`** — same, but both operands are zero/bg-padded to `2H×2W`
  to suppress wrap-around, then center-cropped. (Available; the main pipeline
  currently calls `convolve_with_psf_gpu` on a pre-padded ROI.)

### Bow-tie combination masks (`get_combination_masks_not_equal_gpu`)

Each aperture contributes the frequency wedge closest to its own orientation.
Wedge boundaries are the angular **bisectors** with neighboring angles in the
mod-180 sense, so unequal angular spacing is handled correctly.

```
1. per-angle half-widths hw_left[i], hw_right[i] from sorted neighbors (mod 180)
2. mask_stack[i] = 1[ -hw_left[i] ≤ d_signed ≤ hw_right[i] ]      (N, H, W)
3. coverage = Σ_i mask_stack[i]
   • coverage == 0 → assign pixel to the angularly closest aperture
   • coverage  > 1 → keep only the angularly closest aperture
4. assert min(coverage) == max(coverage) == 1
```

Step 3 is the **coverage-correction pass**: after it, the masks form an exact
partition of the frequency plane (every pixel covered exactly once). The whole
routine is vectorized — angles broadcast to `(N,1,1)`, pixel grids to `(1,H,W)`,
one kernel launch per stage.

`get_combination_mask_gpu` is the simpler equal-spacing variant
(`|θ_local| < π/2N`), kept for reference; it does **not** guarantee coverage.

### Missing-pixel fill (`fill_missing_pixels_gpu`)

For aperture `i`, a pixel is filled iff it is covered by *some* aperture but not
by `i` itself: `need_fill = any_contributor ∧ ¬weights[i]`.

```
contributors_np  (H, W, K)  ← padded contributor IDs, −1 = empty slot
valid_mask       (H, W, K)  ← contributors ≥ 0
candidate_pixels[hw,k,:] = img_flat[ contrib_flat[hw,k], hw, : ]   # advanced index
avg_pixels = Σ_k candidate · valid / Σ_k valid                     # (HW, 3)
filled = where(need_fill, avg_pixels, original)
```

The replacement value depends only on the pixel location, not on which aperture
is being filled, so `avg_pixels` is computed once and broadcast across `N`.
`contributors.clamp(min=0)` keeps the gather index in bounds; the validity mask
zeroes the padded slots before the mean.

### Post-fill edge softening (in `main`)

Filled pixels are sharp where the observed pixels are blurred, which creates a
visible discontinuity at the footprint boundary. Each filled image is convolved
with its own `PSF(φᵢ)` and blended back:

```
soft_w = G_σ=500 ( w_stack[i] )              # separable 1-D Gaussians, F.conv2d
soft_w = soft_w² (3 − 2·soft_w)              # smoothstep S-curve
FINAL_IMG[i] = soft_w · filled + (1 − soft_w) · convolved
```

`σ = 500` px is deliberately enormous — the intent is a canvas-wide ramp, not a
local feather. Inside the footprint `soft_w → 1` (original); far outside
`soft_w → 0` (blurred to match).

### Fusion core (`combine_unequal_images_gpu`)

```
Ĝ        = Σ_i M_i · G_i                          # (H, W, 3) complex
Ĝ[c,c,:] = DC_MEAN                                # calibration
Σ_m      = Σ_i M_i · MTF_i                        # (H, W) effective MTF
k_r      = k · (1 + β·(r/r_max)²),   β = 10
H        = MTF_ideal / (Σ_m + k_r)
H[c,c]   = 1                                      # calibration
rec_no_h = |IFFT2( ifftshift(Ĝ) )|
rec      = |IFFT2( ifftshift(Ĝ · H) )|
rec     ← rec · target_mean / cur_mean            # per-channel rescale
```

- **`k` is not constant across frequency.** `k_radial = k(1 + β(r/r_max)²)`
  applies stronger regularization at high spatial frequency (noise-dominated)
  and weaker near DC (signal-dominated). Larger `k` → smoother; smaller `k` →
  sharper but noisier.
- **DC pinning.** Setting `Ĝ[c,c] = DC_MEAN` forces every group to share the same
  zero-frequency component (i.e. mean brightness). Setting `H[c,c] = 1` prevents
  the filter from immediately undoing that. Without both, each group's `Σ_m`
  differs at DC and groups reconstruct at different brightness levels — the
  classic per-group seam.
- The final multiplicative rescale to `target_mean = Re(DC_MEAN)/C_SIZE²` corrects
  residual mean drift introduced by taking `|·|` of the complex inverse FFT.
- Commented-out alternatives kept in source: `H = MTF_ideal/(Σ_m + k)` (plain
  inverse) and `H = MTF_ideal·Σ_m/(Σ_m² + k)` (textbook Wiener).

### Pixel groups (`build_weight_maps`)

A pixel's **group** is the exact set of apertures whose footprints contain it.

```
weight_stack[i] = footprint(φ_i)                       (N, H, W) bool
key(r,c)        = tuple( i : weight_stack[i,r,c] )
GROUP_ID_ARR[r,c] = group_map_ori[key]                 int32, 0 = no coverage
GROUP_MAP       = {id → tuple_of_angle_indices}, GROUP_MAP[0] = None
```

Every pixel in a group sees the same subset of apertures, so a single fused
reconstruction serves the whole group. Groups are reconstructed independently
and stitched. Note `main` shrinks the footprint (`L−50, W−50`) when building
groups, so the group boundaries sit inside the true observed region.

### Soft-weight stitching (`fill_corr_px_gpu`, v3)

```
w_g   = G_σ=15( 1[GROUP_ID_ARR == g] )          # separable Gaussian, F.conv2d
acc_num      += w_g · rec_g
acc_den      += w_g
acc_num_no_h += w_g · rec_no_h_g
acc_den_no_h += w_g
...
combined = acc_num / (acc_den + 1e-6)           # partition-of-unity average
```

`combined(x) = Σ_g w_g(x)·rec_g(x) / Σ_g w_g(x)`. Because `Σ_g 1[GROUP==g] ≡ 1`
over the covered region and Gaussian blur is linear, `Σ_g w_g ≡ 1` there too —
the division is a no-op in the interior and only matters at the outer border.
Adjacent groups now overlap in a `~3σ` transition band instead of butting
against each other, which is what removes the hard seams of v2.

The function also NaN/Inf-checks `reconstructed_fft` and returns every
intermediate (`masks`, `sum_m_mtf`, `h_filter`, both FFTs) for debugging.

## Outputs

Written under `SAVE_PATH`'s parent, in `single_results/`:

| artifact | content |
|---|---|
| `{SAVE_PATH}` | final combined image, after `H`-filter |
| `{SAVE_PATH}_no_h.jpg` | final combined image, before `H`-filter |
| `ROI/` | raw rotated crops, per angle |
| `IMG_SINGLE/` | blurred + noised + re-stitched observations, per angle |
| `IMG_CROP_FILL/` | after missing-pixel fill, per angle |
| `IMG_FINAL/` | after outside-footprint blur blend, per angle |
| `SINGLE_FFTS/` | `fftshift(FFT2(IMG_CROP[i]))`, `.npy` |
| `G_IMG_FFTS/` | `fftshift(FFT2(FINAL_IMG[i]))`, `.npy` — the fusion inputs |
| `{g}.jpg`, `{g}_no_h.jpg` | per-group reconstruction |
| `{g}_masks.npz` | per-group bow-tie masks |
| `{g}_fft.npy`, `{g}_sum_m_mtf.npy`, `{g}_h_filter.npy` | per-group diagnostics |
| `psf.jpg`, `aperture.jpg` | circular reference PSF (log-scaled) and aperture |
| `max_values.txt` | `GROUP_MAP[g]` ↔ `max(rec_no_h)` per group, brightness audit |

## Failure modes and guards

| failure | symptom | guard |
|---|---|---|
| Per-group brightness mismatch | visible tiling; `max_values.txt` varies widely across groups | DC pinning (`Ĝ[c,c] = DC_MEAN`, `H[c,c] = 1`) + multiplicative mean rescale |
| Hard seams between groups | sharp lines along group boundaries | v3 soft weights: `σ = 15` Gaussian feather + `acc_num/acc_den` |
| Wedge/star artifacts in the spectrum | radial spokes in the reconstruction | verify `assert` in the coverage-correction pass; masks must partition exactly |
| Noise blow-up at high frequency | grain amplified where `Σ_m → 0` | radial `k_radial = k(1+β(r/r_max)²)`; raise `REG_K` or `β` |
| Over-smoothing | result softer than `_no_h` at low frequency | lower `REG_K`; check `MTF_ideal` cutoff matches `AP_L` |
| Wrap-around / dark edges | dark halo at canvas border | `pad_to_size(BG_COLOR)` before convolution; `conv_pad_2D_gpu` if worse |
| Boundary discontinuity from filling | sharp ring at each footprint edge | smoothstep blend with `PSF(φᵢ)`-convolved version |
| `fill_missing_pixels_gpu` slowness | seconds spent in Python loops | the `max_k` scan and `contributors_np` build are `O(H·W)` on CPU; cache per canvas size |
| NaN/Inf in fused spectrum | black or saturated group | printed warning in `fill_corr_px_gpu`; usually `Σ_m + k_r → 0`, raise `k` |

## Tunable knobs

| knob | location | effect |
|---|---|---|
| `REG_K` | global | base Wiener regularization; sharpness ↔ noise tradeoff |
| `beta = 10` | `combine_unequal_images_gpu` | how much harder to regularize at high frequency |
| `NOISE` | global | toggle Poisson noise in the forward model |
| `peak = 1000` | `add_poisson_noise` | photon count at intensity 1.0; SNR |
| `BIN_FACTOR` | global | detector binning along the blur direction |
| `sigma = 15` | `fill_corr_px_gpu` | group-seam feather width; 10–30 reasonable |
| `sigma = 500` | `main`, edge softening | ramp width for the filled-vs-blurred blend |
| `L−30 / L−50` | `main`, weight maps | how far group boundaries are inset from the true footprint |
| `AP_L : AP_W` | global | aperture anisotropy; drives how narrow each frequency wedge is |
| `N_APERTURES` | global | angular sampling density of the frequency plane |
