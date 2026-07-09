# Asymmetric_ap_recon
​

## Setup
Follow these steps to set up the environment:
``` 
conda create -n neuws python=3.9
conda activate neuws
pip install -r requirements.txt
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```
We assume access to a GPU with CUDA 11.3.1 installed/supported.

## Dataset
The image for illustration is included in this repository as test3_1600_1200.jpg

## Reconstruct Experimental Data

The result is saved to SAVE_PATH as defined in each py file. Please refer to the latest version for reconstruction algorithims. 

## Per-angle intermediate outputs (single_results/)
All per-angle artifacts are written under <SAVE_PATH parent>/single_results/, inside a run directory named {REG_K}_{NOISE}_{BIN_FACTOR} (e.g. 0.05_True_10). The six subfolders trace the forward model and pre-fusion processing chain in order.
ROI/ holds the raw rotated crops, one per angle. The L×W window rotates with φ, but the image content keeps its original orientation. No blur, no noise. Written as image_{i}_phi{φ}.jpg by crop_and_rotate.
IMG_SINGLE/ holds the simulated single-aperture observation: the ROI convolved with PSF(φ=0), corrupted with Poisson noise, binned along the blur direction, then rotated back to φ and blended onto the C_SIZE² canvas. This is what a real detector behind aperture i would record. Written as image_{i}_phi{φ}.jpg after rotate_and_blend.
SINGLE_FFTS/ stores fftshift(FFT2(IMG_SINGLE[i])) as complex64 arrays of shape (C_SIZE, C_SIZE, 3), named {i}_phi{φ}.npy. These are used only to derive the shared DC_MEAN calibration constant — they are not the fusion inputs.
IMG_CROP_FILL/ holds the result of missing-pixel fill: pixels lying inside some aperture's footprint but outside this one's are replaced by the mean over all contributing apertures, as recorded in CONTR_MAP. Written as image_{i}_phi{φ}.jpg by fill_missing_pixels_gpu.
IMG_FINAL/ holds the edge-softened images. Each filled image is convolved with its own PSF(φᵢ) and smoothstep-blended back against the unconvolved version, so that filled (sharp) and observed (blurred) regions transition smoothly across the footprint boundary instead of meeting at a visible ring. Written as image_{i}_phi{φ}.jpg.
G_IMG_FFTS/ stores fftshift(FFT2(IMG_FINAL[i])), again complex64 (C_SIZE, C_SIZE, 3) named {i}_phi{φ}.npy. These are the actual fusion inputs Gᵢ consumed by combine_unequal_images_gpu.
The chain runs ROI → IMG_SINGLE → (SINGLE_FFTS) → IMG_CROP_FILL → IMG_FINAL → G_IMG_FFTS → per-group reconstruction. Comparing IMG_SINGLE against IMG_CROP_FILL isolates the fill step; comparing IMG_CROP_FILL against IMG_FINAL isolates the boundary-softening step. The two FFT folders exist separately because SINGLE_FFTS reflects the uncorrected observation, whose DC values are averaged to form the calibration target, while G_IMG_FFTS reflects the fully pre-processed image that enters the bow-tie fusion.