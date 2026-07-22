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
All per-angle artifacts are written under <SAVE_PATH parent>/single_results/, inside a run directory named {REG_K}_{NOISE}_{BIN_FACTOR}_{SIGMA} (e.g. 0.05_True_10_25). 

The reconstructed image is saved in this directory, named as {N_APERTURE}_combined_result.jpg, or {N_APERTURE}_combined_result_no_h.jpg (before h filter).

```{SAVE_PATH}
│
├── {N_APERTURE}_combined_result.jpg: Final reconstruction result
├── {N_APERTURE}_combined_result_no_h.jpg: Reconstruction result before deconvolution filter
└── single_results: results for each single aperture combination results (for check and debug)
    │
    ├── results for a single combination (named as group number): xx_fft.npy, xx_h_filter.npy, xx_masks.npz, xx_no_h.jpg, xx_sum_m_mtf.jpg, xx.jpg
    │   
    ├── G_IMG_FFTS: ffts for each single aperture images (after filling)
    ├── IMG_CROP_FILL: filled images for each single aperture
    ├── IMG_FINAL: filled image after soft weighting and edge blurring
    ├── IMG_SINGLE: each single aperture images
    ├── SINGLE_FFTS: ffts for each raw single apertures (before filling)
    ├── SOFT_WT: soft weight map for each filling missing pixels

