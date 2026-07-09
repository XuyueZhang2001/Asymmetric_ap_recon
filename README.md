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
