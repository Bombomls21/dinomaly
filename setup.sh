#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Setting up Dinomaly Environment ===${NC}"

# Create conda environment
echo -e "${YELLOW}Creating conda environment...${NC}"
conda create -n dinomaly2 python=3.8.12 -y

# Activate environment
echo -e "${YELLOW}Activating environment...${NC}"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate dinomaly2

# Install PyTorch with CUDA
echo -e "${YELLOW}Installing PyTorch with CUDA 11.3...${NC}"
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# Install conda packages
echo -e "${YELLOW}Installing conda packages...${NC}"
conda install numpy=1.18 scipy=1.4 -c conda-forge -y
conda install matplotlib=3.2 pandas=1.3 pillow=9.0 -c conda-forge -y
conda install scikit-learn=0.22 scikit-image=0.19 -c conda-forge -y

# Install pip packages
echo -e "${YELLOW}Installing additional packages...${NC}"
pip install opencv-python-headless==4.6.0.66 \
            tqdm==4.64.1 \
            tabulate==0.9.0 \
            ptflops==0.7 \
            timm==0.9.12

# Install additional required packages
echo -e "${YELLOW}Installing colorama, einops, fvcore, iopath, omegaconf...${NC}"
pip install colorama \
            einops \
            fvcore \
            iopath \
            omegaconf

# Install xformers for better performance
echo -e "${YELLOW}Installing xformers...${NC}"
pip install xformers==0.0.13

# Verify installation
echo -e "${GREEN}=== Verifying Installation ===${NC}"
python -c "
import torch
import torchvision
import numpy as np
import cv2
import timm
from sklearn import metrics
import colorama
import einops
import fvcore
import iopath
import omegaconf

print('✅ PyTorch version:', torch.__version__)
print('✅ CUDA available:', torch.cuda.is_available())
print('✅ NumPy version:', np.__version__)
print('✅ OpenCV version:', cv2.__version__)
print('✅ Timm version:', timm.__version__)
print('✅ All packages imported successfully!')
"

echo -e "${GREEN}=== Setup Complete! ===${NC}"
echo -e "${YELLOW}To activate the environment, run: conda activate dinomaly${NC}"
