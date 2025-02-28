#!/bin/bash
export PATH=$HOME/.local/bin:$PATH

# Install required system packages
sudo apt-get update && sudo apt-get install -y \
    gcc \
    make \
    python3-dev \
    libpython3-dev \
    libglib2.0-dev \
    libgl1-mesa-glx \
    libsm6 \
    libxrender1 \
    libxext6

# Upgrade pip
python3 -m ensurepip --default-pip
python3 -m pip install --upgrade pip setuptools wheel

# Install pycocotools separately
pip install numpy cython
pip install pycocotools
