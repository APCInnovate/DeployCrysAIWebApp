#!/bin/bash

# Ensure Python3 and pip are available
export PATH=$HOME/.local/bin:$PATH

# Upgrade pip
python3 -m ensurepip --default-pip
python3 -m pip install --upgrade pip

# Install Detectron2
python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

