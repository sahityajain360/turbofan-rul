#!/bin/bash
# Base WSL env for the Mamba / SSL work (M7.6). Run as root in the Ubuntu WSL distro.
# Creates mambaenv on D drive with CUDA PyTorch + the data stack.
set -e
export DEBIAN_FRONTEND=noninteractive

echo "=== apt: python tooling ==="
apt-get update -y
apt-get install -y python3-pip python3-venv

echo "=== venv: /mnt/d/INTERNSHIP/PREDICTIVE_MAINTENANCE/mambaenv ==="
python3 -m venv /mnt/d/INTERNSHIP/PREDICTIVE_MAINTENANCE/mambaenv
. /mnt/d/INTERNSHIP/PREDICTIVE_MAINTENANCE/mambaenv/bin/activate
python -m pip install --no-cache-dir --upgrade pip wheel setuptools

echo "=== torch (cu121) ==="
pip install --no-cache-dir torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

echo "=== data + ml stack ==="
pip install --no-cache-dir "numpy<2" pandas h5py scikit-learn scipy pyarrow matplotlib seaborn psutil lightgbm shap

echo "=== verify torch + cuda ==="
python - <<'PY'
import torch
print("TORCH", torch.__version__, "| CUDA build", torch.version.cuda, "| available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("DEVICE", torch.cuda.get_device_name(0))
PY
echo "BASE_SETUP_DONE"

