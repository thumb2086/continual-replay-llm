#!/bin/sh
# WSL compress env setup (CPU-only: venv + pip, no GPU touched)
set -x
/home/thumb/vllm-env/bin/python -m venv --system-site-packages /home/thumb/wsl-compress
/home/thumb/wsl-compress/bin/pip install -q transformers numba
/home/thumb/wsl-compress/bin/python -c "import torch, transformers, numba; print(torch.__version__, torch.cuda.is_available())"
echo SETUP_DONE
