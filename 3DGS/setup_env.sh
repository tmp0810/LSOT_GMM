#!/usr/bin/env bash
# Run inside a dedicated Python 3.10--3.12 environment with a CUDA toolkit.
set -euo pipefail
task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ghap_path="$task_root/3DGS/third_party/GHAP"
ghap_commit=27b3a581c861069cc97e5629e216dc3d1f42a98a
pytorch3d_commit=33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba
install_torch=0
if [[ "${1:-}" == "--install-torch" ]]; then install_torch=1; shift; fi
if (( $# )); then echo "Usage: bash 3DGS/setup_env.sh [--install-torch]" >&2; exit 2; fi
python - <<'PY'
import sys
if not (3,10) <= sys.version_info[:2] <= (3,12):
    raise SystemExit('Use a dedicated Python 3.10, 3.11 or 3.12 environment for these CUDA extensions.')
PY
if (( install_torch )); then
  python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
fi
command -v nvcc >/dev/null || { echo "CUDA toolkit/nvcc is required to build the renderer." >&2; exit 1; }
python - <<'PY'
import torch, torchvision
if not torch.cuda.is_available():
    raise SystemExit('PyTorch cannot use CUDA. Check GPU allocation, driver and PyTorch build.')
print('PyTorch:',torch.__version__,'CUDA:',torch.version.cuda,'GPU:',torch.cuda.get_device_name())
PY
export MAX_JOBS="${MAX_JOBS:-8}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;9.0}"
python -m pip install -r "$task_root/3DGS/requirements.txt"
python -m pip install -e "$task_root"
if [[ ! -d "$ghap_path/.git" ]]; then
  git clone https://github.com/taowang0105/GHAP.git "$ghap_path"
fi
if [[ -n "$(git -C "$ghap_path" status --porcelain --untracked-files=no)" ]]; then
  echo "GHAP checkout has local changes; use a clean checkout rather than overwriting them." >&2
  exit 1
fi
git -C "$ghap_path" checkout --detach "$ghap_commit"
# The OpenGL viewer is unnecessary. Initialize only the CUDA extension submodules.
git -C "$ghap_path" submodule update --init --recursive -- \
  submodules/diff-gaussian-rasterization submodules/simple-knn submodules/fused-ssim
python -m pip install --no-build-isolation "$ghap_path/submodules/diff-gaussian-rasterization"
python -m pip install --no-build-isolation "$ghap_path/submodules/simple-knn"
python -m pip install --no-build-isolation "$ghap_path/submodules/fused-ssim"
python -m pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@$pytorch3d_commit"
python "$task_root/HMW_code/tools/build_hcp.py"
python "$task_root/3DGS/preflight.py" --config "$task_root/3DGS/configs/hmw_protocol.json"
python -m pytest "$task_root/3DGS/tests" -q
