#!/usr/bin/env bash
set -euo pipefail
job=/home/zpxiang/ghap_pristine_compare_20260921_01
out=/data/zpxiang/ghap_pristine_finetune_20260921_01
py=/home/zpxiang/hot_bw_applications_20260904_01/envs/reggs/bin/python
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1 LC_ALL=C LANG=C PYTHONUTF8=1
export PYTHONPATH="$job/code:$job/pytorch3d_official:/data/zpxiang/ghap_sanc_hot_20260904_01/python_packages:/data/zpxiang/ghap_blockmw_20260906_01/python_packages"
mkdir -p "$out"
finish() { s=$?; echo "$s" > "$out/job.exit"; if (( s != 0 )); then echo FAILED > "$out/status.txt"; fi; }
trap finish EXIT
echo WAITING_FOR_IDLE_GPU1 > "$out/status.txt"
idle=0
while (( idle < 6 )); do
 read -r used util < <(nvidia-smi -i 1 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr ',' ' ')
 if (( used < 1000 && util < 5 )); then idle=$((idle+1)); else idle=0; fi
 sleep 5
done
cd "$job/code"
sha256sum -c /data/zpxiang/ghap_pristine_compare_20260921_01/code_before.sha256 > "$out/code_check_before.log"
echo PREPARING > "$out/status.txt"
"$py" "$job/prepare_finetune.py"
for scene in train truck; do
 if [[ "$scene" = train ]]; then
  dataset=/data/zpxiang/nanogs_hcp_assets_20260919/datasets/tandt/train
  resolution=1
 else
  dataset=/data/zpxiang/ghap_sanc_hot_20260904_01/datasets/truck
  resolution=-1
 fi
 for method in official_ghap hilbert_optimized; do
  arm="$out/$scene/$method"
  echo "TRAINING $scene $method" > "$out/status.txt"
  /usr/bin/time -p -o "$arm/train_wall.txt" "$py" train_and_prune.py \
   -s "$dataset" -m "$arm" -r "$resolution" --eval --data_device cpu \
   --start_checkpoint "$arm/start.pth" --iterations 35000 --disable_viewer \
   --test_iterations 35000 --save_iterations 35000 --checkpoint_iterations 35000 \
   > "$arm/train.log" 2>&1
  echo "EVALUATING $scene $method" > "$out/status.txt"
  "$py" render.py -m "$arm" --iteration 35000 --skip_train > "$arm/render.log" 2>&1
  "$py" metrics.py -m "$arm" > "$arm/metrics.log" 2>&1
 done
done
sha256sum -c /data/zpxiang/ghap_pristine_compare_20260921_01/code_before.sha256 > "$out/code_check_after.log"
echo COMPLETE > "$out/status.txt"
