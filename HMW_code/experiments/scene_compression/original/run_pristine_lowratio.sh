#!/usr/bin/env bash
set -euo pipefail
job=/home/zpxiang/ghap_pristine_lowratio_20260923_01
out=/data/zpxiang/ghap_pristine_lowratio_20260923_01
official=/home/zpxiang/ghap_pristine_compare_20260921_01
optimized=/home/zpxiang/ghap_hcp_batched_20260921_01
oldjob=/home/zpxiang/ghap_gmm_fixed_20260920_01
py=/home/zpxiang/hot_bw_applications_20260904_01/envs/reggs/bin/python
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1 LC_ALL=C LANG=C PYTHONUTF8=1
export HCP_EXTENSION="$oldjob/base.so"
export PYTHONPATH="$official/code:$official/pytorch3d_official:$optimized:$oldjob/source:$oldjob/source/experiments:/home/zpxiang/ghap_fair_block_20260907_01/code:/data/zpxiang/ghap_sanc_hot_20260904_01/python_packages:/data/zpxiang/ghap_blockmw_20260906_01/python_packages"
monitor_pid=''
finish() {
 s=$?
 if [[ -n "$monitor_pid" ]]; then kill "$monitor_pid" 2>/dev/null || true; fi
 echo "$s" > "$out/job.exit"
 if (( s != 0 )); then echo FAILED > "$out/status.txt"; fi
}
trap finish EXIT
cd "$official/code"
sha256sum -c /data/zpxiang/ghap_pristine_compare_20260921_01/code_before.sha256 > "$out/code_check_before.log"
"$py" -c 'import gmm_sampler,scene,pytorch3d,hcp_block_lloyd_batched; print(gmm_sampler.__file__,scene.__file__,pytorch3d.__file__,hcp_block_lloyd_batched.__file__)' > "$out/import_preflight.txt"
echo WAITING_FOR_ANY_IDLE_GPU > "$out/status.txt"
candidate=''; idle=0
while (( idle < 6 )); do
 selected=''
 while read -r idx used util; do
  if (( used < 1000 && util < 5 )); then selected="$idx"; break; fi
 done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | tr ',' ' ')
 if [[ -n "$selected" && "$selected" = "$candidate" ]]; then idle=$((idle+1)); else idle=0; candidate="$selected"; fi
 sleep 5
done
export CUDA_VISIBLE_DEVICES="$candidate"
echo "$candidate" > "$out/gpu_index.txt"
nvidia-smi > "$out/gpu_before.txt"
nvidia-smi -i "$candidate" --query-gpu=timestamp,index,utilization.gpu,memory.used,temperature.gpu,clocks.sm,power.draw --format=csv -l 2 > "$out/gpu_trace.csv" &
monitor_pid=$!
for ratio in 0.01 0.001; do
 export KEEP_RATIO="$ratio" LOWRATIO_ROOT="$out/rho_$ratio"
 if [[ "$ratio" = 0.001 ]]; then export HCP_BATCH_SIZE=2; else export HCP_BATCH_SIZE=16; fi
 mkdir -p "$LOWRATIO_ROOT"
 echo "COMPRESSING $ratio GPU $candidate" > "$out/status.txt"
 "$py" "$job/benchmark.py" > "$LOWRATIO_ROOT/compression.log" 2>&1
 for scene in train truck; do
  for method in official_ghap hilbert_optimized; do
   arm="$LOWRATIO_ROOT/pre/$scene/$method"
   echo "EVALUATING_PRE $ratio $scene $method" > "$out/status.txt"
   "$py" render.py -m "$arm" --iteration 1 --skip_train > "$arm/render.log" 2>&1
   "$py" metrics.py -m "$arm" > "$arm/metrics.log" 2>&1
  done
 done
 "$py" "$job/prepare_finetune.py" > "$LOWRATIO_ROOT/prepare.log" 2>&1
 for scene in train truck; do
  if [[ "$scene" = train ]]; then
   dataset=/data/zpxiang/nanogs_hcp_assets_20260919/datasets/tandt/train; resolution=1
  else
   dataset=/data/zpxiang/ghap_sanc_hot_20260904_01/datasets/truck; resolution=-1
  fi
  for method in official_ghap hilbert_optimized; do
   arm="$LOWRATIO_ROOT/post/$scene/$method"
   echo "TRAINING $ratio $scene $method" > "$out/status.txt"
   /usr/bin/time -p -o "$arm/train_wall.txt" "$py" train_and_prune.py \
    -s "$dataset" -m "$arm" -r "$resolution" --eval --data_device cpu \
    --start_checkpoint "$arm/start.pth" --iterations 35000 --disable_viewer \
    --test_iterations 35000 --save_iterations 35000 --checkpoint_iterations 35000 \
    > "$arm/train.log" 2>&1
   echo "EVALUATING_POST $ratio $scene $method" > "$out/status.txt"
   "$py" render.py -m "$arm" --iteration 35000 --skip_train > "$arm/render.log" 2>&1
   "$py" metrics.py -m "$arm" > "$arm/metrics.log" 2>&1
  done
 done
done
sha256sum -c /data/zpxiang/ghap_pristine_compare_20260921_01/code_before.sha256 > "$out/code_check_after.log"
"$py" "$job/summarize.py" > "$out/summary.log" 2>&1
echo COMPLETE > "$out/status.txt"
