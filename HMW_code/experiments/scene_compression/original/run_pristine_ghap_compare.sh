#!/usr/bin/env bash
set -euo pipefail
job=/home/zpxiang/ghap_pristine_compare_20260921_01
out=/data/zpxiang/ghap_pristine_compare_20260921_01
optimized=/home/zpxiang/ghap_hcp_batched_20260921_01
oldjob=/home/zpxiang/ghap_gmm_fixed_20260920_01
py=/home/zpxiang/hot_bw_applications_20260904_01/envs/reggs/bin/python
gpu=${GHAP_COMPARE_GPU:-1}
export CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1 LC_ALL=C LANG=C PYTHONUTF8=1
export HCP_EXTENSION="$oldjob/base.so"
export PYTHONPATH="$job/code:$job/pytorch3d_official:$optimized:$oldjob/source:$oldjob/source/experiments:/home/zpxiang/ghap_fair_block_20260907_01/code:/data/zpxiang/ghap_sanc_hot_20260904_01/python_packages:/data/zpxiang/ghap_blockmw_20260906_01/python_packages"
monitor_pid=''
finish() {
 s=$?
 if [[ -n "$monitor_pid" ]]; then kill "$monitor_pid" 2>/dev/null || true; fi
 echo "$s" > "$out/job.exit"
 if (( s != 0 )); then echo FAILED > "$out/status.txt"; fi
}
trap finish EXIT
cd "$job/code"
"$py" -c 'import gmm_sampler, scene, pytorch3d; from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix; print(gmm_sampler.__file__, scene.__file__, pytorch3d.__file__)' > "$out/import_preflight.txt"
echo "WAITING_FOR_IDLE_GPU${gpu}" > "$out/status.txt"
idle=0
while (( idle < 6 )); do
 read -r used util < <(nvidia-smi -i "$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr ',' ' ')
 if (( used < 1000 && util < 5 )); then idle=$((idle+1)); else idle=0; fi
 sleep 5
done
echo RUNNING > "$out/status.txt"
nvidia-smi > "$out/gpu_before.txt"
nvidia-smi -i "$gpu" --query-gpu=timestamp,index,utilization.gpu,memory.used,temperature.gpu,clocks.sm,power.draw --format=csv -l 1 > "$out/gpu_trace.csv" &
monitor_pid=$!
find "$job/code" -type f -name '*.py' -print0 | sort -z | xargs -0 sha256sum > "$out/code_before.sha256"
sha256sum "$optimized/hcp_block_lloyd_batched.py" "$optimized/hcp_batched_solver.py" >> "$out/code_before.sha256"
"$py" "$job/benchmark.py"
sha256sum -c "$out/code_before.sha256" > "$out/code_check.log"
echo EVALUATING > "$out/status.txt"
for scene in train truck; do
 for method in official_ghap hilbert_optimized; do
  arm="$out/$scene/$method"
  "$py" render.py -m "$arm" --iteration 1 --skip_train > "$out/${scene}_${method}_render.log" 2>&1
  "$py" metrics.py -m "$arm" > "$out/${scene}_${method}_metrics.log" 2>&1
 done
done
echo COMPLETE > "$out/status.txt"
