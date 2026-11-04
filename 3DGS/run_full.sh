#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config="${1:-$task_root/3DGS/configs/hmw_protocol.json}"
if (( $# )); then shift; fi
python "$task_root/3DGS/prepare_data.py" --config "$config"
python "$task_root/3DGS/preflight.py" --config "$config" --renderer --scene train
python "$task_root/3DGS/run.py" --config "$config" "$@"
