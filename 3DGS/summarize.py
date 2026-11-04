import _bootstrap
import argparse
from pathlib import Path
from lsot3dgs.reporting import summarize,summarize_scaling

if __name__=="__main__":
    p=argparse.ArgumentParser(description="Collect successful, incomplete and failed configurations")
    p.add_argument("--results-dir",required=True,type=Path)
    args=p.parse_args()
    function=summarize_scaling if (args.results_dir/"scaling_config.json").exists() else summarize
    rows=function(args.results_dir)
    print(f"Wrote {len(rows)} rows to {args.results_dir}")
