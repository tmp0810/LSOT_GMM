"""CSV summaries, scaling plots and matched-view image panels."""
import csv
import json
from pathlib import Path

import numpy as np

from .common import write_json


LABELS={"mw2":"MW2","ghap":"GHAP","hmw-local":"HMW-local","identity":"Original"}


def label(method):
    if method in LABELS:
        return LABELS[method]
    mode,kind=method.split("-",1)
    return f"{'min-' if mode=='min' else ''}LSOT-{kind}"


def write_csv(path,rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=keys); writer.writeheader(); writer.writerows(rows)


def summarize(root):
    root=Path(root); rows=[]; timing=[]
    for jobfile in sorted(root.glob("*/*/*/job.json")):
        folder=jobfile.parent; job=json.loads(jobfile.read_text())
        base={"scene":job["scene"]["name"],"retention":job["retention"],"method":job["method"],
              "status":"complete" if (folder/"complete.json").exists() else "incomplete",
              "folder":str(folder)}
        if (folder/"failure.json").exists() and base["status"]!="complete":
            failure=json.loads((folder/"failure.json").read_text())
            base.update(status=failure.get("status","failed"),error=failure["message"])
        if (folder/"compression.json").exists():
            comp=json.loads((folder/"compression.json").read_text())
            base.update(source_count=comp["source_count"],requested_count=comp["requested_count"],
                        actual_count=comp["actual_count"],actual_retention=comp["actual_retention"],
                        compression_seconds=comp.get("compression_seconds_median",comp["compression_seconds"]),
                        gpu_peak_allocated_bytes=comp.get("gpu_peak_allocated_bytes",0),
                        cpu_peak_rss_bytes=comp.get("cpu_peak_rss_bytes"),
                        all_blocks_converged=comp.get("all_blocks_converged"),objective=comp.get("objective"))
            for stage,seconds in comp["timings"].items():
                timing.append({**{k:base[k] for k in ("scene","retention","method")},
                               "stage":stage,"seconds":seconds,"timing_repeat":0})
        found=False
        for phase in ("pre","post"):
            path=folder/phase/"evaluation.json"
            if path.exists():
                metrics=json.loads(path.read_text())
                row={**base,"phase":phase,**metrics["metrics"],"test_count":metrics["test_count"],
                     "split_sha256":metrics["split_sha256"],"render_fps":metrics.get("render_fps")}
                if phase=="post" and (folder/"post/finetune.json").exists():
                    row["finetune_seconds"]=json.loads((folder/"post/finetune.json").read_text())["seconds"]
                rows.append(row); found=True
        if not found:
            rows.append({**base,"phase":"compression_only"})
    write_csv(root/"summary.csv",rows); write_csv(root/"timing_breakdown.csv",timing)
    write_csv(root/"quality.csv",[r for r in rows if r["phase"] in ("pre","post")])
    runtime=[]; seen=set()
    for row in rows:
        key=(row["scene"],row["retention"],row["method"])
        if key not in seen:
            seen.add(key)
            runtime.append({k:v for k,v in row.items() if k not in
                            ("phase","PSNR","SSIM","LPIPS","test_count","split_sha256","render_fps")})
    write_csv(root/"runtime.csv",runtime)
    write_json(root/"summary.json",rows)
    return rows


def summarize_scaling(root):
    root=Path(root); rows=[]
    for path in sorted(root.glob("*/*/N_*/*/result.json")):
        data=json.loads(path.read_text()); c=data["case"]
        rows.append({"scene":c["scene"],"n":c["n"],"m":c["m"],"method":c["method"],
                     "projections":c["solver"]["projections"],"status":data["status"],
                     **{key:data.get(key) for key in ("cold_solver_seconds","warm_solver_seconds","objective",
                         "cpu_peak_rss_bytes","gpu_peak_allocated_bytes","message")}})
    exact={(r["scene"],r["n"],r["m"]):r for r in rows if r["method"]=="mw2" and r["status"]=="complete"}
    for row in rows:
        baseline=exact.get((row["scene"],row["n"],row["m"]))
        if baseline and row["status"]=="complete":
            row["speedup_vs_mw2"]=baseline["cold_solver_seconds"]/row["cold_solver_seconds"]
            row["relative_cost_gap"]=(row["objective"]-baseline["objective"])/max(abs(baseline["objective"]),1e-15)
        else:
            row["speedup_vs_mw2"]=None; row["relative_cost_gap"]=None
    write_csv(root/"scaling.csv",rows); write_json(root/"scaling_summary.json",rows)
    return rows


def plots(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root=Path(root); dest=root/"figures"; dest.mkdir(exist_ok=True)
    if (root/"scaling_config.json").exists():
        rows=summarize_scaling(root)
        for scene in sorted({r["scene"] for r in rows}):
            ratios=sorted({round(r["m"]/r["n"],10) for r in rows if r["scene"]==scene})
            for rho in ratios:
                subset=[r for r in rows if r["scene"]==scene and abs(r["m"]/r["n"]-rho)<1e-9]
                for field,title,ylabel in (("cold_solver_seconds","runtime","Solver time (s)"),
                    ("speedup_vs_mw2","speedup","MW2 time / method time"),
                    ("gpu_peak_allocated_bytes","gpu_memory","Peak allocated GPU memory (GiB)"),
                    ("cpu_peak_rss_bytes","cpu_memory","Worker peak RSS (GiB)"),
                    ("relative_cost_gap","cost_gap","Relative lifted cost gap to MW2")):
                    fig,ax=plt.subplots(figsize=(6.5,4))
                    for method in dict.fromkeys(r["method"] for r in subset):
                        curve=sorted((r for r in subset if r["method"]==method and r["status"]=="complete"
                                      and r.get(field) is not None),key=lambda r:r["n"])
                        if curve:
                            values=[r[field]/1024**3 if "bytes" in field else r[field] for r in curve]
                            ax.plot([r["n"] for r in curve],values,"o-",label=label(method))
                    ax.set_xscale("log"); ax.set_xlabel("Source components N"); ax.set_ylabel(ylabel)
                    if field=="cold_solver_seconds": ax.set_yscale("log")
                    ax.grid(alpha=.25); ax.legend(fontsize=8); fig.tight_layout()
                    fig.savefig(dest/f"{scene}_rho_{rho:g}_{title}.png",dpi=200); plt.close(fig)
    else:
        rows=summarize(root)
        # Stage breakdown is repeat 0; total timing in summary is the repeat median.
        timing=list(csv.DictReader((root/"timing_breakdown.csv").open()))
        for scene,rho in sorted({(r["scene"],r["retention"]) for r in timing}):
            selected=[r for r in timing if r["scene"]==scene and r["retention"]==rho]
            methods=list(dict.fromkeys(r["method"] for r in selected))
            stages=list(dict.fromkeys(r["stage"] for r in selected))
            fig,ax=plt.subplots(figsize=(9,5)); bottom=np.zeros(len(methods))
            for stage in stages:
                values=np.array([sum(float(r["seconds"]) for r in selected
                                     if r["method"]==method and r["stage"]==stage) for method in methods])
                ax.bar(range(len(methods)),values,bottom=bottom,label=stage.removesuffix("_seconds"))
                bottom+=values
            ax.set_xticks(range(len(methods)),[label(m) for m in methods],rotation=35,ha="right")
            ax.set_ylabel("Stage time (s), measured repeat 0")
            ax.legend(fontsize=6,ncol=2); fig.tight_layout()
            fig.savefig(dest/f"{scene}_rho_{rho}_stages.png",dpi=180); plt.close(fig)
        for scene in sorted({r["scene"] for r in rows}):
            for phase in ("pre","post"):
                for metric in ("PSNR","SSIM","LPIPS"):
                    fig,ax=plt.subplots(figsize=(6.5,4)); plotted=False
                    for method in dict.fromkeys(r["method"] for r in rows):
                        curve=sorted((r for r in rows if r["scene"]==scene and r["phase"]==phase
                            and r["method"]==method and r["status"]=="complete" and metric in r),
                            key=lambda r:r["retention"])
                        if curve:
                            if method=="identity":
                                ax.axhline(curve[0][metric],ls="--",c="gray",label="Original")
                            else:
                                ax.plot([r["retention"] for r in curve],[r[metric] for r in curve],"o-",label=label(method))
                            plotted=True
                    if plotted:
                        ax.set_xscale("log"); ax.set_xlabel("Retention"); ax.set_ylabel(metric)
                        ax.grid(alpha=.25); ax.legend(fontsize=8); fig.tight_layout()
                        fig.savefig(dest/f"{scene}_{phase}_{metric}.png",dpi=200)
                    plt.close(fig)
        # Objective histories are diagnostics, including increases and capped blocks.
        for path in root.glob("*/*/*/history.jsonl"):
            data=[json.loads(line) for line in path.read_text().splitlines() if line]
            if not data: continue
            fig,ax=plt.subplots(figsize=(6.5,4))
            for block in sorted({r["block"] for r in data})[:12]:
                curve=[r for r in data if r["block"]==block]
                ax.plot([r["iteration"] for r in curve],[r["objective"] for r in curve],label=f"block {block}")
            ax.set_xlabel("Outer update"); ax.set_ylabel("Refreshed objective")
            ax.legend(fontsize=7,ncol=2); fig.tight_layout()
            name="_".join(path.parent.relative_to(root).parts)
            fig.savefig(dest/f"{name}_objective.png",dpi=180); plt.close(fig)


def panels(root,scene,retention,methods,phase="post",views=(0,),crop=None):
    """Same test camera indices and optional pixel crop for every method; no distortion."""
    from PIL import Image,ImageOps,ImageDraw
    root=Path(root); dest=root/"figures"; dest.mkdir(exist_ok=True)
    folders=[]
    for method in methods:
        if method=="identity":
            folders.append(root/scene/"original/identity/pre")
        else:
            folders.append(root/scene/f"rho_{retention:g}"/method/phase)
    manifests=[json.loads((folder/"camera_manifest.json").read_text()) for folder in folders]
    if len({m["split_sha256"] for m in manifests})!=1:
        raise ValueError("Cannot compare images from different camera splits")
    for index in views:
        if len({m["views"]["test"][index]["image_name"] for m in manifests})!=1:
            raise ValueError("Selected view indices do not name the same camera")
        images=[]
        for folder in folders:
            im=Image.open(folder/"test"/folder.name/"renders"/f"{index:05d}.png").convert("RGB")
            images.append(im.crop(tuple(crop)) if crop else im)
        gt=Image.open(folders[0]/"test"/folders[0].name/"gt"/f"{index:05d}.png").convert("RGB")
        if crop: gt=gt.crop(tuple(crop))
        images=[gt]+images; names=["GT"]+[label(m) for m in methods]
        if len({im.size for im in images})!=1:
            raise ValueError("Images differ in resolution; refusing to stretch them")
        width,height=images[0].size
        canvas=Image.new("RGB",(width*len(images),height+36),"white")
        draw=ImageDraw.Draw(canvas)
        for j,(im,name) in enumerate(zip(images,names)):
            canvas.paste(im,(j*width,36)); draw.text((j*width+10,10),name,fill="black")
        canvas.save(dest/f"{scene}_rho_{retention:g}_{phase}_view_{index:05d}{'_crop' if crop else ''}.png")
