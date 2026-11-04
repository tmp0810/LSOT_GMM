"""Input validation and byte-level manifests; no implicit checkpoint substitution."""
from pathlib import Path

from .common import digest,sha256,write_json
from .model_io import load_scene


def prepare_scene(setting,*,require_dataset=True):
    checkpoint=Path(setting["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint {checkpoint}. Set scenes.<name>.checkpoint; "
                                "or explicitly train a new backbone with 3DGS/train_backbone.py.")
    scene=load_scene(checkpoint,spatial_lr_scale=setting.get("spatial_lr_scale"))
    expected=setting.get("backbone_iterations")
    if expected is not None and scene.iteration is not None and expected!=scene.iteration:
        raise ValueError(f"Checkpoint iteration {scene.iteration} differs from requested {expected}")
    record={"checkpoint":str(checkpoint),"checkpoint_sha256":sha256(checkpoint),
            "gaussian_count":scene.count,"sh_degree":scene.degree,"source_iteration":scene.iteration,
            "spatial_lr_scale":scene.spatial_lr_scale,
            "backbone_provenance":setting.get("backbone_provenance","user_supplied")}
    if scene.degree!=setting.get("sh_degree",3):
        raise ValueError(f"Checkpoint SH degree {scene.degree} does not match scene config")
    if require_dataset:
        root=Path(setting.get("dataset",""))
        image_dir=root/setting.get("images","images")
        if not (root/"sparse/0").is_dir() or not image_dir.is_dir():
            raise FileNotFoundError(f"Expected COLMAP scene with sparse/0 and {image_dir}")
        camera_files=[p for p in (root/"sparse/0").iterdir()
                      if p.name in ("cameras.bin","images.bin","points3D.bin",
                                    "cameras.txt","images.txt","points3D.txt")]
        if not any(p.name.startswith("cameras.") for p in camera_files) or not any(
                p.name.startswith("images.") for p in camera_files):
            raise FileNotFoundError("Missing COLMAP cameras/images files")
        images=sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in (".png",".jpg",".jpeg"))
        if not images:
            raise ValueError(f"No images in {image_dir}")
        files={str(p.relative_to(root)):sha256(p) for p in sorted(camera_files+images)}
        record.update(dataset=str(root.resolve()),input_files=files,image_count=len(images),
                      dataset_sha256=digest(files),resolution=setting.get("resolution",-1),
                      split="pinned GHAP COLMAP --eval; actual camera manifest checked after loading")
    record["input_sha256"]=digest(record)
    return record
