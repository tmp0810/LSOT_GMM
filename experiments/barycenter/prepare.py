"""Prepare image clouds/GMMs or reproduce bundled masks from the notebook.

python -m experiments.barycenter.prepare --output-dir data/barycenter
python -m experiments.barycenter.prepare --recover-notebook-output /path/reference.ipynb --mask-output /tmp/masks.json
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import zlib

from matplotlib import colormaps
import numpy as np
from PIL import Image
from threadpoolctl import threadpool_limits

from .data import image_example, SHAPE_NAMES, NOTEBOOK_URL


def recover_masks(notebook, output):
    """Recover foreground support from cell 18's saved viridis rendering.

    This does not fabricate unavailable original PNGs. The supported notebook
    blob is pinned; the manifest describes the recovery and its limitations.
    """
    notebook_bytes = Path(notebook).read_bytes()
    saved = json.loads(notebook_bytes)
    displays = [item for item in saved["cells"][18]["outputs"] if "image/png" in item.get("data", {})]
    encoded = displays[0]["data"]["image/png"]
    if isinstance(encoded, list):
        encoded = "".join(encoded)
    image_bytes = base64.b64decode(encoded)
    expected_display = "432c4d928f2509cb41cae4b3bca8831b381a5f4af6378728b89ac0c009f5b798"
    if hashlib.sha256(image_bytes).hexdigest() != expected_display:
        raise ValueError("Saved display does not match the pinned MW2 notebook output")
    image = np.asarray(Image.open(io.BytesIO(image_bytes)))[..., :3]
    if image.shape[:2] != (301, 1271):
        raise ValueError("Unexpected saved display; use notebook blob 78ed4c246b4bc89cbd98ecdacd4d66b9a0e4d02a")
    palette = (colormaps["viridis"](np.arange(256))[:, :3]*255).astype(np.uint8)
    codes = (image[..., 0].astype(np.int64)<<16)+(image[..., 1].astype(np.int64)<<8)+image[..., 2]
    unique, inverse = np.unique(codes, return_inverse=True)
    colors = np.column_stack((unique>>16, (unique>>8)&255, unique&255))
    distances = ((colors[:, None, :].astype(float)-palette[None].astype(float))**2).sum(-1)
    blue = (255-distances.argmin(-1)[inverse]).reshape(image.shape[:2])
    result = {"source_notebook": NOTEBOOK_URL,
              "source_notebook_blob": "78ed4c246b4bc89cbd98ecdacd4d66b9a0e4d02a",
              "source_cell": 18,
              "source_display_sha256": expected_display,
              "provenance": "Foreground masks recovered from the saved input-image display (inverse viridis lookup at 128x128 pixel centers). Original PNGs are absent from gmmot. The redcross and duck foreground masks match the corresponding POT images exactly. Raw PNG identity for star and batman cannot be verified.",
              "shape": [128, 128], "encoding": "zlib-compressed numpy.packbits, bitorder=big, C order", "masks": {}}
    for name, left, width in zip(SHAPE_NAMES, [33.5, 353.5, 674., 995.], [266.25, 266.5, 266.5, 266.5]):
        x = np.floor(left+(np.arange(128)+.5)*width/128).astype(int)
        y = np.floor(10.+(np.arange(128)+.5)*width/128).astype(int)
        mask = blue[y[:, None], x[None, :]] < 255
        result["masks"][name] = {
            "data": base64.b64encode(zlib.compress(np.packbits(mask).tobytes())).decode(),
            "foreground_pixels": int(mask.sum()),
            "mask_sha256": hashlib.sha256(mask.astype(np.uint8).tobytes()).hexdigest(),
            "display_sampling": {"left": left, "top": 10., "width": width}}
    Path(output).write_text(json.dumps(result, indent=2)+"\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/barycenter"))
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--components", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--recover-notebook-output", type=Path)
    parser.add_argument("--mask-output", type=Path, default=Path("recovered_masks.json"))
    args = parser.parse_args(argv)
    if args.components < 1 or args.seed < 0:
        parser.error("components must be positive and seed nonnegative")
    if args.recover_notebook_output:
        recover_masks(args.recover_notebook_output, args.mask_output)
        print(f"Recovered masks saved to {args.mask_output}")
        return
    with threadpool_limits(limits=1):
        case = image_example(args.output_dir, image_dir=args.image_dir,
                             components=args.components, seed=args.seed)
    print(f"Prepared {len(case.inputs)} image GMMs in {args.output_dir}")


if __name__ == "__main__":
    main()
