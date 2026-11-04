"""Render generated single-page figure PDFs for visual QA using Poppler."""
from pathlib import Path
import argparse
import math
import shutil
import subprocess
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--pdftoppm", default="pdftoppm")
    args = parser.parse_args()
    command = shutil.which(args.pdftoppm)
    if command is None:
        parser.error("pdftoppm not installed")
    source = args.directory.resolve()
    dest = source / "qa"
    dest.mkdir(exist_ok=True)
    paths = sorted(source.glob("*.pdf"))
    if not paths:
        parser.error("no PDFs to render")
    tile_w,tile_h = 760,520
    sheet = Image.new("RGB", (tile_w*3, tile_h*math.ceil(len(paths)/3)), "#e9edf0")
    draw = ImageDraw.Draw(sheet)
    for index,path in enumerate(paths):
        prefix = dest / path.stem
        subprocess.run([command, "-png", "-singlefile", "-r", "144", str(path), str(prefix)], check=True)
        im = Image.open(prefix.with_suffix(".png")).convert("RGB")
        im.thumbnail((tile_w-24,tile_h-46))
        x,y = (index%3)*tile_w,(index//3)*tile_h
        draw.text((x+10,y+10), path.name, fill="black")
        sheet.paste(im, (x+(tile_w-im.width)//2, y+36+(tile_h-40-im.height)//2))
    target = dest / "contact_sheet.png"
    sheet.save(target)
    print(target)


if __name__ == "__main__":
    main()
