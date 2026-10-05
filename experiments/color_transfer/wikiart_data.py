"""Pinned WikiArt pair manifest and on-demand image preparation.

The upstream repository commits URL and pair lists, not the painting files.
Downloaded images are cached locally and are never committed to this project.
"""

from dataclasses import dataclass
from hashlib import sha256
from html.parser import HTMLParser
from io import BytesIO
import json
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from PIL import Image, ImageOps, UnidentifiedImageError


UPSTREAM_REPO = "https://github.com/tmp0810/Sliced-Amortized-OT"
UPSTREAM_REVISION = "59234586542d905d951eae1af7b03566733e528f"
MANIFEST_DIR = Path(__file__).resolve().parent / "wikiart"
DEFAULT_PAIR_INDICES = (0, 2, 3, 5, 14)
MAX_DOWNLOAD_BYTES = 40 * 1024 * 1024
# The upstream WikiArt page for this one artwork returns 404. Commons lists
# the same painting by Franz Marc, with a stable original-file URL.
FALLBACK_IMAGES = {
    "cheerful-forms-1914": {
        "page_url": "https://commons.wikimedia.org/wiki/File:Marc_-_Cheerful_Forms,_1914,_Hoberg,_Jansen_237.jpg",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/9/92/Marc_-_Cheerful_Forms%2C_1914%2C_Hoberg%2C_Jansen_237.jpg",
    },
}


@dataclass(frozen=True)
class PaintingPair:
    index: int
    source: str
    target: str

    @property
    def name(self):
        return f"pair_{self.index:02d}_{self.source}__{self.target}"


def load_manifest(directory=MANIFEST_DIR):
    """Validate that every requested painting has exactly one source URL."""
    directory = Path(directory)
    urls = {}
    for line in (directory / "wikiart-urls.txt").read_text(encoding="utf-8").splitlines():
        url = line.strip()
        if not url or url.startswith("#"):
            continue
        parsed = urlparse(url)
        slug = parsed.path.rstrip("/").rsplit("/", 1)[-1]
        if (parsed.scheme != "https" or parsed.netloc != "www.wikiart.org"
                or not parsed.path.startswith("/en/") or not slug or slug in urls):
            raise ValueError(f"Invalid or duplicate WikiArt URL: {url}")
        urls[slug] = url
    pairs = []
    for line in (directory / "pairs.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 2 or any(slug not in urls for slug in fields):
            raise ValueError(f"Pair does not reference two known paintings: {line}")
        pairs.append(PaintingPair(len(pairs), *fields))
    if not pairs:
        raise ValueError("WikiArt pair manifest is empty")
    return pairs, urls


def select_pairs(pairs, indices):
    if not indices or any(type(i) is not int or not 0 <= i < len(pairs) for i in indices):
        raise ValueError(f"Pair indices must be in [0, {len(pairs) - 1}]")
    if len(set(indices)) != len(indices):
        raise ValueError("Pair indices must be distinct")
    return [pairs[i] for i in indices]


class _ArtworkImageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.primary = None
        self.fallback = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "meta" and (attrs.get("property") == "og:image"
                              or attrs.get("name") == "twitter:image"):
            self.primary = self.primary or attrs.get("content")
        if tag == "img":
            for key in ("data-src", "src", "data-original"):
                value = attrs.get(key, "")
                if "uploads" in value and "wikiart.org" in value:
                    self.fallback = self.fallback or value


def _read_url(url, *, timeout, opener, max_bytes):
    request = Request(url, headers={
        "User-Agent": "Mozilla/5.0 (compatible; LSOT-GMM-WikiArt/1.0)",
        "Accept": "text/html,image/*;q=0.9,*/*;q=0.8",
    })
    with opener(request, timeout=timeout) as response:
        data = response.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise ValueError(f"Download exceeds {max_bytes} bytes: {url}")
    return data


def artwork_image_url(page_url, html):
    parser = _ArtworkImageParser()
    parser.feed(html.decode("utf-8", errors="replace"))
    candidate = parser.primary or parser.fallback
    if not candidate:
        raise ValueError(f"WikiArt page has no artwork image: {page_url}")
    image_url = urljoin(page_url, candidate)
    parsed = urlparse(image_url)
    if parsed.scheme != "https" or not (parsed.hostname or "").endswith(".wikiart.org"):
        raise ValueError(f"Unexpected WikiArt image host: {image_url}")
    return image_url


def _checked_image(path):
    try:
        with Image.open(path) as opened:
            opened.verify()
        with Image.open(path) as opened:
            if min(opened.size) < 2:
                raise ValueError(f"Image is too small: {path}")
            return list(opened.size)
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Invalid painting image: {path}") from exc


def prepare_images(pairs, urls, image_dir, *, download=True, force=False,
                   timeout=30, opener=urlopen):
    """Fetch only unique paintings in selected pairs, or verify local images.

    A pre-existing .jpg/.jpeg/.png can be supplied manually. Downloads are
    converted once to RGB PNG after EXIF correction. Experiment resizing and
    GMM fitting happen later inside run_experiment, identically for all methods.
    """
    image_dir = Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    slugs = list(dict.fromkeys(slug for pair in pairs for slug in (pair.source, pair.target)))
    inventory = {}
    for slug in slugs:
        existing = next((image_dir / f"{slug}{ext}" for ext in (".png", ".jpg", ".jpeg")
                        if (image_dir / f"{slug}{ext}").is_file()), None)
        page_url = urls[slug]
        image_url = None
        downloaded_sha = None
        download_page_url = page_url
        used_fallback = False
        if existing is None or force:
            if not download:
                raise FileNotFoundError(f"Missing painting {slug} in {image_dir}; "
                                        f"download it from {page_url} or omit --no-download")
            try:
                try:
                    html = _read_url(page_url, timeout=timeout, opener=opener,
                                     max_bytes=2 * 1024 * 1024)
                except HTTPError as error:
                    if error.code != 404 or slug not in FALLBACK_IMAGES:
                        raise
                    used_fallback = True
                    download_page_url = FALLBACK_IMAGES[slug]["page_url"]
                    image_url = FALLBACK_IMAGES[slug]["image_url"]
                else:
                    image_url = artwork_image_url(page_url, html)
                raw = _read_url(image_url, timeout=timeout, opener=opener,
                                max_bytes=MAX_DOWNLOAD_BYTES)
                downloaded_sha = sha256(raw).hexdigest()
                with Image.open(BytesIO(raw)) as opened:
                    rgb = ImageOps.exif_transpose(opened).convert("RGB")
                    if min(rgb.size) < 2:
                        raise ValueError("Downloaded image has fewer than two pixels on one side")
                    destination = image_dir / f"{slug}.png"
                    temporary = image_dir / f"{slug}.png.part"
                    try:
                        rgb.save(temporary, format="PNG")
                        temporary.replace(destination)
                    finally:
                        temporary.unlink(missing_ok=True)
                existing = destination
            except Exception as exc:
                raise RuntimeError(f"Could not prepare {slug} from {page_url}: {exc}. "
                                   f"Place {slug}.jpg or {slug}.png in {image_dir} "
                                   "and rerun with --no-download.") from exc
        size = _checked_image(existing)
        file_hash = sha256(existing.read_bytes()).hexdigest()
        sidecar = image_dir / f"{slug}.source.json"
        if image_url is not None:
            source = {"sha256": file_hash, "download_sha256": downloaded_sha,
                      "image_url": image_url, "download_page_url": download_page_url,
                      "used_fallback": used_fallback}
            sidecar.write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")
        elif sidecar.is_file():
            source = json.loads(sidecar.read_text(encoding="utf-8"))
            if source.get("sha256") == file_hash:
                image_url = source["image_url"]
                downloaded_sha = source["download_sha256"]
                download_page_url = source["download_page_url"]
                used_fallback = source["used_fallback"]
        inventory[slug] = {
            "path": str(existing.resolve()), "page_url": page_url,
            "download_page_url": download_page_url, "used_fallback": used_fallback,
            "image_url": image_url, "download_sha256": downloaded_sha,
            "sha256": file_hash, "size": size,
        }
    return inventory


def save_inventory(path, pairs, inventory):
    content = {
        "upstream_repo": UPSTREAM_REPO, "upstream_revision": UPSTREAM_REVISION,
        "upstream_pair_file": "data_color_transfer/paintings/pairs.txt",
        "upstream_url_file": "data_color_transfer/wikiart-urls.txt",
        "pairs": [vars(pair) | {"directory": pair.name} for pair in pairs],
        "images": inventory,
    }
    Path(path).write_text(json.dumps(content, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
