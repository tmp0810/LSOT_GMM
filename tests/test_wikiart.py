"""Offline checks of the pinned pair list, image preparation and paired report."""

import csv
from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from PIL import Image

from experiments.color_transfer.wikiart import run_pairs, summarize_pairs
from experiments.color_transfer.wikiart_data import (
    FALLBACK_IMAGES, artwork_image_url, load_manifest, prepare_images, select_pairs,
)


@dataclass
class FakeConfig:
    source: str = ""
    target: str = ""
    output_dir: str = ""
    download_reference: bool = False

    def validate(self):
        pass


class WikiArtTests(unittest.TestCase):
    def test_upstream_manifest_is_complete_and_valid(self):
        pairs, urls = load_manifest()
        self.assertEqual(len(pairs), 19)
        self.assertEqual([p.index for p in pairs], list(range(19)))
        self.assertTrue(all(p.source in urls and p.target in urls for p in pairs))
        with self.assertRaisesRegex(ValueError, "distinct"):
            select_pairs(pairs, [0, 0])

    def test_download_image_validate_cache_and_do_not_fetch_twice(self):
        pairs, urls = load_manifest()
        selected = select_pairs(pairs, [0])
        sample = BytesIO()
        Image.new("RGB", (5, 7), (90, 135, 180)).save(sample, format="JPEG")
        calls = []

        def opener(request, timeout):
            url = request.full_url
            calls.append(url)
            if url in urls.values():
                slug = url.rsplit("/", 1)[-1]
                return BytesIO(f'<meta property="og:image" content="https://uploads1.wikiart.org/{slug}.jpg">'.encode())
            return BytesIO(sample.getvalue())

        with tempfile.TemporaryDirectory() as directory:
            inventory = prepare_images(selected, urls, directory, opener=opener)
            self.assertEqual(len(inventory), 2)
            self.assertEqual(len(calls), 4)
            for item in inventory.values():
                self.assertEqual(item["size"], [5, 7])
                self.assertEqual(len(item["sha256"]), 64)
                with Image.open(item["path"]) as image:
                    self.assertEqual(image.mode, "RGB")
                    self.assertEqual(image.format, "PNG")
            again = prepare_images(selected, urls, directory, download=False,
                                   opener=lambda *a, **kw: self.fail("unexpected download"))
            self.assertEqual({s: v["sha256"] for s, v in again.items()},
                             {s: v["sha256"] for s, v in inventory.items()})
            Image.new("RGB", (3, 3)).save(Path(directory) / f"{selected[0].source}.jpg")
            self.assertEqual(prepare_images(selected, urls, directory, download=False)
                             [selected[0].source]["path"], inventory[selected[0].source]["path"])

    def test_no_outside_image_host(self):
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            artwork_image_url("https://www.wikiart.org/en/x/a",
                              b'<meta property="og:image" content="https://example.org/a.jpg">')

    def test_404_fallback_is_explicit_and_preserved_in_cache(self):
        pairs, urls = load_manifest()
        selected = select_pairs(pairs, [18])
        sample = BytesIO()
        Image.new("RGB", (4, 3), (11, 22, 33)).save(sample, format="JPEG")
        alternative = FALLBACK_IMAGES["cheerful-forms-1914"]

        def opener(request, timeout):
            if request.full_url == urls["cheerful-forms-1914"]:
                raise HTTPError(request.full_url, 404, "Not Found", None, None)
            if request.full_url == alternative["image_url"]:
                return BytesIO(sample.getvalue())
            if request.full_url in urls.values():
                return BytesIO(b'<meta property="og:image" content="https://uploads1.wikiart.org/other.jpg">')
            return BytesIO(sample.getvalue())

        with tempfile.TemporaryDirectory() as directory:
            item = prepare_images(selected, urls, directory, opener=opener)["cheerful-forms-1914"]
            self.assertTrue(item["used_fallback"])
            self.assertEqual(item["download_page_url"], alternative["page_url"])
            cached = prepare_images(selected, urls, directory, download=False)["cheerful-forms-1914"]
            self.assertEqual(cached["image_url"], alternative["image_url"])
            self.assertEqual(cached["sha256"], item["sha256"])

    def test_paired_differences_use_matching_seed_and_equal_pair_weights(self):
        rows = []
        for pair, mw2, lsot in ((0, 0.20, 0.15), (2, 0.10, 0.15)):
            common = {"pair_index": pair, "pair_name": f"pair_{pair}",
                      "source_slug": "a", "target_slug": "b", "seed": 7,
                      "L": 0, "K0": 50, "K1": 50,
                      "relative_cost_gap": 0.0, "transport_ms_mean": 1.0,
                      "identity_color_w2": 0.30, "identity_color_sw2": 0.25,
                      "guided_color_w2": None, "guided_color_sw2": None}
            rows.append({**common, "method": "MW2", "cost_squared": 0.04,
                         "color_w2": mw2, "color_sw2": mw2})
            rows.append({**common, "method": "min-LSOT-SMix", "L": 100,
                         "cost_squared": 0.05, "relative_cost_gap": 0.25,
                         "color_w2": lsot, "color_sw2": lsot})
        pair_summary, methods = summarize_pairs(rows)
        selected = [r for r in pair_summary if r["method"] == "min-LSOT-SMix"]
        self.assertAlmostEqual(selected[0]["delta_color_w2_to_mw2"], -0.05)
        self.assertAlmostEqual(selected[1]["delta_color_w2_to_mw2"], 0.05)
        method = next(r for r in methods if r["method"] == "min-LSOT-SMix")
        self.assertEqual(method["n_pairs"], 2)
        self.assertEqual(method["wins_delta_color_w2_to_mw2"], 1)
        self.assertAlmostEqual(method["mean_delta_color_w2_to_mw2"], 0.0)
        self.assertEqual(method["wins_delta_color_w2_to_identity"], 2)

    def test_run_pairs_records_manifest_and_resumes_only_same_inputs(self):
        pairs, _ = load_manifest()
        chosen = select_pairs(pairs, [0, 5])
        calls = []

        def fake_run(setting):
            calls.append(setting)
            result = [
                {"method": method, "seed": 0, "L": length, "K0": 50, "K1": 50,
                 "cost_squared": 0.2, "relative_cost_gap": 0,
                 "transport_ms_mean": 1, "color_sw2": 0.2, "color_w2": w2,
                 "guided_color_sw2": "", "guided_color_w2": "",
                 "identity_color_sw2": 0.4, "identity_color_w2": 0.4}
                for method, length, w2 in (("MW2", 0, 0.2), ("min-LSOT-SMix", 100, 0.18))
            ]
            path = Path(setting.output_dir) / "metrics.csv"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(result[0]))
                writer.writeheader()
                writer.writerows(result)
            return result

        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            image_dir = base / "images"
            image_dir.mkdir()
            for slug in {s for pair in chosen for s in (pair.source, pair.target)}:
                Image.new("RGB", (5, 5), (1, 2, 3)).save(image_dir / f"{slug}.png")
            with patch.dict(sys.modules, {"experiments.color_transfer.run":
                                         types.SimpleNamespace(run_experiment=fake_run)}):
                kwargs = dict(pair_indices=[0, 5], download=False)
                run_pairs(FakeConfig(), base / "results", image_dir, **kwargs)
                self.assertEqual(len(calls), 2)
                partial = base / "results" / chosen[0].name / "metrics.csv"
                partial.write_text(partial.read_text().splitlines()[0] + "\n")
                run_pairs(FakeConfig(), base / "results", image_dir, **kwargs)
                self.assertEqual(len(calls), 3)
                run_pairs(FakeConfig(), base / "results", image_dir, **kwargs)
                self.assertEqual(len(calls), 3)
            manifest = json.loads((base / "results/dataset.json").read_text())
            self.assertEqual([p["index"] for p in manifest["pairs"]], [0, 5])
            with (base / "results/method_summary.csv").open(newline="") as stream:
                methods = list(csv.DictReader(stream))
            lsot = next(item for item in methods if item["method"] == "min-LSOT-SMix")
            self.assertEqual(lsot["n_pairs"], "2")
            self.assertEqual(lsot["wins_delta_color_w2_to_mw2"], "2")


if __name__ == "__main__":
    unittest.main()
