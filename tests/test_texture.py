"""Checks for the texture-specific operators and the common pipeline."""
import json

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.nn.functional import unfold

from experiments.texture.data import adsn, aggregate_patches, extract_patches
from experiments.texture.run import main
from experiments.texture.synthesis import METHODS


def test_patch_layout_matches_stori_and_overlap_recovers_image():
    image = np.random.default_rng(11).random((9, 13, 3))
    patches = extract_patches(image, 3)
    expected = unfold(torch.from_numpy(image).permute(2, 0, 1)[None], 3)[0].T
    torch.testing.assert_close(torch.from_numpy(patches), expected, rtol=0, atol=0)
    rebuilt = aggregate_patches(torch.from_numpy(patches), image.shape[:2], 3)
    np.testing.assert_allclose(rebuilt.numpy(), image, atol=5e-16, rtol=0)


def test_adsn_convolves_one_noise_field_for_all_rgb_channels():
    image = np.random.default_rng(13).random((4, 7, 3))
    h, w, _ = image.shape
    mean = image.mean(axis=(0, 1))
    noise = np.random.default_rng(17).standard_normal((2*h, 2*w))
    expected = np.broadcast_to(mean, image.shape).copy()
    for i in range(h):
        for j in range(w):
            shifted = np.roll(noise, (i, j), axis=(0, 1))[:h, :w]
            expected += shifted[..., None]*(image[i, j]-mean)/np.sqrt(h*w)
    np.testing.assert_allclose(adsn(image, 17), expected, atol=1e-14, rtol=0)


def test_texture_cli_shares_mixtures_and_saves_feasible_plans(tmp_path):
    y, x = np.mgrid[:18, :22]
    pixels = np.stack((90 + 55*np.cos(x/2), 100 + 45*np.sin(y/2),
                       130 + 40*np.cos((x+y)/2)), axis=-1).astype(np.uint8)
    img = tmp_path/"test.png"
    Image.fromarray(pixels).save(img)
    config = tmp_path/"test.json"
    output = tmp_path/"out"
    config.write_text(json.dumps(dict(images=[str(img)], component_counts=[2, 3],
        output_dir=str(output), L=4, max_side=22, fit_patches=180, em_iterations=30,
        sw_samples=64, w_samples=32, eval_directions=12, batch_size=53,
        timing_repeats=2, download=False, plots=False)))
    main(["--config", str(config)])
    rows = pd.read_csv(output/"metrics.csv")
    assert len(rows) == 2*len(METHODS)
    assert set(rows.method) == set(METHODS)
    assert len(pd.read_csv(output/"timings.csv")) == 2*len(rows)
    assert len(pd.read_csv(output/"adsn_metrics.csv")) == 1
    assert np.isfinite(rows[["patch_sw2", "patch_w2", "autocov_rmse", "total_ms"]]).all().all()
    assert rows.plan_marginal_l1.max() < 1e-10
    assert (rows.method_ms >= rows.plan_ms).all()
    for k, frame in rows.groupby("K"):
        costs = frame.set_index("method").plan_cost_squared
        assert (costs >= costs["MW2"]-1e-10).all()
        folder = output/"test"/"seed_0"/f"K{k}"
        with np.load(folder/"mixtures.npz") as gmm:
            assert gmm["source_means"].shape == (k, 27)
            assert gmm["target_covariances"].shape == (k, 27, 27)
            for method in METHODS:
                with np.load(folder/f"{method}_plan.npz") as plan:
                    for side, index in (("source", "rows"), ("target", "cols")):
                        marginal = np.bincount(plan[index], plan["mass"], minlength=k)
                        np.testing.assert_allclose(marginal, gmm[f"{side}_weights"], atol=1e-10)
                with Image.open(folder/f"{method}.png") as image:
                    assert image.size == (22, 18)
        for method in METHODS:
            if method.startswith("min-"):
                assert costs[method] <= costs[method.replace("min-", "avg-", 1)] + 1e-10
