"""Checks for normalization/cache, missing references and resource status semantics."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

from .scaling import prepare_data, evaluate
from .scaling_resources import supervise, MemoryMeter
from .data import load_gmm, synthetic_example
from .reference import mw2_barycenter


def test_normalized_shared_em_cache(tmp_path):
    args = SimpleNamespace(image_dir=None, em_steps=3, em_tolerance=.001)
    paths, meta = prepare_data(tmp_path, 2, 0, args)
    times = [Path(p).stat().st_mtime_ns for p in paths]
    again, cached = prepare_data(tmp_path, 2, 0, args)
    assert paths == again and meta == cached
    assert times == [Path(p).stat().st_mtime_ns for p in paths]
    for path in paths:
        g = load_gmm(path)
        assert g.count == 2
        assert bool(((g.means >= 0) & (g.means <= 1)).all())
        assert torch.linalg.eigvalsh(g.covariances).min() >= 1e-6 - 1e-12
    assert all(row['settings']['coordinate_divisor'] == 127 for row in meta)


def test_missing_reference_does_not_invent_zero_gaps():
    data = synthetic_example()
    args = SimpleNamespace(test_kinds=['B1D'], test_L=4, eval_samples=16)
    result = evaluate(data.inputs[0], data.inputs, [.25]*4, args, 0)
    assert result['mw2_objective'] > 0
    assert result['relative_mw2_gap'] is None
    assert result['sw2_to_reference'] is None
    assert result['mw2_to_reference'] is None
    assert result['test_min_B1D'] >= result['mw2_objective'] - 1e-10


def test_reference_timings_do_not_change_result():
    data = synthetic_example(); times={}
    first = mw2_barycenter(data.inputs, [.25]*4, fast=True, iterations=3, timings=times)
    other = mw2_barycenter(data.inputs, [.25]*4, fast=True, iterations=3)
    assert first.cost_squared == pytest.approx(other.cost_squared, abs=1e-12)
    assert set(times) == {'tuple_setup_ms','gaussian_cost_ms','constraint_setup_ms','lp_ms'}
    assert min(times.values()) >= 0


@pytest.mark.skipif(not Path('/proc/self/status').exists(), reason='Linux RSS watchdog')
@pytest.mark.parametrize('failure', ['timeout', 'memory_limit', 'killed_unknown'])
def test_watchdog_distinguishes_limits_and_unknown_kills(tmp_path, failure):
    script = tmp_path/'child.py'
    script.write_text("import time,json,os,signal\nfrom pathlib import Path\n"
                     f"Path({str(tmp_path/'ready.json')!r}).write_text(json.dumps(dict(monotonic=time.monotonic())))\n"
                     + ("os.kill(os.getpid(),signal.SIGKILL)\n" if failure == 'killed_unknown'
                        else "time.sleep(10)\n"))
    result=supervise([sys.executable,str(script)],tmp_path,
                     timeout_s=.05 if failure=='timeout' else 5,
                     memory_limit_gb=.000001 if failure=='memory_limit' else 0,
                     poll_s=.01)
    assert result['status'] == failure
    assert result['total_ms'] is None
    assert result['status'] != 'oom'


@pytest.mark.skipif(not Path('/proc/self/status').exists(), reason='Linux RSS sampling')
def test_solver_memory_window():
    with MemoryMeter('cpu',interval=.001) as meter:
        payload=bytearray(2*1024*1024)
    assert meter.stats['cpu_peak_rss_bytes'] >= meter.stats['cpu_rss_baseline_bytes'] > 0
    assert meter.stats['cuda_peak_allocated_bytes'] == 0


def test_synthetic_scaling_distinct_components_mass_and_cache(tmp_path):
    args = SimpleNamespace(data='synthetic', synthetic_spread=.5)
    paths, records = prepare_data(tmp_path/'first', 10, 7, args)
    repeated, same = prepare_data(tmp_path/'second', 10, 7, args)
    bases = synthetic_example().inputs
    for i, (path, other) in enumerate(zip(paths, repeated)):
        a, b = load_gmm(path), load_gmm(other)
        torch.testing.assert_close(a.means, b.means, rtol=0, atol=0)
        assert a.count == 10 and len(torch.unique(a.means, dim=0)) == 10
        assert torch.linalg.eigvalsh(a.covariances).min() > 0
        offset = 0
        for parent, n in enumerate(records[i]['parent_counts']):
            assert float(a.weights[offset:offset+n].sum()) == pytest.approx(float(bases[i].weights[parent]))
            offset += n
        assert records[i]['fit_ms'] == 0 and records[i]['converged'] is None
    times = [Path(p).stat().st_mtime_ns for p in paths]
    prepare_data(tmp_path/'first', 10, 7, args)
    assert times == [Path(p).stat().st_mtime_ns for p in paths]
    changed, _ = prepare_data(tmp_path/'newseed', 10, 8, args)
    assert not torch.equal(load_gmm(paths[0]).means, load_gmm(changed[0]).means)
    with pytest.raises(ValueError):
        prepare_data(tmp_path/'small', 3, 0, args)
