"""The main barycenter grid compares the four projection families end to end."""
import json

import numpy as np
import pandas as pd
import pytest
import torch

from experiments.barycenter.data import load_gmm
from experiments.barycenter.fast_solver import FastObjective
from experiments.barycenter.simple_initialization_grid import main
from lsot.projections import sample_projection_bank


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_default_grid_uses_all_min_projections(tmp_path, device):
    main(['--cases', 'synthetic', '--component-counts', '4', '--grid-size', '3',
          '--steps', '12', '--device', device, '--no-plots', '--output-dir', str(tmp_path)])
    summary = pd.read_csv(tmp_path/'summary.csv')
    kinds = ('Mix', 'SMix', 'B', 'B1D')
    assert set(summary['name']) == {'MW2', *(f'min-LSOT-{kind}' for kind in kinds)}
    assert (summary['grid_total_ms_median'] >= summary['transport_time']).all()
    assert summary.loc[summary['name'].eq('MW2'), 'runtime_device'].eq('cpu').all()
    assert summary.loc[summary['name'].ne('MW2'), 'runtime_device'].str.startswith(device).all()
    case = tmp_path/'synthetic'/'K4'
    inputs = [load_gmm(case/'inputs'/f'input_{j}.npz') for j in range(4)]
    initial = None
    for kind in kinds:
        method = case/'repeat_00'/f'min-LSOT-{kind}'
        node = method/'barycenters'/'r01_c01.npz'
        initial_gmm = load_gmm(method/'initializations'/node.name)
        params = (initial_gmm.weights, initial_gmm.means, initial_gmm.covariances)
        if initial is None:
            initial = params
        for a, b in zip(params, initial):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        record = json.loads((method/'records'/node.with_suffix('.json').name).read_text())
        bank = sample_projection_bank(2, 512, kind=kind, seed=1000003)
        with torch.no_grad():
            selected_loss = float(FastObjective(inputs, [0.25]*4, bank, kind, 'min')(load_gmm(node)))
        assert record['validation_objective'] == pytest.approx(selected_loss, abs=1e-12)
        assert record['selected_step'] in (0, 10, 12)
        assert np.isfinite(record['final_mw2'])
