import json
from dataclasses import asdict
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from torch import nn

from experiments.comparison import ComparisonConfig, restore_completed_runs, restore_training_run, summarize_run
from experiments.metrics import evaluate_model
from experiments.plots import _band, _mean_sd
from experiments.vorticity_comparison import summarize_vorticity_run
from experiments.vorticity_metrics import evaluate_vorticity_model
from test_vorticity_metrics import wave_data


class FailingStep(nn.Module):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold
        self.coordinates = SimpleNamespace(project=lambda x: x)

    def forward(self, state, control):
        return torch.zeros_like(state)

    def step(self, state, control, dt, **options):
        assert options['method'] == 'avf'
        if state.abs().amax() > self.threshold:
            raise RuntimeError('AVF step did not converge after 100 iterations')
        return state


@pytest.mark.parametrize('vorticity', [False, True])
def test_failure_keeps_other_trajectories_and_raw_diagnostics(vorticity):
    data, _, _ = wave_data()
    model = FailingStep(9 if vorticity else 1.5)
    evaluate = evaluate_vorticity_model if vorticity else evaluate_model
    result = evaluate(model, data, [1, 0], device='cpu', method='avf', record_failures=True)
    assert model.training
    assert result['failures'] == [{
        'trajectory_index': 0, 'source_trajectory_index': 1, 'step': 1, 'time': 0.1,
        'reason': 'AVF step did not converge after 100 iterations',
    }]
    assert torch.isfinite(result['prediction'][:, 0]).all()
    assert torch.isnan(result['prediction'][0, 1:]).all()
    assert torch.isfinite(result['prediction'][1]).all()
    assert torch.isfinite(result['predicted_rhs']).all()
    if vorticity:
        assert torch.isnan(result['vorticity_prediction'][0, 1:]).all()
        assert torch.isfinite(result['vorticity_consistency_nrmse_time'][1]).all()
    summarize = summarize_vorticity_run if vorticity else summarize_run
    summary = summarize({'test': {'wave': result}})
    assert summary['test']['wave']['final_nrmse'] is None
    assert summary['test']['wave']['failures'] == result['failures']
    json.dumps(summary, allow_nan=False)


def test_unrelated_runtime_errors_still_raise():
    data, _, _ = wave_data()
    model = FailingStep(1)
    def broken(*args, **kwargs):
        raise RuntimeError('implementation bug')
    model.step = broken
    with pytest.raises(RuntimeError, match='implementation bug'):
        evaluate_model(model, data, [0], device='cpu', method='avf', record_failures=True)
    assert model.training


def test_failed_plot_has_marker_full_time_and_no_survivor_average():
    fig, ax = plt.subplots()
    _band(ax, [0, 1, 2], [[[0, 1, np.nan]], [[0, 2, 3]]], 'PHFNO', 'blue')
    assert ax.get_xlim() == (0, 2)
    assert 'failure' in ax.collections[0].get_label()
    assert any('failure' in text.get_text() for text in ax.get_legend().get_texts())
    assert np.isnan(ax.lines[-1].get_ydata()[-1])
    assert any('individual rollouts' in line.get_label() for line in ax.lines)
    assert any(np.array_equal(line.get_ydata(), [0, 2, 3]) for line in ax.lines)
    assert _mean_sd([1, np.nan]) == 'Unavailable (failure)'
    fig.canvas.draw()
    plt.close(fig)


def test_resume_requires_matching_protocol_and_checkpoint(tmp_path):
    config = ComparisonConfig(device='cpu', seeds=(8,))
    model = nn.Linear(2, 2)
    run = {'model': 'PHFNO', 'seed': 8, 'best_step': 100, 'test': {'wave': {}}}
    checkpoint = {'config': asdict(config), 'state_dict': model.state_dict(),
                  'model': 'PHFNO', 'seed': 8, 'best_step': 100,
                  'training_run': {'best_step': 100, 'history': [{'step': 100}]}}
    path = tmp_path / 'PHFNO_8.pt'
    torch.save(checkpoint, path)
    restored = nn.Linear(2, 2)
    assert restore_training_run(restored, path, config, name="PHFNO", seed=8) == checkpoint['training_run']
    torch.testing.assert_close(restored.weight, model.weight)
    with pytest.raises(ValueError, match="identity"):
        restore_training_run(restored, path, config, name="FNO", seed=8)
    record = {'config': asdict(config), 'families': ['wave'], 'dataset_sha256': {'wave': 'abc'}, 'runs': [run]}
    torch.save(record, tmp_path / 'results.pt')
    target = {**record, 'runs': []}
    restore_completed_runs(target, tmp_path)
    assert target['runs'] == [run]
    target['dataset_sha256'] = {'wave': 'changed'}
    with pytest.raises(ValueError, match='dataset_sha256'):
        restore_completed_runs(target, tmp_path)
    checkpoint['best_step'] = 200
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match='does not match'):
        restore_completed_runs({**record, 'runs': []}, tmp_path)


def test_interrupted_evaluation_manifest_protects_dataset_identity(tmp_path):
    record = {'config': asdict(ComparisonConfig(device='cpu')), 'families': ['wave'],
              'dataset_sha256': {'wave': 'original'}}
    (tmp_path / 'config.json').write_text(json.dumps(record))
    with pytest.raises(ValueError, match='dataset_sha256'):
        restore_completed_runs({**record, 'dataset_sha256': {'wave': 'changed'}}, tmp_path)
