import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg

from experiments.long_rollout_plots import (
    failure_curves,
    failure_fraction,
    kinetic_energy_curves,
    rollout_curves,
    vorticity_constraints,
)


@pytest.fixture
def results():
    times = np.array([0.0, 0.025, 1.0, 10.0, 20.0])
    runs = []
    for representation in ("velocity", "vorticity"):
        for model in ("PHFNO", "FNO"):
            tests = {}
            for family in ("wave", "taylor_green", "random"):
                values = np.array([[0, 1, 1e60, np.nan, np.nan], [0, 3, 5, 7, 9]])
                tests[family] = {
                    "times": times,
                    "trajectory_indices": [16, 19],
                    "nrmse_time": values,
                    "kinetic_energy": values + 1,
                    "true_kinetic_energy": np.ones_like(values),
                    "vorticity_nrmse_time": values,
                    "vorticity_consistency_nrmse_time": values,
                    "vorticity_divergence_rms": values,
                    "vorticity_mean_norm": values,
                    "failures": [{"trajectory_index": 16, "step": 3, "time": 10.0,
                                  "reason": "nonfinite model prediction"}],
                }
            runs.append({"representation": representation, "model": model, "seed": 7, "test": tests})
    return {"families": ["wave", "taylor_green", "random"], "runs": runs}


@pytest.mark.parametrize("plot, axes_count", [
    (rollout_curves, 6), (kinetic_energy_curves, 6),
    (vorticity_constraints, 12), (failure_curves, 6),
])
def test_long_rollout_figures_render_full_interval_without_footers(results, plot, axes_count):
    figure = plot(results)
    try:
        FigureCanvasAgg(figure).draw()
        assert len(figure.axes) == axes_count
        assert not [text for text in figure.texts if text is not figure._suptitle]
        for axis in figure.axes:
            assert axis.get_xlim() == (0, 20)
            if plot is not failure_curves:
                assert axis.get_yscale() == "symlog"
                assert axis.get_ylim()[1] > 1e60
    finally:
        plt.close(figure)


def test_failed_trajectories_do_not_disappear_from_mean(results):
    figure = rollout_curves(results)
    try:
        axis = figure.axes[0]
        mean = next(line for line in axis.lines if line.get_label() == "PHFNO · seed 7")
        np.testing.assert_allclose(mean.get_ydata()[:3], [0, 2, 5e59])
        assert np.isnan(mean.get_ydata()[3:]).all()
        assert any(np.array_equal(line.get_ydata(), [0, 3, 5, 7, 9]) for line in axis.lines)
        assert any(line.get_marker() == "x" and 10 in line.get_xdata() for line in axis.lines)
        assert "velocity-trained" in axis.get_title()
        assert "vorticity-trained" in figure.axes[3].get_title()
    finally:
        plt.close(figure)


@pytest.mark.parametrize("maximum", [0.1, 1e60])
def test_error_ticks_are_readable_for_small_errors_and_divergence(results, maximum):
    for run in results["runs"]:
        for metrics in run["test"].values():
            metrics["nrmse_time"] = metrics["nrmse_time"] * (maximum / 1e60)
    figure = rollout_curves(results)
    try:
        FigureCanvasAgg(figure).draw()
        for axis in figure.axes:
            ticks = axis.get_yticks()
            upper = axis.get_ylim()[1]
            visible = ticks[(ticks >= 0) & (ticks <= upper)]
            assert 3 <= len(visible) <= 9
            assert upper > maximum
            assert axis.get_yscale() == "symlog"
    finally:
        plt.close(figure)


def test_failure_fraction_merges_nan_and_nonconvergence_by_heldout_index(results):
    metrics = results["runs"][0]["test"]["wave"]
    metrics["failures"].append({"trajectory_index": 19, "step": 4, "time": 20.0,
                                "reason": "implicit integrator did not converge"})
    np.testing.assert_allclose(failure_fraction(metrics), [0, 0, 0, 0.5, 1])
    metrics["failures"].append(dict(metrics["failures"][0]))
    np.testing.assert_allclose(failure_fraction(metrics), [0, 0, 0, 0.5, 1])


def test_notebook_is_code_only_and_uses_separate_cuda_evaluation():
    path = Path(__file__).resolve().parents[1] / "ipynb" / "compare_long_rollout.ipynb"
    notebook = json.loads(path.read_text())
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert all(cell["cell_type"] == "code" for cell in notebook["cells"])
    assert notebook["metadata"]["kernelspec"]["name"] == "research"
    assert 'Path(sys.prefix).name != "research"' in source
    assert "torch.cuda.is_available()" in source
    assert 'device="cuda"' in source
    assert "run_long_comparison(" in source
    assert '"phfno_fno_long_rollout"' in source
    assert source.count("plt.show()") == 4


def test_notebook_failure_summary_includes_nonfinite_diagnostics(results, capsys):
    path = Path(__file__).resolve().parents[1] / "ipynb" / "compare_long_rollout.ipynb"
    notebook = json.loads(path.read_text())
    source = "".join(notebook["cells"][1]["source"])
    summary_source = source[source.index('print("Training state'):]
    for run in results["runs"]:
        run["best_step"] = 100
        for metrics in run["test"].values():
            for key, value in metrics.items():
                if isinstance(value, np.ndarray):
                    metrics[key] = torch.from_numpy(value.copy())
            metrics["vorticity_mean_norm"][1, 2:] = torch.nan
    exec(summary_source, {"results": results, "torch": torch, "failure_fraction": failure_fraction})
    rows = capsys.readouterr().out.splitlines()[1:]
    assert len(rows) == 4
    assert all("6/6" in row and "1.000" in row for row in rows)
