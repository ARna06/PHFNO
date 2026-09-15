import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg

from experiments.vorticity_plots import representation_rollouts, representation_summary, vorticity_constraints


@pytest.fixture
def results_pair():
    results_pair = []
    for representation_factor in (1, 2):
        runs = []
        for model_index, model in enumerate(("PHFNO", "FNO")):
            for seed in (1, 3):
                factor = representation_factor * (model_index + 1) * seed
                runs.append({
                    "model": model,
                    "seed": seed,
                    "train_seconds": 10 * factor,
                    "parameter_count": 1000 * (model_index + 1),
                    "test": {
                        family: {
                            "times": torch.tensor([0.0, 0.5, 1.0]),
                            "nrmse_time": factor * torch.tensor([[0.0, 1.0, 2.0], [0.0, 3.0, 4.0]]),
                            "rhs_relative_error_time": factor * torch.tensor([[1.0, 3.0], [3.0, 5.0]]),
                            "vorticity_nrmse_time": factor * torch.ones(2, 3),
                            "vorticity_consistency_nrmse_time": 0.1 * factor * torch.ones(2, 3),
                            "vorticity_divergence_rms": torch.zeros(2, 3),
                        }
                        for family in ("wave", "random", "taylor_green")
                    },
                })
        results_pair.append({"families": ["wave", "random", "taylor_green"], "runs": runs})
    return results_pair


@pytest.mark.parametrize("plot, axes_count", [(representation_summary, 1), (representation_rollouts, 3), (vorticity_constraints, 9)])
def test_vorticity_figures_render_without_footers(results_pair, plot, axes_count):
    figure = plot(results_pair[1]) if plot is vorticity_constraints else plot(*results_pair)
    try:
        FigureCanvasAgg(figure).draw()
        assert len(figure.axes) == axes_count
        assert not [text for text in figure.texts if text is not figure._suptitle]
    finally:
        plt.close(figure)


def test_representation_summary_compares_velocity_metrics(results_pair):
    figure = representation_summary(*results_pair)
    try:
        table = figure.axes[0].tables[0]
        assert len(table.get_celld()) == 5 * 6
        assert [table[row, 0].get_text().get_text() for row in range(1, 5)] == [
            "PHFNO · velocity", "PHFNO · vorticity", "FNO · velocity", "FNO · vorticity",
        ]
        assert table[1, 1].get_text().get_text() == "5 ± 2.5"
        assert table[1, 2].get_text().get_text() == "6 ± 3"
        assert table[1, 3].get_text().get_text() == "6 ± 3"
        assert table[1, 4].get_text().get_text() == "20 ± 10"
        assert table[1, 5].get_text().get_text() == "1,000"
        assert table[2, 1].get_text().get_text() == "10 ± 5"
        assert all("validation" not in cell.get_text().get_text().lower() for cell in table.get_celld().values())
    finally:
        plt.close(figure)


def test_representation_rollouts_average_trajectories_before_seeds(results_pair):
    figure = representation_rollouts(*results_pair)
    try:
        ax = figure.axes[0]
        lines = {line.get_label(): line for line in ax.lines}
        np.testing.assert_allclose(lines["PHFNO · velocity"].get_ydata(), [0, 4, 6])
        np.testing.assert_allclose(lines["PHFNO · vorticity"].get_ydata(), [0, 8, 12])
        assert lines["PHFNO · velocity"].get_linestyle() == "-"
        assert lines["PHFNO · vorticity"].get_linestyle() == "--"
        assert lines["PHFNO · velocity"].get_color() == lines["PHFNO · vorticity"].get_color()
        vertices = ax.collections[0].get_paths()[0].vertices
        np.testing.assert_allclose(np.unique(vertices[vertices[:, 0] == 1.0, 1]), [3, 9])
    finally:
        plt.close(figure)


def test_vorticity_constraints_keep_raw_data_and_average_seeds(results_pair):
    results = results_pair[1]
    figure = vorticity_constraints(results)
    try:
        np.testing.assert_allclose(figure.axes[0].lines[0].get_ydata(), [4, 4, 4])
        np.testing.assert_allclose(figure.axes[3].lines[0].get_ydata(), [0.4, 0.4, 0.4])
        np.testing.assert_allclose(figure.axes[6].lines[0].get_ydata(), [1e-10] * 3)
        assert figure.axes[6].get_yscale() == "log"
        for run in results["runs"]:
            assert run["test"]["wave"]["vorticity_divergence_rms"].count_nonzero() == 0
    finally:
        plt.close(figure)
