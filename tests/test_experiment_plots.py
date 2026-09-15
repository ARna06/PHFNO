import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg

from experiments.plots import equation_curves, field_slices, forcing_curves, learning_curves, physics_curves, rollout_curves, summary_table


@pytest.fixture
def results():
    families = ["wave", "random", "taylor_green"]
    truth = torch.ones(1, 3, 3, 4, 4, 4)
    truth[:, 1] *= 0.8
    truth[:, 2] *= 0.6
    runs = []
    for model_index, model in enumerate(("PHFNO", "FNO")):
        for seed in (0, 1):
            factor = 0.1 * (1 + model_index + seed)
            prediction = truth + factor
            metrics = {
                "times": torch.tensor([0.0, 0.5, 1.0]),
                "truth": truth.clone(),
                "prediction": prediction,
                "initial_rms": torch.ones(1),
                "nrmse_time": torch.tensor([[0.0, factor, 2 * factor]]),
                "kinetic_energy": 0.5 * prediction.square().sum(dim=2).mean(dim=(2, 3, 4)),
                "true_kinetic_energy": 0.5 * truth.square().sum(dim=2).mean(dim=(2, 3, 4)),
                "divergence_rms": torch.zeros(1, 3),
                "true_divergence_rms": torch.zeros(1, 3),
                "rhs_relative_error_time": torch.full((1, 2), factor),
                "secant_rhs_relative_error_time": torch.full((1, 2), 0.02),
                "power_residual_time": torch.tensor([[-factor, factor]]),
                "reference_power_scale": torch.tensor([0.2]),
                "force_nrmse_time": torch.full((1, 2), factor),
            }
            elapsed = {(0, 0): [0, 1, 3], (0, 1): [0.2, 1.2, 2], (1, 0): [0, 0.4, 0.8], (1, 1): [0.1, 0.5, 1]}
            history = [
                {"step": step, "optimization_seconds": second, "validation_nrmse": error + factor}
                for step, second, error in zip((0, 10, 20), elapsed[model_index, seed], (0.5, 0.3, 0.1))
            ]
            runs.append({
                "model": model,
                "seed": seed,
                "history": history,
                "threshold_nrmse": 0.15,
                "threshold_step": 20 if seed == 0 else None,
                "parameter_count": 1000,
                "best_validation_nrmse": history[-1]["validation_nrmse"],
                "train_seconds": history[-1]["optimization_seconds"],
                "test": {family: metrics.copy() for family in families},
            })
    return {"families": families, "runs": runs, "persistence_validation_nrmse": 0.3}


@pytest.mark.parametrize(
    "plot, axes_count",
    [(learning_curves, 2), (rollout_curves, 3), (physics_curves, 6), (equation_curves, 6), (forcing_curves, 3), (field_slices, 7), (summary_table, 1)],
)
def test_comparison_figures_render(results, plot, axes_count):
    figure = plot(results)
    try:
        FigureCanvasAgg(figure).draw()
        assert len(figure.axes) == axes_count
        assert figure._suptitle or figure.axes[0].get_title()
    finally:
        plt.close(figure)


def test_learning_curves_use_shared_time_ranges_and_seed_means(results):
    figure = learning_curves(results)
    try:
        step_lines = {line.get_label(): line for line in figure.axes[0].lines}
        time_lines = {line.get_label(): line for line in figure.axes[1].lines}
        np.testing.assert_allclose(step_lines["PHFNO"].get_ydata(), [0.65, 0.45, 0.25])
        np.testing.assert_allclose(time_lines["PHFNO"].get_xdata()[[0, -1]], [0.2, 2.0])
        np.testing.assert_allclose(time_lines["FNO"].get_xdata()[[0, -1]], [0.1, 0.8])
    finally:
        plt.close(figure)


def test_physics_plot_does_not_change_zero_divergence(results):
    figure = physics_curves(results)
    try:
        assert results["runs"][0]["test"]["wave"]["divergence_rms"].count_nonzero() == 0
        assert min(figure.axes[3].lines[0].get_ydata()) == pytest.approx(1e-10)
    finally:
        plt.close(figure)


def test_field_slices_share_scales_and_default_to_first_seed(results):
    figure = field_slices(results)
    try:
        images = [axis.images[0] for axis in figure.axes[:5]]
        assert images[0].get_clim() == images[1].get_clim() == images[2].get_clim()
        assert images[3].get_clim() == images[4].get_clim()
        np.testing.assert_allclose(images[3].get_array(), 0.1, atol=1e-7)
        np.testing.assert_allclose(images[4].get_array(), 0.2, atol=1e-7)
    finally:
        plt.close(figure)


def test_summary_counts_only_runs_that_reached_the_target(results):
    figure = summary_table(results)
    try:
        table = figure.axes[0].tables[0]
        assert table[1, 4].get_text().get_text() == "1 / 2"
        assert table[1, 5].get_text().get_text() == "20"
        assert table[2, 4].get_text().get_text() == "1 / 2"
    finally:
        plt.close(figure)
