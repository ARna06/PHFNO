import matplotlib.pyplot as plt
import numpy as np

from .plots import COLORS, FAMILY_NAMES, _band, _figure, _mean_sd, _model_runs


def representation_summary(velocity_results, vorticity_results):
    rows = []
    for name in COLORS:
        for representation, results in (("velocity", velocity_results), ("vorticity", vorticity_results)):
            runs = _model_runs(results, name)
            rollout, final, derivative = [], [], []
            for run in runs:
                metrics = [run["test"][family] for family in results["families"]]
                rollout.append(np.mean([np.asarray(item["nrmse_time"])[:, 1:].mean() for item in metrics]))
                final.append(np.mean([np.asarray(item["nrmse_time"])[:, -1].mean() for item in metrics]))
                derivative.append(np.mean([np.asarray(item["rhs_relative_error_time"]).mean() for item in metrics]))
            rows.append([
                f"{name} · {representation}",
                _mean_sd(rollout),
                _mean_sd(final),
                _mean_sd(derivative),
                _mean_sd([run["train_seconds"] for run in runs]),
                f"{runs[0]['parameter_count']:,}",
            ])
    columns = ["Model · fitted state", "Mean velocity\nrollout NRMSE", "Final velocity\nNRMSE", "Velocity derivative\nrelative error", "Optimization\nseconds", "Real parameters"]
    fig, ax = plt.subplots(figsize=(16, 3.7), layout="constrained")
    ax.set_axis_off()
    table = ax.table(cellText=rows, colLabels=columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#d9e1e5")
        if row == 0:
            cell.set_facecolor("#eef3f5")
            cell.set_text_props(weight="bold")
        elif column == 0:
            cell.set_text_props(color=COLORS[rows[row - 1][0].split(" · ")[0]], weight="bold")
    ax.set_title("Both representations scored in velocity · mean ± standard deviation across seeds", fontsize=14, pad=12)
    return fig


def representation_rollouts(velocity_results, vorticity_results):
    families = velocity_results["families"]
    fig, axes = _figure(1, len(families), height=4.5)
    for ax, family in zip(axes.flat, families):
        for name, color in COLORS.items():
            for representation, results, style in (("velocity", velocity_results, "-"), ("vorticity", vorticity_results, "--")):
                runs = _model_runs(results, name)
                times = np.asarray(runs[0]["test"][family]["times"])
                values = [run["test"][family]["nrmse_time"] for run in runs]
                _band(ax, times, values, f"{name} · {representation}", color)
                ax.lines[-1].set_linestyle(style)
        ax.set_title(FAMILY_NAMES.get(family, family))
        ax.set_xlabel("Time")
        ax.set_ylabel("Velocity rollout RMSE / initial RMS")
        ax.set_ylim(bottom=0)
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("Velocity and vorticity inputs · mean and range across seeds", fontsize=14)
    return fig


def vorticity_constraints(results):
    families = results["families"]
    quantities = (
        ("vorticity_nrmse_time", "Vorticity RMSE / initial RMS"),
        ("vorticity_consistency_nrmse_time", "Curl reconstruction mismatch / initial RMS"),
        ("vorticity_divergence_rms", "RMS vorticity divergence"),
    )
    fig, axes = _figure(len(quantities), len(families), height=10)
    for column, family in enumerate(families):
        times = np.asarray(results["runs"][0]["test"][family]["times"])
        for row, (key, label) in enumerate(quantities):
            ax = axes[row, column]
            for name, color in COLORS.items():
                values = [run["test"][family][key] for run in _model_runs(results, name)]
                _band(ax, times, values, name, color, floor=1e-10 if row == 2 else None)
            ax.set_xlabel("Time")
            ax.set_ylabel(label)
            if row == 2:
                ax.set_yscale("log")
            else:
                ax.set_ylim(bottom=0)
        axes[0, column].set_title(FAMILY_NAMES.get(family, family))
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("Raw vorticity predictions · mean and range across seeds", fontsize=14)
    return fig
