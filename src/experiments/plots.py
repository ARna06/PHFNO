import matplotlib.pyplot as plt
import numpy as np


COLORS = {"PHFNO": "#086788", "FNO": "#dd5e41"}
FAMILY_NAMES = {"wave": "Forced wave", "random": "Random initial flow", "taylor_green": "Taylor–Green flow"}


def _figure(rows, columns, width=15, height=4):
    fig, axes = plt.subplots(rows, columns, figsize=(width, height), squeeze=False, layout="constrained")
    fig.set_facecolor("white")
    for ax in axes.flat:
        ax.set_facecolor("white")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    return fig, axes


def _band(ax, x, values, label, color, floor=None):
    values = np.asarray(values)
    values = values.reshape(len(values), -1, len(x)).mean(axis=1)
    mean = values.mean(axis=0)
    low, high = values.min(axis=0), values.max(axis=0)
    if floor is not None:
        mean, low, high = [np.maximum(value, floor) for value in (mean, low, high)]
    ax.plot(x, mean, label=label, color=color, linewidth=2)
    ax.fill_between(x, low, high, color=color, alpha=0.15, linewidth=0)


def _model_runs(results, name):
    return [run for run in results["runs"] if run["model"] == name]


def learning_curves(results):
    fig, axes = _figure(1, 2, width=13, height=4.5)
    for name, color in COLORS.items():
        runs = _model_runs(results, name)
        histories = [run["history"] for run in runs]
        steps = sorted(set.intersection(*[{row["step"] for row in history} for history in histories]))
        values = [[next(row["validation_nrmse"] for row in history if row["step"] == step) for step in steps] for history in histories]
        _band(axes[0, 0], steps, values, name, color)
        starts = [history[0]["optimization_seconds"] for history in histories]
        ends = [history[-1]["optimization_seconds"] for history in histories]
        seconds = np.linspace(max(starts), min(ends), 200)
        interpolated = [np.interp(seconds, [row["optimization_seconds"] for row in history], [row["validation_nrmse"] for row in history]) for history in histories]
        _band(axes[0, 1], seconds, interpolated, name, color)
    for ax in axes.flat:
        ax.axhline(results["persistence_validation_nrmse"], color="#737373", linestyle="--", linewidth=1.4, label="Persistence")
        ax.axhline(results["runs"][0]["threshold_nrmse"], color="#333333", linestyle=":", linewidth=1.2, label="Shared target")
        ax.set_ylabel("Validation RMSE / training RMS")
        ax.set_yscale("log")
    axes[0, 0].set_xlabel("Optimizer steps")
    axes[0, 1].set_xlabel("Optimization time (seconds)")
    axes[0, 0].legend(frameon=False, fontsize=9)
    axes[0, 0].set_title("Learning per update")
    axes[0, 1].set_title("Learning per second · validation time excluded")
    fig.suptitle("One-step prediction · mean and range across seeds", fontsize=14)
    return fig


def rollout_curves(results):
    families = results["families"]
    fig, axes = _figure(1, len(families), height=4.3)
    for ax, family in zip(axes.flat, families):
        reference = results["runs"][0]["test"][family]
        times = np.asarray(reference["times"])
        for name, color in COLORS.items():
            values = [run["test"][family]["nrmse_time"] for run in _model_runs(results, name)]
            _band(ax, times, values, name, color)
        truth = np.asarray(reference["truth"])
        persistence = np.sqrt(np.mean((truth[:, :1] - truth) ** 2, axis=(2, 3, 4, 5)))
        persistence /= np.asarray(reference["initial_rms"])[:, None]
        ax.plot(times, persistence.mean(axis=0), color="#737373", linestyle="--", label="Persistence")
        ax.set_title(FAMILY_NAMES.get(family, family))
        ax.set_xlabel("Time")
        ax.set_ylabel("Rollout RMSE / initial RMS")
        ax.set_ylim(bottom=0)
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("Held-out rollouts · mean and range across seeds", fontsize=14)
    return fig


def physics_curves(results):
    families = results["families"]
    fig, axes = _figure(2, len(families), height=7)
    for column, family in enumerate(families):
        reference = results["runs"][0]["test"][family]
        times = np.asarray(reference["times"])
        for row, key in enumerate(("kinetic_energy", "divergence_rms")):
            ax = axes[row, column]
            for name, color in COLORS.items():
                values = [run["test"][family][key] for run in _model_runs(results, name)]
                _band(ax, times, values, name, color, floor=1e-10 if row else None)
            true_value = np.asarray(reference["true_" + key]).mean(axis=0)
            ax.plot(times, np.maximum(true_value, 1e-10) if row else true_value, color="#333333", linestyle="--", label="Navier–Stokes reference")
            ax.set_xlabel("Time")
            ax.set_ylabel("RMS divergence" if row else "Physical kinetic energy")
            if row:
                ax.set_yscale("log")
            else:
                ax.set_title(FAMILY_NAMES.get(family, family))
                ax.set_ylim(bottom=0)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Physics along autonomous rollouts · raw model fields", fontsize=14)
    return fig


def equation_curves(results):
    families = results["families"]
    fig, axes = _figure(2, len(families), height=7)
    for column, family in enumerate(families):
        reference = results["runs"][0]["test"][family]
        times = np.asarray(reference["times"])[:-1]
        for name, color in COLORS.items():
            metrics = [run["test"][family] for run in _model_runs(results, name)]
            _band(axes[0, column], times, [item["rhs_relative_error_time"] for item in metrics], name, color)
            residuals = [np.abs(np.asarray(item["power_residual_time"])) / np.asarray(item["reference_power_scale"])[:, None] for item in metrics]
            _band(axes[1, column], times, residuals, name, color)
        axes[0, column].plot(times, np.asarray(reference["secant_rhs_relative_error_time"]).mean(axis=0), color="#737373", linestyle="--", label="Snapshot secant vs. true derivative")
        axes[0, column].set_title(FAMILY_NAMES.get(family, family))
        axes[0, column].set_ylabel("Relative derivative error")
        axes[1, column].set_ylabel("Absolute power residual / reference scale")
        for ax in axes[:, column]:
            ax.set_xlabel("Time")
            ax.set_ylim(bottom=0)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Equation checks at clean held-out states · mean and seed range", fontsize=14)
    return fig


def forcing_curves(results):
    families = results["families"]
    fig, axes = _figure(1, len(families), height=4.3)
    for ax, family in zip(axes.flat, families):
        reference = results["runs"][0]["test"][family]
        times = np.asarray(reference["times"])[:-1]
        for name, color in COLORS.items():
            values = [run["test"][family]["force_nrmse_time"] for run in _model_runs(results, name)]
            _band(ax, times, values, name, color)
        ax.set_title(FAMILY_NAMES.get(family, family))
        ax.set_xlabel("Time")
        ax.set_ylabel("Forcing response RMSE / forcing RMS")
        ax.set_ylim(bottom=0)
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("Effect of the control · model response compared with the applied force", fontsize=14)
    return fig


def _mean_sd(values):
    values = np.asarray(values)
    return f"{values.mean():.3g} ± {values.std():.2g}"


def summary_table(results):
    fig, ax = plt.subplots(figsize=(16, 2.8), layout="constrained")
    ax.set_axis_off()
    rows = []
    for name in COLORS:
        runs = _model_runs(results, name)
        hits = [run["threshold_step"] for run in runs if run["threshold_step"] is not None]
        final_errors = [
            np.mean([np.asarray(run["test"][family]["nrmse_time"])[:, -1].mean() for family in results["families"]])
            for run in runs
        ]
        rows.append([
            name,
            f"{runs[0]['parameter_count']:,}",
            _mean_sd([run["best_validation_nrmse"] for run in runs]),
            _mean_sd([run["train_seconds"] for run in runs]),
            f"{len(hits)} / {len(runs)}",
            f"{np.mean(hits):.0f}" if hits else "—",
            _mean_sd(final_errors),
        ])
    columns = ["Model", "Real parameters", "Selected validation\nNRMSE", "Optimization\nseconds", "Target reached", "Mean step to target\n(reached runs)", "Final rollout\nNRMSE"]
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
            cell.set_text_props(color=COLORS[rows[row - 1][0]], weight="bold")
    ax.set_title("Comparison summary · mean ± standard deviation across seeds", fontsize=14, pad=12)
    return fig


def field_slices(results, family="random", seed=None):
    if seed is None:
        seed = results["runs"][0]["seed"]
    selected = {run["model"]: run["test"][family] for run in results["runs"] if run["seed"] == seed}
    if set(selected) != set(COLORS):
        raise ValueError("The requested seed must have both PHFNO and FNO results")
    truth = np.asarray(selected["PHFNO"]["truth"])[0, -1, 0]
    middle = truth.shape[-1] // 2
    fields = [truth[:, :, middle]]
    fields.extend(np.asarray(selected[name]["prediction"])[0, -1, 0, :, :, middle] for name in COLORS)
    errors = [np.abs(field - fields[0]) for field in fields[1:]]
    velocity_limit = max(np.max(np.abs(field)) for field in fields)
    error_limit = max(np.max(error) for error in errors)
    fig, axes = _figure(1, 5, width=17, height=3.8)
    labels = ("Reference", "PHFNO", "FNO", "PHFNO absolute error", "FNO absolute error")
    for index, (ax, field, title) in enumerate(zip(axes.flat, fields + errors, labels)):
        error = index >= 3
        image = ax.imshow(field.T, origin="lower", extent=(0, 1, 0, 1), cmap="magma" if error else "RdBu_r", vmin=0 if error else -max(velocity_limit, 1e-12), vmax=max(error_limit if error else velocity_limit, 1e-12))
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(False)
        if index == 2:
            fig.colorbar(image, ax=list(axes[0, :3]), label="Velocity component 1", shrink=0.8, pad=0.02)
        if index == 4:
            fig.colorbar(image, ax=list(axes[0, 3:]), label="Absolute error", shrink=0.8, pad=0.02)
    final_time = float(np.asarray(selected["PHFNO"]["times"])[-1])
    fig.suptitle(f"{FAMILY_NAMES.get(family, family)} · time {final_time:g} · z = 0.5 · seed {seed}", fontsize=14)
    return fig
