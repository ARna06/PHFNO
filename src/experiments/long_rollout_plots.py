"""Full-horizon comparison plots that keep failed rollouts visible."""

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterSciNotation, MaxNLocator, ScalarFormatter, SymmetricalLogLocator

from .plots import COLORS, FAMILY_NAMES, _figure


REPRESENTATIONS = ("velocity", "vorticity")
SEED_STYLES = ("-", "--", ":", "-.")


def _runs(results, representation):
    return sorted(
        (run for run in results["runs"] if run["representation"] == representation),
        key=lambda run: (list(COLORS).index(run["model"]), run["seed"]),
    )


def _style(results, seed):
    seeds = sorted({run["seed"] for run in results["runs"]})
    return SEED_STYLES[seeds.index(seed) % len(SEED_STYLES)]


def _failure_times(metrics):
    """Include recorded solver failures and nonfinite metric values."""
    times = np.asarray(metrics["times"])
    failures = {float(item["time"]) for item in metrics.get("failures", [])}
    for key, value in metrics.items():
        if key.startswith("true_") or key in ("times", "failures", "trajectory_indices"):
            continue
        values = np.asarray(value)
        if values.ndim == 2 and values.shape[-1] == len(times):
            for failed in ~np.isfinite(values):
                if failed.any():
                    failures.add(float(times[np.flatnonzero(failed)[0]]))
    return sorted(failures)


def failure_fraction(metrics):
    """Cumulative failed fraction; logged nonconvergence counts as a failure."""
    times = np.asarray(metrics["times"])
    shape = np.asarray(metrics["nrmse_time"]).shape
    failed = np.zeros(shape, dtype=bool)
    for key, value in metrics.items():
        if key.startswith("true_") or key in ("times", "failures", "trajectory_indices"):
            continue
        values = np.asarray(value)
        if values.shape == shape:
            failed |= ~np.isfinite(values)
    indices = metrics.get("trajectory_indices")
    if indices is not None:
        positions = {int(index): row for row, index in enumerate(indices)}
        for item in metrics.get("failures", []):
            failed[positions[int(item["trajectory_index"])], times >= item["time"]] = True
        return np.maximum.accumulate(failed, axis=1).mean(axis=0)

    # Older metric dictionaries may omit original trajectory IDs. Failure
    # records and NaN tails describe the same failures, so never add counts.
    first_failure = {}
    for item in metrics.get("failures", []):
        index = int(item["trajectory_index"])
        first_failure[index] = min(first_failure.get(index, np.inf), item["time"])
    recorded = np.array([sum(time >= start for start in first_failure.values()) for time in times])
    observed = np.maximum.accumulate(failed, axis=1).sum(axis=0)
    return np.maximum(recorded, observed) / shape[0]


def _series(ax, results, run, metrics, key, marker_height):
    times = np.asarray(metrics["times"])
    values = np.asarray(metrics[key], dtype=np.float64)
    color = COLORS[run["model"]]
    style = _style(results, run["seed"])
    # Individual finite paths remain visible even after another sample fails.
    ax.plot(times, values.T, color=color, linestyle=style, alpha=0.15, linewidth=0.7)
    # The mean and band use the fixed cohort: NaNs stop them, never select a
    # shrinking, apparently better-performing subset of surviving samples.
    ax.plot(times, values.mean(axis=0), color=color, linestyle=style,
            linewidth=1.6, label=f"{run['model']} · seed {run['seed']}")
    ax.fill_between(times, values.min(axis=0), values.max(axis=0), color=color, alpha=0.06)
    failures = _failure_times(metrics)
    if failures:
        ax.plot(failures, [marker_height] * len(failures), "x", color=color,
                markersize=6, transform=ax.get_xaxis_transform(), clip_on=False)
    ax.set_xlim(times[0], times[-1])


def _legend(ax, failure_markers=True):
    handles, labels = ax.get_legend_handles_labels()
    if failure_markers:
        handles.append(Line2D([], [], marker="x", linestyle="none", color="#333333"))
        labels.append("Numerical / solver failure")
    ax.legend(handles, labels, frameon=False, fontsize=8, loc="upper left")


def _scale(ax, linthresh):
    """Readable ticks for both small errors and many-decade divergence."""
    ax.set_yscale("symlog", linthresh=linthresh)
    ax.set_ylim(bottom=0)
    if ax.get_ylim()[1] <= linthresh:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-3, 3))
    else:
        locator = SymmetricalLogLocator(base=10, linthresh=linthresh)
        locator.set_params(numticks=7)
        ax.yaxis.set_major_locator(locator)
        formatter = LogFormatterSciNotation(base=10, linthresh=linthresh)
    ax.yaxis.set_major_formatter(formatter)


def _representation_curves(results, key, ylabel, title, linthresh, reference_key=None):
    families = results["families"]
    fig, axes = _figure(2, len(families), height=8)
    for row, representation in enumerate(REPRESENTATIONS):
        runs = _runs(results, representation)
        for column, family in enumerate(families):
            ax = axes[row, column]
            for index, run in enumerate(runs):
                metrics = run["test"][family]
                _series(ax, results, run, metrics, key, 0.98 - 0.025 * index)
            if reference_key is not None:
                reference = runs[0]["test"][family]
                values = np.asarray(reference[reference_key])
                times = np.asarray(reference["times"])
                ax.plot(times, values.T, color="#333333", alpha=0.15, linewidth=0.7)
                ax.plot(times, values.mean(axis=0), color="#333333", linestyle="--",
                        linewidth=1.8, label="Navier–Stokes reference")
            ax.set_title(f"{FAMILY_NAMES.get(family, family)} · {representation}-trained", fontsize=11)
            ax.set_xlabel("Time")
            ax.set_ylabel(ylabel)
            _scale(ax, linthresh)
        _legend(axes[row, 0])
    fig.suptitle(title + " · seed means and individual trajectories", fontsize=14)
    return fig


def rollout_curves(results):
    return _representation_curves(
        results, "nrmse_time", "Velocity RMSE / initial velocity RMS",
        "Autonomous velocity error, time 0–20", linthresh=1,
    )


def kinetic_energy_curves(results):
    return _representation_curves(
        results, "kinetic_energy", "Physical kinetic energy",
        "Kinetic energy, time 0–20", linthresh=1e-6, reference_key="true_kinetic_energy",
    )


def vorticity_constraints(results):
    families = results["families"]
    quantities = (
        ("vorticity_nrmse_time", "Vorticity RMSE / initial vorticity RMS", 1),
        ("vorticity_consistency_nrmse_time", "Curl reconstruction mismatch / initial RMS", 1e-4),
        ("vorticity_divergence_rms", "RMS raw-vorticity divergence", 1e-6),
        ("vorticity_mean_norm", "Norm of spatial mean raw vorticity", 1e-6),
    )
    fig, axes = _figure(len(quantities), len(families), height=13)
    runs = _runs(results, "vorticity")
    for column, family in enumerate(families):
        for row, (key, ylabel, linthresh) in enumerate(quantities):
            ax = axes[row, column]
            for index, run in enumerate(runs):
                _series(ax, results, run, run["test"][family], key, 0.98 - 0.025 * index)
            ax.set_xlabel("Time")
            ax.set_ylabel(ylabel, fontsize=10)
            _scale(ax, linthresh)
        axes[0, column].set_title(FAMILY_NAMES.get(family, family))
    _legend(axes[0, 0])
    fig.suptitle("Vorticity-trained models · raw-vorticity checks, time 0–20", fontsize=14)
    return fig


def failure_curves(results):
    families = results["families"]
    fig, axes = _figure(2, len(families), height=7)
    for row, representation in enumerate(REPRESENTATIONS):
        for column, family in enumerate(families):
            ax = axes[row, column]
            for run in _runs(results, representation):
                metrics = run["test"][family]
                times = np.asarray(metrics["times"])
                ax.step(times, failure_fraction(metrics), where="post", color=COLORS[run["model"]],
                        linestyle=_style(results, run["seed"]), linewidth=1.8,
                        label=f"{run['model']} · seed {run['seed']}")
                ax.set_xlim(times[0], times[-1])
            ax.set_ylim(-0.02, 1.02)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
            ax.set_title(f"{FAMILY_NAMES.get(family, family)} · {representation}-trained", fontsize=11)
            ax.set_xlabel("Time")
            ax.set_ylabel("Cumulative fraction of failed rollouts")
        _legend(axes[row, 0], failure_markers=False)
    fig.suptitle("Numerical failures and integrator nonconvergence, time 0–20", fontsize=14)
    return fig
