"""Evaluation-only extensions of the saved velocity and vorticity comparisons."""

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path

import torch

from nsdata.initial_conditions import exact_wave_trajectory
from nsdata.solver import PeriodicNavierStokes
from .comparison import ComparisonConfig, build_model
from .metrics import _power, _rms
from .vorticity_metrics import VelocityFromVorticity
from .vorticity_operators import PeriodicVorticity


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@torch.no_grad()
def extend_reference(data, indices, final_time=20.0, dt=0.025, device="cuda"):
    """Solve anew from saved clean initial states, with the original physical inputs."""
    if not math.isfinite(dt) or dt <= 0 or not math.isfinite(final_time) or final_time <= 0:
        raise ValueError("final_time and dt must be finite and positive")
    intervals = round(final_time / dt)
    if intervals < 1 or not math.isclose(intervals * dt, final_time, abs_tol=1e-12):
        raise ValueError("final_time must be an integer multiple of dt")
    indices = list(indices)
    if not indices or min(indices) < 0 or max(indices) >= len(data["clean"]):
        raise ValueError("indices must select existing held-out trajectories")
    metadata = data["metadata"]
    if metadata["domain"] != [[0.0, 1.0]] * 3 or not metadata["periodic"]:
        raise ValueError("Expected the periodic unit cube")
    if not torch.allclose(data["times"].double().diff(), torch.tensor(dt, dtype=torch.float64)):
        raise ValueError("Evaluation dt must match the trained snapshot spacing")
    times = torch.arange(intervals + 1, dtype=torch.float64) * dt
    initial = data["clean"][indices, 0].to(device=device, dtype=torch.float64)
    basis = data["forcing_basis"].to(device=device, dtype=torch.float64)
    controls = metadata["forcing_amplitude"] * torch.cos(
        2 * math.pi * metadata["forcing_frequency"] * times[:-1]
    )
    controls = controls[None, :, None].expand(len(indices), -1, -1).to(device)
    forcing = torch.einsum("btm,mcxyz->btcxyz", controls, basis)
    if metadata["kind"] == "wave":
        clean = exact_wave_trajectory(initial, times.to(device), metadata["viscosity"], forcing=forcing)
    else:
        solver = PeriodicNavierStokes(initial.shape[-1], metadata["viscosity"], device=device)
        clean = solver.solve(initial, times, max_dt=metadata["max_dt"],
                             cfl=metadata["cfl"], forcing=forcing)
    clean = clean.cpu().float()
    if not torch.isfinite(clean).all():
        raise FloatingPointError("Reference trajectory is nonfinite")
    return {
        "clean": clean, "times": times, "controls": controls.cpu().float(),
        "forcing_basis": basis.cpu().float(), "grid": data["grid"].clone(),
        "source_indices": indices,
        "metadata": {**metadata, "n_trajectories": len(indices), "n_snapshots": intervals + 1,
                     "final_time": final_time, "dt": dt, "noise_level": 0.0,
                     "noise_type": "none", "noise_seed": None, "noise_projection": None,
                     "generation_device": str(device), "generation_dtype": "torch.float64",
                     "initial_state_source": "saved held-out clean velocity at t=0"},
    }


@torch.no_grad()
def guarded_rollout(model, initial, controls, times):
    """Native autonomous steps; keep finite histories and explicit per-item failures."""
    times = torch.as_tensor(times, dtype=torch.float64, device="cpu")
    if times.ndim != 1 or times.numel() < 2 or not torch.isfinite(times).all() or not (times.diff() > 0).all():
        raise ValueError("times must be finite and strictly increasing")
    if controls.shape[:2] != (len(initial), len(times) - 1):
        raise ValueError("controls must supply every rollout interval")
    if not torch.isfinite(initial).all() or not torch.isfinite(controls).all():
        raise ValueError("initial states and controls must be finite")
    prediction = torch.full((len(initial), len(times), *initial.shape[1:]),
                            torch.nan, dtype=initial.dtype, device="cpu")
    state = model.coordinates.project(initial)
    active = torch.arange(len(initial), device=initial.device)
    prediction[:, 0] = state.cpu()
    failures = []
    was_training = model.training
    model.eval()
    try:
        for step, dt in enumerate(times.diff(), start=1):
            if not len(active):
                break
            control = controls[active, step - 1]
            reasons = {}
            try:
                candidate = model.step(state, control, dt.item())
            except RuntimeError as error:
                if not str(error).startswith("Gonzalez step did not converge"):
                    raise
                # One divergent item must not terminate the other trajectories.
                candidate = torch.full_like(state, torch.nan)
                for local in range(len(active)):
                    try:
                        candidate[local:local + 1] = model.step(
                            state[local:local + 1], control[local:local + 1], dt.item()
                        )
                    except RuntimeError as item_error:
                        if not str(item_error).startswith("Gonzalez step did not converge"):
                            raise
                        reasons[local] = str(item_error)
            finite = torch.isfinite(candidate).flatten(1).all(dim=1)
            for local in (~finite).nonzero().flatten().tolist():
                failures.append({"trajectory_index": active[local].item(), "step": step,
                                 "time": times[step].item(),
                                 "reason": reasons.get(local, "nonfinite predicted state")})
            # Preserve the attempted nonfinite snapshot; subsequent missing states are NaN.
            prediction[active.cpu(), step] = candidate.cpu()
            active, state = active[finite], candidate[finite]
    finally:
        model.train(was_training)
    return prediction, failures


@torch.no_grad()
def evaluate_long_model(model, reference, representation, device="cuda"):
    """Velocity-space diagnostics plus unprojected vorticity checks in small batches."""
    if representation not in ("velocity", "vorticity"):
        raise ValueError("representation must be velocity or vorticity")
    parameter = next((p for p in model.parameters() if not p.is_complex()), None)
    dtype = parameter.dtype if parameter is not None else reference["clean"].dtype
    initial = reference["clean"][:, 0].to(device=device, dtype=dtype)
    controls = reference["controls"].to(device=device, dtype=dtype)
    times = reference["times"].double()
    operators = PeriodicVorticity(initial.shape[-1], device=device, dtype=dtype)
    state = operators.curl(initial) if representation == "vorticity" else initial
    raw, failures = guarded_rollout(model, state, controls, times)
    prediction = torch.empty_like(raw) if representation == "vorticity" else raw
    # Reuse reconstruction, including the initial mean and prescribed mean-force integral.
    adapter = VelocityFromVorticity(model, operators, reference["forcing_basis"].to(device=device, dtype=dtype))
    diagnostic_ops = PeriodicVorticity(initial.shape[-1], device=device, dtype=torch.float64)
    solver = PeriodicNavierStokes(initial.shape[-1], reference["metadata"]["viscosity"], device=device)
    initial_rms = _rms(initial.double().unsqueeze(1)).cpu().clamp_min(1e-12)
    omega_rms = _rms(operators.curl(initial).double().unsqueeze(1)).cpu().clamp_min(1e-12)
    # Integrate the mean from the prescribed input, never from later reference states.
    mean = initial.mean(dim=(-3, -2, -1)).unsqueeze(1)
    increments = adapter._mean_force(controls).double() * times.diff().to(device)[None, :, None]
    means = mean.double() + torch.cat((torch.zeros_like(mean).double(), increments.cumsum(1)), dim=1)
    parts = {}
    for start in range(0, len(times), 16):
        section = slice(start, start + 16)
        omega = raw[:, section].to(device=device, dtype=dtype)
        if representation == "vorticity":
            velocity = operators.velocity(omega, mean=means[:, section].to(dtype))
            prediction[:, section] = velocity.cpu()
        else:
            velocity = omega
        velocity = velocity.double()
        truth = reference["clean"][:, section].to(device=device, dtype=torch.float64)
        shape = truth.shape[:2]
        values = {
            "nrmse_time": _rms(velocity - truth) / initial_rms.to(device),
            "kinetic_energy": 0.5 * _power(velocity, velocity),
            "true_kinetic_energy": 0.5 * _power(truth, truth),
            "divergence_rms": solver.divergence(velocity.flatten(0, 1)).reshape(*shape, -1).square().mean(-1).sqrt(),
            "true_divergence_rms": solver.divergence(truth.flatten(0, 1)).reshape(*shape, -1).square().mean(-1).sqrt(),
        }
        if representation == "vorticity":
            omega = omega.double()
            truth_omega = diagnostic_ops.curl(truth)
            values.update({
                "vorticity_nrmse_time": _rms(omega - truth_omega) / omega_rms.to(device),
                "vorticity_consistency_nrmse_time": _rms(omega - diagnostic_ops.curl(velocity)) / omega_rms.to(device),
                "vorticity_divergence_rms": diagnostic_ops.divergence(omega).flatten(2).square().mean(-1).sqrt(),
                "vorticity_mean_norm": omega.mean(dim=(-3, -2, -1)).norm(dim=-1),
                "enstrophy": 0.5 * _power(omega, omega),
                "true_enstrophy": 0.5 * _power(truth_omega, truth_omega),
            })
        for key, value in values.items():
            parts.setdefault(key, []).append(value.cpu())
    result = {key: torch.cat(value, dim=1) for key, value in parts.items()}
    for failure in failures:
        failure["trajectory_index"] = reference["source_indices"][failure["trajectory_index"]]
    result.update(times=times, initial_rms=initial_rms[:, 0], prediction=prediction, failures=failures,
                  trajectory_indices=reference["source_indices"])
    if representation == "vorticity":
        result["vorticity_prediction"] = raw
    return result


def load_selected_checkpoint(path, representation, summary, device="cuda"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("state_representation", "velocity") != representation:
        raise ValueError(f"Checkpoint representation mismatch: {path}")
    if json.loads(json.dumps(checkpoint["config"])) != json.loads(json.dumps(summary["config"])):
        raise ValueError(f"Checkpoint configuration mismatch: {path}")
    run = next(item for item in summary["runs"]
               if (item["model"], item["seed"]) == (checkpoint["model"], checkpoint["seed"]))
    best = min(run["history"], key=lambda row: row["validation_nrmse"])
    if checkpoint["best_step"] != run["best_step"] or checkpoint["best_step"] != best["step"]:
        raise ValueError(f"Checkpoint is not the validation-selected step: {path}")
    config = replace(ComparisonConfig(**checkpoint["config"]), device=device)
    grid_size = summary["evaluation_grid_size"]
    model = build_model(checkpoint["model"], config, (grid_size,) * 3)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, {"model": checkpoint["model"], "seed": checkpoint["seed"],
                   "representation": representation, "best_step": checkpoint["best_step"],
                   "best_validation_nrmse": best["validation_nrmse"],
                   "checkpoint_sha256": file_sha256(path)}


def _finite_number(value):
    number = float(value)
    return number if math.isfinite(number) else None


def summarize_long_results(results):
    """No survivor-only averages: any failed final trajectory invalidates that mean."""
    summary = {key: value for key, value in results.items() if key != "runs"}
    summary["runs"] = []
    for run in results["runs"]:
        record = {key: value for key, value in run.items() if key != "test"}
        record["test"] = {}
        for family, metrics in run["test"].items():
            times = metrics["times"]
            sample = {"failures": metrics["failures"], "prediction_file": metrics["prediction_file"]}
            for key in ("nrmse_time", "kinetic_energy", "true_kinetic_energy", "divergence_rms",
                        "vorticity_nrmse_time", "vorticity_consistency_nrmse_time",
                        "vorticity_divergence_rms", "vorticity_mean_norm", "enstrophy"):
                if key in metrics:
                    sample[key] = {f"t={time:g}": _finite_number(metrics[key][:, (times - time).abs().argmin()].mean())
                                   for time in (1, 5, 10, 20)}
            record["test"][family] = sample
        summary["runs"].append(record)
    return summary


def run_long_comparison(dataset_dir, checkpoint_dirs, output_dir, device="cuda"):
    """Run/cache the fixed t=20 protocol; never write to input artifact directories."""
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This comparison requires CUDA in the research environment")
    device = str(torch.device("cuda", torch.device(device).index
                              if torch.device(device).index is not None else torch.cuda.current_device()))
    if set(checkpoint_dirs) != {"velocity", "vorticity"}:
        raise ValueError("Supply both velocity and vorticity checkpoint directories")
    dataset_dir, output_dir = Path(dataset_dir).resolve(), Path(output_dir).resolve()
    checkpoint_dirs = {key: Path(path).resolve() for key, path in checkpoint_dirs.items()}
    for source in (dataset_dir, *checkpoint_dirs.values()):
        if output_dir == source or source in output_dir.parents or output_dir in source.parents:
            raise ValueError("Evaluation output must be separate from source directories")
    summaries = {key: json.loads((path / "summary.json").read_text()) for key, path in checkpoint_dirs.items()}
    baseline = summaries["velocity"]
    families, indices = baseline["families"], baseline["config"]["test_indices"]
    hashes = {family: file_sha256(dataset_dir / f"{family}.pt") for family in families}
    checkpoints = {}
    for representation, summary in summaries.items():
        if summary["families"] != families or summary["dataset_sha256"] != hashes:
            raise ValueError("Both saved comparisons must use these exact datasets and families")
        for key in ("seeds", "train_indices", "validation_indices", "test_indices"):
            if summary["config"][key] != baseline["config"][key]:
                raise ValueError(f"Saved comparison split/seed mismatch: {key}")
        for name in ("PHFNO", "FNO"):
            for seed in summary["config"]["seeds"]:
                path = checkpoint_dirs[representation] / f"{name}_{seed}.pt"
                checkpoints[f"{representation}/{name}_{seed}"] = file_sha256(path)
    manifest = {
        "format_version": 1, "final_time": 20.0, "dt": 0.025, "n_snapshots": 801,
        "dataset_sha256": hashes, "checkpoint_sha256": checkpoints,
        "summary_sha256": {key: file_sha256(path / "summary.json") for key, path in checkpoint_dirs.items()},
        "test_indices": indices, "seeds": baseline["config"]["seeds"], "families": families,
        "device": str(device), "hardware": torch.cuda.get_device_name(device),
        "torch_version": str(torch.__version__), "reference_dtype": "torch.float64",
        "model_dtype": "torch.float32", "diagnostic_dtype": "torch.float64",
        "integrators": {"PHFNO": "Gonzalez (checkpoint default)", "FNO": "Euler (checkpoint default)"},
        "protocol": "Clean held-out t=0; autonomous recurrence with prescribed left-endpoint forcing; no resets",
        "error_normalization": "Per-trajectory initial componentwise velocity RMS",
        "failure_policy": "Record nonfinite states or Gonzalez nonconvergence per trajectory; no restart; NaN tail",
        "aggregation": "Equal weight for trajectories and seeds; no omission of failed trajectories",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "config.json"
    if not manifest_path.exists() and any(output_dir.iterdir()):
        raise ValueError("Existing output has no provenance manifest; choose a new output directory")
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Existing output has different provenance; choose a new output directory")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if (output_dir / "results.pt").exists():
        return torch.load(output_dir / "results.pt", map_location="cpu", weights_only=True)
    results = {**manifest, "references": {}, "runs": []}
    for family in families:
        path = output_dir / "references" / f"{family}.pt"
        if not path.exists():
            print(f"Generating clean {family} reference: 801 snapshots, t=0–20", flush=True)
            source = torch.load(dataset_dir / f"{family}.pt", map_location="cpu", weights_only=True, mmap=True)
            reference = extend_reference(source, indices, device=device)
            # A fresh solve must reproduce the original overlapping clean horizon.
            overlap = source["times"].numel()
            discrepancy = _rms((reference["clean"][:, :overlap] - source["clean"][indices]).double())
            discrepancy /= _rms(source["clean"][indices, :1].double()).clamp_min(1e-12)
            reference["metadata"]["overlap_max_nrmse"] = discrepancy.max().item()
            if discrepancy.max() > 1e-5:
                raise ValueError("Extended reference disagrees with the original clean t=0–1 solution")
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(reference, path)
            del source, reference
        reference = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        results["references"][family] = {"path": str(path.relative_to(output_dir)), "metadata": reference["metadata"]}
    for representation, summary in summaries.items():
        summary["evaluation_grid_size"] = reference["clean"].shape[-1]
        for name in ("PHFNO", "FNO"):
            for seed in summary["config"]["seeds"]:
                model, run = load_selected_checkpoint(checkpoint_dirs[representation] / f"{name}_{seed}.pt",
                                                      representation, summary, device)
                run["test"] = {}
                for family in families:
                    metric_path = output_dir / representation / f"{name}_{seed}_{family}_metrics.pt"
                    if metric_path.exists():
                        metrics = torch.load(metric_path, map_location="cpu", weights_only=True)
                    else:
                        print(f"Rolling out {representation} {name}, seed {seed}, {family}", flush=True)
                        reference = torch.load(output_dir / "references" / f"{family}.pt",
                                               map_location="cpu", weights_only=True, mmap=True)
                        metrics = evaluate_long_model(model, reference, representation, device)
                        prediction_path = metric_path.with_name(f"{name}_{seed}_{family}_prediction.pt")
                        predictions = {key: metrics.pop(key) for key in ("prediction", "vorticity_prediction") if key in metrics}
                        metric_path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(predictions, prediction_path)
                        metrics["prediction_file"] = str(prediction_path.relative_to(output_dir))
                        torch.save(metrics, metric_path)
                        del predictions
                    run["test"][family] = metrics
                results["runs"].append(run)
                del model
    torch.save(results, output_dir / "results.pt")
    (output_dir / "summary.json").write_text(json.dumps(summarize_long_results(results), indent=2, allow_nan=False) + "\n")
    return results
