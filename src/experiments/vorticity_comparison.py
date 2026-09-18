from dataclasses import asdict
import json
from pathlib import Path

import torch

from .comparison import (
    ComparisonConfig, build_model, copy_state, parameter_count, prepare_transitions,
    summarize_run, train_model, restore_completed_runs, restore_training_run, finite_summary,
)
from .vorticity_metrics import evaluate_vorticity_model
from .vorticity_operators import PeriodicVorticity


@torch.no_grad()
def vorticity_dataset(data, device="cuda", batch_size=32):
    operators = PeriodicVorticity(data["clean"].shape[-1], device=device, dtype=data["clean"].dtype)
    result = {key: data[key] for key in ("times", "controls", "grid") if key in data}
    for key in ("clean", "noisy", "forcing", "forcing_basis"):
        field = data[key]
        flat = field.reshape(-1, *field.shape[-4:])
        transformed = [operators.curl(batch.to(device)).cpu() for batch in flat.split(batch_size)]
        result[key] = torch.cat(transformed).reshape_as(field)
    result["metadata"] = {**data["metadata"], "state_representation": "vorticity"}
    result["sha256"] = data.get("sha256")
    return result


def validate_velocity_reference(reference, datasets, config):
    expected = asdict(config)
    # Legacy records cannot prove matched integration, accumulation and loss protocols.
    recorded = reference["config"]
    ComparisonConfig.from_record(recorded)
    for key, value in expected.items():
        # Hardware and progress output may differ without changing the experiment protocol.
        if key in ("device", "progress"):
            continue
        if key not in recorded:
            raise ValueError(f"The velocity reference is missing protocol field {key}")
        actual = tuple(recorded[key]) if isinstance(value, tuple) else recorded[key]
        if actual != value:
            raise ValueError(f"The velocity reference uses different {key}")
    if list(reference["families"]) != list(datasets):
        raise ValueError("The velocity reference uses different flow families")
    for family, data in datasets.items():
        if data.get("sha256") != reference["dataset_sha256"].get(family):
            raise ValueError(f"The velocity reference uses different data for {family}")
    expected_runs = {(model, seed) for model in ("PHFNO", "FNO") for seed in config.seeds}
    actual_runs = [(run["model"], run["seed"]) for run in reference["runs"]]
    if set(actual_runs) != expected_runs or len(actual_runs) != len(expected_runs):
        raise ValueError("The velocity reference must contain exactly one run per model and seed")


def summarize_vorticity_run(run):
    summary = summarize_run(run)
    for family, metrics in run["test"].items():
        summary["test"][family].update({
            "vorticity_rollout_nrmse": metrics["vorticity_nrmse_time"][:, 1:].mean().item(),
            "vorticity_final_nrmse": metrics["vorticity_nrmse_time"][:, -1].mean().item(),
            "vorticity_rhs_relative_error": metrics["vorticity_rhs_relative_error_time"].mean().item(),
            "vorticity_divergence_rms": metrics["vorticity_divergence_rms"][:, 1:].mean().item(),
            "vorticity_consistency_nrmse": metrics["vorticity_consistency_nrmse_time"][:, 1:].mean().item(),
            "vorticity_mean_norm": metrics["vorticity_mean_norm"][:, 1:].mean().item(),
        })
    return finite_summary(summary)


def run_vorticity_comparison(datasets, config, output_dir, velocity_reference=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if velocity_reference is not None:
        validate_velocity_reference(velocity_reference, datasets, config)
        reference = {**velocity_reference, "runs": [summarize_run(run) for run in velocity_reference["runs"]]}
        (output_dir / "velocity_reference.json").write_text(json.dumps(reference, indent=2, allow_nan=False) + "\n")
    if torch.device(config.device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this run; select the research kernel with GPU access")
    grid_shape = next(iter(datasets.values()))["clean"].shape[-3:]
    if any(data["clean"].shape[-3:] != grid_shape for data in datasets.values()):
        raise ValueError("All families must use the same spatial grid")
    transformed = {family: vorticity_dataset(data, config.device) for family, data in datasets.items()}
    training = prepare_transitions(transformed, config.train_indices, config.device)
    validation = prepare_transitions(transformed, config.validation_indices, config.device)
    del transformed
    scale = training["field"].square().mean().sqrt().item()
    if scale <= 0:
        raise ValueError("Training vorticity must have positive RMS")
    persistence = (validation["field"] - validation["clean_target"]).square().mean().sqrt().item() / scale
    hardware = torch.cuda.get_device_name(config.device) if torch.device(config.device).type == "cuda" else "CPU"
    # Record evaluation explicitly so velocity/vorticity runs have an auditable protocol.
    results = {"config": asdict(config), "evaluation_method": config.integration_method,
               "families": list(datasets), "training_rms": scale,
               "persistence_validation_nrmse": persistence, "hardware": hardware,
               "torch_version": str(torch.__version__), "runs": [],
               "state_representation": "vorticity", "validation_representation": "vorticity",
               "test_representation": "reconstructed velocity with raw vorticity diagnostics",
               "dataset_sha256": {name: data.get("sha256") for name, data in datasets.items()}}
    restore_completed_runs(results, output_dir)
    completed = {(run["model"], run["seed"]) for run in results["runs"]}
    (output_dir / "config.json").write_text(json.dumps({key: value for key, value in results.items()
                                                      if key != "runs"}, indent=2) + "\n")
    for seed_index, seed in enumerate(config.seeds):
        names = ("PHFNO", "FNO") if seed_index % 2 == 0 else ("FNO", "PHFNO")
        for name in names:
            if (name, seed) in completed:
                continue
            torch.manual_seed(seed)
            if torch.device(config.device).type == "cuda":
                torch.cuda.manual_seed_all(seed)
            model = build_model(name, config, grid_shape)
            checkpoint_path = output_dir / f"{name}_{seed}.pt"
            run = restore_training_run(model, checkpoint_path, config, name=name, seed=seed,
                                       representation="vorticity")
            if run is None:
                run = train_model(model, training, validation, config, seed, scale, persistence * 0.5)
            run.update(model=name, seed=seed, parameter_count=parameter_count(model),
                       evaluation_method=config.integration_method)
            torch.save({"state_dict": copy_state(model), "config": asdict(config), "model": name,
                        "seed": seed, "best_step": run["best_step"],
                        "best_optimizer_updates": run["best_optimizer_updates"],
                        "state_representation": "vorticity", "training_run": run},
                       checkpoint_path)
            # Match training AVF settings through both the evaluator and velocity adapter.
            run["test"] = {family: evaluate_vorticity_model(model, data, config.test_indices, config.device,
                                                           method=config.integration_method,
                                                           solver_options=config.solver_options, record_failures=True)
                           for family, data in datasets.items()}
            results["runs"].append(run)
            torch.save(results, output_dir / "results.pt")
            summary = {**results, "runs": [summarize_vorticity_run(item) for item in results["runs"]]}
            (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
            del model
    return results
