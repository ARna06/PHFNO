from dataclasses import asdict, dataclass
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import torch
from tqdm.auto import tqdm

from phfno import FNOBaseline, PHFNO
from .losses import h1_loss
from .metrics import evaluate_model


@dataclass(frozen=True)
class ComparisonConfig:
    seeds: tuple = (8, 18, 28)
    train_indices: tuple = (0, 1)
    validation_indices: tuple = (2,)
    test_indices: tuple = (3,)
    cutoff: tuple = (5, 5, 5)
    hidden_channels: int = 16
    n_layers: int = 2
    mlp_width: int = 64
    steps: int = 600
    batch_size: int = 12
    learning_rate: float = 1e-3
    eval_every: int = 25
    gradient_clip: float = 1.0
    integration_method: str = "avf"
    theta_update_start: int = 20
    theta_update_interval: int = 5
    device: str = "cpu"
    progress: bool = True
    # Append versioned fields to preserve the existing positional constructor order.
    protocol_version: int = 2
    loss_definition: str = "fourier_h1"
    # Share explicit solver settings across training, validation and held-out rollouts.
    avf_max_iterations: int = 50
    avf_rtol: float = 1e-6
    avf_atol: float = 1e-8
    avf_quadrature_points: int = 4

    def __post_init__(self):
        # Reject incompatible saved protocols instead of silently relabelling them.
        if self.protocol_version != 2 or self.loss_definition != "fourier_h1":
            raise ValueError("Unsupported comparison protocol_version or loss_definition")
        splits = [set(self.train_indices), set(self.validation_indices), set(self.test_indices)]
        if any(not split for split in splits):
            raise ValueError("Every trajectory split must be nonempty")
        if any(splits[i] & splits[j] for i in range(3) for j in range(i)):
            raise ValueError("Training, validation and test trajectories must be disjoint")
        if self.steps < 1 or self.batch_size < 1 or self.eval_every < 1:
            raise ValueError("steps, batch_size and eval_every must be positive")
        if self.theta_update_start < 0 or self.theta_update_interval < 1:
            raise ValueError("theta update schedule values must be nonnegative and positive")
        if self.integration_method != "avf":
            raise ValueError("comparison integration_method must be 'avf'")
        # Only the implemented Gauss-Legendre rules can define a valid solver protocol.
        if (not isinstance(self.avf_max_iterations, int)
                or isinstance(self.avf_max_iterations, bool) or self.avf_max_iterations < 1):
            raise ValueError("avf_max_iterations must be a positive integer")
        if self.avf_quadrature_points not in (1, 2, 4):
            raise ValueError("avf_quadrature_points must be 1, 2, or 4")
        if any(not math.isfinite(value) or value < 0 for value in (self.avf_rtol, self.avf_atol)):
            raise ValueError("AVF tolerances must be finite and nonnegative")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be nonempty and distinct")

    @property
    def solver_options(self):
        # Translate recorded experiment settings to the public model solver interface.
        return {"max_iterations": self.avf_max_iterations, "rtol": self.avf_rtol,
                "atol": self.avf_atol, "quadrature_points": self.avf_quadrature_points}

    @classmethod
    def from_record(cls, record):
        # Missing metadata cannot establish how a legacy checkpoint was trained.
        required = {"protocol_version", "loss_definition", "integration_method",
                    "theta_update_start", "theta_update_interval", "avf_max_iterations",
                    "avf_rtol", "avf_atol", "avf_quadrature_points"}
        missing = sorted(required - record.keys())
        if missing:
            raise ValueError(f"Legacy or incomplete comparison protocol; missing {', '.join(missing)}")
        # JSON stores tuple fields as lists; normalize them for cross-format comparisons.
        values = dict(record)
        for key in ("seeds", "train_indices", "validation_indices", "test_indices", "cutoff"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)


def load_datasets(directory, families=("wave", "taylor_green", "random")):
    datasets = {}
    for family in families:
        path = Path(directory) / f"{family}.pt"
        data = torch.load(path, map_location="cpu", weights_only=True)
        data["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        datasets[family] = data
    return datasets


def prepare_transitions(datasets, indices, device):
    fields, noisy_targets, clean_targets, controls, intervals = [], [], [], [], []
    for data in datasets.values():
        indices_tensor = torch.tensor(indices, dtype=torch.long)
        if min(indices) < 0 or max(indices) >= data["clean"].shape[0]:
            raise ValueError("Trajectory index is outside the dataset")
        noisy = data["noisy"][indices_tensor].float()
        clean = data["clean"][indices_tensor].float()
        times = data["times"].float()
        if noisy.shape != clean.shape or clean.shape[1] != times.numel():
            raise ValueError("Snapshot shapes and times must agree")
        if not (times.diff() > 0).all():
            raise ValueError("Snapshot times must be strictly increasing")
        fields.append(noisy[:, :-1].flatten(0, 1))
        noisy_targets.append(noisy[:, 1:].flatten(0, 1))
        clean_targets.append(clean[:, 1:].flatten(0, 1))
        controls.append(data["controls"][indices_tensor].float().flatten(0, 1))
        intervals.append(times.diff().repeat(len(indices)))
    values = (fields, noisy_targets, clean_targets, controls, intervals)
    names = ("field", "noisy_target", "clean_target", "control", "dt")
    return {name: torch.cat(parts).to(device) for name, parts in zip(names, values)}


def build_model(name, config, grid_shape):
    options = dict(cutoff=config.cutoff, state_channels=3, control_channels=1,
                   hidden_channels=config.hidden_channels, n_layers=config.n_layers)
    if name == "PHFNO":
        model = PHFNO(**options, mlp_width=config.mlp_width, parameter_grid=grid_shape)
    elif name == "FNO":
        model = FNOBaseline(**options)
    else:
        raise ValueError(f"Unknown model: {name}")
    return model.to(config.device)


def parameter_count(model):
    return sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters() if p.requires_grad)


def copy_state(model):
    return {key: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else deepcopy(value)
            for key, value in model.state_dict().items() if key != "_metadata"}


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def predict_next(model, transitions, indices, method="avf", solver_options=None):
    field = transitions["field"][indices]
    # Keep standalone callers compatible while propagating recorded solver settings.
    options = {} if solver_options is None else {"solver_options": solver_options}
    return model.step(
        field, transitions["control"][indices], transitions["dt"][indices], method=method,
        **options,
    )


@torch.no_grad()
def validation_error(model, transitions, scale, batch_size, method="avf", solver_options=None):
    model.eval()
    squared_error = 0.0
    elements = 0
    for start in range(0, len(transitions["field"]), batch_size):
        indices = slice(start, start + batch_size)
        # Validation scores the same discrete dynamics used by the training objective.
        prediction = predict_next(model, transitions, indices, method=method,
                                  solver_options=solver_options)
        difference = prediction - transitions["clean_target"][indices]
        squared_error += difference.square().sum().item()
        elements += difference.numel()
    return (squared_error / elements) ** 0.5 / scale


def train_model(model, training, validation, config, seed, scale, threshold):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    generator = torch.Generator().manual_seed(seed + 1000)
    batches = torch.randint(len(training["field"]), (config.steps, config.batch_size),
                            generator=generator).to(config.device)
    model.train()
    # Initialize lazy model components before timing; this backward pass makes no update.
    warmup = predict_next(model, training, batches[0], method=config.integration_method,
                          solver_options=config.solver_options)
    warmup.square().mean().backward()
    optimizer.zero_grad(set_to_none=True)
    initial_error = validation_error(
        model, validation, scale, config.batch_size, method=config.integration_method,
        solver_options=config.solver_options,
    )
    if not torch.isfinite(torch.tensor(initial_error)):
        raise FloatingPointError("Initial validation error is nonfinite")
    # Retain minibatch step for compatibility and record actual Adam updates separately.
    optimizer_updates = 0
    history = [{"step": 0, "optimizer_updates": 0, "train_nrmse": None, "train_h1_loss": None, "validation_nrmse": initial_error,
                "optimization_seconds": 0.0}]
    best_error, best_step = initial_error, 0
    best_optimizer_updates = 0
    best_state = copy_state(model)
    threshold_step = 0 if initial_error <= threshold else None
    threshold_optimizer_updates = 0 if initial_error <= threshold else None
    threshold_seconds = 0.0 if initial_error <= threshold else None
    elapsed = 0.0
    loss_sum = torch.zeros((), device=config.device)
    squared_error_sum = torch.zeros((), device=config.device)
    segment_steps = 0
    model.train()
    synchronize(config.device)
    started = perf_counter()
    progress = tqdm(range(1, config.steps + 1), desc=f"{type(model).__name__} seed {seed}",
                    unit="step", disable=not config.progress, mininterval=1.0)
    for step in progress:
        indices = batches[step - 1]
        if step <= config.theta_update_start:
            group_start, update_interval = step, 1
        else:
            group_start = (
                config.theta_update_start + 1
                + (step - config.theta_update_start - 1) // config.theta_update_interval
                * config.theta_update_interval
            )
            update_interval = min(
                config.theta_update_interval, config.steps - group_start + 1
            )
        if step == group_start:
            optimizer.zero_grad(set_to_none=True)
        prediction = predict_next(model, training, indices, method=config.integration_method,
                                  solver_options=config.solver_options)
        target = training["noisy_target"][indices]
        # H1 penalizes both field and spatial-gradient errors; average within each group.
        loss = h1_loss(prediction, target) / scale**2
        (loss / update_interval).backward()
        # End groups relative to their start, including offset and shortened final groups.
        if step == group_start + update_interval - 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip, error_if_nonfinite=True)
            optimizer.step()
            optimizer_updates += 1
        loss_sum += loss.detach()
        squared_error_sum += (prediction.detach() - target).square().mean() / scale**2
        segment_steps += 1
        if step % config.eval_every == 0 or step == config.steps:
            synchronize(config.device)
            elapsed += perf_counter() - started
            error = validation_error(
                model, validation, scale, config.batch_size,
                method=config.integration_method,
                solver_options=config.solver_options,
            )
            if not torch.isfinite(torch.tensor(error)):
                raise FloatingPointError(f"Validation error is nonfinite at step {step}")
            history.append({"step": step, "optimizer_updates": optimizer_updates,
                            "train_nrmse": (squared_error_sum.item() / segment_steps)**0.5,
                            "train_h1_loss": loss_sum.item() / segment_steps,
                            "validation_nrmse": error, "optimization_seconds": elapsed})
            progress.set_postfix(validation=f"{error:.4f}", refresh=False)
            if error < best_error:
                best_error, best_step = error, step
                best_optimizer_updates = optimizer_updates
                best_state = copy_state(model)
            if threshold_step is None and error <= threshold:
                threshold_step, threshold_seconds = step, elapsed
                threshold_optimizer_updates = optimizer_updates
            loss_sum.zero_()
            squared_error_sum.zero_()
            segment_steps = 0
            model.train()
            synchronize(config.device)
            started = perf_counter()
    # Test only the checkpoint selected by clean held-out one-step validation.
    model.load_state_dict(best_state)
    model.eval()
    return {"history": history, "best_step": best_step, "best_validation_nrmse": best_error,
            "train_seconds": elapsed, "threshold_nrmse": threshold,
            "threshold_step": threshold_step, "threshold_seconds": threshold_seconds,
            "optimizer_updates": optimizer_updates, "best_optimizer_updates": best_optimizer_updates,
            "threshold_optimizer_updates": threshold_optimizer_updates}


def summarize_run(run):
    summary = {key: value for key, value in run.items() if key != "test"}
    summary["test"] = {}
    for family, metrics in run["test"].items():
        summary["test"][family] = {
            "rollout_nrmse": metrics["nrmse_time"][:, 1:].mean().item(),
            "final_nrmse": metrics["nrmse_time"][:, -1].mean().item(),
            "rhs_relative_error": metrics["rhs_relative_error_time"].mean().item(),
            "divergence_rms": metrics["divergence_rms"][:, 1:].mean().item(),
            "normalized_power_error": (metrics["power_residual_time"].abs()
                                       / metrics["reference_power_scale"][:, None]).mean().item(),
            "force_nrmse": metrics["force_nrmse_time"].mean().item(),
        }
    return summary


def run_comparison(datasets, config, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if torch.device(config.device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this run; select the research kernel with GPU access")
    grid_shape = next(iter(datasets.values()))["clean"].shape[-3:]
    if any(data["clean"].shape[-3:] != grid_shape for data in datasets.values()):
        raise ValueError("All families must use the same spatial grid")
    training = prepare_transitions(datasets, config.train_indices, config.device)
    validation = prepare_transitions(datasets, config.validation_indices, config.device)
    scale = training["field"].square().mean().sqrt().item()
    if scale <= 0:
        raise ValueError("Training velocities must have positive RMS")
    persistence = (validation["field"] - validation["clean_target"]).square().mean().sqrt().item() / scale
    hardware = torch.cuda.get_device_name(config.device) if torch.device(config.device).type == "cuda" else "CPU"
    # Persist the actual evaluation method alongside the versioned training protocol.
    results = {"config": asdict(config), "evaluation_method": config.integration_method,
               "families": list(datasets), "training_rms": scale,
               "persistence_validation_nrmse": persistence, "hardware": hardware,
               "torch_version": str(torch.__version__), "runs": [],
               "dataset_sha256": {name: data.get("sha256") for name, data in datasets.items()}}
    (output_dir / "config.json").write_text(json.dumps({key: value for key, value in results.items()
                                                      if key != "runs"}, indent=2) + "\n")
    for seed_index, seed in enumerate(config.seeds):
        names = ("PHFNO", "FNO") if seed_index % 2 == 0 else ("FNO", "PHFNO")
        for name in names:
            torch.manual_seed(seed)
            if torch.device(config.device).type == "cuda":
                torch.cuda.manual_seed_all(seed)
            model = build_model(name, config, grid_shape)
            run = train_model(model, training, validation, config, seed, scale, persistence * 0.5)
            run.update(model=name, seed=seed, parameter_count=parameter_count(model),
                       evaluation_method=config.integration_method)
            checkpoint = copy_state(model)
            torch.save({"state_dict": checkpoint, "config": asdict(config), "model": name,
                        "seed": seed, "best_step": run["best_step"],
                        "best_optimizer_updates": run["best_optimizer_updates"]}, output_dir / f"{name}_{seed}.pt")
            # Never let model-specific public defaults change held-out discrete dynamics.
            run["test"] = {family: evaluate_model(model, data, config.test_indices, config.device,
                                                 method=config.integration_method,
                                                 solver_options=config.solver_options)
                           for family, data in datasets.items()}
            results["runs"].append(run)
            torch.save(results, output_dir / "results.pt")
            summary = {**results, "runs": [summarize_run(item) for item in results["runs"]]}
            (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
            del model
    return results
