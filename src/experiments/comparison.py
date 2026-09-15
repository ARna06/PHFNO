from dataclasses import asdict, dataclass
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from time import perf_counter

import torch
from tqdm.auto import tqdm

from phfno import FNOBaseline, PHFNO
from .metrics import evaluate_model


@dataclass(frozen=True)
class ComparisonConfig:
    seeds: tuple = (7, 17, 27)
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
    device: str = "cuda"
    progress: bool = True

    def __post_init__(self):
        splits = [set(self.train_indices), set(self.validation_indices), set(self.test_indices)]
        if any(not split for split in splits):
            raise ValueError("Every trajectory split must be nonempty")
        if any(splits[i] & splits[j] for i in range(3) for j in range(i)):
            raise ValueError("Training, validation and test trajectories must be disjoint")
        if self.steps < 1 or self.batch_size < 1 or self.eval_every < 1:
            raise ValueError("steps, batch_size and eval_every must be positive")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be nonempty and distinct")


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


def predict_next(model, transitions, indices):
    field = transitions["field"][indices]
    dt = transitions["dt"][indices].reshape(-1, 1, 1, 1, 1)
    return model.coordinates.project(field) + dt * model(field, transitions["control"][indices])


@torch.no_grad()
def validation_error(model, transitions, scale, batch_size):
    model.eval()
    squared_error = 0.0
    elements = 0
    for start in range(0, len(transitions["field"]), batch_size):
        indices = slice(start, start + batch_size)
        prediction = predict_next(model, transitions, indices)
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
    warmup = predict_next(model, training, batches[0])
    warmup.square().mean().backward()
    optimizer.zero_grad(set_to_none=True)
    initial_error = validation_error(model, validation, scale, config.batch_size)
    if not torch.isfinite(torch.tensor(initial_error)):
        raise FloatingPointError("Initial validation error is nonfinite")
    history = [{"step": 0, "train_nrmse": None, "validation_nrmse": initial_error,
                "optimization_seconds": 0.0}]
    best_error, best_step = initial_error, 0
    best_state = copy_state(model)
    threshold_step = 0 if initial_error <= threshold else None
    threshold_seconds = 0.0 if initial_error <= threshold else None
    elapsed = 0.0
    loss_sum = torch.zeros((), device=config.device)
    segment_steps = 0
    model.train()
    synchronize(config.device)
    started = perf_counter()
    progress = tqdm(range(1, config.steps + 1), desc=f"{type(model).__name__} seed {seed}",
                    unit="step", disable=not config.progress, mininterval=1.0)
    for step in progress:
        indices = batches[step - 1]
        optimizer.zero_grad(set_to_none=True)
        prediction = predict_next(model, training, indices)
        loss = (prediction - training["noisy_target"][indices]).square().mean() / scale**2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip, error_if_nonfinite=True)
        optimizer.step()
        loss_sum += loss.detach()
        segment_steps += 1
        if step % config.eval_every == 0 or step == config.steps:
            synchronize(config.device)
            elapsed += perf_counter() - started
            error = validation_error(model, validation, scale, config.batch_size)
            if not torch.isfinite(torch.tensor(error)):
                raise FloatingPointError(f"Validation error is nonfinite at step {step}")
            history.append({"step": step, "train_nrmse": (loss_sum.item() / segment_steps)**0.5,
                            "validation_nrmse": error, "optimization_seconds": elapsed})
            progress.set_postfix(validation=f"{error:.4f}", refresh=False)
            if error < best_error:
                best_error, best_step = error, step
                best_state = copy_state(model)
            if threshold_step is None and error <= threshold:
                threshold_step, threshold_seconds = step, elapsed
            loss_sum.zero_()
            segment_steps = 0
            model.train()
            synchronize(config.device)
            started = perf_counter()
    model.load_state_dict(best_state)
    model.eval()
    return {"history": history, "best_step": best_step, "best_validation_nrmse": best_error,
            "train_seconds": elapsed, "threshold_nrmse": threshold,
            "threshold_step": threshold_step, "threshold_seconds": threshold_seconds}


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
    results = {"config": asdict(config), "families": list(datasets), "training_rms": scale,
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
            run.update(model=name, seed=seed, parameter_count=parameter_count(model))
            checkpoint = copy_state(model)
            torch.save({"state_dict": checkpoint, "config": asdict(config), "model": name,
                        "seed": seed, "best_step": run["best_step"]}, output_dir / f"{name}_{seed}.pt")
            run["test"] = {family: evaluate_model(model, data, config.test_indices, config.device)
                           for family, data in datasets.items()}
            results["runs"].append(run)
            torch.save(results, output_dir / "results.pt")
            summary = {**results, "runs": [summarize_run(item) for item in results["runs"]]}
            (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
            del model
    return results
