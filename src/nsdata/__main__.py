import argparse
from pathlib import Path

import torch

from .dataset import generate_dataset, resolve_device, save_dataset


def main():
    parser = argparse.ArgumentParser(description="Generate periodic 3D Navier-Stokes trajectories")
    parser.add_argument("--kind", choices=("wave", "taylor_green", "random", "all"), default="all")
    parser.add_argument("--output-dir", type=Path, default=Path("datasets"))
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--trajectories", type=int, default=4)
    parser.add_argument("--snapshots", type=int, default=21)
    parser.add_argument("--final-time", type=float, default=1.0)
    parser.add_argument("--viscosity", type=float, default=0.01)
    parser.add_argument("--initial-rms", type=float, default=0.2)
    parser.add_argument("--forcing-amplitude", type=float, default=0.1)
    parser.add_argument("--forcing-frequency", type=float, default=1.0)
    parser.add_argument("--cutoff", type=int, default=2)
    parser.add_argument("--max-dt", type=float, default=0.005)
    parser.add_argument("--noise-level", type=float, default=0.01)
    parser.add_argument("--project-noise", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("float32", "float64"), default="float64")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    kinds = ("wave", "taylor_green", "random") if args.kind == "all" else (args.kind,)
    paths = [args.output_dir / f"{kind}.pt" for kind in kinds]
    if not args.overwrite:
        for path in paths:
            if path.exists() or path.with_suffix(".json").exists():
                parser.error(f"{path} already exists; choose another directory or pass --overwrite")
    print(f"Generating on {device}", flush=True)
    for kind, path in zip(kinds, paths):
        print(f"Generating {kind}", flush=True)
        dataset = generate_dataset(
            kind=kind,
            grid_size=args.grid_size,
            n_trajectories=args.trajectories,
            n_snapshots=args.snapshots,
            final_time=args.final_time,
            viscosity=args.viscosity,
            initial_rms=args.initial_rms,
            forcing_amplitude=args.forcing_amplitude,
            forcing_frequency=args.forcing_frequency,
            cutoff=args.cutoff,
            max_dt=args.max_dt,
            noise_level=args.noise_level,
            project_noise=args.project_noise,
            seed=args.seed,
            device=device,
            dtype=getattr(torch, args.precision),
        )
        save_dataset(dataset, path, overwrite=args.overwrite)
        print(f"Saved {path}: {tuple(dataset['clean'].shape)}", flush=True)


if __name__ == "__main__":
    main()
