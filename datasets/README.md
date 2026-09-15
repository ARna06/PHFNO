# Synthetic flow datasets

These are small 3D incompressible Navier–Stokes datasets on the unit periodic
cube, with time running from zero to one. Each contains clean velocities, noisy
observations, recorded external forcing, and the viscous term. The generators
use standard PyTorch operations and FFTs.

## Generate

Run from the project root in the `research` environment:

```bash
PYTHONPATH=src python -m nsdata --kind all --output-dir datasets
```

The defaults generate four trajectories of each kind on a `16 × 16 × 16` grid
with 21 saved times. Simulation uses float64; stored fields use float32. Each
trajectory starts at a componentwise RMS velocity of 0.2. These are starter
datasets for checking training, not resolved turbulence benchmarks.

| File | Initial state and evolution |
|---|---|
| `wave.pt` | Random-phase transverse sine waves with exact forced evolution |
| `taylor_green.pt` | Randomly shifted Taylor–Green fields evolved numerically |
| `random.pt` | Random low-frequency divergence-free fields evolved numerically |

Each `.pt` file has a `.json` manifest containing the parameters and summary
diagnostics. Data tensors are ignored by Git. Existing files are protected;
pass `--overwrite` to regenerate them intentionally.

The starter datasets were generated on an NVIDIA H100 using CUDA. The complete
test suite passed with CUDA available, including checks of the numerical solver
and both model interfaces.

## Forcing and dissipation

The physical equation is

$$
\partial_t\mathbf v+(\mathbf v\cdot\nabla)\mathbf v
=-\nabla p+\nu\Delta\mathbf v+\mathbf f,
\qquad \nabla\cdot\mathbf v=0.
$$

The viscosity defaults to 0.01. There is no extra damping term. All datasets use
the same fixed, divergence-free forcing pattern:

$$
\mathbf g(x,y,z)=\sqrt{\frac65}
\begin{pmatrix}1\\0\\-2\end{pmatrix}
\sin\!\left(2\pi(2x+3y+z)\right),
\qquad \mathbf f(t,x)=u_j\mathbf g(x),\quad t_j\le t<t_{j+1}.
$$

The pattern has unit RMS when averaging over its three components and spatial
points. The scalar controls are sampled at the saved interval's left endpoint
and held constant over that entire interval:

$$
u_j=0.1\cos(2\pi t_j).
$$

Use `--forcing-amplitude`, `--forcing-frequency`, and `--viscosity` to change
these choices. Setting `--forcing-amplitude 0` gives unforced trajectories.
The numerical generator requires at least ten grid points per side for the
default forcing pattern to fit its dealiased frequency range.

Taylor–Green and random flows use a Fourier pressure projection, strict
two-thirds dealiasing, and fourth-order Runge–Kutta integration. The internal
time step is limited by the requested maximum, advection, diffusion, and forcing;
it is separate from the saved snapshot spacing. The wave generator solves its
forced linear mode exactly over each interval.

## Noise

Gaussian observation noise is added after solving the dynamics. Its standard
deviation is 1% of each trajectory's initial componentwise RMS by default:
0.002 with the default initial RMS. This scale stays fixed over time. It is
measurement noise, not stochastic forcing.

Use `--noise-level 0` for clean observations or `--project-noise` to project noise
onto the divergence-free, dealiased space. Projected noise is correlated and its
RMS is lower; the saved noise scale describes the Gaussian noise before that
projection. Independent noise is not required to be divergence-free.

## Stored tensors

Load with `torch.load(path, weights_only=True)`.

| Key | Shape or meaning |
|---|---|
| `clean`, `noisy` | `[trajectory, time, 3, x, y, z]` |
| `times` | Saved times, including zero and the final time |
| `grid` | One-dimensional grid coordinates, excluding the periodic endpoint |
| `controls` | `[trajectory, time-1, 1]`, held constant between saved times |
| `forcing_basis` | `[1, 3, x, y, z]`, the shared spatial input pattern |
| `forcing` | `[trajectory, time-1, 3, x, y, z]`, physical external acceleration |
| `viscous_term` | `[trajectory, time, 3, x, y, z]`, viscosity times the velocity Laplacian |
| `noise_sigma` | Noise standard deviation for each trajectory before optional projection |
| `diagnostics` | Kinetic energy, divergence RMS, mean velocity, dissipation rate, input power, energy rate |
| `metadata` | Generation parameters, seeds, units/conventions, device and precision |

Diagnostics are computed from the stored clean fields. Dissipation is a positive
rate; input power and energy rate are evaluated at each interval's left endpoint.
Their values describe the continuous energy balance, not a finite-difference
identity between saved snapshots. Pressure is eliminated through the Fourier
projection and is not stored.

To use the existing models, select three state channels and one control channel.
The solver's retained frequency range is fixed by its grid and two-thirds filter;
choose the model's Fourier cutoff accordingly. The initial wave and forcing need
cutoffs of at least `(2, 3, 1)` to be represented without truncation.

## Device and reproducibility

Generation uses CUDA by default and fails clearly if it is unavailable. Explicit
`--device cpu` supports small CPU checks; `--device auto` allows an automatic
CPU fallback. Local random generators leave
global RNG state unchanged. Reusing a seed and configuration reproduces a run
on the same backend; floating-point results may differ between CPU and CUDA.

```bash
PYTHONPATH=src python -m nsdata --kind random --grid-size 32 --trajectories 8 --device cuda --output-dir datasets/random32
```

Numerical checks include exact decay and forced-mode solutions, a Taylor–Green
acceleration check, an energy balance, and time-step convergence. Refine the grid
and time step before using a dataset to draw physical or comparative conclusions.
