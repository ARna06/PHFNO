# Fourier port-Hamiltonian model

This implementation learns how a spatial field changes over time. It combines
an FNO from [NeuralOperator](https://github.com/neuraloperator/neuraloperator)
with a small PyTorch model that separates energy, dissipation, and external input.

Start with [the assembly notebook](ipynb/01_assemble_phfno.ipynb). It builds both
models, checks their basic properties, and runs a few training steps on synthetic
trajectories. The reusable code lives in `src/phfno/`.

## How the pieces fit together

```mermaid
flowchart TD
    field["Current state field"] --> coords["Fourier coordinates z"]
    coords --> energy["Energy network H"]
    energy --> grad["PyTorch autodiff: gradient of H"]
    coords --> damping["Damping network d"]
    coords --> grid["Reconstruct on a fixed internal grid"]
    grid --> fno["NeuralOperator FNO"]
    fno --> factors["Fourier factors a, b and input map B"]
    grad --> dynamics["Port-Hamiltonian state derivative"]
    damping --> dynamics
    factors --> dynamics
    input["External input u"] --> dynamics
    dynamics --> step["Euler step"]
    step --> output["Reconstruct the next state field"]
```

A rollout repeats this process, using each predicted state as the next input.

### Represent the field with Fourier coordinates

The model works on a uniform periodic grid. Instead of evolving every grid value
independently, it keeps a chosen set of Fourier frequencies and packs their real
and imaginary parts into a real vector, called `z`.

$$
z = E_N v, \qquad v_N = E_N^{-1}z.
$$

`encode` converts a field to these coordinates, dropping frequencies outside the
cutoff. `decode` reconstructs the retained field on a requested grid. Every grid
side must contain at least `2 * cutoff + 1` points, without duplicating the
periodic endpoint.

The normalization preserves the field's squared size:

$$
\|z\|^2 = \frac{1}{K}\sum_x\sum_c |v_{N,c}(x)|^2.
$$

Here, `K` is the number of spatial points. This matters because it lets us use
ordinary coordinate gradients without an extra grid-size correction.

### Learn the energy and the operator factors

Two small PyTorch networks read `z`. One returns the scalar energy `H`; the other
returns a damping value `d`.

The FNO reads the reconstructed field and produces output fields for `a`, `b`,
and each column of the input map `B`. These outputs are converted back to Fourier
coordinates. The FNO supplies the lifting, spectral layers, and output projection;
there are no custom kernels.

The FNO always runs on `parameter_grid`, which is fixed when the model is built.
This lets the same coordinate dynamics be evaluated on different external grids.
Changing the Fourier cutoff requires a new model because it changes the size of
`z` and the scalar networks.

### Build the state derivative

PyTorch autodiff computes the energy gradient:

$$
e(z) = \nabla_z H(z).
$$

The learned factors then define the energy-exchange and damping terms:

$$
J(z)e = \frac12\left[a(z)\bigl(b(z)^Te\bigr)
                         -b(z)\bigl(a(z)^Te\bigr)\right],
\qquad R(z)e = d(z)^2e.
$$

The full update direction is

$$
\dot z = \bigl(J(z)-R(z)\bigr)e(z) + B(z)u.
$$

The code applies `J` using two dot products rather than building a large matrix.
Its skew symmetry prevents it from adding energy. Squaring `d` makes the damping
nonnegative. The input `u` can add or remove energy through `B`.

With the corresponding output `y`, the continuous model satisfies

$$
y = B(z)^Te(z), \qquad
\frac{dH}{dt} = -d(z)^2\|e(z)\|^2 + y^Tu.
$$

This identity concerns the learned energy. It does not establish that the network
has recovered the physical energy of a particular system. The two-factor form
also limits `J` to rank at most two, which may restrict what the model can learn.

### Take time steps and train

Time evolution uses a simple Euler step:

$$
z_{j+1} = z_j + \Delta t_j\,f_\theta(z_j,u_j).
$$

Autodiff remains active through the energy gradient during training. A prediction
loss can therefore update the energy network, the damping network, and the FNO,
including across several rollout steps.

Euler is easy to inspect, but it does not guarantee decreasing energy for a finite
time step. Autodiff is the only implemented energy-gradient method; a discrete
gradient method can be added later if needed.

## The FNO comparison model

`FNOBaseline` uses an ordinary FNO to predict the state derivative. It broadcasts
the control values across the spatial grid and projects the output into the same
retained Fourier space. Both models expose the same interface:

| Call | Result |
|---|---|
| `model(field, control)` | State derivative on the input grid |
| `model.step(field, control, dt)` | Next state after one Euler step |
| `model.rollout(initial, controls, times)` | Full predicted trajectory, including the initial state |

Fields have shape `[batch, channels, *grid]`. A control has shape
`[batch, control_channels]`; rollout controls have shape
`[batch, len(times)-1, control_channels]`. Passing `None` supplies zero controls.
Both models project the initial field into the retained Fourier space.

The baseline evaluates its FNO on the external grid. Matching its width and depth
with the structured model does not match parameter counts or computation cost.
The notebook demonstrates assembly and training; it does not report a benchmark
or establish that either architecture performs better.

## Where to find the code

| File | Responsibility |
|---|---|
| [fourier.py](src/phfno/fourier.py) | Fourier coordinates and field reconstruction |
| [model.py](src/phfno/model.py) | Learned energy, damping, factors, and structured dynamics |
| [integrators.py](src/phfno/integrators.py) | Autodiff energy gradient and Euler step |
| [baseline.py](src/phfno/baseline.py) | FNO comparison model |
| [assembly notebook](ipynb/01_assemble_phfno.ipynb) | Worked example and short training loop |
| [tests/](tests/) | Small checks of the mathematics and gradient flow |

## Run it

Select the **research** kernel in the notebook. It already adds `src/` to its
import path. The local environment uses PyTorch `2.11.0+cu128` and
`neuraloperator==2.0.0`; validation covers CPU and CUDA.

```bash
conda activate research
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
```

For an editable installation in an environment with PyTorch available:

```bash
python -m pip install -e '.[notebook,test]'
```

Use float32 for the FNO. During evaluation, use `model.eval()` with
`torch.no_grad()`. The energy gradient is computed locally inside that context;
`torch.inference_mode()` disables the differentiation it needs.

The tests cover Fourier normalization, projection, multidimensional inputs,
energy identities, gradients through every network, and short rollouts. They
also check the synthetic flow generators, prescribed forcing, viscous terms,
observation noise, and saved data. All 82 tests passed in about 3 seconds with
CUDA available. The assembly notebook was also executed successfully.

## Synthetic 3D flow data

The `nsdata` package generates exact forced waves, Taylor–Green flows, and random
smooth divergence-free flows on the unit periodic cube. The datasets contain
clean and noisy velocities, scalar controls, external forcing, viscous terms,
and energy diagnostics. The [dataset guide](datasets/README.md) describes the
equations, tensor shapes, and generation options.

```bash
PYTHONPATH=src python -m nsdata --kind all --output-dir datasets
```

By default, this creates four trajectories per kind on a `16³` grid, with 21
snapshots over times zero to one. Generation uses CUDA by default; pass
`--device cpu` for small CPU checks. Use three state channels and one
control channel when connecting these data to either model.
