**Codebase audit — 17 September 2026**

## AVF comparison update — 18 September 2026

The two time-0–1 comparison notebooks have now completed in the research
environment on NVIDIA H100 NVL GPUs: 25 executed cells, 15 inline plots, and no
cell errors. Twelve validation-selected checkpoints and two result files were
generated. Short CUDA replays reproduced their saved predictions and failure
locations. The full test suite passes: **268 passed**; the long-rollout notebook's
research kernel metadata mismatch is fixed.

The velocity-trained mean final NRMSE is **0.174 for PHFNO** and **0.099 for FNO**,
averaged equally over flow families and seeds. Vorticity-trained PHFNO has two
AVF solver failures on random-flow trajectory 31: seed 8 at time 0.375, and seed
18 at time 0.025. FNO has no recorded rollout failures in these runs. A solver
failure does not by itself prove physical instability. Raw-vorticity constraint
defects also remain; inverse-curl reconstruction's low velocity divergence is
not evidence that the learned raw vorticity is consistent.

These experiments do not show a general PHFNO advantage or establish time-20
accuracy for the new checkpoints. The earlier audit and remediation record below
are retained as history. Current artifacts are under
`results/local_smoke_test_seeds_8_18_28/` and `results/phfno_fno_vorticity_avf/`.
The [README reproduction guide](README.md#reproduce-the-notebook-results) explains
dataset generation, training, checkpoint inference, and the distinction between
saved results and a fresh run. Binary datasets/checkpoints remain local under
the repository's existing ignore rules.

## Earlier remediation and original findings

<!-- Remediation update: retain the original audit below as a historical record. -->
**Remediation completed.** The findings below describe the code before these
changes; their original line references and test results are historical.

- Training, validation, test evaluation, and long rollouts now pass the same
  recorded AVF method and solver settings. Public model defaults are preserved.
- AVF checks the equation at the returned endpoint, rejects nonfinite values,
  and uses a norm over the whole state. FNO and PhFNO both integrate Fourier
  coordinates so their convergence tolerances have the same normalization.
- AVF training solves the implicit adjoint separately from the state, fixing
  equilibrium sensitivities. The backward solve is normalized against arbitrary
  loss scaling. It supports first-order training gradients, including the
  energy Hessian used inside PhFNO; higher-order derivatives of the complete AVF
  step are explicitly unsupported.
- Accumulation uses group-relative boundaries, and histories record minibatch
  iterations and actual optimizer updates separately.
- Versioned protocol checks reject ambiguous legacy records; Python 3.10 hashing
  works; plots reject incomplete seed sets; notebook setup and mathematical
  documentation are corrected. Comments explain the changes and main model steps.

The complete CPU suite now reports **252 passed, 5 skipped** (Python 3.10.20,
PyTorch 2.14.0, NeuralOperator 2.0.0). CUDA was unavailable. Saved experiments
were not rewritten or retrained; their metrics need regeneration under the new
protocol. The documented limitations remain: fixed-point solves can fail for
noncontractive dynamics, and generic AVF does not guarantee a discrete energy
law for the model's state-dependent structure. Solver settings are configurable
through `solver_options` and the corresponding `ComparisonConfig.avf_*` fields.

**Original audit follows.**

The current comparison trains and validates both models with AVF, but evaluates their trajectories with different integrators. The AVF implementation also has reproducible convergence and differentiation problems. Existing AVF-labelled rollout metrics should be regenerated after fixing method propagation.

Scope: all source modules in `src/phfno`, `src/experiments`, and `src/nsdata`; notebook code; test coverage and execution; documentation; saved experiment configurations; and selected saved checkpoints. No implementation, tests, datasets, or experiment results were changed. This report is the only intentional repository addition.

**Actual integration paths**

| Entry point | PhFNO | FNO |
|---|---|---|
| `train_model` / `predict_next` | Explicit AVF in Fourier coordinates | Explicit AVF on projected fields |
| `validation_error` | Explicit AVF | Explicit AVF |
| Public `step` / `rollout`, method omitted | Gonzalez | Euler |
| `evaluate_model` | Gonzalez through the default | Euler through the default |
| Vorticity evaluation adapter | Gonzalez through the default | Euler through the default |
| `guarded_rollout` / long evaluation | Gonzalez through the default | Euler through the default |
| Assembly notebook | Explicit Euler | Explicit Euler |

The reference Navier–Stokes solver uses RK4, with exact evolution for the special wave dataset. That is an intentional reference-data choice, not a missing AVF call.

**1. [P1] Held-out evaluation drops the training integration method.**

Locations: [metrics.py:36](/Users/sarve/Downloads/PHFNO-main/src/experiments/metrics.py:36), [vorticity_metrics.py:45](/Users/sarve/Downloads/PHFNO-main/src/experiments/vorticity_metrics.py:45), [long_rollout.py:94](/Users/sarve/Downloads/PHFNO-main/src/experiments/long_rollout.py:94), [model.py:122](/Users/sarve/Downloads/PHFNO-main/src/phfno/model.py:122), [baseline.py:36](/Users/sarve/Downloads/PHFNO-main/src/phfno/baseline.py:36).

Training and validation explicitly pass `config.integration_method`, but test evaluation calls `model.rollout(...)` without a method. The vorticity adapter repeats that omission, and long evaluation calls `model.step(...)` without a method. A checkpoint's configuration does not change these Python defaults. Consequently, validation selection and reported test performance concern different discrete dynamics. The long-run manifest also hard-codes Gonzalez/Euler, and its failure handler recognizes only Gonzalez nonconvergence.

Verified against seed 1 checkpoints in `results/local_smoke_test_seeds_1_20_avf_retry`: stored wave and random predictions matched default-method rollouts exactly, with maximum difference zero. Explicit AVF changed the predictions:

| Model / family | Stored final NRMSE | Explicit AVF final NRMSE | Maximum field difference over rollout |
|---|---:|---:|---:|
| FNO / wave | 0.362891 | 0.356859 | 0.019287 |
| FNO / random | 0.427665 | 0.425246 | 0.014219 |
| PhFNO / wave | 0.796772 | 0.790252 | 0.020323 |
| PhFNO / random | 0.829703 | 0.829028 | 0.011963 |

Fix: pass the selected method through every evaluation and adapter layer, persist the actual evaluation method, and handle the chosen solver's failures. Handle legacy checkpoints explicitly rather than assuming new defaults describe their training protocol.

**2. [P1] AVF can accept a large endpoint residual or a nonfinite state.**

Location: [integrators.py:75](/Users/sarve/Downloads/PHFNO-main/src/phfno/integrators.py:75).

The stopping criterion tests `candidate - next_z`, then returns `candidate`. This is the fixed-point equation residual at the previous iterate, not at the returned endpoint. Without contraction, the endpoint residual can be much larger. There is also no finiteness guard: an infinite residual and infinite scale satisfy `inf <= inf`.

Reproduced with the default solver settings:

- Float64, `f(z) = -10000*z`, `z0 = 1e-16`, `dt = 1`: returns `4.9990001e-9`. The actual AVF equation residual is `2.5e-5`, versus tolerance approximately `1e-8`.
- Float32, `f(z) = -1e20*z`, `z0 = 1`, `dt = 1`: returns `inf` instead of reporting failure.

Fix: reject nonfinite iterates and residuals, and verify the AVF equation at the endpoint being returned. The Gonzalez implementation already performs an endpoint residual check.

**3. [P2] AVF applies different stopping norms to fields and coordinates.**

Location: [integrators.py:76](/Users/sarve/Downloads/PHFNO-main/src/phfno/integrators.py:76).

`vector_norm(..., dim=-1)` measures a complete coordinate vector for PhFNO, but only one spatial line for FNO. For a 3D field the comparison is separately applied to every channel and spatial line. This makes convergence depend on tensor layout and can cause unequal iteration counts or solver failures in the two models.

Reproduction: use two constant field channels initially `(1, 0)`, dynamics `(0, x - 4*y)`, and `dt = 0.1`. AVF accepts the state represented as `[1, 128]`, but raises after eight iterations for the same state and dynamics represented as `[1, 2, 4, 4, 4]`.

Fix: define a norm over the entire state per batch item, with field normalization consistent with the Fourier Parseval convention. Flattening alone addresses layout dependence; normalization also matters for cross-resolution absolute tolerances.

**4. [P2] State convergence does not ensure convergence of AVF gradients.**

Location: [integrators.py:69](/Users/sarve/Downloads/PHFNO-main/src/phfno/integrators.py:69).

Autograd differentiates only the iterations executed before the state-based early return. At an equilibrium the state can converge immediately while its sensitivity to parameters still needs further iterations.

Reproduction: `f(z, theta) = -10*z + theta`, `z0 = 0`, `theta = 0`, `dt = 0.1`. The exact implicit AVF sensitivity is `0.1 / 1.5 = 0.0666667`. The implementation returns gradient `0.05`, a 25% error. This persists with `max_iterations=64`, `rtol=1e-12`, and `atol=1e-14`; the state still exits immediately. A centered finite difference gives `0.0666666687`, and `torch.autograd.gradcheck` fails.

Fix: differentiate the converged implicit equation, or use a justified iteration strategy that also converges sensitivities. Successful backpropagation and finite gradients alone do not establish the correct implicit gradient.

**5. [P2] Gradient accumulation updates at the wrong group boundary.**

Location: [comparison.py:164](/Users/sarve/Downloads/PHFNO-main/src/experiments/comparison.py:164).

Groups are aligned relative to `theta_update_start`, but `optimizer.step()` uses the absolute condition `step % update_interval == 0`. These schedules agree only for particular configurations. A shortened final group also changes the modulus. Incorrectly early updates reuse earlier gradients, and a subsequent group reset can discard gradients that never contributed to an update.

Observed optimizer calls:

- `steps=23, start=20, interval=5`: updates at 21 and 23 after warmup; the final group should update only at 23.
- `steps=14, start=3, interval=5`: updates at 5, 10, and 14 after warmup; intended group ends are 8, 13, and 14.
- The standard `steps=600, start=20, interval=5` configuration happens to align correctly.

Fix: trigger updates at `step == group_start + update_interval - 1` and test offset starts and partial final groups.

**6. [P2] Representation-comparison validation ignores the new protocol fields.**

Location: [vorticity_comparison.py:26](/Users/sarve/Downloads/PHFNO-main/src/experiments/vorticity_comparison.py:26).

`validate_velocity_reference` checks architecture, seeds, data, and older optimizer fields, but omits `integration_method`, `theta_update_start`, and `theta_update_interval`. A reference explicitly labelled Euler with updates every batch is accepted against a new AVF configuration with delayed updates. This confounds velocity-versus-vorticity comparisons. Loading old dictionaries through `ComparisonConfig(**old_config)` also silently supplies the new AVF and accumulation defaults when those fields are absent.

Fix: version experiment protocols and compare the complete relevant configuration, including integrator settings, optimizer schedule, and loss definition. Explicitly migrate or reject legacy records with missing protocol information.

**7. [P2] Checkpoint loading fails on supported Python 3.10.**

Locations: [long_rollout.py:19](/Users/sarve/Downloads/PHFNO-main/src/experiments/long_rollout.py:19), [pyproject.toml:9](/Users/sarve/Downloads/PHFNO-main/pyproject.toml:9).

The package declares Python >=3.10, but `file_sha256` uses `hashlib.file_digest`, which is unavailable on Python 3.10. This caused the checkpoint-loader test to fail in the installed `research` environment and prevents the long-comparison hashing path from running there.

Fix: use chunked hashing compatible with Python 3.10, or consistently raise and document the minimum Python version.

**8. [P2] Learning curves label minibatch iterations as optimizer steps.**

Locations: [comparison.py:196](/Users/sarve/Downloads/PHFNO-main/src/experiments/comparison.py:196), [plots.py:53](/Users/sarve/Downloads/PHFNO-main/src/experiments/plots.py:53).

History `step`, `best_step`, and `threshold_step` count minibatch evaluations. After accumulation was introduced, these are no longer Adam update counts. Under the default schedule, 600 recorded iterations correspond to 136 optimizer updates; 3,000 correspond to 616. The plot still says “Optimizer steps” and “Learning per update,” and the experiment guide repeats the old interpretation.

Fix: record both counters and label each plot and summary with the counter actually used.

**Mathematical limitations of the current PhFNO AVF path**

AVF is correctly recognizable as Gauss–Legendre quadrature of the vector field along the segment between endpoints. That alone does not preserve the learned port-Hamiltonian energy balance when `J(z)`, `R(z)`, and `B(z)` are evaluated at every quadrature state. The familiar exact AVF energy/dissipation result assumes constant structure matrices; see [Celledoni et al., 2012](https://arxiv.org/abs/1202.4555).

The limitation is reproducible within the model's rank-two skew structure. Set `H(x,y)=(x*x+y*y)/2`, `J(x,y)=[[0,-x],[x,0]]`, and `R=0`, with no controls. Then `f=(-x*y,x*x)` conserves H continuously. From `(1,1)` with `dt=0.1`, the PhFNO AVF step increases H from 1 to approximately `1.000156869`. With a tightly converged solve, the increase is approximately `0.000156878`; it is not caused by the eight-iteration limit. The integrand is quadratic, so four-point quadrature is exact for this example.

For the actual SiLU energy network, four-point quadrature introduces an additional approximation to the energy-gradient integral. Thus neither learned-energy monotonicity nor exact discrete passivity follows from selecting `method="avf"`. If that property is required, use a discrete-gradient construction with a common skew/positive-semidefinite structure at the step level and explicitly verify its energy balance. If the objective is a shared generic integrator comparison, describe that narrower property accurately.

The fixed-point solver is also limited to eight iterations with no model-level solver options or substepping. For linear decay `f=-10*z`, `z0=1`, `dt=0.1`, it raises after eight iterations with residual `3.906e-3`, although the implicit step has the well-defined endpoint `1/3`. Even an unlimited fixed-point iteration does not converge for this linear problem when `abs(rate*dt/2) >= 1`. Expose solver settings and distinguish nonlinear-solver failure from instability of the implicit method or learned physical dynamics.

**Documentation, notebooks, and saved artifacts**

- [README.md:128](/Users/sarve/Downloads/PHFNO-main/README.md:128) labels an Euler equation as AVF, and lines 135–137 say a discrete-gradient method can be added even though Gonzalez is implemented. `implementation.md` describes AVF as the default although the public model defaults remain Gonzalez/Euler.
- Training uses the Fourier H1 loss in [comparison.py:179](/Users/sarve/Downloads/PHFNO-main/src/experiments/comparison.py:179), whereas `implementation.md` and `experiments.md` describe MSE. The vorticity guide still says Euler training.
- The first cell of `ipynb/compare_phfno_fno.ipynb` loads results from an absolute user-specific path before the training setup. A fresh checkout without that result file cannot run the notebook in order. A subsequent cell requires CUDA even though the training configuration sets `device="cpu"`.
- The plotting cell selects the unfinished AVF retry experiment: its saved results contain PhFNO seed 1 and FNO seeds 1 and 2, only 3 of the configured 40 runs. Current plots average these unequal seed sets without a completeness check. The earlier AVF directory has only 1 of 40 configured runs. These are partial artifacts, not completed twenty-seed comparisons.
- `results/phfno_fno_large/summary.json` describes 600-step runs, while the velocity reference embedded in the vorticity results describes 3,000-step runs with different selected checkpoints. They should not be assumed to be the same experiment. Legacy records lacking an integrator field do not establish AVF training.

**Validation performed**

Environment: Python 3.10.20, PyTorch 2.14.0, NeuralOperator 2.0.0, CPU. CUDA was unavailable.

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /Users/sarve/miniconda3/envs/research/bin/python -m pytest -q
```

Result: **187 passed, 7 failed, 5 skipped**, in 13.50 seconds. Failure breakdown:

- Four training tests still use `ScalarModel`, which has no `step` method after the AVF interface change.
- One FNO test expects the obsolete “only Euler” error message.
- One checkpoint-loader test exposes the Python 3.10 hashing failure.
- One notebook test expects the `research` kernelspec, but the long-rollout notebook records `python3`.

Additional runtime checks reproduced findings 1–6, the energy counterexample, and the linear-decay solver limit. Explicit two-step AVF rollouts for both actual model classes propagated finite gradients to all trainable parameters, the input fields, and controls. That confirms connectivity; the equilibrium counterexample separately shows why connectivity is insufficient to validate sensitivities.

The existing Fourier, physical reference solver, vorticity operator, and discrete Gonzalez tests largely passed. No additional defect was established in those core mathematical operations during this audit. Passing tests do not establish the correctness of every untested configuration, and GPU behavior was not validated.

Missing regression coverage includes method propagation into test and long rollouts, AVF endpoint residuals and overflow, field-versus-coordinate norm consistency, equilibrium sensitivities, AVF learned-energy behavior, nonaligned accumulation groups, and comparison of legacy versus current training protocols. Existing AVF unit tests cover linear decay and batch timestep broadcasting but do not cover these failure modes.
