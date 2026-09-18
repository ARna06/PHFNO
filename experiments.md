# Comparing the two models on synthetic flow

This experiment asks whether the port-Hamiltonian model learns the available trajectories more quickly than the FNO baseline, and whether its predictions agree with the dynamics that generated them. It uses 96 trajectories and repeats training with three initialization seeds. The curves can reveal useful behavior and failures within this setup; they do not establish general superiority across flow regimes.

## The reference data

The data contain 32 trajectories from each of three families: a decaying, forced wave; Taylor–Green initial conditions; and random smooth, divergence-free initial conditions. Each trajectory has 41 snapshots on a periodic 24 × 24 × 24 grid over times zero to one. Viscosity is 0.01. The reference equation is

$$
\partial_t v = -\mathbb P[(v\cdot\nabla)v]+\nu\Delta v+f_j,
\qquad \nabla\cdot v=0.
$$

The projection removes the pressure gradient and keeps the velocity divergence-free. The prescribed force is held constant within each saved time interval:

$$
f_j(x,y,z)=a_j\sqrt{\frac65}
\begin{pmatrix}1\\0\\-2\end{pmatrix}
\sin\!\left(2\pi(2x+3y+z)\right),
\qquad a_j=0.1\cos(2\pi t_j).
$$

<!-- Audit fix: reference generation shares the notebook's CUDA-or-CPU device selection. -->
The wave has an exact forced evolution. The other two families use the Fourier solver with RK4 and a strict two-thirds frequency truncation. These are numerical or analytical references, rather than experimental measurements. Generation uses CUDA when available or CPU otherwise, double precision, and seed 142; the stored velocity tensors use single precision. Gaussian observation noise is added after solving, and the clean reference remains available for evaluation.

The comparison loads the larger datasets from `datasets/large/`. The earlier twelve-trajectory pilot remains in `datasets/` as a separate dataset.

## Keep whole trajectories apart

Within every family, trajectories 0 through 23 are used for training, 24 through 27 for validation, and 28 through 31 for testing. No trajectory contributes snapshots to more than one split. This gives 72 training trajectories, 12 validation trajectories, and 12 test trajectories. There are 2,880 training transitions in total.

The stored families reuse the same noise seed and therefore share noise arrays at corresponding trajectory indices. Keeping the same indices in every split avoids placing one of those noise arrays in both training and testing.

## What each model learns

<!-- Audit fix: the AVF segment starts at the Fourier-projected observed state. -->
Both models receive the current noisy velocity and the scalar forcing control. They predict a velocity derivative, which advances the state using AVF:

$$
\widehat v_{j+1}=\Pi_N\widetilde v_j
\;+\;\Delta t\int_0^1
F_\theta\bigl((1-\xi)\Pi_N\widetilde v_j+\xi\widehat v_{j+1},a_j\bigr)\,d\xi,
\qquad \Delta t=0.025.
$$

The Fourier cutoff is 7 in every direction. On this grid it retains every mode kept by the data generator. The FNO width is 16 with two layers; the port-Hamiltonian scalar networks have width 64, and its internal grid is also 24 × 24 × 24. The existing model definitions are used directly.

<!-- Audit fix: training uses Fourier H1; NRMSE remains a separate validation metric. -->
Training compares the predicted next state with the next noisy observation using
the Fourier H1 loss. Let \(e=\widehat v_{j+1}-\widetilde v_{j+1}\) and let
\(\widehat e\) denote its orthonormal FFT. A single RMS scale is computed from
the current noisy training states and reused throughout:

$$
s^2=\operatorname{mean}_{\mathrm{training}}|\widetilde v_j|^2,
\qquad
\mathcal L=\frac{\operatorname{mean}_{\mathrm{batch},c,k}
\left[(1+4\pi^2\|k\|^2)|\widehat e_c(k)|^2\right]}{s^2}.
$$

Validation starts from noisy current states but compares the predicted next states with clean references using ordinary field RMSE divided by the same training scale. Clean equations and derivatives are used for diagnostics, not as extra training targets.

## Make the training comparison reproducible

<!-- Audit fix: the current notebook uses seeds 8/18/28 and counts minibatches separately from Adam updates. -->
Training uses Adam with learning rate 0.001, gradient clipping at 1, batches of 24, and seeds 8, 18, and 28. Each run takes 3,000 minibatch iterations, with validation every 100 iterations. Both models receive the same sampled batches for each seed. The comparison uses
the implicit AVF step for both models. The notebook configuration and the saved
run configuration record these settings. Losses are evaluated every transition,
but parameter updates occur every step through step 20 and then every five
minibatches; use `theta_update_interval=10` for the ten-step ablation. The default
3,000-iteration schedule makes 616 Adam updates. Histories record both counts.

Width, depth, optimizer, data, and update budget are shared. Parameter count and computational cost are not identical. Report both models' trainable parameter counts as real scalar values, counting each complex parameter as two real values.

<!-- Audit fix: evaluate the selected checkpoint with its training method and label the plotted counter accurately. -->
Validation selects the best checkpoint independently for each run. Only after this selection is the checkpoint evaluated on the held-out test trajectories, using the same AVF method and solver settings. Training curves show validation error against minibatch iterations and elapsed training seconds. CUDA is synchronized around timing; construction, warm-up, validation, plotting, and testing are excluded from the training clock.

For test curves, the four held-out trajectories within each family are averaged separately for each seed. The shaded bands then show the minimum and maximum across the three seed averages. These bands are not confidence intervals. Overall final-error summaries first average equally over families and their held-out trajectories within each seed, then summarize the three seed values.

A common target, set at half the validation persistence error, provides one additional comparison of the recorded steps and seconds needed to reach the same error. A run that never reaches it is reported as such.

## Check predictions against the equations

Each test rollout starts from the clean initial velocity and then uses its own predictions, together with the prescribed control sequence. No later reference state is fed back into the rollout. Errors are shown separately for each family. One normalization uses the initial velocity RMS so that a nearly decayed wave does not create a misleadingly large denominator effect; ordinary relative L2 errors are also retained.

The physical checks include kinetic energy, raw velocity divergence, and the error in the learned derivative evaluated on true saved states. Predictions are not projected to divergence-free fields before measuring divergence. The reference derivative uses the same spatial discretization and left-endpoint forcing as the generator.

$$
K(v)=\frac12\langle |v|^2\rangle,
\qquad
\frac{dK}{dt}=\langle v\cdot f\rangle-\nu\langle |\nabla v|^2\rangle.
$$

Angle brackets denote the spatial average. The physical power residual compares the learned instantaneous kinetic-energy rate with forcing work minus viscous dissipation, on true states:

$$
r_K=\langle v\cdot F_\theta(v,a)\rangle
-\left(\langle v\cdot f\rangle-\nu\langle|\nabla v|^2\rangle\right).
$$

The diagnostics also compare the learned change caused by switching the control on with the known forcing field. Its error uses the forcing RMS across the whole trajectory, so intervals where the control is nearly zero do not cause division by a tiny instantaneous force.

## Read the curves with the model assumptions in view

The port-Hamiltonian model has a skew operator of rank at most two and scalar damping. Its learned energy is not constrained to equal physical kinetic energy. Satisfying its own energy identity therefore does not establish that it has recovered Navier–Stokes energy transfer or viscous dissipation.

<!-- Audit fix: a common AVF solver does not impose a discrete energy identity for state-dependent structure. -->
The AVF comparison shares an integration method between the models. Its
quadrature evaluates PhFNO's state-dependent structure along the endpoint
segment, so exact discrete energy conservation, dissipation, and passivity are
not guaranteed. Public model defaults remain Gonzalez for PhFNO and Euler for
FNO; the comparison passes AVF explicitly.

Both models learn their control response through an FNO without positional information. The baseline broadcasts the scalar control; the port-Hamiltonian model constructs its input map from the velocity field using its FNO. Shifting the velocity field shifts the learned control response in both models, whereas the prescribed forcing pattern stays fixed in space. Neither input interface can therefore produce this fixed force independently of the state for arbitrary translated states. The port-Hamiltonian energy network can break the translation symmetry of its autonomous dynamics, but it does not remove this restriction on the input map. This comparison reflects these input interfaces as well as optimization and the port-Hamiltonian structure; it cannot establish a general advantage over other FNO formulations.

AVF training still learns changes across finite snapshot intervals. These differ from
instantaneous derivatives, even with perfect clean data. Reference Euler error and
secant-derivative error remain diagnostics for the reference solver, not the
training integrator.

## Run and inspect

Running [the comparison notebook](ipynb/compare_phfno_fno.ipynb) retrains the models by default. With the notebook dependencies installed, tqdm uses a widget that updates in place for each training run. Figures appear directly in the notebook, without footer captions. The velocity slice uses the first seed and the first held-out trajectory, chosen before looking at the results.

<!-- Audit fix: follow the active notebook output and reject legacy protocol inference. -->
The configuration, training histories, selected checkpoints, test diagnostics, and summary are saved under `results/local_smoke_test_seeds_8_18_28/`, including the tensor results and JSON summary. Saved protocol records include the loss, integrator settings, and accumulation schedule. Legacy records missing those fields must be regenerated before use in a current comparison.

The notebook contains only code; this document explains what its outputs measure. Conclusions should follow the completed curves and held-out errors, including cases where the FNO learns faster or neither model reproduces the reference well.

## What the larger run showed

<!-- Audit fix: retain historical measurements without presenting them as a verified current AVF comparison. -->
The historical H100 report described 3,000 recorded training iterations for both
models and all three seeds. These measurements predate the audit fixes and are
not validated results for the current AVF protocol. In particular, the current
`results/phfno_fno_large/summary.json` describes 600 iterations, whereas the
vorticity experiment's saved velocity reference describes 3,000. They must not
be treated as the same run. The historical reported averages were:

| Measure | PHFNO | FNO |
|---|---:|---:|
| Selected one-step validation NRMSE | 0.01897 | 0.01954 |
| Recorded iterations to the shared target | 233 | 267 |
| Optimization seconds to the target | 1.98 | 1.71 |
| Optimization seconds for all 3,000 iterations | 25.11 | 19.21 |
| Final held-out rollout NRMSE | 0.05999 | 0.05077 |

In that report, PHFNO reached a slightly lower one-step validation error and used fewer recorded iterations to reach the shared target on average. FNO reached that target sooner in elapsed time and had lower final rollout error. Target crossings were only checked every 100 iterations, and three seeds give a limited view of training variability. These historical measurements need regeneration with consistent integration and complete seed sets before drawing conclusions about the current protocol.

Both models followed the broad kinetic-energy decay, but the equation checks exposed remaining errors. Mean derivative errors were roughly 23–64%, predicted divergence was far above the reference, and the learned forcing response remained weak. Good energy curves alone would overstate how much of the Navier–Stokes dynamics the models recovered.

Validation error uses the fixed training RMS. Final rollout error uses each test trajectory's initial RMS and is averaged across the held-out trajectories and flow types. These two columns therefore measure different prediction tasks and use different normalizations.
