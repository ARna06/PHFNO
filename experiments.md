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

The wave has an exact forced evolution. The other two families use the Fourier solver with RK4 and a strict two-thirds frequency truncation. These are numerical or analytical references, rather than experimental measurements. Generation uses CUDA, double precision, and seed 142; the stored velocity tensors use single precision. Gaussian observation noise is added after solving, and the clean reference remains available for evaluation.

The comparison loads the larger datasets from `datasets/large/`. The earlier twelve-trajectory pilot remains in `datasets/` as a separate dataset.

## Keep whole trajectories apart

Within every family, trajectories 0 through 23 are used for training, 24 through 27 for validation, and 28 through 31 for testing. No trajectory contributes snapshots to more than one split. This gives 72 training trajectories, 12 validation trajectories, and 12 test trajectories. There are 2,880 training transitions in total.

The stored families reuse the same noise seed and therefore share noise arrays at corresponding trajectory indices. Keeping the same indices in every split avoids placing one of those noise arrays in both training and testing.

## What each model learns

Both models receive the current noisy velocity and the scalar forcing control. They predict a velocity derivative, which advances the state using Euler:

$$
\widehat v_{j+1}=\Pi_N\widetilde v_j
+\Delta t\,F_\theta(\widetilde v_j,a_j),
\qquad \Delta t=0.025.
$$

The Fourier cutoff is 7 in every direction. On this grid it retains every mode kept by the data generator. The FNO width is 16 with two layers; the port-Hamiltonian scalar networks have width 64, and its internal grid is also 24 × 24 × 24. The existing model definitions are used directly.

Training compares the predicted next state with the next noisy observation. A single RMS scale is computed from the current noisy training states and reused throughout:

$$
s^2=\operatorname{mean}_{\mathrm{training}}|\widetilde v_j|^2,
\qquad
\mathcal L=\frac{\operatorname{mean}|\widehat v_{j+1}-\widetilde v_{j+1}|^2}{s^2}.
$$

Validation starts from noisy current states but compares the predicted next states with clean references, using the same training scale. Clean equations and derivatives are used for diagnostics, not as extra training targets.

## Make the training comparison reproducible

Training uses Adam with learning rate 0.001, gradient clipping at 1, batches of 24, and seeds 7, 17, and 27. Each run takes 3,000 optimizer steps, with validation every 100 steps. Both models receive the same sampled batches for each seed. The notebook configuration and the saved run configuration record these settings.

Width, depth, optimizer, data, and update budget are shared. Parameter count and computational cost are not identical. Report both models' trainable parameter counts as real scalar values, counting each complex parameter as two real values.

Validation selects the best checkpoint independently for each run. Only after this selection is the checkpoint evaluated on the held-out test trajectories. Training curves show validation error against both optimizer steps and elapsed training seconds. CUDA is synchronized around timing; construction, warm-up, validation, plotting, and testing are excluded from the training clock.

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

The baseline receives a broadcast scalar control and has no positional embedding. Its translation symmetry restricts its ability to produce a fixed spatial forcing pattern from that control. This experiment retains that interface, so any difference can reflect this limitation as well as optimization and the port-Hamiltonian structure. It should not be described as a general advantage over all FNO formulations.

Euler training learns changes across finite snapshot intervals. These differ from instantaneous derivatives, even with perfect clean data. For the wave here, the unforced secant derivative over one interval is approximately 6.6% smaller in magnitude than its instantaneous derivative. Reference Euler error and secant-derivative error are retained to expose this effect. Finite-step energy differences likewise need not satisfy the instantaneous energy balance exactly.

## Run and inspect

Running [the comparison notebook](ipynb/compare_phfno_fno.ipynb) retrains the models by default, with tqdm showing training progress. Figures appear directly in the notebook. The configuration, training histories, selected checkpoints, test diagnostics, and summary are saved under `results/phfno_fno_large/`, including the tensor results and JSON summary.

The notebook contains only code; this document explains what its outputs measure. Conclusions should follow the completed curves and held-out errors, including cases where the FNO learns faster or neither model reproduces the reference well.
