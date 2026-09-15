# Fitting the models in vorticity coordinates

This experiment changes the fitted state from velocity to vorticity. It uses the same saved trajectories, split, model definitions, optimizer, Fourier cutoff, and training budget as the velocity experiment. Both PHFNO and FNO receive the transformed state, so the comparison separates the choice of state from the choice of model.

The existing velocity experiment is kept as the reference. The new notebook is [compare_vorticity.ipynb](ipynb/compare_vorticity.ipynb), and its outputs are saved under `results/phfno_fno_vorticity/`.

## Change the state and the forcing

On the periodic unit cube, define

$$
\omega=\nabla\times v,
\qquad g=\nabla\times f.
$$

The transform uses ordinary PyTorch FFTs. For a Fourier wave vector, the angular frequency and transformed vorticity are

$$
\kappa=2\pi k,
\qquad \widehat\omega(k)=i\kappa\times\widehat v(k).
$$

Clean velocities, noisy observations, the forcing basis, and the full forcing field are all transformed this way. The temporal controls stay the same. Curl does not remove forcing; it changes the spatial input field.

Noise is transformed along with the observations. No new independent noise is added to vorticity. Differentiation emphasizes high frequencies, so the resulting noise differs from the original velocity noise. On an even grid, Nyquist planes are removed consistently in curl and inverse curl. All resolved clean modes used by the generator are retained.

## The physical reference

For divergence-free vorticity and zero-mean velocity, the kinetic-energy functional is

$$
\mathcal H[\omega]=\frac12\langle\omega,(-\Delta)^{-1}\omega\rangle,
\qquad \frac{\delta\mathcal H}{\delta\omega}=(-\Delta)^{-1}\omega.
$$

Angle brackets denote the spatial integral, which equals the spatial average on the unit cube. The reference operators are

$$
J[\omega]\phi=\nabla\times\bigl((\nabla\times\phi)\times\omega\bigr),
\qquad R\phi=\nu(-\Delta)^2\phi.
$$

They give

$$
\partial_t\omega=(J[\omega]-R)\frac{\delta\mathcal H}{\delta\omega}+g
=\nabla\times(v\times\omega)+\nu\Delta\omega+\nabla\times f.
$$

The curl of the cross product includes both transport and vortex stretching. The implementation contains these physical reference operators, with the same two-thirds Fourier truncation as the data generator. Small tests check skew symmetry, nonnegative dissipation, the kinetic-energy identity, and agreement with the curl of the velocity solver's full derivative.

For forced flow, the enstrophy balance also includes forcing work:

$$
\frac{d}{dt}\frac12\|\omega\|^2
=\langle\omega,(\omega\cdot\nabla)v\rangle
-\nu\|\nabla\omega\|^2+\langle\omega,g\rangle.
$$

The stretching term is absent in two dimensions. The forcing contribution belongs in both the two-dimensional and three-dimensional balances.

## What is learned

The physical reference operators above are used for verification. The learned models retain their existing Hamiltonian, skew factors, damping, and control interfaces. In particular, changing the input to vorticity does not force the learned Hamiltonian to equal kinetic energy or make the learned dissipation equal the physical operator.

Training applies the existing Euler step to noisy vorticity and fits the next noisy vorticity observation. The loss uses one RMS scale computed from the training vorticity. Validation uses noisy current states and clean next vorticity, and selects the best checkpoint independently for each model and seed.

This changes the error being minimized: curl gives higher frequencies greater weight. Native vorticity and velocity validation losses therefore should not be compared numerically. The four-model comparison uses held-out velocity errors after reconstruction. The learning curves within the vorticity experiment compare PHFNO and FNO using the same vorticity loss.

## Recover velocity without losing the mean

For nonzero frequencies, inverse curl gives

$$
\widehat v(k)=\frac{i\kappa\times\widehat\omega(k)}{|\kappa|^2}.
$$

The zero-frequency multiplier is set to zero. Mean velocity is carried separately from the initial state and the known mean force:

$$
\overline v(t_{j+1})=\overline v(t_j)+\Delta t_j\,\overline f_j.
$$

The present datasets have zero mean velocity and zero mean forcing up to storage roundoff. The adapter preserves the mean component rather than silently discarding it. A future experiment with varying nonzero mean velocities would also need to give that mean to the learned dynamics, because it affects vorticity transport.

## Check what reconstruction hides

An arbitrary network output need not be a valid vorticity field. Inverse curl discards its constant and longitudinal components and always produces divergence-free velocity. Small reconstructed velocity divergence is therefore a property of the decoder, not evidence that the network learned incompressibility.

The experiment saves and scores raw predicted vorticity before reconstruction. Alongside raw vorticity error, divergence, and spatial mean, it measures the reconstruction mismatch

$$
\frac{\operatorname{RMS}\bigl(\widehat\omega-\nabla\times\widehat v\bigr)}
{\operatorname{RMS}(\omega_0)}.
$$

The velocity diagnostics use the same test trajectories and normalization as the existing experiment: trajectory error, physical kinetic energy, true Navier–Stokes derivative error, forcing response, and power balance. Dataset hashes and training settings are checked before comparing the two representations.

Plots appear inside the notebook without footer captions. Each training run uses an in-place tqdm widget. Shaded bands average test trajectories within each seed first, then show the minimum and maximum across seeds.

## Results of the comparison

The H100 run completed 3,000 updates for each model and all three seeds. The following errors use reconstructed velocity, normalized by each trajectory's initial velocity RMS and averaged over the held-out trajectories, flow types, and seeds:

| Model | Fitted state | Final velocity NRMSE |
|---|---|---:|
| PHFNO | Velocity | 0.06020 |
| PHFNO | Vorticity | 0.08993 |
| FNO | Velocity | 0.05079 |
| FNO | Vorticity | 0.07940 |

Final rollout error increased by about 49% for PHFNO and 56% for FNO. Most of the deterioration came from random flows. The instantaneous velocity derivative checks improved on average, particularly for Taylor–Green flow, but those gains did not translate into better autonomous rollouts.

Raw predicted vorticity also developed components that inverse curl cannot reconstruct. On random flows, the average reconstruction mismatch was 0.0708 for PHFNO and 0.0317 for FNO, using the initial vorticity RMS as the scale. These hidden components can still affect later learned vorticity updates. Forcing-response errors remained high for both models.

This run does not establish an advantage from curl reparameterization alone. It changes the state, the weighting of the fitting loss, and the reconstruction of predictions. It does not test a model constrained to the physical kinetic-energy Hamiltonian and the exact physical operators.

The numerical velocity reference used for this comparison is saved as `velocity_reference.json` beside the new results, so later reruns of the original experiment do not change the recorded comparison.
