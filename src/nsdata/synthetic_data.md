# Synthetic Navier–Stokes data with forcing and viscous dissipation

The data describe a three-component incompressible velocity on a periodic unit cube, observed from time zero to one. Each trajectory starts from a smooth, divergence-free field. A prescribed external force drives the flow, viscosity removes kinetic energy, and Gaussian noise is added afterward to represent measurement error.

## Domain and equation

The spatial domain and governing equation are

$$
\Omega=(\mathbb{R}/\mathbb{Z})^3,\qquad t\in[0,1],
$$

$$
\begin{aligned}
\partial_t\mathbf{v}+(\mathbf{v}\cdot\nabla)\mathbf{v}
&=-\nabla p+\nu\Delta\mathbf{v}+\mathbf{f},\\
\nabla\cdot\mathbf{v}&=0.
\end{aligned}
$$

Space is sampled uniformly without duplicating the periodic endpoint. Saved times include both zero and one. The numerical solver takes smaller internal steps when required by advection, diffusion, or forcing.

The starter settings are:

| Quantity | Value |
|---|---|
| Spatial grid | 16 points per axis |
| Saved snapshots | 21, spaced by 0.05 |
| Trajectories per initial-condition family | 4 |
| Initial componentwise RMS velocity | 0.2 |
| Kinematic viscosity | 0.01 |
| Forcing control amplitude | 0.1 |
| Forcing control frequency | One cycle per unit time |
| Maximum numerical time step | 0.005 |
| Noise standard deviation | 1% of initial componentwise RMS |
| Random seed | 42 |

The componentwise RMS convention averages over both spatial points and velocity components:

$$
\text{RMS}(\mathbf{v})
=\left(\frac{1}{3K}\sum_{j=1}^{K}\|\mathbf{v}(x_j)\|^2\right)^{1/2}.
$$

Simulation uses double precision by default. Saved velocity, force, and viscous fields use single precision for use with the neural models. These small datasets are intended for initial experiments, rather than claims about resolved turbulence.

## Prescribed external forcing

Every trajectory uses the same fixed spatial forcing pattern:

$$
\mathbf{g}(x,y,z)=\sqrt{\frac{6}{5}}
\begin{pmatrix}
1\\
0\\
-2
\end{pmatrix}
\sin\!\left(2\pi(2x+3y+z)\right).
$$

The polarization is perpendicular to the wavevector, so the force is divergence-free. The normalization makes its componentwise RMS equal to one. Using the same pattern across trajectories avoids an unrecorded change in the physical dynamics when the initial condition changes.

A scalar control sets the force strength. At each saved time, evaluate a cosine and hold its value constant until the next saved time:

$$
u_j=A_f\cos(2\pi\omega t_j),
$$

$$
\mathbf{f}(t,x)=u_j\mathbf{g}(x),
\qquad t_j\le t<t_{j+1}.
$$

The defaults are

$$
A_f=0.1,\qquad \omega=1.
$$

The force therefore changes direction during the trajectory. This is a piecewise-constant deterministic input, not white-noise forcing. The scalar control and its full spatial force field are both retained in the dataset. The control convention also matches a neural model that takes one input value per prediction interval.

A zero forcing amplitude gives an unforced case. Numerical trajectories with nonzero forcing need at least ten points per spatial axis so the forcing modes fit inside the retained frequency range.

## Viscous dissipation

The only physical dissipation is the Navier–Stokes viscous term:

$$
\mathbf{d}_{\text{visc}}=\nu\Delta\mathbf{v}.
$$

In Fourier space, it acts separately on each mode:

$$
\widehat{\mathbf{d}}_{\text{visc}}(k)
=-4\pi^2\nu\|k\|^2\widehat{\mathbf{v}}(k).
$$

Higher frequencies decay faster. There is no added linear drag or separate uniform damping term. The viscous acceleration is stored at every saved clean velocity field, along with the viscosity used to generate the trajectory.

For periodic incompressible flow, kinetic energy, dissipation rate, and supplied power are

$$
E(t)=\frac{1}{2}\int_\Omega\|\mathbf{v}\|^2\,\text{d}x.
$$

$$
\begin{aligned}
\mathcal{D}(t)
&=\nu\int_\Omega\|\nabla\mathbf{v}\|_{\text{F}}^2\,\text{d}x\\
&=-\int_\Omega\mathbf{v}\cdot\mathbf{d}_{\text{visc}}\,\text{d}x.
\end{aligned}
$$

$$
\mathcal{P}(t)=\int_\Omega\mathbf{v}\cdot\mathbf{f}\,\text{d}x.
$$

They obey the continuous balance

$$
\frac{\text{d}E}{\text{d}t}=\mathcal{P}(t)-\mathcal{D}(t).
$$

Dissipation is nonnegative, but energy need not decrease when external input supplies power. Input power and the energy rate are recorded at the left endpoint of each saved interval. They are instantaneous quantities, not a claim that a finite difference between saved energies is exactly equal to those rates.

## Initial-condition families

### Exact forced waves

Choose a random phase for each trajectory:

$$
\mathbf{v}_0(x,y,z)=\frac{A}{\sqrt{5}}
\begin{pmatrix}
1\\
0\\
-2
\end{pmatrix}
\sin\!\left(2\pi(2x+3y+z)+\phi\right).
$$

Normalize the initial field to the chosen RMS velocity. The forcing has the same wavevector and polarization, so nonlinear advection remains zero even as the phase and amplitude evolve. Over each interval, the exact solution is

$$
\mathbf{v}_{j+1}
=e^{-\lambda\Delta t_j}\mathbf{v}_j
+\frac{1-e^{-\lambda\Delta t_j}}{\lambda}\,u_j\mathbf{g},
$$

$$
\lambda=56\pi^2\nu.
$$

At zero viscosity, the fraction is replaced by the interval length. Without forcing, this reduces to exponential viscous decay. These trajectories provide an exact reference for checking the numerical solver and basic learning behavior. They do not exercise nonlinear interactions between distinct wavevectors.

### Taylor–Green fields

Use periodic spatial shifts of the initial field

$$
\mathbf{v}_0=A
\begin{pmatrix}
\sin(2\pi x)\cos(2\pi y)\cos(2\pi z)\\
-\cos(2\pi x)\sin(2\pi y)\cos(2\pi z)\\
0
\end{pmatrix}.
$$

Each shifted field is divergence-free and is normalized to the chosen initial RMS. The external forcing pattern remains fixed. Subsequent fields are obtained by numerical integration of the forced Navier–Stokes equations.

### Random smooth fields

Generate random real fields, transform them to Fourier coefficients, retain only low frequencies, and project each nonzero mode perpendicular to its wavevector:

$$
\widehat{\mathbf{v}}_0(k)
=\left(I-\frac{kk^T}{\|k\|^2}\right)\widehat{\mathbf{w}}(k),
\qquad k\ne0.
$$

The mean is zero. The initial cutoff is two in each direction, and the RMS is normalized separately for each trajectory. Starting from real fields preserves conjugate symmetry. Several retained modes can interact nonlinearly during subsequent forced evolution.

## Numerical evolution

Taylor–Green and random fields use a Fourier solver. Velocity is projected onto divergence-free modes, eliminating the pressure gradient from the evolution. The nonlinear term is evaluated through velocity crossed with vorticity, followed by the same pressure projection.

To prevent unresolved quadratic products from folding into retained frequencies, apply a strict two-thirds filter along every spatial axis:

$$
|k_i|<\frac{M}{3}.
$$

The strict inequality matters at the boundary. The state, nonlinear update, and forcing are kept within this range. Fourth-order Runge–Kutta advances the Fourier coefficients. Its internal step is limited by the requested maximum, advection, diffusion, forcing acceleration, and the next saved time.

The force stays constant throughout every internal step belonging to the same saved interval. Pressure is eliminated by projection and is not stored. A finer grid and smaller time step should be checked before drawing physical conclusions from the generated trajectories.

## Gaussian observation noise

Noise is added only after the clean trajectory has been generated:

$$
\widetilde{\mathbf{v}}_j
=\mathbf{v}_j+\sigma\boldsymbol{\epsilon}_j.
$$

$$
\boldsymbol{\epsilon}_j\sim\mathcal{N}(0,I),
\qquad
\sigma=\eta\,\text{RMS}(\mathbf{v}_0).
$$

The default relative noise level is 0.01, giving a standard deviation of 0.002. This scale stays fixed throughout the trajectory. Clean and noisy observations are retained together; noise does not alter the external force, viscosity, or underlying clean solution.

Independent observation noise generally breaks incompressibility. An optional projection makes the noise divergence-free and removes frequencies outside the solver's retained range. The resulting noise is correlated, with lower RMS. The recorded noise scale refers to the Gaussian noise before this projection.

## What each trajectory contains

Each trajectory retains the clean velocity, noisy velocity, saved times, spatial grid, scalar controls, shared forcing pattern, full external force, and viscous acceleration. It also retains the random seeds and generation parameters.

The accompanying diagnostics are kinetic energy, divergence RMS, mean velocity, viscous dissipation rate, input power, and the continuous energy rate. They are computed from the stored clean fields so they describe the data actually supplied to a model. Numerical tests check exact forced evolution, the driven mean flow, energy balance, incompressibility, time-step convergence, and reproducibility.
