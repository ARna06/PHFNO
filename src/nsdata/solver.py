import math

import torch


class PeriodicNavierStokes:
    def __init__(
        self,
        grid_size: int,
        viscosity: float,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ):
        if isinstance(grid_size, bool) or not isinstance(grid_size, int) or grid_size < 4:
            raise ValueError("grid_size must be an integer of at least 4")
        if not math.isfinite(viscosity) or viscosity < 0:
            raise ValueError("viscosity must be finite and nonnegative")
        if dtype not in (torch.float32, torch.float64):
            raise TypeError("dtype must be torch.float32 or torch.float64")
        self.grid_size = grid_size
        self.viscosity = float(viscosity)
        self.dtype = dtype
        modes = torch.fft.fftfreq(grid_size, d=1 / grid_size, device=device, dtype=dtype).round()
        indices = torch.stack(torch.meshgrid(modes, modes, modes, indexing="ij"))
        self.device = modes.device
        self.k = 2 * math.pi * indices
        self.ik = (1j * self.k).unsqueeze(0)
        self.k2 = self.k.square().sum(dim=0)
        self.inverse_k2 = self.k2.clamp_min(1).reciprocal()
        self.inverse_k2[0, 0, 0] = 0
        self.dealias_mask = (indices.abs() < grid_size / 3).all(dim=0)
        self.nyquist_mask = (indices.abs() < grid_size / 2).all(dim=0)
        largest_k2 = (self.k2 * self.dealias_mask).max().item()
        self.diffusion_dt = (
            0.5 / (self.viscosity * largest_k2) if self.viscosity else math.inf
        )

    def _validate_field(self, field: torch.Tensor, spectral: bool = False):
        expected_shape = (3,) + (self.grid_size,) * 3
        if field.ndim != 5 or field.shape[0] == 0 or field.shape[1:] != expected_shape:
            raise ValueError("field must have shape [batch, 3, grid_size, grid_size, grid_size]")
        expected_dtype = self.dtype
        if spectral:
            expected_dtype = torch.complex64 if self.dtype == torch.float32 else torch.complex128
        if field.dtype != expected_dtype:
            raise TypeError(f"field dtype must be {expected_dtype}")
        if field.device != self.device:
            raise ValueError(f"field device must be {self.device}")

    def leray_project(self, spectrum: torch.Tensor, dealias: bool = True):
        """Apply the Fourier-space Leray projector and optional dealiasing."""
        parallel = (self.k * spectrum).sum(dim=1, keepdim=True) * self.inverse_k2
        mask = self.dealias_mask if dealias else self.nyquist_mask
        return (spectrum - self.k * parallel) * mask

    def project(self, field: torch.Tensor, dealias: bool = True):
        self._validate_field(field)
        spectrum = torch.fft.fftn(field, dim=(-3, -2, -1), norm="forward")
        spectrum = self.leray_project(spectrum, dealias=dealias)
        return torch.fft.ifftn(spectrum, dim=(-3, -2, -1), norm="forward").real

    def divergence(self, field: torch.Tensor):
        self._validate_field(field)
        spectrum = torch.fft.fftn(field, dim=(-3, -2, -1), norm="forward")
        divergence = 1j * (self.k * spectrum).sum(dim=1)
        return torch.fft.ifftn(divergence, dim=(-3, -2, -1), norm="forward").real

    def viscous_term(self, field: torch.Tensor):
        self._validate_field(field)
        spectrum = torch.fft.fftn(field, dim=(-3, -2, -1), norm="forward")
        viscous_spectrum = -self.viscosity * self.k2 * spectrum
        return torch.fft.ifftn(viscous_spectrum, dim=(-3, -2, -1), norm="forward").real

    def rhs(self, spectrum: torch.Tensor, forcing_spectrum: torch.Tensor | None = None):
        self._validate_field(spectrum, spectral=True)
        if forcing_spectrum is not None:
            self._validate_field(forcing_spectrum, spectral=True)
            if forcing_spectrum.shape != spectrum.shape:
                raise ValueError("forcing_spectrum must have the same shape as spectrum")
        spectrum = self.leray_project(spectrum)
        velocity = torch.fft.ifftn(spectrum, dim=(-3, -2, -1), norm="forward").real
        curl_spectrum = torch.linalg.cross(self.ik, spectrum, dim=1)
        vorticity = torch.fft.ifftn(curl_spectrum, dim=(-3, -2, -1), norm="forward").real
        nonlinear = torch.linalg.cross(velocity, vorticity, dim=1)
        nonlinear = torch.fft.fftn(nonlinear, dim=(-3, -2, -1), norm="forward")
        derivative = self.leray_project(nonlinear) - self.viscosity * self.k2 * spectrum
        derivative[:, :, 0, 0, 0] = 0
        if forcing_spectrum is not None:
            derivative = derivative + self.leray_project(forcing_spectrum)
        return derivative

    def _step(self, spectrum: torch.Tensor, dt: float, forcing_spectrum=None):
        k1 = self.rhs(spectrum, forcing_spectrum)
        k2 = self.rhs(spectrum + 0.5 * dt * k1, forcing_spectrum)
        k3 = self.rhs(spectrum + 0.5 * dt * k2, forcing_spectrum)
        k4 = self.rhs(spectrum + dt * k3, forcing_spectrum)
        return self.leray_project(spectrum + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6)

    @torch.no_grad()
    def solve(
        self, initial: torch.Tensor, times, max_dt: float = 0.005,
        cfl: float = 0.4, forcing: torch.Tensor | None = None,
    ):
        self._validate_field(initial)
        if not torch.isfinite(initial).all():
            raise ValueError("initial field must be finite")
        if not math.isfinite(max_dt) or max_dt <= 0:
            raise ValueError("max_dt must be finite and positive")
        if not math.isfinite(cfl) or not 0 < cfl <= 1:
            raise ValueError("cfl must be finite and in (0, 1]")
        times = torch.as_tensor(times, device="cpu", dtype=torch.float64)
        if times.ndim != 1 or times.numel() == 0 or not torch.isfinite(times).all():
            raise ValueError("times must be a nonempty finite one-dimensional sequence")
        if times[0] != 0 or not torch.all(times[1:] > times[:-1]):
            raise ValueError("times must start at zero and be strictly increasing")
        if forcing is not None:
            expected_shape = (initial.shape[0], times.numel() - 1, *initial.shape[1:])
            if not isinstance(forcing, torch.Tensor) or forcing.shape != expected_shape:
                raise ValueError("forcing must have shape [batch, time - 1, 3, grid_size, grid_size, grid_size]")
            if forcing.dtype != self.dtype:
                raise TypeError(f"forcing dtype must be {self.dtype}")
            if forcing.device != self.device:
                raise ValueError(f"forcing device must be {self.device}")
            if not torch.isfinite(forcing).all():
                raise ValueError("forcing must be finite")
        spectrum = torch.fft.fftn(initial, dim=(-3, -2, -1), norm="forward")
        spectrum = self.leray_project(spectrum)
        velocity = torch.fft.ifftn(spectrum, dim=(-3, -2, -1), norm="forward").real
        snapshots = [velocity]
        current_time = 0.0
        for interval, target_time in enumerate(times[1:].tolist()):
            forcing_spectrum = None
            forcing_dt = math.inf
            if forcing is not None:
                forcing_spectrum = torch.fft.fftn(
                    forcing[:, interval], dim=(-3, -2, -1), norm="forward"
                )
                forcing_spectrum = self.leray_project(forcing_spectrum)
                acceleration = torch.fft.ifftn(
                    forcing_spectrum, dim=(-3, -2, -1), norm="forward"
                ).real.abs().sum(dim=1).amax().item()
                if acceleration:
                    forcing_dt = math.sqrt(cfl / (self.grid_size * acceleration))
            while current_time < target_time:
                speed = velocity.abs().sum(dim=1).amax().item()
                advection_dt = cfl / (self.grid_size * speed) if speed else math.inf
                dt = min(max_dt, advection_dt, self.diffusion_dt, forcing_dt, target_time - current_time)
                if current_time + dt <= current_time:
                    raise RuntimeError("time step is too small to advance the solution")
                spectrum = self._step(spectrum, dt, forcing_spectrum)
                if not torch.isfinite(spectrum).all():
                    raise RuntimeError("solution became nonfinite; reduce max_dt or initial amplitude")
                velocity = torch.fft.ifftn(spectrum, dim=(-3, -2, -1), norm="forward").real
                current_time = min(current_time + dt, target_time)
            snapshots.append(velocity)
        return torch.stack(snapshots, dim=1)
