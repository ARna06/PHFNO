import math

import torch


class PeriodicVorticity(torch.nn.Module):
    def __init__(self, grid_size, device="cpu", dtype=torch.float64):
        super().__init__()
        if isinstance(grid_size, bool) or not isinstance(grid_size, int) or grid_size < 4:
            raise ValueError("grid_size must be an integer of at least 4")
        if dtype not in (torch.float32, torch.float64):
            raise TypeError("dtype must be torch.float32 or torch.float64")
        self.grid_size = grid_size
        modes = torch.fft.fftfreq(grid_size, d=1 / grid_size, device=device, dtype=dtype).round()
        indices = torch.stack(torch.meshgrid(modes, modes, modes, indexing="ij"))
        k = 2 * math.pi * indices
        k2 = k.square().sum(dim=0)
        inverse_k2 = k2.clamp_min(1).reciprocal()
        inverse_k2[0, 0, 0] = 0
        self.register_buffer("k", k)
        self.register_buffer("k2", k2)
        self.register_buffer("inverse_k2", inverse_k2)
        self.register_buffer("nyquist_mask", (indices.abs() < grid_size / 2).all(dim=0))
        self.register_buffer("dealias_mask", (indices.abs() < grid_size / 3).all(dim=0))

    def _validate(self, field):
        if field.ndim < 4 or field.shape[-4:] != (3,) + (self.grid_size,) * 3:
            raise ValueError("field must have shape [..., 3, grid_size, grid_size, grid_size]")
        if field.dtype != self.k.dtype:
            raise TypeError(f"field dtype must be {self.k.dtype}")
        if field.device != self.k.device:
            raise ValueError(f"field device must be {self.k.device}")

    def _spectrum(self, field):
        self._validate(field)
        return torch.fft.fftn(field, dim=(-3, -2, -1), norm="forward")

    def _field(self, spectrum):
        return torch.fft.ifftn(spectrum, dim=(-3, -2, -1), norm="forward").real

    def _curl_spectrum(self, spectrum):
        x, y, z = spectrum.unbind(dim=-4)
        kx, ky, kz = self.k.unbind(dim=0)
        return 1j * torch.stack((ky * z - kz * y, kz * x - kx * z, kx * y - ky * x), dim=-4)

    def curl(self, field):
        return self._field(self._curl_spectrum(self._spectrum(field)) * self.nyquist_mask)

    def velocity(self, omega, mean=None):
        spectrum = self._curl_spectrum(self._spectrum(omega))
        velocity = self._field(spectrum * self.inverse_k2 * self.nyquist_mask)
        if mean is not None:
            mean = torch.as_tensor(mean, device=omega.device, dtype=omega.dtype)
            if mean.shape != omega.shape[:-3]:
                raise ValueError("mean must have shape omega.shape[:-3]")
            velocity = velocity + mean[..., None, None, None]
        return velocity

    def inverse_laplacian(self, field):
        return self._field(self._spectrum(field) * self.inverse_k2 * self.nyquist_mask)

    def kinetic_energy(self, omega):
        return 0.5 * (omega * self.inverse_laplacian(omega)).sum(dim=-4).mean(dim=(-3, -2, -1))

    def divergence(self, field):
        spectrum = self._spectrum(field)
        divergence = 1j * (self.k * spectrum).sum(dim=-4) * self.nyquist_mask
        return self._field(divergence)

    def project(self, omega):
        spectrum = self._spectrum(omega)
        parallel = (self.k * spectrum).sum(dim=-4, keepdim=True) * self.inverse_k2
        spectrum = (spectrum - self.k * parallel) * self.nyquist_mask * (self.k2 > 0)
        return self._field(spectrum)

    def apply_j(self, omega, phi):
        self._validate(omega)
        self._validate(phi)
        if omega.shape != phi.shape:
            raise ValueError("omega and phi must have the same shape")
        phi = self._field(self._spectrum(phi) * self.dealias_mask)
        product = torch.linalg.cross(self.curl(phi), omega, dim=-4)
        spectrum = self._spectrum(product) * self.dealias_mask
        return self._field(self._curl_spectrum(spectrum))

    def apply_r(self, phi, viscosity):
        if not math.isfinite(viscosity) or viscosity < 0:
            raise ValueError("viscosity must be finite and nonnegative")
        spectrum = self._spectrum(phi) * (viscosity * self.k2.square()) * self.nyquist_mask
        return self._field(spectrum)

    def reference_rhs(self, omega, forcing=None, viscosity=0, mean_velocity=None):
        omega = self._field(self._spectrum(self.project(omega)) * self.dealias_mask)
        phi = self.inverse_laplacian(omega)
        velocity = self.velocity(omega, mean=mean_velocity)
        product = torch.linalg.cross(velocity, omega, dim=-4)
        spectrum = self._spectrum(product) * self.dealias_mask
        rate = self._field(self._curl_spectrum(spectrum)) - self.apply_r(phi, viscosity)
        if forcing is not None:
            self._validate(forcing)
            if forcing.shape != omega.shape:
                raise ValueError("forcing must have the same shape as omega")
            spectrum = self._spectrum(forcing) * self.dealias_mask
            rate = rate + self._field(self._curl_spectrum(spectrum))
        return rate
