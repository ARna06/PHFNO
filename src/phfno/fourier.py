from itertools import product
from math import prod, sqrt
from numbers import Integral
from typing import Sequence

import torch
from torch import Tensor, nn


class RealFourierCoordinates(nn.Module):

    def __init__(self, cutoff: Sequence[int], channels: int):
        super().__init__()
        if not isinstance(channels, Integral) or isinstance(channels, bool) or channels < 1:
            raise ValueError("channels must be a positive integer")
        cutoff = tuple(cutoff)
        if not cutoff or any(
            not isinstance(n, Integral) or isinstance(n, bool) or n < 0
            for n in cutoff
        ):
            raise ValueError("cutoff must contain one or more nonnegative integers")
        self.cutoff = tuple(int(n) for n in cutoff)
        self.channels = int(channels)
        self.spatial_dim = len(self.cutoff)
        self.modes_per_channel = prod(2 * n + 1 for n in self.cutoff)
        self.coordinate_dim = self.channels * self.modes_per_channel
        # Real fields have conjugate Fourier pairs; store one representative per pair.
        representatives = [
            k
            for k in product(*(range(-n, n + 1) for n in self.cutoff))
            if any(k) and next(component for component in k if component) > 0
        ]
        self.register_buffer(
            "modes",
            torch.tensor(representatives, dtype=torch.long).reshape(-1, self.spatial_dim),
        )

    def _grid_shape(self, grid_shape: Sequence[int]) -> tuple[int, ...]:
        shape = tuple(grid_shape)
        if len(shape) != self.spatial_dim or any(
            not isinstance(n, Integral) or isinstance(n, bool) or n < 1 for n in shape
        ):
            raise ValueError(f"grid_shape must contain {self.spatial_dim} positive integers")
        if any(size <= 2 * cutoff for size, cutoff in zip(shape, self.cutoff)):
            raise ValueError("each grid side must be at least 2 * cutoff + 1")
        return tuple(int(n) for n in shape)

    @staticmethod
    def _check_dtype(tensor: Tensor) -> None:
        if tensor.dtype not in (torch.float32, torch.float64):
            raise TypeError("Fourier coordinates require real float32 or float64 tensors")

    def encode(self, field: Tensor) -> Tensor:
        self._check_dtype(field)
        if field.ndim != self.spatial_dim + 2 or field.shape[1] != self.channels:
            raise ValueError(
                f"field must have shape [batch, {self.channels}, "
                f"{self.spatial_dim} spatial dimensions]"
            )
        self._grid_shape(field.shape[2:])
        # Forward normalization and sqrt(2) packing make the coordinate squared
        # norm equal the spatial mean of the retained field's channel-summed energy.
        spectrum = torch.fft.fftn(field, dim=tuple(range(2, field.ndim)), norm="forward")
        zero = spectrum[(slice(None), slice(None), *((0,) * self.spatial_dim))].real
        modes = self.modes.to(device=field.device)
        selected = spectrum[(slice(None), slice(None), *modes.unbind(dim=1))]
        pairs = torch.stack((selected.real, selected.imag), dim=-1).flatten(start_dim=2)
        coordinates = torch.cat((zero.unsqueeze(-1), sqrt(2) * pairs), dim=-1)
        return coordinates.reshape(field.shape[0], self.coordinate_dim)

    def decode(self, coordinates: Tensor, grid_shape: Sequence[int]) -> Tensor:
        self._check_dtype(coordinates)
        if coordinates.ndim != 2 or coordinates.shape[1] != self.coordinate_dim:
            raise ValueError(f"coordinates must have shape [batch, {self.coordinate_dim}]")
        shape = self._grid_shape(grid_shape)
        packed = coordinates.reshape(coordinates.shape[0], self.channels, self.modes_per_channel)
        dtype = torch.complex128 if coordinates.dtype == torch.float64 else torch.complex64
        spectrum = torch.zeros(
            (coordinates.shape[0], self.channels, *shape),
            dtype=dtype,
            device=coordinates.device,
        )
        spectrum[(slice(None), slice(None), *((0,) * self.spatial_dim))] = packed[..., 0]
        pairs = packed[..., 1:].reshape(coordinates.shape[0], self.channels, -1, 2)
        coefficients = torch.complex(pairs[..., 0], pairs[..., 1]) / sqrt(2)
        modes = self.modes.to(device=coordinates.device)
        spectrum[(slice(None), slice(None), *modes.unbind(dim=1))] = coefficients
        # Restore the conjugate partners before reconstructing a real periodic field.
        spectrum[(slice(None), slice(None), *(-modes).unbind(dim=1))] = coefficients.conj()
        return torch.fft.ifftn(
            spectrum, dim=tuple(range(2, spectrum.ndim)), norm="forward"
        ).real

    def project(self, field: Tensor) -> Tensor:
        return self.decode(self.encode(field), field.shape[2:])
