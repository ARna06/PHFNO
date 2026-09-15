import math

import pytest
import torch

from phfno.fourier import RealFourierCoordinates


@pytest.mark.parametrize("cutoff, grid", [((0,), (1,)), ((2,), (7,)), ((1, 2), (5, 7))])
def test_coordinate_roundtrip_and_parseval(cutoff, grid):
    transform = RealFourierCoordinates(cutoff, channels=2)
    generator = torch.Generator().manual_seed(1)
    z = torch.randn(3, transform.coordinate_dim, generator=generator, dtype=torch.float64)
    field = transform.decode(z, grid)
    torch.testing.assert_close(transform.encode(field), z)
    spatial_norm = field.square().flatten(start_dim=2).mean(dim=-1).sum(dim=-1)
    torch.testing.assert_close(spatial_norm, z.square().sum(dim=-1))


def test_mixed_sign_frequencies_and_resolution_change():
    transform = RealFourierCoordinates((2, 2), channels=1)
    encoded = []
    for grid in ((7, 9), (11, 13)):
        x, y = torch.meshgrid(
            torch.arange(grid[0], dtype=torch.float64) / grid[0],
            torch.arange(grid[1], dtype=torch.float64) / grid[1],
            indexing="ij",
        )
        field = (1.2 + 0.7 * torch.cos(2 * math.pi * (x - 2 * y))
                 - 0.3 * torch.sin(2 * math.pi * (2 * x + y)))[None, None]
        z = transform.encode(field)
        torch.testing.assert_close(transform.decode(z, grid), field)
        encoded.append(z)
    torch.testing.assert_close(encoded[0], encoded[1])


def test_projection_discards_unretained_frequencies():
    transform = RealFourierCoordinates((1,), channels=1)
    x = torch.arange(11, dtype=torch.float64) / 11
    low = (0.3 + torch.cos(2 * math.pi * x))[None, None]
    field = low + (0.8 * torch.sin(6 * math.pi * x))[None, None]
    torch.testing.assert_close(transform.project(field), low)
    torch.testing.assert_close(transform.project(transform.project(field)), low)


def test_differentiation_and_functional_gradient_scaling():
    transform = RealFourierCoordinates((1, 1), channels=2)
    generator = torch.Generator().manual_seed(2)
    z = torch.randn(1, transform.coordinate_dim, generator=generator, dtype=torch.float64)
    z.requires_grad_()
    assert torch.autograd.gradcheck(lambda value: transform.decode(value, (3, 3)), (z,))
    field = transform.decode(z, (3, 3))

    energy = 0.5 * field.square().flatten(start_dim=2).mean(dim=-1).sum()
    gradient = torch.autograd.grad(energy, z, create_graph=True)[0]
    torch.testing.assert_close(gradient, z)
    curvature = torch.autograd.grad(gradient.square().sum(), z)[0]
    torch.testing.assert_close(curvature, 2 * z)


def test_encode_supports_first_and_second_derivatives():
    transform = RealFourierCoordinates((1,), channels=1)
    field = torch.randn(1, 1, 5, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(transform.encode, (field,))
    assert torch.autograd.gradgradcheck(transform.encode, (field,))


def test_invalid_shapes_and_dtypes():
    with pytest.raises(ValueError, match="nonnegative integers"):
        RealFourierCoordinates((-1,), channels=1)
    with pytest.raises(ValueError, match="positive integer"):
        RealFourierCoordinates((1,), channels=0)
    transform = RealFourierCoordinates((2,), channels=1)
    with pytest.raises(ValueError, match="2 \\* cutoff"):
        transform.encode(torch.zeros(1, 1, 4))
    with pytest.raises(ValueError, match="positive integers"):
        transform.decode(torch.zeros(1, 5), (float("inf"),))
    with pytest.raises(ValueError, match="field must have shape"):
        transform.encode(torch.zeros(1, 2, 5))
    with pytest.raises(TypeError, match="real float32 or float64"):
        transform.encode(torch.zeros(1, 1, 5, dtype=torch.complex64))
