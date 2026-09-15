import math

import pytest
import torch

from experiments.losses import h1_loss


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_constant_fields_use_mean_squared_error_normalization(dtype):
    values = torch.arange(6, dtype=dtype).reshape(2, 3, 1, 1, 1)
    prediction = values.expand(-1, -1, 6, 8, 10)
    target = torch.full_like(prediction, 0.25)
    expected = (prediction - target).square().mean()
    torch.testing.assert_close(h1_loss(prediction, target), expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_three_dimensional_mode_uses_physical_derivative_weights(dtype):
    axes = [torch.arange(size, dtype=dtype) / size for size in (8, 10, 12)]
    x, y, z = torch.meshgrid(*axes, indexing="ij")
    mode = torch.cos(2 * math.pi * (x + 2 * y + 3 * z))
    prediction = (0.4 + 1.3 * mode)[None, None]
    expected = 0.4**2 + 1.3**2 / 2 * (1 + 4 * math.pi**2 * 14)
    torch.testing.assert_close(h1_loss(prediction, torch.zeros_like(prediction)),
                               prediction.new_tensor(expected))


@pytest.mark.parametrize("shape", [(12,), (12, 8), (12, 8, 6)])
def test_loss_and_gradient_follow_an_analytic_mode(shape):
    axis = torch.arange(shape[0], dtype=torch.float64) / shape[0]
    mode = torch.sin(4 * math.pi * axis).reshape(-1, *([1] * (len(shape) - 1)))
    mode = mode.expand(shape)[None, None]
    amplitude = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    prediction = 0.4 + amplitude * mode
    target = torch.full_like(prediction, 0.25, requires_grad=True)
    loss = h1_loss(prediction, target)
    weight = 1 + 16 * math.pi**2
    torch.testing.assert_close(loss, 0.15**2 + amplitude.square() * weight / 2)
    amplitude_gradient, target_gradient = torch.autograd.grad(loss, (amplitude, target))
    torch.testing.assert_close(amplitude_gradient, amplitude * weight)
    expected_target_gradient = -2 * (0.15 + amplitude * weight * mode) / target.numel()
    torch.testing.assert_close(target_gradient, expected_target_gradient)


def test_equal_fields_have_zero_loss_and_gradient():
    prediction = torch.randn(2, 3, 4, 6, 8, dtype=torch.float64, requires_grad=True)
    loss = h1_loss(prediction, prediction.detach().clone())
    loss.backward()
    assert loss.item() == 0
    torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction))


@pytest.mark.parametrize("shape", [(2, 3), (2, 3, 0)])
def test_invalid_field_shapes_are_rejected(shape):
    field = torch.empty(shape)
    with pytest.raises(ValueError, match="shapes"):
        h1_loss(field, field)


def test_mismatched_shapes_and_dtypes_are_rejected():
    field = torch.zeros(2, 3, 4)
    with pytest.raises(ValueError, match="shapes"):
        h1_loss(field, field[:1])
    with pytest.raises(ValueError, match="same dtype"):
        h1_loss(field, field.double())
    with pytest.raises(ValueError, match="float32 or float64"):
        h1_loss(field.long(), field.long())
