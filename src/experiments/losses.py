import torch


def h1_loss(prediction, target):
    if prediction.shape != target.shape or prediction.ndim < 3 or prediction.numel() == 0:
        raise ValueError("Fields must have matching [batch, channels, *spatial] shapes")
    if prediction.dtype not in (torch.float32, torch.float64):
        raise ValueError("Fields must use float32 or float64")
    if prediction.dtype != target.dtype or prediction.device != target.device:
        raise ValueError("Fields must have the same dtype and device")
    shape = prediction.shape[2:]
    weight = prediction.new_ones(shape)
    for axis, size in enumerate(shape):
        frequencies = torch.fft.fftfreq(size, d=1 / size, device=prediction.device,
                                       dtype=prediction.dtype)
        view_shape = [1] * len(shape)
        view_shape[axis] = size
        weight = weight + (2 * torch.pi * frequencies).square().reshape(view_shape)
    spectrum = torch.fft.fftn(prediction - target, dim=tuple(range(2, prediction.ndim)),
                              norm="ortho")
    return (weight * spectrum.abs().square()).mean()
