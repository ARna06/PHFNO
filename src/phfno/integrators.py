import torch


def energy_gradient(energy, z, create_graph=None):
    if torch.is_inference_mode_enabled():
        raise RuntimeError("Energy gradients need autograd; use torch.no_grad(), not inference_mode()")
    if z.ndim != 2:
        raise ValueError("z must have shape [batch, coordinates]")
    if create_graph is None:
        create_graph = torch.is_grad_enabled()
    with torch.enable_grad():
        if not z.requires_grad:
            z = z.detach().requires_grad_(True)
        h = energy(z)
        if h.shape != z.shape[:1]:
            raise ValueError("energy(z) must return one scalar per batch item")
        if not h.requires_grad:
            return torch.zeros_like(z)
        gradient = torch.autograd.grad(
            h.sum(), z, create_graph=create_graph, allow_unused=True
        )[0]
        return torch.zeros_like(z) if gradient is None else gradient


def euler_step(rhs, z, u, dt):
    dt = torch.as_tensor(dt, device=z.device, dtype=z.dtype)
    if dt.ndim != 0 or not torch.isfinite(dt).item() or dt.item() <= 0:
        raise ValueError("dt must be a finite positive scalar")
    return z + dt * rhs(z, u)
