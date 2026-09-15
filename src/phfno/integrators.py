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


def gonzalez_gradient(energy, z, next_z):
    if z.ndim != 2 or next_z.shape != z.shape:
        raise ValueError("States must have matching shapes [batch, coordinates]")
    midpoint = (z + next_z) / 2
    gradient = energy_gradient(energy, midpoint)
    difference = next_z - z
    distance_squared = difference.square().sum(dim=-1, keepdim=True)
    scale = torch.maximum(z.square().sum(dim=-1, keepdim=True),
                          next_z.square().sum(dim=-1, keepdim=True)).clamp_min(1)
    small = distance_squared <= torch.finfo(z.dtype).eps * scale
    denominator = torch.where(small, torch.ones_like(distance_squared), distance_squared)
    remainder = (energy(next_z) - energy(z)).unsqueeze(-1) - (gradient * difference).sum(dim=-1, keepdim=True)
    correction = torch.where(small, torch.zeros_like(remainder), remainder / denominator)
    return gradient + correction * difference


def gonzalez_step(energy, factors, z, u, dt, max_iterations=50, rtol=None, atol=None):
    dt = torch.as_tensor(dt, device=z.device, dtype=z.dtype)
    if dt.shape == (z.shape[0],):
        dt = dt.unsqueeze(-1)
    if dt.ndim != 0 and dt.shape != (z.shape[0], 1):
        raise ValueError("dt must be a scalar or one value per batch item")
    if not torch.isfinite(dt).all() or not (dt > 0).all():
        raise ValueError("dt must be finite and positive")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    rtol = 10 * torch.finfo(z.dtype).eps if rtol is None else rtol
    atol = 10 * torch.finfo(z.dtype).eps if atol is None else atol
    if rtol < 0 or atol < 0:
        raise ValueError("Tolerances must be nonnegative")
    forcing = factors.apply_b(u)
    next_z = z
    for _ in range(max_iterations):
        effort = gonzalez_gradient(energy, z, next_z)
        candidate = z + dt * (factors.apply_j(effort) - factors.apply_r(effort) + forcing)
        residual = torch.linalg.vector_norm(candidate - next_z, dim=-1)
        scale = torch.maximum(torch.linalg.vector_norm(z, dim=-1),
                              torch.linalg.vector_norm(candidate, dim=-1))
        if torch.isfinite(candidate).all() and (residual <= atol + rtol * scale).all():
            with torch.no_grad():
                effort = gonzalez_gradient(energy, z, candidate)
                update = dt * (factors.apply_j(effort) - factors.apply_r(effort) + forcing)
                residual = torch.linalg.vector_norm(candidate - z - update, dim=-1)
            if (residual <= atol + rtol * scale).all():
                return candidate
        next_z = candidate
    raise RuntimeError(f"Gonzalez step did not converge after {max_iterations} iterations "
                       f"(residual {residual.max().item():.3e})")
