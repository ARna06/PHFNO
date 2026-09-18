import math
from numbers import Integral

import torch


def _validate_dt(dt, z):
    dt = torch.as_tensor(dt, device=z.device, dtype=z.dtype)
    if dt.ndim == 0:
        valid_shape = True
    else:
        valid_shape = dt.shape == (z.shape[0],) or dt.shape == (z.shape[0], 1)
    if not valid_shape or not torch.isfinite(dt).all() or not (dt > 0).all():
        raise ValueError("dt must be a finite positive scalar or one value per batch item")
    if dt.ndim != 0:
        dt = dt.reshape((z.shape[0],) + (1,) * (z.ndim - 1))
    return dt


def energy_gradient(energy, z, create_graph=None):
    if torch.is_inference_mode_enabled():
        raise RuntimeError("Energy gradients need autograd; use torch.no_grad(), not inference_mode()")
    if z.ndim != 2:
        raise ValueError("z must have shape [batch, coordinates]")
    if create_graph is None:
        # Training differentiates through this gradient; evaluation only needs its value.
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
    dt = _validate_dt(dt, z)
    return z + dt * rhs(z, u)


def avf_step(rhs, z, u, dt, max_iterations=50, quadrature_points=4,
             rtol=1e-6, atol=1e-8):
    """Advance an ODE with an implicit Average Vector Field step.

    The line integral in the AVF method is evaluated with Gauss-Legendre
    quadrature, and the implicit endpoint is solved by fixed-point iteration.
    Use damped updates when ordinary fixed-point iteration is slow or unstable.
    Acceptance checks the original equation residual, not the damped update.
    Training differentiates the converged equation with an implicit adjoint
    solve, so state convergence cannot prematurely stop parameter sensitivities.
    This supports first-order training gradients, including the energy Hessian
    needed by PhFNO. Higher-order differentiation of the step is not supported.
    """
    dt = _validate_dt(dt, z)
    # A larger, configurable budget admits moderately contractive steps; this
    # remains a fixed-point solver and reports failure for noncontractive steps.
    if (not isinstance(max_iterations, Integral) or isinstance(max_iterations, bool)
            or max_iterations < 1 or quadrature_points not in (1, 2, 4)):
        raise ValueError("max_iterations must be positive and quadrature_points must be 1, 2, or 4")
    if not math.isfinite(rtol) or not math.isfinite(atol) or rtol < 0 or atol < 0:
        raise ValueError("rtol and atol must be finite and nonnegative")
    quadrature = {
        1: ([0.5], [1.0]),
        2: ([0.2113248654051871, 0.7886751345948129], [0.5, 0.5]),
        4: ([0.06943184420297371, 0.33000947820757187,
             0.6699905217924281, 0.9305681557970262],
            [0.1739274225687269, 0.3260725774312731,
             0.3260725774312731, 0.1739274225687269]),
    }
    nodes, weights = (
        torch.tensor(values, device=z.device, dtype=z.dtype)
        for values in quadrature[quadrature_points]
    )
    def norm(value):
        # Measure the entire state per sample, independently of its tensor layout.
        return torch.linalg.vector_norm(value.flatten(start_dim=1), dim=1)

    def fixed_point(endpoint):
        # Hold the control fixed and integrate the vector field along the segment.
        displacement = endpoint - z
        update = torch.zeros_like(z)
        for node, weight in zip(nodes, weights):
            update = update + weight * rhs(z + node * displacement, u)
        return z + dt * update

    def solve(mapping, initial, reference, label):
        current = initial
        previous_residual = None
        damping = 1.0
        for iteration in range(max_iterations):
            candidate = mapping(current)
            residual = norm(candidate - current)
            scale = torch.maximum(norm(reference), norm(current))
            if not (torch.isfinite(candidate).all() and torch.isfinite(residual).all()
                    and torch.isfinite(scale).all()):
                raise RuntimeError(f"{label} did not converge: nonfinite state or residual")
            converged = residual <= atol + rtol * scale
            if converged.all():
                return current
            if iteration == 8:
                damping = 0.5
            elif iteration > 8 and ((residual > previous_residual) & ~converged).any():
                damping = max(damping * 0.5, 1 / 64)
            previous_residual = residual
            current = current + damping * (candidate - current)
        raise RuntimeError(
            f"{label} did not converge after {max_iterations} iterations "
            f"(residual {residual.max().item():.3e})"
        )

    # Solve values without retaining every network evaluation in the graph.
    with torch.no_grad():
        predictor = z.detach() + dt * rhs(z.detach(), u)
        next_z = solve(fixed_point, predictor, z.detach(), "AVF step")
    if not torch.is_grad_enabled():
        return next_z

    endpoint = next_z.detach().requires_grad_(True)
    mapped = fixed_point(endpoint)
    if not mapped.requires_grad:
        return next_z

    def implicit_backward(gradient):
        # For y=G(y), solve (I-dG/dy)^T lambda=gradient with matrix-free
        # vector-Jacobian products. This converges sensitivities even at equilibrium.
        # Autograd may represent an unused output gradient as None.
        if gradient is None:
            return None
        if torch.is_grad_enabled():
            raise RuntimeError("AVF implicit backward supports first-order gradients only")
        # Normalize the right-hand side so absolute tolerances do not make the
        # learned sensitivity depend on arbitrary loss scaling or tiny loss values.
        gradient_scale = gradient.abs().amax()
        if gradient_scale == 0:
            return gradient
        gradient = gradient / gradient_scale
        def adjoint_map(adjoint):
            product = torch.autograd.grad(
                mapped, endpoint, adjoint, retain_graph=True, allow_unused=True,
            )[0]
            return gradient if product is None else gradient + product

        adjoint = solve(adjoint_map, gradient, gradient, "AVF backward solve")
        return adjoint * gradient_scale

    # Hook an internal proxy so Jacobian products do not reenter the hook and
    # local derivatives with respect to the returned state remain ordinary ones.
    proxy = mapped.clone()
    proxy.register_hook(implicit_backward)
    # Preserve the solved value exactly while differentiating through one map.
    return next_z + (proxy - proxy.detach())


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
