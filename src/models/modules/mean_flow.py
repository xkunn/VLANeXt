"""MeanFlow for action chunks. Times run from clean actions (0) to noise (1)."""

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


def mean_flow_loss(field, actions, diagonal_fraction, weight_power=0.0, return_metrics=False):
    actions = actions.float()
    noise = torch.randn_like(actions)
    batch = actions.shape[0]
    endpoints = torch.rand(2, batch, device=actions.device)
    r, t = endpoints.amin(dim=0), endpoints.amax(dim=0)
    diagonal = torch.rand_like(t) < diagonal_fraction
    # Draw diagonal times uniformly as in the original Flow Matching training.
    t = torch.where(diagonal, endpoints[0], t)
    r = torch.where(diagonal, t, r)
    velocity = noise - actions
    noisy_actions = actions + t[:, None, None] * velocity

    # Differentiate in normalized time, including the head's timestep scaling.
    # The math attention backend supports forward AD for RoPE policy heads.
    with sdpa_kernel(SDPBackend.MATH):
        pred, derivative = torch.func.jvp(
            field, (noisy_actions, r, t), (velocity, torch.zeros_like(r), torch.ones_like(t))
        )
    correction = (t - r)[:, None, None] * derivative.detach().float()
    target = (velocity - correction).detach()
    mse = (pred.float() - target).square().flatten(1).mean(1)
    # MeanFlow Eq. 22: weight each example independently, outside the gradient.
    loss = (mse / (mse.detach() + 1e-3).pow(weight_power)).mean() if weight_power else mse.mean()
    if not return_metrics:
        return loss
    return loss, {
        "raw_mse": mse.detach().mean(),
        "correction_rms": correction.square().mean().sqrt(),
        "velocity_rms": velocity.detach().square().mean().sqrt(),
        "prediction_rms": pred.detach().float().square().mean().sqrt(),
    }


def sample_mean_flow(field, noise, steps):
    action = noise.float()
    for step in range(steps, 0, -1):
        t = action.new_full((action.shape[0],), step / steps)
        r = action.new_full((action.shape[0],), (step - 1) / steps)
        action = action - (t - r)[:, None, None] * field(action, r, t).float()
    return action
