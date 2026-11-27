import unittest

import torch
import torch.nn.functional as F

from src.models.modules.mean_flow import mean_flow_loss, sample_mean_flow
from src.models.modules.policies import ActionDiffusionTransformerMoE


class MeanFlowTests(unittest.TestCase):
    def test_analytic_target_and_stopped_gradient(self):
        actions = torch.arange(12).reshape(2, 3, 2).float()
        for diagonal_fraction in (0.0, 1.0):
            with self.subTest(diagonal_fraction=diagonal_fraction):
                torch.manual_seed(17)
                velocity = torch.randn_like(actions) - actions
                endpoints = torch.rand(2, len(actions))
                r, t = endpoints.amin(0), endpoints.amax(0)
                if diagonal_fraction == 1.0:
                    r = t = endpoints[0]
                r, t = r[:, None, None], t[:, None, None]
                z = actions + t * velocity
                weight = torch.tensor(0.3, requires_grad=True)
                expected_pred = weight * (z + 2 * t + 3 * r)
                # dr/dt = 0. Both dz/dt and explicit time dependence contribute.
                target = velocity - (t - r) * weight.detach() * (velocity + 2)
                expected = F.mse_loss(expected_pred, target)
                expected_grad, = torch.autograd.grad(expected, weight)

                def field(z, r, t):
                    return weight * (z + 2 * t[:, None, None] + 3 * r[:, None, None])

                torch.manual_seed(17)
                loss = mean_flow_loss(field, actions, diagonal_fraction)
                grad, = torch.autograd.grad(loss, weight)
                torch.testing.assert_close(loss, expected)
                torch.testing.assert_close(grad, expected_grad)

    def test_robust_weights_are_per_example_and_detached(self):
        actions = torch.zeros(2, 2, 3)
        pred = torch.tensor([1., 100.], requires_grad=True)
        def field(z, r, t):
            return pred[:, None, None].expand_as(z)

        torch.manual_seed(9)
        velocity = torch.randn_like(actions)
        mse = (pred[:, None, None] - velocity).square().mean((1, 2))
        expected = (mse / (mse.detach() + 1e-3).sqrt()).mean()
        torch.manual_seed(9)
        loss, metrics = mean_flow_loss(field, actions, 0.5, 0.5, return_metrics=True)
        torch.testing.assert_close(loss, expected)
        torch.testing.assert_close(metrics["raw_mse"], mse.detach().mean())
        torch.testing.assert_close(torch.autograd.grad(loss, pred)[0], torch.autograd.grad(expected, pred)[0])
        self.assertEqual(metrics["correction_rms"].item(), 0)

    def test_sampling_interval_sign_and_endpoints(self):
        noise = torch.randn(2, 8, 7)
        for steps in (1, 4):
            calls = []
            def field(z, r, t):
                calls.append((r[0].item(), t[0].item()))
                return torch.full_like(z, 2)
            result = sample_mean_flow(field, noise, steps)
            torch.testing.assert_close(result, noise - 2)
            self.assertEqual(calls, [((i-1)/steps, i/steps) for i in range(steps, 0, -1)])

    def test_warm_start_preserves_released_velocity(self):
        kwargs = dict(action_dim=7, vlm_hidden_size=16, hidden_size=16, depth=1, num_heads=4)
        original = ActionDiffusionTransformerMoE(**kwargs)
        # The head's default final layer is zero. Use a nonconstant prediction.
        torch.nn.init.normal_(original.final_layer.linear.weight)
        extended = ActionDiffusionTransformerMoE(**kwargs, condition_on_interval=True)
        missing, unexpected = extended.load_state_dict(original.state_dict(), strict=False)
        self.assertEqual(missing, ['t_embedder.interval_proj.weight'])
        self.assertEqual(unexpected, [])
        z, t, cond = torch.randn(2, 8, 7), torch.rand(2) * 1000, [torch.randn(2, 4, 16)]
        torch.testing.assert_close(extended(z, t, cond, interval=t / 2), original(z, t, cond))


if __name__ == "__main__":
    unittest.main()
