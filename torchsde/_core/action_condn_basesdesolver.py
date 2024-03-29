from . import base_solver
from ..types import Scalar, Tensor, Dict, Tensors, Tuple
import torch
from . import adaptive_stepping
from . import better_abc
from . import interp
from .base_sde import BaseSDE
from .._brownian import BaseBrownian
from ..settings import NOISE_TYPES
import warnings
import torchcde


class ActionControlledSolver(base_solver.BaseSDESolver):

    def __init__(self, sde, actions, action_encode_net,states, **kwargs):
        self.actions = actions
        self.states = states
        self.action_encode_net = action_encode_net
        super(ActionControlledSolver, self).__init__(sde=sde, **kwargs)

    def integrate(self, y0: Tensor, ts: Tensor, extra0: Tensors) -> Tuple[Tensor, Tensors]:
        """Integrate along trajectory.

        Args:
            y0: Tensor of size (batch_size, d)
            ts: Tensor of size (T,).
            extra0: Any extra state for the solver.

        Returns:
            ys, where ys is a Tensor of size (T, batch_size, d).
            extra_solver_state, which is a tuple of Tensors of shape (T, ...), where ... is arbitrary and
                solver-dependent.
        """
        assert ts.shape[0] == self.actions.shape[0]+1, f'Shape of time horizon and actions do not match {ts.shape}{self.actions.shape}.'

        step_size = self.dt

        prev_t = curr_t = ts[0]
        encoded_z = self.action_encode_net(torch.cat((y0, self.actions[0], self.states[0]), dim=1))
        prev_y = curr_y = encoded_z
        curr_extra = extra0

        ys = [encoded_z]
        prev_error_ratio = None

        for i, out_t in enumerate(ts[1:]):
            while curr_t < out_t:
                next_t = min(curr_t + step_size, ts[-1])
                if self.adaptive:
                    # Take 1 full step.
                    next_y_full, _ = self.step(curr_t, next_t, curr_y, curr_extra)
                    # Take 2 half steps.
                    midpoint_t = 0.5 * (curr_t + next_t)
                    midpoint_y, midpoint_extra = self.step(curr_t, midpoint_t, curr_y, curr_extra)
                    next_y, next_extra = self.step(midpoint_t, next_t, midpoint_y, midpoint_extra)

                    # Estimate error based on difference between 1 full step and 2 half steps.
                    with torch.no_grad():
                        error_estimate = adaptive_stepping.compute_error(next_y_full, next_y, self.rtol, self.atol)
                        step_size, prev_error_ratio = adaptive_stepping.update_step_size(
                            error_estimate=error_estimate,
                            prev_step_size=step_size,
                            prev_error_ratio=prev_error_ratio
                        )

                    if step_size < self.dt_min:
                        warnings.warn("Hitting minimum allowed step size in adaptive time-stepping.")
                        step_size = self.dt_min
                        prev_error_ratio = None

                    # Accept step.
                    if error_estimate <= 1 or step_size <= self.dt_min:
                        prev_t, prev_y = curr_t, curr_y
                        curr_t, curr_y, curr_extra = next_t, next_y, next_extra
                else:
                    prev_t, prev_y = curr_t, curr_y
                    curr_y, curr_extra = self.step(curr_t, next_t, curr_y, curr_extra)
                    if i < len(self.actions)-1:
                        curr_y = torch.cat((curr_y, self.actions[i+1], self.states[i+1]), dim=1)
                    else:
                        curr_y = torch.cat((curr_y, torch.zeros_like(self.actions[0]),torch.zeros_like(self.states[0])), dim=1)
                    curr_y = self.action_encode_net(curr_y)
                    curr_t = next_t
            ys.append(interp.linear_interp(t0=prev_t, y0=prev_y, t1=curr_t, y1=curr_y, t=out_t))

        return torch.stack(ys, dim=0), curr_extra