# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Train a latent SDE on data from a stochastic Lorenz attractor.

Reproduce the toy example in Section 7.2 of https://arxiv.org/pdf/2001.01328.pdf

To run this file, first run the following to install extra requirements:
pip install fire

To run, execute:
python -m examples.latent_sde_lorenz
"""
import logging
import os

import fire
from cde_utils import CDEFunc, CDEFuncPost
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchcde
import tqdm
from torch import nn
from torch import optim
from torch.distributions import Normal
from envs.gym_utils import get_encoded_env_samples

import torchsde

os.environ['CUDA_VISIBLE_DEVICES'] = '2'


class LinearScheduler(object):
    def __init__(self, iters, maxval=1.0):
        self._iters = max(1, iters)
        self._val = maxval / self._iters
        self._maxval = maxval

    def step(self):
        self._val = min(self._maxval, self._val + self._maxval / self._iters)

    @property
    def val(self):
        return self._val


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Encoder, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size)
        self.lin = nn.Linear(hidden_size, output_size)

    def forward(self, inp):
        out, _ = self.gru(inp)
        out = self.lin(out)
        return out


class LatentSDE(nn.Module):
    sde_type = "stratonovich"
    noise_type = "diagonal"

    def __init__(self, data_size, latent_size, action_size, context_size, hidden_size, actions):
        super(LatentSDE, self).__init__()
        # Encoder.
        self.encoder = Encoder(input_size=data_size, hidden_size=hidden_size, output_size=context_size)
        self.qz0_net = nn.Linear(context_size, latent_size + latent_size)
        self.actions = actions
        # Decoder.
        self.func_prior = CDEFunc(latent_size, hidden_size, action_size)
        self.func_post = CDEFuncPost(latent_size, context_size, hidden_size, action_size)
        # This needs to be an element-wise function for the SDE to satisfy diagonal noise.
        self.g_nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1, hidden_size),
                    nn.Softplus(),
                    nn.Linear(hidden_size, 1),
                    nn.Sigmoid()
                )
                for _ in range(latent_size)
            ]
        )
        self.projector = nn.Sequential(
            nn.Linear(latent_size, latent_size * 2),
            nn.Tanh(),
            nn.Linear(latent_size * 2, latent_size * 4),
            nn.Tanh(),
            nn.Linear(latent_size * 4, data_size))
        self.pz0_mean = nn.Parameter(torch.zeros(1, latent_size))
        self.pz0_logstd = nn.Parameter(torch.zeros(1, latent_size))

        self._ctx = None
        self.encode_actions = nn.Linear(latent_size + action_size, latent_size)

    def contextualize(self, ctx):
        self._ctx = ctx  # A tuple of tensors of sizes (T,), (T, batch_size, d).

    # posterior drift including f(x)*dX/dt
    def f(self, t, z):
        # control_gradient is of shape (..., input_channels)
        control_gradient = self.actions.derivative(t)
        ts, ctx = self._ctx
        i = min(torch.searchsorted(ts, t, right=True), len(ts) - 1)
        z_hat = torch.cat((z, ctx[i]), dim=1)
        out = self.func_post.prod(t, z_hat, control_gradient)
        return out

    # prior including f(x)*dX/dt
    def h(self, t, z):
        control_gradient = self.actions.derivative(t)

        out = self.func_prior.prod(t, z, control_gradient)
        return out

    def g(self, t, y):  # Diagonal diffusion.
        y = torch.split(y, split_size_or_sections=1, dim=1)
        out = [g_net_i(y_i) for (g_net_i, y_i) in zip(self.g_nets, y)]
        return torch.cat(out, dim=1)

    def forward(self, xs, ts, actions_est, noise_std, adjoint=False, method="reversible_heun"):
        # Contextualization is only needed for posterior inference.
        ctx = self.encoder(torch.flip(xs, dims=(1,)))
        ctx = torch.flip(ctx, dims=(1,))
        ctx = torch.transpose(ctx, 0, 1)
        self.contextualize((ts, ctx))

        qz0_mean, qz0_logstd = self.qz0_net(ctx[0]).chunk(chunks=2, dim=1)

        z0 = qz0_mean + qz0_logstd.exp() * torch.randn_like(qz0_mean)

        if adjoint:
            # Must use the argument `adjoint_params`, since `ctx` is not part of the input to `f`, `g`, and `h`.
            adjoint_params = (
                    (ctx,) +
                    tuple(self.func_post.parameters()) + tuple(self.g_nets.parameters()) + tuple(
                self.func_prior.parameters())
            )
            zs, log_ratio = torchsde.sdeint_adjoint(
                self, z0, ts, adjoint_params=adjoint_params, dt=0.2, logqp=True, method=method,
                adjoint_method='adjoint_reversible_heun')

        else:
            zs, log_ratio = torchsde.sdeint(self, z0, ts, dt=1e-2, logqp=True, method=method)

        _xs = self.projector(zs)
        _xs = torch.transpose(_xs, 0,1)
        xs_dist = Normal(loc=_xs, scale=noise_std)
        log_pxs = xs_dist.log_prob(xs).sum(dim=(1, 2)).mean(dim=0)
        qz0 = torch.distributions.Normal(loc=qz0_mean, scale=qz0_logstd.exp())
        pz0 = torch.distributions.Normal(loc=self.pz0_mean, scale=self.pz0_logstd.exp())
        logqp0 = torch.distributions.kl_divergence(qz0, pz0).sum(dim=0).mean(dim=0)
        logqp_path = log_ratio.sum(dim=1).mean(dim=0)
        return log_pxs, logqp0 + logqp_path

    @torch.no_grad()
    def sample(self, batch_size, ts, bm=None):
        eps = torch.randn(size=(batch_size, *self.pz0_mean.shape[1:]), device=self.pz0_mean.device)
        z0 = self.pz0_mean + self.pz0_logstd.exp() * eps
        zs = torchsde.sdeint(self, z0, ts, names={'drift': 'h'}, dt=0.2, bm=bm)
        zs = np.transpose(zs, (1, 0, 2))
        # Most of the time in ML, we don't sample the observation noise for visualization purposes.
        _xs = self.projector(zs)
        return _xs


def log_MSE(xs, ts, latent_sde, bm_vis, global_step, train_dir):
    xs_model = latent_sde.sample(batch_size=xs.size(0), ts=ts, bm=bm_vis).cpu().numpy()
    mse_loss = nn.MSELoss()
    with torch.no_grad():
        # xs_T = np.transpose(xs_model, (1, 0, 2))
        # xs_true = get_obs_from_initial_state(xs_T[:, 0, :], xs_T.shape[0], xs_T.shape[1])
        # print(xs_true.shape, xs_model.shape)
        loss = mse_loss(xs[0, :, :], torch.tensor(xs_model[0, :, :]))
        # xs_m_t = np.transpose(xs_model, (1, 0, 2))
        # xs_t = np.transpose(xs, (1, 0, 2))
        plot_gym_results(xs, xs_model, 0, False, f'{train_dir}/recon_{global_step:06d}')
    logging.info(f'current loss: {loss:.4f}, global_step: {global_step:06d},\n')


def plot_gym_results(X, Xrec, idx=0, show=False, fname='reconstructions.png'):
    tt = X.shape[1]
    D = np.ceil(X.shape[2]).astype(int)
    nrows = np.ceil(D / 3).astype(int)
    lag = X.shape[1] - Xrec.shape[1]
    plt.figure(2, figsize=(20, 40))
    for i in range(D):
        plt.subplot(nrows, 3, i + 1)
        plt.plot(range(0, tt), X[idx, :, i], 'r.-')
        plt.plot(range(lag, tt), Xrec[idx, :, i], 'b.-')
    plt.savefig(fname)
    if show is False:
        plt.close()


def main(
        batch_size=32,
        latent_size=8,
        context_size=64,
        hidden_size=128,
        lr_init=1e-3,
        t0=0.3,
        t1=50.,
        lr_gamma=0.997,
        num_iters=5000,
        kl_anneal_iters=400,
        pause_every=50,
        noise_std=0.01,
        dt=0.2,
        adjoint=True,
        train_dir='./dump/lorenz/',
        method="reversible_heun",
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"), filename=f'{train_dir}/log.txt')
    steps = 500
    xs, ts, actions = get_encoded_env_samples('Hopper-v2', 'sac_hopper', batch_size, steps, device)
    # actions_spline = np.transpose(actions, (1, 0, 2))
    coeffs = torchcde.natural_cubic_coeffs(actions, ts[:-1])
    action_est = torchcde.CubicSpline(coeffs)

    latent_sde = LatentSDE(
        data_size=xs.shape[-1],
        action_size=actions.shape[-1],
        latent_size=latent_size,
        context_size=context_size,
        hidden_size=hidden_size,
        actions=action_est
    ).to(device)
    optimizer = optim.Adam(params=latent_sde.parameters(), lr=lr_init)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=lr_gamma)
    kl_scheduler = LinearScheduler(iters=kl_anneal_iters)

    # Fix the same Brownian motion for visualization.
    bm_vis = torchsde.BrownianInterval(
        t0=t0, t1=t1, size=(batch_size, latent_size,), device=device, levy_area_approximation="space-time")
    log_MSE(xs, ts, latent_sde, bm_vis, 0, train_dir)

    for global_step in tqdm.tqdm(range(1, num_iters + 1)):
        latent_sde.zero_grad()
        log_pxs, log_ratio = latent_sde(xs, ts, action_est, noise_std, adjoint, method)
        loss = -log_pxs + log_ratio * kl_scheduler.val
        loss.backward()
        torch.nn.utils.clip_grad_norm(parameters=latent_sde.parameters(), max_norm=20, norm_type=2.0)
        optimizer.step()
        scheduler.step()
        kl_scheduler.step()

        if global_step % pause_every == 0:
            lr_now = optimizer.param_groups[0]['lr']
            logging.warning(
                f'global_step: {global_step:06d}, lr: {lr_now:.5f}, '
                f'log_pxs: {log_pxs:.4f}, log_ratio: {log_ratio:.4f} loss: {loss:.4f}, kl_coeff: {kl_scheduler.val:.4f}\n'
            )
            img_path = os.path.join(train_dir, f'global_step_{global_step:06d}.pdf')
            log_MSE(xs, ts, latent_sde, bm_vis, global_step, train_dir)


if __name__ == "__main__":
    fire.Fire(main)
