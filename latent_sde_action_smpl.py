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
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from torch import nn
from torch import optim
from torch.distributions import Normal
from envs.gym_utils import get_encoded_env_samples, get_training_data, get_obs_from_initial_state
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

    def __init__(self, data_size, latent_size, context_size, hidden_size, action_dim, t0=0,
                 skip_every=5,
                 t1=2, dt=0.01):
        super(LatentSDE, self).__init__()
        # Encoder.
        self.encoder = Encoder(input_size=data_size, hidden_size=hidden_size, output_size=context_size)
        self.qz0_net = nn.Linear(context_size, latent_size + latent_size)
        self.t0 = t0
        self.t1 = t1
        self.dt = dt
        self.skip_every = skip_every
        # Decoder.
        self.f_net = nn.Sequential(
            nn.Linear(latent_size + context_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, latent_size),
        )
        self.h_net = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, latent_size),
        )
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
            nn.Linear(latent_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, data_size))

        latent_and_action_size = latent_size + action_dim + data_size
        self.action_encode_net = nn.Sequential(
            nn.Linear(latent_and_action_size, latent_size))
        self.pz0_mean = nn.Parameter(torch.zeros(1, latent_size))
        self.pz0_logstd = nn.Parameter(torch.zeros(1, latent_size))

        self._ctx = None

    def contextualize(self, ctx):
        self._ctx = ctx  # A tuple of tensors of sizes (T,), (T, batch_size, d).

    def contextualize_time(self, i):
        self.ti = i

    def f(self, t, y):
        ctx = self._ctx
        return self.f_net(torch.cat((y, ctx[self.ti]), dim=1))

    def h(self, t, y):
        return self.h_net(y)

    def g(self, t, y):  # Diagonal diffusion.
        y = torch.split(y, split_size_or_sections=1, dim=1)
        out = [g_net_i(y_i) for (g_net_i, y_i) in zip(self.g_nets, y)]
        return torch.cat(out, dim=1)

    def forward(self, xs, ts, noise_std, adjoint=False, method="reversible_heun", actions=None):
        # Contextualization is only needed for posterior inference.
        ctx = self.encoder(torch.flip(xs, dims=(0,)))
        ctx = torch.flip(ctx, dims=(0,))
        self.contextualize(ctx)
        sampled_t = list(t for t in range(ts.shape[0] - 1) if t % self.skip_every == 0)
        qz0_mean, qz0_logstd = self.qz0_net(ctx[0]).chunk(chunks=2, dim=1)
        z0 = qz0_mean + qz0_logstd.exp() * torch.randn_like(qz0_mean)
        zs = torch.reshape(z0, (1, z0.shape[0], z0.shape[1]))
        t_horizon = torch.linspace(self.t0, self.t1, self.skip_every)
        xs_ = self.projector(zs[-1, :, :])
        predicted_xs = xs_.reshape(1, xs_.shape[0], xs_.shape[1])

        for i in sampled_t:
            self.contextualize_time(i)
            if i == 0:
                latent_and_data = torch.cat((zs[-1, :, :], actions[i, :, :], xs[0, :, :]), dim=1)
            elif i < ts.shape[0] - 1:
                latent_and_data = torch.cat((zs[-1, :, :], actions[i, :, :], predicted_xs[-1, :, :]), dim=1)
            else:
                latent_and_data = torch.cat((zs[-1, :, :], torch.zeros_like(actions[0]), predicted_xs[-1, :, :]),
                                            dim=1)
            z_encoded = self.action_encode_net(latent_and_data)
            if adjoint:
                # Must use the argument `adjoint_params`, since `ctx` is not part of the input to `f`, `g`, and `h`.
                adjoint_params = (
                        (ctx,) +
                        tuple(self.f_net.parameters()) + tuple(self.g_nets.parameters()) + tuple(
                    self.h_net.parameters())
                )
                z_pred, log_ratio = torchsde.sdeint_adjoint(
                    self, z_encoded, t_horizon, adjoint_params=adjoint_params, dt=self.dt, logqp=True, method=method,
                    adjoint_method='adjoint_reversible_heun', action_encode_net=self.action_encode_net,
                    states=xs)
                xs_ = self.projector(z_pred)
                # xs_ = xs_.reshape(1, xs_.shape[0], xs_.shape[1])
                predicted_xs = torch.cat((predicted_xs, xs_), dim=0)
                zs = torch.cat((zs, z_pred), dim=0)
                if i == 0:
                    cum_log_ratio = log_ratio
                else:
                    cum_log_ratio = torch.cat((cum_log_ratio, log_ratio), dim=0)
            else:
                zs, log_ratio = torchsde.sdeint(self, z_encoded, ts, dt=1e-2, logqp=True, method=method)
        predicted_xs = predicted_xs[:-1, :, :]
        xs_dist = Normal(loc=predicted_xs, scale=noise_std)
        log_pxs = xs_dist.log_prob(xs).sum(dim=(0, 2)).mean(dim=0)

        qz0 = torch.distributions.Normal(loc=qz0_mean, scale=qz0_logstd.exp())
        pz0 = torch.distributions.Normal(loc=self.pz0_mean, scale=self.pz0_logstd.exp())
        logqp0 = torch.distributions.kl_divergence(qz0, pz0).sum(dim=1).mean(dim=0)
        logqp_path = cum_log_ratio.sum(dim=0).mean(dim=0)
        return log_pxs, logqp0 + logqp_path

    @torch.no_grad()
    def sample_fromx0(self, x0, ts, bm=None, actions=None, zs=None):

        t_horizon = torch.linspace(self.t0, self.t1, 10)
        predicted_xs = x0.reshape(1, x0.shape[0], x0.shape[1])
        for i in range(ts.shape[0] - 1):
            if i == 0:
                latent_and_data = torch.cat((zs[-1, :, :], actions[i, :, :], x0), dim=1)
            elif i < ts.shape[0] - 1:
                latent_and_data = torch.cat((zs[-1, :, :], actions[i, :, :], predicted_xs[-1, :, :]), dim=1)
            else:
                latent_and_data = torch.cat((zs[-1, :, :], torch.zeros_like(actions[0]), predicted_xs[-1, :, :]),
                                            dim=1)
            z_encoded = self.action_encode_net(latent_and_data)
            z_pred = torchsde.sdeint(self, z_encoded, t_horizon, dt=self.dt, names={'drift': 'h'}, bm=bm)
            # Most of the time in ML, we don't sample the observation noise for visualization purposes.

            x0 = self.projector(z_pred[-1, :, :])
            x0 = x0.reshape(1, x0.shape[0], x0.shape[1])
            predicted_xs = torch.cat((predicted_xs, x0), dim=0)
            zs = torch.cat((zs, z_pred[-1, :, :].reshape(1, z_pred.shape[1], z_pred.shape[2])), dim=0)
        return predicted_xs

    @torch.no_grad()
    def sample(self, batch_size, ts, bm=None, actions=None, xs=None):
        eps = torch.randn(size=(batch_size, *self.pz0_mean.shape[1:]), device=self.pz0_mean.device)
        t_horizon = torch.linspace(0, 2, 10)
        z0 = self.pz0_mean + self.pz0_logstd.exp() * eps
        zs = torch.reshape(z0, (1, z0.shape[0], z0.shape[1]))
        xs_ = self.projector(zs[-1, :, :])
        predicted_xs = xs_.reshape(1, xs_.shape[0], xs_.shape[1])
        for i in range(ts.shape[0] - 1):
            if i == 0:
                latent_and_data = torch.cat((zs[-1, :, :], actions[i, :, :], xs[0, :, :]), dim=1)
            elif i < ts.shape[0] - 1:
                latent_and_data = torch.cat((zs[-1, :, :], actions[i, :, :], predicted_xs[-1, :, :]), dim=1)
            else:
                latent_and_data = torch.cat((zs[-1, :, :], torch.zeros_like(actions[0]), predicted_xs[-1, :, :]),
                                            dim=1)
            z_encoded = self.action_encode_net(latent_and_data)
            z_pred = torchsde.sdeint(self, z_encoded, t_horizon, dt=0.1, names={'drift': 'h'}, bm=bm)
            # Most of the time in ML, we don't sample the observation noise for visualization purposes.

            xs_ = self.projector(z_pred[-1, :, :])
            xs_ = xs_.reshape(1, xs_.shape[0], xs_.shape[1])
            predicted_xs = torch.cat((predicted_xs, xs_), dim=0)
            zs = torch.cat((zs, z_pred[-1, :, :].reshape(1, z_pred.shape[1], z_pred.shape[2])), dim=0)
        return predicted_xs


def log_MSE(xs, ts, latent_sde, bm_vis, global_step, train_dir, steps):
    eps = torch.randn(size=(xs.size(1), *latent_sde.pz0_mean.shape[1:]), device=latent_sde.pz0_mean.device)
    z0 = latent_sde.pz0_mean + latent_sde.pz0_logstd.exp() * eps
    z0 = torch.reshape(z0, (1, z0.shape[0], z0.shape[1]))
    x0 = latent_sde.projector(z0[-1, :, :])
    xs, actions = get_obs_from_initial_state(x0, xs.size(1), steps=steps)
    xs_model = latent_sde.sample_fromx0(x0=x0, ts=ts, bm=bm_vis, actions=actions, zs=z0).cpu().numpy()
    mse_loss = nn.MSELoss()
    with torch.no_grad():
        loss = mse_loss(xs[:, 0, :], torch.tensor(xs_model[:, 0, :]))
        xs_m_t = np.transpose(xs_model, (1, 0, 2))
        xs_t = np.transpose(xs, (1, 0, 2))
        plot_gym_results(xs_t, xs_m_t, 0, False, f'{train_dir}/recon_{global_step:06d}')
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
        batch_size=64,
        latent_size=8,
        context_size=64,
        hidden_size=128,
        lr_init=1e-3,
        t0=0,
        t1=10,
        lr_gamma=0.9997,
        num_iters=5000,
        kl_anneal_iters=700,
        pause_every=50,
        noise_std=0.01,
        skip_every=5,
        dt=0.2,
        train_batch_size=8,
        adjoint=True,
        train_dir='./dump/lorenz/',
        method="reversible_heun",
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"), filename=f'{train_dir}/log.txt')
    steps = 100
    dt = (t1 - t0) / 100
    train_data, data_dim, action_dim = get_training_data('Hopper-v2', 'sac_hopper', batch_size, steps, device, t0, t1,
                                                         train_batch_size=train_batch_size)

    latent_sde = LatentSDE(
        data_size=data_dim,
        latent_size=latent_size,
        context_size=context_size,
        hidden_size=hidden_size,
        action_dim=action_dim,
        skip_every=skip_every,
        t0=0,
        t1=2,
        dt=dt,
    ).to(device)
    optimizer = optim.Adam(params=latent_sde.parameters(), lr=lr_init)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=lr_gamma)
    kl_scheduler = LinearScheduler(iters=kl_anneal_iters)

    for global_step in tqdm.tqdm(range(1, num_iters + 1)):
        for i, batch in enumerate(train_data):
            xs, ts, actions = batch
            xs, actions, ts = torch.permute(xs, (1, 0, 2)), torch.permute(actions, (1, 0, 2)), torch.permute(ts, (1, 0))
            # Fix the same Brownian motion for visualization.
            bm_vis = torchsde.BrownianInterval(
                t0=t0, t1=t1, size=(xs.shape[1], latent_size), device=device, levy_area_approximation="space-time")
            if global_step == 1:
                log_MSE(xs, ts, latent_sde, bm_vis, global_step, train_dir, steps)
            latent_sde.zero_grad()
            log_pxs, log_ratio = latent_sde(xs, ts, noise_std, adjoint, method, actions)
            loss = -log_pxs + log_ratio * kl_scheduler.val
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(parameters=latent_sde.parameters(), max_norm=10, norm_type=2.0)
            optimizer.step()
            scheduler.step()
            kl_scheduler.step()

            if global_step % pause_every == 0 or global_step == 1:
                lr_now = optimizer.param_groups[0]['lr']
                logging.warning(
                    f'global_step: {global_step:06d}, lr: {lr_now:.5f}, '
                    f'log_pxs: {log_pxs:.4f}, log_ratio: {log_ratio:.4f} loss: {loss:.4f}, kl_coeff: {kl_scheduler.val:.4f}\n'
                )
                log_MSE(xs, ts, latent_sde, bm_vis, global_step, train_dir, steps)


if __name__ == "__main__":
    fire.Fire(main)
