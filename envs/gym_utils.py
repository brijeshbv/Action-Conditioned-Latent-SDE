from stable_baselines3 import SAC
import os
import torch
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import gym
from envs.pseudo_gym import PseudoGym
from torch.utils.data import TensorDataset, DataLoader

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_obs_from_initial_state(x0, batch_size, steps):
    env = PseudoGym()
    model = SAC.load('envs/trained_envs/sac_hopper', device=device)
    buffer = np.array([], dtype=np.float32)
    action_buffer = np.array([], dtype=np.float32)
    for i in range(batch_size):
        env.set_internal_state(x0[i].detach().numpy())
        obs = env.get_obs()
        observations = np.array([obs], dtype=np.float32)
        actions = np.array([], dtype=np.float32)
        for j in range(steps - 1):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = compt_step(env, action)
            observations = np.vstack((observations, obs))
            if len(actions) == 0:
                actions = np.array([action], dtype=np.float32)
            else:
                actions = np.vstack((actions, action))
        if i == 0:
            buffer = np.array([observations])
            action_buffer = np.array([actions])
        else:
            buffer = np.append(buffer, [observations], axis=0)
            action_buffer = np.append(action_buffer, [actions], axis=0)
    data_mean = buffer.mean(axis=1)
    for i in range(buffer.shape[1]):
        buffer[:, i, :] = buffer[:, i, :] - data_mean
    buffer = np.transpose(buffer, (1, 0, 2))
    action_buffer = np.transpose(action_buffer, (1, 0, 2))
    return torch.tensor(buffer, dtype=torch.float32), torch.tensor(action_buffer, dtype=torch.float32)


def vis(xs, ts, img_path, num_samples=10):
    fig = plt.figure(figsize=(20, 9))
    gs = gridspec.GridSpec(1, 2)
    ax00 = fig.add_subplot(gs[0, 0], projection='3d')
    ax01 = fig.add_subplot(gs[0, 1], projection='3d')

    # Left plot: data.
    z1, z2, z3 = np.split(xs.cpu().numpy(), indices_or_sections=3, axis=-1)
    print(z1.shape)
    [ax00.plot(z1[:, i, 0], z2[:, i, 0], z3[:, i, 0]) for i in range(num_samples)]
    ax00.scatter(z1[:, :num_samples, 0], z2[:, :num_samples, 0], z3[:, :10, 0], marker='x')
    ax00.set_yticklabels([])
    ax00.set_xticklabels([])
    ax00.set_zticklabels([])
    ax00.set_xlabel('$z_1$', labelpad=0., fontsize=16)
    ax00.set_ylabel('$z_2$', labelpad=.5, fontsize=16)
    ax00.set_zlabel('$z_3$', labelpad=0., horizontalalignment='center', fontsize=16)
    ax00.set_title('Data', fontsize=20)
    xlim = ax00.get_xlim()
    ylim = ax00.get_ylim()
    zlim = ax00.get_zlim()

    plt.savefig(img_path)
    plt.close()


def plot_action_results(X, idx=0, show=False, fname='reconstructions.png'):
    tt = X.shape[1]
    D = np.ceil(X.shape[2]).astype(int)
    nrows = np.ceil(D).astype(int)
    plt.figure(2, figsize=(20, 40))
    for i in range(D):
        plt.subplot(nrows, 1, i + 1)
        plt.plot(range(0, tt), X[idx, :, i], 'r.-')
    plt.savefig(fname)
    if show is False:
        plt.close()


def compt_reset(env):
    obs = env.reset()
    if type(obs) is np.ndarray:
        return obs
    else:
        obs, extra = obs
        return obs


def compt_step(env, action):
    op = env.step(action)
    if len(op) == 4:
        return op
    else:
        obs, reward, done, info, extra = op
        return obs, reward, done, info


def get_encoded_env_samples(env, model_file, batch_size, steps, device, t0=0., t1=2.):
    env = gym.make(env)
    model = SAC.load(f'envs/trained_envs/{model_file}', device=device)
    data_buffer = np.array([], dtype=np.float32)
    action_buffer = np.array([], dtype=np.float32)
    obs = compt_reset(env)
    for i in range(batch_size):
        observations = np.array([obs], dtype=np.float32)
        actions = np.array([], dtype=np.float32)
        for j in range(steps - 1):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = compt_step(env, action)
            observations = np.vstack((observations, obs))
            if j == 0:
                actions = np.array([action], dtype=np.float32)
            else:
                actions = np.vstack((actions, action))
        if i == 0:
            data_buffer = np.array([observations])
            action_buffer = np.array([actions])
        else:
            data_buffer = np.append(data_buffer, [observations], axis=0)
            action_buffer = np.append(action_buffer, [actions], axis=0)
    ts = torch.linspace(t0, t1, steps=steps, device=device)
    ts = ts.repeat(data_buffer.shape[0], 1)
    data_mean = data_buffer.mean(axis=1)
    for i in range(data_buffer.shape[1]):
        data_buffer[:,i,:] = data_buffer[:,i,:] - data_mean
    print(data_buffer.shape)
    return torch.tensor(data_buffer, dtype=torch.float32), ts, torch.tensor(action_buffer, dtype=torch.float32)


def get_training_data(env, model_file, batch_size, steps, device, t0=0., t1=2., train_batch_size=8):
    xs, ts, a = get_encoded_env_samples(env, model_file, batch_size, steps, device, t0, t1)
    train_dataset = TensorDataset(xs, ts, a)
    data_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=0)
    return data_loader, xs.shape[-1], a.shape[-1]


def get_env_samples(env, model_file, batch_size, steps, device, t0=0., t1=2.):
    env = gym.make(env)
    model = SAC.load(model_file, device=device)
    data_buffer = np.array([], dtype=np.float32)

    for i in range(batch_size):
        obs = env.reset()
        observations = np.array([obs], dtype=np.float32)
        actions = np.array([], dtype=np.float32)
        for j in range(steps - 1):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            observations = np.vstack((observations, obs))
        if i == 0:
            data_buffer = np.array([observations])
        else:
            data_buffer = np.append(data_buffer, [observations], axis=0)
    ts = torch.linspace(t0, t1, steps=steps, device=device)
    data_buffer = np.transpose(data_buffer, (1, 0, 2))
    data_mean = data_buffer.mean(axis=0)
    for i in range(data_buffer.shape[0]):
        data_buffer[i] = data_buffer[i] - data_mean
    print(data_buffer.shape)
    return torch.tensor(data_buffer, dtype=torch.float32), ts

def plot_gym_results(X, Xrec, idx=0, show=False, fname='reconstructions.png'):
    tt = X.shape[1]
    D = np.ceil(X.shape[2]).astype(int)
    nrows = np.ceil(D / 3).astype(int)
    plt.figure(2, figsize=(20, 40))
    for i in range(D):
        plt.subplot(nrows, 3, i + 1)
        plt.plot(range(0, tt), X[idx, :, i], 'r.-')
        # plt.plot(range(0, tt), Xrec[idx, :, i], 'b.-')
    plt.savefig(fname)
    if show is False:
        plt.close()

if __name__ == "__main__":
    data_buffer, ts, actions = get_encoded_env_samples('Hopper-v2', 'sac_hopper', 16, 50, device)

    for i in range(10):
        plot_gym_results(data_buffer, None, i, True, "hopper-data")

