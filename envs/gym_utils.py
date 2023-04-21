from stable_baselines3 import SAC
import os
import torch
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import gym
from envs.pseudo_gym import PseudoGym
from torch.utils.data import TensorDataset, DataLoader
from rl_zoo3.enjoy import get_encoded_env_samples, get_trained_model

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_obs_from_initial_state(x0, batch_size, steps):
    env = PseudoGym()
    model = SAC.load('envs/trained_envs/sac_hopper', device=device)
    buffer = np.array([], dtype=np.float32)
    action_buffer = np.array([], dtype=np.float32)
    for i in range(batch_size):
        env.set_internal_state(x0[i].cpu().detach().numpy())
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
    return torch.tensor(buffer, dtype=torch.float32).to(device), torch.tensor(action_buffer, dtype=torch.float32).to(device)



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


# def get_encoded_env_samples(env, model_file, batch_size, steps, device, t0=0., t1=2., reset_data=True):
#     env = gym.make(env)
#     model = SAC.load(f'envs/trained_envs/{model_file}', device=device)
#     data_buffer = np.array([], dtype=np.float32)
#     action_buffer = np.array([], dtype=np.float32)
#     obs = compt_reset(env)
#     for i in range(batch_size):
#         if reset_data:
#             obs = compt_reset(env)
#         observations = np.array([obs], dtype=np.float32)
#         actions = np.array([], dtype=np.float32)
#         for j in range(steps - 1):
#             action, _states = model.predict(obs, deterministic=True)
#             obs, reward, done, info = compt_step(env, action)
#             observations = np.vstack((observations, obs))
#             if j == 0:
#                 actions = np.array([action], dtype=np.float32)
#             else:
#                 actions = np.vstack((actions, action))
#         if i == 0:
#             data_buffer = np.array([observations])
#             action_buffer = np.array([actions])
#         else:
#             data_buffer = np.append(data_buffer, [observations], axis=0)
#             action_buffer = np.append(action_buffer, [actions], axis=0)
#     ts = torch.linspace(t0, t1, steps=steps, device=device)
#     ts = ts.repeat(data_buffer.shape[0], 1).to(device)
#     data_mean = data_buffer.mean(axis=1)
#     for i in range(data_buffer.shape[1]):
#         data_buffer[:, i, :] = data_buffer[:, i, :] - data_mean
#     print(data_buffer.shape)
#     return torch.tensor(data_buffer, dtype=torch.float32).to(device), ts, torch.tensor(action_buffer, dtype=torch.float32).to(device)


def get_training_data(batch_size, steps, device, t0=0., t1=2., train_batch_size=8, reset_data=True):
    xs, ts, a = get_encoded_env_samples( batch_size, steps, device, t0, t1, reset_data)
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
    data_buffer, ts, actions = get_encoded_env_samples('Hopper-v2', 'sac_hopper', 16, 300, device, reset_data=False)

    for i in range(10):
        plot_gym_results(data_buffer, None, i, True, "hopper-data")
