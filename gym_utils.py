
from stable_baselines3 import SAC
import os
import torch
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import gym
from pseudo_gym import PseudoGym

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def render_mujoco(xs = None):
    env = PseudoGym()
    obs = env.reset()
    print(xs.shape)
    for i in range(xs.shape[0]):
        env.set_internal_state(xs[i])
        env.render()

def render_2_mujoco(xs_1 = None, xs_2 = None):
    env1 = PseudoGym()
    env2 = PseudoGym()
    obs = env1.reset()
    obs = env2.reset()
    print(xs_1.shape)
    for i in range(xs_1.shape[0]):
        env2.set_internal_state(xs_2[i])
        env2.render()
        env1.set_internal_state(xs_1[i])
        env1.render()
    env1.close()
    env2.close()

def get_obs_from_initial_state(x0, batch_size, steps):
    env = PseudoGym()
    model = SAC.load('sac_pendulum', device=device)
    buffer = np.array([], dtype=np.float32)
    for i in range(batch_size):
        env.set_internal_state(x0[i])
        obs = env.get_obs()
        observations = np.array([obs], dtype=np.float32)
        for j in range(steps - 1):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info, extra = env.step(action)
            observations = np.vstack((observations, obs))
        if i == 0:
            buffer = np.array([observations])
        else:
            buffer = np.append(buffer, [observations], axis=0)
    buffer = np.transpose(buffer, (1, 0, 2))
    return torch.tensor(buffer, dtype=torch.float32)

    
    
def vis(xs, ts, img_path, num_samples=10):
    fig = plt.figure(figsize=(20, 9))
    gs = gridspec.GridSpec(1, 2)
    ax00 = fig.add_subplot(gs[0, 0], projection='3d')
    ax01 = fig.add_subplot(gs[0, 1], projection='3d')

    # Left plot: data.
    z1, z2, z3 = np.split(xs.cpu().numpy(), indices_or_sections=3, axis=-1)
    print(z1.shape)
    [ax00.plot(z1[ :,i, 0], z2[ :,i, 0], z3[:,i, 0]) for i in range(num_samples)]
    ax00.scatter(z1[:,:num_samples, 0], z2[:,:num_samples ,0], z3[ :,:10,0], marker='x')
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

def get_env_samples(env, model_file, batch_size, steps, device, t0=0., t1=2.):
    env = gym.make(env)
    model = SAC.load(model_file, device=device)
    data_buffer = np.array([], dtype=np.float32)

    for i in range(batch_size):
        obs, extra = env.reset()
        observations = np.array([obs], dtype=np.float32)
        for j in range(steps - 1):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info, extra = env.step(action)
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




if __name__ == "__main__":
    data_buffer, ts = get_env_samples('HalfCheetah-v2', 'sac_HalfCheetah', 2, 500, device)
    data_buffer = np.transpose(data_buffer, (1, 0, 2))
    render_mujoco(data_buffer[1])
    print(data_buffer.shape)

    img_path = os.path.join("./test/", f'test.pdf')

