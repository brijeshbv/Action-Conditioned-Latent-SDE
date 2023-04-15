import numpy as np
import os
import torch

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt


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


def load_mocap_data_many_walks(data_dir, t0=0.0, t1=2.0, dt=0.1, plot=True):
    from scipy.io import loadmat
    fname = os.path.join(data_dir, 'mocap35.mat')
    mocap_data = loadmat(fname)

    Xtest = mocap_data['Xtest']
    Ytest = dt * np.arange(0, Xtest.shape[1], dtype=np.float32)
    Ytest = np.tile(Ytest, [Xtest.shape[0], 1])
    Xval = mocap_data['Xval']
    Yval = dt * np.arange(0, Xval.shape[1], dtype=np.float32)
    Yval = np.tile(Yval, [Xval.shape[0], 1])
    Xtr = mocap_data['Xtr']
    Ytr = dt * np.arange(0, Xtr.shape[1], dtype=np.float32)
    Ytr = np.tile(Ytr, [Xtr.shape[0], 1])
    Xtr = np.transpose(Xtr, (1, 0, 2))
    Xtest = np.transpose(Xtest, (1, 0, 2))
    Xtr = Xtr[:50,:,:]
    Xtest = Xtest[:50, :, :]
    ts = torch.linspace(t0, t1, steps=Xtr.shape[0])
    return torch.tensor(Xtr, dtype=torch.float32), torch.tensor(Xtest, dtype=torch.float32), ts


if __name__ == "__main__":
    dt = 0.01
    xs, xs_test, ts = load_mocap_data_many_walks('../', t0=0.3, t1=2, dt=0.005)

    num_steps = (ts - ts[0]) / dt
    print(xs.shape, xs_test.shape, ts.shape)
