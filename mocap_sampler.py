import numpy as np
import os
import torch

def load_mocap_data_many_walks(data_dir,t0=0.0, t1=2.0,dt=0.1,plot=True):
	from scipy.io import loadmat
	fname = os.path.join(data_dir, 'mocap35.mat')
	mocap_data = loadmat(fname)

	Xtest = mocap_data['Xtest']
	Ytest = dt*np.arange(0,Xtest.shape[1],dtype=np.float32)
	Ytest = np.tile(Ytest,[Xtest.shape[0],1])
	Xval  = mocap_data['Xval']
	Yval  = dt*np.arange(0,Xval.shape[1],dtype=np.float32)
	Yval  = np.tile(Yval,[Xval.shape[0],1])
	Xtr   = mocap_data['Xtr']
	Ytr   = dt*np.arange(0,Xtr.shape[1],dtype=np.float32)
	Ytr   = np.tile(Ytr,[Xtr.shape[0],1])
	Xtr = np.transpose(Xtr, (1, 0, 2))
	Xtest = np.transpose(Xtest, (1, 0, 2))
	ts = torch.linspace(t0, t1, steps=Xtr.shape[0])
	return torch.tensor(Xtr, dtype=torch.float32), torch.tensor(Xtest, dtype=torch.float32), ts


if __name__ == "__main__":
 dt = 0.01
 xs, xs_test, ts =    load_mocap_data_many_walks('./', t0=0.3, t1=2, dt = 0.005)

 num_steps = (ts - ts[0]) / dt
 print(ts)
 print(num_steps)
 print(xs.shape,xs_test.shape, ts.shape, xs.size(1))