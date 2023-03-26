import numpy as np

arr = np.array([[[0,0],[1,1],[2,2]],[[0,0],[1,1],[2,2]],[[0,0],[1,1],[2,2]],[[0,0],[1,1],[2,2]]])
print(arr)
arr = np.transpose(arr, (1,0,2))
print(arr)