import numpy as np

l = np.load("losses_onlineae.npy")
l = list(np.around(l, decimals=3))
print(l)
