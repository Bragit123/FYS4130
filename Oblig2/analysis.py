import numpy as np
import matplotlib.pyplot as plt

# rem, imm, m1, m2, m4 = np.loadtxt("results.txt", unpack=True)
data = np.loadtxt("results.txt", unpack=True, dtype=complex)

print(np.mean(data[0,:]))

# m = complex(rem, imm)
# print(m)