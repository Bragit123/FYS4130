import numpy as np
import matplotlib.pyplot as plt

T = 0.25
N = 16

## Analytic correlation function
def corr_func_anal(r):
    lam0 = np.exp(1/T) + 2
    lam1 = np.exp(1/T) - 1
    
    num = lam1**N + lam0**r * lam1**(N-r) + lam0**(N-r) * lam1**r
    den = lam0**N + 2 * lam1**N

    return num/den

r_vals, corr_real, corr_imag = np.loadtxt("results.txt", unpack=True)

plt.plot(r_vals, corr_real, label="MC")
# plt.plot(r_vals, corr_func_anal(r_vals), label="Analytic")
plt.legend()
plt.savefig("wow.pdf")