import numpy as np
import matplotlib.pyplot as plt

L = 16
N = L
T = 0.5 # Temperature in units of J. NB: Should be the same as in MC simulation.

def corr_func_anal(r):
    lam0 = np.exp(1/T) + 2
    lam1 = np.exp(1/T) - 1
    
    num = lam1**N + lam0**r * lam1**(N-r) + lam0**(N-r) * lam1**r
    den = lam0**N + 2 * lam1**N

    return num/den

# def corr_func_anal(r):
#     lam0 = np.exp(1/T) + 2
#     lam1 = np.exp(1/T) - 1

#     return (lam1/lam0)**r * 1/(1 + 2*(lam1/lam0)**N)

# rem, imm, m1, m2, m4 = np.loadtxt("results.txt", unpack=True)
r, reC, imC = np.loadtxt("results.txt", unpack=True)

C_anal = corr_func_anal(r)


# Analytic correlation
# def C_anal(r)

# m = complex(rem, imm)
# print(m)

plt.ylabel("$\\Re\\{ C(r) \\}$")
plt.xlabel("$r$")
plt.plot(r, reC, label="MC simulation")
plt.plot(r, imC, label="MC simulation")
# plt.plot(r, C_anal, label="Analytic")
plt.legend()
plt.savefig("corr_func.pdf")