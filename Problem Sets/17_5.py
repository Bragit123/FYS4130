import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import k

V0 = 10
V1 = 100
n = 1000
V = np.linspace(V0, V1, n)

k = 1
N = 1
a = 1
b = 1

Tc = 8*a/(27*k*b)
Tl = Tc/1.5 # Less than Tc
Tg = Tc*1.5 # Greater than Tc

def find_P(T):
    P = N*k*T/(V-b*N) - a*N*N/(V*V)
    return P

def find_kT(T):
    kT_inv = N*k*T*V/(V-b*N)**2 - 2*a*N*N/(V*V)
    return 1/kT_inv

def find_alpha(T):
    alpha = N*k/(V-b*N) * find_kT(T)
    return alpha

cV = 3/2 * k

def find_cP(T):
    cP = cV + T*V*find_alpha(T)**2 / (N * find_kT(T))
    return cP

# Make arrays with elements [T<Tc, Tc, T>Tc]
P = []
kT = []
alpha = []
cP = []

for T in [Tl, Tc, Tg]:
    P.append(find_P(T))
    kT.append(find_kT(T))
    alpha.append(find_alpha(T))
    cP.append(find_cP(T))

funcs = [kT, alpha, cP]
func_names = ["kT", "alpha", "cP"]
T_names = ["less", "Tc", "greater"]
T_titles = ["$T<T_c$", "$T=T_c$", "$T>T_c$"]
colors = ["red", "green", "blue"]
styles = ["solid", "dashed", "dotted"]

## Plot
for i in range(3):
    plt.figure()
    plt.xlabel("P")
    plt.title(T_titles[i])
    for f in range(3):
        plt.loglog(P[i], funcs[f][i], color=colors[f], label=func_names[f])
    plt.legend()
    plt.savefig(f"17_5_{T_names[i]}.pdf")

for i in range(3):
    plt.figure()
    plt.xlabel("P")
    plt.title(T_titles[i])
    for f in range(3):
        plt.loglog(P[i], 1/funcs[f][i], color=colors[f], label=func_names[f]+"$^{-1}$")
    plt.legend()
    plt.savefig(f"17_5_{T_names[i]}_inv.pdf")
