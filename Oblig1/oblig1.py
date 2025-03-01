import numpy as np
import matplotlib.pyplot as plt
from numba import njit

alpha = 1
gamma = 10
V = 1
T = 1

# def find_F(Nx, Ny, Nz):
#     Nvals = [Nx, Ny, Nz]
#     Nilog = 0
#     cross_terms = 0
#     for i in range(3):
#         Ni = Nvals[i]
#         Nii = Nvals[(i+1) % 3]
#         Niii = Nvals[(i+2) % 3]
#         Nilog += Ni*np.log(alpha*Ni/V)
#         cross_terms += Nii*Niii
#     ans = T * (Nilog + gamma/V * cross_terms)
#     return ans

def find_F(nx, ny, nz):
    nvals = [nx, ny, nz]
    nilog = 0
    cross_terms = 0
    for i in range(3):
        ni = nvals[i]
        nii = nvals[(i+1) % 3]
        niii = nvals[(i+2) % 3]
        nilog += ni*np.log(alpha*ni)
        cross_terms += nii*niii
    ans = T*V * (nilog + gamma * cross_terms)
    return ans

num_points = 100
nlow = 0.25
nhigh = 0.35
n_vals = np.linspace(nlow, nhigh, num_points)
F_vals = np.zeros(num_points)
nx_vals = np.zeros(num_points)
ny_vals = np.zeros(num_points)
nz_vals = np.zeros(num_points)
ni_vals = [nx_vals, ny_vals, nz_vals]

grid_size = 1000

for i in range(num_points):
    n = n_vals[i]
    nxy_range = np.linspace(n/1000, n/10, grid_size)
    nx, ny = np.meshgrid(nxy_range, nxy_range)
    nz = n-nx-ny
    Fi = find_F(nx, ny, nz)
    eq_ind = np.argmin(Fi)
    F_eq = Fi.flatten()[eq_ind]
    F_vals[i] = F_eq
    
    nx_eq = nx.flatten()[eq_ind]
    ny_eq = ny.flatten()[eq_ind]
    nz_eq = nz.flatten()[eq_ind]

    ni_vals[0][i] = nx_eq
    ni_vals[1][i] = ny_eq
    ni_vals[2][i] = nz_eq
    # nx_vals[i] = nx_eq
    # ny_vals[i] = ny_eq
    # nz_vals[i] = nz_eq

# ddF = np.gradient(np.gradient(F))

# Locate phase transition from instabillity
ddF_n = np.gradient(np.gradient(F_vals))
trans_ind = np.argwhere(ddF_n <= 0)
trans_start_ind = trans_ind[0]
trans_end_ind = trans_ind[-1]

n_trans_start = n_vals[trans_start_ind]
n_trans_end = n_vals[trans_end_ind]
pre_trans_ind = np.arange(trans_start_ind[0])
post_trans_ind = np.arange(trans_end_ind[0], num_points)

# Values before phase transition
n_pre = n_vals[pre_trans_ind]
nx_pre = nx_vals[pre_trans_ind]
ny_pre = ny_vals[pre_trans_ind]
nz_pre = nz_vals[pre_trans_ind]

# Values after phase transition
n_post = n_vals[post_trans_ind]
nx_post = nx_vals[post_trans_ind]
ny_post = ny_vals[post_trans_ind]
nz_post = nz_vals[post_trans_ind]

plt.figure()
plt.title("$F(n)$ at equilibrium")
plt.xlabel("$n=N/V$")
plt.ylabel("$F$")
plt.plot(n_vals, F_vals)
# plt.plot(n_vals, np.gradient(F_vals), "g")
plt.savefig("F_n.pdf")

plt.figure()
plt.title("Second derivative of $F(n)$ at equilibrium")
plt.xlabel("$n=N/V$")
plt.ylabel("$\\left(\\frac{d^2F}{dn^2}\\right)_{V,T}$")
plt.plot(n_vals, ddF_n)
plt.axhline(0, linestyle="dashed", color="black")
plt.axvline(n_trans_start, linestyle="dashed", color="black")
plt.axvline(n_trans_end, linestyle="dashed", color="black")
plt.savefig("ddF_n.pdf")

plt.figure()
plt.title("Number of each rod $n_i$ at equilibrium")
plt.xlabel("$n$")
plt.ylabel("$n_i$")
plt.plot(n_vals, nx_vals, linestyle="solid", color="blue", label="$n_x$")
plt.plot(n_vals, ny_vals, linestyle="dashed", color="red", label="$n_y$")
plt.plot(n_vals, nz_vals, linestyle="solid", color="green", label="$n_z$")
plt.axvline(n_trans_start, linestyle="dashed", color="black")
plt.axvline(n_trans_end, linestyle="dashed", color="black")
plt.legend()
plt.savefig("ni_n.pdf")

# nz_sort_ind = np.argsort(nz_vals)
# dF_dnz_sort = np.gradient(F_vals[nz_sort_ind], nz_vals[nz_sort_ind])
# plt.figure()
# plt.title("Derivative of $F(n_z)$")
# plt.xlabel("$n_z$")
# plt.ylabel("$\\frac{dF}{dn_z}$")
# plt.plot(n_vals[nz_sort_ind], dF_dnz_sort, linestyle="solid", color="blue", label="$\\frac{dF}{dn_z}$")
# # plt.plot(n_vals, ny_vals, linestyle="dashed", color="red", label="$n_y$")
# # plt.plot(n_vals, nz_vals, linestyle="solid", color="green", label="$n_z$")
# plt.axvline(n_trans_start, linestyle="dashed", color="black")
# plt.axvline(n_trans_end, linestyle="dashed", color="black")
# plt.legend()
# plt.savefig("dF_nz.pdf")

nx_pre_mean = np.mean(nx_pre)
ny_pre_mean = np.mean(ny_pre)
nz_pre_mean = np.mean(nz_pre)


def mui(nx, ny, nz, i):
    ni = [nx, ny, nz]
    ans = 1
    for j in range(3):
        if j == i:
            ans += np.log(alpha * ni[j])
        else:
            ans += gamma * ni[j]
    return T * ans

mux = mui(nx_vals, ny_vals, nz_vals, 0)
muy = mui(nx_vals, ny_vals, nz_vals, 1)
muz = mui(nx_vals, ny_vals, nz_vals, 2)

plt.figure()
plt.title("Chemical potential of each rod orientation")
plt.xlabel("$n$")
plt.ylabel("$\\mu_i$")
plt.plot(n_vals, mux, linestyle="solid", color="blue", label="$\\mu_x$")
plt.plot(n_vals, muy, linestyle="dashed", color="red", label="$\\mu_y$")
plt.plot(n_vals, muz, linestyle="solid", color="green", label="$\\mu_z$")
plt.axvline(n_trans_start, linestyle="dashed", color="black")
plt.axvline(n_trans_end, linestyle="dashed", color="black")
plt.legend()
plt.savefig("mui_n.pdf")

def find_P(nx, ny, nz):
    return T * (nx + ny + nz + gamma*(nx*ny + ny*nz + nz*nx))

P_vals = find_P(nx_vals, ny_vals, nz_vals)

plt.figure()
plt.title("Pressure at equilibrium")
plt.xlabel("$n$")
plt.ylabel("$P$")
plt.plot(n_vals, P_vals, linestyle="solid", color="blue", label="$P$")
plt.axvline(n_trans_start, linestyle="dashed", color="black")
plt.axvline(n_trans_end, linestyle="dashed", color="black")
plt.legend()
plt.savefig("P_n.pdf")

plt.figure()
plt.title("Chemical potential of each rod orientation")
plt.xlabel("$\\mu_i$")
plt.ylabel("$n$")
plt.plot(mux, n_vals, linestyle="solid", color="blue", label="$\\mu_x$")
plt.plot(muy, n_vals, linestyle="dashed", color="red", label="$\\mu_y$")
plt.plot(muz, n_vals, linestyle="solid", color="green", label="$\\mu_z$")
plt.axhline(n_trans_start, linestyle="dashed", color="black")
plt.axhline(n_trans_end, linestyle="dashed", color="black")
plt.legend()
plt.savefig("n_mui.pdf")