import numpy as np
import matplotlib.pyplot as plt

## Set constants
alpha = 1
gamma = 10
V = 1
T = 1

## Function for finding F
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

## Function for finding mu
def mui(nx, ny, nz, i):
    # i = 0,1,2 for mux,muy,muz
    ni = [nx, ny, nz]
    ans = 1
    for j in range(3):
        if j == i:
            ans += np.log(alpha * ni[j])
        else:
            ans += gamma * ni[j]
    return T * ans

## Function for finding P
def find_P(nx, ny, nz):
    return T * (nx + ny + nz + gamma*(nx*ny + ny*nz + nz*nx))

## Define region of n to consider, and create arrays
num_points = 100
# nlow = 0.01
nlow = 0.25
# nhigh = 0.4
nhigh = 0.35
n_vals = np.linspace(nlow, nhigh, num_points)
F_vals = np.zeros(num_points)
nx_vals = np.zeros(num_points)
ny_vals = np.zeros(num_points)
nz_vals = np.zeros(num_points)
ni_vals = [nx_vals, ny_vals, nz_vals]

grid_size = 1000

## Find nx, ny, nz and F at equilibrium
for i in range(num_points):
    n = n_vals[i]
    nxy_range = np.linspace(n/1000, n/3, grid_size)
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

## Locate phase transition from instabillity
ddF_n = np.gradient(np.gradient(F_vals))
trans_ind = np.argwhere(ddF_n <= 0)
trans_start_ind = trans_ind[0]
trans_end_ind = trans_ind[-1]

n_trans_start = n_vals[trans_start_ind]
n_trans_end = n_vals[trans_end_ind]
pre_trans_ind = np.arange(trans_start_ind[0])
post_trans_ind = np.arange(trans_end_ind[0], num_points)

print(f"Phase transition starts at n = {n_trans_start}.")
print(f"Phase transition ends at n = {n_trans_end}.")

## Values before phase transition
n_pre = n_vals[pre_trans_ind]
nx_pre = nx_vals[pre_trans_ind]
ny_pre = ny_vals[pre_trans_ind]
nz_pre = nz_vals[pre_trans_ind]

## Values after phase transition
n_post = n_vals[post_trans_ind]
nx_post = nx_vals[post_trans_ind]
ny_post = ny_vals[post_trans_ind]
nz_post = nz_vals[post_trans_ind]

## Pressure
P_vals = find_P(nx_vals, ny_vals, nz_vals)

## Chemical potential
mux = mui(nx_vals, ny_vals, nz_vals, 0)
muy = mui(nx_vals, ny_vals, nz_vals, 1)
muz = mui(nx_vals, ny_vals, nz_vals, 2)
mu = mux + muy + muz

## Make plots
plt.figure()
plt.title("$F(n)$ at equilibrium")
plt.xlabel("$n=N/V$")
plt.ylabel("$F$")
plt.plot(n_vals, F_vals)
plt.axvline(n_trans_start, linestyle="dashed", color="black")
plt.axvline(n_trans_end, linestyle="dashed", color="black")
plt.savefig("Figures_3b/F_n.pdf")


plt.figure()
plt.title("Second derivative of $F(n)$ at equilibrium")
plt.xlabel("$n=N/V$")
plt.ylabel("$\\left(\\frac{d^2F}{dn^2}\\right)_{V,T}$")
plt.plot(n_vals, ddF_n)
plt.axhline(0, linestyle="dashed", color="black")
plt.axvline(n_trans_start, linestyle="dashed", color="black")
plt.axvline(n_trans_end, linestyle="dashed", color="black")
plt.savefig("Figures_3b/ddF_n.pdf")


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
plt.savefig("Figures_3b/ni_n.pdf")


# mu_start_ind = trans_start_ind[0]-20
mu_start_ind = 0
plt.figure()
plt.title("Chemical potential of each rod orientation")
plt.xlabel("$n$")
plt.ylabel("$\\mu_i$")
# plt.plot(n_vals[mu_start_ind:], mu[mu_start_ind:], linestyle="solid", color="blue", label="$\\mu$")
plt.plot(n_vals, mux, linestyle="solid", color="blue", label="$\\mu_x$")
plt.plot(n_vals, muy, linestyle="dashed", color="red", label="$\\mu_y$")
plt.plot(n_vals, muz, linestyle="solid", color="green", label="$\\mu_z$")
# plt.plot(n_vals[mu_start_ind:], mux[mu_start_ind:], linestyle="solid", color="blue", label="$\\mu_x$")
# plt.plot(n_vals[mu_start_ind:], muy[mu_start_ind:], linestyle="dashed", color="red", label="$\\mu_y$")
# plt.plot(n_vals[mu_start_ind:], muz[mu_start_ind:], linestyle="solid", color="green", label="$\\mu_z$")
plt.axvline(n_trans_start, linestyle="dashed", color="black")
plt.axvline(n_trans_end, linestyle="dashed", color="black")
plt.legend()
plt.savefig("Figures_3b/mui_n.pdf")


plt.figure()
plt.title("Pressure at equilibrium")
plt.xlabel("$n$")
plt.ylabel("$P$")
plt.plot(n_vals, P_vals, linestyle="solid", color="blue", label="$P$")
plt.axvline(n_trans_start, linestyle="dashed", color="black")
plt.axvline(n_trans_end, linestyle="dashed", color="black")
plt.legend()
plt.savefig("Figures_3b/P_n.pdf")


plt.figure()
plt.title("Chemical potential of each rod orientation")
plt.xlabel("$\\mu_i$")
plt.ylabel("$n$")
plt.plot(mux[mu_start_ind:], n_vals[mu_start_ind:], linestyle="solid", color="blue", label="$\\mu_x$")
plt.plot(muy[mu_start_ind:], n_vals[mu_start_ind:], linestyle="dashed", color="red", label="$\\mu_y$")
plt.plot(muz[mu_start_ind:], n_vals[mu_start_ind:], linestyle="solid", color="green", label="$\\mu_z$")
plt.axhline(n_trans_start, linestyle="dashed", color="black")
plt.axhline(n_trans_end, linestyle="dashed", color="black")
plt.legend()
plt.savefig("Figures_3b/n_mui.pdf")


G_new = F_vals + P_vals*V
plt.figure()
plt.title("Gibbs free energy")
plt.xlabel("$n$")
plt.ylabel("$G$")
plt.plot(n_vals, G_new)
plt.savefig("Figures_3b/G_n.pdf")