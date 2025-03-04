import numpy as np
import matplotlib.pyplot as plt

## Set constants
alpha = 1
gamma = 10
T = 1
N = 1


## Function for finding F
def find_F(Nx, Ny, Nz, V):
    Nvals = [Nx, Ny, Nz]
    Nilog = 0
    cross_terms = 0
    for i in range(3):
        Ni = Nvals[i]
        Nii = Nvals[(i+1) % 3]
        Niii = Nvals[(i+2) % 3]
        Nilog += Ni*np.log(alpha*Ni/V)
        cross_terms += Nii*Niii
    ans = T * (Nilog + gamma * cross_terms/V)
    return ans

## Define region of P and N to consider, and create arrays
num_points = 1000
Plow = 0.4
Phigh = 0.7
P_vals = np.linspace(Plow, Phigh, num_points)
V_vals = np.zeros(num_points)
F_vals = np.zeros(num_points)
G_vals = np.zeros(num_points)

Nx_vals = np.zeros(num_points)
Ny_vals = np.zeros(num_points)
Nz_vals = np.zeros(num_points)
Ni_vals = [Nx_vals, Ny_vals, Nz_vals]

grid_size = 100

for i in range(num_points):
    P = P_vals[i]
    Nxy_range = np.linspace(N/100, N/10, grid_size)
    Nx, Ny = np.meshgrid(Nxy_range, Nxy_range)
    Nz = N-Nx-Ny
    Nxyz = Nx*Ny + Ny*Nz + Nz*Nx
    V = T*N/(2*P) * np.sqrt(1 + 4*gamma*P/T * Nxyz/(N*N))
    Fi = find_F(Nx, Ny, Nz, V)
    Gi = Fi + P*V
    
    eq_ind = np.argmin(Gi)

    G_eq = Gi.flatten()[eq_ind]
    G_vals[i] = G_eq

    F_eq = Fi.flatten()[eq_ind]
    F_vals[i] = F_eq

    V_vals[i] = V.flatten()[eq_ind]

    Nx_eq = Nx.flatten()[eq_ind]
    Ny_eq = Ny.flatten()[eq_ind]
    Nz_eq = Nz.flatten()[eq_ind]

    Ni_vals[0][i] = Nx_eq
    Ni_vals[1][i] = Ny_eq
    Ni_vals[2][i] = Nz_eq

mu_vals = G_vals/N

## Make plots
plt.figure()
plt.title("$G(P)$ at equilibrium")
plt.xlabel("$P$")
plt.ylabel("$G$")
# plt.plot(V_vals, P_vals)
# plt.plot(V_vals, F_vals)
# plt.plot(V_vals, G_vals)
plt.plot(V_vals, np.gradient(np.gradient(G_vals)))
# plt.axvline(n_trans_start, linestyle="dashed", color="black")
# plt.axvline(n_trans_end, linestyle="dashed", color="black")
plt.savefig("Figures_3c/G_n.pdf")