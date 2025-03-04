import numpy as np
import matplotlib.pyplot as plt

alpha = 1
gamma = 10
T = 1
N = 1000

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
        cross_terms += Nii*Niii/V
    ans = T * (Nilog + gamma * cross_terms)
    return ans

Plow = 0.5
Phigh = 5
num_points = 1000

P_vals = np.linspace(Plow, Phigh, num_points)
V_vals = np.zeros(num_points)
F_vals = np.zeros(num_points)
Nx_values = np.zeros(num_points)
Ny_values = np.zeros(num_points)
Nz_values = np.zeros(num_points)
Ni_vals = [Nx_values, Ny_values, Nz_values]

grid_size = 100
for i in range(num_points):
    P = P_vals[i]
    Nxy_range = np.linspace(N/1e10, N/3, grid_size)
    Nx, Ny = np.meshgrid(Nxy_range, Nxy_range)
    Nz = N-Nx-Ny
    Nxyz = Nx*Ny + Ny*Nz + Nz*Nx

    V = T*N/(2*P) * (1 + np.sqrt(1 + 4*P*gamma*Nxyz/(T*N*N)))
    
    Fi = find_F(Nx, Ny, Nz, V)
    eq_ind = np.argmin(Fi)
    F_eq = Fi.flatten()[eq_ind]
    F_vals[i] = F_eq

    Nx_eq = Nx.flatten()[eq_ind]
    Ny_eq = Ny.flatten()[eq_ind]
    Nz_eq = Nz.flatten()[eq_ind]

    Ni_vals[0][i] = Nx_eq
    Ni_vals[1][i] = Ny_eq
    Ni_vals[2][i] = Nz_eq
    
    V_eq = V.flatten()[eq_ind]
    V_vals[i] = V_eq



plt.figure()
plt.plot(P_vals, V_vals)
plt.savefig("Figures_3c/PV.pdf")