import numpy as np
import matplotlib.pyplot as plt

## Set constants
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

##Define region of P to consider, and create arrays
Plow = 1
Phigh = 5
num_points = 1000

P_vals = np.linspace(Plow, Phigh, num_points)
V_vals = np.zeros(num_points)
F_vals = np.zeros(num_points)
G_vals = np.zeros(num_points)
Nx_vals = np.zeros(num_points)
Ny_vals = np.zeros(num_points)
Nz_vals = np.zeros(num_points)
Ni_vals = [Nx_vals, Ny_vals, Nz_vals]

grid_size = 100

## Find values at equilibrium
for i in range(num_points):
    P = P_vals[i]
    Nxy_range = np.linspace(N/1e10, N/3, grid_size)
    Nx, Ny = np.meshgrid(Nxy_range, Nxy_range)
    Nz = N-Nx-Ny
    Nxyz = Nx*Ny + Ny*Nz + Nz*Nx

    V = T*N/(2*P) * (1 + np.sqrt(1 + 4*P*gamma*Nxyz/(T*N*N)))
    
    Fi = find_F(Nx, Ny, Nz, V)
    Gi = Fi + P*V
    eq_ind = np.argmin(Fi)
    
    F_eq = Fi.flatten()[eq_ind]
    F_vals[i] = F_eq
    G_eq = Gi.flatten()[eq_ind]
    G_vals[i] = G_eq

    Nx_eq = Nx.flatten()[eq_ind]
    Ny_eq = Ny.flatten()[eq_ind]
    Nz_eq = Nz.flatten()[eq_ind]

    Ni_vals[0][i] = Nx_eq
    Ni_vals[1][i] = Ny_eq
    Ni_vals[2][i] = Nz_eq
    
    V_eq = V.flatten()[eq_ind]
    V_vals[i] = V_eq

## Locate phase transition from instability
ddG_P = np.gradient(np.gradient(G_vals))
trans_ind = np.argwhere(ddG_P >= 0)
# ddF_n = np.gradient(np.gradient(F_vals))
# trans_ind = np.argwhere(ddF_n >= 0)
trans_start_ind = trans_ind[0]
trans_end_ind = trans_ind[-1]

P_trans_start = P_vals[trans_start_ind]
P_trans_end = P_vals[trans_end_ind]
pre_trans_ind = np.arange(trans_start_ind[0])
post_trans_ind = np.arange(trans_end_ind[0], num_points)

print(f"Phase transition starts at P = {P_trans_start}.")
print(f"Phase transition ends at P = {P_trans_end}.")

## Values before phase transition
P_pre = P_vals[pre_trans_ind]
Ni_pre = [Ni_vals[i][pre_trans_ind] for i in range(3)]

## Values after phase transition
P_post = P_vals[post_trans_ind]
Ni_post = [Ni_vals[i][post_trans_ind] for i in range(3)]

## Chemical potential
mux = G_vals/Nx_vals
muy = G_vals/Ny_vals
muz = G_vals/Nz_vals
mu = mux + muy + muz

## Make plots
plt.figure()
plt.plot(P_vals, V_vals)
plt.savefig("Figures_3c/P_V.pdf")

plt.figure()
plt.title("$G(P)$ at equilibrium")
plt.xlabel("$P$")
plt.ylabel("$G$")
plt.plot(P_vals, G_vals)
plt.axvline(P_trans_start, linestyle="dashed", color="black")
plt.axvline(P_trans_end, linestyle="dashed", color="black")
plt.savefig("Figures_3c/G_P.pdf")

plt.figure()
plt.title("Second derivative of $G(P)$ at equilibrium")
plt.xlabel("$P$")
plt.ylabel("$\\left(\\frac{d^2G}{dP^2}\\right)_{T,N}$")
plt.plot(P_vals, ddG_P)
plt.axvline(P_trans_start, linestyle="dashed", color="black")
plt.axvline(P_trans_end, linestyle="dashed", color="black")
plt.savefig("Figures_3c/ddG_P.pdf")

plt.figure()
plt.title("$PV$ diagram at equilibrium")
plt.xlabel("$P$")
plt.ylabel("$V$")
plt.plot(P_vals, V_vals)
plt.axvline(P_trans_start, linestyle="dashed", color="black")
plt.axvline(P_trans_end, linestyle="dashed", color="black")
plt.savefig("Figures_3c/P_V.pdf")

plt.figure()
plt.title("Concentration of each rod orientation $n_i$ at equilibrium")
plt.xlabel("$P$")
plt.ylabel("$\\frac{n_i}{n}$")
plt.plot(P_vals, Nx_vals/N, linestyle="solid", color="blue", label="$n_x$")
plt.plot(P_vals, Ny_vals/N, linestyle="dashed", color="red", label="$n_y$")
plt.plot(P_vals, Nz_vals/N, linestyle="solid", color="green", label="$n_z$")
plt.axvline(P_trans_start, linestyle="dashed", color="black")
plt.axvline(P_trans_end, linestyle="dashed", color="black")
plt.legend()
plt.savefig("Figures_3c/ni_P.pdf")

plt.figure()
plt.title("Chemical potential of each rod orientation")
plt.xlabel("$P$")
plt.ylabel("$\\mu_i$")
# plt.plot(P_vals, mu, linestyle="solid", color="blue", label="$\\mu$")
plt.plot(P_vals, mux, linestyle="solid", color="blue", label="$\\mu_x$")
plt.plot(P_vals, muy, linestyle="dashed", color="red", label="$\\mu_y$")
plt.plot(P_vals, muz, linestyle="solid", color="green", label="$\\mu_z$")
plt.axvline(P_trans_start, linestyle="dashed", color="black")
plt.axvline(P_trans_end, linestyle="dashed", color="black")
plt.legend()
plt.savefig("Figures_3c/mui_P.pdf")

