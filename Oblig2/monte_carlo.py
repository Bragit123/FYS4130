# This code was created by looking at the skeleton code for C++, rewriting it into python, fixing it and
# generalizing to two dimensions, and then have ChatGPT optimize it using numba jit (and some other
# optimization methods). The reason for this is that my own code was too slow to run many simulations
# for different temperatures and lattice sizes. I have added detailed comments to most of the code to
# make sure I understand the code in its entirety. To see the old, non-optimized version written by me,
# see old_monte_carlo.py. In particular numba jit does not work well on recursive functions, which is
# why the flip_and_build function has been rewritten using loops.

import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
from numba.typed import List

np.random.seed(100)

def compute_neighbors(d, L):
    ## This method computes the neighbors of each point on the lattice for more efficient lookup in the MC simulation
    """
    d = dimension of system (1D chain or 2D lattice)
    L = system size (number of points along each axis)
    """
    if d == 1:
        N = L
        neighbors = np.zeros((N, 2), dtype=np.int32)
        for i in range(N):
            neighbors[i, 0] = (i + 1) % L  # Right
            neighbors[i, 1] = (i - 1 + L) % L  # Left
    elif d == 2:
        N = L * L
        neighbors = np.zeros((N, 4), dtype=np.int32)
        for x in range(L):
            for y in range(L):
                i = x * L + y
                neighbors[i, 0] = ((x + 1) % L) * L + y # Right
                neighbors[i, 1] = x * L + (y + 1) % L # Up
                neighbors[i, 2] = ((x - 1 + L) % L) * L + y # Left
                neighbors[i, 3] = x * L + (y - 1 + L) % L # Down
    else:
        raise ValueError("Only 1D and 2D supported")
    
    return neighbors # 2D array where neighbors[i,:] gives the two/four neighbors of the i'th element


@njit
def flip_and_build_loop_jit(S, neighbors, P_connect, q, start_i, d):
    """
    S = array with spin state of each site
    neighbors = 2D array with neighbors of each site (should be calculated by compute_neighbors()
    P_connect = probability of connecting to a neighboring site with the same spin state (before flipping)
    q = Number of spin states (should be three for our model)
    start_i = index of the site where we want to start building a cluster
    d = dimension of the system (1D chain or 2D lattice)
    """
    old_state = S[start_i]
    new_state = (old_state + 1) % q # Increase the spin value by one (modulo )
    stack = np.empty(len(S), dtype=np.int32) # Initialize a "stack" that keeps track of which sites have been visited
    stack[0] = start_i
    stack_ptr = 1 # "pointer" to keep track of which site we are considering at any time

    while stack_ptr > 0: # Keep going through neighbors as long as we haven't exausted all neighbors
        stack_ptr -= 1
        i = stack[stack_ptr] # Determine which site we came from (that we are considering the neighbors of)

        if S[i] != old_state: # Skip neighbors with spins that are different from the site we started with
            continue

        S[i] = new_state

        dir_arr = range(2) if d == 1 else range(4) # Two neighbors if d=1, four if d=2
        for dir in dir_arr:
            # Go through each neighbor of the current site
            j = neighbors[i, dir]
            if S[j] == old_state: # Only consider neighbors with spin equal to the site we started with
                if np.random.rand() < P_connect: # Connect to the new site with a probability P_connect
                    stack[stack_ptr] = j # Add the site to the stack
                    stack_ptr += 1


## Analytic correlation function
def corr_func_anal(r, T, N):
    lam0 = np.exp(1/T) + 2
    lam1 = np.exp(1/T) - 1
    
    num = lam1**N + lam0**r * lam1**(N-r) + lam0**(N-r) * lam1**r
    den = lam0**N + 2 * lam1**N

    return num/den


def run_MC(d, L, T, q=3, n_clusters=5, n_steps_eq=10000, n_steps=10000):
    # Run Monte Carlo simulation
    """
    d = dimension of the system (1D chain or 2D lattice)
    L = system size (number of points along each axis)
    T = temperature (in units of J)
    q = Number of spin states (should be three for our model)
    n_clusters = number of clusters to consider at each iteration
    n_steps_eq = number of equilibrium steps
    n_steps = number of measurement steps
    """
    N = L if d == 1 else L * L # Find the number of sites N depending on system dimension
    S = np.zeros(N, dtype=np.int32) # Initialize an array containing the spin value at each site
    P_connect = 1 - np.exp(-1/T) # The connection probability
    neighbors = compute_neighbors(d, L) # Compute the neighbors of each site
    m_spin = np.array([
        1,
        np.exp(2*np.pi/3 * 1j),
        np.exp(4*np.pi/3 * 1j)
    ]) # The magnetization given the spin state

    # Reach equilibrium:
    for _ in range(n_steps_eq):
        for _ in range(n_clusters):
            flip_and_build_loop_jit(S, neighbors, P_connect, q, np.random.randint(0, N), d)

    mit = np.zeros((N, n_steps), dtype=np.complex128) # array containing the magnetization m_i of each site at each time step
    mt = np.zeros(n_steps, dtype=np.complex128) # Array containing the magnetization per site m at each time step
    for t in range(n_steps):
        for _ in range(n_clusters):
            flip_and_build_loop_jit(S, neighbors, P_connect, q, np.random.randint(0, N), d)
        mit[:, t] = m_spin[S]
        mt[t] = np.mean(m_spin[S])

    mi_mean = np.mean(mit, axis=1) # Mean magnetization at each step
    m0t_mit = np.conj(mit[0, :]) * mit # m_0* times m_i at each site i and time step
    corr_func = np.mean(m0t_mit, axis=1) - np.conj(mi_mean[0]) * mi_mean # Correlation function

    m_mean = np.mean(mt) # Mean magnetization per site over every time step
    m_mean_square = np.mean(np.abs(mt)**2) # Mean square magnetization
    m_mean_quad = np.mean(np.abs(mt)**4) # Mean quadruple magnetization

    # Return dictionary with results
    results = {
        "corr_func": corr_func,
        "m_mean": m_mean,
        "m_mean_square": m_mean_square,
        "m_mean_quad": m_mean_quad
    }
    return results


#### Problem 2b

# Set parameters
d = 1
L = 16
T_vals = [0.25, 0.5]
T_text = ["T_0_25", "T_0_5"]

# Run simulation for each T
for i, T in enumerate(T_vals):
    print(f"Problem 2b: {i}/{len(T_vals)}", end="\r")

    r_vals = np.arange(L if d == 1 else L*L) # Number of r values to consider depends on the dimension
    corr_anal = corr_func_anal(r_vals, T, len(r_vals)) # Compute the analytical correlation function
    results = run_MC(d, L, T) # Run the Monte Carlo simulation
    corr_MC = results["corr_func"]

    # Plot the correlation function
    plt.figure()
    plt.title(f"Correlation function for 1D spin chain with temperature $T={T:.2f}$J")
    plt.xlabel("$r$")
    plt.ylabel("$C(r)$")
    plt.plot(r_vals, corr_anal, "k", label="Analytic")
    plt.plot(r_vals, np.real(corr_MC), "or", label="MC simulation")
    plt.legend()
    plt.savefig(f"Figures/corr_func_1D_{T_text[i]}.pdf")


#### Problem 2c and 2d

# Set parameters
d = 2
L = 16
n_T = 100 # Number of T values to run MC simulation for
T_c = 0.9950 # Critical temperature given in problem
T_min = 0.1
T_max = 3
T_vals = np.linspace(T_min, T_max, n_T)

# Initialize arrays to store results
m_mean_vals = np.zeros(n_T, dtype=complex)
m_mean_square_vals = np.zeros(n_T)
m_mean_quad_vals = np.zeros(n_T)

for i in range(n_T):
    T = T_vals[i]
    print(f"Problem 2c-d: {i+1}/{n_T}", end="\r")
    results = run_MC(d, L, T, n_clusters=1, n_steps_eq=10000, n_steps=10000) # Run the Monte Carlo simulation
    m_mean_vals[i] = results["m_mean"]
    m_mean_square_vals[i] = results["m_mean_square"]
    m_mean_quad_vals[i] = results["m_mean_quad"]

# Plot average magnetization
plt.figure()
plt.title(f"Real part of average magnetization per site")
plt.xlabel("$T [J]$")
plt.ylabel("$\\text{Re}\\left(\\langle m \\rangle\\right)$")
plt.plot(T_vals, np.real(m_mean_vals), label="Magnetization")
plt.axvline(T_c, label="$T_c$", linestyle="dashed", color="black")
plt.legend()
plt.savefig(f"Figures/m_mean_real.pdf")

# Plot average square magnetization
plt.figure()
plt.title(f"Average square magnetization per site")
plt.xlabel("$T [J]$")
plt.ylabel("$\\langle |m|^2 \\rangle$")
plt.plot(T_vals, m_mean_square_vals, label="Square magnetization")
plt.axvline(T_c, label="$T_c$", linestyle="dashed", color="black")
plt.legend()
plt.savefig(f"Figures/m_mean_square.pdf")


#### Problem 2f

# Set parameters
d = 2
L_vals = np.array([8, 16, 32])
n_L = len(L_vals) # Number of L values to run MC simulation for
n_T = 100 # Number of T values to run MC simulation for
T_min = 0.1
T_max = 3
T_vals = np.linspace(T_min, T_max, n_T)

# Initialize array to store results for Gamma
gamma_vals = np.zeros((n_T, n_L))

for l in range(n_L):
    L = L_vals[l]

    # Initialize arrays for moments of the magnetization
    m_mean_square_vals = np.zeros(n_T)
    m_mean_quad_vals = np.zeros(n_T)

    for t in range(n_T):
        T = T_vals[t]
        print(f"Problem 2f: {l+1}/{n_L} | {t+1}/{n_T}", end="\r")
        results = run_MC(d, L, T, n_clusters=10, n_steps_eq=10000, n_steps=10000) # Run MC simulation
        m_mean_square_vals[t] = results["m_mean_square"]
        m_mean_quad_vals[t] = results["m_mean_quad"]
    
    gamma_vals[:, l] = m_mean_quad_vals / m_mean_square_vals**2 # Compute Gamma and add to the array

# Plot Gamma
plt.figure()
plt.title(f"Gamma")
plt.xlabel("$T [J]$")
plt.ylabel("$\\Gamma$")
for l in range(n_L):
    plt.plot(T_vals, gamma_vals[:,l], label=f"L = {L_vals[l]}")
plt.axvline(T_c, label="$T_c$", linestyle="dashed", color="black")
plt.legend()
plt.savefig(f"Figures/gamma.pdf")