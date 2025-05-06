# This code was created by looking at the skeleton code for C++, rewriting it into python, fixing it and
# generalizing to two dimensions. I then realized that the code was way too slow to be used for subproblems
# after 2b, so I had ChatGPT fix it by rewriting it using numba jit and other optimization methods.

import numpy as np
import matplotlib.pyplot as plt
from numba import njit

np.random.seed(100)

# Precompute neighbors for the lattice
def compute_neighbors(d, L):
    if d == 1:
        N = L
        neighbors = np.zeros((N, 2), dtype=np.int32)
        for i in range(N):
            neighbors[i, 0] = (i + 1) % L  # right
            neighbors[i, 1] = (i - 1 + L) % L  # left
    elif d == 2:
        N = L * L
        neighbors = np.zeros((N, 4), dtype=np.int32)
        for x in range(L):
            for y in range(L):
                i = x * L + y
                neighbors[i, 0] = ((x + 1) % L) * L + y     # right
                neighbors[i, 1] = x * L + (y + 1) % L       # up
                neighbors[i, 2] = ((x - 1 + L) % L) * L + y # left
                neighbors[i, 3] = x * L + (y - 1 + L) % L   # down
    else:
        raise ValueError("Only 1D and 2D supported")
    return neighbors

@njit
def flip_and_build_loop_jit(S, M_count, neighbors, P_connect, q, start_i, d):
    old_state = S[start_i]
    new_state = old_state
    while new_state == old_state:
        new_state = np.random.randint(0, q)

    stack = np.empty(len(S), dtype=np.int32)
    stack[0] = start_i
    stack_ptr = 1

    while stack_ptr > 0:
        stack_ptr -= 1
        i = stack[stack_ptr]

        if S[i] != old_state:
            continue

        S[i] = new_state
        M_count[old_state] -= 1
        M_count[new_state] += 1

        dir_arr = range(2) if d == 1 else range(4)
        for dir in dir_arr:
            j = neighbors[i, dir]
            if S[j] == old_state:
                if np.random.rand() < P_connect:
                    stack[stack_ptr] = j
                    stack_ptr += 1

@njit
def corr_func_anal(r_vals, T, N):
    lam0 = np.exp(1/T) + 2
    lam1 = np.exp(1/T) - 1
    result = np.empty(len(r_vals))
    den = lam0**N + 2 * lam1**N
    for idx in range(len(r_vals)):
        r = r_vals[idx]
        num = lam1**N + lam0**r * lam1**(N - r) + lam0**(N - r) * lam1**r
        result[idx] = num / den
    return result

def run_MC(d, L, T, q=3, n_clusters=10, n_steps_eq=10000, n_steps=50000):
    N = L if d == 1 else L * L
    S = np.zeros(N, dtype=np.int32)
    P_connect = 1 - np.exp(-1/T)
    neighbors = compute_neighbors(d, L)
    m_spin = np.array([
        1,
        np.exp(2*np.pi/3 * 1j),
        np.exp(4*np.pi/3 * 1j)
    ])
    M_count = np.zeros(q, dtype=np.int32)
    M_count[0] = N

    for _ in range(n_steps_eq):
        for _ in range(n_clusters):
            flip_and_build_loop_jit(S, M_count, neighbors, P_connect, q, np.random.randint(0, N), d)

    mt = np.zeros(n_steps, dtype=np.complex128)
    S_mean = np.zeros(n_steps)
    for t in range(n_steps):
        for _ in range(n_clusters):
            flip_and_build_loop_jit(S, M_count, neighbors, P_connect, q, np.random.randint(0, N), d)
        mt[t] = np.mean(m_spin[S])
        S_mean[t] = np.mean(S)

    m_mean = np.mean(mt)
    m_mean_square = np.mean(np.abs(mt)**2)
    m_mean_quad = np.mean(np.abs(mt)**4)

    return {
        "m_mean": m_mean,
        "m_mean_square": m_mean_square,
        "m_mean_quad": m_mean_quad,
        "S_mean": S_mean
    }

# --- Finite-size scaling study ---
d = 2
Ls = [8, 16, 32]
n_points = 40
T_min = 1e-2
T_max = 3.0
T_vals = np.linspace(T_min, T_max, n_points)

plt.figure()

for L in Ls:
    gamma_vals = np.zeros(n_points)
    for i, T in enumerate(T_vals):
        print(f"L = {L}, T index {i+1}/{n_points}", end="\r")
        results = run_MC(d, L, T)
        m2 = results["m_mean_square"]
        m4 = results["m_mean_quad"]
        gamma_vals[i] = m4 / (m2**2 + 1e-12)

    plt.plot(T_vals, gamma_vals, label=f"L={L}")

plt.title("Binder cumulant \u0393 vs. T for different system sizes")
plt.xlabel("$T / J$")
plt.ylabel("$\\Gamma$")
plt.legend()
plt.grid(True)
plt.savefig("gamma_crossing.pdf")
