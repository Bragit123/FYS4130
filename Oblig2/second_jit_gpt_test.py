# This code was created by looking at the skeleton code for C++, rewriting it into python, fixing it and
# generalizing to two dimensions. I then realized that the code was way too slow to be used for subproblems
# after 2b, so I had ChatGPT fix it by rewriting it using numba jit and other optimization methods.

import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
from numba.typed import List

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

# d = 2
# L = 16
# neighbors = compute_neighbors(2, L)
# S = np.arange(L*L).reshape((L, L))
# print(S[11:14,7:10])
# print(neighbors[S[12,8]])


@njit
def flip_and_build_loop_jit(S, M_count, neighbors, P_connect, q, start_i, d):
    old_state = S[start_i]
    new_state = (old_state + 1) % q
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

def run_MC(d, L, T, q=3, n_clusters=5, n_steps_eq=10000, n_steps=10000):
    N = L if d == 1 else L * L
    S = np.zeros(N, dtype=np.int32)
    P_connect = 1 - np.exp(-1/T)
    neighbors = compute_neighbors(d, L)
    m_spin = np.array([
        1,
        np.exp(2*np.pi/3 * 1j),
        np.exp(-2*np.pi/3 * 1j)
        # np.exp(4*np.pi/3 * 1j)
    ])
    M_count = np.zeros(q, dtype=np.int32)
    M_count[0] = N

    for _ in range(n_steps_eq):
        for _ in range(n_clusters):
            flip_and_build_loop_jit(S, M_count, neighbors, P_connect, q, np.random.randint(0, N), d)

    mit = np.zeros((N, n_steps), dtype=np.complex128)
    mt = np.zeros(n_steps, dtype=np.complex128)
    S_mean = np.zeros(n_steps)
    for t in range(n_steps):
        for _ in range(n_clusters):
            flip_and_build_loop_jit(S, M_count, neighbors, P_connect, q, np.random.randint(0, N), d)
        mit[:, t] = m_spin[S]
        # mt[t] = np.sum(M_count * m_spin) / N
        mt[t] = np.mean(m_spin[S])
        S_mean[t] = np.mean(S)

        # mit[:, t] = m_spin[S]
        # mt[t] = np.sum(mit[:, t])

    mi_mean = np.mean(mit, axis=1)
    m0t_mit = np.conj(mit[0, :]) * mit
    corr_func = np.mean(m0t_mit, axis=1) - np.conj(mi_mean[0]) * mi_mean
    
    # mt = np.sum(mit, axis=0)
    # mt = np.mean(mit, axis=0)
    # m_mean = np.mean(mt)
    # m_mean = np.mean(mit)
    m_mean = np.mean(mt)
    m_mean_square = np.mean(np.abs(mt)**2)
    m_mean_quad = np.mean(np.abs(mt)**4)

    # m_mean = np.mean(mt)
    # m_abs2_mean = np.mean(np.abs(mt)**2)
    # m_abs4_mean = np.mean(np.abs(mt)**4)

    return {
        "corr_func": corr_func,
        "m_mean": m_mean,
        "m_mean_square": m_mean_square,
        "m_mean_quad": m_mean_quad,
        "S_mean": S_mean
    }


# --- Problem 2b ---
d = 1
L = 16
T_vals = [0.25, 0.5]
T_text = ["T_0_25", "T_0_5"]

for i, T in enumerate(T_vals):
    print(f"Problem 2b: {i}/{len(T_vals)}", end="\r")
    r_vals = np.arange(L if d == 1 else L*L)
    corr_anal = corr_func_anal(r_vals, T, len(r_vals))
    results = run_MC(d, L, T)
    corr_MC = results["corr_func"]

    plt.figure()
    plt.title(f"Correlation function for 1D spin chain with temperature $T={T:.2f}$J")
    plt.xlabel("$r$")
    plt.ylabel("$C(r)$")
    plt.plot(r_vals, corr_anal, "k", label="Analytic")
    plt.plot(r_vals, np.real(corr_MC), "or", label="MC simulation")
    plt.legend()
    plt.savefig(f"Figures/corr_func_1D_{T_text[i]}.pdf")


#### --- Problem 2c and 2d --- ####
d = 2
L = 16
n_points = 100 # Number of data points to run MC simulation for
# T_min = 1e-5
# T_max = 1e4
# T_vals = np.linspace(T_min, T_max, n_points)
# T_vals = np.logspace(np.log10(T_min), np.log10(T_max), n_points)
T_c = 0.9950
T_min = 0.1
T_max = 3
T_vals = np.linspace(T_min, T_max, n_points)
m_mean_vals = np.zeros(n_points, dtype=complex)
m_mean_square_vals = np.zeros(n_points)
m_mean_quad_vals = np.zeros(n_points)
S_vals = np.zeros((n_points, 20000))

for i in range(n_points):
    T = T_vals[i]
    print(f"Problem 2c-d: {i+1}/{n_points}", end="\r")
    results = run_MC(d, L, T, n_clusters=1, n_steps_eq=10000, n_steps=10000)
    m_mean_vals[i] = results["m_mean"]
    m_mean_square_vals[i] = results["m_mean_square"]
    m_mean_quad_vals[i] = results["m_mean_quad"]
    # S_vals[i,:] = results["S_mean"]

# plt.figure()
# plt.plot(np.arange(20000), S_vals[0,:], ".")
# plt.savefig("Figures/S_vals.pdf")

plt.figure()
plt.title(f"Real part of average magnetization per site")
plt.xlabel("$T [J]$")
plt.ylabel("$\\text{Re}\\left(\\langle m \\rangle\\right)$")
plt.plot(T_vals, np.real(m_mean_vals), label="Magnetization")
plt.axvline(T_c, label="$T_c$", linestyle="dashed", color="black")
plt.legend()
plt.savefig(f"Figures/m_mean_real.pdf")

plt.figure()
plt.title(f"Average square magnetization per site")
plt.xlabel("$T [J]$")
plt.ylabel("$\\langle |m|^2 \\rangle$")
plt.plot(T_vals, m_mean_square_vals, label="Square magnetization")
plt.axvline(T_c, label="$T_c$", linestyle="dashed", color="black")
plt.legend()
plt.savefig(f"Figures/m_mean_square.pdf")

# plt.figure()
# plt.title(f"Average quadruple magnetization per site")
# plt.xscale("log")
# plt.xlabel("$T [J]$")
# plt.ylabel("$\\langle |m|^4 \\rangle$")
# plt.plot(T_vals, m_mean_quad_vals, label="Quadruple magnetization")
# plt.axvline(T_c, label="$T_c$", linestyle="dashed", color="black")
# plt.legend()
# plt.savefig(f"m_mean_quad.pdf")

# eps = 1e-12
# gamma = m_mean_quad_vals / (m_mean_square_vals**2 + eps)

# plt.figure()
# plt.title(f"Gamma")
# plt.xscale("log")
# plt.xlabel("$T [J]$")
# plt.ylabel("$\\Gamma$")
# plt.plot(T_vals, gamma/T_vals, "k", label="$\\Gamma$")
# plt.legend()
# plt.savefig(f"gamma.pdf")




#### --- Problem 2f --- ####
d = 2
L_vals = np.array([8, 16, 32])
n_T = 100 # Number of data points to run MC simulation for
T_min = 0.1
T_max = 3
n_L = len(L_vals)
T_vals = np.linspace(T_min, T_max, n_T)
# T_vals = np.logspace(np.log10(T_min), np.log10(T_max), n_T)
gamma_vals = np.zeros((n_T, n_L))

for l in range(n_L):
    m_mean_square_vals = np.zeros(n_T)
    m_mean_quad_vals = np.zeros(n_T)
    L = L_vals[l]
    for t in range(n_T):
        T = T_vals[t]
        print(f"Problem 2f: {l+1}/{n_L} | {t+1}/{n_T}", end="\r")
        results = run_MC(d, L, T, n_clusters=10, n_steps_eq=10000, n_steps=10000)
        m_mean_square_vals[t] = results["m_mean_square"]
        m_mean_quad_vals[t] = results["m_mean_quad"]
    gamma_vals[:, l] = m_mean_quad_vals / m_mean_square_vals**2

plt.figure()
plt.title(f"Gamma")
plt.xlabel("$T [J]$")
plt.ylabel("$\\Gamma$")
for l in range(n_L):
    plt.plot(T_vals, gamma_vals[:,l], label=f"L = {L_vals[l]}")
plt.axvline(T_c, label="$T_c$", linestyle="dashed", color="black")
plt.legend()
plt.savefig(f"Figures/gamma.pdf")