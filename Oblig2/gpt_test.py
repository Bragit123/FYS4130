import numpy as np
import matplotlib.pyplot as plt
from numba import jit

np.random.seed(100)

#### Initialization and Utility Functions

def initialize_model(d, L, T, q=3):
    model = {
        "d": d,
        "L": L,
        "T": T,
        "q": q,
        "P_connect": 1 - np.exp(-1 / T)
    }

    if d == 1:
        N = L
        indices = np.arange(N).reshape(N, 1)
        x_arr = indices[:, 0]
        y_arr = np.zeros(N, dtype=int)
    elif d == 2:
        N = L * L
        indices = np.arange(N).reshape(L, L)
        y_arr, x_arr = np.meshgrid(np.arange(L), np.arange(L))
        x_arr = x_arr.flatten()
        y_arr = y_arr.flatten()
    else:
        raise ValueError("Dimension d must be 1 or 2")

    S = np.zeros(N, dtype=int)
    M_count = np.zeros(q)
    M_count[0] = N

    m_spin = np.array([
        1,
        np.exp(2 * np.pi / 3 * 1j),
        np.exp(4 * np.pi / 3 * 1j)
    ])

    model.update({
        "N": N,
        "indices": indices,
        "x_arr": x_arr,
        "y_arr": y_arr,
        "S": S,
        "M_count": M_count,
        "m_spin": m_spin
    })

    return model

def neighbor_index(model, i, dir):
    x = model["x_arr"][i]
    y = model["y_arr"][i]
    L = model["L"]
    indices = model["indices"]

    if dir == 0:   # Right
        return indices[(x + 1) % L, y]
    elif dir == 1: # Up
        return indices[x, (y + 1) % L]
    elif dir == 2: # Left
        return indices[(x - 1 + L) % L, y]
    elif dir == 3: # Down
        return indices[x, (y - 1 + L) % L]
    else:
        raise ValueError("Invalid direction")

#### Monte Carlo Cluster Step

@jit(nopython=True)
def flip_and_build_loop(model, start_i):
    stack = [start_i]
    old_state = model["S"][start_i]
    new_state = (old_state + 1) % model["q"]

    while stack:
        i = stack.pop()

        if model["S"][i] != old_state:
            continue

        model["S"][i] = new_state
        model["M_count"][old_state] -= 1
        model["M_count"][new_state] += 1

        dir_arr = [0, 2] if model["d"] == 1 else [0, 1, 2, 3]

        for dir in dir_arr:
            j = neighbor_index(model, i, dir)
            if model["S"][j] == old_state and np.random.uniform() < model["P_connect"]:
                stack.append(j)

#### Correlation Function (Analytic)

def corr_func_anal(model, r):
    lam0 = np.exp(1 / model["T"]) + 2
    lam1 = np.exp(1 / model["T"]) - 1

    N = model["N"]
    r = np.array(r)

    num = lam1**N + lam0**r * lam1**(N - r) + lam0**(N - r) * lam1**r
    den = lam0**N + 2 * lam1**N

    return num / den

#### Full Monte Carlo Simulation

def run_MC(model, n_clusters=5, n_steps_eq=10000, n_steps=10000):
    N = model["N"]
    S = model["S"]
    m_spin = model["m_spin"]

    # Equilibration
    for _ in range(n_steps_eq):
        for _ in range(n_clusters):
            flip_and_build_loop(model, np.random.randint(0, N))

    # Measurements
    mit = np.zeros((N, n_steps), dtype=complex)
    for t in range(n_steps):
        for _ in range(n_clusters):
            flip_and_build_loop(model, np.random.randint(0, N))
        mit[:, t] = m_spin[S]

    mi_mean = np.mean(mit, axis=1)
    mt = np.sum(mit, axis=0)
    m_mean = np.mean(mt)

    m0t_mit = np.conj(mit[0, :]) * mit
    corr_func = np.mean(m0t_mit, axis=1) - np.conj(mi_mean[0]) * mi_mean

    return {
        "corr_func": corr_func,
        "m_mean": m_mean
    }

#### Problem 2b – 1D Simulations

d = 1
L = 16
T_vals = [0.25, 0.5]
T_text = ["T_0_25", "T_0_5"]

for i in range(len(T_vals)):
    T = T_vals[i]
    model = initialize_model(d, L, T)
    r_vals = np.arange(0, model["N"])
    corr_anal = corr_func_anal(model, r_vals)
    results = run_MC(model)
    corr_MC = results["corr_func"]

    plt.figure()
    plt.title(f"Correlation function for 1D spin chain with temperature $T={T}$J")
    plt.xlabel("$r$")
    plt.ylabel("$C(r)$")
    plt.plot(r_vals, corr_anal, "k", label="Analytic")
    plt.plot(r_vals, np.real(corr_MC), "or", label="MC simulation")
    plt.legend()
    plt.savefig(f"corr_func_1D_{T_text[i]}.pdf")

# #### Problem 2c – Uncomment for 2D Simulations

# d = 2
# n_points = 6
# T_vals = np.linspace(0.1, 10, n_points)
# m_mean_vals = np.zeros(n_points)

# for i in range(n_points):
#     model = initialize_model(d, L, T_vals[i])
#     results = run_MC(model)
#     m_mean_vals[i] = results["m_mean"]

# plt.figure()
# plt.title("Real part of average magnetization per site")
# plt.xlabel("$T [J]$")
# plt.ylabel("$\\text{Re}\\left(\\langle m \\rangle\\right)$")
# plt.plot(T_vals, np.real(m_mean_vals), "or")
# plt.savefig("m_mean_real.pdf")
