import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from numba.typed import List

np.random.seed(100)

#### --- JIT-compatible functions --- ####

@njit
def neighbor_index(i, dir, x_arr, y_arr, indices, L):
    x = x_arr[i]
    y = y_arr[i]

    if dir == 0:
        return indices[(x + 1) % L, y]
    elif dir == 1:
        return indices[x, (y + 1) % L]
    elif dir == 2:
        return indices[(x - 1 + L) % L, y]
    elif dir == 3:
        return indices[x, (y - 1 + L) % L]
    else:
        return -1

@njit
def flip_and_build_loop_jit(start_i, S, M_count, q, d, P_connect, x_arr, y_arr, indices, L):
    stack = List()
    stack.append(start_i)

    old_state = S[start_i]
    new_state = (old_state + 1) % q

    while len(stack) > 0:
        i = stack.pop()

        if S[i] != old_state:
            continue

        S[i] = new_state
        M_count[old_state] -= 1
        M_count[new_state] += 1

        dir_arr = [0, 2] if d == 1 else [0, 1, 2, 3]

        for dir in dir_arr:
            j = neighbor_index(i, dir, x_arr, y_arr, indices, L)
            if S[j] == old_state and np.random.rand() < P_connect:
                stack.append(j)

@njit
def corr_func_anal_jit(r, T, N):
    lam0 = np.exp(1 / T) + 2
    lam1 = np.exp(1 / T) - 1

    num = lam1**N + lam0**r * lam1**(N - r) + lam0**(N - r) * lam1**r
    den = lam0**N + 2 * lam1**N

    return num / den

#### --- Model runner (non-class) --- ####

def initialize_model(d, L, T):
    q = 3
    if d == 1:
        N = L
        indices = np.arange(N).reshape(N, 1)
        x_arr = indices[:, 0]
        y_arr = np.zeros(N, dtype=np.int32)
    else:
        N = L * L
        indices = np.arange(N).reshape(L, L)
        y_arr, x_arr = np.meshgrid(np.arange(L), np.arange(L))
        x_arr = x_arr.flatten()
        y_arr = y_arr.flatten()

    S = np.zeros(N, dtype=np.int32)
    M_count = np.zeros(q, dtype=np.int32)
    M_count[0] = N

    P_connect = 1 - np.exp(-1 / T)

    m_spin = np.array([
        1,
        np.exp(2 * np.pi / 3 * 1j),
        np.exp(4 * np.pi / 3 * 1j)
    ])

    model = {
        "d": d,
        "L": L,
        "T": T,
        "q": q,
        "N": N,
        "indices": indices,
        "x_arr": x_arr,
        "y_arr": y_arr,
        "S": S,
        "M_count": M_count,
        "P_connect": P_connect,
        "m_spin": m_spin
    }
    return model

def run_MC(model, n_clusters=5, n_steps_eq=10000, n_steps=10000):
    d = model["d"]
    L = model["L"]
    T = model["T"]
    q = model["q"]
    N = model["N"]
    indices = model["indices"]
    x_arr = model["x_arr"]
    y_arr = model["y_arr"]
    S = model["S"]
    M_count = model["M_count"]
    P_connect = model["P_connect"]
    m_spin = model["m_spin"]

    for _ in range(n_steps_eq):
        for _ in range(n_clusters):
            flip_and_build_loop_jit(
                np.random.randint(0, N), S, M_count, q, d, P_connect,
                x_arr, y_arr, indices, L
            )

    mit = np.zeros((N, n_steps), dtype=np.complex128)
    for t in range(n_steps):
        for _ in range(n_clusters):
            flip_and_build_loop_jit(
                np.random.randint(0, N), S, M_count, q, d, P_connect,
                x_arr, y_arr, indices, L
            )
        mit[:, t] = m_spin[S]

    mi_mean = np.mean(mit, axis=1)
    mt = np.sum(mit, axis=0)
    m_mean = np.mean(mt)
    m0t_mit = np.conj(mit[0, :]) * mit
    corr_func = np.mean(m0t_mit, axis=1) - np.conj(mi_mean[0]) * mi_mean

    return {"corr_func": corr_func, "m_mean": m_mean}

#### --- Run Problem 2b --- ####
d = 1
L = 16
T_vals = [0.25, 0.5]
T_text = ["T_0_25", "T_0_5"]

for i in range(len(T_vals)):
    T = T_vals[i]
    model = initialize_model(d, L, T)
    r_vals = np.arange(0, model["N"])
    corr_anal = np.array([corr_func_anal_jit(r, T, model["N"]) for r in r_vals])
    results = run_MC(model)
    corr_MC = results["corr_func"]

    plt.figure()
    plt.title(f"Correlation function for 1D spin chain with temperature $T={T_vals[i]}$J")
    plt.xlabel("$r$")
    plt.ylabel("$C(r)$")
    plt.plot(r_vals, corr_anal, "k", label="Analytic")
    plt.plot(r_vals, np.real(corr_MC), "or", label="MC simulation")
    plt.legend()
    plt.savefig(f"corr_func_1D_{T_text[i]}.pdf")


#### --- Run Problem 2c --- ####
d = 2
L = 16
n_points = 20 # Number of data points to run MC simulation for
T_min = 1e-10
T_max = 1e10
T_vals = np.linspace(T_min, T_max, n_points)
m_mean_vals = np.zeros(n_points, dtype=complex)

for i in range(n_points):
    T = T_vals[i]
    print(f"T = {T}")
    model = initialize_model(d, L, T)
    results = run_MC(model, n_clusters=1, n_steps_eq=100000, n_steps=100000)
    m_mean_vals[i] = results["m_mean"]

plt.figure()
plt.title(f"Real part of average magnetization per site")
plt.xlabel("$T [J]$")
plt.ylabel("$\\text{Re}\\left(\\langle m \\rangle\\right)$")
plt.plot(T_vals, np.real(m_mean_vals), "r", label="Magnetization")
plt.legend()
plt.savefig(f"m_mean_real.pdf")