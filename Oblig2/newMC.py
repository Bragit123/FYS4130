## This code was written by looking through the C++ skeleton code, and
## translating it into python, then expanding on that.

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(100)


#### Problem 2a)

d = 1 # Dimension of system. Should be 1 or 2 (1D chain or 2D lattice)
L = 16 # System size
T = 0.25 # Temperature in units of J

q = 3 # Number of spin states

## Define index and position arrays depending on system dimension
if d == 1:
    N = L
    indices = np.arange(N).reshape(N, 1)
    x_arr = indices[:,0]
    y_arr = np.zeros(N, dtype=int)
elif d == 2:
    N = L*L
    indices = np.arange(N).reshape(L, L)
    y_arr, x_arr = np.meshgrid(np.arange(L), np.arange(L))
    x_arr = x_arr.flatten()
    y_arr = y_arr.flatten()

S = np.zeros(N, dtype=int) # Array containing spin state of each site
P_connect = 1 - np.exp(-1/T) # Connection probability

# Possible values of m_j from spin
m_spin = np.array([
        1,
        np.exp(2*np.pi/3 * 1j),
        np.exp(4*np.pi/3 * 1j)
])

# Initialize magnetization counter
M_count = np.zeros(q)
M_count[0] = N

def neighbor_index(self, i, dir):
    # Find position from index
    x = x_arr[i]
    y = y_arr[i]

    if dir == 0:
        # Right
        return indices[(x+1)%L, y]
    elif dir == 1:
        # Up
        return indices[x, (y+1)%L]
    elif dir == 2:
        # Left
        return indices[(x-1+L)%L, y]
    elif dir == 3:
        # Down
        return indices[x, (y-1+L)%L]
    else:
        raise ValueError("dir must be 0, 1, 2 or 3 (right, up, left, down)")

def flip_and_build(self, i):
    # Run one cluster starting from i by going through neighbours recursively
    old_state = S[i]
    new_state = (S[i]+1)%q

    S[i] = new_state # Flip the spin
    
    # Count spins
    M_count[old_state] -= 1
    M_count[new_state] += 1

    if d == 1:
        dir_arr = [0, 2] # Only right and left for 1D systems
    else:
        dir_arr = [0, 1, 2, 3] # All 4 directions for 2D systems

    for dir in dir_arr:
        j = neighbor_index(i, dir)
        if (S[j] == old_state):
            if (np.random.uniform() < P_connect):
                flip_and_build(j)

## Analytic correlation function
def corr_func_anal(self, r):
    lam0 = np.exp(1/T) + 2
    lam1 = np.exp(1/T) - 1
    
    num = lam1**N + lam0**r * lam1**(N-r) + lam0**(N-r) * lam1**r
    den = lam0**N + 2 * lam1**N

    return num/den

def run_MC(self, d, L, T, n_clusters=5, n_steps_eq=10000, n_steps=10000):
    ## Equilibrate
    for t in range(n_steps_eq):
        for c in range(n_clusters):
            flip_and_build(np.random.randint(0, N))
    
    ## Run measurements
    mit = np.zeros((N, n_steps), dtype=complex)

    for t in range(n_steps):
        for c in range(n_clusters):
            flip_and_build(np.random.randint(0, N))
        
        mit[:, t] = m_spin[S] # Local magnetization m_i

    mi_mean = np.mean(mit, axis=1) # Mean local magnetization

    mt = np.sum(mit, axis=0) # Total magnetization
    m_mean = np.mean(mt) # Mean total magnetization

    m0t_mit = np.conj(mit[0,:]) * mit # Conjugate of m_0 times m_i for each time step and site i

    corr_func = np.mean(m0t_mit, axis=1) - np.conj(mi_mean[0]) * mi_mean # Correlation function as function of site i

    results = {
        "corr_func": corr_func,
        "m_mean": m_mean
    }

    return results