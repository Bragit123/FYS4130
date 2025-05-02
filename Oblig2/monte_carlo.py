import numpy as np
from scipy.constants import k
import matplotlib.pyplot as plt

np.random.seed(100)

q = 3 # q spin states
L = 16 # System size
N = L # Total number of spins
T = 0.5 # Temperature in units of J

P_connect = 1 - np.exp(-1/T) # Connection probability
# P_connect = 0.3 # Connection probability

n_clusters = 1 # Number of clusters per MC step
n_steps_eq = 10000 # Number of equilibrium MC steps
n_steps = 10000 # Number of measurement MC steps

# S = np.zeros(N, dtype=int) # Array containing spin state of each site
S = np.random.randint(0, 3, N) # Array containing spin state of each site

m_spin = np.array([
        1,
        np.exp(2*np.pi/3 * 1j),
        np.exp(4*np.pi/3 * 1j)
]) # Possible values of m_j from spin

# Initialize magnetization counter
M_count = np.zeros(q)
M_count[0] = N

def index(x):
    # Find index of position
    return x

def xpos(i):
    # Find position of index
    return i%L

def neighbor_index(i, dir):
    x = xpos(i)
    if dir == 0:
        # Right
        return index((x+1)%L)
    elif dir == 1:
        # Left
        return index((x-1+L)%L)
    else:
        raise ValueError("dir must be 0 or 1")

def flip_and_build(i):
    # Run one cluster starting from i by going through neighbours recursively
    old_state = S[i]
    new_state = (S[i]+1)%q

    S[i] = new_state # Flip the spin
    
    # Count spins
    M_count[old_state] -= 1
    M_count[new_state] += 1

    for dir in range(2):
        j = neighbor_index(i, dir)
        if (S[j] == old_state):
            if (np.random.uniform() < P_connect):
                flip_and_build(j)


## Equilibrate
for t in range(n_steps_eq):
    for c in range(n_clusters):
        flip_and_build(np.random.randint(0, N))

## Run measurements
mit = np.zeros((N, n_steps), dtype=complex)

for t in range(n_steps):
    for c in range(n_clusters):
        flip_and_build(np.random.randint(0,N))
    
    mit[:, t] = m_spin[S]

m0t_mit = np.conj(mit[0,:]) * mit
mi_mean = np.mean(mit, axis=1)

corr_func = np.mean(m0t_mit, axis=1) - np.conj(mi_mean[0]) * mi_mean

# print(corr_func)



## Analytic correlation function
def corr_func_anal(r):
    lam0 = np.exp(1/T) + 2
    lam1 = np.exp(1/T) - 1
    
    num = lam1**N + lam0**r * lam1**(N-r) + lam0**(N-r) * lam1**r
    den = lam0**N + 2 * lam1**N

    return num/den

## Plot
r_vals = np.arange(0, N)
corr_anal = corr_func_anal(r_vals)

plt.xlabel("$r$")
plt.ylabel("$C(r)$")
plt.plot(r_vals, corr_anal, label="Analytic")
plt.plot(r_vals, np.real(corr_func), label="Real")
plt.plot(r_vals, np.imag(corr_func), label="Imaginary")
plt.legend()
plt.savefig("test.pdf")