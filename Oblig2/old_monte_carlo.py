## This code was written by looking through the C++ skeleton code, and
## translating it into python, then expanding on that.

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(100)


#### Problem 2a)

class Spin_Model:
    def __init__(self, d, L, T):
        self.d = d # Dimension of system. Should be 1 or 2 (1D chain or 2D lattice)
        self.L = L # System size
        self.T = T # Temperature in units of J

        self.q = 3 # Number of spin states

        ## Define index and position arrays depending on system dimension
        if d == 1:
            self.N = L
            self.indices = np.arange(self.N).reshape(self.N, 1)
            self.x_arr = self.indices[:,0]
            self.y_arr = np.zeros(self.N, dtype=int)
        elif d == 2:
            self.N = L*L
            self.indices = np.arange(self.N).reshape(L, L)
            y_arr, x_arr = np.meshgrid(np.arange(L), np.arange(L))
            self.x_arr = x_arr.flatten()
            self.y_arr = y_arr.flatten()
        
        self.S = np.zeros(self.N, dtype=int) # Array containing spin state of each site
        self.P_connect = 1 - np.exp(-1/T) # Connection probability

        # Possible values of m_j from spin
        self.m_spin = np.array([
                1,
                np.exp(2*np.pi/3 * 1j),
                np.exp(4*np.pi/3 * 1j)
        ])

        # Initialize magnetization counter
        self.M_count = np.zeros(self.q)
        self.M_count[0] = self.N
    
    def neighbor_index(self, i, dir):
        # Find position from index
        x = self.x_arr[i]
        y = self.y_arr[i]

        if dir == 0:
            # Right
            return self.indices[(x+1)%self.L, y]
        elif dir == 1:
            # Up
            return self.indices[x, (y+1)%self.L]
        elif dir == 2:
            # Left
            return self.indices[(x-1+self.L)%self.L, y]
        elif dir == 3:
            # Down
            return self.indices[x, (y-1+self.L)%self.L]
        else:
            raise ValueError("dir must be 0, 1, 2 or 3 (right, up, left, down)")

    def flip_and_build(self, i):
        # Run one cluster starting from i by going through neighbours recursively
        old_state = self.S[i]
        new_state = (self.S[i]+1)%self.q

        self.S[i] = new_state # Flip the spin
        
        # Count spins
        self.M_count[old_state] -= 1
        self.M_count[new_state] += 1

        if self.d == 1:
            dir_arr = [0, 2] # Only right and left for 1D systems
        else:
            dir_arr = [0, 1, 2, 3] # All 4 directions for 2D systems

        for dir in dir_arr:
            j = self.neighbor_index(i, dir)
            if (self.S[j] == old_state):
                if (np.random.uniform() < self.P_connect):
                    self.flip_and_build(j)

    def flip_and_build_loop(self, start_i):
        stack = [start_i]
        old_state = self.S[start_i]
        new_state = (old_state + 1) % self.q

        while stack:
            i = stack.pop()

            # Skip already-flipped spins
            if self.S[i] != old_state:
                continue

            # Flip the spin
            self.S[i] = new_state
            self.M_count[old_state] -= 1
            self.M_count[new_state] += 1

            # Determine directions based on dimensionality
            if self.d == 1:
                dir_arr = [0, 2]  # Left and right
            else:
                dir_arr = [0, 1, 2, 3]  # Up, right, down, left

            # Check neighbors
            for dir in dir_arr:
                j = self.neighbor_index(i, dir)
                if self.S[j] == old_state:
                    if np.random.uniform() < self.P_connect:
                        stack.append(j)


    ## Analytic correlation function
    def corr_func_anal(self, r):
        lam0 = np.exp(1/self.T) + 2
        lam1 = np.exp(1/self.T) - 1
        
        num = lam1**self.N + lam0**r * lam1**(self.N-r) + lam0**(self.N-r) * lam1**r
        den = lam0**self.N + 2 * lam1**self.N

        return num/den

    def run_MC(self, n_clusters=5, n_steps_eq=10000, n_steps=10000):
        ## Equilibrate
        for t in range(n_steps_eq):
            for c in range(n_clusters):
                self.flip_and_build(np.random.randint(0, self.N))
        
        ## Run measurements
        mit = np.zeros((self.N, n_steps), dtype=complex)

        for t in range(n_steps):
            for c in range(n_clusters):
                self.flip_and_build_loop(np.random.randint(0, self.N))
            
            mit[:, t] = self.m_spin[self.S] # Local magnetization m_i

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


#### Problem 2b)

## Set parameters
d = 1
L = 16
T_vals = [0.25, 0.5]
T_text = ["T_0_25", "T_0_5"]

for i in range(len(T_vals)):
    T = T_vals[i]
    model = Spin_Model(d, L, T)
    r_vals = np.arange(0, model.N)
    corr_anal = model.corr_func_anal(r_vals) # Find analytic correlation function
    results = model.run_MC() # Run MC simulation
    corr_MC = results["corr_func"]

    ## Plot results
    plt.figure()
    plt.title(f"Correlation function for 1D spin chain with temperature $T={T_vals[i]}$J")
    plt.xlabel("$r$")
    plt.ylabel("$C(r)$")
    plt.plot(r_vals, corr_anal, "k", label="Analytic")
    plt.plot(r_vals, np.real(corr_MC), "or", label="MC simulation")
    plt.legend()
    plt.savefig(f"corr_func_1D_{T_text[i]}.pdf")



# #### Problem 2c)

# d = 2
# n_points = 6 # Number of data points to run MC simulation for
# T_min = 0
# T_max = 100
# T_vals = np.linspace(T_min, T_max, n_points)
# m_mean_vals = np.zeros(n_points)

# model = Spin_Model(d=2, L=L, T=T_vals[0])
# for i in range(n_points):
#     model.T = T_vals[i]
#     results = model.run_MC()
#     m_mean_vals[i] = results["m_mean"]

# plt.figure()
# plt.title(f"Real part of average magnetization per site")
# plt.xlabel("$T [J]$")
# plt.ylabel("$\\text{Re}\\left(\\langle m \\rangle\\right)$")
# plt.plot(T_vals, np.real(m_mean_vals), "or")
# plt.legend()
# plt.savefig(f"m_mean_real.pdf")