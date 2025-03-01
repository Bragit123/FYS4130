import numpy as np
import matplotlib.pyplot as plt
import sys

np.random.seed(100)

def rand_add(p, N):
    rnd = np.random.uniform(0, 1, size=(N,))
    res = rnd <= p

    s = np.sum(res)

    return s

# Set values
if len(sys.argv) < 3:
    print("Include N and p as arguments when running the program.")
    sys.exit()
N = int(sys.argv[1])
p = float(sys.argv[2])

num_iter = 1000
data = np.zeros(num_iter)

# Run simulation
for i in range(num_iter):
    data[i] = rand_add(p, N)

# Find theoretical distribution
bin_prob = np.zeros(N+1)
bin_coeff = 1
n_vals = np.arange(0, N+1)
for n in n_vals:
    bin_prob[n] = bin_coeff * p**n * (1-p)**(N-n)
    bin_coeff = (N-n)/(n+1) * bin_coeff

plt.figure()
plt.xlabel("Results")
plt.ylabel("Number of occurrences")
plt.hist(data, density=True, bins=10, label="Simulated results")
plt.plot(n_vals, bin_prob, "r", label="Binomial distribution")
plt.legend()
plt.savefig("3_9_hist.pdf")