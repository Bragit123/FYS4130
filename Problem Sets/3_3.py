import numpy as np
import matplotlib.pyplot as plt

def roll_dice(S, N, print_results=False):
    data = np.random.randint(1, S+1, size=(N,))

    mean = np.mean(data)
    std = np.std(data)
    var = std**2

    mean_anal = (S+1)/2
    var_anal = (S+1)*(S-1)/12
    std_anal = 1/2 * np.sqrt((S+1)*(S-1)/3)

    mean_err = np.abs(mean-mean_anal)/mean_anal
    var_err = np.abs(var-var_anal)/var_anal
    std_err = np.abs(std-std_anal)/std_anal
    
    if print_results:
        print(f"S = {S}, N = {N}:")
        print(f"  Score | Result | True | Error ")
        print(f"  mean  | {mean:>6.1f} | {mean_anal:>4.1f} | {mean_err:>5.2f}")
        print(f"  var   | {var:>6.1f} | {var_anal:>4.1f} | {var_err:>5.2f}")
        print(f"  std   | {std:>6.1f} | {std_anal:>4.1f} | {std_err:>5.2f}")
        print()
    
    return (mean_err, var_err, std_err)

S_vals = [6, 50]
N_vals = [10, 100, 1000]

# for S in S_vals:
#     for N in N_vals:
#         roll_dice(S, N, True)

S = 20
good_mean = None
good_std = None
N_vals = np.arange(1, 1000)
for N in N_vals:
    scores = roll_dice(S, N, False)
    if scores[0] <= 0.01 and good_mean == None:
        good_mean = N
    if scores[2] <= 0.01 and good_std == None:
        good_std = N

print(f"Reached mean <= 0.01 at N = {good_mean}")
print(f"Reached std <= 0.01 at N = {good_std}")