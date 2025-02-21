import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const

## Problem 17.2
k = const.k
a = 1
b = 1
N = 100

def P_vdW(T, V):
    P = N*k*T/(V-b*N) - a*N*N/(V*V)
    return P

def P_ideal(T,V):
    P = N*k*T/V
    return P

def T_vdW(P, V):
    # Inverse of P(T) from above
    T = P*(V-b*N)/(N*k) + a*N*N/(V*V)
    return T

def T_ideal(P, V):
    # Inverse of P(T) from above
    T = P*V/(N*k)
    return T

Tc = 8*a/(27*k*b)
Vc = 3*b*N
Pc = a/(27*b*b)

Tbelow = Tc/1.5
Tabove = Tc*1.5

V = np.linspace(Vc/2,Vc*2, 1001)

PvdWbelow = P_vdW(Tbelow, V)
PvdWc = P_vdW(Tc, V)
PvdWabove = P_vdW(Tabove, V)

Pidealbelow = P_ideal(Tbelow, V)
Pidealc = P_ideal(Tc, V)
Pidealabove = P_ideal(Tabove, V)

plt.figure()
plt.title("Solid = van der Waals ; Dashed = Ideal gas")
plt.xlabel("$V/V_c$")
plt.ylabel("$P/P_c$")
plt.plot(V/Vc, PvdWbelow/Pc, "b", label="$T < T_c$")
plt.plot(V/Vc, PvdWc/Pc, "r", label="$T = T_c$")
plt.plot(V/Vc, PvdWabove/Pc, "g", label="$T > T_c$")
plt.plot(V/Vc, Pidealbelow/Pc, "b--")
plt.plot(V/Vc, Pidealc/Pc, "r--")
plt.plot(V/Vc, Pidealabove/Pc, "g--")
plt.legend()
plt.savefig("pressure.pdf")

print("For T<Tc the second derivative switches sign")

Pbelow = Pc/1.5
Pabove = Pc*1.5

TvdWbelow = T_vdW(Pbelow, V)
TvdWc = T_vdW(Pc, V)
TvdWabove = T_vdW(Pabove, V)

Tidealbelow = T_ideal(Pbelow, V)
Tidealc = T_ideal(Pc, V)
Tidealabove = T_ideal(Pabove, V)

plt.figure()
plt.title("Solid = van der Waals ; Dashed = Ideal gas")
plt.xlabel("$V/V_c$")
plt.ylabel("$T/T_c$")
plt.plot(V/Vc, PvdWbelow/Pc, "b", label="$T < T_c$")
plt.plot(V/Vc, PvdWc/Pc, "r", label="$T = T_c$")
plt.plot(V/Vc, PvdWabove/Pc, "g", label="$T > T_c$")
plt.plot(V/Vc, Pidealbelow/Pc, "b--")
plt.plot(V/Vc, Pidealc/Pc, "r--")
plt.plot(V/Vc, Pidealabove/Pc, "g--")
plt.legend()
plt.savefig("temperature.pdf")