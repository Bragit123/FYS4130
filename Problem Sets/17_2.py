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
plt.plot(V/Vc, TvdWbelow/Tc, "b", label="$P < P_c$")
plt.plot(V/Vc, TvdWc/Tc, "r", label="$P = P_c$")
plt.plot(V/Vc, TvdWabove/Tc, "g", label="$P > P_c$")
plt.plot(V/Vc, Tidealbelow/Tc, "b--")
plt.plot(V/Vc, Tidealc/Tc, "r--")
plt.plot(V/Vc, Tidealabove/Tc, "g--")
plt.legend()
plt.savefig("temperature.pdf")

X = 0
def mu_vdW(T,V):
    mu = -k*T*(np.log((V-b*N)/N)+3/2*np.log(k*T)+X) + N*k*T*(b/(V-b*N)+1/N)-2*a*N/V
    return mu

def mu_ideal(T,V):
    mu = -k*T*(np.log(V/N)+3/2*np.log(k*T)+X-1)
    return mu

muvdWbelow = mu_vdW(Pbelow, V)
muvdWc = mu_vdW(Pc, V)
muvdWabove = mu_vdW(Pabove, V)

muidealbelow = mu_ideal(Pbelow, V)
muidealc = mu_ideal(Pc, V)
muidealabove = mu_ideal(Pabove, V)

muc = mu_vdW(Tc,Vc)

plt.figure()
plt.title("Solid = van der Waals ; Dashed = Ideal gas")
plt.xlabel("$P/P_c$")
plt.ylabel("$\\mu/\\mu_c$")
plt.plot(P_vdW(TvdWbelow,V)/Pc, muvdWbelow/muc, "b", label="$P < P_c$")
plt.plot(P_vdW(Tc,V)/Pc, muvdWc/muc, "r", label="$P = P_c$")
plt.plot(P_vdW(TvdWabove,V)/Pc, muvdWabove/muc, "g", label="$P > P_c$")
plt.plot(P_ideal(Tidealbelow,V)/Pc, muidealbelow/muc, "b--")
plt.plot(P_ideal(Tc,V)/Pc, muidealc/muc, "r--")
plt.plot(P_ideal(Tidealabove,V)/Pc, muidealabove/muc, "g--")
plt.legend()
plt.savefig("chemical_potential.pdf")

plt.figure()
plt.title("Solid = van der Waals ; Dashed = Ideal gas")
plt.xlabel("$P/P_c$")
plt.ylabel("$V/V_c$")
plt.plot(PvdWbelow/Pc, V/Vc, "b", label="$T < T_c$")
plt.plot(PvdWc/Pc, V/Vc, "r", label="$T = T_c$")
plt.plot(PvdWabove/Pc, V/Vc, "g", label="$T > T_c$")
plt.plot(Pidealbelow/Pc, V/Vc, "b--")
plt.plot(Pidealc/Pc, V/Vc, "r--")
plt.plot(Pidealabove/Pc, V/Vc, "g--")
plt.legend()
plt.savefig("volume.pdf")