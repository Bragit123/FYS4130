import numpy as np
import matplotlib.pyplot as plt

alpha = 1
gamma = 10

N = 1000
eps = 1e-10
nx = np.linspace(eps, 1, N)
ny = np.linspace(eps, 1, N)
nz = np.linspace(eps, 1, N)

nx, ny, nz = np.meshgrid(nx, ny, nz)

S_V = -(nx*np.log(alpha*nx) + ny*np.log(alpha*ny) + nz*np.log(alpha*nz) + gamma*(nx*ny + ny*nz + nz*nx))
# dS_Vdnx = -1-np.log(alpha*nx)-gamma*(ny+nz)
# dS_Vdny = -1-np.log(alpha*ny)-gamma*(nx+nz)
# dS_Vdnz = -1-np.log(alpha*nz)-gamma*(ny+nx)

maxind = np.argmax(S_V)
print(maxind)
nxmax = nx.flatten()[maxind]
nymax = ny.flatten()[maxind]
nzmax = nz.flatten()[maxind]

print(nxmax, nymax, nzmax)


for i in range(8):
    plt.subplot(4, 2,i+1)
    plt.imshow(S_V[:,:,i*10])
    plt.colorbar()
plt.savefig("entropy1.pdf")
for i in range(8):
    plt.subplot(4, 2,i+1)
    plt.plot(S_V[:,i,i*10], label=f"i={i}")
    plt.legend()
plt.savefig("entropy.pdf")

T = 1
Nx = Ny = Nz = 100
NN = Nx+Ny+Nz
beta = Nx*Ny + Ny*Nz + Nz*Nx
P = np.linspace(eps,10000,N)
V = 0.5 * NN*T/P * (-1 + np.sqrt(1+gamma*beta*P/(T*NN**2)))

F = T*(Nx*np.log(alpha*Nx/V) + Ny*np.log(alpha*Ny/V) + Nz*np.log(alpha*Nz/V) + gamma*beta/V)
G = F+P*V
print(np.any(np.gradient(np.gradient(G))>0))
print(np.any(np.gradient(np.gradient(F))>0))
plt.figure()
plt.plot(P, G)
plt.plot(P, F)
plt.savefig("gibbs.pdf")