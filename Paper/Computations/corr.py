import numpy as np
import time

t0 = time.time()

omega0 = 2 * np.pi * 1.3e9 # in Hz
gamma = 2 * np.pi * 0.15 # in Hz
delomega0 = 2 * np.pi * 3 # in Hz
tau = 1 / (5 * np.pi) # in s
fj = 45 # in Hz
F0 = 1
Domega = 0
x0 = 0
v0 = 0
mode = 'Gaussian'

tcut = 15 # in s
T = 25 # in s
num = 10000

dt = min(tau, 1 / (2 * np.pi * fj)) / 10 # in s
Ncut = int(np.ceil(tcut / dt))
N = int(np.ceil(T / dt))

rho = gamma * tau * (2 + gamma * tau) / ((2 + gamma * tau) ** 2 + (4 * np.pi * fj * tau) ** 2)
alpha = 4 * delomega0 ** 2 * rho / gamma ** 2
print(f'alpha: {alpha:.3f}')
print(f'Transient: {np.exp(-gamma * tcut / 2):.2e}')

xf = lambda xi, vi, delomegai, ti: (np.exp(-gamma * dt / 2) * (xi * np.cos((omega0 + delomegai) * dt) + vi / omega0 * np.sin((omega0 + delomegai) * dt))
                                    + F0 * np.exp(1j * (omega0 + Domega) * (dt + ti)) / omega0 / (2 * delomegai - 2 * Domega + 1j * gamma) * (1 - np.exp(-gamma * dt / 2 + 1j * (delomegai - Domega) * dt)))
vf = lambda xi, vi, delomegai, ti: (np.exp(-gamma * dt / 2) * (-omega0 * xi * np.sin((omega0 + delomegai) * dt) + vi * np.cos((omega0 + delomegai) * dt))
                                    + 1j * F0 * np.exp(1j * (omega0 + Domega) * (dt + ti)) / (2 * delomegai - 2 * Domega + 1j * gamma) * (1 - np.exp(-gamma * dt / 2 + 1j * (delomegai - Domega) * dt)))

xs = np.zeros((num, N), dtype = complex)
vs = np.zeros((num, N), dtype = complex)
for i in range(num):
    if mode == 'Gaussian':
        etahat = delomega0 * np.random.randn(N)
        eta = np.exp(-np.arange(N) * dt / tau) * (etahat[0] + np.sqrt(1 - np.exp(-2 * dt / tau)) * np.insert(np.cumsum(etahat[1:] * np.exp(np.arange(1, N) * dt / tau)), 0, 0))
        phihat = delomega0 * np.random.randn(N)
        phi = np.exp(-np.arange(N) * dt / tau) * (phihat[0] + np.sqrt(1 - np.exp(-2 * dt / tau)) * np.insert(np.cumsum(phihat[1:] * np.exp(np.arange(1, N) * dt / tau)), 0, 0))
    if mode == 'DMP':
        etaflips = np.random.choice([1, -1], N, p = [(1 + np.exp(-dt / tau)) / 2, (1 - np.exp(-dt / tau)) / 2])
        eta = delomega0 * np.random.choice([1, -1]) * np.cumprod(etaflips)
        phiflips = np.random.choice([1, -1], N, p = [(1 + np.exp(-dt / tau)) / 2, (1 - np.exp(-dt / tau)) / 2])
        phi = delomega0 * np.random.choice([1, -1]) * np.cumprod(phiflips)
    delomega = eta * np.cos(2 * np.pi * fj * np.arange(N) * dt) + phi * np.sin(2 * np.pi * fj * np.arange(N) * dt)

    x = [x0]
    v = [v0]
    for j in range(N - 1):
        x.append(xf(x[j], v[j], delomega[j], j * dt))
        v.append(vf(x[j], v[j], delomega[j], j * dt))
    x = np.array(x)
    v = np.array(v)

    xs[i] = x
    vs[i] = v

xmean = np.mean(xs[:,Ncut:], axis = 0)
xstd = np.std(xs[:,Ncut:], axis = 0) / np.sqrt(num)
Cx = np.mean(xs[:,Ncut:] * np.conj(xs[:,Ncut,None]), axis = 0)
Cxstd = np.std(xs[:,Ncut:] * np.conj(xs[:,Ncut,None]), axis = 0) / np.sqrt(num)
Kx = (Cx - xmean * np.conj(xmean[0])) * np.exp(-1j * omega0 * np.arange(N - Ncut) * dt)
Kxstd = np.sqrt(Cxstd ** 2 + xstd ** 2 * np.abs(xmean[0]) ** 2 + np.abs(xmean) ** 2 * xstd[0] ** 2)

np.savez('../Data/corr.npz',
         omega0 = omega0,
         gamma = gamma,
         delomega0 = delomega0,
         tau = tau,
         fj = fj,
         mode = mode,
         tcut = tcut,
         T = T,
         Kx = Kx,
         Kxstd = Kxstd)
print(f'Total time: {time.time() - t0:.2f} s')
