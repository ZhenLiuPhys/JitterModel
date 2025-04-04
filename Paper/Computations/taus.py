import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import time
import sys

t0 = time.time()
cores = 20

omega0 = 2 * np.pi * 1.3e9 # in Hz
gamma = 2 * np.pi * 0.15 # in Hz
delomega0 = 2 * np.pi * 3 # in Hz
fj = 45 # in Hz
F0 = 1
Domega = 0
x0 = 0
v0 = 0
mode = 'DMP'

T = 15 # in s
num = 1000
Ntaus = 300
taumin = 0.001 # in s
taumax = 0.1 # in s
taus = np.logspace(np.log10(taumin), np.log10(taumax), Ntaus)

xf = lambda xi, vi, delomegai, ti, dt: (np.exp(-gamma * dt / 2) * (xi * np.cos((omega0 + delomegai) * dt) + vi / omega0 * np.sin((omega0 + delomegai) * dt))
                                        + F0 * np.exp(1j * (omega0 + Domega) * (dt + ti)) / omega0 / (2 * delomegai - 2 * Domega + 1j * gamma) * (1 - np.exp(-gamma * dt / 2 + 1j * (delomegai - Domega) * dt)))
vf = lambda xi, vi, delomegai, ti, dt: (np.exp(-gamma * dt / 2) * (-omega0 * xi * np.sin((omega0 + delomegai) * dt) + vi * np.cos((omega0 + delomegai) * dt))
                                        + 1j * F0 * np.exp(1j * (omega0 + Domega) * (dt + ti)) / (2 * delomegai - 2 * Domega + 1j * gamma) * (1 - np.exp(-gamma * dt / 2 + 1j * (delomegai - Domega) * dt)))

def simulate(params):
    tau = params[0]
    dt = min(tau, 1 / (2 * np.pi * fj)) / 10 # in s
    N = int(np.ceil(T / dt))
    
    if mode == 'Gaussian':
        etahat = delomega0 * np.random.randn(N)
        eta = [etahat[0]]
        for i in range(N - 1):
            eta.append(np.exp(-dt / tau) * eta[i] + np.sqrt(1 - np.exp(-2 * dt / tau)) * etahat[i + 1])
        phihat = delomega0 * np.random.randn(N)
        phi = [phihat[0]]
        for i in range(N - 1):
            phi.append(np.exp(-dt / tau) * phi[i] + np.sqrt(1 - np.exp(-2 * dt / tau)) * phihat[i + 1])
    if mode == 'DMP':
        etaflips = np.random.choice([1, -1], N, p = [(1 + np.exp(-dt / tau)) / 2, (1 - np.exp(-dt / tau)) / 2])
        eta = delomega0 * np.random.choice([1, -1]) * np.cumprod(etaflips)
        phiflips = np.random.choice([1, -1], N, p = [(1 + np.exp(-dt / tau)) / 2, (1 - np.exp(-dt / tau)) / 2])
        phi = delomega0 * np.random.choice([1, -1]) * np.cumprod(phiflips)
    delomega = eta * np.cos(2 * np.pi * fj * np.arange(N) * dt) + phi * np.sin(2 * np.pi * fj * np.arange(N) * dt)

    x = [x0]
    v = [v0]
    for i in range(N - 1):
        x.append(xf(x[i], v[i], delomega[i], i * dt, dt))
        v.append(vf(x[i], v[i], delomega[i], i * dt, dt))

    if params[1] == num - 1:
        print(f'Last tau={tau:.5f} run finished at t={time.time() - t0:.2f} s')
        sys.stdout.flush()
    return(np.abs(x[-1]) ** 2)

if __name__ == '__main__':
    pool = mp.Pool(cores)
    xsqr = pool.map_async(simulate, [(tau, n) for tau in taus for n in range(num)]).get()
    pool.close()
    pool.join()
print(f'All runs finished at t={time.time() - t0:.2f} s')
sys.stdout.flush()

xsqr = np.array(xsqr).reshape(Ntaus, num)
xsqrmean = np.mean(xsqr, axis = 1)
xsqrstd = np.std(xsqr, axis = 1) / np.sqrt(num)

np.savez('../Data/taus_DMP.npz',
         omega0 = omega0,
         gamma = gamma,
         delomega0 = delomega0,
         fj = fj,
         T = T,
         taus = taus,
         xsqrmean = xsqrmean,
         xsqrstd = xsqrstd)
print(f'Finished at t={time.time() - t0:.2f} s')
sys.stdout.flush()
