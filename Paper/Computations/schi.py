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
tau = 1 / (5 * np.pi) # in s
fj = 45 # in Hz
F0 = 1
x0 = 0
v0 = 0
mode = 'Gaussian'

tcut = 15 # in s
T = 35 # in s
num = 100

dt = min(tau, 1 / (2 * np.pi * fj)) / 10 # in s
Ncut = int(np.ceil(tcut / dt))
N = int(np.ceil(T / dt))
maxfreq = 2 * max(gamma / (2 * np.pi), delomega0 / (2 * np.pi), 1 / (2 * np.pi * tau), fj) # in Hz
Nmax = int(np.ceil(maxfreq * ((N - Ncut) * dt)))
freqs = np.arange(-Nmax, Nmax + 1) / ((N - Ncut) * dt) # in Hz

xf = lambda xi, vi, delomegai, ti, Domega: (np.exp(-gamma * dt / 2) * (xi * np.cos((omega0 + delomegai) * dt) + vi / omega0 * np.sin((omega0 + delomegai) * dt))
                                            + F0 * np.exp(1j * (omega0 + Domega) * (dt + ti)) / omega0 / (2 * delomegai - 2 * Domega + 1j * gamma) * (1 - np.exp(-gamma * dt / 2 + 1j * (delomegai - Domega) * dt)))
vf = lambda xi, vi, delomegai, ti, Domega: (np.exp(-gamma * dt / 2) * (-omega0 * xi * np.sin((omega0 + delomegai) * dt) + vi * np.cos((omega0 + delomegai) * dt))
                                            + 1j * F0 * np.exp(1j * (omega0 + Domega) * (dt + ti)) / (2 * delomegai - 2 * Domega + 1j * gamma) * (1 - np.exp(-gamma * dt / 2 + 1j * (delomegai - Domega) * dt)))

def simulate(params):
    Dfreq = params[0]
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
        x.append(xf(x[i], v[i], delomega[i], i * dt, 2 * np.pi * Dfreq))
        v.append(vf(x[i], v[i], delomega[i], i * dt, 2 * np.pi * Dfreq))

    x = np.array(x[Ncut:])
    xfft = np.fft.fft(x * np.exp(-1j * omega0 * np.arange(Ncut, N) * dt)) * dt
    if params[1] == num - 1:
        print(f'Last Dfreq={Dfreq:.2f} run finished at t={time.time() - t0:.2f} s')
        sys.stdout.flush()
    return(np.concatenate((xfft[-Nmax:], xfft[:Nmax + 1])))

if __name__ == '__main__':
    pool = mp.Pool(cores)
    xffts = pool.map_async(simulate, [(Dfreq, n) for Dfreq in freqs for n in range(num)]).get()
    pool.close()
    pool.join()
print(time.time() - t0)
sys.stdout.flush()

xffts = np.array(xffts).reshape(2 * Nmax + 1, num, 2 * Nmax + 1) # shape is (Dfreqs, samples, freqs)
Schi = np.mean(np.abs(xffts) ** 2, axis = 1) / ((N - Ncut) * dt)
Schistd = np.std(np.abs(xffts) ** 2, axis = 1) / ((N - Ncut) * dt * np.sqrt(num))

np.savez('../Data/schi.npz',
         omega0 = omega0,
         gamma = gamma,
         delomega0 = delomega0,
         tau = tau,
         fj = fj,
         tcut = tcut,
         T = T,
         freqs = freqs,
         Schi = Schi,
         Schistd = Schistd)
print(time.time() - t0)
sys.stdout.flush()
