import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 14})
fig, ax = plt.subplots(figsize = (6., 4.))

omega0 = 2 * np.pi * 1.3e9 # in Hz
gamma = 2 * np.pi * 0.15 # in Hz
delomega0 = 2 * np.pi * 3 # in Hz
tau = 1 / (5 * np.pi) # in s
fj = 45 # in Hz
F0 = 1
Domega = 0
x0 = 0
v0 = 0

dt = min(tau, 1 / (2 * np.pi * fj)) / 10 # in s
T = 50 # in s
N = int(np.ceil(T / dt))

xsqrinfty = np.abs(F0) ** 2 / omega0 ** 2 / gamma ** 2
xsqrnum = 0.87
ax.plot(np.arange(N) * dt,  (1 - np.exp(-gamma * np.arange(N) * dt / 2)) ** 2, color = 'k', label = 'No jittering')

xf = lambda xi, vi, delomegai, ti: (np.exp(-gamma * dt / 2) * (xi * np.cos((omega0 + delomegai) * dt) + vi / omega0 * np.sin((omega0 + delomegai) * dt))
                                    + F0 * np.exp(1j * (omega0 + Domega) * (dt + ti)) / omega0 / (2 * delomegai - 2 * Domega + 1j * gamma) * (1 - np.exp(-gamma * dt / 2 + 1j * (delomegai - Domega) * dt)))
vf = lambda xi, vi, delomegai, ti: (np.exp(-gamma * dt / 2) * (-omega0 * xi * np.sin((omega0 + delomegai) * dt) + vi * np.cos((omega0 + delomegai) * dt))
                                    + 1j * F0 * np.exp(1j * (omega0 + Domega) * (dt + ti)) / (2 * delomegai - 2 * Domega + 1j * gamma) * (1 - np.exp(-gamma * dt / 2 + 1j * (delomegai - Domega) * dt)))

etahat = delomega0 * np.random.randn(N)
etaG = [etahat[0]]
for i in range(N - 1):
    etaG.append(np.exp(-dt / tau) * etaG[i] + np.sqrt(1 - np.exp(-2 * dt / tau)) * etahat[i + 1])
etaG = np.array(etaG)
phihat = delomega0 * np.random.randn(N)
phiG = [phihat[0]]
for i in range(N - 1):
    phiG.append(np.exp(-dt / tau) * phiG[i] + np.sqrt(1 - np.exp(-2 * dt / tau)) * phihat[i + 1])
phiG = np.array(phiG)
delomegaG = etaG * np.cos(2 * np.pi * fj * np.arange(N) * dt) + phiG * np.sin(2 * np.pi * fj * np.arange(N) * dt)

xG = [x0]
vG = [v0]
for i in range(N - 1):
    xG.append(xf(xG[i], vG[i], delomegaG[i], i * dt))
    vG.append(vf(xG[i], vG[i], delomegaG[i], i * dt))
xG = np.array(xG)
vG = np.array(vG)
ax.plot(np.arange(N) * dt, np.abs(xG) ** 2 / xsqrinfty, color = (0.317647, 0.654902, 0.752941), label = 'Gaussian')

etaflips = np.random.choice([1, -1], N, p = [(1 + np.exp(-dt / tau)) / 2, (1 - np.exp(-dt / tau)) / 2])
etaDMP = delomega0 * np.random.choice([1, -1]) * np.cumprod(etaflips)
phiflips = np.random.choice([1, -1], N, p = [(1 + np.exp(-dt / tau)) / 2, (1 - np.exp(-dt / tau)) / 2])
phiDMP = delomega0 * np.random.choice([1, -1]) * np.cumprod(phiflips)
delomegaDMP = etaDMP * np.cos(2 * np.pi * fj * np.arange(N) * dt) + phiDMP * np.sin(2 * np.pi * fj * np.arange(N) * dt)

xDMP = [x0]
vDMP = [v0]
for i in range(N - 1):
    xDMP.append(xf(xDMP[i], vDMP[i], delomegaDMP[i], i * dt))
    vDMP.append(vf(xDMP[i], vDMP[i], delomegaDMP[i], i * dt))
xDMP = np.array(xDMP)
vDMP = np.array(vDMP)
ax.plot(np.arange(N) * dt, np.abs(xDMP) ** 2 / xsqrinfty, color = (1., 0.721569, 0.219608), label = 'DMP')

ax.axhline(xsqrnum, ls = '--', color = '0.7')

ax.set_xlim(0, T)
ax.set_ylim(0, 1.05)
ax.set_xlabel(r'$t$\,[s]')
ax.set_ylabel(r'$\frac{|x(t)|^2}{|x_0(\infty)|^2}$')

ax.tick_params(which = 'both', direction = 'in')
secxax = ax.secondary_xaxis('top')
secxax.tick_params(which = 'both', direction = 'in')
plt.setp(secxax.get_xticklabels(), visible = False)
secyax = ax.secondary_yaxis('right')
secyax.tick_params(which = 'both', direction = 'in')
plt.setp(secyax.get_yticklabels(), visible = False)

ax.legend(loc = 'lower right')

fig.tight_layout()
#fig.show()
fig.savefig('comparison.pdf')
