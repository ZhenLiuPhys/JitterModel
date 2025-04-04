from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1235)

plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 14})
fig = plt.figure(figsize = (6., 8.))
grid = GridSpec(2, 1, figure = fig, hspace = 0)
ax = fig.add_subplot(grid[0])
ax2 = fig.add_subplot(grid[1])

omega0 = 2 * np.pi * 1.3e9 # in Hz
gamma = 2 * np.pi * 0.15 # in Hz
delomega0 = 2 * np.pi * 3 # in Hz
tau = 1 / (5 * np.pi) # in s
fj = 45 # in Hz
fj2 = 10 # in Hz
F0 = 1
Domega = 0
x0 = 0
v0 = 0

dt = min(tau, 1 / (2 * np.pi * fj)) / 10 # in s
T = 25 # in s
N = int(np.ceil(T / dt))
T0 = 15 # in s
Tf = 25 # in s

xsqrinfty = np.abs(F0) ** 2 / omega0 ** 2 / gamma ** 2

xf = lambda xi, vi, delomegai, ti: (np.exp(-gamma * dt / 2) * (xi * np.cos((omega0 + delomegai) * dt) + vi / omega0 * np.sin((omega0 + delomegai) * dt))
                                    + F0 * np.exp(1j * (omega0 + Domega) * (dt + ti)) / omega0 / (2 * delomegai - 2 * Domega + 1j * gamma) * (1 - np.exp(-gamma * dt / 2 + 1j * (delomegai - Domega) * dt)))
vf = lambda xi, vi, delomegai, ti: (np.exp(-gamma * dt / 2) * (-omega0 * xi * np.sin((omega0 + delomegai) * dt) + vi * np.cos((omega0 + delomegai) * dt))
                                    + 1j * F0 * np.exp(1j * (omega0 + Domega) * (dt + ti)) / (2 * delomegai - 2 * Domega + 1j * gamma) * (1 - np.exp(-gamma * dt / 2 + 1j * (delomegai - Domega) * dt)))

etahat = delomega0 * np.random.randn(N)
eta = [etahat[0]]
for i in range(N - 1):
    eta.append(np.exp(-dt / tau) * eta[i] + np.sqrt(1 - np.exp(-2 * dt / tau)) * etahat[i + 1])
eta = np.array(eta)
phihat = delomega0 * np.random.randn(N)
phi = [phihat[0]]
for i in range(N - 1):
    phi.append(np.exp(-dt / tau) * phi[i] + np.sqrt(1 - np.exp(-2 * dt / tau)) * phihat[i + 1])
phi = np.array(phi)
delomega = eta * np.cos(2 * np.pi * fj * np.arange(N) * dt) + phi * np.sin(2 * np.pi * fj * np.arange(N) * dt)

x = [x0]
v = [v0]
for i in range(N - 1):
    x.append(xf(x[i], v[i], delomega[i], i * dt))
    v.append(vf(x[i], v[i], delomega[i], i * dt))
x = np.array(x)
v = np.array(v)
dpower = (np.abs(x[1:]) ** 2 - np.abs(x[:-1]) ** 2) / (xsqrinfty * dt)
theta = np.angle(np.conj(x) * F0 * np.exp(1j * (omega0 + Domega) * np.arange(N) * dt))
theta[np.where(theta <= -np.pi / 2)[0]] += 2 * np.pi

etahat2 = delomega0 * np.random.randn(N)
eta2 = [etahat2[0]]
for i in range(N - 1):
    eta2.append(np.exp(-dt / tau) * eta2[i] + np.sqrt(1 - np.exp(-2 * dt / tau)) * etahat2[i + 1])
eta2 = np.array(eta2)
phihat2 = delomega0 * np.random.randn(N)
phi2 = [phihat2[0]]
for i in range(N - 1):
    phi2.append(np.exp(-dt / tau) * phi2[i] + np.sqrt(1 - np.exp(-2 * dt / tau)) * phihat2[i + 1])
phi2 = np.array(phi2)
delomega2 = eta2 * np.cos(2 * np.pi * fj2 * np.arange(N) * dt) + phi2 * np.sin(2 * np.pi * fj2 * np.arange(N) * dt)

x2 = [x0]
v2 = [v0]
for i in range(N - 1):
    x2.append(xf(x2[i], v2[i], delomega2[i], i * dt))
    v2.append(vf(x2[i], v2[i], delomega2[i], i * dt))
x2 = np.array(x2)
v2 = np.array(v2)
dpower2 = (np.abs(x2[1:]) ** 2 - np.abs(x2[:-1]) ** 2) / (xsqrinfty * dt)
theta2 = np.angle(np.conj(x2) * F0 * np.exp(1j * (omega0 + Domega) * np.arange(N) * dt))
start = int(T0 / dt)
for ind in np.where(theta2[start + 1:] - theta2[start:-1] > 1.9 * np.pi)[0]:
    theta2[start + ind + 1:] -= 2 * np.pi
for ind in np.where(theta2[start + 1:] - theta2[start:-1] < -1.9 * np.pi)[0]:
    theta2[start + ind + 1:] += 2 * np.pi

intervals = [[17.3, 18.0], [19.8, 20.4]]
intervals2 = [[16.4, 17.1], [22.7, 23.4]]
for interval in intervals:
    ax.axvspan(interval[0], interval[1], color = (0.317647, 0.654902, 0.752941), alpha = 0.3)
for interval in intervals2:
    ax.axvspan(interval[0], interval[1], color =  (1., 0.721569, 0.219608), alpha = 0.3)
ax.axhline(0, color = '0.7')
ax.plot((np.arange(N - 1) + 0.5) * dt, dpower, color = (0.317647, 0.654902, 0.752941), label = f'$f_j={fj}$\,Hz')
ax.plot((np.arange(N - 1) + 0.5) * dt, dpower2, color = (1., 0.721569, 0.219608), label = f'$f_j={fj2}$\,Hz')
ax.set_xlim(T0, Tf)
ax.set_ylim(-1., 1)
ax.set_ylabel(r'$\frac d{dt}\frac{|x(t)|^2}{|x_0(\infty)|^2}$\,[s$^{-1}$]')

ax.tick_params(which = 'both', direction = 'in')
plt.setp(ax.get_xticklabels(), visible = False)
ax.set_xticks([15, 17, 19, 21, 23, 25])
ax.set_yticks([-1., -0.5, 0, 0.5, 1.])
secxax = ax.secondary_xaxis('top')
secxax.tick_params(which = 'both', direction = 'in')
secxax.set_xticks([15, 17, 19, 21, 23, 25])
plt.setp(secxax.get_xticklabels(), visible = False)
secyax = ax.secondary_yaxis('right')
secyax.tick_params(which = 'both', direction = 'in')
ax.set_yticks([-1., -0.5, 0, 0.5, 1.])
plt.setp(secyax.get_yticklabels(), visible = False)

ax.legend(loc = 'upper right')

for interval in intervals:
    ax2.axvspan(interval[0], interval[1], color = (0.317647, 0.654902, 0.752941), alpha = 0.3)
for interval in intervals2:
    ax2.axvspan(interval[0], interval[1], color =  (1., 0.721569, 0.219608), alpha = 0.3)
ax2.axhline(np.pi / 2, color = '0.7')
ax2.plot(np.arange(N) * dt, theta, color = (0.317647, 0.654902, 0.752941))
ax2.plot(np.arange(N) * dt, theta2, color = (1., 0.721569, 0.219608))
ax2.set_xlim(T0, Tf)
ax2.set_ylim(-0.75 * np.pi, 1.75 * np.pi)
ax2.set_xlabel(r'$t$\,[s]')
ax2.set_ylabel(r'$\theta(t)$')

ax2.tick_params(which = 'both', direction = 'in')
ax2.set_xticks([15, 17, 19, 21, 23, 25])
ax2.set_yticks([-np.pi / 2, 0, np.pi / 2, np.pi, 3 * np.pi / 2])
ax2.set_yticklabels([r'$-\frac\pi2$', '0', r'$\frac\pi2$', r'$\pi$', r'$\frac{3\pi}2$'])
secxax2 = ax2.secondary_xaxis('top')
secxax2.tick_params(which = 'both', direction = 'in')
secxax2.set_xticks([15, 17, 19, 21, 23, 25])
plt.setp(secxax2.get_xticklabels(), visible = False)
secyax2 = ax2.secondary_yaxis('right')
secyax2.tick_params(which = 'both', direction = 'in')
secyax2.set_ticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
plt.setp(secyax2.get_yticklabels(), visible = False)

ax.set_axisbelow(False)
ax2.set_axisbelow(False)
fig.tight_layout()
#fig.show()
fig.savefig('phase.pdf')
