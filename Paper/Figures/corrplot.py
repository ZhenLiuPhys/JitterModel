import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 14})
fig, ax = plt.subplots(figsize = (6., 4.))

data = np.load('../Data/corr.npz')
omega0 = data['omega0']
gamma = data['gamma']
delomega0 = data['delomega0']
tau = data['tau']
fj = data['fj']
tcut = data['tcut']
T = data['T']
Kx = data['Kx']
Kxstd = data['Kxstd']

corr_fit = lambda t, tau1, tau2, fosc, c1, c2, c3: (c1 * np.cos(2 * np.pi * fosc * t) + c2 * np.sin(2 * np.pi * fosc * t)) * np.exp(-t / tau1) + c3 * np.exp(-t / tau2)
#corr_fit = lambda t, tau1, tau2, fosc, c1, c2, c3, tau3, fosc2, c4, c5: (c1 * np.cos(2 * np.pi * fosc * t) + c2 * np.sin(2 * np.pi * fosc * t)) * np.exp(-t / tau1) + c3 * np.exp(-t / tau2) + (c4 * np.cos(2 * np.pi * fosc2 * t) + c5 * np.sin(2 * np.pi * fosc2 * t)) * np.exp(-t / tau3)
TAU1 = tau
TAU2 = 2 / gamma
FOSC = fj
C1 = 4 * delomega0 ** 2 * tau ** 2 * ((4 * np.pi * fj * tau) ** 2 + gamma ** 2 * tau ** 2 - 4)
C2 = 4 * delomega0 ** 2 * tau ** 2 * 16 * np.pi * fj * tau
C3 = 8 * delomega0 ** 2 * tau / gamma * ((4 * np.pi * fj * tau) ** 2 - gamma ** 2 * tau ** 2 + 4)
CORR_PARAMS = np.array([TAU1, TAU2, FOSC, C1, C2, C3])
#TAU3 = 2 * tau
#FOSC2 = 2 * fj
#C4 = 0
#C5 = 0
#CORR_PARAMS = np.array([TAU1, TAU2, FOSC, C1, C2, C3, TAU3, FOSC2, C4, C5])

dt = min(tau, 1 / (2 * np.pi * fj)) / 10 # in s
Ncut = int(np.ceil(tcut / dt))
N = int(np.ceil(T / dt))

xsqrinfty = 1 / omega0 ** 2 / gamma ** 2
std = np.sqrt(Kxstd ** 2 + np.exp(-gamma * tcut) * np.abs(Kx) ** 2)
popt, pcov, infodict, mesg, ier = optimize.curve_fit(corr_fit, np.arange(N - Ncut) * dt, Kx.real / xsqrinfty, sigma = std / xsqrinfty, p0 = CORR_PARAMS, full_output = True)

ax.plot(np.arange(N - Ncut) * dt, corr_fit(np.arange(N - Ncut) * dt, *popt), color = 'k', label = 'Fit')
ax.fill_between(np.arange(N - Ncut) * dt, (Kx.real - std) / xsqrinfty, (Kx.real + std) / xsqrinfty, color = (0.317647, 0.654902, 0.752941), alpha = 0.5, zorder = 2)
ax.plot(np.arange(N - Ncut) * dt, Kx.real / xsqrinfty, color = (0.317647, 0.654902, 0.752941), lw = 0.5, label = 'Data')

ax.set_xlim(1e-3, T - tcut)
ax.set_xscale('log')
ax.set_ylim(-0.01, 0.13)
ax.set_xlabel(r'$t$\,[s]')
ax.set_ylabel(r'$\frac{\hat K_x(t)}{|x_0(\infty)|^2}$')

ax.tick_params(which = 'both', direction = 'in')
secxax = ax.secondary_xaxis('top')
secxax.tick_params(which = 'both', direction = 'in')
plt.setp(secxax.get_xticklabels(), visible = False)
secyax = ax.secondary_yaxis('right')
secyax.tick_params(which = 'both', direction = 'in')
plt.setp(secyax.get_yticklabels(), visible = False)

legend = ax.legend(loc = 'lower left')
legend.legend_handles[1].set_linewidth(1.5)

print(f'Fit tau1 = {popt[0]:.3f}')
print(f'Fit tau2 = {popt[1]:.2f}')
print(f'Fit fK = {popt[2]:.1f}')
a = -1 + 12 * np.pi ** 2 * fj ** 2 * tau ** 2 + 3 * delomega0 ** 2 * tau ** 2
b = 1 + 36 * np.pi ** 2 * fj ** 2 * tau ** 2 - 4.5 * delomega0 ** 2 * tau ** 2
beta1 = -gamma / 2 - (2 - (1 + 1j * np.sqrt(3)) / 2 * a / (np.sqrt(a ** 3 + b ** 2) + b) ** (1 / 3) + (1 - 1j * np.sqrt(3)) / 2 * (np.sqrt(a ** 3 + b ** 2) + b) ** (1 / 3)) / (3 * tau)
beta2 = -gamma / 2 - (2 + a / (np.sqrt(a ** 3 + b ** 2) + b) ** (1 / 3) - (np.sqrt(a ** 3 + b ** 2) + b) ** (1 / 3)) / (3 * tau)
print(f'Analytic tau1 = {-1 / beta1.real:.3f}')
print(f'Analytic tau2 = {-1 / beta2:.2f}')
print(f'Analytic fK = {beta1.imag / (2 * np.pi):.1f}')

fig.tight_layout()
fig.show()
#fig.savefig('corr.pdf')
