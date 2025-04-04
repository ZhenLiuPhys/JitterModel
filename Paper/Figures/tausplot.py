import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 14})
fig, ax = plt.subplots(figsize = (6., 4.))

dataG = np.load('../Data/taus_gaussian.npz')
omega0 = dataG['omega0']
gamma = dataG['gamma']
delomega0 = dataG['delomega0']
fj = dataG['fj']
T = dataG['T']
taus = dataG['taus']
tau0 = 1 / (5 * np.pi) # in s
xsqrmeanG = dataG['xsqrmean']
xsqrstdG = dataG['xsqrstd']

dataDMP = np.load('../Data/taus_DMP.npz')
xsqrmeanDMP = dataDMP['xsqrmean']
xsqrstdDMP = dataDMP['xsqrstd']

rho = gamma * taus * (2 + gamma * taus) / ((2 + gamma * taus) ** 2 + (4 * np.pi * fj * taus) ** 2)
alpha = 4 * delomega0 ** 2 * rho / gamma ** 2
ax.axvline(tau0, ls = '--', color = '0.7')
ax.plot(taus, 1 / (1 + alpha), color = 'k', label = 'Analytic')

xsqrinfty = 1 / omega0 ** 2 / gamma ** 2
stdG = np.sqrt(xsqrstdG ** 2 + np.exp(-gamma * T) * xsqrmeanG ** 2)
stdDMP = np.sqrt(xsqrstdDMP ** 2 + np.exp(-gamma * T) * xsqrmeanDMP ** 2)

ax.fill_between(taus, (xsqrmeanG - stdG) / xsqrinfty, (xsqrmeanG + stdG) / xsqrinfty, color = (0.317647, 0.654902, 0.752941), alpha = 0.5, zorder = 2)
ax.plot(taus, xsqrmeanG / xsqrinfty, color = (0.317647, 0.654902, 0.752941), lw = 0.5, label = 'Gaussian')
ax.fill_between(taus, (xsqrmeanDMP - stdDMP) / xsqrinfty, (xsqrmeanDMP + stdDMP) / xsqrinfty, color = (1., 0.721569, 0.219608), alpha = 0.5, zorder = 2)
ax.plot(taus, xsqrmeanDMP / xsqrinfty, color = (1., 0.721569, 0.219608), lw = 0.5, label = 'DMP')

ax.set_xlim(taus[0], taus[-1])
ax.set_xscale('log')
ax.set_ylim(0.37, 1.)
ax.set_xlabel(r'$\tau$\,[s]')
ax.set_ylabel(r'$\frac{\langle|x(t)|^2\rangle_\infty}{|x_0(\infty)|^2}$')

ax.tick_params(which = 'both', direction = 'in')
secxax = ax.secondary_xaxis('top')
secxax.tick_params(which = 'both', direction = 'in')
plt.setp(secxax.get_xticklabels(), visible = False)
secyax = ax.secondary_yaxis('right')
secyax.tick_params(which = 'both', direction = 'in')
plt.setp(secyax.get_yticklabels(), visible = False)

legend = ax.legend(loc = 'upper left')
legend.legend_handles[1].set_linewidth(1.5)
legend.legend_handles[2].set_linewidth(1.5)

fig.tight_layout()
#fig.show()
fig.savefig('taus.pdf')
