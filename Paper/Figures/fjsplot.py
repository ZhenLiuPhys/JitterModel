from matplotlib import ticker
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 14})
fig, ax = plt.subplots(figsize = (6., 4.))

dataG = np.load('../Data/fjs_gaussian.npz')
omega0 = dataG['omega0']
gamma = dataG['gamma']
delomega0 = dataG['delomega0']
tau = dataG['tau']
T = dataG['T']
fjs = dataG['fjs']
fj0 = 45 # in Hz
xsqrmeanG = dataG['xsqrmean']
xsqrstdG = dataG['xsqrstd']

dataDMP = np.load('../Data/fjs_DMP.npz')
xsqrmeanDMP = dataDMP['xsqrmean']
xsqrstdDMP = dataDMP['xsqrstd']

rho = gamma * tau * (2 + gamma * tau) / ((2 + gamma * tau) ** 2 + (4 * np.pi * fjs * tau) ** 2)
alpha = 4 * delomega0 ** 2 * rho / gamma ** 2
ax.axvline(fj0, ls = '--', color = '0.7')
ax.plot(fjs, 1 / (1 + alpha), color = 'k', label = 'Analytic')

xsqrinfty = 1 / omega0 ** 2 / gamma ** 2
stdG = np.sqrt(xsqrstdG ** 2 + np.exp(-gamma * T) * xsqrmeanG ** 2)
stdDMP = np.sqrt(xsqrstdDMP ** 2 + np.exp(-gamma * T) * xsqrmeanDMP ** 2)

ax.fill_between(fjs, (xsqrmeanG - stdG) / xsqrinfty, (xsqrmeanG + stdG) / xsqrinfty, color = (0.317647, 0.654902, 0.752941), alpha = 0.5, zorder = 2)
ax.plot(fjs, xsqrmeanG / xsqrinfty, color = (0.317647, 0.654902, 0.752941), lw = 0.5, label = 'Gaussian')
ax.fill_between(fjs, (xsqrmeanDMP - stdDMP) / xsqrinfty, (xsqrmeanDMP + stdDMP) / xsqrinfty, color = (1., 0.721569, 0.219608), alpha = 0.5, zorder = 2)
ax.plot(fjs, xsqrmeanDMP / xsqrinfty, color = (1., 0.721569, 0.219608), lw = 0.5, label = 'DMP')

ax.set_xlim(fjs[0], fjs[-1])
ax.set_xscale('log')
ax.set_ylim(1e-2, 1.)
ax.set_yscale('log')
ax.set_xlabel(r'$f_j$\,[Hz]')
ax.set_ylabel(r'$\frac{\langle|x(t)|^2\rangle_\infty}{|x_0(\infty)|^2}$')

class CustomTicker(ticker.LogFormatterSciNotation): 
    def __call__(self, x, pos = None): 
        if x not in np.concatenate((0.1 * np.arange(1, 10), np.arange(1, 10), 10 * np.arange(1, 10))): 
            return ticker.LogFormatterSciNotation.__call__(self, x, pos = None) 
        else:
            return "{x:g}".format(x = x)

ax.tick_params(which = 'both', direction = 'in')
ax.yaxis.set_major_formatter(CustomTicker())
secxax = ax.secondary_xaxis('top')
secxax.tick_params(which = 'both', direction = 'in')
plt.setp(secxax.get_xticklabels(), visible = False)
secyax = ax.secondary_yaxis('right')
secyax.tick_params(which = 'both', direction = 'in')
plt.setp(secyax.get_yticklabels(), visible = False)

#legend = ax.legend(loc = 'upper left')
#legend.legend_handles[1].set_linewidth(1.5)
#legend.legend_handles[2].set_linewidth(1.5)

fig.tight_layout()
#fig.show()
fig.savefig('fjs.pdf')
