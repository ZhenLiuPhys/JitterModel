from matplotlib import ticker
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 14})
fig, ax = plt.subplots(figsize = (6., 4.))

dataG = np.load('../Data/hist_gaussian.npz')
xhistG = dataG['xhist']
dataDMP = np.load('../Data/hist_DMP.npz')
xhistDMP = dataDMP['xhist']

xlim1 = 0.17
xlim2 = 1.0
ylim1 = 3e-3
ylim2 = 10.
ax.set_yscale('log')
ax.set_xlim(xlim1, xlim2)
ax.set_ylim(ylim1, ylim2)
ax.set_xlabel(r'$\frac{|x(T)|^2}{|x_0(\infty)|^2}$')
ax.set_ylabel('Probability')

xsqrnum = 0.87
ax.axvline(xsqrnum, ls = '--', color = '0.4')

ax.hist(xhistDMP, bins = np.linspace(xlim1, xlim2, 50), density = True, color = (1., 0.721569, 0.219608), alpha = 0.5, edgecolor = (1., 0.721569, 0.219608), label = 'DMP')
ax.hist(xhistG, bins = np.linspace(xlim1, xlim2, 50), density = True, color = (0.317647, 0.654902, 0.752941), alpha = 0.5, edgecolor = (0.317647, 0.654902, 0.752941), label = 'Gaussian')

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

handles, labels = ax.get_legend_handles_labels()
legend = ax.legend(handles[::-1], labels[::-1], loc = 'upper left')

fig.tight_layout()
fig.show()
#fig.savefig('hist.pdf')
