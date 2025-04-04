from matplotlib import ticker
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 14})
fig, ax = plt.subplots(figsize = (6., 4.))

data = np.load('../Data/schi.npz')
freqs = data['freqs']

gammaF = 2 * np.pi * 0.75 # in Hz
SF = lambda Dfreq: 4 * gammaF / (gammaF ** 2 + (4 * np.pi * Dfreq) ** 2)
SN = SF(0)

ax.plot(freqs, SF(freqs), color = (0.317647, 0.654902, 0.752941), lw = 1., label = 'Signal')
ax.plot(freqs, np.full_like(freqs, SN), color = (1., 0.721569, 0.219608), lw = 1., label = 'Noise')

ax.set_yscale('log')
ax.set_xlim(freqs[0], freqs[-1])
ax.set_ylim(1e-5, 2)
ax.set_xlabel(r'$f_F-f_0\,[\mathrm{Hz}]$')
ax.set_ylabel(r'$\frac{S_F(f_F)}{|F_0|^2}\,[\mathrm{Hz}^{-1}]$')

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

ax.legend(loc = 'upper right', bbox_to_anchor = (0.98, 0.91))

fig.tight_layout()
#fig.show()
fig.savefig('sf.pdf')
