from matplotlib import ticker
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 14})
fig, ax = plt.subplots(figsize = (6., 4.))

data = np.load('../Data/schi.npz')
omega0 = data['omega0']
gamma = data['gamma']
delomega0 = data['delomega0']
tau = data['tau']
fj = data['fj']
tcut = data['tcut']
T = data['T']
freqs = data['freqs']
Schi = data['Schi']
Schistd = data['Schistd']

gammaF = 2 * np.pi * 0.75 # in Hz
xsqrinfty = 1 / omega0 ** 2 / gamma ** 2
SF = lambda Dfreq: 4 * gammaF / (gammaF ** 2 + (4 * np.pi * Dfreq) ** 2)
SN = SF(0)

dt = min(tau, 1 / (2 * np.pi * fj)) / 10 # in s
Ncut = int(np.ceil(tcut / dt))
N = int(np.ceil(T / dt))
maxfreq = 2 * max(gamma / (2 * np.pi), delomega0 / (2 * np.pi), 1 / (2 * np.pi * tau), fj) # in Hz
Nmax = int(np.ceil(maxfreq * ((N - Ncut) * dt)))
freqs = np.arange(-Nmax, Nmax + 1) / ((N - Ncut) * dt) # in Hz

Sj = lambda f: delomega0 ** 2 / tau * (1 / (tau ** -2 + 4 * np.pi ** 2 * (f - fj) ** 2) + 1 / (tau ** -2 + 4 * np.pi ** 2 * (f + fj) ** 2))
Schi_smooth = lambda f, Dfreq: 4 * Sj(f - Dfreq) / omega0 ** 2 / (gamma ** 2 + (4 * np.pi * Dfreq) ** 2) / (gamma ** 2 + (4 * np.pi * f) ** 2)
Schi_delta_noj = lambda Dfreq: 1 / omega0 ** 2 / (gamma ** 2 + (4 * np.pi * Dfreq) ** 2)
Schi_delta = lambda Dfreq: Schi_delta_noj(Dfreq) * (1 - 8 * delomega0 ** 2 * tau * ((2 + 4j * np.pi * Dfreq * tau + gamma * tau) / (gamma + 4j * np.pi * Dfreq) / ((2 + 4j * np.pi * Dfreq * tau + gamma * tau) ** 2 + (4 * np.pi * fj * tau) ** 2)).real)
delta = (N - Ncut) * dt * np.identity(2 * Nmax + 1)

Sx_sig = np.sum(Schi * SF(freqs[:,None]), axis = 0) / (N - Ncut) / dt / xsqrinfty
Sx_sig_std = np.sqrt(np.sum((Schistd * SF(freqs[:,None]) / (N - Ncut) / dt / xsqrinfty) ** 2, axis = 0) + np.exp(-gamma * tcut) * Sx_sig ** 2)
Sx_sig_noj = np.sum(Schi_delta_noj(freqs[:,None]) * delta * SF(freqs[:,None]), axis = 0) / (N - Ncut) / dt / xsqrinfty
Sx_sig_anal = np.sum((Schi_delta(freqs[:,None]) * delta + Schi_smooth(freqs[None], freqs[:,None])) * SF(freqs[:,None]), axis = 0) / (N - Ncut) / dt / xsqrinfty

Sx_noise = np.sum(Schi * SN, axis = 0) / (N - Ncut) / dt / xsqrinfty
Sx_noise_std = np.sqrt(np.sum((Schistd * SN / (N - Ncut) / dt / xsqrinfty) ** 2, axis = 0) + np.exp(-gamma * tcut) * Sx_noise ** 2)
Sx_noise_noj = np.sum(Schi_delta_noj(freqs[:,None]) * delta * SN, axis = 0) / (N - Ncut) / dt / xsqrinfty
Sx_noise_anal = np.sum((Schi_delta(freqs[:,None]) * delta + Schi_smooth(freqs[None], freqs[:,None])) * SN, axis = 0) / (N - Ncut) / dt / xsqrinfty

ax.plot(freqs, Sx_sig_anal, color = (0.317647, 0.654902, 0.752941), linestyle = ':', lw = 1.)
ax.plot(freqs, Sx_sig_noj, color = (0.317647, 0.654902, 0.752941), linestyle = '--', lw = 1.)
ax.fill_between(freqs, Sx_sig - Sx_sig_std, Sx_sig + Sx_sig_std, color = (0.317647, 0.654902, 0.752941), alpha = 0.5, zorder = 2)
ax.plot(freqs, Sx_sig, color = (0.317647, 0.654902, 0.752941), lw = 0.5)
ax.plot(freqs, Sx_noise_anal, color = (1., 0.721569, 0.219608), linestyle = ':', lw = 1.)
ax.plot(freqs, Sx_noise_noj, color = (1., 0.721569, 0.219608), linestyle = '--', lw = 1.)
ax.fill_between(freqs, Sx_noise - Sx_noise_std, Sx_noise + Sx_noise_std, color = (1., 0.721569, 0.219608), alpha = 0.5, zorder = 2)
ax.plot(freqs, Sx_noise, color = (1., 0.721569, 0.219608), lw = 0.5)

ax.set_yscale('log')
ax.set_xlim(freqs[0], freqs[-1])
ax.set_ylim(1e-8, 1)
ax.set_xlabel(r'$f-f_0\,[\mathrm{Hz}]$')
ax.set_ylabel(r'$\frac{S_x(f)}{|x_0(\infty)|^2}\,[\mathrm{Hz}^{-1}]$')

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

inset = ax.inset_axes([0.1, 0.6, 0.3, 0.3], xlim = (-0.2, 0.2), ylim = (0.1, 1.))
inset.set_yscale('log')
inset.plot(freqs, Sx_sig_anal, color = (0.317647, 0.654902, 0.752941), linestyle = ':', lw = 1.)
inset.plot(freqs, Sx_sig_noj, color = (0.317647, 0.654902, 0.752941), linestyle = '--', lw = 1.)
inset.fill_between(freqs, Sx_sig - Sx_sig_std, Sx_sig + Sx_sig_std, color = (0.317647, 0.654902, 0.752941), alpha = 0.5, zorder = 2)
inset.plot(freqs, Sx_sig, color = (0.317647, 0.654902, 0.752941), lw = 0.5)
inset.plot(freqs, Sx_noise_anal, color = (1., 0.721569, 0.219608), linestyle = ':', lw = 1.)
inset.plot(freqs, Sx_noise_noj, color = (1., 0.721569, 0.219608), linestyle = '--', lw = 1.)
inset.fill_between(freqs, Sx_noise - Sx_noise_std, Sx_noise + Sx_noise_std, color = (1., 0.721569, 0.219608), alpha = 0.5, zorder = 2)
inset.plot(freqs, Sx_noise, color = (1., 0.721569, 0.219608), lw = 0.5)

inset.tick_params(which = 'both', direction = 'in')
insetsecxax = inset.secondary_xaxis('top')
insetsecxax.tick_params(which = 'both', direction = 'in')
plt.setp(insetsecxax.get_xticklabels(), visible = False)
insetsecyax = inset.secondary_yaxis('right')
insetsecyax.tick_params(which = 'both', direction = 'in')
plt.setp(insetsecyax.get_yticklabels(), visible = False)

inset.set_xticks([-0.2, 0., 0.2])
inset.set_xticklabels([r'$-0.2$', r'$0$', r'$0.2$'])
insetsecxax.set_ticks([-0.2, 0., 0.2])
inset.yaxis.set_major_formatter(CustomTicker())
inset.yaxis.set_minor_formatter(ticker.NullFormatter())
insetsecyax.yaxis.set_minor_formatter(ticker.NullFormatter())

ax.indicate_inset_zoom(inset, ec = '0.5')
fig.tight_layout()
fig.show()
#fig.savefig('sx.pdf')
