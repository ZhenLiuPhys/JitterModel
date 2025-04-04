from matplotlib import ticker
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 14})
fig, ax = plt.subplots(figsize = (6., 4.))

delf0 = 3 # in Hz
T = 0.1 # in s

# Params: tau (in s), fj (in Hz), mode, color, label
# tau plot
params = [[1 / (0.5 * np.pi), 45, 'Gaussian', (0.317647, 0.654902, 0.752941), r'$1/\pi\tau=0.5$\,Hz'],
          [1 / (5 * np.pi), 45, 'Gaussian', 'k', r'$1/\pi\tau=5$\,Hz'],
          [1 / (50 * np.pi), 45, 'Gaussian', (0.921569, 0.494118, 0.431373), r'$1/\pi\tau=50$\,Hz']]
textlabel = r'$f_j=45$\,Hz'
legendfont = 11
filename = 'jittering_tau.pdf'
# fj plot
##params = [[1 / (5 * np.pi), 0, 'Gaussian', (0.317647, 0.654902, 0.752941), r'$f_j=0$\,Hz'],
##          [1 / (5 * np.pi), 45, 'Gaussian', 'k', r'$f_j=45$\,Hz'],
##          [1 / (5 * np.pi), 200, 'Gaussian', (0.921569, 0.494118, 0.431373), r'$f_j=200$\,Hz']]
##textlabel = r'$1/\pi\tau=5$\,Hz'
##legendfont = 12
##filename = 'jittering_fj.pdf'
# modes plot
##params = [[1 / (5 * np.pi), 45, 'Gaussian', 'k', r'Gaussian'],
##          [1 / (5 * np.pi), 45, 'DMP', (0.317647, 0.654902, 0.752941), r'DMP']]
##textlabel = r'$f_j=45$\,Hz,~$1/\pi\tau=5$\,Hz'
##legendfont = 14
##filename = 'jittering_modes.pdf'

for param in params:
    tau, fj, mode, color, label = param
    if fj != 0: dt = min(tau, 1 / (2 * np.pi * fj)) / 10 # in s
    else: dt = tau / 100 # in s
    N = int(np.ceil(T / dt))

    if mode == 'Gaussian':
        etahat = delf0 * np.random.randn(N)
        eta = np.exp(-np.arange(N) * dt / tau) * (etahat[0] + np.sqrt(1 - np.exp(-2 * dt / tau)) * np.insert(np.cumsum(etahat[1:] * np.exp(np.arange(1, N) * dt / tau)), 0, 0))
        phihat = delf0 * np.random.randn(N)
        phi = np.exp(-np.arange(N) * dt / tau) * (phihat[0] + np.sqrt(1 - np.exp(-2 * dt / tau)) * np.insert(np.cumsum(phihat[1:] * np.exp(np.arange(1, N) * dt / tau)), 0, 0))
    if mode == 'DMP':
        etaflips = np.random.choice([1, -1], N, p = [(1 + np.exp(-dt / tau)) / 2, (1 - np.exp(-dt / tau)) / 2])
        eta = delf0 * np.random.choice([1, -1]) * np.cumprod(etaflips)
        phiflips = np.random.choice([1, -1], N, p = [(1 + np.exp(-dt / tau)) / 2, (1 - np.exp(-dt / tau)) / 2])
        phi = delf0 * np.random.choice([1, -1]) * np.cumprod(phiflips)
    delf = eta * np.cos(2 * np.pi * fj * np.arange(N) * dt) + phi * np.sin(2 * np.pi * fj * np.arange(N) * dt)

    ax.plot(np.arange(N) * dt, delf, color = color, label = label)

ax.set_xlim(0, T)
ax.set_ylim(-4 * delf0, 4 * delf0)
ax.set_xlabel(r'$t$\,[s]')
ax.set_ylabel(r'$\delta f(t)$\,[Hz]')

ax.tick_params(which = 'both', direction = 'in')
ax.xaxis.set_major_locator(ticker.FixedLocator(0.02 * np.arange(6)))
ax.xaxis.set_major_formatter(ticker.FixedFormatter(['0', '0.02', '0.04', '0.06', '0.08', '0.1']))
ax.yaxis.set_major_locator(ticker.FixedLocator(delf0 * np.arange(-4, 5, 2)))
secxax = ax.secondary_xaxis('top')
secxax.tick_params(which = 'both', direction = 'in')
plt.setp(secxax.get_xticklabels(), visible = False)
secyax = ax.secondary_yaxis('right')
secyax.tick_params(which = 'both', direction = 'in')
plt.setp(secyax.get_yticklabels(), visible = False)

ax.legend(loc = 'lower center', ncols = len(params), fontsize = legendfont)
ax.text(0.096, 10.6, textlabel, ha = 'right', va = 'top', bbox = dict(boxstyle = 'round', facecolor = 'white', alpha = 0.5))

fig.tight_layout()
#fig.show()
fig.savefig(filename)
