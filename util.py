# Author: Demetris Marnerides
import torch
from torch import nn
import time
import warnings
import matplotlib
from matplotlib import pyplot as plt


## https://stackoverflow.com/questions/22873410/how-do-i-fix-the-deprecation-warning-that-comes-with-pylab-pause
warnings.filterwarnings("ignore", ".*GUI is implemented.*")


def wait_and_exit():
    while True:
        try:
            time.sleep(0.1)
        except KeyboardInterrupt:
            print()
            break
    exit()


## https://stackoverflow.com/questions/45729092/make-interactive-matplotlib-window-not-pop-to-front-on-each-update-windows-7
def _mypause(interval):
    backend = matplotlib.rcParams['backend']
    if backend in matplotlib.rcsetup.interactive_bk:
        figManager = matplotlib._pylab_helpers.Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw()
            canvas.start_event_loop(interval)
            return

    # No on-screen figure is active, so sleep() is all we need.
    time.sleep(interval)


_funcs = dict(
    line=lambda x: 0.7 * x,
    pow_4=lambda x: x ** 4,
    exponential=lambda x: (x + 1).exp() - 2,
    poly=lambda x: 0.1 * x ** 4 + 0.3 * x ** 3 + 0.2 * x ** 2,
    sine=lambda x: 0.4 * (torch.sin(8 * x + 0.2) + 1),
    sine2=lambda x: 0.4 * (torch.sin(15 * x + 0.2) + 1),
)

PAUSE_TIME = [1]


def set_pause_time(pause):
    PAUSE_TIME[0] = pause


def get_data(linetype, num_points, noise=0):
    x = torch.ones(num_points).uniform_(0, 1)
    y = _funcs[linetype](x)
    if noise > 0:
        y += torch.zeros_like(y).normal_(0, noise)
    return x, y


class Plotter:
    def __init__(self, title='Regression'):
        self.title = title

    def __enter__(self):
        plt.figure(1)
        plt.ion()
        plt.clf()
        plt.axis([-0.1, 1.1, -0.1, 1.1])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(self.title)
        for (x, y, mode) in kept:
            plt.plot(x, y, **_PLOT_DATA[mode])
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        plt.show(block=False)
        _mypause(PAUSE_TIME[0])
        plt.ioff()


def plot(x, y, mode, keep=True):
    x, y = _to_numpy(x, y)
    if keep:
        _store(x, y, mode)
    with Plotter():
        plt.scatter(x, y, **_PLOT_DATA[mode])


kept = []


def _store(x, y, mode):
    kept.append((x.copy(), y.copy(), mode))


def _to_numpy(*args):
    return tuple(x.detach().numpy() if torch.is_tensor(x) else x for x in args)


def plot_function(func, keep=True):
    x = torch.linspace(0, 1, 200)
    if isinstance(func, nn.Module):
        y = func(x.unsqueeze(-1))[:, -1]
    else:
        y = func(x)
    x, y = _to_numpy(x, y)
    if keep:
        _store(x, y, 'function')
    with Plotter():
        plt.plot(x, y, c='blue', marker='')


_PLOT_DATA = {
    'function': dict(c='blue', marker='', linestyle='-'),
    'training': dict(c='red', marker='x', linestyle='None'),
    'validation': dict(c='green', marker='P', linestyle='None'),
}
