import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import scipy
# from shimmingtoolbox.optimizer.basic_optimizer import Optimizer as Opt
from shimmingtoolbox.optimizer.lsq_optimizer import LSQ_Optimizer as Opt

class IndexTracker:
    def __init__(self, ax, X, title, vmin=None, vmax=None, cmap=None):
        self.ax = ax
        ax.set_title(title)

        if cmap is None:
            cmap = plt.get_cmap('rainbow')
            cmap.set_bad('black')
        
        if vmin is None:
            vmin=np.nanmin(X)

        if vmax is None:
            vmax=np.nanmax(X)

        self.X = X
        self.rows, self.cols, self.slices = X.shape
        self.ind = self.slices//2

        self.im = ax.imshow(self.X[:, :, self.ind], vmin=vmin, vmax=vmax, cmap=cmap)
        self.update()

    def onscroll(self, event):
        ind_max = self.slices
        if event.button == 'up':
            self.ind = (self.ind + 1)
        else:
            self.ind = (self.ind - 1)
        if self.ind >= ind_max:
            self.ind = ind_max - 1
        if self.ind < 0:
            self.ind = 0
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_ylabel('slice %s (scroll)' % self.ind)
        self.im.axes.figure.canvas.draw()

def compare_shim(coils, unshimmed, mask, mask_origin=(0, 0, 0), bounds=None, magnitude=None, magnitude_mask=None):

    print('Making optimizer')
    opt = Opt(coils)
    print('Optimizer made')

    mask_range = tuple([slice(mask_origin[i], mask_origin[i] + mask.shape[i]) for i in range(3)])
    mask_vec = mask.reshape((-1,))

    currents = opt.optimize(unshimmed, mask, mask_origin=mask_origin, bounds=bounds)
    print("Currents:")
    print(currents)
    shimmed = np.sum(currents.reshape((1, 1, 1, -1)) * coils, axis=3, keepdims=False) + unshimmed
    print(f'Standard deviations -- Unshimmed:{round(np.std(unshimmed[mask_range] * mask), 3)}, ' +
            f'Shimmed: {round(np.std(shimmed[mask_range] * mask), 3)}')

    if magnitude is not None:
        if magnitude_mask is None:
            magnitude_mask = np.ones_like(magnitude)
        
        cmap = plt.get_cmap('bone')
        cmap.set_bad('black')
        mag_fig, mag_ax = plt.subplots(1, 1)
        X_mag = magnitude * magnitude_mask
        X_mag[magnitude_mask == 0] = np.nan
        tracker_mag = IndexTracker(mag_ax, X_mag, 'Magnitude', cmap=cmap)
        mag_fig.canvas.mpl_connect('scroll_event', tracker_mag.onscroll)
        plt.show(block=False)

    coil_fig, coil_ax = plt.subplots(1, 1)
    X_coil = coils[:, :, :, -3]
    X_coil[X_coil == 0] = np.nan
    tracker_coil = IndexTracker(coil_ax, X_coil, 'Coil demo')
    coil_fig.canvas.mpl_connect('scroll_event', tracker_coil.onscroll)
    plt.show(block=False)

    scroll_fig, scroll_axs = plt.subplots(2, 1)
    scroll_fig.tight_layout(pad=3)
    X_unsh = unshimmed[mask_range] * mask
    X_unsh[mask == 0] = np.nan
    X_sh = shimmed[mask_range] * mask
    X_sh[mask == 0] = np.nan
    scale_max = max(abs(np.nanmax(X_unsh)), abs(np.nanmax(X_sh)), abs(np.nanmin(X_unsh)), abs(np.nanmin(X_sh)))
    scale_min = -1 * scale_max
    tracker_unsh = IndexTracker(scroll_axs[0], X_unsh, 'Unshimmed', vmin=scale_min, vmax=scale_max)
    tracker_sh = IndexTracker(scroll_axs[1], X_sh, 'Shimmed', vmin=scale_min, vmax=scale_max)
    scroll_fig.canvas.mpl_connect('scroll_event', tracker_unsh.onscroll)
    scroll_fig.canvas.mpl_connect('scroll_event', tracker_sh.onscroll)
    plt.show(block=False)

    hist_fig, hist_axs = plt.subplots(2, 1)
    hist_fig.tight_layout(pad=3)
    hist_axs[0].hist(np.reshape(unshimmed[mask_range] * mask, (-1,))[mask_vec != 0], range=[scale_min, scale_max], bins=100)
    hist_axs[0].set_title('Unshimmed')
    hist_axs[1].hist(np.reshape(shimmed[mask_range] * mask, (-1,))[mask_vec != 0], range=[scale_min, scale_max], bins=100)
    hist_axs[1].set_title('Shimmed')
    plt.show()