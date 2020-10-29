import numpy as np
import matplotlib.pyplot as plt

class Slice_Plotter:
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
        self.ind = 0

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

