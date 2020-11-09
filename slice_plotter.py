import numpy as np
import matplotlib.pyplot as plt

class Slice_Plotter:
    def __init__(self, ax, X, title='', vmin=None, vmax=None, cmap=None, patches=None):
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

        if patches is None:
            patches = [[] for _ in range(self.slices)]
        self.patches = patches

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
        self.ax.patches = []
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_ylabel('slice %s (scroll)' % self.ind)
        self.ax.patches = []
        for patch in self.patches[self.ind]:
            self.ax.add_patch(patch)
        self.im.axes.figure.canvas.draw()

    def add_patch(self, patch, ind):
        self.patches[ind].append(patch)
        self.update()

    def pop_patch(self, ind):
        if len(self.patches[ind]) > 0:
            self.patches[ind].pop()
        self.update()

def quick_slice_plot(X, title='', cmap=None, vmin=None, vmax=None, patches=None):
    quick_fig, quick_ax = plt.subplots(1, 1)
    plotter_quick = Slice_Plotter(quick_ax, np.transpose(X, axes=(1, 0, 2)), title=title, cmap=cmap, patches=patches)
    quick_fig.canvas.mpl_connect('scroll_event', plotter_quick.onscroll)
    plt.show(block=True)
