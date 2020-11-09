import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.io as sio

from slice_plotter import Slice_Plotter
from slice_plotter import quick_slice_plot
from matplotlib.widgets import RectangleSelector
from matplotlib.patches import Rectangle

def select_callback_link(link_target, plotter):
    def select_callback(eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        plotter.pop_patch(link_target[4])
        link_target[0:5] = (x1, x2, y1, y2, plotter.ind)
        patch = Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='r')
        plotter.add_patch(patch, plotter.ind)
    return select_callback

def close_on_enter(event):
    if event.key == 'enter':
        plt.close()
    if event.key == 'q':
        print('Quitting...')
        quit()

def select_mask(mag, mag_mask, z_range=3, title_id=None, int_output=False, confirm=True, mask_type='int32'):

    cmap = plt.get_cmap('bone')
    cmap.set_bad('black')

    good = False
    artery_mask = np.zeros_like(mag, dtype='float32')

    title = f'Magnitude -- select target region'
    if title_id is not None:
        title = title_id + ' ' + title

    while not good:
        rect_vars = np.zeros(5).astype('int32')
        while np.all(rect_vars == 0):
            mag_fig, mag_ax = plt.subplots(1, 1)
            mag_plotter = Slice_Plotter(mag_ax, np.transpose(mag * mag_mask, axes=(1, 0, 2)), title=title, cmap=cmap)
            mag_fig.canvas.mpl_connect('scroll_event', mag_plotter.onscroll)
            selector = RectangleSelector(mag_ax, select_callback_link(rect_vars, mag_plotter),
                                        drawtype='box', useblit=True,
                                        button=[1],  # left click only
                                        minspanx=5, minspany=5,
                                        spancoords='pixels',
                                        interactive=True)


            mag_fig.canvas.mpl_connect('scroll_event', mag_plotter.onscroll)
            mag_fig.canvas.mpl_connect('key_press_event', close_on_enter)
            print(f'Locate arteries on labeling plane')
            plt.show()
            plt.close()
            del selector

        x1 = int(np.floor(np.min(rect_vars[0:2])))
        x2 = int(np.ceil(np.max(rect_vars[0:2])))
        y1 = int(np.floor(np.min(rect_vars[2:4])))
        y2 = int(np.ceil(np.max(rect_vars[2:4])))
        z = rect_vars[4]

        x_slice = slice(x1, x2 + 1)
        y_slice = slice(y1, y2 + 1)
        z_slice = slice(max(0, z - (z_range - 1) // 2), min(mag.shape[2] - 1, z + z_range // 2 + 1))
        artery_mask[x_slice, y_slice, z_slice] = 1

        artery_mask[artery_mask == 0] = np.nan
        artery_mask = artery_mask * mag_mask
        
        


        mag_fig, mag_ax = plt.subplots(1, 1)
        mag_plotter = Slice_Plotter(mag_ax, np.transpose((mag * artery_mask)[..., z_slice], axes=(1, 0, 2)), f'Target mask', cmap=cmap)
        mag_fig.canvas.mpl_connect('scroll_event', mag_plotter.onscroll)
        mag_fig.canvas.mpl_connect('key_press_event', close_on_enter)

        print('Confirm mask -- close when done')
        plt.show(block=True)
        plt.close()

        valid = False
        if not confirm:
            valid = True
            good = True
        while not valid:
            confirm = input('Good? [Y/N/Q]: ').upper()
            valid = True
            if confirm == 'Y':
                good = True
            elif confirm == 'N':
                good = False
            elif confirm == 'Q':
                print('Quitting...')
                quit()
            else:
                valid = False

        if int_output:
            artery_mask[artery_mask == np.nan] = 0
            artery_mask = artery_mask.astype(mask_type)

    return artery_mask