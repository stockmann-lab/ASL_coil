import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.io as sio

from slice_plotter import Slice_Plotter

cmap = plt.get_cmap('bone')
cmap.set_bad('black')
view_only = (input('Magnitude only? [Y/N] ') == 'Y')
if view_only:
    check_fmap = (input('Check field map? [Y/N] ') == 'Y')
if not view_only:
    recheck = (input('Check previous? [Y/N] ') == 'Y')

for fn in range(25):
    # Check if exists
    if not os.path.exists(f'maps/connectome_L1/P{fn + 1}_mag.npy'):
        mag = sio.loadmat(f'data/connectome_L1/P{fn + 1}_mag.mat')['mag']
        np.save(f'maps/connectome_L1/P{fn + 1}_mag.npy', mag)
    if not os.path.exists(f'maps/connectome_L1/P{fn + 1}_fmap.npy'):
        fmap = sio.loadmat(f'data/connectome_L1/P{fn + 1}_fmap.mat')['fmap']
        np.save(f'maps/connectome_L1/P{fn + 1}_fmap.npy', fmap)
    mag = np.load(f'maps/connectome_L1/P{fn + 1}_mag.npy')
    fmap = np.load(f'maps/connectome_L1/P{fn + 1}_fmap.npy')
    X, Y, Z = mag.shape
    mag_mask = np.where(mag > 0.25 * np.mean(mag), 1, np.nan)
    artery_mask = np.zeros((X, Y, Z))

    good = False
    mask_path = f'maps/connectome_L1/P{fn + 1}_target_mask.npy'

    if view_only:
        mag_fig, mag_ax = plt.subplots(1, 1)
        plotter_mag = Slice_Plotter(mag_ax, np.transpose((mag * mag_mask), axes=(1, 0, 2)), f'P{fn+1} Magnitude', cmap=cmap)
        mag_fig.canvas.mpl_connect('scroll_event', plotter_mag.onscroll)
        plt.show(block=True)
        plt.close()

        if check_fmap:
            mag_fig, mag_ax = plt.subplots(1, 1)
            plotter_mag = Slice_Plotter(mag_ax, np.transpose((fmap * mag_mask), axes=(1, 0, 2)), f'P{fn+1} Magnitude', cmap=cmap)
            mag_fig.canvas.mpl_connect('scroll_event', plotter_mag.onscroll)
            plt.show(block=True)
            plt.close()

        continue

    if os.path.exists(mask_path):
        if not recheck:
            continue

        print('Existing mask detected')
        print('Loading existed mask:')

        artery_mask = np.load(mask_path)
        z_min = np.min(np.argwhere(artery_mask == 1)[:, 2])
        z_max = np.max(np.argwhere(artery_mask == 1)[:, 2])
        z = z_min
        if z_max - z_min > 1:
            z += 1

        mag_fig, mag_ax = plt.subplots(1, 1)
        plotter_mag = Slice_Plotter(mag_ax, np.transpose((mag * artery_mask)[:, :, max(0, z-1):z+2], axes=(1, 0, 2)), f'P{fn+1} Target mask', cmap=cmap)
        mag_fig.canvas.mpl_connect('scroll_event', plotter_mag.onscroll)

        print('Confirm mask -- close when done')
        plt.show(block=True)
        plt.close()

        valid = False
        while not valid:
            confirm = input('Good? [Y/N/Quit]: ')
            valid = True
            if confirm == 'Y':
                good = True
            elif confirm == 'N':
                good = False
            elif confirm == 'Quit':
                print('Quitting...')
                quit()
            else:
                valid = False

    while not good:
        mag_fig, mag_ax = plt.subplots(1, 1)
        plotter_mag = Slice_Plotter(mag_ax, np.transpose((mag * mag_mask), axes=(1, 0, 2)), f'P{fn+1} Magnitude', cmap=cmap)
        mag_fig.canvas.mpl_connect('scroll_event', plotter_mag.onscroll)
        print(f'Locate arteries -- close when done')
        print('[Labeling plane]')
        print('[x_i, y_i] x4')
        plt.show(block=True)
        plt.close()

        z = int(input())
        arteries = []
        artery_mask = np.zeros((X, Y, Z))

        for artery_n in range(4):
            a = tuple([int(val) for val in input().split()])
            arteries.append(a)
            artery_mask[a[0]-3:a[0] + 4, a[1]-3:a[1]+4, max(0, z-1):z+2] = 1

        artery_mask[artery_mask == 0] = np.nan
        artery_mask = artery_mask * mag_mask
        
        mag_fig, mag_ax = plt.subplots(1, 1)
        plotter_mag = Slice_Plotter(mag_ax, np.transpose((mag * artery_mask)[:, :, max(0, z-1):z+2], axes=(1, 0, 2)), f'P{fn+1} Target mask', cmap=cmap)
        mag_fig.canvas.mpl_connect('scroll_event', plotter_mag.onscroll)

        print('Confirm mask -- close when done')
        plt.show(block=True)
        plt.close()

        valid = False
        while not valid:
            confirm = input('Good? [Y/N/Quit]: ')
            valid = True
            if confirm == 'Y':
                good = True
            elif confirm == 'N':
                good = False
            elif confirm == 'Quit':
                print('Quitting...')
                quit()
            else:
                valid = False
    
    if os.path.exists(mask_path):
        os.remove(mask_path)
    np.save(mask_path, artery_mask)


    

    
