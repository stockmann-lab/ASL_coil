import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

import artery_map as am
import slice_plotter as slp

import save_masks_to_mat

# Clear mask_folder on start
clear_folder = True

# Automatically save without editing any files
autosave = True

# Mat file where TOF is saved:
matfile = 'data/tof.mat'
mat_varname = 'tof'

# Mat out file (only useful if autosave is true):
out_file = 'LOCAL/artery_masks/mask.mat'

# Output path:
mask_folder = 'LOCAL/artery_masks'

# Arteries to select before stopping:
artery_count = 4

# Thickness expansion, to capture more than just the 
thicken_x = 0
thicken_y = 0
thicken_z = 0 

# Fractional threshold for boundary
threshold = 0.5

# Maximum range of propagation
delta_x = -1
delta_y = -1
delta_z = 15

# Click radius for finding local maximum
delta_select = 8

def click_link(link, plotter):
    def click_select(event):
        link[0:3] = int(event.xdata), int(event.ydata), plotter.ind
        plt.close()
    return click_select

def mask_link(want, reselect):
    def decide_mask(event):
        if event.key in ['W', 'w']:
            want[0] = True
            print("Want")
        elif event.key in ['D', 'd']:
            want[0] = False
            print("Don't want")
        elif event.key in ['R', 'r']:
            reselect[0] = True
            print("Reselect")
        else:
            return
        plt.close()
    return decide_mask

if clear_folder:
    for filename in os.listdir(mask_folder):
        if filename.endswith(".npy") or filename.endswith(".mat"):
            os.remove(filename)

for _ in range(artery_count):
    reselect = [True]
    red_cmap = plt.get_cmap('bone')
    red_cmap.set_bad('red')
    mag = sio.loadmat(matfile)[mat_varname]
    continuity_index = 0

    while reselect[0]: 
        reselect[0] = False

        fig, current_ax = plt.subplots()
        plotter = slp.Slice_Plotter(current_ax, np.transpose(mag, axes=(1, 0, 2)), f'TOF -- click to select artery', cmap=red_cmap)
        fig.canvas.mpl_connect('scroll_event', plotter.onscroll)

        xyz = [-1, -1, -1]
        fig.canvas.mpl_connect('button_press_event', click_link(xyz, plotter))

        plotter.ind = continuity_index
        plotter.update()

        plt.show()
        plt.close()
        continuity_index = plotter.ind

        de = delta_select
        local = mag[xyz[0] - de: xyz[0] + de, xyz[1] - de: xyz[1] + de, xyz[2]:xyz[2]+1]
        edit = np.argwhere(local == local.max())[0, :] - np.array([de, de, 0])
        xyz = (np.array(xyz) + edit).tolist()
        artery = am.bfs_boundary(mag, xyz, threshold=0.5, dxyz_max=[delta_x, delta_y, delta_z], from_neighbor=False, from_average=False)
        mask = am.thicken(artery, thicken_x, thicken_y, thicken_z)

        mask_vis = np.where(mask, np.nan, mag)
        want = [True]

        fig, current_ax = plt.subplots()
        plotter = slp.Slice_Plotter(current_ax, np.transpose(mask_vis, axes=(1, 0, 2)), f"Want / Don't want or reslect (press W/D/R)", cmap=red_cmap)
        fig.canvas.mpl_connect('scroll_event', plotter.onscroll)
        fig.canvas.mpl_connect('key_press_event', mask_link(want, reselect))

        plotter.ind = continuity_index
        plotter.update()

        plt.show()
        plt.close()
        continuity_index = plotter.ind

    path_label = 'want'
    if want[0]:
        mask = np.where(mask, 1, 0)
    else:
        mask = np.where(mask, 3, 0)
        path_label = 'dont_want'

    i = 0
    while os.path.exists(f"{mask_folder}/{path_label}_{i}.npy"):
        i += 1
    np.save(f"{mask_folder}/{path_label}_{i}.npy", mask)

if autosave:
    save_masks_to_mat.save_to_mat(mask_folder, out_file)