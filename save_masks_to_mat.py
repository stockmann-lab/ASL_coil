import os
import sys

import numpy as np
import scipy.io as sio

def save_to_mat():
    mask_folder = 'LOCAL/artery_masks'
    out_file = 'LOCAL/artery_masks/mask.mat'
    mask = 0

    i = 0
    while os.path.exists(f"{mask_folder}/want_{i}.npy"):
        temp = np.load(f"{mask_folder}/want_{i}.npy")
        print(f"Loaded {mask_folder}/want_{i}.npy")
        mask = np.where(temp != 0, temp, mask)
        i += 1

    i = 0
    while os.path.exists(f"{mask_folder}/dont_want_{i}.npy"):
        temp = np.load(f"{mask_folder}/dont_want_{i}.npy")
        print(f"Loaded {mask_folder}/dont_want_{i}.npy")
        mask = np.where(temp != 0, temp, mask)
        i += 1

    sio.savemat(out_file, {'mask' : mask})

if __name__ == '__main__':
    save_to_mat()