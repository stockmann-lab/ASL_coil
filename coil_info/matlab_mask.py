import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.io as sio

from slice_plotter import Slice_Plotter
from select_mask import select_mask


IN_FPATH = 'data/connectome_L1/P1_mag.mat'
IN_VARNAME = 'mag'
OUT_FPATH = 'LOCAL/test.mat'
OUT_VARNAME = 'mask'
SLAB_THICKNESS = 3

magnitude = sio.loadmat(IN_FPATH)[IN_VARNAME]
X, Y, Z = magnitude.shape
mag_mask = np.where(magnitude > 0.25 * np.mean(magnitude), 1, np.nan)
artery_mask = select_mask(magnitude, mag_mask, z_range=SLAB_THICKNESS, int_output=True, confirm=False, mask_type='uint16')
out_dict = {OUT_VARNAME : artery_mask, 'label' : 'target mask', 'TEMP' : magnitude}
if os.path.exists(OUT_FPATH):
    os.remove(OUT_FPATH)
sio.savemat(OUT_FPATH, out_dict)