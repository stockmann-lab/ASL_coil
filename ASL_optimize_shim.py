#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import sys

import numpy as np
import scipy
import scipy.io as sio
from compare_shim import compare_shim
from biot_savart_neck import generate_neck_coils

if not os.path.exists('coils/head_coils.npy'):
    head_coils = sio.loadmat('coil_maps.mat')['coil_maps']
    np.save('coils/head_coils.npy', head_coils)
if not os.path.exists('coils/neck_coils.npy'):
    generate_neck_coils('coils/neck_coils.npy')
if not os.path.exists('maps/magnitude.npy'):
    mag = sio.loadmat('subject1_mag.mat')['subject1_mag'].astype('float')
    np.save('maps/magnitude.npy', mag)
if not os.path.exists('maps/unshimmed.npy'):
    unshimmed = sio.loadmat('subject1_fm.mat')['subject1_fm']
    np.save('maps/unshimmed.npy', unshimmed)

m_x, m_y, m_z = mask_origin = (52, 40, 0)
m_X, m_Y, m_Z = (25, 41, 8)

head_coils = np.load('coils/head_coils.npy')  # X, Y, Z, N
neck_coils = np.load('coils/neck_coils.npy')
mag = np.load('maps/magnitude.npy')
unshimmed = np.load('maps/unshimmed.npy')

selection_mask = np.zeros(unshimmed.shape)
selection_mask[m_x:m_x+m_X, m_y:m_y+m_Y, m_z:m_z+m_Z] = 1

mag_mask = np.where(mag > 1 * np.mean(mag), 1, 0)
mask = mag_mask[m_x:m_x+m_X, m_y:m_y+m_Y, m_z:m_z+m_Z]

# coils = head_coils
# coils = neck_coils
coils = np.concatenate((head_coils,neck_coils), axis=3)

n = coils.shape[3]
currents = np.zeros(n)
bounds = [(-3, 3) for channel in currents]

# Add scalar
currents = np.append(currents, 0)
bounds.append((-1e+6, 1e+6))
coils = np.concatenate((coils, np.ones(coils.shape[:-1] + (1,))), axis=3)

print(f'Coils shape: {coils.shape}')
print(f'Mag shape: {mag.shape}')
print(f'Unshimmed shape: {unshimmed.shape}')
print(f'Mask shape: {mask.shape}')

compare_shim(coils, unshimmed, mask, mask_origin=mask_origin, bounds=bounds, magnitude=mag, magnitude_mask=mag_mask)