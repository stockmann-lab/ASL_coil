#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import os

from shimmingtoolbox.coils.biot_savart import biot_savart

def generate_neck_coils(filepath):

    if os.path.exists(filepath):
        return False

    scale = np.array([1.7, 1.7, 2])

    centers = np.array([ [69, 20, -3],
                        [95, 32, -3],
                        [104, 60, -3],
                        [96, 93, -3],
                        [69, 103, -3],
                        [104, 60, 0],
                        [69, 60, -2]]) * scale
    normals = [ (0, 1, 0),
                (1, 1, 0),
                (1, 0, 0),
                (1, -1, 0),
                (0, -1, 0),
                (1, 0, 0),
                (0, 0, 1)]
    radii = [30, 30, 30, 30, 30, 12, 70]
    segment_numbers = [13, 13, 13, 13, 13, 9, 25]
    fov_min_idx = np.array((25, 25, 0))
    fov_max_idx = np.array((100, 95, 10))
    fov_n = fov_max_idx - fov_min_idx + 1
    fov_min = fov_min_idx * scale
    fov_max = fov_max_idx * scale

    neck_coils = np.zeros((128, 128, 75, 7))
    neck_coils[fov_min_idx[0]:fov_max_idx[0] + 1, fov_min_idx[1]:fov_max_idx[1] + 1, fov_min_idx[2]:fov_max_idx[2] + 1, :] = 2 * biot_savart(centers, normals, radii, segment_numbers, fov_min, fov_max, fov_n)

    np.save(filepath, neck_coils)

