#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import os

import coils.biot_savart as bsv

scale = np.array([1.7, 1.7, 2])

centers = np.array([ [69, 20, -3],
                    [95, 32, -3],
                    [104, 60, -3],
                    [96, 93, -3],
                    [69, 103, -3],
                    [104, 60, 0],
                    [69, 60, -2]]) * scale
normals = [ (0, 1, 0),
            (1, -1, 0),
            (1, 0, 0),
            (1, 1, 0),
            (0, -1, 0),
            (1, 0, 0),
            (0, 0, 1)]
radii = [30, 30, 30, 30, 30, 12, 70]
segment_numbers = [13, 13, 13, 13, 13, 9, 25]

bsv.plot_segments(bsv.generate_segments(centers, normals, radii, segment_numbers))
