#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import os

import biot_savart as bsv
import matplotlib.pyplot as plt

centers = [(0, 0, 0)]
normals = [(0, 0, 1)]
radii = [10]
segment_numbers = [3]

close_axis = np.zeros(50)
far_axis = np.zeros(50)
close_off = np.zeros(50)
far_off = np.zeros(50)

i = 0
for x in range(8, 58):
    segment_numbers = [x]
    maps = bsv.biot_savart(bsv.generate_segments(centers, normals, radii, segment_numbers), (0, 0, 1), (0, 15, 30), (1, 3, 4))
    close_axis[i] = maps[0, 0, 0]
    far_axis[i] = maps[0, 0, 3]
    close_off[i] = maps[0, 2, 0]
    far_off[i] = maps[0, 2, 3]
    i += 1

plt.plot(np.arange(8, 58), close_axis / close_axis[-1])
plt.plot(np.arange(8, 58), far_axis / far_axis[-1])
plt.plot(np.arange(8, 58), close_off / close_off[-1])
plt.plot(np.arange(8, 58), far_off / far_off[-1])
plt.plot(np.arange(8, 58), np.ones(50) * 1.1, 'r:')
plt.plot(np.arange(8, 58), np.ones(50) * 0.9, 'r:')
plt.plot(np.arange(8, 58), np.ones(50) * 1.05, 'y:')
plt.plot(np.arange(8, 58), np.ones(50) * 0.95, 'y:')
plt.plot(np.arange(8, 58), np.ones(50) * 1.01, 'g:')
plt.plot(np.arange(8, 58), np.ones(50) * 0.99, 'g:')
plt.show()