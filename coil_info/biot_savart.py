#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Code ported and refactored from Jason Stockmann and Fa-Hsuan Lin, "Magnetic field by Biot-Savart's Law"
# http://maki.bme.ntu.edu.tw/?page_id=333

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

MU0 = 1.256637e-6  # [H/m]
H_GYROMAGNETIC_RATIO = 42.577478518e+6  # [Hz/T]

# TODO Docstring
def biot_savart(loop_list, fov_min, fov_max, fov_n, points=None):
    """
    Creates coil profiles for arbitrary loops, for use in multichannel shim examples that do not match spherical
    harmonics
    Args:
        centers (list): List of 3D float center points for each loop in mm
        normals (list): List of 3D float normal vectors for each loop in mm
        radii (list): List of float radii for each loop in mm
        segment_numbers (list): List of integer number of segments for each loop approximation
        fov_min (tuple): Low 3D float corner of coil profile field of view (x, y, z) in mm
        fov_max (tuple): Inclusive high 3D float corner of coil profile field of view (x, y, z) in mm
        fov_n (tuple): Integer number of points for each dimension (x, y, z) in mm

    Returns:
        numpy.ndarray: (|X|, |Y|, |Z|, |centers|) coil profiles of magnetic field z-component in Hz/A -- (X, Y, Z, Channel)

    """
    channels = len(loop_list)
    ranges = []
    for i in range(3):
        ranges.append(np.linspace(fov_min[i], fov_max[i], num=fov_n[i]))
    x, y, z = np.meshgrid(ranges[0], ranges[1], ranges[2], indexing='ij', sparse=True)
    scale = (np.array(fov_max) - np.array(fov_min)) / (np.array(fov_n) - 1)

    profiles = np.zeros((x.size, y.size, z.size, channels))
    ch = 0
    for segments in loop_list:
        print(f'Loop {ch + 1}')
        n = 0
        n_max = segments.shape[2]
        for segment in np.split(segments, segments.shape[2], axis=2):
            l = np.average(segment, axis=0).reshape(3)
            dl = (segment[1] - segment[0]).reshape(3)
            if points is None:
                for i in range(x.size):
                    for j in range(y.size):
                        for k in range(z.size):
                            bz = _z_field(l, dl, np.asarray([x[i, 0, 0], y[0, j, 0], z[0, 0, k]]))
                            if np.isnan(bz):
                                profiles[i, j, k, ch] = bz
                            if not np.isnan(profiles[i, j, k, ch]):
                                profiles[i, j, k, ch] += bz
            else:
                for point_n in range(points.shape[0]):
                    pt = points[point_n]
                    idx = (pt / scale).astype('int32')
                    bz = _z_field(l, dl, pt)
                    i, j, k = idx[0], idx[1], idx[2]
                    if np.isnan(bz):
                        profiles[i, j, k, ch] = bz
                    if not np.isnan(profiles[i, j, k, ch]):
                        profiles[i, j, k, ch] += bz
            if n % 5 == 4:
                print(f'Segment {n + 1}/{n_max}')
            n += 1
        ch += 1

    coils = profiles * H_GYROMAGNETIC_RATIO  # [Hz/A]
    return coils

# TODO Docstring
def parameterize_segments(functions, ranges, segment_ns):
    seg_list = []
    for ch in range(len(functions)):
        function = functions[ch]
        start, stop = ranges[ch]
        segment_n = segment_ns[ch]
        x_vals = np.concatenate((np.linspace(start, stop, segment_n), np.array([start])), axis=0)
        f_return = np.array([function(x) for x in x_vals])
        points = f_return.T.reshape((1, 3, -1))
        seg_list.append(np.concatenate((points[:, :, :-1], points[:, :, 1:]), axis=0))
        
    return seg_list

# TODO Docstring
def generate_segments(centers, normal_axes, radii, segment_numbers, major_axes=None, major_radii=None):
    seg_list = []
    for ch in range(len(centers)):
        if major_axes is not None and major_radii is not None:
            major_axis = major_axes[ch]
            major_radius = major_radii[ch]
        else:
            major_axis = major_radius = None
        seg_list.append (_loop_segments(np.asarray(centers[ch]), np.asarray(normal_axes[ch]), radii[ch], segment_numbers[ch],
             major_axis=major_axis, major_radius=major_radius))
    return seg_list

# TODO Docstring
def plot_segments(seg_list, block=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for segments in seg_list:
        _3d_plot(segments, ax)

    x_lim, y_lim, z_lim = ax.get_xlim(), ax.get_ylim(), ax.get_zlim()
    
    x_scale = x_lim[1] - x_lim[0]
    y_scale = y_lim[1] - y_lim[0]
    z_scale = z_lim[1] - z_lim[0]

    scale = max(x_scale, y_scale, z_scale)

    ax.set_xlim([x * scale/x_scale for x in x_lim])
    ax.set_ylim([y * scale/y_scale for y in y_lim])
    ax.set_zlim([0.75 * z * scale/z_scale for z in z_lim])
    plt.show(block=block)
    plt.clf()

# TODO Docstring
def _3d_plot(segments, ax):
    xs = np.concatenate((segments[0, 0, :], segments[1, 0, -1:]), axis=0)
    ys = np.concatenate((segments[0, 1, :], segments[1, 1, -1:]), axis=0)
    zs = np.concatenate((segments[0, 2, :], segments[1, 2, -1:]), axis=0)
    ax.plot(xs, ys, zs)

# TODO Docstring
def _loop_segments(center, normal, minor_radius, segment_num, major_axis=None, major_radius=None):
    """Creates loop segments for loop approximation, given loop details
    Args:
        center (numpy.ndarray): 3D center point of loop in arbitrary units
        normal (numpy.ndarray): 3D normal vector to loop in arbitrary units
        minor_radius (float): Minimum loop radius in arbitrary units
        segment_num (int): Number of segments for loop approximation

    Keyword Args:
        major_axis (numpy.ndarray): Default None, major axis of ellipse (must be orthogonal to normal)
        major_radius (float): Default None, major radius of elliptical loop

    Returns:
        numpy.ndarray: (2, 3, segment_num) array of segments (segment start [0] or end [1]; x, y, z ; segment number)
    """
    
    x_ax = np.array([1, 0, 0])
    z_ax = np.array([0, 0, 1])
    if major_axis is not None:
        assert(major_radius is not None)
        assert(np.dot(normal, major_axis) == 0)
        major_axis = np.asarray(major_axis)
    else:
        major_radius = minor_radius
        if np.dot(normal, x_ax) != 0:
            major_axis = np.cross(normal, z_ax)
        else:
            major_axis = np.cross(normal, x_ax)
    
    segments = np.zeros((2, 3, segment_num))

    theta = np.linspace(0, 2 * np.pi, num=segment_num+1, endpoint=True).reshape((1, segment_num+1))
    segments[0, :-1, :] = np.concatenate((major_radius * np.cos(theta[:, :-1]), minor_radius * np.sin(theta[:, :-1])), axis=0)  # Start points
    segments[1, :-1, :] = np.concatenate((major_radius * np.cos(theta[:, 1:]), minor_radius * np.sin(theta[:, 1:])), axis=0)  # End points

    segments = np.round(segments, decimals=9)

    # normal = normal / np.linalg.norm(normal)
    # major_axis = major_axis / np.linalg.norm(major_axis)
    r = np.cross(z_ax, normal)
    if np.linalg.norm(r) == 0:
        r = np.array([1, 0, 0])
    normal_rotate = _rotate_to(z_ax, normal, r)
    major_rotate = _rotate_to(normal_rotate @ x_ax, major_axis, normal)
    segments =  major_rotate @ normal_rotate @ segments
    return segments + center.reshape((1, 3, 1))

# TODO Docstring
def _rotate_to(v1, v2, r):
    assert(np.round(np.linalg.norm(r), decimals=9) != 0)
    assert(np.round(np.dot(v1, r), decimals=9) == 0)
    assert(np.round(np.dot(v2, r), decimals=9) == 0)

    v1_hat = v1 / np.linalg.norm(v1)
    v2_hat = v2 / np.linalg.norm(v2)
    r_hat = r / np.linalg.norm(r)

    cs = np.dot(v1_hat, v2_hat)
    sn = np.linalg.norm(np.cross(v1_hat, v2_hat))

    if sn != 0:
        r = r_hat * sn
        r_cross_mat = np.array([[0, -r[2], r[1]],
                            [r[2], 0, -r[0]],
                            [-r[1], r[0], 0]])
        return np.identity(3) + r_cross_mat + r_cross_mat @ r_cross_mat * (1 - cs)/sn**2
    else:
        basis_shift = np.vstack((r_hat, v1_hat, np.cross(v1_hat, r_hat)))
        flip_mat = np.array([[1, 0, 0],
                            [0, cs, 0],
                            [0, 0, cs]])
        return basis_shift.T @ flip_mat @ basis_shift


def _z_field(l, dl, r):
    """Calculate z-field at point r from line segment centered at l with length dl
    Args:
        l (numpy.ndarray): Line segment center in m
        dl (numpy.ndarray): Line segment vector in m
        r (numpy.ndarray): Target point in m

    Returns:
        float: z-component of magnetic field at r in T/A
    """
    l, dl, r = l / 1000, dl / 1000, r / 1000  # Convert mm to m
    rp = r - l
    rp_norm = np.linalg.norm(rp)
    if rp_norm == 0:
        return np.nan
    b_per_i = MU0 / (4 * np.pi) * np.cross(dl, rp) / rp_norm ** 3
    return b_per_i[2]  # [T/A]
