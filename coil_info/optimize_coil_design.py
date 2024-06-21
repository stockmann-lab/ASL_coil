import numpy as np
import time
import scipy.optimize as opt
import biot_savart as bsv
import matplotlib.pyplot as plt
from slice_plotter import Slice_Plotter

SEG_NUM = 15
COIL_N = 8
DAT_NUM = 25
SCALE = np.array([2, 2, 2])
CYL_CENTER = np.array([45, 45, 0]) * SCALE
CYL_SCALE = np.array([40, 40, 40]) * SCALE + np.array([0, 0, 1])
PI = np.pi
PREV_RES = 0
FEV = 0

def shim_residuals(coef, unshimmed_vec, coil_mat):
        """
        Objective function to minimize
        Args:
            coef (numpy.ndarray): 1D array of channel coefficients
            unshimmed_vec (numpy.ndarray): 1D flattened array (point) of the masked unshimmed map
            coil_mat (numpy.ndarray): 2D flattened array (point, channel) of masked coils
                (axis 0 must align with unshimmed_vec)

        Returns:
            numpy.ndarray: Residuals for least squares optimization -- equivalent to flattened shimmed vector

        """
        res = unshimmed_vec + np.sum(coil_mat * coef, axis=1, keepdims=False)
        return res

def optimize_shim(coils, unshimmed, mask, mask_origin=(0, 0, 0), bounds=None):
        """
        Optimize unshimmed volume by varying current to each channel

        Args:
            coils (numpy.ndarray): X, Y, Z, N coil map
            unshimmed (numpy.ndarray): 3D B0 map
            mask (numpy.ndarray): 3D integer mask used for the optimizer (only consider voxels with non-zero values).
            mask_origin (tuple): Mask origin if mask volume does not cover unshimmed volume
            bounds (list): List of ``(min, max)`` pairs for each coil channels. None
               is used to specify no bound.

        Returns:
            numpy.ndarray: Coefficients corresponding to the coil profiles that minimize the objective function
                           (coils.size)
        """

        # cmap = plt.get_cmap('bone')
        # cmap.set_bad('black')
        # mag_fig, mag_ax = plt.subplots(1, 1)
        # plotter_mag = Slice_Plotter(mag_ax, np.transpose((unshimmed), axes=(1, 0, 2)), f'Unshimmed Full', cmap=cmap)
        # mag_fig.canvas.mpl_connect('scroll_event', plotter_mag.onscroll)
        # plt.show(block=True)
        # plt.close()

        mask_range = tuple([slice(mask_origin[i], mask_origin[i] + mask.shape[i]) for i in range(3)])
        mask_vec = mask.reshape((-1,))

        # Least squares solver
        N = coils.shape[3]
        # Reshape coil profile: X, Y, Z, N --> [mask.shape], N
        #   --> N, [mask.shape] --> N, mask.size --> mask.size, N
        coil_mat = np.reshape(np.transpose(coils[mask_range], axes=(3, 0, 1, 2)),
                                (N, -1)).T
        coil_mat = coil_mat[mask_vec != 0, :]  # masked points x N

        unshimmed = unshimmed[mask_range]
        unshimmed_vec = np.reshape(unshimmed, (-1,)) # mV
        unshimmed_vec = unshimmed_vec[mask_vec != 0]  # mV'

        # Set up output currents and optimize
        if bounds is not None:
            bounds = np.asarray(bounds)
        currents_0 = np.zeros(N)
        currents_sp = opt.least_squares(shim_residuals, currents_0,
                                        args=(unshimmed_vec, coil_mat), bounds=bounds)

        currents = currents_sp.x
        residuals = np.asarray(currents_sp.fun)

        return (currents, residuals)

def design_residuals_func(fieldmaps, masks, coil_count):
    total_mask = np.zeros_like(masks[0])

    fov_n = masks[0].shape
    fov_min = np.zeros(3).astype('int32')
    fov_max = (fov_min + fov_n - 1) * SCALE

    for mask in masks:
        total_mask[mask == 1] = 1
    mask_origins = []
    for m_i in range(len(masks)):
        coord_ranges = [[], [], []]
        for coord in range(3):
            coord_ranges[coord] = (np.min(np.argwhere(masks[m_i] == 1)[:, coord]), np.max(np.argwhere(masks[m_i] == 1)[:, coord]))
        mask_origins.append(tuple([coord_range[0] for coord_range in coord_ranges]))
        mask = np.zeros_like(masks[m_i])
        mask[masks[m_i] == 1] = 1
        mask_range = tuple([slice(coord_range[0], coord_range[1] + 1) for coord_range in coord_ranges])
        masks[m_i] = mask[mask_range]

    points = np.argwhere(total_mask == 1) * SCALE
    print(f'{points.shape[0]} target voxels')
    
    def design_residuals(param_vec):
        global FEV
        print(param_vec)
        residuals = []
        coils = generate_coils(param_vec, coil_count, fov_min, fov_max, fov_n, points=points)
        for patient_n in range(len(fieldmaps)):
            unshimmed = fieldmaps[patient_n]
            mask = masks[patient_n]
            mask_origin = mask_origins[patient_n]
            _, res = optimize_shim(coils, unshimmed, mask, mask_origin=mask_origin, bounds=(np.ones(coil_count) * -6, np.ones(coil_count) * 6))
            residuals.append(res)
        res = np.concatenate(residuals, axis=0)
        global PREV_RES
        print(f'Max residual difference: {np.max(res - PREV_RES)}')
        PREV_RES = res
        return res
    return design_residuals

def unpack_param_vec(param_vec, coil_count):
    centers = np.split(param_vec[:coil_count * 2], coil_count)
    major_phis = param_vec[coil_count * 2: coil_count * 3].tolist()
    major_radii = param_vec[coil_count * 3: coil_count * 4].tolist()
    minor_radii = param_vec[coil_count * 4: coil_count * 5].tolist()

    return centers, major_phis, major_radii, minor_radii

def cylinder_loop_func(center, major_phi, major_r, minor_r):
    # phi = 0 = x_cart = z_cyl
    def cylinder_loop(phi):
        cart_point = np.array([major_r * np.cos(phi), minor_r * np.sin(phi), 0]) # z, theta
        cart_point = bsv._rotate_to(np.array([1, 0, 0]), np.array([np.cos(major_phi), np.sin(major_phi), 0]), np.array([0, 0, 1])) @ cart_point
        cart_point += [center[0], center[1], 0]
        z, theta = cart_point[0], cart_point[1]
        point = np.array([np.cos(theta), np.sin(theta), z]) * CYL_SCALE + CYL_CENTER
        return point
    return cylinder_loop

def generate_coils(param_vec, coil_count, fov_min, fov_max, fov_n, points=None):
    centers, major_phis, major_radii, minor_radii = unpack_param_vec(param_vec, coil_count)
    print('Generating coils...')
    functions = [cylinder_loop_func(centers[ch], major_phis[ch], major_radii[ch], minor_radii[ch]) for ch in range(coil_count)]
    segments = bsv.parameterize_segments(functions, [[0, 2 * PI * (SEG_NUM-1)/SEG_NUM] for _ in range(coil_count)], [SEG_NUM for _ in range(coil_count)])
    # bsv.plot_segments(segments, block=False)
    coils = bsv.biot_savart(segments, fov_min, fov_max, fov_n, points=points)
    print('Generated coils')

    global FEV
    if FEV % 10 == 0:
        print(f'--- RUNTIME: {int((time.time() - START_TIME) / 3600)} hr {int((time.time() - START_TIME) / 60) % 60} min ')
    FEV += 1
    print(f'Function evaluations: {FEV}')

    return coils

masks = []
fieldmaps = []
for fn in range(DAT_NUM):
    fmap = np.load(f'maps/connectome_L1/P{fn + 1}_fmap.npy')
    mask = np.load(f'maps/connectome_L1/P{fn + 1}_target_mask.npy')
    fieldmaps.append(fmap)
    masks.append(mask)
coil_count = COIL_N

res_func = design_residuals_func(fieldmaps, masks, coil_count)


centers = np.array([[0, n * 2 * PI/coil_count] for n in range(coil_count)])
major_phi = np.array([n * 2 * PI/coil_count for n in range(coil_count)])
major_radii = np.array([0.5 for _ in range(coil_count)])
minor_radii = np.array([0.2 for _ in range(coil_count)])

param_vec_0 = np.concatenate((centers.reshape(-1), major_phi, major_radii, minor_radii), axis=0)

centers, major_phis, major_radii, minor_radii = unpack_param_vec(param_vec_0, coil_count)
print('Plotting original')
functions = [cylinder_loop_func(centers[ch], major_phis[ch], major_radii[ch], minor_radii[ch]) for ch in range(coil_count)]
segments = bsv.parameterize_segments(functions, [[0, 2 * PI * (SEG_NUM-1)/SEG_NUM] for _ in range(coil_count)], [SEG_NUM for _ in range(coil_count)])
bsv.plot_segments(segments, block=True)

START_TIME = time.time()
params_sp = opt.least_squares(res_func, param_vec_0, verbose=2, jac='2-point', max_nfev=(coil_count * 250))
param_vec = params_sp.x
print('')
print('')
print('--- COMPLETED ---')
print(f'--- RUNTIME: {((time.time() - START_TIME) / 3600) // 1} hr {(((time.time() - START_TIME) / 60) // 1) % 60} min ')

np.save(f'LOCAL/param_vec_SEG{SEG_NUM}_COIL{COIL_N}', param_vec)
print(param_vec)

centers, major_phis, major_radii, minor_radii = unpack_param_vec(param_vec_0, coil_count)
print('Plotting original')
functions = [cylinder_loop_func(centers[ch], major_phis[ch], major_radii[ch], minor_radii[ch]) for ch in range(coil_count)]
segments = bsv.parameterize_segments(functions, [[0, 2 * PI * (SEG_NUM-1)/SEG_NUM] for _ in range(coil_count)], [SEG_NUM for _ in range(coil_count)])
bsv.plot_segments(segments, block=False)

centers, major_phis, major_radii, minor_radii = unpack_param_vec(param_vec, coil_count)
print('Plotting final')
functions = [cylinder_loop_func(centers[ch], major_phis[ch], major_radii[ch], minor_radii[ch]) for ch in range(coil_count)]
segments = bsv.parameterize_segments(functions, [[0, 2 * PI * (SEG_NUM-1)/SEG_NUM] for _ in range(coil_count)], [SEG_NUM for _ in range(coil_count)])
bsv.plot_segments(segments, block=True)


