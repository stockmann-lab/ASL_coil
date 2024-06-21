import numpy as np
from scipy.interpolate import RegularGridInterpolator as rgi

# TODO Docstring
def grid_align(V, T, V_scale=None, V_origin=None, bounds_error=False, fill_value=np.nan):
    assert(len(V.shape) == len(T.shape))
    if V_scale is None:
        V_scale = np.array(T.shape) / np.array(V.shape)
    if V_origin is None:
        V_origin = np.zeros(len(V.shape))
    assert(not np.any(V_scale == 0))
    V_axes = [np.linspace(V_origin[ax], V_origin[ax] + (V.shape[ax] - 1) * V_scale[ax], num=V.shape[ax]) for ax in range(len(V.shape))]
    interp_func = rgi(V_axes, V, bounds_error=bounds_error, fill_value=fill_value)
    points = np.argwhere(np.ones_like(T))
    return interp_func(points).reshape(T.shape)
