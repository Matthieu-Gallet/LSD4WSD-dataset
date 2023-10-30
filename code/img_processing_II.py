import numpy as np
from numba import jit


@jit(nopython=True)
def SAR_patch_clean(x, windows_size, step, start=0):
    xsize, ysize, _ = x.shape
    f = [
        x[i : i + windows_size, j : j + windows_size, :]
        for i in range(start, xsize, step)
        for j in range(start, ysize, step)
        if (
            x[i : i + windows_size, j : j + windows_size, :].shape[:2]
            == (windows_size, windows_size)
        )
        & (np.all(x[i : i + windows_size, j : j + windows_size, :] != -999))
        & (np.all(x[i : i + windows_size, j : j + windows_size, :2] > 0))
    ]
    return f
