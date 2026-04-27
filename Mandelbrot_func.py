import numpy as np
from numba import njit, prange

def mandelbrot_naive(rows: int, cols: int, max_iter: int = 100) -> np.ndarray:
    """
    Compute Mandelbrot set using a naive double for-loop.

    Args:
        rows: Number of rows (image height).
        cols: Number of columns (image width).
        max_iter: Maximum number of iterations per pixel.

    Returns:
        2D array of iteration counts.
    """
    x = np.linspace(-2, 1, cols)
    y = np.linspace(-1.5, 1.5, rows)
    screen = x + y[:, None] * 1j

    iter_count = np.zeros((rows, cols), dtype=np.int16)

    for i in range(rows):
        for j in range(cols):
            z = 0 + 0j
            for k in range(max_iter):
                z = z * z + screen[i, j]
                if abs(z) > 2:
                    iter_count[i, j] = k
                    break
            else:
                iter_count[i, j] = max_iter

    return iter_count

def mandelbrot_numpy(rows: int, cols: int, max_iter: int = 50) -> np.ndarray:
    """
    Compute Mandelbrot set using NumPy vectorization.

    Args:
        rows: Image height.
        cols: Image width.
        max_iter: Maximum iterations.

    Returns:
        2D array of iteration counts.
    """
    x = np.linspace(-2, 1, cols)
    y = np.linspace(-1.5, 1.5, rows)
    C = x + y[:, None] * 1j

    Z = np.zeros_like(C, dtype=np.complex128)
    iter_count = np.zeros(C.shape, dtype=np.int16)

    mask = np.ones(C.shape, dtype=bool)

    for t in range(max_iter):
        Z[mask] = Z[mask] ** 2 + C[mask]
        diverged = np.abs(Z) > 2
        iter_count[mask & diverged] = t
        mask &= ~diverged

    iter_count[mask] = max_iter
    return iter_count

from numba import njit, prange

@njit(parallel=True)
def mandelbrot_parallel(rows: int, cols: int, max_iter: int) -> np.ndarray:
    
    """
    Parallel Mandelbrot computation using Numba prange.

    Args:
        rows: Image height.
        cols: Image width.
        max_iter: Maximum iterations.

    Returns:
        2D array of iteration counts.
    """
        
    x = np.linspace(-2.0, 1.0, cols)
    y = np.linspace(-1.5, 1.5, rows)

    result = np.zeros((rows, cols), dtype=np.int32)

    for i in prange(rows):
        for j in range(cols):
            c_r = x[j]
            c_i = y[i]
            zr = 0.0
            zi = 0.0

            for k in range(max_iter):
                zr_new = zr * zr - zi * zi + c_r   
                zi = 2.0 * zr * zi + c_i
                zr = zr_new
                if zr * zr + zi * zi > 4.0:        
                    result[i, j] = k
                    break
            else:
                result[i, j] = max_iter

    return result