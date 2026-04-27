import numpy as np
from dask import delayed
import dask
from dask.distributed import Client
from numba import njit
from multiprocessing import Pool
import time
import statistics
from typing import Tuple, List, Optional


@njit(cache=True)
def mandelbrot_pixel(
    c_real: float, 
    c_imag: float, 
    max_iter: int
    ) -> int:
    """
    Compute the number of iterations for a single complex point
    in the Mandelbrot set.

    Args:
        c_real: Real part of the complex number c.
        c_imag: Imaginary part of the complex number c.
        max_iter: Maximum number of iterations.

    Returns:
        Number of iterations before divergence (or max_iter if bounded).
    """
    z_real = z_imag = 0.0
    for i in range(max_iter):
        zr2 = z_real * z_real
        zi2 = z_imag * z_imag
        if zr2 + zi2 > 4.0:
            return i
        z_imag = 2.0 * z_real * z_imag + c_imag
        z_real = zr2 - zi2 + c_real
    return max_iter


@njit(cache=True)
def mandelbrot_chunk(
    row_start: int,
    row_end: int,
    N: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    max_iter: int
) -> np.ndarray:
    
    """
    Compute a horizontal chunk (subset of rows) of the Mandelbrot image.

    Args:
        row_start: Starting row index (inclusive).
        row_end: Ending row index (exclusive).
        N: Image resolution (NxN grid).
        x_min, x_max: Horizontal bounds in complex plane.
        y_min, y_max: Vertical bounds in complex plane.
        max_iter: Maximum iterations per pixel.

    Returns:
        2D NumPy array of shape (row_end - row_start, N) with iteration counts.
    """
    out = np.empty((row_end - row_start, N), dtype=np.int32)
    dx = (x_max - x_min) / N
    dy = (y_max - y_min) / N

    for r in range(row_end - row_start):
        c_imag = y_min + (r + row_start) * dy
        for col in range(N):
            out[r, col] = mandelbrot_pixel(
                x_min + col * dx,
                c_imag,
                max_iter
            )
    return out


def mandelbrot_serial(
    N: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    max_iter: int = 100
) -> np.ndarray:
    """
    Compute the full Mandelbrot image using a single process.

    Args:
        N: Image resolution (NxN).
        x_min, x_max, y_min, y_max: Bounds of the complex plane.
        max_iter: Maximum iterations per pixel.

    Returns:
        2D NumPy array of shape (N, N).
    """
    return mandelbrot_chunk(0, N, N, x_min, x_max, y_min, y_max, max_iter)


def _worker(args: Tuple[int, int, int, float, float, float, float, int]) -> np.ndarray:
    """
    Helper function for multiprocessing.

    Args:
        args: Tuple of arguments for mandelbrot_chunk.

    Returns:
        Computed chunk as NumPy array.
    """
    return mandelbrot_chunk(*args)


def mandelbrot_parallel(
    N: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    max_iter: int = 100,
    n_workers: int = 4,
    n_chunks: Optional[int] = None,
    pool: Optional[Pool] = None
) -> np.ndarray:
    
    """
    Compute the Mandelbrot image using multiprocessing.

    Args:
        N: Image resolution.
        x_min, x_max, y_min, y_max: Complex plane bounds.
        max_iter: Maximum iterations per pixel.
        n_workers: Number of worker processes.
        n_chunks: Number of chunks to divide the work into.
        pool: Optional pre-created multiprocessing Pool.

    Returns:
        Full Mandelbrot image as a 2D NumPy array.
    """
    if n_chunks is None:
        n_chunks = n_workers

    chunk_size = max(1, N // n_chunks)
    chunks: List[Tuple[int, int, int, float, float, float, float, int]] = []

    row = 0
    while row < N:
        row_end = min(row + chunk_size, N)
        chunks.append((row, row_end, N, x_min, x_max, y_min, y_max, max_iter))
        row = row_end

    if pool is not None:
        return np.vstack(pool.map(_worker, chunks))

    # Warm-up (important for Numba JIT)
    tiny = [(0, 8, 8, x_min, x_max, y_min, y_max, max_iter)]
    with Pool(processes=n_workers) as p:
        p.map(_worker, tiny)
        parts = p.map(_worker, chunks)

    return np.vstack(parts)


def mandelbrot_dask(
    N: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    max_iter: int = 100,
    n_chunks: int = 32
) -> np.ndarray:
    
    """
    Compute the Mandelbrot image using Dask for parallel execution.

    Args:
        N: Image resolution.
        x_min, x_max, y_min, y_max: Complex plane bounds.
        max_iter: Maximum iterations per pixel.
        n_chunks: Number of chunks (tasks) for Dask.

    Returns:
        Full Mandelbrot image as a 2D NumPy array.
    """
    chunk_size = max(1, N // n_chunks)
    tasks = []
    row = 0

    while row < N:
        row_end = min(row + chunk_size, N)
        tasks.append(
            delayed(mandelbrot_chunk)(
                row, row_end, N, x_min, x_max, y_min, y_max, max_iter
            )
        )
        row = row_end

    parts = dask.compute(*tasks)
    return np.vstack(parts)


if __name__ == "__main__":
    """
    Benchmark Mandelbrot implementations using Dask.

    Sweeps over different chunk sizes and measures:
    - Execution time
    - Speedup relative to single chunk
    - Load imbalance factor (LIF)
    """
    N, max_iter = 1024, 100
    X_MIN, X_MAX, Y_MIN, Y_MAX = -2.5, 1.0, -1.25, 1.25

    p = 8  # number of workers

    client = Client("tcp://10.92.0.190:8786")

    # Warm-up to trigger JIT compilation
    client.run(lambda: mandelbrot_chunk(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, 10))

    n_chunks_list = list(range(1, 33))
    times: List[float] = []

    for n_chunks in n_chunks_list:
        t_runs: List[float] = []
        for _ in range(3):
            t0 = time.perf_counter()
            mandelbrot_dask(
                N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter, n_chunks
            )
            t_runs.append(time.perf_counter() - t0)
        times.append(statistics.median(t_runs))

    T1 = times[0]

    vs1x = [t / T1 for t in times]
    speedup = [T1 / t for t in times]
    LIF = [p * (t / T1) - 1 for t in times]

    print("\nn_chunks | time (s) | vs 1x | speedup | LIF")
    print("-" * 55)
    for n, t, v, s, lif in zip(n_chunks_list, times, vs1x, speedup, LIF):
        print(f"{n:8d} | {t:8.3f} | {v:6.2f} | {s:7.2f} | {lif:8.3f}")

    print("\n--- Summary ---")
    print(f"n_chunks optimal : {n_chunks_list[times.index(min(times))]}")
    print(f"t_min            : {min(times):.3f} s")
    print(f"LIF_min          : {min(LIF):.3f}")

    client.close()