import numpy as np
import pytest

# Import your actual implementation functions.
# If you keep them in a .py file extracted from the notebook, adjust the
# module name below accordingly (e.g. "from mandelbrot import ...").
from Mandelbrot_func import (
    mandelbrot_naive,
    mandelbrot_numpy,
    mandelbrot_parallel,
)

""""
---------------------------------------------------------------------------
1. Analytically provable pixel values — parametrized unit test
---------------------------------------------------------------------------

3 x 3 grid so coordinate maping definitios produce predictable values:

  x = linspace(-2, 1, cols=3) = [-2.0, -0.5,  1.0]
  y = linspace(-1.5, 1.5, rows=3) = [-1.5,  0.0,  1.5]

This gives us the following c values at each [row, col]:

  [1, 1] → c = -0.5 + 0j  (inside the set)
    z cycles away from 0 but stays bounded → never escapes → max_iter = 100

  [1, 2] → c = 1.0 + 0j   (outside the set)
    z_1 = 1.0, z_2 = 2.0, z_3 = 5.0 → |z_3| > 2, escapes at k=2

  [0, 2] → c = 1.0 - 1.5j (far outside)
    z_1 = 1 - 1.5j, |z_1|^2 = 1 + 2.25 = 3.25 > 4 → False, continues
    z_2 = (1-1.5j)^2 + (1-1.5j) = (1 - 3 - 3j) + (1-1.5j) = -1 - 4.5j
    |z_2| = sqrt(1 + 20.25) ≈ 4.61 > 2  → escapes at k=1

(row_idx, col_idx, expected, reason)
"""
KNOWN_PIXEL_CASES = [
    (1, 1, 100, "c=-0.5+0j is inside the set, never escapes"),
    (1, 2,   2, "c=1+0j: z1=1, z2=2, z3=5>2, escapes at iteration k=2"),
    (0, 2,   1, "c=1-1.5j: z2 has magnitude ~4.61>2, escapes at k=1"),
]

@pytest.mark.parametrize("row, col, expected, reason", KNOWN_PIXEL_CASES)
def test_naive_known_pixel_values(
    row: int, col: int, expected: int, reason: str
) -> None:
    """mandelbrot_naive produces analytically correct escape counts."""
    result = mandelbrot_naive(rows=3, cols=3, max_iter=100)
    assert result[row, col] == expected, (
        f"pixel [{row},{col}]: expected {expected} ({reason}), got {result[row, col]}"
    )


"""
---------------------------------------------------------------------------
2. Cross-validation: naive is oracle, numpy must agree exactly
---------------------------------------------------------------------------
The naive double loop is the simplest correct implementation, the
oracle. Any vectorised version must produce identical integer escape counts
on every pixel of a small grid (slides: "naive loop as oracle, test that
NumPy/Numba/multiprocessing agree on a 32×32 grid").
"""

def test_numpy_matches_naive_oracle() -> None:
    """mandelbrot_numpy must produce identical results to mandelbrot_naive on a 32×32 grid."""
    rows, cols, max_iter = 32, 32, 50

    oracle = mandelbrot_naive(rows, cols, max_iter).astype(np.int32)
    result = mandelbrot_numpy(rows, cols, max_iter).astype(np.int32)

    np.testing.assert_array_equal(
        result, oracle,
        err_msg="mandelbrot_numpy disagrees with naive oracle",
    )


"""
---------------------------------------------------------------------------
3. Cross-validation: parallel Numba must agree with naive oracle
---------------------------------------------------------------------------
mandelbrot_parallel uses prange and explicit arithmetic for coordinates
instead of linspace. We verify it produces the same output as naive on a
small grid, catching any off-by-one in coordinate mapping or differences
in the escape condition.
"""

def test_parallel_matches_naive_oracle() -> None:
    """mandelbrot_parallel must produce identical results to mandelbrot_naive on a 32×32 grid."""
    rows, cols, max_iter = 32, 32, 50

    oracle = mandelbrot_naive(rows, cols, max_iter).astype(np.int32)

    mandelbrot_parallel(4, 4, max_iter)        
    result = mandelbrot_parallel(rows, cols, max_iter).astype(np.int32)

    np.testing.assert_array_equal(
        result, oracle,
        err_msg=(
            "mandelbrot_parallel disagrees with naive oracle — "
            "check coordinate mapping or escape condition in prange version"
        ),
    )