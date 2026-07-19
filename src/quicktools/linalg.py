"""Linear algebra: matrix operations, determinants, inverses, linear systems.

Matrices are represented as lists of lists, e.g. [[1, 2], [3, 4]].
No external dependencies (no numpy) — pure Python.
"""
import math


def matrix_add(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    """Add two matrices of the same shape."""
    return [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(a))]


def matrix_multiply(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    """Multiply two matrices (a is m x n, b is n x p, result is m x p)."""
    rows_a, cols_a = len(a), len(a[0])
    rows_b, cols_b = len(b), len(b[0])
    if cols_a != rows_b:
        raise ValueError(f"Cannot multiply {rows_a}x{cols_a} by {rows_b}x{cols_b}")
    result = [[0.0] * cols_b for _ in range(rows_a)]
    for i in range(rows_a):
        for j in range(cols_b):
            result[i][j] = sum(a[i][k] * b[k][j] for k in range(cols_a))
    return result


def transpose(a: list[list[float]]) -> list[list[float]]:
    """Return the transpose of a matrix."""
    return [list(row) for row in zip(*a)]


def identity(n: int) -> list[list[float]]:
    """Return the n x n identity matrix."""
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def determinant(a: list[list[float]]) -> float:
    """Determinant of a square matrix via cofactor expansion (Laplace expansion)."""
    n = len(a)
    if n != len(a[0]):
        raise ValueError("Matrix must be square")
    if n == 1:
        return a[0][0]
    if n == 2:
        return a[0][0] * a[1][1] - a[0][1] * a[1][0]
    det = 0.0
    for col in range(n):
        minor = [row[:col] + row[col + 1:] for row in a[1:]]
        sign = 1 if col % 2 == 0 else -1
        det += sign * a[0][col] * determinant(minor)
    return det


def inverse(a: list[list[float]]) -> list[list[float]]:
    """Inverse of a square matrix using Gauss-Jordan elimination."""
    n = len(a)
    aug = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(a)]

    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot_row][col]) < 1e-12:
            raise ValueError("Matrix is singular and cannot be inverted")
        aug[col], aug[pivot_row] = aug[pivot_row], aug[col]

        pivot_val = aug[col][col]
        aug[col] = [x / pivot_val for x in aug[col]]

        for r in range(n):
            if r != col:
                factor = aug[r][col]
                aug[r] = [aug[r][k] - factor * aug[col][k] for k in range(2 * n)]

    return [row[n:] for row in aug]


def solve_linear_system(a: list[list[float]], b: list[float]) -> list[float]:
    """Solve Ax = b for x, given square matrix A and vector b."""
    n = len(a)
    aug = [a[i][:] + [b[i]] for i in range(n)]

    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot_row][col]) < 1e-12:
            raise ValueError("System has no unique solution (singular matrix)")
        aug[col], aug[pivot_row] = aug[pivot_row], aug[col]

        pivot_val = aug[col][col]
        aug[col] = [x / pivot_val for x in aug[col]]

        for r in range(n):
            if r != col:
                factor = aug[r][col]
                aug[r] = [aug[r][k] - factor * aug[col][k] for k in range(n + 1)]

    return [row[n] for row in aug]


def trace(a: list[list[float]]) -> float:
    """Sum of the diagonal elements of a square matrix."""
    return sum(a[i][i] for i in range(len(a)))

def dot_product(v1: list[float], v2: list[float]) -> float:
    """Dot product of two equal-length vectors."""
    if len(v1) != len(v2):
        raise ValueError("Vectors must be the same length")
    return sum(a * b for a, b in zip(v1, v2))


def cross_product(v1: list[float], v2: list[float]) -> list[float]:
    """Cross product of two 3D vectors."""
    if len(v1) != 3 or len(v2) != 3:
        raise ValueError("Cross product requires two 3D vectors")
    return [
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0],
    ]


def vector_magnitude(v: list[float]) -> float:
    """Euclidean length (magnitude) of a vector."""
    return math.sqrt(sum(x ** 2 for x in v))


def normalize_vector(v: list[float]) -> list[float]:
    """Scale a vector to unit length (magnitude 1)."""
    mag = vector_magnitude(v)
    if mag == 0:
        raise ValueError("Cannot normalize the zero vector")
    return [x / mag for x in v]


def vector_add(v1: list[float], v2: list[float]) -> list[float]:
    """Add two equal-length vectors."""
    if len(v1) != len(v2):
        raise ValueError("Vectors must be the same length")
    return [a + b for a, b in zip(v1, v2)]


def vector_subtract(v1: list[float], v2: list[float]) -> list[float]:
    """Subtract v2 from v1 (equal-length vectors)."""
    if len(v1) != len(v2):
        raise ValueError("Vectors must be the same length")
    return [a - b for a, b in zip(v1, v2)]


def scalar_multiply(a: list[list[float]], scalar: float) -> list[list[float]]:
    """Multiply every element of a matrix by a scalar."""
    return [[x * scalar for x in row] for row in a]


def matrix_power(a: list[list[float]], n: int) -> list[list[float]]:
    """Raise a square matrix to the n-th power (n >= 0) via repeated multiplication."""
    if n < 0:
        raise ValueError("n must be non-negative")
    size = len(a)
    result = identity(size)
    for _ in range(n):
        result = matrix_multiply(result, a)
    return result


def is_symmetric(a: list[list[float]]) -> bool:
    """Return True if a square matrix equals its own transpose."""
    return a == transpose(a)


def rank(a: list[list[float]]) -> int:
    """Rank of a matrix, computed via Gaussian elimination to row echelon form."""
    mat = [row[:] for row in a]
    rows, cols = len(mat), len(mat[0])
    rank_count = 0
    for col in range(cols):
        pivot_row = None
        for r in range(rank_count, rows):
            if abs(mat[r][col]) > 1e-12:
                pivot_row = r
                break
        if pivot_row is None:
            continue
        mat[rank_count], mat[pivot_row] = mat[pivot_row], mat[rank_count]
        for r in range(rank_count + 1, rows):
            factor = mat[r][col] / mat[rank_count][col]
            mat[r] = [mat[r][k] - factor * mat[rank_count][k] for k in range(cols)]
        rank_count += 1
        if rank_count == rows:
            break
    return rank_count