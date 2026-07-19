"""Combinatorics: permutations, combinations, and counting sequences."""
import math


def permutations_count(n: int, r: int) -> int:
    """Number of ways to arrange r items out of n, order matters: nPr."""
    if r > n or r < 0:
        return 0
    return math.factorial(n) // math.factorial(n - r)


def combinations_count(n: int, r: int) -> int:
    """Number of ways to choose r items out of n, order doesn't matter: nCr."""
    if r > n or r < 0:
        return 0
    return math.factorial(n) // (math.factorial(r) * math.factorial(n - r))


def catalan_number(n: int) -> int:
    """The nth Catalan number — counts things like valid bracket sequences and binary tree shapes."""
    return combinations_count(2 * n, n) // (n + 1)


def fibonacci(n: int) -> int:
    """The nth Fibonacci number (0-indexed: fibonacci(0) == 0)."""
    if n < 0:
        raise ValueError("n must be non-negative")
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def binomial_expansion_coeffs(n: int) -> list[int]:
    """Coefficients of (x + y)^n, i.e. row n of Pascal's triangle."""
    return [combinations_count(n, k) for k in range(n + 1)]