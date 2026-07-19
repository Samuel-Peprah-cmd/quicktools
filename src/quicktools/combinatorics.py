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

def multinomial_coefficient(n: int, ks: list[int]) -> int:
    """Number of ways to divide n items into groups of sizes ks (which must sum to n)."""
    if sum(ks) != n:
        raise ValueError("The group sizes must sum to n")
    result = math.factorial(n)
    for k in ks:
        result //= math.factorial(k)
    return result


def stirling_second_kind(n: int, k: int) -> int:
    """Number of ways to partition a set of n objects into k non-empty subsets."""
    if k == 0:
        return 1 if n == 0 else 0
    if k > n or n < 0:
        return 0
    if k == n:
        return 1
    total = 0
    for j in range(k + 1):
        sign = (-1) ** (k - j)
        total += sign * combinations_count(k, j) * (j ** n)
    return total // math.factorial(k)


def derangements_count(n: int) -> int:
    """Number of permutations of n items where no item ends up in its original position."""
    if n == 0:
        return 1
    if n == 1:
        return 0
    a, b = 1, 0
    for i in range(2, n + 1):
        a, b = b, (i - 1) * (a + b)
    return b


def integer_partitions_count(n: int) -> int:
    """Number of ways to write n as a sum of positive integers, order not mattering."""
    if n < 0:
        return 0
    partitions = [1] + [0] * n
    for k in range(1, n + 1):
        for i in range(k, n + 1):
            partitions[i] += partitions[i - k]
    return partitions[n]


def combinations_with_replacement_count(n: int, r: int) -> int:
    """Number of ways to choose r items from n types, with repetition allowed and order not mattering."""
    return combinations_count(n + r - 1, r)