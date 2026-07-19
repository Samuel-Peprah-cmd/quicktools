"""Number theory: primes, modular arithmetic, and related functions."""


def extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    """Return (g, x, y) such that a*x + b*y = g = gcd(a, b)."""
    if a == 0:
        return b, 0, 1
    g, x1, y1 = extended_gcd(b % a, a)
    return g, y1 - (b // a) * x1, x1


def mod_inverse(a: int, m: int) -> int:
    """Return the modular multiplicative inverse of a modulo m."""
    g, x, _ = extended_gcd(a % m, m)
    if g != 1:
        raise ValueError(f"No modular inverse exists for {a} mod {m}")
    return x % m


def sieve_of_eratosthenes(limit: int) -> list[int]:
    """Return all prime numbers up to and including `limit`."""
    if limit < 2:
        return []
    is_prime_arr = [True] * (limit + 1)
    is_prime_arr[0] = is_prime_arr[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime_arr[i]:
            for multiple in range(i * i, limit + 1, i):
                is_prime_arr[multiple] = False
    return [i for i, prime in enumerate(is_prime_arr) if prime]


def prime_factors(n: int) -> dict[int, int]:
    """Return the prime factorization of n as {prime: exponent}."""
    if n < 2:
        return {}
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors


def euler_totient(n: int) -> int:
    """Count integers up to n that are coprime with n (Euler's totient function)."""
    result = n
    factors = prime_factors(n)
    for p in factors:
        result -= result // p
    return result


def is_coprime(a: int, b: int) -> bool:
    """Return True if a and b share no common factors other than 1."""
    import math
    return math.gcd(a, b) == 1