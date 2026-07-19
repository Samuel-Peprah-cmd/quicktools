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

def chinese_remainder_theorem(remainders: list[int], moduli: list[int]) -> int:
    """Solve a system of congruences x = remainders[i] (mod moduli[i]) for all i,
    returning the smallest non-negative solution (moduli must be pairwise coprime)."""
    total_modulus = 1
    for m in moduli:
        total_modulus *= m
    result = 0
    for r, m in zip(remainders, moduli):
        partial = total_modulus // m
        result += r * partial * mod_inverse(partial, m)
    return result % total_modulus


def sum_of_divisors(n: int) -> int:
    """Sum of all positive divisors of n, including 1 and n itself."""
    if n < 1:
        raise ValueError("n must be a positive integer")
    total = 0
    for i in range(1, int(n ** 0.5) + 1):
        if n % i == 0:
            total += i
            if i != n // i:
                total += n // i
    return total


def is_perfect_number(n: int) -> bool:
    """Return True if n equals the sum of its proper divisors (e.g. 6, 28, 496)."""
    if n < 1:
        return False
    return sum_of_divisors(n) - n == n


def is_amicable_pair(a: int, b: int) -> bool:
    """Return True if a and b form an amicable pair (each equals the sum of the other's proper divisors)."""
    return (sum_of_divisors(a) - a == b) and (sum_of_divisors(b) - b == a) and a != b


def digital_root(n: int) -> int:
    """Repeatedly sum the digits of n until a single digit remains."""
    n = abs(n)
    while n >= 10:
        n = sum(int(d) for d in str(n))
    return n


def collatz_sequence(n: int) -> list[int]:
    """Return the Collatz sequence starting at n, ending at 1."""
    if n < 1:
        raise ValueError("n must be a positive integer")
    sequence = [n]
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        sequence.append(n)
    return sequence


def next_prime(n: int) -> int:
    """Return the smallest prime strictly greater than n."""
    candidate = n + 1
    while True:
        is_p = True
        for i in range(2, int(candidate ** 0.5) + 1):
            if candidate % i == 0:
                is_p = False
                break
        if is_p and candidate > 1:
            return candidate
        candidate += 1


def nth_prime(n: int) -> int:
    """Return the nth prime number (1-indexed: nth_prime(1) == 2)."""
    if n < 1:
        raise ValueError("n must be a positive integer")
    count = 0
    candidate = 1
    while count < n:
        candidate = next_prime(candidate)
        count += 1
    return candidate


def is_palindromic_number(n: int) -> bool:
    """Return True if n reads the same forwards and backwards."""
    s = str(n)
    return s == s[::-1]