"""Math utilities: primes, number theory, and basic statistics."""
import math


def is_prime(n: int) -> bool:
    """Return True if n is a prime number."""
    if n < 2:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True


def gcd(a: int, b: int) -> int:
    """Greatest common divisor of a and b."""
    return math.gcd(a, b)


def lcm(a: int, b: int) -> int:
    """Least common multiple of a and b."""
    return abs(a * b) // math.gcd(a, b)


def mean(numbers: list[float]) -> float:
    """Arithmetic mean of a list of numbers."""
    if not numbers:
        raise ValueError("mean() requires at least one number")
    return sum(numbers) / len(numbers)


def median(numbers: list[float]) -> float:
    """Median of a list of numbers."""
    if not numbers:
        raise ValueError("median() requires at least one number")
    nums = sorted(numbers)
    n = len(nums)
    mid = n // 2
    if n % 2 == 0:
        return (nums[mid - 1] + nums[mid]) / 2
    return nums[mid]


def std_dev(numbers: list[float]) -> float:
    """Population standard deviation of a list of numbers."""
    if not numbers:
        raise ValueError("std_dev() requires at least one number")
    m = mean(numbers)
    variance = sum((x - m) ** 2 for x in numbers) / len(numbers)
    return math.sqrt(variance)


def variance(numbers: list[float]) -> float:
    """Population variance of a list of numbers."""
    if not numbers:
        raise ValueError("variance() requires at least one number")
    m = mean(numbers)
    return sum((x - m) ** 2 for x in numbers) / len(numbers)


def is_perfect_square(n: int) -> bool:
    """Return True if n is a perfect square."""
    if n < 0:
        return False
    root = int(math.isqrt(n))
    return root * root == n


def harmonic_mean(numbers: list[float]) -> float:
    """Harmonic mean of a list of positive numbers."""
    if not numbers:
        raise ValueError("harmonic_mean() requires at least one number")
    return len(numbers) / sum(1 / x for x in numbers)


def geometric_mean(numbers: list[float]) -> float:
    """Geometric mean of a list of positive numbers."""
    if not numbers:
        raise ValueError("geometric_mean() requires at least one number")
    product = 1.0
    for x in numbers:
        product *= x
    return product ** (1 / len(numbers))


def mode(numbers: list[float]) -> list[float]:
    """Return the most frequently occurring value(s) in a list of numbers."""
    if not numbers:
        raise ValueError("mode() requires at least one number")
    counts: dict[float, int] = {}
    for x in numbers:
        counts[x] = counts.get(x, 0) + 1
    max_count = max(counts.values())
    return [val for val, cnt in counts.items() if cnt == max_count]


def percentile(numbers: list[float], p: float) -> float:
    """The p-th percentile (0-100) of a list of numbers, using linear interpolation."""
    if not numbers:
        raise ValueError("percentile() requires at least one number")
    if not (0 <= p <= 100):
        raise ValueError("p must be between 0 and 100")
    data = sorted(numbers)
    index = (p / 100) * (len(data) - 1)
    lower = int(index)
    upper = min(lower + 1, len(data) - 1)
    frac = index - lower
    return data[lower] + frac * (data[upper] - data[lower])


def covariance(x: list[float], y: list[float]) -> float:
    """Population covariance between two equal-length lists of numbers."""
    if len(x) != len(y) or not x:
        raise ValueError("covariance() requires two non-empty lists of equal length")
    mx, my = mean(x), mean(y)
    return sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / len(x)


def correlation(x: list[float], y: list[float]) -> float:
    """Pearson correlation coefficient between two equal-length lists of numbers."""
    return covariance(x, y) / (std_dev(x) * std_dev(y))


def data_range(numbers: list[float]) -> float:
    """Difference between the maximum and minimum values in a list."""
    if not numbers:
        raise ValueError("data_range() requires at least one number")
    return max(numbers) - min(numbers)