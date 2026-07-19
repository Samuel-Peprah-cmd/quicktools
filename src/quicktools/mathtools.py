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