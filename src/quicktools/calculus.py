"""Calculus: numerical differentiation, integration, and series approximations."""
import math
from typing import Callable


def derivative(f: Callable[[float], float], x: float, h: float = 1e-6) -> float:
    """Approximate f'(x) using the central difference method."""
    return (f(x + h) - f(x - h)) / (2 * h)


def second_derivative(f: Callable[[float], float], x: float, h: float = 1e-4) -> float:
    """Approximate f''(x) using the central difference method."""
    return (f(x + h) - 2 * f(x) + f(x - h)) / (h ** 2)


def integral_trapezoidal(f: Callable[[float], float], a: float, b: float, n: int = 1000) -> float:
    """Approximate the definite integral of f from a to b using the trapezoidal rule."""
    if n <= 0:
        raise ValueError("n must be positive")
    h = (b - a) / n
    total = (f(a) + f(b)) / 2
    for i in range(1, n):
        total += f(a + i * h)
    return total * h


def integral_simpson(f: Callable[[float], float], a: float, b: float, n: int = 1000) -> float:
    """Approximate the definite integral of f from a to b using Simpson's rule (n must be even)."""
    if n % 2 != 0:
        n += 1
    h = (b - a) / n
    total = f(a) + f(b)
    for i in range(1, n):
        coeff = 4 if i % 2 != 0 else 2
        total += coeff * f(a + i * h)
    return total * h / 3


def taylor_series_exp(x: float, terms: int = 15) -> float:
    """Approximate e^x using its Taylor series expansion around 0."""
    return sum((x ** n) / math.factorial(n) for n in range(terms))


def taylor_series_sin(x: float, terms: int = 15) -> float:
    """Approximate sin(x) using its Taylor series expansion around 0."""
    return sum(((-1) ** n) * (x ** (2 * n + 1)) / math.factorial(2 * n + 1) for n in range(terms))


def taylor_series_cos(x: float, terms: int = 15) -> float:
    """Approximate cos(x) using its Taylor series expansion around 0."""
    return sum(((-1) ** n) * (x ** (2 * n)) / math.factorial(2 * n) for n in range(terms))


def newtons_method(f: Callable[[float], float], x0: float, tol: float = 1e-10, max_iter: int = 100) -> float:
    """Find a root of f near x0 using Newton's method."""
    x = x0
    for _ in range(max_iter):
        fx = f(x)
        if abs(fx) < tol:
            return x
        dfx = derivative(f, x)
        if dfx == 0:
            raise ZeroDivisionError("Derivative is zero; Newton's method failed")
        x = x - fx / dfx
    return x