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

def bisection_method(f, a: float, b: float, tol: float = 1e-10, max_iter: int = 200) -> float:
    """Find a root of f within [a, b] using the bisection method. Requires f(a) and f(b) to have opposite signs."""
    if f(a) * f(b) > 0:
        raise ValueError("f(a) and f(b) must have opposite signs")
    for _ in range(max_iter):
        mid = (a + b) / 2
        if abs(f(mid)) < tol or (b - a) / 2 < tol:
            return mid
        if f(a) * f(mid) < 0:
            b = mid
        else:
            a = mid
    return (a + b) / 2


def secant_method(f, x0: float, x1: float, tol: float = 1e-10, max_iter: int = 200) -> float:
    """Find a root of f near x0 and x1 using the secant method (no derivative required)."""
    for _ in range(max_iter):
        fx0, fx1 = f(x0), f(x1)
        if fx1 - fx0 == 0:
            raise ZeroDivisionError("Secant method failed: division by zero")
        x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        if abs(x2 - x1) < tol:
            return x2
        x0, x1 = x1, x2
    return x1


def partial_derivative(f, point: list[float], index: int, h: float = 1e-6) -> float:
    """Approximate the partial derivative of a multivariable function f at `point`,
    with respect to the variable at position `index`."""
    point_plus = list(point)
    point_minus = list(point)
    point_plus[index] += h
    point_minus[index] -= h
    return (f(*point_plus) - f(*point_minus)) / (2 * h)


def arc_length(f, a: float, b: float, n: int = 1000) -> float:
    """Approximate the arc length of y = f(x) from x=a to x=b."""
    def integrand(x: float) -> float:
        return math.sqrt(1 + derivative(f, x) ** 2)
    return integral_simpson(integrand, a, b, n)


def limit_approximation(f, x: float, h: float = 1e-6) -> float:
    """Approximate the limit of f(t) as t approaches x, by averaging values just before and after x
    (useful for functions undefined exactly at x, e.g. removable discontinuities)."""
    return (f(x - h) + f(x + h)) / 2