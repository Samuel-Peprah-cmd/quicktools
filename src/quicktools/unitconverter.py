"""Unit conversion utilities: temperature, distance, weight, volume, speed, and data storage."""


def celsius_to_fahrenheit(c: float) -> float:
    """Convert Celsius to Fahrenheit."""
    return c * 9 / 5 + 32


def fahrenheit_to_celsius(f: float) -> float:
    """Convert Fahrenheit to Celsius."""
    return (f - 32) * 5 / 9


def celsius_to_kelvin(c: float) -> float:
    """Convert Celsius to Kelvin."""
    return c + 273.15


def kelvin_to_celsius(k: float) -> float:
    """Convert Kelvin to Celsius."""
    return k - 273.15


def km_to_miles(km: float) -> float:
    """Convert kilometers to miles."""
    return km * 0.621371


def miles_to_km(miles: float) -> float:
    """Convert miles to kilometers."""
    return miles / 0.621371


def meters_to_feet(m: float) -> float:
    """Convert meters to feet."""
    return m * 3.28084


def feet_to_meters(ft: float) -> float:
    """Convert feet to meters."""
    return ft / 3.28084


def kg_to_pounds(kg: float) -> float:
    """Convert kilograms to pounds."""
    return kg * 2.20462


def pounds_to_kg(lb: float) -> float:
    """Convert pounds to kilograms."""
    return lb / 2.20462


def liters_to_gallons(liters: float) -> float:
    """Convert liters to US gallons."""
    return liters * 0.264172


def gallons_to_liters(gallons: float) -> float:
    """Convert US gallons to liters."""
    return gallons / 0.264172


def mph_to_kmh(mph: float) -> float:
    """Convert miles per hour to kilometers per hour."""
    return mph * 1.60934


def kmh_to_mph(kmh: float) -> float:
    """Convert kilometers per hour to miles per hour."""
    return kmh / 1.60934


def convert_bytes(value: float, from_unit: str, to_unit: str) -> float:
    """Convert a data storage value between units: 'B', 'KB', 'MB', 'GB', 'TB' (base 1024)."""
    units = {"B": 0, "KB": 1, "MB": 2, "GB": 3, "TB": 4}
    if from_unit not in units or to_unit not in units:
        raise ValueError(f"Units must be one of {list(units.keys())}")
    from_bytes = value * (1024 ** units[from_unit])
    return from_bytes / (1024 ** units[to_unit])