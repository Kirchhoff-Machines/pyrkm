"""
Simple validation utilities for input checking.
"""

from __future__ import annotations


def validate_positive_int(value: int, name: str) -> None:
    """Validate that a value is a positive integer.

    Parameters
    ----------
    value : int
        The value to validate.
    name : str
        The name of the parameter for error messages.

    Raises
    ------
    ValueError
        If the value is not a positive integer.
    """
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}")


def validate_positive_float(value: float, name: str) -> None:
    """Validate that a value is a positive float.

    Parameters
    ----------
    value : float
        The value to validate.
    name : str
        The name of the parameter for error messages.

    Raises
    ------
    ValueError
        If the value is not a positive float.
    """
    if not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{name} must be a positive number, got {value}")


def validate_probability(value: float, name: str) -> None:
    """Validate that a value is a valid probability (between 0 and 1).

    Parameters
    ----------
    value : float
        The value to validate.
    name : str
        The name of the parameter for error messages.

    Raises
    ------
    ValueError
        If the value is not between 0 and 1.
    """
    if not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1, got {value}")
