"""
Tests for internal validation utilities.
"""

from __future__ import annotations

import pytest

from pyrkm._validation import validate_positive_float, validate_positive_int, validate_probability


class TestValidatePositiveInt:
    """
    Tests for validate_positive_int function.
    """

    def test_valid_positive_int(self) -> None:
        """
        Test with valid positive integer.
        """
        validate_positive_int(5, "test_param")  # Should not raise

    def test_zero_raises_error(self) -> None:
        """
        Test that zero raises ValueError.
        """
        with pytest.raises(ValueError, match="test_param must be a positive integer"):
            validate_positive_int(0, "test_param")

    def test_negative_raises_error(self) -> None:
        """
        Test that negative value raises ValueError.
        """
        with pytest.raises(ValueError, match="test_param must be a positive integer"):
            validate_positive_int(-5, "test_param")

    def test_float_raises_error(self) -> None:
        """
        Test that float raises ValueError.
        """
        with pytest.raises(ValueError, match="test_param must be a positive integer"):
            validate_positive_int(5.5, "test_param")


class TestValidatePositiveFloat:
    """
    Tests for validate_positive_float function.
    """

    def test_valid_positive_float(self) -> None:
        """
        Test with valid positive float.
        """
        validate_positive_float(5.5, "test_param")  # Should not raise

    def test_valid_positive_int(self) -> None:
        """
        Test with valid positive integer.
        """
        validate_positive_float(5, "test_param")  # Should not raise

    def test_zero_raises_error(self) -> None:
        """
        Test that zero raises ValueError.
        """
        with pytest.raises(ValueError, match="test_param must be a positive number"):
            validate_positive_float(0, "test_param")

    def test_negative_raises_error(self) -> None:
        """
        Test that negative value raises ValueError.
        """
        with pytest.raises(ValueError, match="test_param must be a positive number"):
            validate_positive_float(-5.5, "test_param")


class TestValidateProbability:
    """
    Tests for validate_probability function.
    """

    def test_valid_probability_zero(self) -> None:
        """
        Test with probability of 0.
        """
        validate_probability(0.0, "test_param")  # Should not raise

    def test_valid_probability_one(self) -> None:
        """
        Test with probability of 1.
        """
        validate_probability(1.0, "test_param")  # Should not raise

    def test_valid_probability_half(self) -> None:
        """
        Test with probability of 0.5.
        """
        validate_probability(0.5, "test_param")  # Should not raise

    def test_negative_raises_error(self) -> None:
        """
        Test that negative value raises ValueError.
        """
        with pytest.raises(ValueError, match="test_param must be between 0 and 1"):
            validate_probability(-0.1, "test_param")

    def test_greater_than_one_raises_error(self) -> None:
        """
        Test that value greater than 1 raises ValueError.
        """
        with pytest.raises(ValueError, match="test_param must be between 0 and 1"):
            validate_probability(1.5, "test_param")
