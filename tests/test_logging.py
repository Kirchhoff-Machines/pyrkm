"""
Tests for internal logging utilities.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from pyrkm._logging import setup_logger


class TestSetupLogger:
    """
    Tests for setup_logger function.
    """

    def test_console_only_logger(self) -> None:
        """
        Test logger with console output only.
        """
        logger = setup_logger("test_logger")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_logger"
        assert logger.level == logging.INFO

    def test_logger_with_file(self) -> None:
        """
        Test logger with file output.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            logger = setup_logger("test_file_logger", log_file=log_file)

            assert isinstance(logger, logging.Logger)
            assert log_file.exists()

            # Test logging works
            logger.info("Test message")
            with open(log_file) as f:
                content = f.read()
                assert "Test message" in content

    def test_logger_level(self) -> None:
        """
        Test logger with custom level.
        """
        logger = setup_logger("test_debug", level=logging.DEBUG)
        assert logger.level == logging.DEBUG

    def test_logger_creates_directory(self) -> None:
        """
        Test logger creates parent directories.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "subdir" / "nested" / "test.log"
            setup_logger("test_nested", log_file=log_file)

            assert log_file.exists()
            assert log_file.parent.exists()

    def test_logger_no_duplicate_handlers(self) -> None:
        """
        Test that calling setup_logger twice doesn't add duplicate handlers.
        """
        logger1 = setup_logger("test_dup")
        handler_count1 = len(logger1.handlers)

        logger2 = setup_logger("test_dup")
        handler_count2 = len(logger2.handlers)

        assert handler_count1 == handler_count2
        assert logger1 is logger2  # Same logger instance
