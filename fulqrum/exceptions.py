# Fulqrum
# Copyright (C) 2024, IBM
# pylint: disable=no-name-in-module
"""Fulqrum exceptions"""


class FulqrumError(Exception):
    """Base class for errors raised by Fulqrum"""

    def __init__(self, *message):
        """Set the error message."""
        super().__init__(" ".join(message))
        self.message = " ".join(message)

    def __str__(self):
        """Return the message."""
        return repr(self.message)
