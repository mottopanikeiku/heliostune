"""Expected HeliosTune domain and trust-boundary failures."""

from __future__ import annotations


class HeliostuneError(Exception):
    """Base class for failures safe to present to a command-line user."""


class SchemaError(HeliostuneError, ValueError):
    """A decoded value does not satisfy a HeliosTune schema."""


class ArtifactError(HeliostuneError):
    """An artifact could not be read, verified, or committed safely."""


class ProtocolError(HeliostuneError, ValueError):
    """A requested collection or replay violates its frozen protocol."""
