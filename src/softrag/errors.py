"""Exception hierarchy for softrag.

Every error raised by the library derives from :class:`SoftragError`, so callers
can catch the whole surface with a single ``except`` while still discriminating
specific failures when they care.
"""

from __future__ import annotations

__all__ = [
    "SoftragError",
    "ConfigurationError",
    "MissingDependencyError",
    "StoreError",
    "SchemaVersionError",
    "DimensionMismatchError",
    "IngestionError",
    "UnsupportedFormatError",
    "ExtractionError",
    "ProviderError",
    "EmbeddingError",
    "ChatError",
]


class SoftragError(Exception):
    """Base class for every error raised by softrag."""


class ConfigurationError(SoftragError):
    """The engine was configured in a way that cannot work."""


class MissingDependencyError(ConfigurationError):
    """An optional dependency is required for the requested feature.

    The message always names the extra to install, so the fix is copy-pasteable.
    """

    def __init__(self, package: str, *, extra: str, feature: str) -> None:
        self.package = package
        self.extra = extra
        self.feature = feature
        super().__init__(
            f"{feature} requires the '{package}' package, which is not installed.\n"
            f"Install it with:  pip install 'softrag[{extra}]'"
        )


class StoreError(SoftragError):
    """Something went wrong at the SQLite storage layer."""


class SchemaVersionError(StoreError):
    """The database was written by an incompatible version of softrag."""

    def __init__(self, found: int, expected: int, path: str) -> None:
        self.found = found
        self.expected = expected
        super().__init__(
            f"Database {path!r} uses schema version {found}, but this version of "
            f"softrag speaks version {expected}. "
            + (
                "The file was written by a newer softrag; upgrade the library."
                if found > expected
                else "Run `softrag migrate` or re-index into a fresh database file."
            )
        )


class DimensionMismatchError(StoreError):
    """An embedding of the wrong width was handed to a store already in use."""

    def __init__(self, expected: int, got: int) -> None:
        self.expected = expected
        self.got = got
        super().__init__(
            f"This database stores {expected}-dimensional vectors but the embedder "
            f"returned {got} dimensions. You are most likely using a different "
            f"embedding model than the one the index was built with. Either switch "
            f"back to the original model or re-index into a new database file."
        )


class IngestionError(SoftragError):
    """Content could not be turned into indexable text."""


class UnsupportedFormatError(IngestionError):
    """No extractor is registered for this kind of input."""


class ExtractionError(IngestionError):
    """An extractor was found but failed on this particular input."""


class ProviderError(SoftragError):
    """A pluggable model backend misbehaved."""


class EmbeddingError(ProviderError):
    """The embedding backend failed or returned something unusable."""


class ChatError(ProviderError):
    """The chat backend failed or returned something unusable."""
