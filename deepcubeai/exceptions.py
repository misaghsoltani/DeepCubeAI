from __future__ import annotations


class DeepCubeAIError(Exception):
    """Base package error."""


class ConfigError(DeepCubeAIError):
    """Invalid or missing configuration detected."""


class ExternalServiceError(DeepCubeAIError):
    """Failures when interacting with external services."""
