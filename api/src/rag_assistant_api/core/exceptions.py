class RAGAssistantError(Exception):
    """Base application error."""


class UnsupportedSourceError(RAGAssistantError):
    """Raised when a file or source type is unsupported."""


class ProviderConfigurationError(RAGAssistantError):
    """Raised when a provider is missing required configuration."""
