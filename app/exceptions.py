class DigestError(Exception):
    """Expected error that can be shown to a CLI user."""


class InputError(DigestError):
    pass


class RetrievalError(DigestError):
    pass


class ParseError(DigestError):
    pass


class ConfigurationError(DigestError):
    pass
