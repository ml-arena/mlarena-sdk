class MLArenaError(Exception):
    """Base exception for mlarena SDK."""
    pass


class AuthenticationError(MLArenaError):
    """Raised when API key is invalid or missing."""
    pass


class SubmissionError(MLArenaError):
    """Raised when agent submission fails."""
    pass


class ChallengeNotFoundError(MLArenaError):
    """Raised when challenge is not found."""
    pass


# Deprecated spelling, kept so `except CompetitionNotFoundError` in existing
# notebooks still catches the same error. It is the same class, not a subclass,
# so isinstance checks against either name behave identically.
CompetitionNotFoundError = ChallengeNotFoundError
