class MLArenaError(Exception):
    """Base exception for mlarena SDK."""
    pass


class AuthenticationError(MLArenaError):
    """Raised when API key is invalid or missing."""
    pass


class SubmissionError(MLArenaError):
    """Raised when agent submission fails."""
    pass


class CompetitionNotFoundError(MLArenaError):
    """Raised when competition is not found."""
    pass
