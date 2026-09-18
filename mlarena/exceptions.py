class MLArenaError(Exception):
    """Base exception for mlarena SDK."""
    pass


class AuthenticationError(MLArenaError):
    """Raised when API key is invalid or missing (HTTP 401)."""
    pass


class PermissionDeniedError(AuthenticationError):
    """Raised when the token is valid but the action is not allowed (HTTP 403).

    A subclass of :class:`AuthenticationError` because that is what a 403 used
    to raise, and the SDK is a published client — `except AuthenticationError`
    in an existing notebook still catches it.
    """
    pass


class SubmissionError(MLArenaError):
    """Raised when a submission operation fails."""
    pass


class NotFoundError(MLArenaError):
    """Base for "the thing you asked for is not there" (HTTP 404).

    Catch this when you do not care *what* was missing. The platform answers
    404 rather than 403 for a resource the caller may not see, so this is also
    what a submission that belongs to someone else raises.
    """
    pass


class ChallengeNotFoundError(NotFoundError):
    """Raised when a challenge — or a course, module, lesson or dataset — is not found."""
    pass


class SubmissionNotFoundError(NotFoundError):
    """Raised when a submission is not found, or is not yours to read.

    The server does not distinguish the two on purpose: a 403 on a submission
    id would confirm that the id exists.
    """
    pass


# Deprecated spelling, kept so `except CompetitionNotFoundError` in existing
# notebooks still catches the same error. It is the same class, not a subclass,
# so isinstance checks against either name behave identically.
CompetitionNotFoundError = ChallengeNotFoundError
