class MLArenaError(Exception):
    """Base exception for mlarena SDK.

    An error raised for an HTTP reply carries that reply: `status_code` is
    its HTTP status and `body` its decoded JSON object, so a caller can read
    what the server said beyond the message — a deploy refused with 409
    exposes `err.body["deployment_limits"]` / `err.body["active_submission_limits"]`,
    the same two blocks `submission_deploy_status()` returns. Both are None
    when the error is not an HTTP refusal (a local validation, a timeout, a
    `submit(wait=True)` that ended `deploy_failed` — there `body` is the final
    status payload and `status_code` is None) or when the reply had no JSON
    object body. The message is unchanged from earlier releases.
    """

    def __init__(self, message: str = "", *,
                 status_code: int | None = None,
                 body: dict | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


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


class MaintenanceError(MLArenaError):
    """Raised when the platform is in maintenance mode (HTTP 503).

    Every route but login/logout answers `{"error": <admin message>,
    "maintenance_mode": true}` while an admin has maintenance on (admins pass
    through). The message is the admin-set one; retry once it is over. A 503
    without that flag (a data source proxy whose upstream is down, an
    ingress) is not maintenance and stays a plain `MLArenaError`.
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


class ChatSessionNotFoundError(NotFoundError):
    """Raised when a chat session is not found, or is not yours to read.

    A session is readable by its user, their teammates, the challenge's
    creator / assistants and admins; anyone else gets the same 404 as for an
    id that does not exist (see `SubmissionNotFoundError`).
    """
    pass


# Deprecated spelling, kept so `except CompetitionNotFoundError` in existing
# notebooks still catches the same error. It is the same class, not a subclass,
# so isinstance checks against either name behave identically.
CompetitionNotFoundError = ChallengeNotFoundError
