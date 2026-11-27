class MLArenaError(Exception):
    """Base exception for mlarena SDK.

    An error raised for an HTTP reply carries that reply: `status_code` is
    its HTTP status and `body` its decoded JSON object, so a caller can read
    what the server said beyond the message — a deploy refused with 409
    exposes `err.body["deployment_limits"]` / `err.body["active_submission_limits"]`,
    the same two blocks `submission_deploy_status()` returns. A refusal with
    a cause a program may act on names it in `err.body["reason"]`, beside
    `error` (the sentence, which is the message here):

    - `"out_of_credits"` (409): the owner of the challenge has no compute
      credits left on the balance its machine draws from. A benchmark run, a
      re-score, an update of a started challenge's files, a self-test and
      Start are refused with it; nothing is written, and the call succeeds
      again once the owner has credits (`credits()`).
    - `"on_hold"` (409): the challenge is paused (its owner's credits are
      used up). A chat conversation or message and an exercise Check are
      refused with it; the sentence names no credit. A deploy is not
      refused: its runs wait.
    - `"test_runs_cap"` (409): the owner of the challenge already has 3
      test runs queued or running (benchmark runs and exercise self-tests,
      all their challenges together): `run_benchmark` and
      `self_test_exercises` queue nothing; call again once one finished.
    - `"datasets_total_cap"` (409): `upload_dataset_file` would take the
      datasets of the challenge's owner over 20 GiB in total; nothing is
      stored (`err.body["used_bytes"]`, `["size_bytes"]`, `["max_bytes"]`).
    - `"enrollment_required"` (403): `EnrollmentRequiredError`.
    - `"terms_required"` (403, `PermissionDeniedError`): a challenge was
      about to be created (built or copied, in a course or as a competition;
      a module or a course copied with a challenge in it) by an account that
      has not accepted the terms of use. Nothing is created. The message
      names the page of the terms and `client.accept_terms()`, which the
      account's holder calls once, after reading them; the SDK never
      accepts on its own. The compositions that would create a course first
      (`create_challenge_for_class`, `use_example_challenge` into a new
      course) raise it before they create anything.
    - `"files_unreadable"` (503): the challenge's files could not be read;
      nothing was changed, try again.

    `status_code` and `body` are None
    when the error is not an HTTP refusal (a local validation, a timeout, a
    `submit(wait=True)` that ended `deploy_failed` — there `body` is the final
    status payload and `status_code` is None) or when the reply had no JSON
    object body. The message is unchanged from earlier releases.

    `submission_id` is the submission a `submit()` call created before it
    failed (a refused upload or deploy, a failed wait), so the caller can
    deploy it later or delete it; None on every other error.
    """

    def __init__(self, message: str = "", *,
                 status_code: int | None = None,
                 body: dict | None = None,
                 submission_id: int | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body
        self.submission_id = submission_id


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


class EnrollmentRequiredError(PermissionDeniedError):
    """Raised when taking part in a course challenge needs an enrollment the
    caller does not have (HTTP 403, `reason: "enrollment_required"`).

    A challenge of a course is taken part in by the course's members: a
    submission, an upload, a deploy, a team, a chat message or an exercise
    Check is refused to anyone else, even on a public course whose page they
    read. `course_id` is the course to enroll in (`enroll_in_course` with its
    join code, `course(slug)["join_code"]` on a public course that takes
    enrollment); the message is the server's. A subclass of
    `PermissionDeniedError`, what the same 403 raised before 4.23.0.
    """

    @property
    def course_id(self) -> int:
        return self.body["course_id"]


class CourseEditedOnGithubError(MLArenaError):
    """Raised when a write reaches a course edited on GitHub (HTTP 409,
    `edited_in: "github"`).

    A course is edited in one place: the console (and the SDK, with a
    personal key) or a GitHub repository, whose workflow pushes it with the
    course's key (`mlarena course push`). In GitHub mode every write of its
    content (lessons, modules, a challenge's files, config, page and data)
    is refused with this error, and so is a push of the course with a
    personal key; operations (enrollment, staff, Start and Stop, dates and
    visibility) stay open. The message is the server's sentence; `repository`
    is the `owner/name` the course is edited in. Edit the repository, or
    switch the course back with `switch_course_edited_in(course_id,
    "console")`, which revokes its keys.
    """

    @property
    def repository(self) -> str:
        return self.body["repository"]


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


class WaitTimeoutError(SubmissionError):
    """Raised when a wait on a submission (`tail_logs(timeout_sec=)`,
    `submit(wait=True, timeout_sec=)`) runs out before the deploy settled.

    The submission is still running: `body` is its last status payload
    (`submission()`), so `err.body["id"]` and `err.body["status"]` say what
    to poll. A subclass of `SubmissionError`, which is what the timeout used
    to raise.
    """
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
    staff (its owner and assistants) and admins; anyone else gets the same 404 as for an
    id that does not exist (see `SubmissionNotFoundError`).
    """
    pass


class ExerciseAttemptNotFoundError(NotFoundError):
    """Raised when an exercise attempt is not found, or is not yours to read.

    An attempt is readable by its group (the user, or their team), the
    challenge's staff (its owner and assistants), admins and the student's
    teachers; anyone else gets the same 404 as for an id that does not exist (see
    `SubmissionNotFoundError`).
    """
    pass


# Deprecated spelling, kept so `except CompetitionNotFoundError` in existing
# notebooks still catches the same error. It is the same class, not a subclass,
# so isinstance checks against either name behave identically.
CompetitionNotFoundError = ChallengeNotFoundError
