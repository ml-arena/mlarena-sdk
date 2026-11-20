from mlarena.bundle import make_bundle
from mlarena.chat import ChatConversation
from mlarena.ci_auth import PROD_BASE_URL, Credentials, resolve_credentials
from mlarena.client import MLArenaClient
from mlarena.exceptions import (
    AuthenticationError,
    ChallengeNotFoundError,
    ChatSessionNotFoundError,
    CompetitionNotFoundError,  # deprecated alias of ChallengeNotFoundError
    ExerciseAttemptNotFoundError,
    MaintenanceError,
    MLArenaError,
    NotFoundError,
    PermissionDeniedError,
    SubmissionError,
    SubmissionNotFoundError,
    WaitTimeoutError,
)

__version__ = "4.13.0"


def connect(
    api_key: str | None = None,
    base_url: str = PROD_BASE_URL,
    *,
    allow_insecure: bool = False,
    scope: str = "user",
    audience: str | None = None,
) -> MLArenaClient:
    """Connect to ML Arena.

    The credential is resolved in this order (`mlarena.ci_auth`):

    1. `api_key`;
    2. the `MLARENA_API_KEY` environment variable;
    3. inside a GitHub Actions job with `permissions: id-token: write`, the
       job's OIDC token, exchanged for a short-lived `mlo_…` token of the
       account the repository is linked to. No secret is stored.

    Args:
        api_key: Full bearer token in the canonical form
            ``mlk_<scope>_<lookup>_<secret>`` (copy it from your Profile page).
            Older `key_id:key_pass` colon-separated form is no longer supported.
        base_url: ML Arena server URL.
        allow_insecure: Opt-in escape hatch to allow ``http://`` URLs against
            non-loopback hosts. ``http://localhost`` and ``http://127.0.0.1``
            are always allowed without this flag (local dev). Default
            ``False`` because the bearer token is sent on every request and
            cleartext transport would expose it to anyone on the network path.
        scope: GitHub Actions only: the scope asked for the `mlo_` token
            (``user`` | ``creator`` | ``teacher``). An API key carries its own.
        audience: GitHub Actions only: the OIDC audience, the platform's
            public URL. Defaults to ``MLARENA_OIDC_AUDIENCE``, else `base_url`.

    Returns:
        MLArenaClient instance; its ``credentials`` say which of the three
        sources was used.

    Raises:
        AuthenticationError: If there is no credential, if the token doesn't
            look like a real `mlk_…` token, or if the GitHub Actions exchange
            is refused (the message is the server's). The backend ultimately
            decides authenticity on the first authenticated request.
    """
    def resolve():
        return resolve_credentials(
            api_key, base_url=base_url, scope=scope, audience=audience,
            allow_insecure=allow_insecure,
        )

    credentials = resolve()
    return MLArenaClient(
        token=credentials.token, base_url=base_url, credentials=credentials,
        # An exchanged token lives 15 minutes; a long `submit(wait=True)`
        # outlives it, so the client exchanges a new one when it runs out.
        renew_credentials=resolve if credentials.expires_at_ts is not None else None,
    )
