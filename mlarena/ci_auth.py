"""Which credential a client or the `mlarena` CLI sends.

One resolution order, used by `mlarena.connect()` and by the CLI
(`resolve_credentials`):

1. the `api_key` argument (an `mlk_…` key; an `mlc_…` course key is
   accepted there too) or the `course_key` argument (an `mlc_…` key), never
   both;
2. the `MLARENA_API_KEY` or the `MLARENA_COURSE_KEY` environment variable,
   never both;
3. inside a GitHub Actions job that has `permissions: id-token: write`
   (`ACTIONS_ID_TOKEN_REQUEST_URL` and `ACTIONS_ID_TOKEN_REQUEST_TOKEN` are
   set): the job's OIDC token, exchanged at `POST /api/auth/oidc/github` for a
   short-lived `mlo_…` token. No secret is stored in the repository. Skipped
   with `oidc=False` (the course commands: a course is pushed with its key).

With none of them the call raises, naming them. Nothing is guessed: a
key that is set but malformed is an error, never a reason to try the next
source.

A **course key** (`mlc_<lookup>_<secret>`) belongs to one course: it is
minted in the course's Settings › GitHub (`create_course_key`) and stored as
the repository secret `MLARENA_COURSE_KEY`. It reaches two routes only:
`GET /api/auth/course_key` (which course it is) and that course's bundle push.
`ci_context_headers` is what a push sends about the GitHub run it comes from.
"""

import os
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from urllib.parse import quote, urlsplit

import requests

from mlarena.exceptions import (
    AuthenticationError,
    MLArenaError,
    PermissionDeniedError,
)

PROD_BASE_URL = "https://ml-arena.com"

ENV_API_KEY = "MLARENA_API_KEY"
ENV_COURSE_KEY = "MLARENA_COURSE_KEY"
ENV_BASE_URL = "MLARENA_BASE_URL"
ENV_OIDC_AUDIENCE = "MLARENA_OIDC_AUDIENCE"
# Set by the Actions runner for a job with `permissions: id-token: write`.
ENV_ACTIONS_OIDC_URL = "ACTIONS_ID_TOKEN_REQUEST_URL"
ENV_ACTIONS_OIDC_TOKEN = "ACTIONS_ID_TOKEN_REQUEST_TOKEN"

# The backend route that trades a GitHub OIDC token for an `mlo_` token, and
# the keys of its body and reply (`backend/app/auth/routes/oidc.py`).
OIDC_EXCHANGE_PATH = "/api/auth/oidc/github"
OIDC_SCOPES = ("user", "author")
# The two scopes `author` replaced (4.23.0): a published client's spelling,
# warned and sent as `author`. The server reads them as `author` too.
LEGACY_OIDC_SCOPES = ("creator", "teacher")

# Which course a course key belongs to (`backend/app/auth/routes/course_key.py`):
# `CourseKeyWhoamiOut{course_id, slug, edited_in, expires_at_ts}`.
COURSE_KEY_WHOAMI_PATH = "/api/auth/course_key"
# The `scope` of a course key's `Credentials`: it has no API-key scope, it
# pushes its course.
COURSE_KEY_SCOPE = "course"

KIND_API_KEY = "api_key"
KIND_COURSE_KEY = "course_key"
KIND_GITHUB_ACTIONS = "github_actions"

# The GitHub run a course push comes from, sent as request headers of the
# bundle push and recorded on the course's run row as reported (never
# trusted). Header -> the Actions variable it is read from; the run URL is
# built from four of them (`ci_context_headers`).
CI_HEADER_REPOSITORY = "X-MLArena-CI-Repository"
CI_HEADER_COMMIT = "X-MLArena-CI-Commit"
CI_HEADER_REF = "X-MLArena-CI-Ref"
CI_HEADER_RUN_URL = "X-MLArena-CI-Run-Url"
CI_HEADER_ACTOR = "X-MLArena-CI-Actor"

_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


@dataclass(frozen=True)
class Credentials:
    """The bearer token a client sends, and where it came from.

    `kind` is `"api_key"` (an `mlk_…` key; `scope` is the key's own scope
    segment), `"course_key"` (an `mlc_…` key of one course; `scope` is
    `"course"`) or `"github_actions"` (an `mlo_…` token from the OIDC exchange;
    `scope` is the scope that was asked for, and `username`, `expires_at_ts`
    and `source_revision` are the exchange's reply: the account the
    repository is linked to and the commit the run is building).
    """

    token: str
    kind: str
    scope: str
    username: str | None = None
    # As served: an ISO 8601 UTC timestamp ("2026-11-18T10:15:00Z").
    expires_at_ts: str | None = None
    source_revision: dict | None = None

    def expires_within(self, seconds: float) -> bool:
        """True when the token has an expiry that is less than `seconds`
        away. An API key has none, and a course key is not renewed (its
        expiry is the one `GET /api/auth/course_key` serves)."""
        if self.expires_at_ts is None:
            return False
        expires_at = datetime.fromisoformat(self.expires_at_ts.replace("Z", "+00:00"))
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        return expires_at - datetime.now(timezone.utc) < timedelta(seconds=seconds)


def check_transport(base_url: str, allow_insecure: bool) -> None:
    """Refuse to send a credential in cleartext to a non-loopback host."""
    parsed = urlsplit(base_url)
    if parsed.scheme != "https":
        is_local = parsed.hostname in _LOCAL_HOSTS
        if not is_local and not allow_insecure:
            raise AuthenticationError(
                f"Refusing to send bearer token over '{parsed.scheme}://'. "
                "Use an https:// base_url, or pass allow_insecure=True if you "
                "are pointing at a non-public dev server you control."
            )


def check_token_shape(token: str) -> str:
    """Validate a bearer token's shape and return its kind.

    `mlk_<scope>_<lookup>_<secret>` is an API key, `mlc_<lookup>_<secret>` a
    course key, `mlo_<lookup>_<secret>` a CI token. The backend decides
    authenticity on the first request.
    """
    if token and token.startswith("mlc_"):
        parts = token.split("_")
        if len(parts) != 3 or any(not part for part in parts):
            raise AuthenticationError(
                "Invalid course key shape. Expected 'mlc_<lookup>_<secret>' "
                "(copy it from the course's Settings › GitHub)."
            )
        return KIND_COURSE_KEY
    if token and token.startswith("mlo_"):
        parts = token.split("_")
        if len(parts) != 3 or any(not part for part in parts):
            raise AuthenticationError(
                "Invalid CI token shape. Expected 'mlo_<lookup>_<secret>'."
            )
        return KIND_GITHUB_ACTIONS
    if not token or not token.startswith("mlk_"):
        raise AuthenticationError(
            "Invalid api_key format. Expected a token of the form "
            "'mlk_<scope>_<lookup>_<secret>' (copy it from your Profile page)."
        )
    parts = token.split("_", 3)
    if len(parts) != 4 or any(not part for part in parts):
        raise AuthenticationError(
            "Invalid api_key shape. Expected exactly 4 underscore-separated parts "
            "(prefix, scope, lookup, secret)."
        )
    return KIND_API_KEY


def oidc_audience(base_url: str, audience: str | None = None,
                  environ=None) -> str:
    """The audience the GitHub OIDC token is asked for: the platform's public
    URL. `audience`, else `MLARENA_OIDC_AUDIENCE`, else the base URL (they
    differ only on a local stack, where the API and the console have two
    ports)."""
    environ = os.environ if environ is None else environ
    return audience or environ.get(ENV_OIDC_AUDIENCE) or base_url.rstrip("/")


def in_github_actions_oidc(environ=None) -> bool:
    """True inside an Actions job that may request an OIDC token."""
    environ = os.environ if environ is None else environ
    return bool(environ.get(ENV_ACTIONS_OIDC_URL)
                and environ.get(ENV_ACTIONS_OIDC_TOKEN))


# Both requests of this module carry a credential: no redirect is followed.
_HTTP_OPTIONS = {"allow_redirects": False, "timeout": 30}


def _http(method: str, url: str, **kwargs) -> requests.Response:
    """The one HTTP call of this module (tests replace it)."""
    return requests.request(method, url, **kwargs)


def _reason(resp: requests.Response) -> str:
    """The server's own message (`{"error": …}`), else the reply's text."""
    try:
        body = resp.json()
    except ValueError:
        return resp.text
    if isinstance(body, dict) and body.get("error"):
        return body["error"]
    return resp.text


def _body(resp: requests.Response) -> dict | None:
    try:
        body = resp.json()
    except ValueError:
        return None
    return body if isinstance(body, dict) else None


def fetch_github_oidc_token(audience: str, environ=None) -> str:
    """Ask the Actions runner for this job's OIDC token (a JWT)."""
    environ = os.environ if environ is None else environ
    request_url = environ[ENV_ACTIONS_OIDC_URL]
    separator = "&" if "?" in request_url else "?"
    resp = _http(
        "GET",
        f"{request_url}{separator}audience={quote(audience, safe='')}",
        headers={"Authorization": f"bearer {environ[ENV_ACTIONS_OIDC_TOKEN]}"},
        **_HTTP_OPTIONS,
    )
    if resp.status_code != 200:
        raise AuthenticationError(
            f"GitHub Actions refused to issue an OIDC token "
            f"(HTTP {resp.status_code}): {_reason(resp)}",
            status_code=resp.status_code, body=_body(resp),
        )
    return resp.json()["value"]


def exchange_github_oidc_token(base_url: str, token: str, scope: str) -> dict:
    """Trade a GitHub OIDC token for a short-lived `mlo_` token.

    Mirrors `POST /api/auth/oidc/github` `{"token", "scope"}` →
    `{"access_token", "expires_at_ts", "username", "source_revision"}`. A
    refusal raises with the server's message unchanged (it says what to fix:
    link GitHub in Profile, add the repository to the trusted ones, …):
    `AuthenticationError` on 401, `PermissionDeniedError` on 403.
    """
    resp = _http(
        "POST", f"{base_url.rstrip('/')}{OIDC_EXCHANGE_PATH}",
        headers={"Content-Type": "application/json"},
        json={"token": token, "scope": scope},
        **_HTTP_OPTIONS,
    )
    if resp.status_code == 200:
        return resp.json()
    reason = _reason(resp)
    if resp.status_code == 401:
        cls = AuthenticationError
    elif resp.status_code == 403:
        cls = PermissionDeniedError
    else:
        cls = MLArenaError
        reason = f"GitHub OIDC exchange failed (HTTP {resp.status_code}): {reason}"
    raise cls(reason, status_code=resp.status_code, body=_body(resp))


def ci_context_headers(environ=None) -> dict:
    """The `X-MLArena-CI-*` headers of a course push, read from the GitHub
    Actions variables of the job (empty outside a GitHub Actions job).

    The course's run row records them as the run reports them: repository
    (`GITHUB_REPOSITORY`), commit (`GITHUB_SHA`), ref (`GITHUB_REF`), actor
    (`GITHUB_ACTOR`) and the run's URL,
    `$GITHUB_SERVER_URL/$GITHUB_REPOSITORY/actions/runs/$GITHUB_RUN_ID/attempts/$GITHUB_RUN_ATTEMPT`.
    A variable that is not set sends no header (the run URL needs all four).
    """
    environ = os.environ if environ is None else environ
    if environ.get("GITHUB_ACTIONS") != "true":
        return {}
    headers = {}
    for header, name in ((CI_HEADER_REPOSITORY, "GITHUB_REPOSITORY"),
                         (CI_HEADER_COMMIT, "GITHUB_SHA"),
                         (CI_HEADER_REF, "GITHUB_REF"),
                         (CI_HEADER_ACTOR, "GITHUB_ACTOR")):
        if environ.get(name):
            headers[header] = environ[name]
    parts = [environ.get(name) for name in (
        "GITHUB_SERVER_URL", "GITHUB_REPOSITORY", "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT")]
    if all(parts):
        server, repository, run_id, attempt = parts
        headers[CI_HEADER_RUN_URL] = (
            f"{server.rstrip('/')}/{repository}/actions/runs/{run_id}/attempts/{attempt}")
    return headers


def _key_credentials(token: str, source: str, scope: str) -> Credentials:
    """The `Credentials` of an `mlk_` or `mlc_` key named by `source` (the
    argument or the variable it came from, for the error)."""
    if source in ("course_key", ENV_COURSE_KEY):
        if token == "":
            raise AuthenticationError(
                f"{source} is set but empty. In a GitHub Actions workflow, add the "
                "repository secret MLARENA_COURSE_KEY (the course's Settings › GitHub "
                "creates the key) and pass it to the step as "
                "`env: MLARENA_COURSE_KEY: ${{ secrets.MLARENA_COURSE_KEY }}`."
            )
        if check_token_shape(token) != KIND_COURSE_KEY:
            raise AuthenticationError(
                f"{source} holds a key that is not a course key: expected "
                "'mlc_<lookup>_<secret>' (the course's Settings › GitHub). A personal "
                "key goes in MLARENA_API_KEY."
            )
        return Credentials(token=token, kind=KIND_COURSE_KEY, scope=COURSE_KEY_SCOPE)
    kind = check_token_shape(token)
    if kind == KIND_API_KEY:
        return Credentials(token=token, kind=kind, scope=token.split("_", 3)[1])
    if kind == KIND_COURSE_KEY:
        return Credentials(token=token, kind=kind, scope=COURSE_KEY_SCOPE)
    # An `mlo_` token passed as it is: the scope it was exchanged for.
    return Credentials(token=token, kind=kind, scope=scope)


def resolve_credentials(api_key: str | None = None, *,
                        course_key: str | None = None,
                        base_url: str = PROD_BASE_URL,
                        scope: str = "user",
                        audience: str | None = None,
                        allow_insecure: bool = False,
                        oidc: bool = True,
                        environ=None) -> Credentials:
    """Resolve the bearer token to send (module docstring: the order).

    `scope` and `audience` matter only for the GitHub Actions exchange: an
    API key carries its own, and a course key pushes its course. `oidc=False`
    never tries the exchange: without a key the call raises.
    """
    environ = os.environ if environ is None else environ
    check_transport(base_url, allow_insecure)

    if api_key is not None and course_key is not None:
        raise AuthenticationError("pass api_key= or course_key=, not both")
    if api_key is not None:
        return _key_credentials(api_key, "api_key", scope)
    if course_key is not None:
        return _key_credentials(course_key, "course_key", scope)
    env_key, env_course_key = environ.get(ENV_API_KEY), environ.get(ENV_COURSE_KEY)
    if env_key is not None and env_course_key is not None:
        raise AuthenticationError(
            f"both {ENV_API_KEY} and {ENV_COURSE_KEY} are set: unset one. A course "
            f"edited on GitHub is pushed with its course key ({ENV_COURSE_KEY}); "
            f"everything else takes a personal key ({ENV_API_KEY})."
        )
    if env_key is not None:
        return _key_credentials(env_key, ENV_API_KEY, scope)
    if env_course_key is not None:
        return _key_credentials(env_course_key, ENV_COURSE_KEY, scope)

    if oidc and in_github_actions_oidc(environ):
        if scope in LEGACY_OIDC_SCOPES:
            warnings.warn(
                f"scope={scope!r} is deprecated: the {scope} scope is now "
                "'author'; pass scope='author'",
                DeprecationWarning,
                stacklevel=3,
            )
            scope = "author"
        if scope not in OIDC_SCOPES:
            raise MLArenaError(
                f"Unknown scope {scope!r}; expected one of {', '.join(OIDC_SCOPES)}"
            )
        jwt = fetch_github_oidc_token(
            oidc_audience(base_url, audience, environ), environ)
        reply = exchange_github_oidc_token(base_url, jwt, scope)
        token = reply["access_token"]
        check_token_shape(token)
        return Credentials(
            token=token, kind=KIND_GITHUB_ACTIONS, scope=scope,
            username=reply["username"],
            expires_at_ts=reply["expires_at_ts"],
            source_revision=reply["source_revision"],
        )

    if not oidc:
        raise AuthenticationError(
            f"No ML Arena key. A course is pushed with its course key: set "
            f"{ENV_COURSE_KEY} (the course's Settings › GitHub creates it; in a "
            f"workflow, the repository secret of that name), or {ENV_API_KEY} with a "
            f"personal author key for a course edited in the console."
        )
    hint = ""
    if environ.get("GITHUB_ACTIONS") == "true":
        hint = (" This GitHub Actions job cannot request an OIDC token: add "
                "`permissions: id-token: write` (and `contents: read`) to it.")
    raise AuthenticationError(
        "No ML Arena credential. Use one of: (1) pass api_key= "
        "(mlarena.connect(api_key=\"mlk_…\")); (2) set the MLARENA_API_KEY "
        "environment variable (or MLARENA_COURSE_KEY, a course's key); (3) run "
        "inside a GitHub Actions job with "
        "`permissions: id-token: write`, from a repository linked to your "
        "account (Profile → GitHub)." + hint
    )
