from urllib.parse import urlsplit

from mlarena.client import MLArenaClient
from mlarena.exceptions import (
    AuthenticationError,
    CompetitionNotFoundError,
    MLArenaError,
    SubmissionError,
)

__version__ = "0.2.0"

_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def connect(
    api_key: str,
    base_url: str = "https://ml-arena.com",
    *,
    allow_insecure: bool = False,
) -> MLArenaClient:
    """Connect to ML Arena.

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

    Returns:
        MLArenaClient instance.

    Raises:
        AuthenticationError: If the token doesn't look like a real `mlk_…`
            token. The backend ultimately decides authenticity on the first
            authenticated request.
    """
    if not api_key or not api_key.startswith("mlk_"):
        raise AuthenticationError(
            "Invalid api_key format. Expected a token of the form "
            "'mlk_<scope>_<lookup>_<secret>' (copy it from your Profile page)."
        )
    parts = api_key.split("_", 3)
    if len(parts) != 4 or any(not part for part in parts):
        raise AuthenticationError(
            "Invalid api_key shape. Expected exactly 4 underscore-separated parts "
            "(prefix, scope, lookup, secret)."
        )
    parsed = urlsplit(base_url)
    if parsed.scheme != "https":
        is_local = parsed.hostname in _LOCAL_HOSTS
        if not is_local and not allow_insecure:
            raise AuthenticationError(
                f"Refusing to send bearer token over '{parsed.scheme}://'. "
                "Use an https:// base_url, or pass allow_insecure=True if you "
                "are pointing at a non-public dev server you control."
            )
    return MLArenaClient(token=api_key, base_url=base_url)
