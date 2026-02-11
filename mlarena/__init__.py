from mlarena.client import MLArenaClient
from mlarena.exceptions import MLArenaError, AuthenticationError, SubmissionError, CompetitionNotFoundError

__version__ = "0.1.0"


def connect(api_key, base_url="https://ml-arena.com"):
    """
    Connect to ML Arena.

    Args:
        api_key: API key in format "key_id:key_pass" (from your Profile page)
        base_url: ML Arena server URL

    Returns:
        MLArenaClient instance

    Raises:
        AuthenticationError: If api_key format is invalid
    """
    if not api_key or ":" not in api_key:
        raise AuthenticationError(
            "Invalid api_key format. Expected 'key_id:key_pass'. "
            "Get your keys from your Profile page on ML Arena."
        )

    key_id, key_pass = api_key.split(":", 1)
    if not key_id or not key_pass:
        raise AuthenticationError("Both key_id and key_pass must be non-empty.")

    return MLArenaClient(key_id=key_id, key_pass=key_pass, base_url=base_url.rstrip("/"))
