"""SDK tests for the terms of use an author accepts before their first
challenge (`backend/app/services/terms.py`, `backend/app/auth/routes/terms.py`).

- `accept_terms()` mirrors `POST /api/auth/terms/accept`; `me()` serves
  `terms_accepted_at_ts`.
- Every route that creates a challenge answers 403 `{"error": …, "reason":
  "terms_required"}` until then: the SDK raises it as a
  `PermissionDeniedError` whose body is the server's and whose message adds
  where the terms are read and how they are accepted. It never accepts them
  for the caller.

The compositions that create a course first are in
`test_course_challenge_helpers.py`. Offline: the client's single HTTP
chokepoint (`MLArenaClient._request`) is replaced by a recorder.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mlarena
from mlarena.ci_auth import KIND_GITHUB_ACTIONS, Credentials
from mlarena.client import TERMS_REQUIRED, TERMS_REQUIRED_REASON, MLArenaClient

ACCEPT = ("POST", "/auth/terms/accept")
ACCEPTED_AT = "2026-11-26T10:15:00Z"
# The body of the refusal, as the backend writes it (`services/terms.py`).
SERVER_SENTENCE = (
    "Accept the terms of use before you create a challenge: once per account, in the "
    "console when you create your first challenge, or with POST /api/auth/terms/accept."
)
REFUSAL = {"error": SERVER_SENTENCE, "reason": "terms_required"}
CI_TOKEN = "mlo_0123456789abcdef_0123456789abcdef0123456789abcdef"


class FakeResponse:
    def __init__(self, status_code, json_data):
        self.status_code = status_code
        self._json = json_data
        self.text = ""

    def json(self):
        return self._json


class Recorder:
    def __init__(self, status, data):
        self.calls = []
        self.reply = (status, data)

    def __call__(self, method, url, **kwargs):
        self.calls.append({"method": method, "path": url.split("/api", 1)[1], **kwargs})
        return FakeResponse(*self.reply)

    def signature(self):
        return [(call["method"], call["path"]) for call in self.calls]


def make_client(status, data, scope="author"):
    client = mlarena.connect(api_key=f"mlk_{scope}_abcd1234_deadbeef0000",
                             base_url="https://example.com")
    client._request = Recorder(status, data)
    return client, client._request


def make_ci_client(status, data):
    """The client of a GitHub Actions run: an `mlo_` token of the exchange."""
    credentials = Credentials(token=CI_TOKEN, kind=KIND_GITHUB_ACTIONS, scope="author",
                              username="alice")
    client = MLArenaClient(token=CI_TOKEN, base_url="https://example.com",
                           credentials=credentials)
    client._request = Recorder(status, data)
    return client, client._request


# The four routes that create a challenge, each through its SDK method.
CREATING_CALLS = {
    "build in a course": lambda c: c.add_course_challenge(7, "Maze", kernel_version="flex_v1"),
    "copy into a course": lambda c: c.add_course_challenge(7, copy_from_challenge_id=12),
    "competition": lambda c: c.create_challenge("Open maze", "flex_v1"),
    "competition copied": lambda c: c.create_challenge(copy_from_challenge_id=12),
    "module copied": lambda c: c.copy_module_into_course(7, 3),
    "course duplicated": lambda c: c.duplicate_course(7),
}


# ---- accepting ---------------------------------------------------------------------

def test_accept_terms_posts_no_body_and_returns_the_date():
    client, rec = make_client(200, {"terms_accepted_at_ts": ACCEPTED_AT})

    assert client.accept_terms() == {"terms_accepted_at_ts": ACCEPTED_AT}

    assert rec.signature() == [ACCEPT]
    call = rec.calls[0]
    assert call["json"] is None and call["params"] is None
    assert call["headers"]["Authorization"].startswith("Bearer mlk_author_")
    assert "Content-Type" not in call["headers"]


def test_accepting_again_returns_what_the_server_answers():
    """Once per account: the server keeps the first date, the method adds
    no state of its own."""
    client, rec = make_client(200, {"terms_accepted_at_ts": "2026-03-01T00:00:00Z"})

    assert client.accept_terms() == client.accept_terms() == {
        "terms_accepted_at_ts": "2026-03-01T00:00:00Z"}
    assert rec.signature() == [ACCEPT, ACCEPT]


@pytest.mark.parametrize("sentence", [
    "Key scope 'user' cannot access this route: it takes an 'author' key",
    "A GitHub Actions token cannot accept the terms of use for the account: "
    "accept them in the console or with an API key",
])
def test_who_may_not_accept_is_refused_in_the_servers_words(sentence):
    client, _ = make_client(403, {"error": sentence}, scope="user")

    with pytest.raises(mlarena.PermissionDeniedError) as raised:
        client.accept_terms()

    assert str(raised.value) == sentence
    assert raised.value.status_code == 403 and raised.value.body == {"error": sentence}


def test_me_serves_when_the_terms_were_accepted():
    me = {
        "is_authenticated": True, "id": 2, "username": "alice",
        "email": "alice@example.com", "avatar_key": None, "is_admin": False,
        "is_human_verified": True, "has_teacher_access": True, "can_create_course": True,
        "max_courses_limit": 10, "active_course_count": 1, "github_login": None,
        "terms_accepted_at_ts": None,
    }
    client, rec = make_client(200, me)

    assert client.me() == me and client.me()["terms_accepted_at_ts"] is None
    assert rec.signature()[0] == ("GET", "/auth/current_user")


# ---- the refusal -------------------------------------------------------------------

def test_the_sdk_names_the_refusal_as_the_backend_does():
    assert (TERMS_REQUIRED, TERMS_REQUIRED_REASON) == (SERVER_SENTENCE, "terms_required")


@pytest.mark.parametrize("name", sorted(CREATING_CALLS))
def test_creating_a_challenge_without_the_terms_is_a_branchable_refusal(name):
    client, rec = make_client(403, REFUSAL)

    with pytest.raises(mlarena.PermissionDeniedError) as raised:
        CREATING_CALLS[name](client)

    err = raised.value
    assert type(err) is mlarena.PermissionDeniedError
    assert err.status_code == 403 and err.body == REFUSAL
    assert err.body["reason"] == "terms_required"
    # The server's sentence, then where to read the terms and how to accept.
    assert str(err) == (
        f"{SERVER_SENTENCE} Read them at https://example.com/terms , then call "
        "client.accept_terms() (once per account, with an author key)."
    )
    # One request: the refused one. Nothing accepts the terms for the caller.
    assert len(rec.calls) == 1 and ACCEPT not in rec.signature()


def test_a_github_actions_run_is_told_that_the_account_holder_accepts():
    """A workflow's token cannot accept (the route refuses it): the message
    does not tell the run to call `accept_terms()` itself."""
    client, rec = make_ci_client(403, REFUSAL)

    with pytest.raises(mlarena.PermissionDeniedError) as raised:
        client.add_course_challenge(7, "Maze", kernel_version="flex_v1")

    assert raised.value.body == REFUSAL
    assert str(raised.value) == (
        f"{SERVER_SENTENCE} Read them at https://example.com/terms . A GitHub Actions "
        "run cannot accept them for the account: its holder accepts once, in the "
        "console or with an API key (client.accept_terms())."
    )
    assert ACCEPT not in rec.signature()


def test_another_403_of_a_creating_route_is_left_as_it_is():
    """The route's own guard answers before the terms (a course that is not
    the caller's): no `reason`, the server's sentence alone."""
    client, _ = make_client(403, {"error": "You do not manage this course"})

    with pytest.raises(mlarena.PermissionDeniedError) as raised:
        client.add_course_challenge(7, "Maze", kernel_version="flex_v1")

    assert str(raised.value) == "You do not manage this course"
    assert "reason" not in raised.value.body
