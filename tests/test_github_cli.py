"""Offline tests for the GitHub Actions path: credential resolution (API key,
`MLARENA_API_KEY`, OIDC exchange), `make_bundle`, the bundle / trusted
repository / same-commit client methods, and the `mlarena` command.

Same harness as the other suites: `MLArenaClient._request` is replaced by a
recorder; the two HTTP calls of `mlarena.ci_auth` go through its `_http`,
replaced the same way. The CLI is driven through `main([...])` with a fake
client.
"""
import io
import os
import sys
import zipfile
from datetime import datetime, timedelta, timezone

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mlarena
from mlarena import ci_auth, cli
from mlarena.bundle import DEFAULT_EXCLUDES, bundle_files, make_bundle, read_bundle
from mlarena.exceptions import (
    AuthenticationError,
    MLArenaError,
    PermissionDeniedError,
    SubmissionError,
    WaitTimeoutError,
)

API_KEY = "mlk_user_abcd1234_deadbeef0000"
CI_TOKEN = "mlo_0123456789abcdef_0123456789abcdef0123456789abcdef"
REVISION = {
    "id": 5, "provider": "github", "repository": "alice/agent",
    "commit_sha": "0123456789abcdef0123456789abcdef01234567",
    "ref": "refs/tags/submission-3", "transport": "actions_oidc",
    "event_name": "push", "run_url": "https://github.com/alice/agent/actions/runs/1",
    "commit_url": "https://github.com/alice/agent/commit/0123456", "created_at_ts": "t",
}

_GITHUB_ENV = (
    "MLARENA_API_KEY", "MLARENA_BASE_URL", "MLARENA_OIDC_AUDIENCE",
    "ACTIONS_ID_TOKEN_REQUEST_URL", "ACTIONS_ID_TOKEN_REQUEST_TOKEN",
    "GITHUB_ACTIONS", "GITHUB_STEP_SUMMARY", "GITHUB_OUTPUT", "GITHUB_SHA", "GITHUB_REF_NAME",
    "GITHUB_REF_TYPE",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """No test reads the developer's (or a CI runner's) own credentials."""
    for name in _GITHUB_ENV:
        monkeypatch.delenv(name, raising=False)


class FakeResponse:
    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json = json_data
        self.text = text

    def json(self):
        if self._json is None:
            raise ValueError("no JSON body")
        return self._json


class Recorder:
    """Stands in for `client._request` and for `ci_auth._http`."""

    def __init__(self, router=None):
        self.calls = []
        self.router = router

    def __call__(self, method, url, **kwargs):
        self.calls.append({"method": method, "url": url,
                           "path": url.split("/api", 1)[-1], **kwargs})
        result = self.router(method, url, kwargs) if self.router else None
        if result is None:
            return FakeResponse(200, {"ok": True})
        return result if isinstance(result, FakeResponse) else FakeResponse(*result)

    @property
    def last(self):
        return self.calls[-1]


def make_client(router=None, scope="user"):
    client = mlarena.connect(api_key=f"mlk_{scope}_abcd1234_deadbeef0000",
                             base_url="https://example.com")
    rec = Recorder(router)
    client._request = rec
    return client, rec


# --------------------------------------------------------------------------- #
# Credential resolution
# --------------------------------------------------------------------------- #


def _actions_env(monkeypatch):
    monkeypatch.setenv("ACTIONS_ID_TOKEN_REQUEST_URL",
                       "https://runner.example/token?api-version=2")
    monkeypatch.setenv("ACTIONS_ID_TOKEN_REQUEST_TOKEN", "runner-secret")


def _iso_in(seconds):
    moment = datetime.now(timezone.utc) + timedelta(seconds=seconds)
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _oidc_http(exchange=None, expires_in=900):
    """`ci_auth._http`: the runner's token endpoint, then the exchange."""
    def router(method, url, kwargs):
        if url.startswith("https://runner.example/"):
            return (200, {"value": "the.github.jwt"})
        if exchange is not None:
            return exchange
        return (200, {"access_token": CI_TOKEN, "expires_at_ts": _iso_in(expires_in),
                      "username": "alice", "source_revision": REVISION})
    return Recorder(router)


def test_explicit_api_key_wins_over_everything(monkeypatch):
    monkeypatch.setenv("MLARENA_API_KEY", "mlk_creator_ffff0000_deadbeef0000")
    _actions_env(monkeypatch)
    http = _oidc_http()
    monkeypatch.setattr(ci_auth, "_http", http)

    client = mlarena.connect(API_KEY, base_url="https://example.com")
    assert client._token == API_KEY
    assert client.credentials.kind == "api_key"
    assert client.credentials.scope == "user"
    assert http.calls == []


def test_env_api_key_wins_over_the_oidc_exchange(monkeypatch):
    monkeypatch.setenv("MLARENA_API_KEY", "mlk_creator_ffff0000_deadbeef0000")
    _actions_env(monkeypatch)
    http = _oidc_http()
    monkeypatch.setattr(ci_auth, "_http", http)

    client = mlarena.connect(base_url="https://example.com", scope="teacher")
    assert client._token == "mlk_creator_ffff0000_deadbeef0000"
    # The key's own scope, not the one asked for.
    assert client.credentials.scope == "creator"
    assert http.calls == []


def test_github_actions_oidc_exchange(monkeypatch):
    _actions_env(monkeypatch)
    http = _oidc_http()
    monkeypatch.setattr(ci_auth, "_http", http)

    client = mlarena.connect(base_url="https://example.com", scope="creator")

    fetch, exchange = http.calls
    # The runner's URL already has a query string: `&audience=`, URL-encoded.
    assert fetch["method"] == "GET"
    assert fetch["url"] == ("https://runner.example/token?api-version=2"
                            "&audience=https%3A%2F%2Fexample.com")
    assert fetch["headers"] == {"Authorization": "bearer runner-secret"}
    assert exchange["method"] == "POST"
    assert exchange["url"] == "https://example.com/api/auth/oidc/github"
    assert exchange["json"] == {"token": "the.github.jwt", "scope": "creator"}
    # Neither call follows a redirect: both carry a credential.
    assert fetch["allow_redirects"] is False and exchange["allow_redirects"] is False

    assert client._token == CI_TOKEN
    creds = client.credentials
    assert (creds.kind, creds.scope, creds.username) == ("github_actions", "creator", "alice")
    assert creds.source_revision == REVISION
    # The mlo_ token is the bearer of every later request, and never printed.
    assert client._headers() == {"Authorization": f"Bearer {CI_TOKEN}"}
    assert CI_TOKEN.split("_")[2] not in repr(client)


def test_an_expiring_ci_token_is_exchanged_again(monkeypatch):
    """The exchanged token lives 15 minutes; `submit --wait` can outlive it."""
    _actions_env(monkeypatch)
    second = "mlo_fedcba9876543210_0123456789abcdef0123456789abcdef"
    state = {"exchanges": 0}

    def router(method, url, kwargs):
        if url.startswith("https://runner.example/"):
            return (200, {"value": "the.github.jwt"})
        state["exchanges"] += 1
        first = state["exchanges"] == 1
        return (200, {"access_token": CI_TOKEN if first else second,
                      # The first token is about to expire, the second is fresh.
                      "expires_at_ts": _iso_in(30 if first else 900),
                      "username": "alice", "source_revision": REVISION})
    monkeypatch.setattr(ci_auth, "_http", Recorder(router))

    client = mlarena.connect(base_url="https://example.com")
    assert client._token == CI_TOKEN
    assert client._headers() == {"Authorization": f"Bearer {second}"}
    assert client.credentials.token == second
    # A fresh token is kept: no exchange per request.
    client._headers()
    assert state["exchanges"] == 2


def test_an_api_key_is_never_renewed(monkeypatch):
    http = _oidc_http()
    monkeypatch.setattr(ci_auth, "_http", http)
    client = mlarena.connect(API_KEY, base_url="https://example.com")
    assert client._renew_credentials is None
    assert client.credentials.expires_within(10**9) is False
    client._headers()
    assert http.calls == []


def test_oidc_audience_defaults_to_base_url_and_is_overridable(monkeypatch):
    _actions_env(monkeypatch)
    http = _oidc_http()
    monkeypatch.setattr(ci_auth, "_http", http)

    # A local stack: the API on :4999, the public URL (the audience) on :3000.
    monkeypatch.setenv("MLARENA_OIDC_AUDIENCE", "http://localhost:3000")
    mlarena.connect(base_url="http://localhost:4999")
    assert http.calls[0]["url"].endswith("&audience=http%3A%2F%2Flocalhost%3A3000")
    assert http.calls[1]["url"] == "http://localhost:4999/api/auth/oidc/github"

    # The argument wins over the environment.
    mlarena.connect(base_url="http://localhost:4999", audience="https://other.example")
    assert http.calls[2]["url"].endswith("&audience=https%3A%2F%2Fother.example")


@pytest.mark.parametrize("status, cls", [(401, AuthenticationError),
                                         (403, PermissionDeniedError)])
def test_oidc_refusal_surfaces_the_server_message_verbatim(monkeypatch, status, cls):
    message = ("alice/agent is not owned by GitHub user bob and is not one of "
               "alice's trusted repositories. Add it at https://example.com/profile "
               "(Trusted repositories), or with the SDK: "
               "client.add_trusted_repository('alice/agent').")
    _actions_env(monkeypatch)
    monkeypatch.setattr(ci_auth, "_http", _oidc_http((status, {"error": message})))

    with pytest.raises(cls) as raised:
        mlarena.connect(base_url="https://example.com")
    assert type(raised.value) is cls
    assert str(raised.value) == message
    assert raised.value.status_code == status


def test_oidc_token_request_refused_by_the_runner(monkeypatch):
    _actions_env(monkeypatch)
    monkeypatch.setattr(ci_auth, "_http",
                        Recorder(lambda *_: FakeResponse(500, None, "boom")))
    with pytest.raises(AuthenticationError, match="refused to issue an OIDC token"):
        mlarena.connect(base_url="https://example.com")


def test_no_credential_names_the_three_ways(monkeypatch):
    with pytest.raises(AuthenticationError) as raised:
        mlarena.connect()
    message = str(raised.value)
    assert "api_key=" in message
    assert "MLARENA_API_KEY" in message
    assert "id-token: write" in message

    # In Actions without the permission, the message says what is missing.
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    with pytest.raises(AuthenticationError, match="cannot request an OIDC token"):
        mlarena.connect()


def test_one_actions_variable_alone_is_not_an_oidc_job(monkeypatch):
    monkeypatch.setenv("ACTIONS_ID_TOKEN_REQUEST_URL", "https://runner.example/token")
    with pytest.raises(AuthenticationError, match="No ML Arena credential"):
        mlarena.connect()


def test_a_malformed_key_is_an_error_not_a_reason_to_try_oidc(monkeypatch):
    _actions_env(monkeypatch)
    http = _oidc_http()
    monkeypatch.setattr(ci_auth, "_http", http)
    monkeypatch.setenv("MLARENA_API_KEY", "not-a-key")
    with pytest.raises(AuthenticationError, match="Invalid api_key format"):
        mlarena.connect()
    with pytest.raises(AuthenticationError, match="Invalid api_key format"):
        mlarena.connect("")
    with pytest.raises(AuthenticationError, match="exactly 4"):
        mlarena.connect("mlk_user_only")
    assert http.calls == []


def test_mlo_token_passes_as_an_explicit_credential():
    client = mlarena.connect(CI_TOKEN, base_url="https://example.com")
    assert client.credentials.kind == "github_actions"
    with pytest.raises(AuthenticationError, match="CI token shape"):
        mlarena.connect("mlo_only", base_url="https://example.com")


def test_cleartext_transport_is_refused_before_any_exchange(monkeypatch):
    _actions_env(monkeypatch)
    http = _oidc_http()
    monkeypatch.setattr(ci_auth, "_http", http)
    with pytest.raises(AuthenticationError, match="Refusing to send"):
        mlarena.connect(base_url="http://example.com")
    assert http.calls == []


# --------------------------------------------------------------------------- #
# make_bundle
# --------------------------------------------------------------------------- #


def _write(root, relpath, content=b"x"):
    path = os.path.join(root, *relpath.split("/"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(content)
    return path


def _names(payload):
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        return archive.namelist()


def test_make_bundle_is_rooted_sorted_and_skips_housekeeping(tmp_path):
    root = str(tmp_path / "challenge")
    for relpath in (
        "env.py", "challenge.toml", "data/train.csv", "data/sub/x.bin",
        ".git/config", ".git/objects/ab/cd", "__pycache__/env.cpython-312.pyc",
        "data/__pycache__/a.pyc", "stray.pyc", ".DS_Store", "data/.DS_Store",
        ".venv/bin/python", "node_modules/pkg/index.js",
        ".ipynb_checkpoints/nb-checkpoint.ipynb", ".gitignore",
    ):
        _write(root, relpath)

    names = _names(make_bundle(root))
    # No wrapping folder; a dotfile that is not housekeeping is kept.
    assert names == [".gitignore", "challenge.toml", "data/sub/x.bin",
                     "data/train.csv", "env.py"]
    assert bundle_files(root) == names
    assert set(DEFAULT_EXCLUDES) == {".git", "__pycache__", "*.pyc", ".DS_Store",
                                     ".venv", "node_modules", ".ipynb_checkpoints"}


def test_make_bundle_is_deterministic(tmp_path):
    root = str(tmp_path / "c")
    _write(root, "env.py", b"class Env: pass\n")
    _write(root, "data/a.csv", b"1,2\n")
    first = make_bundle(root)
    os.utime(os.path.join(root, "env.py"), (1, 1))  # another mtime, same bytes
    assert make_bundle(root) == first
    with zipfile.ZipFile(io.BytesIO(first)) as archive:
        assert archive.read("env.py") == b"class Env: pass\n"


def test_make_bundle_include_and_exclude(tmp_path):
    root = str(tmp_path / "c")
    for relpath in ("a.md", "img/a.png", "notes/draft.md", "big.bin"):
        _write(root, relpath)
    assert _names(make_bundle(root, include=["*.md", "img/*"])) == [
        "a.md", "img/a.png", "notes/draft.md"]
    assert _names(make_bundle(root, exclude=["notes", "*.bin"])) == ["a.md", "img/a.png"]


def test_make_bundle_refuses_symlinks(tmp_path):
    root = str(tmp_path / "c")
    _write(root, "env.py")
    outside = _write(str(tmp_path), "secret.txt")
    os.symlink(outside, os.path.join(root, "link.txt"))
    with pytest.raises(MLArenaError, match="link.txt is a symbolic link"):
        make_bundle(root)
    os.remove(os.path.join(root, "link.txt"))

    os.symlink(str(tmp_path), os.path.join(root, "linked_dir"))
    with pytest.raises(MLArenaError, match="linked_dir is a symbolic link"):
        make_bundle(root)
    os.remove(os.path.join(root, "linked_dir"))

    # A link inside an excluded folder is never looked at (.venv is full of them).
    os.makedirs(os.path.join(root, ".venv"))
    os.symlink(outside, os.path.join(root, ".venv", "python"))
    assert _names(make_bundle(root)) == ["env.py"]


def test_make_bundle_refuses_nothing_and_non_directories(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(MLArenaError, match="Nothing to bundle"):
        make_bundle(str(empty))
    with pytest.raises(MLArenaError, match="Not a directory"):
        make_bundle(str(tmp_path / "missing"))


def test_read_bundle_takes_a_directory_or_an_archive(tmp_path):
    root = str(tmp_path / "c")
    _write(root, "env.py")
    name, payload = read_bundle(root)
    assert name == "bundle.zip" and _names(payload) == ["env.py"]

    archive = _write(str(tmp_path), "repo.tar.gz", b"raw")
    assert read_bundle(archive) == ("repo.tar.gz", b"raw")
    with pytest.raises(MLArenaError, match="neither a directory nor"):
        read_bundle(_write(str(tmp_path), "notes.txt"))
    with pytest.raises(MLArenaError, match="No such directory or archive"):
        read_bundle(str(tmp_path / "nope"))


# --------------------------------------------------------------------------- #
# Client methods: route, verb, payload
# --------------------------------------------------------------------------- #

CHALLENGE_REPORT = {
    "message": "Bundle applied", "dry_run": False,
    "env_files": ["challenge.toml", "env.py"], "env_changed": True,
    "overview_updated": True, "agent_template_updated": False,
    "image_updated": False, "ignored": [".github/workflows/push.yml"],
    "config": {"values": {}, "origin": {"simulation_timeout_sec": "file",
                                        "number_of_agents": "default"},
               "file_lines": {}, "file_applied_at_ts": None, "file_in_sync": True},
    "validation": {"ok": True, "errors": [], "warnings": []},
    "source_revision": None,
}


def test_push_challenge_uploads_a_directory_bundle(tmp_path):
    root = str(tmp_path / "challenge")
    _write(root, "env.py")
    _write(root, ".git/HEAD")
    c, rec = make_client(lambda *_: (200, CHALLENGE_REPORT), scope="creator")

    assert c.push_challenge(12, root) == CHALLENGE_REPORT
    call = rec.last
    assert call["method"] == "PUT"
    assert call["path"] == "/creator_challenge/challenge/12/bundle"
    assert call["params"] is None
    filename, payload = call["files"]["archive"]
    assert filename == "bundle.zip" and _names(payload) == ["env.py"]

    c.push_challenge(12, root, dry_run=True)
    assert rec.last["params"] == {"dry_run": "true"}


def test_push_subpath_of_a_directory_bundles_only_that_folder(tmp_path):
    """One repository, several challenges or courses: `subpath` means the same
    thing for both pushes, the folder inside the directory that is bundled."""
    repo = str(tmp_path / "repo")
    _write(repo, "README.md")
    _write(repo, "challenges/kart/env.py")
    _write(repo, "courses/rl/course.json", b'{"course": {"name": "RL", "slug": "rl"}}')
    _write(repo, "courses/rl/intro/a.md")

    c, rec = make_client(lambda *_: (200, CHALLENGE_REPORT), scope="creator")
    c.push_challenge(12, repo, subpath="challenges/kart")
    assert rec.last["params"] is None  # nothing left for the server to select
    assert _names(rec.last["files"]["archive"][1]) == ["env.py"]

    c, rec = make_client(lambda *_: (200, COURSE_PLAN), scope="teacher")
    c.push_course(repo, subpath="courses/rl")
    # The slug is read from the manifest of the sub-folder.
    assert rec.last["path"] == "/teacher/courses/rl/bundle"
    assert _names(rec.last["files"]["archive"][1]) == ["course.json", "intro/a.md"]

    for bad in ("nope", "../repo", "/etc"):
        with pytest.raises(MLArenaError):
            c.push_course(repo, subpath=bad)
        with pytest.raises(MLArenaError):
            c.push_challenge(12, repo, subpath=bad)


def test_push_subpath_of_an_archive(tmp_path):
    archive = _write(str(tmp_path), "repo.zip", b"zipbytes")
    # The challenge route selects the folder itself (`path` query).
    c, rec = make_client(lambda *_: (200, CHALLENGE_REPORT), scope="creator")
    c.push_challenge(12, archive, subpath="challenges/kart", dry_run=True)
    assert rec.last["params"] == {"path": "challenges/kart", "dry_run": "true"}
    assert rec.last["files"]["archive"] == ("repo.zip", b"zipbytes")
    # The course route has no such query: refused before any request.
    c, rec = make_client(lambda *_: (200, COURSE_PLAN), scope="teacher")
    with pytest.raises(MLArenaError, match="only with a directory"):
        c.push_course(archive, slug="rl", subpath="courses/rl")
    assert rec.calls == []


def test_push_challenge_sends_an_archive_file_as_it_is(tmp_path):
    archive = _write(str(tmp_path), "challenge.zip", b"zipbytes")
    c, rec = make_client(lambda *_: (200, CHALLENGE_REPORT), scope="creator")
    c.push_challenge(12, archive)
    assert rec.last["files"]["archive"] == ("challenge.zip", b"zipbytes")


def test_push_challenge_refusal_keeps_the_server_message(tmp_path):
    root = str(tmp_path / "challenge")
    _write(root, "env.py")
    c, _ = make_client(lambda *_: (400, {"error": "challenge.toml:3: unknown key 'x'"}),
                       scope="creator")
    with pytest.raises(MLArenaError) as raised:
        c.push_challenge(12, root)
    assert "challenge.toml:3: unknown key 'x'" in str(raised.value)
    assert raised.value.status_code == 400


def test_import_challenge_from_github_body():
    c, rec = make_client(lambda *_: (200, CHALLENGE_REPORT), scope="creator")
    c.import_challenge_from_github(12, "alice/kart")
    assert rec.last["method"] == "POST"
    assert rec.last["path"] == "/creator_challenge/challenge/12/import-github"
    assert rec.last["json"] == {"repository": "alice/kart", "ref": None,
                                "path": None, "dry_run": False}
    c.import_challenge_from_github(12, "alice/kart", ref="v1", path="env", dry_run=True)
    assert rec.last["json"] == {"repository": "alice/kart", "ref": "v1",
                                "path": "env", "dry_run": True}


def test_sync_env_from_github_is_gone():
    c, _ = make_client(scope="creator")
    assert not hasattr(c, "sync_env_from_github")


def test_trusted_repository_methods():
    row = {"id": 3, "repository": "org/agents", "repository_id": None,
           "created_at_ts": "t", "last_used_at_ts": None}
    c, rec = make_client(lambda method, *_: (
        (200, [row]) if method == "GET"
        else (201, row) if method == "POST" else (200, {"message": "removed"})))

    assert c.trusted_repositories() == [row]
    assert (rec.last["method"], rec.last["path"]) == (
        "GET", "/profile/github/trusted_repositories")

    assert c.add_trusted_repository("org/agents") == row
    assert (rec.last["method"], rec.last["path"]) == (
        "POST", "/profile/github/trusted_repositories")
    assert rec.last["json"] == {"repository": "org/agents"}

    c.remove_trusted_repository(3)
    assert (rec.last["method"], rec.last["path"]) == (
        "DELETE", "/profile/github/trusted_repositories/3")


def test_update_profile_no_longer_takes_github_page():
    c, _ = make_client()
    with pytest.raises(TypeError):
        c.update_profile(github_page="https://github.com/alice")


def test_create_submission_sends_allow_same_commit_only_when_asked():
    c, rec = make_client(lambda *_: (201, {"submission_id": 11}))
    c.create_submission(4, "s")
    assert rec.last["json"] == {"submission_name": "s"}
    c.create_submission(4, "s", allow_same_commit=True)
    assert rec.last["json"] == {"submission_name": "s", "allow_same_commit": True}


# What `submit()` reads off `challenge()` before it creates anything.
_CHALLENGE_4 = {"entry_filename": "agent.py", "max_upload_files": 10,
                "max_upload_size_bytes": 1 << 20, "runtime_options": []}


def _same_commit_router(method, url, kwargs):
    path = url.split("/api", 1)[1]
    if path == "/challenges/4":
        return (200, _CHALLENGE_4)
    if method == "POST" and path == "/submissions/challenge/4":
        return (409, {"error": "This commit was already submitted", "submission_id": 77})
    if path == "/submissions/submission/77":
        return (200, {"id": 77, "status": "active"})
    raise AssertionError(f"unexpected request {method} {path}")


def test_create_submission_409_body_is_reachable():
    c, _ = make_client(_same_commit_router)
    with pytest.raises(SubmissionError) as raised:
        c.create_submission(4, "s")
    assert raised.value.status_code == 409
    assert raised.value.body["submission_id"] == 77


def test_submit_returns_the_existing_submission_of_the_same_commit(tmp_path):
    agent = _write(str(tmp_path), "agent.py")
    c, rec = make_client(_same_commit_router)
    out = c.submit(4, files=[agent])
    assert out == {"submission_id": 77, "deploy": None, "already_submitted": True,
                   "status": {"id": 77, "status": "active"}}
    # Nothing uploaded, nothing deployed; status() reads the existing one.
    assert [(call["method"], call["path"]) for call in rec.calls] == [
        ("GET", "/challenges/4"),
        ("POST", "/submissions/challenge/4"), ("GET", "/submissions/submission/77")]
    assert c._last_submission_id == 77


def test_submit_reraises_a_409_without_a_submission_id(tmp_path):
    # e.g. a chat challenge: its submission is not created by this route.
    agent = _write(str(tmp_path), "agent.py")
    def router(method, url, kwargs):
        if url.endswith("/challenges/4"):
            return (200, _CHALLENGE_4)
        return (409, {"error": "Chat challenges have no upload"})

    c, _ = make_client(router)
    with pytest.raises(SubmissionError, match="no upload"):
        c.submit(4, files=[agent])


def _upload_only_router(method, url, kwargs):
    path = url.split("/api", 1)[1]
    if path == "/challenges/4":
        return (200, _CHALLENGE_4)
    if method == "POST" and path == "/submissions/challenge/4":
        return (201, {"submission_id": 11})
    if path == "/submissions/submission/11":
        return (200, {"id": 11, "status": "upload_validated", "is_deployable": True,
                      "last_status_message": None})
    if path.endswith("/file"):
        return (200, {"message": "ok"})
    raise AssertionError(f"unexpected request {method} {path}")


def test_submit_without_deploy_stops_after_the_upload_check(tmp_path):
    agent = _write(str(tmp_path), "agent.py")
    c, rec = make_client(_upload_only_router)
    out = c.submit(4, files=[agent], deploy=False, allow_same_commit=True)
    assert out == {"submission_id": 11, "deploy": None}
    assert rec.calls[1]["json"]["allow_same_commit"] is True
    assert not any(call["path"].endswith("/deploy") for call in rec.calls)
    with pytest.raises(SubmissionError, match="wait=True needs deploy=True"):
        c.submit(4, files=[agent], deploy=False, wait=True)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

COURSE_PLAN = {
    "dry_run": False,
    "course": {"id": 100, "slug": "intro-to-rl", "action": "updated"},
    "modules": [{
        "slug": "foundations", "id": 200, "action": "adopted", "renamed_from": "basics",
        "lessons": [
            {"slug": "what-is-rl", "id": 300, "action": "unchanged", "renamed_from": None},
            {"slug": "bandits", "id": None, "action": "created", "renamed_from": None},
        ],
        "orphaned_lessons": ["old-intro"], "pruned_lessons": [],
        "assets": {"written": 2, "removed": 1},
    }],
    "unmanaged_modules": ["extra"],
    "source_revision": REVISION,
    "url": "/courses/intro-to-rl",
    "warnings": [
        "module 'foundations', lesson 'bandits': mlarena:challenge: challenge 9 is not "
        "visible to the students of course 'Intro to RL' (it is private and the course "
        "does not assign it), so they will not see this block",
    ],
}


class FakeClient:
    """What the CLI calls on a client, recorded."""

    def __init__(self, *, kernel="flex_v1", submit_result=None, error=None,
                 credentials=None, challenge_report=None, course_plan=None):
        self.calls = []
        self.kernel = kernel
        self.submit_result = submit_result or {"submission_id": 11,
                                               "deploy": {"message": "queued"}}
        self.error = error
        self.credentials = credentials or ci_auth.Credentials(
            token=API_KEY, kind="api_key", scope="user")
        self.challenge_report = challenge_report or CHALLENGE_REPORT
        self.course_plan = course_plan or COURSE_PLAN

    def _challenge(self):
        return {"id": 4, "slug": "lunar-lander", "name": "Lunar Lander",
                "kernel_version": self.kernel, "submission_filename": (
                    "submission.csv" if self.kernel == "file_v1" else None)}

    def me(self):
        return {"username": "alice", "is_authenticated": True}

    def submission(self, submission_id):
        self.calls.append(("submission", submission_id))
        return {"id": submission_id, "status": "deploy_queue", "score": None}

    def challenge(self, challenge_id):
        self.calls.append(("challenge", challenge_id))
        return self._challenge()

    def challenge_by_slug(self, slug):
        self.calls.append(("challenge_by_slug", slug))
        return self._challenge()

    def submit(self, challenge_id, **kwargs):
        # The files are read now: the CLI may hand over a temporary copy.
        kwargs["file_names"] = [os.path.basename(path) for path in kwargs["files"]]
        kwargs["file_bytes"] = [open(path, "rb").read() for path in kwargs["files"]]
        self.calls.append(("submit", challenge_id, kwargs))
        if self.error is not None:
            raise self.error
        return self.submit_result

    def push_challenge(self, challenge_id, path, **kwargs):
        self.calls.append(("push_challenge", challenge_id, path, kwargs))
        if self.error is not None:
            raise self.error
        return self.challenge_report

    def push_course(self, path, **kwargs):
        self.calls.append(("push_course", path, kwargs))
        if self.error is not None:
            raise self.error
        return self.course_plan


@pytest.fixture
def fake(monkeypatch):
    """Install a FakeClient as the CLI's client; records how it was asked for."""
    def install(client):
        asked = []

        def connect(base_url, scope, audience):
            asked.append({"base_url": base_url, "scope": scope, "audience": audience})
            return client
        monkeypatch.setattr(cli, "_connect", connect)
        client.asked = asked
        return client
    return install


@pytest.fixture
def actions(monkeypatch, tmp_path):
    """A GitHub Actions job: returns the path of its step summary."""
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    return summary


def test_cli_whoami_with_an_api_key(fake, capsys):
    fake(FakeClient())
    assert cli.main(["whoami"]) == 0
    out = capsys.readouterr().out
    assert "username: alice" in out
    assert "API key (scope user)" in out
    assert "https://ml-arena.com" in out


def test_cli_whoami_in_a_github_actions_run(fake, capsys, monkeypatch):
    monkeypatch.setenv("MLARENA_BASE_URL", "http://localhost:4999")
    client = fake(FakeClient(credentials=ci_auth.Credentials(
        token=CI_TOKEN, kind="github_actions", scope="user", username="alice",
        source_revision=REVISION)))
    assert cli.main(["--audience", "http://localhost:3000", "whoami"]) == 0
    out = capsys.readouterr().out
    assert "GitHub Actions run of alice/agent@0123456 (scope user)" in out
    assert "http://localhost:4999" in out
    assert client.asked == [{"base_url": "http://localhost:4999", "scope": "user",
                             "audience": "http://localhost:3000"}]
    assert CI_TOKEN not in out


def test_cli_submit_by_slug_composes_the_sdk_submit(fake, capsys, tmp_path):
    agent = _write(str(tmp_path), "agent.py")
    model = _write(str(tmp_path), "model.pt")
    client = fake(FakeClient())
    code = cli.main(["submit", "--challenge", "lunar-lander", "--files", agent, model,
                     "--name", "v1", "--runtime", "torch"])
    assert code == 0
    assert client.asked[0]["scope"] == "user"
    assert client.calls[0] == ("challenge_by_slug", "lunar-lander")
    _, challenge_id, kwargs = client.calls[1]
    assert challenge_id == 4
    assert kwargs["file_names"] == ["agent.py", "model.pt"]
    assert kwargs["submission_name"] == "v1"
    assert kwargs["runtime"] == "torch"
    assert (kwargs["deploy"], kwargs["wait"], kwargs["allow_same_commit"]) == (
        True, False, False)
    out = capsys.readouterr().out
    assert "Submission #11 (deploy queued)" in out
    assert "https://ml-arena.com/submission/view/4/11" in out


def test_cli_submit_numeric_challenge_and_flags(fake, tmp_path):
    agent = _write(str(tmp_path), "agent.py")
    client = fake(FakeClient())
    assert cli.main(["submit", "--challenge", "4", "--files", agent, "--no-deploy",
                     "--allow-same-commit"]) == 0
    assert client.calls[0] == ("challenge", 4)
    kwargs = client.calls[1][2]
    assert (kwargs["deploy"], kwargs["allow_same_commit"]) == (False, True)
    # Outside GitHub Actions the name is left to the SDK.
    assert kwargs["submission_name"] is None


@pytest.mark.parametrize("env, expected", [
    ({"GITHUB_REF_TYPE": "tag", "GITHUB_REF_NAME": "submission-3",
      "GITHUB_SHA": "0123456789abcdef"}, "submission-3"),
    ({"GITHUB_REF_TYPE": "branch", "GITHUB_REF_NAME": "main",
      "GITHUB_SHA": "0123456789abcdef"}, "gh-0123456"),
    ({"GITHUB_REF_TYPE": "tag", "GITHUB_REF_NAME": "t" * 60,
      "GITHUB_SHA": "0123456789abcdef"}, "t" * 40),
    ({}, None),
])
def test_cli_submit_name_defaults_from_the_run(fake, tmp_path, monkeypatch, env, expected):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    agent = _write(str(tmp_path), "agent.py")
    client = fake(FakeClient())
    assert cli.main(["submit", "--challenge", "4", "--files", agent]) == 0
    assert client.calls[1][2]["submission_name"] == expected


def test_cli_submit_refuses_a_name_over_the_limit(fake, tmp_path, capsys):
    agent = _write(str(tmp_path), "agent.py")
    client = fake(FakeClient())
    assert cli.main(["submit", "--challenge", "4", "--files", agent,
                     "--name", "n" * 41]) == 2
    assert client.calls == []
    assert "limit is 40" in capsys.readouterr().err


def test_cli_submit_dir_takes_its_files_and_refuses_a_subfolder(fake, tmp_path, capsys):
    root = str(tmp_path / "sub")
    _write(root, "agent.py")
    _write(root, "weights.bin")
    _write(root, "__pycache__/agent.pyc")  # housekeeping is skipped, not refused
    client = fake(FakeClient())
    assert cli.main(["submit", "--challenge", "4", "--dir", root]) == 0
    assert client.calls[1][2]["file_names"] == ["agent.py", "weights.bin"]

    _write(root, "utils/helpers.py")
    client = fake(FakeClient())
    assert cli.main(["submit", "--challenge", "4", "--dir", root]) == 2
    err = capsys.readouterr().err
    assert "utils is a folder" in err and "flat list of files" in err
    # Refused before any request: nothing was flattened, nothing was sent.
    assert client.calls == [] and client.asked == []


def test_cli_submit_refuses_a_folder_among_the_files(fake, tmp_path, capsys):
    agent = _write(str(tmp_path), "agent.py")
    folder = tmp_path / "pkg"
    folder.mkdir()
    client = fake(FakeClient())
    assert cli.main(["submit", "--challenge", "4", "--files", agent, str(folder)]) == 2
    assert "is a folder" in capsys.readouterr().err
    assert client.calls == []


def test_cli_submit_usage_errors(fake, tmp_path, capsys):
    agent = _write(str(tmp_path), "agent.py")
    other = _write(str(tmp_path / "b"), "agent.py")
    client = fake(FakeClient())
    assert cli.main(["submit", "--challenge", "4"]) == 2
    assert cli.main(["submit", "--challenge", "4", "--files", agent,
                     "--dir", str(tmp_path)]) == 2
    assert cli.main(["submit", "--challenge", "4", "--files", agent, other]) == 2
    assert cli.main(["submit", "--challenge", "4", "--files", str(tmp_path / "no.py")]) == 2
    assert cli.main(["submit", "--challenge", "4", "--files", agent,
                     "--wait", "--no-deploy"]) == 2
    assert client.calls == []
    # argparse's own usage errors exit 2 as well.
    with pytest.raises(SystemExit) as raised:
        cli.main(["submit"])
    assert raised.value.code == 2
    capsys.readouterr()


def test_cli_submit_warns_about_requirements_txt(fake, tmp_path, capsys):
    agent = _write(str(tmp_path), "agent.py")
    _write(str(tmp_path), "requirements.txt")  # next to the files, not among them
    fake(FakeClient())
    assert cli.main(["submit", "--challenge", "4", "--files", agent]) == 0
    err = capsys.readouterr().err
    assert "requirements.txt is not installed at run time" in err
    assert "--runtime" in err


def test_cli_submit_file_challenge_single_file_is_the_submission_file(fake, tmp_path, capsys):
    predictions = _write(str(tmp_path), "predictions_v3.csv", b"id,y\n1,0\n")
    client = fake(FakeClient(kernel="file_v1"))
    assert cli.main(["submit", "--challenge", "4", "--files", predictions]) == 0
    kwargs = client.calls[1][2]
    # Uploaded under the name the challenge configured, with the same bytes.
    assert kwargs["file_names"] == ["submission.csv"]
    assert kwargs["file_bytes"] == [b"id,y\n1,0\n"]
    assert "as submission.csv" in capsys.readouterr().out

    other = _write(str(tmp_path), "extra.csv")
    client = fake(FakeClient(kernel="file_v1"))
    assert cli.main(["submit", "--challenge", "4", "--files", predictions, other]) == 2
    assert "exactly one file" in capsys.readouterr().err


def test_cli_submit_same_commit_is_reported_as_already_submitted(fake, tmp_path, capsys, actions):
    agent = _write(str(tmp_path), "agent.py")
    fake(FakeClient(submit_result={
        "submission_id": 77, "deploy": None, "already_submitted": True,
        "status": {"id": 77, "status": "active"}}))
    assert cli.main(["submit", "--challenge", "4", "--files", agent]) == 0
    out = capsys.readouterr().out
    assert "already submitted as #77" in out
    assert "/submission/view/4/77" in out
    assert "already submitted as [#77]" in actions.read_text()


def test_cli_submit_wait_prints_the_settled_status(fake, tmp_path, capsys):
    agent = _write(str(tmp_path), "agent.py")
    client = fake(FakeClient(submit_result={
        "submission_id": 11, "deploy": {"message": "queued"},
        "status": {"status": "active", "score": 212.5}}))
    assert cli.main(["submit", "--challenge", "4", "--files", agent, "--wait",
                     "--timeout", "900"]) == 0
    kwargs = client.calls[1][2]
    assert (kwargs["wait"], kwargs["timeout_sec"]) == (True, 900.0)
    # --wait reads the settled status from submit(): no extra request.
    assert [call[0] for call in client.calls] == ["challenge", "submit"]
    assert "deployed, status active, score 212.5" in capsys.readouterr().out


def test_cli_submit_failed_deploy_exits_1_with_the_failure_message(
        fake, tmp_path, capsys, actions):
    agent = _write(str(tmp_path), "agent.py")
    fake(FakeClient(error=SubmissionError(
        "ModuleNotFoundError: No module named 'torch' (failure_owner: participant)")))
    assert cli.main(["submit", "--challenge", "4", "--files", agent, "--wait"]) == 1
    captured = capsys.readouterr()
    assert "error: ModuleNotFoundError: No module named 'torch'" in captured.err
    # The annotation shows on the run page; the summary says it failed.
    assert "::error title=mlarena::ModuleNotFoundError" in captured.out
    assert "**Failed.** ModuleNotFoundError" in actions.read_text()


def test_cli_outside_actions_writes_no_summary_and_no_annotation(
        fake, tmp_path, capsys, monkeypatch):
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))  # GITHUB_ACTIONS unset
    agent = _write(str(tmp_path), "agent.py")
    fake(FakeClient(error=SubmissionError("refused")))
    assert cli.main(["submit", "--challenge", "4", "--files", agent]) == 1
    assert "::error" not in capsys.readouterr().out
    assert not summary.exists()


def test_cli_submit_step_summary(fake, tmp_path, actions):
    agent = _write(str(tmp_path), "agent.py")
    fake(FakeClient())
    assert cli.main(["submit", "--challenge", "4", "--files", agent]) == 0
    text = actions.read_text()
    assert "### ML Arena submission to Lunar Lander" in text
    assert "[#11](https://ml-arena.com/submission/view/4/11): deploy queued." in text


def test_cli_auth_failure_exits_1_with_the_server_message(monkeypatch, tmp_path, capsys, actions):
    agent = _write(str(tmp_path), "agent.py")

    def connect(base_url, scope, audience):
        raise PermissionDeniedError("No ML-Arena account is linked to GitHub user bob",
                                    status_code=403)
    monkeypatch.setattr(cli, "_connect", connect)
    assert cli.main(["submit", "--challenge", "4", "--files", agent]) == 1
    captured = capsys.readouterr()
    assert "error: No ML-Arena account is linked to GitHub user bob" in captured.err
    assert "::error title=mlarena::No ML-Arena account" in captured.out


def test_cli_challenge_push(fake, tmp_path, capsys, actions):
    client = fake(FakeClient())
    code = cli.main(["challenge", "push", str(tmp_path), "--challenge", "lunar-lander",
                     "--path", "env", "--dry-run"])
    assert code == 0
    assert client.asked[0]["scope"] == "creator"
    assert client.calls[1] == ("push_challenge", 4, str(tmp_path),
                               {"subpath": "env", "dry_run": True})
    out = capsys.readouterr().out
    assert "Bundle applied" in out
    assert "env files (2, changed): challenge.toml, env.py" in out
    assert "updated: overview" in out
    assert "config: 1 key(s) set by challenge.toml (simulation_timeout_sec)" in out
    assert "env.py: valid" in out
    assert "ignored: .github/workflows/push.yml" in out
    assert "https://ml-arena.com/challenges/lunar-lander" in out
    assert "### ML Arena challenge push to Lunar Lander" in actions.read_text()


def test_cli_challenge_push_defaults_to_the_current_directory(fake):
    client = fake(FakeClient())
    assert cli.main(["challenge", "push", "--challenge", "4"]) == 0
    assert client.calls[1] == ("push_challenge", 4, ".",
                               {"subpath": None, "dry_run": False})


def test_cli_challenge_push_refused_exits_1(fake, tmp_path, capsys):
    fake(FakeClient(error=MLArenaError(
        "push_challenge failed: challenge.toml:3: unknown key 'x'", status_code=400)))
    assert cli.main(["challenge", "push", str(tmp_path), "--challenge", "4"]) == 1
    assert "challenge.toml:3: unknown key 'x'" in capsys.readouterr().err


def test_cli_challenge_push_invalid_env_exits_1(fake, tmp_path, capsys):
    report = {**CHALLENGE_REPORT, "validation": {
        "ok": False,
        "errors": [{"line": 4, "col": 1, "msg": "class Env has no evaluate method"}],
        "warnings": [{"line": 9, "col": 5, "msg": "unused import"}]}}
    fake(FakeClient(challenge_report=report))
    assert cli.main(["challenge", "push", str(tmp_path), "--challenge", "4"]) == 1
    captured = capsys.readouterr()
    assert "env.py:4:1: class Env has no evaluate method" in captured.out
    assert "warning env.py:9:5: unused import" in captured.out
    assert "the bundle was applied" in captured.err


def test_cli_course_push_prints_the_plan(fake, tmp_path, capsys, actions):
    client = fake(FakeClient())
    assert cli.main(["course", "push", str(tmp_path), "--slug", "intro-to-rl",
                     "--dry-run", "--prune"]) == 0
    assert client.asked[0]["scope"] == "teacher"
    assert client.calls == [("push_course", str(tmp_path),
                             {"slug": "intro-to-rl", "subpath": None,
                              "dry_run": True, "prune": True})]
    out = capsys.readouterr().out
    assert "course intro-to-rl: updated" in out
    assert "  module foundations: adopted (renamed from basics)" in out
    assert "    lesson what-is-rl: unchanged" in out
    assert "    lesson bandits: created" in out
    assert "    lesson old-intro: orphaned" in out
    assert "    assets: 2 written, 1 removed" in out
    assert "  module extra: not in the manifest, untouched" in out
    assert "source: alice/agent@0123456" in out
    assert "https://ml-arena.com/courses/intro-to-rl" in out
    # The push's warnings: a line of the plan, an annotation on the run page
    # and a line of the job summary.
    warning = COURSE_PLAN["warnings"][0]
    assert f"warning: {warning}" in out
    assert f"::warning title=mlarena::{warning}" in out
    summary = actions.read_text()
    assert "### ML Arena course push" in summary
    assert f"warning: {warning}" in summary
    assert "[Open the course](https://ml-arena.com/courses/intro-to-rl)" in summary


def test_cli_course_push_dry_run_and_pruned_lines(fake, tmp_path, capsys):
    plan = {**COURSE_PLAN, "dry_run": True, "source_revision": None,
            "modules": [{**COURSE_PLAN["modules"][0], "orphaned_lessons": [],
                         "pruned_lessons": ["old-intro"]}]}
    client = fake(FakeClient(course_plan=plan))
    assert cli.main(["course", "push", "--path", "courses/rl"]) == 0
    assert client.calls == [("push_course", ".",
                             {"slug": None, "subpath": "courses/rl",
                              "dry_run": False, "prune": False})]
    out = capsys.readouterr().out
    assert "Dry run: nothing was changed" in out
    assert "    lesson old-intro: pruned (deleted)" in out


def test_cli_course_push_refused_exits_1(fake, tmp_path, capsys):
    fake(FakeClient(error=MLArenaError(
        "push_course failed: module 'm': lesson slug 'a' is listed twice",
        status_code=400)))
    assert cli.main(["course", "push", str(tmp_path)]) == 1
    assert "is listed twice" in capsys.readouterr().err


def test_cli_site_url_follows_the_audience(fake, tmp_path, capsys, monkeypatch):
    # Local stack: API on :4999, console (and OIDC audience) on :3000.
    monkeypatch.setenv("MLARENA_OIDC_AUDIENCE", "http://localhost:3000")
    agent = _write(str(tmp_path), "agent.py")
    fake(FakeClient())
    assert cli.main(["--base-url", "http://localhost:4999", "submit",
                     "--challenge", "4", "--files", agent]) == 0
    assert "http://localhost:3000/submission/view/4/11" in capsys.readouterr().out


def test_cli_whoami_refuses_a_credential_the_server_does_not_know(fake, capsys):
    """`current_user` answers 200 to any bearer; `is_authenticated` is the verdict."""
    client = fake(FakeClient())
    client.me = lambda: {"is_authenticated": False, "username": None}
    assert cli.main(["--base-url", "http://localhost:4999", "whoami"]) == 1
    captured = capsys.readouterr()
    assert captured.err.startswith(
        "error: the credential was not accepted by http://localhost:4999")
    assert "username" not in captured.out


def _outputs(path):
    return dict(line.split("=", 1) for line in path.read_text().splitlines())


@pytest.fixture
def github_output(monkeypatch, tmp_path):
    path = tmp_path / "output.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(path))
    return path


def test_cli_submit_outputs(fake, tmp_path, github_output):
    agent = _write(str(tmp_path), "agent.py")
    fake(FakeClient())
    assert cli.main(["submit", "--challenge", "4", "--files", agent]) == 0
    assert _outputs(github_output) == {
        "submission-id": "11", "url": "https://ml-arena.com/submission/view/4/11",
        "already-submitted": "false", "status": "deploy_queue"}


def test_cli_submit_outputs_with_wait_and_same_commit(fake, tmp_path, github_output):
    agent = _write(str(tmp_path), "agent.py")
    fake(FakeClient(submit_result={
        "submission_id": 11, "deploy": {"message": "queued"},
        "status": {"status": "active", "score": 212.5}}))
    assert cli.main(["submit", "--challenge", "4", "--files", agent, "--wait"]) == 0
    assert _outputs(github_output) == {
        "submission-id": "11", "url": "https://ml-arena.com/submission/view/4/11",
        "already-submitted": "false", "status": "active", "score": "212.5"}

    github_output.unlink()
    fake(FakeClient(submit_result={
        "submission_id": 77, "deploy": None, "already_submitted": True,
        "status": {"id": 77, "status": "active"}}))
    assert cli.main(["submit", "--challenge", "4", "--files", agent]) == 0
    assert _outputs(github_output) == {
        "submission-id": "77", "url": "https://ml-arena.com/submission/view/4/77",
        "already-submitted": "true", "status": "active"}


def test_cli_push_outputs(fake, tmp_path, github_output):
    fake(FakeClient())
    assert cli.main(["challenge", "push", str(tmp_path), "--challenge", "4"]) == 0
    assert _outputs(github_output) == {
        "url": "https://ml-arena.com/challenges/lunar-lander", "env-changed": "true"}

    github_output.unlink()
    fake(FakeClient())
    assert cli.main(["course", "push", str(tmp_path), "--slug", "intro-to-rl"]) == 0
    assert _outputs(github_output) == {
        "url": "https://ml-arena.com/courses/intro-to-rl", "course-slug": "intro-to-rl"}


def test_cli_outputs_are_single_line_and_skipped_on_failure(
        fake, tmp_path, github_output, monkeypatch):
    cli._set_outputs({"status": "two\nlines\r\nhere", "flag": True})
    assert github_output.read_text() == "status=two lines here\nflag=true\n"
    github_output.unlink()

    agent = _write(str(tmp_path), "agent.py")
    fake(FakeClient(error=SubmissionError("refused")))
    assert cli.main(["submit", "--challenge", "4", "--files", agent]) == 1
    assert not github_output.exists()
    # Without GITHUB_OUTPUT nothing is written anywhere.
    monkeypatch.delenv("GITHUB_OUTPUT")
    cli._set_outputs({"a": 1})


def test_cli_submit_wait_has_a_default_timeout(fake, tmp_path):
    agent = _write(str(tmp_path), "agent.py")
    client = fake(FakeClient(submit_result={
        "submission_id": 11, "deploy": {}, "status": {"status": "active", "score": None}}))
    assert cli.main(["submit", "--challenge", "4", "--files", agent, "--wait"]) == 0
    assert client.calls[1][2]["timeout_sec"] == 1800 == cli.DEFAULT_WAIT_TIMEOUT_SEC
    # Without --wait there is nothing to time out.
    client = fake(FakeClient())
    assert cli.main(["submit", "--challenge", "4", "--files", agent]) == 0
    assert client.calls[1][2]["timeout_sec"] is None


def test_cli_submit_wait_timeout_exits_1_with_a_clear_message(
        fake, tmp_path, capsys, actions):
    agent = _write(str(tmp_path), "agent.py")
    fake(FakeClient(error=WaitTimeoutError(
        "tail_logs timed out after 60s", body={"id": 11, "status": "deploy_run"})))
    assert cli.main(["submit", "--challenge", "4", "--files", agent, "--wait",
                     "--timeout", "60"]) == 1
    captured = capsys.readouterr()
    assert ("error: submission #11 did not settle within 60 s (status: deploy_run). "
            "It is still running: https://ml-arena.com/submission/view/4/11") in captured.err
    assert "Raise --timeout" in captured.err
    assert "::error title=mlarena::submission #11 did not settle" in captured.out


def test_tail_logs_timeout_is_a_wait_timeout_error():
    status = {"id": 11, "challenge_id": 4, "status": "deploy_run", "is_settled": False,
              "last_status_message": None, "latest_deploy": None, "queue_info": None}
    c, _ = make_client(lambda *_: (200, status))
    with pytest.raises(WaitTimeoutError) as raised:
        for _ in c.tail_logs(4, 11, poll_sec=0, timeout_sec=0.01):
            pass
    assert isinstance(raised.value, SubmissionError)
    assert raised.value.body["id"] == 11


def test_pyyaml_is_a_declared_dependency():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    with open(os.path.join(root, "pyproject.toml"), encoding="utf-8") as fh:
        assert '"pyyaml>=6.0"' in fh.read()


def test_console_script_is_declared():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    with open(os.path.join(root, "pyproject.toml"), encoding="utf-8") as fh:
        assert 'mlarena = "mlarena.cli:main"' in fh.read()
    assert callable(cli.main)
