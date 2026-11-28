"""Offline tests for a course edited in the console or on GitHub (4.26.0):
the Settings › GitHub methods, the course key (`mlc_…`) as a credential, the
`X-MLArena-CI-*` headers of a course push, `CourseEditedOnGithubError`, and
`mlarena course push` / `course status` with a course key.

Same harness as the other suites: `MLArenaClient._request` is replaced by a
recorder (on the class for the CLI, which builds its own client), and
`ci_auth._http` (the OIDC exchange) by one that fails the test if called.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mlarena
from mlarena import ci_auth, cli
from mlarena.client import MLArenaClient
from mlarena.exceptions import (
    AuthenticationError,
    CourseEditedOnGithubError,
    MLArenaError,
)

API_KEY = "mlk_author_abcd1234_deadbeef0000"
COURSE_KEY = "mlc_0123456789abcdef_0123456789abcdef0123456789abcdef"
BASE = "https://example.com"

GITHUB_SENTENCE = ("This course is edited on GitHub (racousin/python-medecine). Edit it "
                   "there, or switch to console editing in Settings › GitHub.")
GITHUB_409 = {"error": GITHUB_SENTENCE, "edited_in": "github",
              "repository": "racousin/python-medecine"}

WHOAMI = {"course_id": 18, "slug": "python-medecine", "edited_in": "connecting",
          "expires_at_ts": "2027-11-27T10:00:00Z"}

PLAN = {
    "dry_run": True,
    "course": {"id": 18, "slug": "python-medecine", "action": "unchanged", "changes": []},
    "modules": [{
        "slug": "intro", "id": 1, "action": "unchanged", "renamed_from": None,
        "changes": [], "lessons": [], "challenges": [
            {"name": "House prices", "id": 40, "action": "unchanged", "changes": []}],
        "orphaned_lessons": [], "pruned_lessons": [],
        "assets": {"written_paths": [], "removed_paths": []},
    }],
    "unmanaged_modules": [], "unmanaged_challenges": [], "source_revision": None,
    "unpublished_modules": ["s9-old"], "removed_challenges": ["Old quiz"],
    "unplaced_challenges": [{"id": 9, "name": "Old quiz", "slug": "old-quiz", "status": "started"}],
    "url": "/courses/python-medecine", "warnings": [], "run_id": 7,
}

GITHUB_OUT = {
    "edited_in": "github",
    "repository": {"repository": "racousin/python-medecine", "branch": "main",
                   "path": "courses/med", "connected_by_user_id": 7,
                   "connected_at_ts": "2026-11-27T10:00:00Z"},
    "keys": [
        {"id": 3, "label": "GitHub Actions", "created_by_user_id": 7,
         "created_by_username": "yann", "created_at_ts": "2026-11-27T10:00:00Z",
         "expires_at_ts": "2027-11-27T10:00:00Z", "last_used_at_ts": None,
         "revoked_at_ts": None, "revoked_reason": None},
        {"id": 1, "label": "Laptop", "created_by_user_id": 8,
         "created_by_username": "olivier", "created_at_ts": "2026-11-20T10:00:00Z",
         "expires_at_ts": "2026-12-20T10:00:00Z", "last_used_at_ts": None,
         "revoked_at_ts": "2026-11-25T10:00:00Z", "revoked_reason": "staff_left"},
    ],
    "runs": [
        {"id": 12, "course_key_id": 3, "repository": "racousin/python-medecine",
         "commit_sha": "abc1234def5678", "ref": "refs/heads/main", "run_url": None,
         "actor_login": "racousin", "is_dry_run": False, "outcome": "applied",
         "message": None, "plan": {}, "created_at_ts": "2026-11-27T11:00:00Z"},
        {"id": 11, "course_key_id": 1, "repository": None, "commit_sha": None,
         "ref": None, "run_url": None, "actor_login": None, "is_dry_run": True,
         "outcome": "refused", "message": "this key was revoked: staff_left",
         "plan": None, "created_at_ts": "2026-11-26T11:00:00Z"},
    ],
    "changes": [],
}

ACTIONS_RUN = {
    "GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": "racousin/python-medecine",
    "GITHUB_SHA": "abc1234def5678abc1234def5678abc1234def56", "GITHUB_REF": "refs/heads/main",
    "GITHUB_ACTOR": "racousin", "GITHUB_SERVER_URL": "https://github.com",
    "GITHUB_RUN_ID": "42", "GITHUB_RUN_ATTEMPT": "2",
}

_ENV = (
    "MLARENA_API_KEY", "MLARENA_COURSE_KEY", "MLARENA_BASE_URL", "MLARENA_OIDC_AUDIENCE",
    "ACTIONS_ID_TOKEN_REQUEST_URL", "ACTIONS_ID_TOKEN_REQUEST_TOKEN",
    "GITHUB_STEP_SUMMARY", "GITHUB_OUTPUT", *ACTIONS_RUN,
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)
    # The OIDC exchange is never what a course command or a key uses.
    def no_exchange(*args, **kwargs):
        raise AssertionError(f"unexpected OIDC call {args}")
    monkeypatch.setattr(ci_auth, "_http", no_exchange)


class FakeResponse:
    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json = json_data
        self.text = ""
        self.content = b""

    def json(self):
        if self._json is None:
            raise ValueError("no JSON body")
        return self._json


class Recorder:
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


def make_client(router=None, **connect):
    connect = connect or {"api_key": API_KEY}
    client = mlarena.connect(base_url=BASE, **connect)
    rec = Recorder(router)
    client._request = rec
    return client, rec


# --------------------------------------------------------------------------- #
# Settings › GitHub methods: one route each
# --------------------------------------------------------------------------- #


def test_course_github_reads_the_mode_repository_keys_and_runs():
    client, rec = make_client(lambda *_: (200, GITHUB_OUT))
    assert client.course_github(18) == GITHUB_OUT
    assert (rec.last["method"], rec.last["path"]) == ("GET", "/teacher/course/18/github")


def test_set_course_repository_sends_the_three_keys():
    client, rec = make_client(lambda *_: (200, GITHUB_OUT))
    client.set_course_repository(18, "racousin/python-medecine")
    assert (rec.last["method"], rec.last["path"]) == (
        "PUT", "/teacher/course/18/github/repository")
    assert rec.last["json"] == {"repository": "racousin/python-medecine",
                                "branch": "main", "path": ""}
    client.set_course_repository(18, "racousin/ms2a-courses", "dev", "ai-engineering")
    assert rec.last["json"] == {"repository": "racousin/ms2a-courses",
                                "branch": "dev", "path": "ai-engineering"}


def test_switch_course_edited_in():
    client, rec = make_client(lambda *_: (200, GITHUB_OUT))
    client.switch_course_edited_in(18, "github")
    assert (rec.last["method"], rec.last["path"], rec.last["json"]) == (
        "POST", "/teacher/course/18/github/switch", {"to": "github"})
    client.switch_course_edited_in(18, to="console")
    assert rec.last["json"] == {"to": "console"}


def test_create_course_key_answers_the_token_once():
    created = {"key": GITHUB_OUT["keys"][0], "token": COURSE_KEY}
    client, rec = make_client(lambda *_: (201, created))
    assert client.create_course_key(18, "GitHub Actions") == created
    assert (rec.last["method"], rec.last["path"]) == ("POST", "/teacher/course/18/keys")
    assert rec.last["json"] == {"label": "GitHub Actions", "expires_in_days": 365}
    client.create_course_key(18, "Laptop", expires_in_days=30)
    assert rec.last["json"] == {"label": "Laptop", "expires_in_days": 30}


def test_revoke_course_key():
    client, rec = make_client(lambda *_: (200, GITHUB_OUT["keys"][1]))
    client.revoke_course_key(18, 1)
    assert (rec.last["method"], rec.last["path"]) == ("DELETE", "/teacher/course/18/keys/1")


def test_course_repository_runs_pages_with_before_id():
    client, rec = make_client(lambda *_: (200, GITHUB_OUT["runs"]))
    assert client.course_repository_runs(18) == GITHUB_OUT["runs"]
    assert (rec.last["path"], rec.last["params"]) == ("/teacher/course/18/github/runs", None)
    client.course_repository_runs(18, before_id=11)
    assert rec.last["params"] == {"before_id": 11}


def test_course_key_whoami():
    client, rec = make_client(lambda *_: (200, WHOAMI), course_key=COURSE_KEY)
    assert client.course_key_whoami() == WHOAMI
    assert (rec.last["method"], rec.last["url"]) == ("GET", f"{BASE}/api/auth/course_key")
    assert rec.last["headers"]["Authorization"] == f"Bearer {COURSE_KEY}"


def test_detach_module_is_gone():
    assert not hasattr(MLArenaClient, "detach_module")


# --------------------------------------------------------------------------- #
# The 409 of a course edited on GitHub
# --------------------------------------------------------------------------- #


def test_a_write_to_a_course_edited_on_github_raises_its_own_error(tmp_path):
    (tmp_path / "course.yaml").write_text("course:\n  name: Med\nmodules: []\n")
    client, _ = make_client(lambda *_: (409, GITHUB_409))
    for call in (lambda: client.update_module(2, title="New"),
                 lambda: client.push_course(str(tmp_path), slug="python-medecine"),
                 lambda: client.update_env_file_content(40, "env.py", "x = 1\n")):
        with pytest.raises(CourseEditedOnGithubError) as raised:
            call()
        assert str(raised.value) == GITHUB_SENTENCE
        assert raised.value.status_code == 409
        assert raised.value.repository == "racousin/python-medecine"
        assert isinstance(raised.value, MLArenaError)
    assert mlarena.CourseEditedOnGithubError is CourseEditedOnGithubError


def test_another_409_is_not_the_github_error():
    client, _ = make_client(lambda *_: (409, {"error": "a module with this slug exists"}))
    with pytest.raises(MLArenaError) as raised:
        client.update_module(2, slug="taken")
    assert not isinstance(raised.value, CourseEditedOnGithubError)
    # A course key's push to a course back in the console: refused, not this class.
    client, _ = make_client(lambda *_: (409, {"error": "edited in the console",
                                              "edited_in": "console"}))
    with pytest.raises(MLArenaError) as raised:
        client.update_module(2, title="x")
    assert not isinstance(raised.value, CourseEditedOnGithubError)


# --------------------------------------------------------------------------- #
# The course key as a credential
# --------------------------------------------------------------------------- #


def test_a_course_key_is_a_credential_of_its_own_kind(monkeypatch):
    client = mlarena.connect(course_key=COURSE_KEY, base_url=BASE)
    assert (client.credentials.kind, client.credentials.scope) == ("course_key", "course")
    assert client._token == COURSE_KEY
    assert "0123456789abcdef0123456789abcdef'" not in repr(client)

    # Passed as api_key= it is read the same way.
    assert mlarena.connect(COURSE_KEY, base_url=BASE).credentials.kind == "course_key"

    monkeypatch.setenv("MLARENA_COURSE_KEY", COURSE_KEY)
    client = mlarena.connect(base_url=BASE)
    assert client.credentials.kind == "course_key"
    assert not client.credentials.expires_within(10 ** 9)


def test_both_keys_at_once_are_refused(monkeypatch):
    with pytest.raises(AuthenticationError, match="not both"):
        mlarena.connect(API_KEY, course_key=COURSE_KEY, base_url=BASE)
    monkeypatch.setenv("MLARENA_API_KEY", API_KEY)
    monkeypatch.setenv("MLARENA_COURSE_KEY", COURSE_KEY)
    with pytest.raises(AuthenticationError, match="both MLARENA_API_KEY and MLARENA_COURSE_KEY"):
        mlarena.connect(base_url=BASE)
    # An explicit argument still wins over the environment.
    assert mlarena.connect(API_KEY, base_url=BASE).credentials.kind == "api_key"


def test_a_course_key_variable_that_is_empty_or_not_a_course_key(monkeypatch):
    # A repository secret that does not exist reaches the job as "".
    monkeypatch.setenv("MLARENA_COURSE_KEY", "")
    with pytest.raises(AuthenticationError, match="set but empty.*repository secret"):
        mlarena.connect(base_url=BASE)
    monkeypatch.setenv("MLARENA_COURSE_KEY", API_KEY)
    with pytest.raises(AuthenticationError, match="not a course key"):
        mlarena.connect(base_url=BASE)
    monkeypatch.setenv("MLARENA_COURSE_KEY", "mlc_only")
    with pytest.raises(AuthenticationError, match="course key shape"):
        mlarena.connect(base_url=BASE)


def test_without_oidc_no_exchange_is_tried(monkeypatch):
    monkeypatch.setenv("ACTIONS_ID_TOKEN_REQUEST_URL", "https://runner.example/token")
    monkeypatch.setenv("ACTIONS_ID_TOKEN_REQUEST_TOKEN", "runner-secret")
    with pytest.raises(AuthenticationError, match="MLARENA_COURSE_KEY"):
        ci_auth.resolve_credentials(base_url=BASE, oidc=False)


# --------------------------------------------------------------------------- #
# The run a course push comes from
# --------------------------------------------------------------------------- #


def test_ci_context_headers_from_the_actions_variables():
    assert ci_auth.ci_context_headers({}) == {}
    # Outside GitHub Actions the variables are not read.
    assert ci_auth.ci_context_headers({**ACTIONS_RUN, "GITHUB_ACTIONS": "false"}) == {}
    assert ci_auth.ci_context_headers(ACTIONS_RUN) == {
        "X-MLArena-CI-Repository": "racousin/python-medecine",
        "X-MLArena-CI-Commit": ACTIONS_RUN["GITHUB_SHA"],
        "X-MLArena-CI-Ref": "refs/heads/main",
        "X-MLArena-CI-Actor": "racousin",
        "X-MLArena-CI-Run-Url":
            "https://github.com/racousin/python-medecine/actions/runs/42/attempts/2",
    }
    # A missing variable sends no header; the URL needs its four parts.
    partial = {key: value for key, value in ACTIONS_RUN.items() if key != "GITHUB_RUN_ID"}
    assert "X-MLArena-CI-Run-Url" not in ci_auth.ci_context_headers(partial)


def test_push_course_with_a_course_key_asks_its_course_and_sends_the_run(
        tmp_path, monkeypatch):
    for name, value in ACTIONS_RUN.items():
        monkeypatch.setenv(name, value)
    # The template's course.yaml names no slug: the key names the course.
    (tmp_path / "course.yaml").write_text("course:\n  name: Med\nmodules: []\n")

    def router(method, url, kwargs):
        if url.endswith("/api/auth/course_key"):
            return (200, WHOAMI)
        return (200, PLAN)
    client, rec = make_client(router, course_key=COURSE_KEY)

    assert client.push_course(str(tmp_path)) == PLAN

    assert [(call["method"], call["path"]) for call in rec.calls] == [
        ("GET", "/auth/course_key"), ("PUT", "/teacher/courses/python-medecine/bundle")]
    headers = rec.last["headers"]
    assert headers["Authorization"] == f"Bearer {COURSE_KEY}"
    assert headers["X-MLArena-CI-Commit"] == ACTIONS_RUN["GITHUB_SHA"]
    assert headers["X-MLArena-CI-Run-Url"].endswith("/actions/runs/42/attempts/2")


def test_push_course_with_a_personal_key_reads_the_manifest_slug(tmp_path):
    (tmp_path / "course.yaml").write_text("course:\n  name: Med\n  slug: med\nmodules: []\n")
    client, rec = make_client(lambda *_: (200, PLAN))
    client.push_course(str(tmp_path))
    assert [call["path"] for call in rec.calls] == ["/teacher/courses/med/bundle"]
    # Outside GitHub Actions: no run to report.
    assert not any(name.startswith("X-MLArena-CI-") for name in rec.last["headers"])


# --------------------------------------------------------------------------- #
# The CLI with a course key
# --------------------------------------------------------------------------- #


@pytest.fixture
def server(monkeypatch):
    """Every client the CLI builds talks to this recorder."""
    def install(router):
        rec = Recorder(router)
        monkeypatch.setattr(MLArenaClient, "_request",
                            lambda self, method, url, **kwargs: rec(method, url, **kwargs))
        return rec
    return install


@pytest.fixture
def actions_run(monkeypatch, tmp_path):
    for name, value in ACTIONS_RUN.items():
        monkeypatch.setenv(name, value)
    # An OIDC-capable job: the course push must still not use it.
    monkeypatch.setenv("ACTIONS_ID_TOKEN_REQUEST_URL", "https://runner.example/token")
    monkeypatch.setenv("ACTIONS_ID_TOKEN_REQUEST_TOKEN", "runner-secret")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "summary.md"))
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "output.txt"))
    return tmp_path


def _course_router(whoami=WHOAMI, plan=PLAN):
    def router(method, url, kwargs):
        if url.endswith("/api/auth/course_key"):
            return (200, whoami)
        if "/bundle" in url:
            return (200, plan)
        return (404, {"error": "no such route"})
    return router


def test_cli_course_push_with_a_course_key_pushes_its_course(
        server, actions_run, monkeypatch, capsys):
    monkeypatch.setenv("MLARENA_COURSE_KEY", COURSE_KEY)
    course = actions_run / "repo"
    course.mkdir()
    (course / "course.yaml").write_text("course:\n  name: Med\nmodules: []\n")
    rec = server(_course_router())

    assert cli.main(["course", "push", str(course)]) == 0

    assert [(call["method"], call["path"]) for call in rec.calls] == [
        ("GET", "/auth/course_key"), ("PUT", "/teacher/courses/python-medecine/bundle")]
    push = rec.last
    assert push["headers"]["Authorization"] == f"Bearer {COURSE_KEY}"
    assert push["headers"]["X-MLArena-CI-Repository"] == "racousin/python-medecine"
    assert push["headers"]["X-MLArena-CI-Ref"] == "refs/heads/main"
    assert push["headers"]["X-MLArena-CI-Actor"] == "racousin"
    assert push["params"] is None          # a push to main: not a dry run
    out = capsys.readouterr().out
    assert "Course python-medecine (edited in: connecting), with its course key" in out
    # The course is connecting: the server answered a plan the run did not ask for.
    assert ("planned (connecting): nothing was changed. Switch to GitHub editing in "
            "Settings › GitHub; the next push then applies") in out
    assert "    challenge House prices: unchanged" in out
    # What the repository no longer lists (backend/PROCESS.md "Course on GitHub").
    assert "  module s9-old: not in the repository, unpublished (its lessons kept)" in out
    assert ("  challenge Old quiz: not in the repository, taken out of its module "
            "(its submissions kept)") in out
    assert ("  challenge Old quiz: in no module (started); stop, archive or delete it in "
            "the console, or list it again") in out
    summary = (actions_run / "summary.md").read_text()
    assert "planned (connecting)" in summary
    outputs = (actions_run / "output.txt").read_text()
    assert "course-slug=python-medecine" in outputs


def test_cli_course_push_with_a_course_key_refuses_another_slug(
        server, monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("MLARENA_COURSE_KEY", COURSE_KEY)
    rec = server(_course_router())
    assert cli.main(["course", "push", str(tmp_path), "--slug", "other-course"]) == 2
    assert "belongs to the course python-medecine" in capsys.readouterr().err
    assert [call["path"] for call in rec.calls] == ["/auth/course_key"]


def test_cli_course_push_without_a_key_never_tries_oidc(server, actions_run, capsys):
    rec = server(_course_router())
    assert cli.main(["course", "push", str(actions_run)]) == 1
    err = capsys.readouterr().err
    assert "No ML Arena key" in err and "MLARENA_COURSE_KEY" in err
    assert rec.calls == []


def test_cli_course_push_with_a_personal_key_on_a_github_course(
        server, monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("MLARENA_API_KEY", API_KEY)
    (tmp_path / "course.yaml").write_text("course:\n  name: Med\n  slug: med\nmodules: []\n")
    rec = server(lambda *_: (409, GITHUB_409))
    assert cli.main(["course", "push", str(tmp_path)]) == 1
    assert GITHUB_SENTENCE in capsys.readouterr().err
    # A personal key asks no course: the manifest's slug.
    assert [call["path"] for call in rec.calls] == ["/teacher/courses/med/bundle"]


def test_cli_course_status_with_a_course_key(server, monkeypatch, capsys):
    monkeypatch.setenv("MLARENA_COURSE_KEY", COURSE_KEY)
    server(_course_router())
    assert cli.main(["course", "status"]) == 0
    out = capsys.readouterr().out
    assert "course:    python-medecine (#18)" in out
    assert "edited in: connecting" in out
    assert "expires 2027-11-27T10:00:00Z" in out
    assert "mlarena course status --course python-medecine" in out


def test_cli_course_status_with_a_personal_key(server, monkeypatch, capsys):
    monkeypatch.setenv("MLARENA_API_KEY", API_KEY)

    def router(method, url, kwargs):
        if url.endswith("/api/academic_courses/python-medecine"):
            return (200, {"id": 18, "slug": "python-medecine"})
        if url.endswith("/api/teacher/course/18/github"):
            return (200, GITHUB_OUT)
        return (404, {"error": "no such route"})
    server(router)

    assert cli.main(["course", "status", "--course", "python-medecine"]) == 0
    out = capsys.readouterr().out
    assert "edited in:  github" in out
    assert "repository: racousin/python-medecine (branch main, folder courses/med)" in out
    assert ("  #3 GitHub Actions: by yann, created 2026-11-27T10:00:00Z, "
            "expires 2027-11-27T10:00:00Z, never used") in out
    assert "  #1 Laptop: by olivier, created 2026-11-20T10:00:00Z, revoked " \
           "2026-11-25T10:00:00Z (staff_left)" in out
    assert ("  #12 2026-11-27T11:00:00Z applied racousin/python-medecine@abc1234 "
            "by @racousin") in out
    assert "  #11 2026-11-26T11:00:00Z refused (dry run) ?: this key was revoked" in out

    assert cli.main(["course", "status"]) == 2
    assert "pass --course" in capsys.readouterr().err
