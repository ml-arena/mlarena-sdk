"""SDK tests for the teacher administration methods (roster, assistants,
exports, course cover) — the routes the console had and the SDK did not.

Same offline harness as ``test_course_content.py``: the client's single HTTP
chokepoint (``MLArenaClient._request``) is replaced by a recorder, so each test
asserts the method → (verb, path, body, params) mapping the parity rule
requires without a network or a backend.

Runs under pytest *or* directly: ``python tests/test_teacher_admin.py``.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import warnings

import mlarena
from mlarena.exceptions import MLArenaError


class FakeResponse:
    def __init__(self, status_code=200, json_data=None, text="", content=b""):
        self.status_code = status_code
        self._json = {} if json_data is None else json_data
        self.text = text
        self._content = content

    def json(self):
        return self._json

    def iter_content(self, chunk_size=1):
        yield self._content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise AssertionError(f"unexpected raise_for_status {self.status_code}")


class Recorder:
    def __init__(self, router=None):
        self.calls = []
        self.router = router

    def __call__(self, method, url, **kwargs):
        path = url.split("/api", 1)[1]
        self.calls.append({"method": method, "path": path, "url": url, **kwargs})
        if self.router is not None:
            result = self.router(method, path, kwargs)
            if result is not None:
                if isinstance(result, FakeResponse):
                    return result
                status, data = result
                return FakeResponse(status, data)
        return FakeResponse(200, {"ok": True})

    @property
    def last(self):
        return self.calls[-1]


def make_client(router=None, scope="teacher"):
    client = mlarena.connect(
        api_key=f"mlk_{scope}_abcd1234_deadbeef0000",
        base_url="https://example.com",
    )
    rec = Recorder(router)
    client._request = rec
    return client, rec


def _expect(cond, msg):
    if not cond:
        raise AssertionError(msg)


# --------------------------------------------------------------------------- #
# Course list + challenge picker
# --------------------------------------------------------------------------- #


def _workspace(challenges=(), courses=(), attention=()):
    """A `GET /author/workspace` body."""
    return {
        "scope": "mine", "challenges": list(challenges), "courses": list(courses),
        "attention": list(attention),
        "quotas": {"owned_challenge_count": len(challenges), "max_courses_limit": 3,
                   "active_course_count": len(courses), "can_create_course": True},
    }


def _workspace_course(course_id, name, role, **over):
    row = {
        "id": course_id, "name": name, "code": None, "slug": name.lower(),
        "visibility": "private", "start_date": None, "end_date": None,
        "archived_at_ts": None, "join_code": f"JOIN{course_id:04d}", "role": role,
        "owner_username": "ada", "enrollment_count": 12, "active_enrollment_count_7d": 4,
        "last_active_at_ts": "2026-11-20T09:00:00Z", "challenges": [],
        "hidden_challenge_ids": [], "enrolled_without_submission": None,
    }
    row.update(over)
    return row


def _workspace_challenge(challenge_id, name, role="owner", status="started"):
    return {
        "challenge": {
            "id": challenge_id, "name": name, "slug": name.lower(), "status": status,
            "end_date_ts": None, "is_public": False, "kernel_version": "file_v1",
            "kind_label": "Upload a file", "user_id": 7, "owner_username": "ada",
            "role": role, "statistics": {"running_submissions": 2, "unique_participants": 2},
        },
        "counters": {"team_count": 0, "submissions_created_7d": 3, "submissions_created_daily": [],
                     "deploy_failed_submissions": 1,
                     "deploy_failed_submissions_by_owner": {
                         "participant": 1, "challenge": 0, "platform": 0, "unattributed": 0},
                     "last_submission_created_at_ts": None, "course_count": 1},
        "created_at_ts": "2026-11-01T10:00:00Z",
        "start_blockers": None, "start_blockers_error": None,
    }


def test_workspace_is_one_read_of_the_author_route():
    body = _workspace(
        challenges=[_workspace_challenge(4, "Churn")],
        courses=[_workspace_course(1, "RL", "owner")],
        attention=[{"kind": "deploy_failed_submissions", "challenge_id": 4,
                    "challenge_name": "Churn", "deploy_failed_submissions": 1,
                    "deploy_failed_submissions_by_owner": {
                        "participant": 1, "challenge": 0, "platform": 0, "unattributed": 0}}],
    )
    for scope in ("creator", "teacher"):
        c, rec = make_client(lambda *_: (200, body), scope=scope)
        out = c.workspace()
        _expect((rec.last["method"], rec.last["path"]) == ("GET", "/author/workspace"), rec.last)
        # No query string: the scope is the server's default, `mine`
        # (everyone's is an admin's session in the console, never a key).
        _expect(not rec.last.get("params"), rec.last.get("params"))
        _expect(len(rec.calls) == 1, rec.calls)
        _expect(out == body, out)


def test_workspace_takes_no_scope():
    """`scope="all"` is session-only on the platform: the SDK has no way to
    ask for it, rather than a parameter every key is refused."""
    c, rec = make_client(lambda *_: (200, _workspace()), scope="creator")
    try:
        c.workspace(scope="all")
    except TypeError:
        pass
    else:
        raise AssertionError("workspace() takes no scope")
    _expect(rec.calls == [], rec.calls)


def test_workspace_refuses_a_user_key_with_the_servers_sentence():
    body = {"error": "Key scope 'user' cannot access this route: it takes a 'creator' or 'teacher' key"}
    c, _rec = make_client(lambda *_: (403, body), scope="user")
    try:
        c.workspace()
    except mlarena.PermissionDeniedError as e:
        _expect("'creator' or 'teacher' key" in str(e), str(e))
    else:
        raise AssertionError("expected PermissionDeniedError")


def test_teacher_courses_is_a_deprecated_composition_of_the_workspace():
    body = _workspace(courses=[
        _workspace_course(1, "RL", "owner"),
        _workspace_course(2, "NLP", "assistant", archived_at_ts="2026-07-01T00:00:00Z"),
    ])
    c, rec = make_client(lambda *_: (200, body))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = c.teacher_courses()
    _expect([w.category for w in caught] == [DeprecationWarning], caught)
    _expect("workspace()" in str(caught[0].message), caught[0].message)
    _expect((rec.last["method"], rec.last["path"]) == ("GET", "/author/workspace"), rec.last)
    rows = out.to_dict("records") if hasattr(out, "columns") else out
    # The old list's columns that the workspace serves, nothing else.
    _expect(set(rows[0]) == {"id", "name", "code", "slug", "visibility", "join_code",
                             "start_date", "end_date", "archived_at_ts", "role"}, rows[0])
    _expect([r["role"] for r in rows] == ["owner", "assistant"], rows)
    _expect(rows[1]["archived_at_ts"] == "2026-07-01T00:00:00Z", rows[1])
    _expect(rows[0]["join_code"] == "JOIN0001", rows[0])


def test_creator_challenges_is_a_deprecated_composition_of_the_workspace():
    body = _workspace(challenges=[
        _workspace_challenge(4, "Churn"), _workspace_challenge(9, "Kart", role="assistant"),
    ])
    c, rec = make_client(lambda *_: (200, body), scope="creator")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = c.creator_challenges()
    _expect([w.category for w in caught] == [DeprecationWarning], caught)
    _expect((rec.last["method"], rec.last["path"]) == ("GET", "/author/workspace"), rec.last)
    _expect([(r["id"], r["name"], r["role"], r["status"]) for r in out] == [
        (4, "Churn", "owner", "started"), (9, "Kart", "assistant", "started")], out)
    # The reference, flat, plus the row's creation time: the old list's shape
    # without the columns the route no longer serves.
    _expect(set(out[0]) == set(body["challenges"][0]["challenge"]) | {"created_at_ts"}, out[0])
    _expect(out[0]["created_at_ts"] == "2026-11-01T10:00:00Z", out[0])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        c.creator_competitions()
    _expect(all(w.category is DeprecationWarning for w in caught) and caught, caught)


# --------------------------------------------------------------------------- #
# One course: workspace read, settings, archive / duplicate / delete, report
# --------------------------------------------------------------------------- #


def test_teacher_course_reads_one_course():
    row = {"id": 7, "name": "RL", "role": "assistant", "enrollment_count": 12,
           "report_recipients": ["dean@example.com"], "archived_at_ts": None}
    c, rec = make_client(lambda *_: (200, row))
    out = c.teacher_course(7)
    _expect(out == row, out)
    _expect((rec.last["method"], rec.last["path"]) == ("GET", "/teacher/course/7"), rec.last)
    _expect(len(rec.calls) == 1, rec.calls)


def test_update_course_sends_the_settings_keys():
    c, rec = make_client(lambda *_: (200, {"id": 7}))
    c.update_course(7, ask_student_identity=False, report_to_assistants=False,
                    report_recipients=["dean@example.com"])
    _expect((rec.last["method"], rec.last["path"]) == ("PUT", "/teacher/course/7"), rec.last)
    _expect(rec.last["json"] == {"ask_student_identity": False,
                                 "report_to_assistants": False,
                                 "report_recipients": ["dean@example.com"]}, rec.last["json"])


def test_archive_and_unarchive_are_one_post_each():
    c, rec = make_client(lambda *_: (200, {"id": 7, "archived_at_ts": "2026-07-01T00:00:00Z"}))
    out = c.archive_course(7)
    _expect(out["archived_at_ts"] is not None, out)
    _expect((rec.last["method"], rec.last["path"]) == ("POST", "/teacher/course/7/archive"),
            rec.last)
    c.unarchive_course(7)
    _expect((rec.last["method"], rec.last["path"]) == ("POST", "/teacher/course/7/unarchive"),
            rec.last)


def test_duplicate_course_sends_the_name_only_when_given():
    c, rec = make_client(lambda *_: (201, {
        "id": 8, "name": "RL (copy)", "left_out_modules": [{"id": 3, "title": "TA notes"}]}))
    out = c.duplicate_course(7)
    _expect(out["id"] == 8, out)
    _expect(out["left_out_modules"] == [{"id": 3, "title": "TA notes"}], out)
    _expect((rec.last["method"], rec.last["path"]) == ("POST", "/teacher/course/7/duplicate"),
            rec.last)
    _expect(rec.last["json"] == {}, rec.last["json"])
    c.duplicate_course(7, name="RL 2027")
    _expect(rec.last["json"] == {"name": "RL 2027"}, rec.last["json"])


def test_the_course_limit_refusal_raises_with_the_platform_reason():
    reason = ("Course limit reached: you own 3 active courses and may own 3. "
              "Archive or delete a course to create another.")
    c, _ = make_client(lambda *_: (403, {"error": reason}))
    for call in (lambda: c.duplicate_course(7), lambda: c.unarchive_course(7)):
        try:
            call()
        except mlarena.MLArenaError as exc:
            _expect(reason in str(exc), str(exc))
        else:
            raise AssertionError("a refused course action did not raise")


def test_delete_course_is_one_delete_and_a_409_says_why():
    plan = {"can_delete": True, "enrollment_count": 0,
            "deleted_modules": [{"id": 4, "title": "Week 2"}],
            "kept_modules": [{"id": 9, "title": "Lessons"}]}
    c, rec = make_client(lambda *_: (200, plan))
    _expect(c.course_deletion(7) == plan, "course_deletion")
    _expect((rec.last["method"], rec.last["path"]) == ("GET", "/teacher/course/7/deletion"),
            rec.last)
    _expect(c.delete_course(7) == plan, "delete_course")
    _expect((rec.last["method"], rec.last["path"]) == ("DELETE", "/teacher/course/7"), rec.last)

    reason = "The course has 4 enrolled user(s): remove them first, or archive the course instead"
    c, _ = make_client(lambda *_: (409, {"error": reason, "enrollment_count": 4}))
    try:
        c.delete_course(7)
    except mlarena.MLArenaError as exc:
        _expect(reason in str(exc), str(exc))
    else:
        raise AssertionError("delete_course on a course with students did not raise")


def test_send_course_report_posts_and_returns_who_got_it():
    sent = {"recipients": ["owner@example.com"], "challenge_count": 2,
            "attachment_count": 4, "report_last_sent_at_ts": "2026-11-20T10:00:00Z"}
    c, rec = make_client(lambda *_: (200, sent))
    _expect(c.send_course_report(7) == sent, "send_course_report")
    _expect((rec.last["method"], rec.last["path"])
            == ("POST", "/teacher/course/7/report/send"), rec.last)

    c, _ = make_client(lambda *_: (429, {"error": "The report was sent a moment ago",
                                         "retry_after_sec": 540}))
    try:
        c.send_course_report(7)
    except mlarena.MLArenaError as exc:
        _expect("sent a moment ago" in str(exc), str(exc))
    else:
        raise AssertionError("a rate-limited send did not raise")


def test_export_course_grades_csv_writes_the_file():
    c, rec = make_client(_csv_router)
    with tempfile.TemporaryDirectory() as d:
        out = c.export_course_grades_csv(5, dest_dir=d)
        _expect(os.path.basename(out) == "grades_course_5.csv", out)
        with open(out) as fh:
            _expect("ana" in fh.read(), "csv body not written")
    _expect(rec.last["path"] == "/teacher/course/5/progress.csv", rec.last["path"])
    _expect(not rec.last["params"], rec.last["params"])


def test_challenges_for_course_lists_the_picker():
    metric = {"key": "reward", "label": "Reward", "source": "score",
              "agg": "mean", "order": "desc", "format": "number",
              "unit": None, "precision": 2, "is_ranking": True,
              "visible": True}
    rows = [{"id": 42, "name": "CartPole-v1", "metric": metric}]
    c, rec = make_client(lambda *_: (200, rows))
    out = c.challenges_for_course()
    _expect(rec.last["path"] == "/teacher/challenges-for-course", rec.last["path"])
    ids = list(out["id"]) if hasattr(out, "columns") else [r["id"] for r in out]
    _expect(ids == [42], ids)
    got = list(out["metric"]) if hasattr(out, "columns") else [r["metric"] for r in out]
    _expect(got == [metric], got)


# --------------------------------------------------------------------------- #
# Roster
# --------------------------------------------------------------------------- #


def _progress_payload():
    def cell(team_id, team_name):
        return {"challenge": {"id": 42}, "team_id": team_id, "team_name": team_name}

    def other():
        return {"challenge": {"id": 43}, "team_id": None, "team_name": None}

    def student(user_id, username, enrolled, team):
        return {"user_id": user_id, "username": username, "student_number": f"S{user_id}",
                "student_email": f"{username}@uni.edu", "enrolled_at_ts": enrolled,
                "last_active_at_ts": None, "passed_count": 0,
                "challenges": [other(), cell(*team)]}

    return {
        "challenges": [
            {"challenge": {"id": 43, "name": "Maze"}, "is_published": True},
            {"challenge": {"id": 42, "name": "CartPole-v1"}, "is_published": True},
        ],
        "lesson_total": 0, "pass_bar_count": 0,
        "students": [
            student(8, "bob", "2026-02-02T09:00:00Z", (3, "Reds")),
            student(7, "ana", "2026-02-01T09:00:00Z", (None, None)),
        ],
    }


def _deprecated(call):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = call()
    _expect(any(issubclass(w.category, DeprecationWarning)
                and "course_progress" in str(w.message) for w in caught), caught)
    return out


def test_course_students_is_built_from_the_progress_route_in_its_old_shape():
    """The roster route is gone: `course_students` composes `course_progress`
    and warns. Same keys, students by enrollment date, the team of the
    challenge named."""
    c, rec = make_client(lambda *_: (200, _progress_payload()))
    out = _deprecated(lambda: c.course_students(5, challenge_id=42))
    _expect(rec.last["method"] == "GET", rec.last["method"])
    _expect(rec.last["path"] == "/teacher/course/5/progress", rec.last["path"])
    _expect(sorted(out) == ["challenges", "selected_challenge_id", "students"], out)
    _expect(out["selected_challenge_id"] == 42, out)
    _expect([ch["id"] for ch in out["challenges"]] == [43, 42], out["challenges"])
    _expect(out["students"] == [
        {"user_id": 7, "username": "ana", "student_number": "S7",
         "student_email": "ana@uni.edu", "team_id": None, "team_name": None,
         "enrolled_at_ts": "2026-02-01T09:00:00Z"},
        {"user_id": 8, "username": "bob", "student_number": "S8",
         "student_email": "bob@uni.edu", "team_id": 3, "team_name": "Reds",
         "enrolled_at_ts": "2026-02-02T09:00:00Z"},
    ], out["students"])


def test_course_students_without_a_challenge_names_no_team():
    c, _rec = make_client(lambda *_: (200, _progress_payload()))
    out = _deprecated(lambda: c.course_students(5))
    _expect(out["selected_challenge_id"] is None, out)
    _expect(all(s["team_id"] is None and s["team_name"] is None for s in out["students"]), out)


def test_course_students_refuses_a_challenge_the_course_does_not_hold():
    c, _rec = make_client(lambda *_: (200, _progress_payload()))
    try:
        _deprecated(lambda: c.course_students(5, challenge_id=999))
    except MLArenaError as err:
        _expect("not attached to this course" in str(err), err)
    else:
        raise AssertionError("expected MLArenaError")


def test_remove_student_deletes_the_enrollment():
    c, rec = make_client(lambda *_: (200, {"message": "Student removed successfully"}))
    out = c.remove_student(5, 7)
    _expect(rec.last["method"] == "DELETE", rec.last["method"])
    _expect(rec.last["path"] == "/teacher/student/5/7", rec.last["path"])
    _expect(out["message"].startswith("Student removed"), out)


# --------------------------------------------------------------------------- #
# Teaching assistants
# --------------------------------------------------------------------------- #


def test_course_assistants_round_trip():
    rows = [{"user_id": 9, "username": "bob", "created_at_ts": "2026-02-01T09:00:00Z"}]
    c, rec = make_client(lambda *_: (200, rows))
    out = c.course_assistants(5)
    _expect(rec.last["path"] == "/teacher/course/5/assistants", rec.last["path"])
    ids = list(out["user_id"]) if hasattr(out, "columns") else [r["user_id"] for r in out]
    _expect(ids == [9], ids)


def test_add_course_assistant_sends_the_username():
    c, rec = make_client(lambda *_: (201, {"message": "ok", "user_id": 9,
                                           "username": "bob"}))
    out = c.add_course_assistant(5, "bob")
    _expect(rec.last["method"] == "POST", rec.last["method"])
    _expect(rec.last["path"] == "/teacher/course/5/assistants", rec.last["path"])
    _expect(rec.last["json"] == {"username": "bob"}, rec.last["json"])
    _expect(out["user_id"] == 9, out)


def test_remove_course_assistant_takes_the_user_id():
    c, rec = make_client(lambda *_: (200, {"message": "ok"}))
    c.remove_course_assistant(5, 9)
    _expect(rec.last["method"] == "DELETE", rec.last["method"])
    _expect(rec.last["path"] == "/teacher/course/5/assistants/9", rec.last["path"])


def test_assistant_errors_surface():
    c, _rec = make_client(lambda *_: (409, {"error": "already a TA"}))
    try:
        c.add_course_assistant(5, "bob")
        raise AssertionError("expected MLArenaError on a 409")
    except MLArenaError:
        pass


# --------------------------------------------------------------------------- #
# Downloads (CSV export + cover)
# --------------------------------------------------------------------------- #


def _csv_router(method, path, kwargs):
    if path.startswith("/teacher/export-csv/") or path.endswith("/progress.csv"):
        return FakeResponse(200, content=b"Rank,Username\n1,ana\n")
    return None


def test_export_course_csv_writes_the_file():
    c, rec = make_client(_csv_router)
    with tempfile.TemporaryDirectory() as d:
        out = c.export_course_csv(5, challenge_id=42, by_participant=True, dest_dir=d)
        _expect(os.path.isfile(out), out)
        _expect(os.path.basename(out) == "leaderboard_course_5_challenge_42.csv", out)
        with open(out) as fh:
            _expect("ana" in fh.read(), "csv body not written")
    _expect(rec.last["path"] == "/teacher/export-csv/5", rec.last["path"])
    _expect(rec.last["params"] == {"by_participant": "true", "challenge_id": 42},
            rec.last["params"])


def test_export_course_csv_defaults_to_by_submission():
    c, rec = make_client(_csv_router)
    with tempfile.TemporaryDirectory() as d:
        c.export_course_csv(5, 42, dest_dir=d)
    _expect(rec.last["params"] == {"by_participant": "false", "challenge_id": 42},
            rec.last["params"])


def test_export_course_csv_requires_the_challenge():
    """The route exports one attached challenge; there is no "first" default."""
    c, rec = make_client(_csv_router)
    try:
        c.export_course_csv(5)
    except TypeError:
        pass
    else:
        raise AssertionError("export_course_csv(course_id) without a challenge_id")
    _expect(rec.calls == [], rec.calls)


# --------------------------------------------------------------------------- #
# Add a challenge to a course (one transactional route)
# --------------------------------------------------------------------------- #


def test_add_course_challenge_is_one_call():
    row = {"id": 9, "module_id": 4, "title": "Week 2", "challenges": []}
    c, rec = make_client(lambda *_: (201, row))
    out = c.add_course_challenge(7, 42, title="Week 2", summary="Read me.",
                                 pass_threshold=0.5)
    _expect(out == row, out)
    _expect(len(rec.calls) == 1, rec.calls)
    _expect(rec.last["method"] == "POST", rec.last["method"])
    _expect(rec.last["path"] == "/teacher/course/7/challenges", rec.last["path"])
    _expect(rec.last["json"] == {
        "challenge_id": 42, "is_published": True, "title": "Week 2",
        "summary": "Read me.", "pass_threshold": 0.5,
    }, rec.last["json"])


def test_remove_course_challenge_is_one_call():
    body = {"message": "Challenge removed", "module_deleted": True}
    c, rec = make_client(lambda *_: (200, body))
    out = c.remove_course_challenge(7, 4)
    _expect(out == body, out)
    _expect(len(rec.calls) == 1, rec.calls)
    _expect(rec.last["method"] == "DELETE", rec.last["method"])
    _expect(rec.last["path"] == "/teacher/course/7/challenges/4", rec.last["path"])


def test_add_course_challenge_sends_only_what_was_given():
    c, rec = make_client(lambda *_: (201, {"id": 1}))
    c.add_course_challenge(7, 42, is_published=False)
    _expect(rec.last["json"] == {"challenge_id": 42, "is_published": False},
            rec.last["json"])


def test_add_course_challenge_refusal_raises_with_the_reason():
    c, _rec = make_client(lambda *_: (400, {"error": "challenge_id 42 has no evaluation"}))
    try:
        c.add_course_challenge(7, 42)
    except MLArenaError as e:
        _expect("has no evaluation" in str(e), str(e))
        _expect(e.status_code == 400, e.status_code)
    else:
        raise AssertionError("expected MLArenaError")


# --------------------------------------------------------------------------- #
# Maintenance mode (503)
# --------------------------------------------------------------------------- #


def test_maintenance_503_is_a_maintenance_error_with_the_admin_message():
    body = {"error": "Back at noon.", "maintenance_mode": True}
    c, _rec = make_client(lambda *_: (503, body))
    try:
        c.workspace()
    except mlarena.MaintenanceError as e:
        _expect(str(e) == "Back at noon.", str(e))
        _expect(e.status_code == 503 and e.body == body, (e.status_code, e.body))
        _expect(isinstance(e, MLArenaError), type(e).__mro__)
    else:
        raise AssertionError("expected MaintenanceError")


def test_a_503_without_the_flag_is_not_maintenance():
    c, _rec = make_client(lambda *_: (503, {"error": "upstream down",
                                             "upstream": "data_collection"}))
    try:
        c.workspace()
    except mlarena.MaintenanceError:
        raise AssertionError("a plain 503 is not maintenance")
    except MLArenaError as e:
        _expect("upstream down" in str(e), str(e))
    else:
        raise AssertionError("expected MLArenaError")


def test_teacher_course_says_whether_there_is_a_cover_not_where_it_lives():
    """A course payload never carries the NFS path — the catalog and the
    landing are public reads (see the backend's `has_cover`)."""
    row = {"id": 1, "name": "RL", "role": "owner", "has_cover": True}
    c, _rec = make_client(lambda *_: (200, row))
    out = c.teacher_course(1)
    _expect(out["has_cover"] is True, out)
    _expect("cover_image_path" not in out, out)


def test_set_course_cover_answers_the_flag():
    def router(method, path, kwargs):
        if path.endswith("/cover"):
            return (201, {"has_cover": True})
        return None

    c, rec = make_client(router)
    with tempfile.TemporaryDirectory() as d:
        image = os.path.join(d, "cover.png")
        with open(image, "wb") as fh:
            fh.write(b"\x89PNG")
        out = c.set_course_cover(5, image)
    _expect(rec.last["method"] == "POST", rec.last["method"])
    _expect(rec.last["path"] == "/teacher/course/5/cover", rec.last["path"])
    _expect(out == {"has_cover": True}, out)


def test_course_cover_downloads_from_the_asset_route():
    def router(method, path, kwargs):
        if path.endswith("/cover"):
            return FakeResponse(200, content=b"\x89PNG")
        return None

    c, rec = make_client(router)
    with tempfile.TemporaryDirectory() as d:
        out = c.course_cover(5, dest_dir=d)
        _expect(os.path.isfile(out), out)
    _expect(rec.last["path"] == "/academic_courses/assets/courses/5/cover",
            rec.last["path"])


def test_course_cover_404_raises():
    c, _rec = make_client(lambda *_: FakeResponse(404, {"error": "No cover image"}))
    with tempfile.TemporaryDirectory() as d:
        try:
            c.course_cover(5, dest_dir=d)
            raise AssertionError("expected an error for a course with no cover")
        except Exception as exc:  # NotFoundError or MLArenaError, both fine
            _expect(not isinstance(exc, AssertionError), repr(exc))


# --------------------------------------------------------------------------- #
# Every new method carries the bearer token
# --------------------------------------------------------------------------- #


def test_all_teacher_admin_methods_send_the_token():
    c, rec = make_client(lambda *_: (200, []))
    c.challenges_for_course()
    c.course_assistants(1)
    for call in rec.calls:
        auth = call["headers"]["Authorization"]
        _expect(auth.startswith("Bearer mlk_teacher_"), auth)


def _run_all():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failures = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as exc:  # noqa: BLE001 — test runner reports all failures
            failures += 1
            print(f"  FAIL  {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return failures


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
