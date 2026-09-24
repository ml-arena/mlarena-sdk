"""Team, `me()` and ranking wire tests.

The twelve `/api/teams/...` routes had no SDK method at all (the console was
the only client), and the two ranking reads renamed or unwrapped what the
backend serves. These pin the route, the verb, the payload keys and the
bearer header for each of them, with the same fake transport the other
suites use: the client's single HTTP chokepoint (`MLArenaClient._request`)
is replaced by a recorder.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mlarena
from mlarena.exceptions import MLArenaError, PermissionDeniedError


class FakeResponse:
    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json = {} if json_data is None else json_data
        self.text = ""

    def json(self):
        return self._json


class Recorder:
    def __init__(self, router=None):
        self.calls = []
        self.router = router

    def __call__(self, method, url, **kwargs):
        path = url.split("/api", 1)[1]
        self.calls.append({"method": method, "path": path, **kwargs})
        if self.router is not None:
            result = self.router(method, path, kwargs)
            if result is not None:
                return FakeResponse(*result)
        return FakeResponse(200, {"ok": True})

    @property
    def last(self):
        return self.calls[-1]


def make_client(router=None, scope="user"):
    c = mlarena.connect(
        api_key=f"mlk_{scope}_abcd1234_deadbeef0000",
        base_url="https://example.com",
    )
    rec = Recorder(router)
    c._request = rec
    return c, rec


TEAM = {
    "id": 3,
    "challenge_id": 4,
    "name": "Team A",
    "looking_for_members": False,
    "created_at_ts": "2026-09-01T10:00:00Z",
    "members": [
        {"id": 9, "user_id": 2, "username": "leader", "avatar_key": None,
         "role": "leader"},
    ],
    "my_role": "leader",
}

INVITATION = {
    "id": 5, "team_id": 3, "team_name": "Team A",
    "user_id": 7, "username": "invitee",
    "sender_id": 2, "sender_username": "leader",
    "status": "pending", "created_at_ts": "2026-09-01T10:05:00Z",
}


# --------------------------------------------------------------------------- #
# Teams — routes, verbs and payload keys
# --------------------------------------------------------------------------- #


def test_challenge_team_route():
    c, rec = make_client(lambda *_: (200, TEAM))
    assert c.challenge_team(4) == TEAM
    assert rec.last["method"] == "GET"
    assert rec.last["path"] == "/teams/challenge/4/team"


def test_my_role_is_served_not_derived():
    """The caller's role comes off the payload — nothing here scans
    `members` for it."""
    c, _ = make_client(lambda *_: (200, TEAM))
    assert c.challenge_team(4)["my_role"] == "leader"


def test_create_team_route_and_body():
    c, rec = make_client(lambda *_: (201, TEAM))
    assert c.create_team(4, "Team A") == TEAM
    assert rec.last["method"] == "POST"
    assert rec.last["path"] == "/teams/challenge/4/team"
    assert rec.last["json"] == {"name": "Team A"}


def test_update_team_sends_both_keys():
    c, rec = make_client(lambda *_: (200, TEAM))
    c.update_team(3, "Team B", True)
    assert rec.last["method"] == "PUT"
    assert rec.last["path"] == "/teams/3"
    assert rec.last["json"] == {"name": "Team B", "looking_for_members": True}


def test_update_team_can_clear_looking_for_members():
    c, rec = make_client(lambda *_: (200, TEAM))
    c.update_team(3, "Team B", None)
    assert rec.last["json"] == {"name": "Team B", "looking_for_members": None}


def test_delete_team_route():
    c, rec = make_client(lambda *_: (200, {"message": "Team deleted successfully"}))
    c.delete_team(3)
    assert (rec.last["method"], rec.last["path"]) == ("DELETE", "/teams/3")


def test_leave_team_route():
    c, rec = make_client(lambda *_: (200, {"message": "Successfully left team"}))
    c.leave_team(3)
    assert (rec.last["method"], rec.last["path"]) == ("POST", "/teams/3/leave")


def test_remove_team_member_takes_the_membership_row_id():
    c, rec = make_client(lambda *_: (200, {"message": "Member removed successfully"}))
    member = TEAM["members"][0]
    c.remove_team_member(3, member["id"])
    assert (rec.last["method"], rec.last["path"]) == ("DELETE", "/teams/3/members/9")


def test_search_teams_sends_both_query_keys():
    c, rec = make_client(lambda *_: (200, {"users": []}))
    c.search_teams(4, "ali")
    assert rec.last["method"] == "GET"
    assert rec.last["path"] == "/teams/search"
    assert rec.last["params"] == {"q": "ali", "challenge_id": 4}


def test_search_teams_rows_carry_the_column_names():
    rows = {"users": [{"id": 7, "username": "alice", "avatar_key": "bottts:x"}]}
    c, _ = make_client(lambda *_: (200, rows))
    assert c.search_teams(4, "ali") == rows


def test_invite_to_team_route_and_body():
    c, rec = make_client(lambda *_: (201, INVITATION))
    assert c.invite_to_team(3, 7) == INVITATION
    assert rec.last["method"] == "POST"
    assert rec.last["path"] == "/teams/3/invitations"
    assert rec.last["json"] == {"user_id": 7}


def test_pending_invitations_route():
    c, rec = make_client(lambda *_: (200, [INVITATION]))
    assert c.pending_invitations(3) == [INVITATION]
    assert rec.last["path"] == "/teams/3/invitations/pending"


def test_received_invitations_route():
    c, rec = make_client(lambda *_: (200, [INVITATION]))
    assert c.received_invitations() == [INVITATION]
    assert rec.last["path"] == "/teams/invitations/received"


def test_respond_to_invitation_route_and_body():
    settled = {**INVITATION, "status": "accepted"}
    c, rec = make_client(lambda *_: (200, settled))
    assert c.respond_to_invitation(5, True)["status"] == "accepted"
    assert rec.last["method"] == "POST"
    assert rec.last["path"] == "/teams/invitations/5/respond"
    assert rec.last["json"] == {"accept": True}


def test_respond_to_invitation_quota_409_carries_the_numbers():
    body = {
        "error": "Joining this team would exceed the challenge's "
                 "active-submission limit",
        "combined_active": 4, "max_active_submissions": 3, "excess": 1,
    }
    c, _ = make_client(lambda *_: (409, body))
    try:
        c.respond_to_invitation(5, True)
        raise AssertionError("expected MLArenaError")
    except MLArenaError as exc:
        assert exc.status_code == 409
        assert exc.body["combined_active"] == 4
        assert exc.body["max_active_submissions"] == 3
        assert exc.body["excess"] == 1


def test_cancel_invitation_route():
    c, rec = make_client(lambda *_: (200, {"message": "Invitation cancelled successfully"}))
    c.cancel_invitation(5)
    assert (rec.last["method"], rec.last["path"]) == ("DELETE", "/teams/invitations/5")


def _every_team_call(c):
    return [
        lambda: c.challenge_team(4),
        lambda: c.create_team(4, "T"),
        lambda: c.update_team(3, "T", False),
        lambda: c.delete_team(3),
        lambda: c.leave_team(3),
        lambda: c.remove_team_member(3, 9),
        lambda: c.search_teams(4, "a"),
        lambda: c.invite_to_team(3, 7),
        lambda: c.pending_invitations(3),
        lambda: c.received_invitations(),
        lambda: c.respond_to_invitation(5, False),
        lambda: c.cancel_invitation(5),
    ]


def test_every_team_call_sends_the_bearer_token():
    def router(method, path, kwargs):
        if method == "POST" and path.endswith("/team"):
            return (201, TEAM)
        if path.endswith("/invitations") and method == "POST":
            return (201, INVITATION)
        return (200, {"ok": True})

    c, rec = make_client(router)
    for call in _every_team_call(c):
        rec.calls.clear()
        call()
        auth = (rec.last.get("headers") or {}).get("Authorization", "")
        assert auth.startswith("Bearer "), rec.last["path"]


def test_a_team_403_is_a_permission_denied_error():
    c, _ = make_client(lambda *_: (403, {"error": "Only team leaders can delete the team"}))
    try:
        c.delete_team(3)
        raise AssertionError("expected PermissionDeniedError")
    except PermissionDeniedError as exc:
        assert "Only team leaders" in str(exc)


# --------------------------------------------------------------------------- #
# me()
# --------------------------------------------------------------------------- #


ME = {
    "is_authenticated": True, "id": 2, "username": "alice",
    "email": "alice@example.com", "avatar_key": None,
    "is_admin": False, "is_teacher": False, "is_creator": True,
    "is_human_verified": True, "has_teacher_access": False,
    "has_creator_access": True, "can_create_course": True,
}


def test_me_route_and_shape():
    c, rec = make_client(lambda *_: (200, ME))
    assert c.me() == ME
    assert (rec.last["method"], rec.last["path"]) == ("GET", "/auth/current_user")
    assert (rec.last.get("headers") or {})["Authorization"].startswith("Bearer ")


def test_the_sdk_offers_no_api_key_management():
    """`/api/auth/api_keys` and its rotate are session-only (a bearer key may
    not mint another key), so the client deliberately has no method for
    them — an SDK method would only ever raise 403."""
    c, _ = make_client()
    assert not hasattr(c, "api_keys")
    assert not hasattr(c, "rotate_api_key")


# --------------------------------------------------------------------------- #
# Ranking — served names, no unwrapping
# --------------------------------------------------------------------------- #


RANKING_PAGE = {
    "rankings": [
        {"user_id": 2, "username": "alice", "avatar_key": None, "rank": 1,
         "current_points": 12.5, "medals_gold": 1, "medals_silver": 0,
         "medals_bronze": 2},
    ],
    "metadata": {"total_pages": 3, "current_page": 1, "total_users": 57,
                 "has_next": True, "has_prev": False},
}


def test_global_ranking_keeps_the_metadata_envelope():
    c, rec = make_client(lambda *_: (200, RANKING_PAGE))
    rows = c.global_ranking(search="ali", page=2, per_page=10)
    assert rec.last["params"] == {"page": 2, "per_page": 10, "search": "ali"}
    # With pandas the rows are a DataFrame carrying `metadata` in `.attrs`.
    if hasattr(rows, "attrs"):
        assert rows.attrs["metadata"] == RANKING_PAGE["metadata"]
        assert list(rows["current_points"]) == [12.5]
        assert list(rows["medals_gold"]) == [1]
    else:
        assert rows == RANKING_PAGE["rankings"]


def test_user_global_rank_is_flat():
    flat = {"user_id": 2, "username": "alice", "rank": 1,
            "current_points": 12.5, "percentile": 98.2,
            "medals_gold": 1, "medals_silver": 0, "medals_bronze": 2}
    c, rec = make_client(lambda *_: (200, flat))
    assert c.user_global_rank(2) == flat
    assert rec.last["path"] == "/ranking/user/2"
