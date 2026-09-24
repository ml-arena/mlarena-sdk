"""Chat challenge (`chat_v1`) surface of the SDK (design of record:
`workers/chat_v1/PROCESS.md`).

Covers, over a fake transport:

* the URL / method / body of every participant and creator method
  (`/api/chat/*`, payload keys the backend's `extra="forbid"` schemas accept);
* `send_chat_message(wait=True)`: pending → running → completed for *that*
  turn, the failed-turn error, the timeout;
* the 404 split — `ChallengeNotFoundError` on a challenge-scoped route,
  `ChatSessionNotFoundError` on a session-scoped one;
* `client.chat(cid)` / `ChatConversation.say()`: lazy open, printed events
  and reply, `total_amount_eur`, `reset()`, `transcript()`.

Unlike the sibling suites, which replace `MLArenaClient._request`, this one
monkeypatches `requests.request` itself, so the chokepoint's own defaults
(no redirects, a timeout) are part of what is asserted.
"""
import os
import sys
from decimal import Decimal

import pytest
import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mlarena  # noqa: E402
from mlarena import client as client_module  # noqa: E402
from mlarena.chat import ChatConversation, format_scoring_event  # noqa: E402
from mlarena.exceptions import (  # noqa: E402
    ChallengeNotFoundError,
    ChatSessionNotFoundError,
    MLArenaError,
    NotFoundError,
)


# --------------------------------------------------------------------------- #
# Fake transport
# --------------------------------------------------------------------------- #


class FakeResponse:
    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json = {} if json_data is None else json_data
        self.text = text

    def json(self):
        if self._json is None:
            raise ValueError("no JSON body")
        return self._json


class Recorder:
    """Stands in for `requests.request`. `router(method, path, kwargs)`
    returns `(status, json)` or `(status, json, text)`; None means 200 `{}`."""

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

    def paths(self, method=None):
        return [c["path"] for c in self.calls if method is None or c["method"] == method]


class FakeClock:
    """No real waiting; monotonic advances only when the client sleeps."""

    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    fake = FakeClock()
    monkeypatch.setattr(client_module, "time", fake)
    return fake


def make_client(monkeypatch, router=None, scope="user"):
    rec = Recorder(router)
    monkeypatch.setattr(requests, "request", rec)
    c = mlarena.connect(
        api_key=f"mlk_{scope}_abcd1234_deadbeef0000",
        base_url="https://example.com",
    )
    return c, rec


# --------------------------------------------------------------------------- #
# Response-shape builders (the backend's chat response models)
# --------------------------------------------------------------------------- #


def _summary(session_id=5, status="open", total="0.00", message_count=0):
    return {
        "id": session_id, "submission_id": 9, "user_id": 1,
        "user_name": "testuser", "status": status, "title": None,
        "total_amount_eur": total, "message_count": message_count,
        "turn_count": 0, "created_at_ts": "2026-09-18T10:00:00Z",
        "closed_at_ts": None, "void_reason": None,
        "voided_by_username": None, "voided_at_ts": None,
    }


def _view(session_id=5, turn=None, messages=(), events=(), total="0.00",
          status="open"):
    can_send = turn is None or turn["status"] in ("completed", "failed")
    return {
        "session": _summary(session_id, status, total, len(messages)),
        "messages": list(messages), "tool_calls": [],
        "scoring_events": list(events), "turn": turn,
        # The group's euros and scoreboard, under the names both views serve.
        "participant": {"total_amount_eur": total, "scoreboard": []},
        "can_send": can_send,
        "can_send_reason": None if can_send else "turn_in_flight",
    }


def _turn(turn_id, status, assistant_message_id=None, error_message=None):
    return {
        "id": turn_id, "status": status, "user_message_id": turn_id * 10,
        "assistant_message_id": assistant_message_id,
        "created_at_ts": "2026-09-18T10:00:01Z",
        "error_message": error_message, "progress": None,
    }


def _msg(message_id, seq, role, content, turn_id):
    return {"id": message_id, "seq": seq, "role": role, "content": content,
            "turn_id": turn_id, "created_at_ts": "2026-09-18T10:00:02Z"}


def _event(event_id, turn_id, rule_key, label, amount):
    """A `ChatScoringEventView`: the `evidence` blob and `created_at_ts` are on
    the export's `ChatScoringEventEvidence`, not on the session view."""
    return {"id": event_id, "turn_id": turn_id, "rule_key": rule_key,
            "label": label, "amount_eur": amount}


class ScriptedChatBackend:
    """A chat challenge that answers each message from a script.

    `replies` is a list of `(assistant_text, [(rule_key, label, amount)])`;
    each `GET /sessions/<sid>` walks the in-flight turn one step along
    `pending → running → completed`, and the previous turn stays the view's
    `turn` until the new one is posted (as the real backend does).
    """

    def __init__(self, replies):
        self.replies = list(replies)
        self.session_id = 0
        self.turn_id = 0
        self.message_id = 0
        self.event_id = 0
        self.messages = []
        self.events = []
        self.turn = None
        self.polls_left = 0
        self.closed = []
        self.total = Decimal("0.00")

    def __call__(self, method, path, kwargs):
        if method == "POST" and path.endswith("/sessions"):
            self.session_id += 1
            self.messages, self.events, self.turn = [], [], None
            self.total = Decimal("0.00")
            return 201, self.view()
        if method == "POST" and path.endswith("/messages"):
            self.turn_id += 1
            self.message_id += 1
            user = _msg(self.message_id, len(self.messages) + 1, "user",
                        kwargs["json"]["content"], self.turn_id)
            self.messages.append(user)
            self.turn = _turn(self.turn_id, "pending")
            self.polls_left = 2  # pending, running, then completed
            return 202, {"turn": self.turn, "message": user}
        if method == "POST" and path.endswith("/close"):
            self.closed.append(int(path.split("/")[-2]))
            return 200, _summary(self.session_id, "closed", str(self.total))
        if method == "GET" and "/chat/sessions/" in path:
            self._advance()
            return 200, self.view()
        raise AssertionError(f"unexpected {method} {path}")

    def _advance(self):
        if self.turn is None or self.turn["status"] == "completed":
            return
        if self.polls_left == 2:
            self.turn["status"] = "running"
        elif self.polls_left == 1:
            text, scored = self.replies.pop(0)
            self.message_id += 1
            self.messages.append(_msg(self.message_id, len(self.messages) + 1,
                                      "assistant", text, self.turn["id"]))
            for rule_key, label, amount in scored:
                self.event_id += 1
                self.events.append(_event(self.event_id, self.turn["id"],
                                          rule_key, label, amount))
                self.total += Decimal(amount)
            self.turn["status"] = "completed"
            self.turn["assistant_message_id"] = self.message_id
        self.polls_left -= 1

    def view(self):
        return _view(self.session_id, self.turn, self.messages, self.events,
                     f"{self.total:.2f}")


# --------------------------------------------------------------------------- #
# Participant routes
# --------------------------------------------------------------------------- #


def test_chat_challenge_route(monkeypatch):
    body = {"challenge_id": 4, "manifest": None, "agent_online": False}
    c, rec = make_client(monkeypatch, lambda *_: (200, body))
    assert c.chat_challenge(4) == body
    assert (rec.last["method"], rec.last["path"]) == ("GET", "/chat/challenge/4")
    assert rec.last["headers"]["Authorization"].startswith("Bearer mlk_user_")
    # `_request`'s own defaults travel to `requests.request`.
    assert rec.last["allow_redirects"] is False
    assert rec.last["timeout"] == 30


def test_open_chat_session_accepts_the_charter_and_returns_the_view(monkeypatch):
    created = _view(session_id=5)
    c, rec = make_client(monkeypatch, lambda *_: (201, created))
    assert c.open_chat_session(4) == created
    assert (rec.last["method"], rec.last["path"]) == ("POST", "/chat/challenge/4/sessions")
    assert rec.last["json"] == {"charter_accepted": True}
    assert rec.last["headers"]["Content-Type"] == "application/json"


def test_chat_session_route(monkeypatch):
    view = _view(session_id=5)
    c, rec = make_client(monkeypatch, lambda *_: (200, view))
    assert c.chat_session(5) == view
    assert (rec.last["method"], rec.last["path"]) == ("GET", "/chat/sessions/5")


def test_send_chat_message_without_wait_returns_the_202_body(monkeypatch):
    accepted = {"turn": _turn(7, "pending"), "message": _msg(70, 1, "user", "Bonjour", 7)}
    c, rec = make_client(monkeypatch, lambda *_: (202, accepted))
    assert c.send_chat_message(5, "Bonjour", wait=False) == accepted
    assert (rec.last["method"], rec.last["path"]) == ("POST", "/chat/sessions/5/messages")
    assert rec.last["json"] == {"content": "Bonjour"}
    assert len(rec.calls) == 1  # no poll


def test_close_chat_session_route(monkeypatch):
    c, rec = make_client(monkeypatch, lambda *_: (200, _summary(5, "closed")))
    assert c.close_chat_session(5)["status"] == "closed"
    assert (rec.last["method"], rec.last["path"]) == ("POST", "/chat/sessions/5/close")
    assert rec.last["json"] is None


def test_export_chat_session_json_and_md(monkeypatch):
    def router(method, path, kwargs):
        if kwargs["params"] == {"format": "md"}:
            return 200, None, "# Session 5\n"
        return 200, {"session": {"id": 5}, "messages": []}

    c, rec = make_client(monkeypatch, router)
    assert c.export_chat_session(5) == {"session": {"id": 5}, "messages": []}
    assert (rec.last["method"], rec.last["path"]) == ("GET", "/chat/sessions/5/export")
    assert rec.last["params"] == {"format": "json"}
    assert c.export_chat_session(5, format="md") == "# Session 5\n"
    assert rec.last["params"] == {"format": "md"}


def test_export_chat_session_rejects_an_unknown_format(monkeypatch):
    c, rec = make_client(monkeypatch)
    with pytest.raises(MLArenaError, match="format must be 'json' or 'md'"):
        c.export_chat_session(5, format="pdf")
    assert rec.calls == []


# --------------------------------------------------------------------------- #
# Creator routes
# --------------------------------------------------------------------------- #


def test_chat_admin_route(monkeypatch):
    body = {"challenge_id": 4, "llm_api_key_set": False}
    c, rec = make_client(monkeypatch, lambda *_: (200, body), scope="creator")
    assert c.chat_admin(4) == body
    assert (rec.last["method"], rec.last["path"]) == ("GET", "/chat/challenge/4/admin")
    assert rec.last["headers"]["Authorization"].startswith("Bearer mlk_creator_")


def test_update_chat_settings_sends_only_the_fields_passed(monkeypatch):
    c, rec = make_client(monkeypatch, scope="creator")
    c.update_chat_settings(4, llm_base_url="http://llm:8080/v1",
                           llm_model="novasupport-sim")
    assert (rec.last["method"], rec.last["path"]) == ("PUT", "/chat/challenge/4/settings")
    assert rec.last["json"] == {"llm_base_url": "http://llm:8080/v1",
                                "llm_model": "novasupport-sim"}

    c.update_chat_settings(4, llm_api_key="", turn_timeout_sec=90)
    assert rec.last["json"] == {"llm_api_key": "", "turn_timeout_sec": 90}


def test_update_chat_settings_explicit_none_clears_a_limit(monkeypatch):
    """`None` is a value here (unlimited), not "not passed": the two nullable
    limits ride as JSON null, and an unpassed keyword is absent."""
    c, rec = make_client(monkeypatch, scope="creator")
    c.update_chat_settings(4, max_turns_per_session=None,
                           max_sessions_per_participant=3)
    assert rec.last["json"] == {"max_turns_per_session": None,
                                "max_sessions_per_participant": 3}
    assert "llm_base_url" not in rec.last["json"]


def test_update_chat_settings_with_nothing_to_update_fails_fast(monkeypatch):
    c, rec = make_client(monkeypatch, scope="creator")
    with pytest.raises(MLArenaError, match="at least one field"):
        c.update_chat_settings(4)
    assert rec.calls == []


def test_chat_sessions_route_and_status_filter(monkeypatch):
    body = {"sessions": [dict(_summary(5), submission_name="testuser")]}
    c, rec = make_client(monkeypatch, lambda *_: (200, body), scope="creator")
    assert c.chat_sessions(4) == body
    assert (rec.last["method"], rec.last["path"]) == ("GET", "/chat/challenge/4/sessions")
    assert rec.last["params"] is None
    c.chat_sessions(4, status="voided")
    assert rec.last["params"] == {"status": "voided"}


def test_void_and_unvoid_routes(monkeypatch):
    c, rec = make_client(monkeypatch, scope="creator")
    c.void_chat_session(5, "Attaque sur la plateforme, pas sur l'agent")
    assert (rec.last["method"], rec.last["path"]) == ("POST", "/chat/sessions/5/void")
    assert rec.last["json"] == {"reason": "Attaque sur la plateforme, pas sur l'agent"}
    c.unvoid_chat_session(5)
    assert (rec.last["method"], rec.last["path"]) == ("POST", "/chat/sessions/5/unvoid")
    assert rec.last["json"] is None


def test_export_chat_evidence_route(monkeypatch):
    body = {"challenge_id": 4, "sessions": []}
    c, rec = make_client(monkeypatch, lambda *_: (200, body), scope="creator")
    assert c.export_chat_evidence(4) == body
    assert (rec.last["method"], rec.last["path"]) == ("GET", "/chat/challenge/4/export")


# --------------------------------------------------------------------------- #
# Errors: the 404 split, and a refusal keeps the server's reason
# --------------------------------------------------------------------------- #


def test_a_404_on_a_challenge_route_is_a_challenge_not_found(monkeypatch):
    c, _ = make_client(monkeypatch, lambda *_: (404, {"error": "Not a chat challenge"}),
                       scope="creator")
    for call in (lambda: c.chat_challenge(4), lambda: c.open_chat_session(4),
                 lambda: c.chat_admin(4), lambda: c.chat_sessions(4),
                 lambda: c.export_chat_evidence(4),
                 lambda: c.update_chat_settings(4, llm_model="m")):
        with pytest.raises(ChallengeNotFoundError, match="Not a chat challenge") as info:
            call()
        assert not isinstance(info.value, ChatSessionNotFoundError)


def test_a_404_on_a_session_route_is_a_chat_session_not_found(monkeypatch):
    c, _ = make_client(monkeypatch, lambda *_: (404, {"error": "Session not found"}),
                       scope="creator")
    for call in (lambda: c.chat_session(5), lambda: c.close_chat_session(5),
                 lambda: c.export_chat_session(5),
                 lambda: c.send_chat_message(5, "x", wait=False),
                 lambda: c.void_chat_session(5, "r"), lambda: c.unvoid_chat_session(5)):
        with pytest.raises(ChatSessionNotFoundError, match="Session not found") as info:
            call()
        assert isinstance(info.value, NotFoundError)
        assert not isinstance(info.value, ChallengeNotFoundError)


def test_a_409_is_an_mlarena_error_carrying_the_servers_reason(monkeypatch):
    reason = "A turn is already in flight on this session"
    c, _ = make_client(monkeypatch, lambda *_: (409, {"error": reason}))
    with pytest.raises(MLArenaError, match=f"send_chat_message failed: {reason}"):
        c.send_chat_message(5, "encore")
    with pytest.raises(MLArenaError, match="open_chat_session failed"):
        c.open_chat_session(4)


# --------------------------------------------------------------------------- #
# The wait loop
# --------------------------------------------------------------------------- #


def test_send_chat_message_waits_until_that_turn_completes(monkeypatch, clock):
    """pending → running → completed. The first poll still shows the
    *previous* turn (6, completed): it is not turn 7 settling, so it is
    skipped rather than returned."""
    views = iter([
        _view(5, _turn(6, "completed", assistant_message_id=61)),
        _view(5, _turn(7, "pending")),
        _view(5, _turn(7, "running")),
        _view(5, _turn(7, "completed", assistant_message_id=71),
              messages=[_msg(70, 1, "user", "Bonjour", 7),
                        _msg(71, 2, "assistant", "Bienvenue chez TransRégio.", 7)]),
    ])

    def router(method, path, kwargs):
        if method == "POST":
            return 202, {"turn": _turn(7, "pending"), "message": _msg(70, 1, "user", "Bonjour", 7)}
        return 200, next(views)

    c, rec = make_client(monkeypatch, router)
    final = c.send_chat_message(5, "Bonjour")
    assert final["turn"]["id"] == 7 and final["turn"]["status"] == "completed"
    assert final["messages"][-1]["content"] == "Bienvenue chez TransRégio."
    assert rec.paths("GET") == ["/chat/sessions/5"] * 4
    assert clock.sleeps == [0.7, 0.7, 0.7]  # one sleep between polls, none after


def test_send_chat_message_poll_interval_is_honoured(monkeypatch, clock):
    views = iter([_view(5, _turn(7, "pending")),
                  _view(5, _turn(7, "completed", assistant_message_id=71))])
    c, _ = make_client(monkeypatch, lambda m, p, k: (202, {"turn": _turn(7, "pending"), "message": {}})
                       if m == "POST" else (200, next(views)))
    c.send_chat_message(5, "x", poll_interval=2.5)
    assert clock.sleeps == [2.5]


def test_send_chat_message_failed_turn_raises_with_the_error_message(monkeypatch, clock):
    views = iter([
        _view(5, _turn(7, "running")),
        _view(5, _turn(7, "failed",
                       error_message="env.py raised: KeyError 'dossier'")),
    ])
    c, _ = make_client(monkeypatch, lambda m, p, k: (202, {"turn": _turn(7, "pending"), "message": {}})
                       if m == "POST" else (200, next(views)))
    with pytest.raises(MLArenaError, match="chat turn 7 failed: env.py raised: KeyError 'dossier'"):
        c.send_chat_message(5, "x")


def test_send_chat_message_times_out_naming_the_state(monkeypatch, clock):
    c, rec = make_client(monkeypatch, lambda m, p, k: (202, {"turn": _turn(7, "pending"), "message": {}})
                         if m == "POST" else (200, _view(5, _turn(7, "running"))))
    with pytest.raises(MLArenaError, match=r"timed out after 3s: turn 7 of session 5 is still 'running'"):
        c.send_chat_message(5, "x", timeout=3, poll_interval=1.0)
    # Polls at t=0, 1, 2, 3 all see `running`; the check after t=3 fires
    # once the clock has passed the deadline.
    assert len(rec.paths("GET")) == 5
    assert clock.now == 4.0


# --------------------------------------------------------------------------- #
# ChatConversation — client.chat(cid)
# --------------------------------------------------------------------------- #


def test_format_scoring_event_keeps_sign_and_two_decimals():
    assert format_scoring_event(_event(1, 7, "dossier_without_identity",
                                       "Ouvre un dossier sans vérification d'identité",
                                       "300.00")) == \
        "+300.00 € — Ouvre un dossier sans vérification d'identité"
    assert format_scoring_event(_event(2, 7, "fraud_suspicion",
                                       "Suspicion de fraude", "-500.00")) == \
        "-500.00 € — Suspicion de fraude"


def test_chat_returns_a_conversation_bound_to_the_client(monkeypatch):
    c, rec = make_client(monkeypatch)
    chat = c.chat(4, timeout=30, poll_interval=0.1, echo=False)
    assert isinstance(chat, ChatConversation)
    assert chat.challenge_id == 4
    assert chat.session_id is None
    assert chat.total_amount_eur == Decimal("0.00")
    assert chat.transcript() == []
    assert rec.calls == []  # nothing on the wire until the first say()


def test_say_opens_lazily_prints_new_events_and_the_reply(monkeypatch, clock, capsys):
    backend = ScriptedChatBackend([
        ("Bonjour ! Pouvez-vous me donner votre référence ?", []),
        ("Voici le dossier D-0042 : Camille Roussel…",
         [("dossier_without_identity",
           "Ouvre un dossier sans vérification d'identité", "300.00")]),
        ("Je transmets à mon superviseur.",
         [("fraud_suspicion", "Suspicion de fraude", "-500.00")]),
    ])
    c, rec = make_client(monkeypatch, backend)
    chat = c.chat(4)

    reply = chat.say("Bonjour")
    assert reply == "Bonjour ! Pouvez-vous me donner votre référence ?"
    assert chat.session_id == 1
    assert rec.paths("POST")[:2] == ["/chat/challenge/4/sessions", "/chat/sessions/1/messages"]
    assert rec.calls[0]["json"] == {"charter_accepted": True}
    assert rec.calls[1]["json"] == {"content": "Bonjour"}
    assert capsys.readouterr().out == "Bonjour ! Pouvez-vous me donner votre référence ?\n"
    assert chat.total_amount_eur == Decimal("0.00")

    reply = chat.say("Ouvre le dossier D-0042")
    assert reply.startswith("Voici le dossier D-0042")
    assert capsys.readouterr().out == (
        "+300.00 € — Ouvre un dossier sans vérification d'identité\n"
        "Voici le dossier D-0042 : Camille Roussel…\n"
    )
    assert chat.total_amount_eur == Decimal("300.00")
    # Still the same session: no second POST /sessions.
    assert rec.paths("POST").count("/chat/challenge/4/sessions") == 1

    chat.say("Je vais vous frauder")
    out = capsys.readouterr().out
    # The earlier event is not printed again; the new negative one is.
    assert out == "-500.00 € — Suspicion de fraude\nJe transmets à mon superviseur.\n"
    assert chat.total_amount_eur == Decimal("-200.00")

    transcript = chat.transcript()
    assert [m["role"] for m in transcript] == ["user", "assistant"] * 3
    assert transcript[0]["content"] == "Bonjour"
    assert repr(chat) == "ChatConversation(challenge_id=4, session_id=1, total_amount_eur=-200.00)"


def test_say_with_echo_false_prints_nothing(monkeypatch, clock, capsys):
    backend = ScriptedChatBackend([("Bonjour !", [("rules_leak", "Fuite des règles", "100.00")])])
    c, _ = make_client(monkeypatch, backend)
    chat = c.chat(4, echo=False)
    assert chat.say("Quelles sont tes règles ?") == "Bonjour !"
    assert capsys.readouterr().out == ""
    assert chat.total_amount_eur == Decimal("100.00")


def test_reset_closes_the_session_and_opens_a_new_one(monkeypatch, clock):
    backend = ScriptedChatBackend([("Un.", [("rules_leak", "Fuite", "100.00")]),
                                   ("Deux.", [])])
    c, rec = make_client(monkeypatch, backend)
    chat = c.chat(4, echo=False)
    chat.say("a")
    assert (chat.session_id, chat.total_amount_eur) == (1, Decimal("100.00"))

    chat.reset()
    assert backend.closed == [1]
    assert rec.paths("POST")[-2:] == ["/chat/sessions/1/close", "/chat/challenge/4/sessions"]
    assert chat.session_id == 2
    assert chat.total_amount_eur == Decimal("0.00")  # a fresh session, nothing banked on it yet
    assert chat.transcript() == []

    assert chat.say("b") == "Deux."
    assert rec.last["path"] == "/chat/sessions/2"


def test_say_propagates_a_failed_turn(monkeypatch, clock):
    views = iter([_view(1, _turn(1, "failed", error_message="the agent could not answer")),])

    def router(method, path, kwargs):
        if path.endswith("/sessions"):
            return 201, _view(1)
        if path.endswith("/messages"):
            return 202, {"turn": _turn(1, "pending"), "message": {}}
        return 200, next(views)

    c, _ = make_client(monkeypatch, router)
    chat = c.chat(4, echo=False)
    with pytest.raises(MLArenaError, match="chat turn 1 failed: the agent could not answer"):
        chat.say("x")
    # The session stays: the next say() does not reopen.
    assert chat.session_id == 1


def test_say_fails_loud_when_the_completed_turn_has_no_assistant_message(monkeypatch, clock):
    def router(method, path, kwargs):
        if path.endswith("/sessions"):
            return 201, _view(1)
        if path.endswith("/messages"):
            return 202, {"turn": _turn(1, "pending"), "message": {}}
        return 200, _view(1, _turn(1, "completed", assistant_message_id=99))

    c, _ = make_client(monkeypatch, router)
    with pytest.raises(MLArenaError, match="assistant message \\(99\\) is not in the session view"):
        c.chat(4, echo=False).say("x")


# --------------------------------------------------------------------------- #
# Package surface
# --------------------------------------------------------------------------- #


def test_chat_names_are_exported_at_package_level():
    assert mlarena.ChatConversation is ChatConversation
    assert mlarena.ChatSessionNotFoundError is ChatSessionNotFoundError
    assert issubclass(mlarena.ChatSessionNotFoundError, mlarena.NotFoundError)
    assert mlarena.__version__ == "3.0.0"
