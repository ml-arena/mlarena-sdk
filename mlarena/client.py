"""ML Arena REST client.

Authentication is via Bearer token of the form `mlk_<scope>_<lookup>_<secret>`,
matching the backend's `auth_required` decorator (see backend/app/auth/decorators.py). The
older `key_id:key_pass` format that earlier README versions documented has been
retired — the canonical form is the full underscore-separated token.

Routes hit the canonical API blueprints under `/api/...`. There is no longer a
separate `/api/sdk/` namespace.
"""

import functools
import inspect
import io
import os
import tempfile
import time
import warnings
from pathlib import Path
from typing import Iterator
from urllib.parse import quote

import requests
import yaml

from mlarena.bundle import bundle_root, check_extract_target, extract_bundle, read_bundle
from mlarena.chat import ChatConversation
from mlarena.ci_auth import KIND_GITHUB_ACTIONS
from mlarena.exercise import format_attempt, source_of
from mlarena.exceptions import (
    AuthenticationError,
    ChallengeNotFoundError,
    ChatSessionNotFoundError,
    EnrollmentRequiredError,
    ExerciseAttemptNotFoundError,
    MaintenanceError,
    MLArenaError,
    NotFoundError,
    PermissionDeniedError,
    SubmissionError,
    SubmissionNotFoundError,
    WaitTimeoutError,
)


class _Unset:
    """Default of a keyword the caller did not pass — distinct from `None`,
    which `update_chat_settings` sends to clear a nullable limit."""

    def __repr__(self) -> str:
        return "UNSET"


_UNSET = _Unset()

# A token that expires is renewed this long before it does.
_TOKEN_RENEW_MARGIN_SEC = 60

# The 403 of every route that creates a challenge, for an account that has
# not accepted the terms of use: its `reason` and its sentence, as the
# backend writes them (`backend/app/services/terms.py`). The compositions
# that create a course before a challenge raise it themselves, before the
# course (`MLArenaClient._refuse_without_terms`).
TERMS_REQUIRED_REASON = "terms_required"
TERMS_REQUIRED = (
    "Accept the terms of use before you create a challenge: once per account, in the "
    "console when you create your first challenge, or with POST /api/auth/terms/accept."
)


# SDK 4.0.0 (score model, D5): the six legacy score keywords of
# `update_settings` are gone, with no alias. `metrics=[...]` replaces them all.
_REMOVED_SCORE_KWARGS = (
    "metric", "metric2", "is_elo_score", "metric_order",
    "frontend_precision", "metrics_schema",
)


def _refuses_removed_score_kwargs(fn):
    """Raise a TypeError naming `metrics` when a removed score keyword is
    passed, instead of Python's bare "unexpected keyword argument"."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        removed = [name for name in _REMOVED_SCORE_KWARGS if name in kwargs]
        if removed:
            raise TypeError(
                f"{fn.__name__}() no longer accepts {', '.join(removed)} "
                "(removed in mlarena-sdk 4.0.0): declare the leaderboard with "
                "metrics=[{key, label, source, agg, order, format, unit, "
                "precision, is_ranking, visible}, ...] — see the README "
                "'Metrics' section"
            )
        return fn(*args, **kwargs)
    return wrapper


class MLArenaClient:
    """Client for the ML Arena REST API.

    Pass the full bearer token returned by your Profile page (`mlk_user_...`
    or `mlk_author_...`; a `mlk_creator_...` or `mlk_teacher_...` key minted
    before 4.23.0 is an `author` key). The token's scope segment dictates
    which endpoints are reachable — a `user`-scope token cannot call an
    `author`-required route. A GitHub Actions run sends the
    short-lived `mlo_…` token of its OIDC exchange instead, which the backend
    reads like a key of the scope it was issued for (`mlarena.connect()`
    resolves either; `credentials` says which).
    """

    def __init__(self, token: str, base_url: str, credentials=None,
                 renew_credentials=None):
        self._token = token
        self._base_url = base_url.rstrip("/")
        # `mlarena.ci_auth.Credentials` when built by `mlarena.connect()`.
        self.credentials = credentials
        # Returns fresh credentials; set for a token that expires (the
        # GitHub Actions exchange), None for an API key.
        self._renew_credentials = renew_credentials
        self._last_submission_id = None
        self._last_challenge = None

    def _headers(self, *, json_body: bool = False) -> dict:
        if (self._renew_credentials is not None
                and self.credentials.expires_within(_TOKEN_RENEW_MARGIN_SEC)):
            self.credentials = self._renew_credentials()
            self._token = self.credentials.token
        h = {"Authorization": f"Bearer {self._token}"}
        if json_body:
            h["Content-Type"] = "application/json"
        return h

    def _url(self, path: str) -> str:
        return f"{self._base_url}/api{path}"

    def _handle_response(
        self,
        resp: requests.Response,
        *,
        not_found: type = ChallengeNotFoundError,
    ) -> requests.Response:
        """Turn a refusal into the exception that names it.

        `not_found` is the class a 404 raises: the submission methods pass
        `SubmissionNotFoundError`, so a missing submission no longer reports
        itself as a missing *challenge*. Both derive from `NotFoundError`.

        The message is always the server's own reason: every error body it
        produces carries it under `error` (see the app-wide HTTP handler in
        `backend/app/__init__.py`). It used to hold Werkzeug's status *name*
        — "Not Found", "Forbidden" — with the reason in a second key this
        client never read.
        """
        if resp.status_code == 401:
            raise _error(AuthenticationError, resp,
                         _safe_error(resp, "Invalid or missing API credentials."))
        if resp.status_code == 403:
            body = _json_body(resp)
            reason = body.get("reason") if body is not None else None
            if reason == TERMS_REQUIRED_REASON:
                # Creating a challenge before the terms of use were accepted
                # (`services.terms`): the server's sentence, then where to
                # read them and how this caller accepts.
                raise _error(PermissionDeniedError, resp, self._terms_required_message(
                    _safe_error(resp, "Access denied")))
            # Taking part in a course challenge without being enrolled in its
            # course (`challenge_access.ChallengeParticipant`).
            cls = (EnrollmentRequiredError if reason == "enrollment_required"
                   else PermissionDeniedError)
            raise _error(cls, resp, _safe_error(resp, "Access denied"))
        if resp.status_code == 404:
            raise _error(not_found, resp, _safe_error(resp, "Not found"))
        if resp.status_code == 503:
            body = _json_body(resp)
            # Only the backend's maintenance gate sets the flag; any other 503
            # (a data source proxy, an ingress) is left to the caller.
            if body is not None and body.get("maintenance_mode") is True:
                raise _error(MaintenanceError, resp,
                             _safe_error(resp, "Platform under maintenance"))
        return resp

    def _terms_required_message(self, sentence: str) -> str:
        """`sentence` (the server's `terms_required` refusal), then where the
        terms are read and how they are accepted. The SDK never accepts them
        for the caller: `accept_terms` is theirs to call, once they have
        read them. A GitHub Actions run cannot (the route refuses its
        token): the account's holder does."""
        terms_url = f"{self._base_url}/terms"
        if self.credentials is not None and self.credentials.kind == KIND_GITHUB_ACTIONS:
            return (f"{sentence} Read them at {terms_url} . A GitHub Actions run "
                    "cannot accept them for the account: its holder accepts once, "
                    "in the console or with an API key (client.accept_terms()).")
        return (f"{sentence} Read them at {terms_url} , then call "
                "client.accept_terms() (once per account, with an author key).")

    def _refuse_without_terms(self) -> None:
        """Raise what a route that creates a challenge answers to an account
        that has not accepted the terms of use (`PermissionDeniedError`, 403,
        `body["reason"] == "terms_required"`), read from `me()`.

        For the compositions that create a course and then a challenge in
        it: called first, so a refusal leaves no empty course behind. A
        credential the platform does not know is left to the first route
        (401)."""
        me = self.me()
        if me["is_authenticated"] and me["terms_accepted_at_ts"] is None:
            raise PermissionDeniedError(
                self._terms_required_message(TERMS_REQUIRED), status_code=403,
                body={"error": TERMS_REQUIRED, "reason": TERMS_REQUIRED_REASON},
            )

    def _request(self, method: str, url: str, **kwargs) -> requests.Response:
        """Single chokepoint for HTTP calls. Disables redirect-following by
        default so the bearer token cannot be replayed to an unintended host
        if the backend ever 3xx-s to one. Sets a default 30s timeout to match
        the prior per-call default.
        """
        kwargs.setdefault("allow_redirects", False)
        kwargs.setdefault("timeout", 30)
        return requests.request(method, url, **kwargs)

    # ---- Profile / account (any token scope; the token identifies the user) ----

    def profile(self) -> dict:
        """Return the authenticated user's profile.

        Mirrors ``GET /api/profile/`` — the same data the console's Profile
        page loads (username, email, bio, links, student identity, avatar_key,
        enrollments). Each enrollment is ``{"course": {"id", "name"},
        "enrolled_at_ts"}``. Works with any valid token scope
        since the token itself identifies the user.
        """
        resp = self._request("GET", self._url("/profile/"), headers=self._headers())
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "profile", resp)
        return resp.json()

    def update_profile(self, *,
                       username: str | None = None,
                       bio: str | None = None,
                       personal_website: str | None = None,
                       student_number: str | None = None,
                       student_email: str | None = None,
                       avatar_key: str | None = None) -> dict:
        """Patch the authenticated user's profile fields.

        Mirrors ``PUT /api/profile/update`` — the same call the console's
        Profile page makes. Only fields explicitly passed are sent; everything
        else is left untouched.

        ``avatar_key`` selects a generated avatar as ``"<style>:<seed>"`` (e.g.
        ``"bottts:hopper"``); pass ``""`` to clear it back to the initials
        default. Allowed styles match the console's avatar gallery — an unknown
        style, or a seed that is empty / over 64 chars / outside
        ``[A-Za-z0-9_-]``, is rejected with a 400.
        """
        body: dict = {}
        if username is not None:
            body["username"] = username
        if bio is not None:
            body["bio"] = bio
        if personal_website is not None:
            body["personal_website"] = personal_website
        if student_number is not None:
            body["student_number"] = student_number
        if student_email is not None:
            body["student_email"] = student_email
        if avatar_key is not None:
            body["avatar_key"] = avatar_key
        if not body:
            raise MLArenaError("update_profile requires at least one field")
        resp = self._request("PUT", self._url("/profile/update"),
            headers=self._headers(json_body=True),
            json=body,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "update_profile", resp)
        return resp.json()

    def me(self) -> dict:
        """Who this token belongs to, and what it may reach.

        Mirrors ``GET /api/auth/current_user`` — the same payload the console
        loads on every page. Keys: ``is_authenticated``, ``id``, ``username``,
        ``email``, ``avatar_key``, ``is_admin``, ``is_human_verified``,
        ``has_teacher_access`` (the teacher flag *or* admin *or* a
        course-assistant row), ``can_create_course``, ``max_courses_limit``
        and ``active_course_count`` (the limit behind it and the active, not
        archived, courses you own; an admin is exempt from the limit),
        ``github_login`` (the GitHub account
        linked to this one, None when there is none; it is linked from the
        console's Profile page, a browser flow) and ``terms_accepted_at_ts``
        (when the account accepted the terms of use, an ISO 8601 UTC string;
        None = not yet, and every method that creates a challenge is refused
        until `accept_terms`). Every key is always present. What you may do
        on one challenge is on that challenge's payloads (``role``,
        ``can_edit``): there is one kind of author.

        Note the route answers to any scope, but the *flags* describe the
        account, not the token: a ``mlk_user_…`` token on an admin account
        still reports ``is_admin: true`` while being refused admin routes.

        Listing or rotating API keys is deliberately **not** offered here:
        ``/api/auth/api_keys`` and ``/api/auth/api_keys/<scope>/rotate``
        answer 403 to a bearer caller, so that a leaked ``user`` key cannot
        mint an ``author`` one. Rotate keys from the console's Profile page.
        """
        resp = self._request("GET", self._url("/auth/current_user"),
                             headers=self._headers())
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "me", resp)
        return resp.json()

    def accept_terms(self) -> dict:
        """Accept the terms of use for your account: once, before your first
        challenge.

        Mirrors ``POST /api/auth/terms/accept`` (no body). Read the terms
        first: ``<base_url>/terms`` (https://ml-arena.com/terms). Until an
        account has accepted them, every method that creates a challenge is
        refused with a `PermissionDeniedError` (403) whose
        ``.body["reason"]`` is ``"terms_required"``: `add_course_challenge`
        (a build and a copy), `create_challenge` (a challenge with no course),
        `copy_module_into_course` and `duplicate_course` when a challenge
        would be copied, and the compositions over them
        (`create_challenge_for_class`, `use_example_challenge`,
        `start_from_example_course`). Nothing is created then. Creating a
        course, a module or a lesson needs no terms. Nobody is exempt, an
        admin included, and each person accepts for themselves: a course's
        assistant who creates a challenge in it has accepted, whether or not
        its owner has.

        Returns ``{"terms_accepted_at_ts": "2026-11-26T10:15:00Z"}``. Once
        per account: a second call changes nothing and returns the first
        date, the one `me()` serves under the same key. The SDK never calls
        this for you.

        Who may call it: the account's holder, with an ``author`` key. A
        ``user`` key is refused (403), and so is the token of a GitHub
        Actions run (403: a workflow acts for the account, accepting is its
        holder's own act, in the console or with a key).
        """
        return self._json_call("POST", "/auth/terms/accept", error_label="accept_terms")

    # ---- GitHub: trusted repositories (any token scope) ----
    #
    # A GitHub Actions run authenticates as the account its repository is
    # linked to. A repository owned by the linked GitHub account
    # (`me()["github_login"]`) is trusted as it is; any other one (an
    # organisation's, a classmate's) must be listed here first.

    def trusted_repositories(self) -> list:
        """The repositories whose GitHub Actions runs may act as you.

        Mirrors ``GET /api/profile/github/trusted_repositories``. Each row:
        ``id``, ``repository`` (``owner/name``), ``repository_id`` (GitHub's
        numeric id, None until the repository's first run), ``created_at_ts``
        and ``last_used_at_ts``.
        """
        resp = self._request("GET",
                             self._url("/profile/github/trusted_repositories"),
                             headers=self._headers())
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "trusted_repositories", resp)
        return resp.json()

    def add_trusted_repository(self, repository: str) -> dict:
        """Trust a repository you do not own: its Actions runs may act as you.

        Mirrors ``POST /api/profile/github/trusted_repositories``
        ``{"repository": "owner/name"}``. Returns the new row (see
        `trusted_repositories`).
        """
        resp = self._request("POST",
                             self._url("/profile/github/trusted_repositories"),
                             headers=self._headers(json_body=True),
                             json={"repository": repository})
        self._handle_response(resp)
        if resp.status_code not in (200, 201):
            raise _failed(MLArenaError, "add_trusted_repository", resp)
        return resp.json()

    def remove_trusted_repository(self, trusted_repository_id: int) -> dict:
        """Stop trusting a repository (the row's ``id``).

        Mirrors ``DELETE /api/profile/github/trusted_repositories/{id}``.
        """
        resp = self._request(
            "DELETE",
            self._url(f"/profile/github/trusted_repositories/{trusted_repository_id}"),
            headers=self._headers())
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "remove_trusted_repository", resp)
        return resp.json()

    # ---- Teams (per-challenge; any token scope) ----
    #
    # A team is a participant action, so every route below is `user`-scoped:
    # any valid token reaches it. One team per challenge per user; the team a
    # submission is attributed to is the one these methods manage.
    #
    # Shapes mirror the response models in backend/app/views/_schemas.py. A
    # team carries `id`, `challenge_id`, `name`,
    # `created_at_ts`, `members` (each: `id` — the membership row id —,
    # `user_id`, `username`, `avatar_key`, `role`) and `my_role`, your own
    # role on it ("leader", "member" or None). An invitation carries `id`,
    # `team_id`, `team_name`, `user_id`, `username`, `sender_id`,
    # `sender_username`, `status` and `created_at_ts`. A team also carries
    # `datagouv_reuse`: the data.gouv.fr reuse it linked (a dict, see
    # `link_datagouv_reuse`) or None.

    def challenge_team(self, challenge_id: int) -> dict | None:
        """Your team for a challenge, or None when you are on no team for it.

        Mirrors ``GET /api/teams/challenge/{id}/team``, which answers 200
        with ``null`` when you have no team there (a normal answer; until
        4.1.0 it was a 404 raised as ``ChallengeNotFoundError``). ``my_role``
        is never None on a returned team. The 404 now means a challenge you
        cannot see, raised as ``ChallengeNotFoundError``.
        """
        resp = self._request("GET", self._url(f"/teams/challenge/{challenge_id}/team"),
                             headers=self._headers())
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "challenge_team", resp)
        return resp.json()

    def create_team(self, challenge_id: int, name: str) -> dict:
        """Create a team for a challenge; you become its leader.

        Mirrors ``POST /api/teams/challenge/{id}/team``. 400 if you are
        already on a team for that challenge.
        """
        resp = self._request("POST", self._url(f"/teams/challenge/{challenge_id}/team"),
                             headers=self._headers(json_body=True),
                             json={"name": name})
        self._handle_response(resp)
        if resp.status_code != 201:
            raise _failed(MLArenaError, "create_team", resp)
        return resp.json()

    def update_team(self, team_id: int, name: str) -> dict:
        """Rename a team (leader only).

        Mirrors ``PUT /api/teams/{team_id}``.
        """
        resp = self._request("PUT", self._url(f"/teams/{team_id}"),
                             headers=self._headers(json_body=True),
                             json={"name": name})
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "update_team", resp)
        return resp.json()

    def delete_team(self, team_id: int) -> dict:
        """Delete a team, its members and its invitations (leader only).

        Mirrors ``DELETE /api/teams/{team_id}``.
        """
        resp = self._request("DELETE", self._url(f"/teams/{team_id}"),
                             headers=self._headers())
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "delete_team", resp)
        return resp.json()

    def leave_team(self, team_id: int) -> dict:
        """Leave a team you are a plain member of.

        Mirrors ``POST /api/teams/{team_id}/leave``. A leader cannot leave
        (400) — they delete the team instead.
        """
        resp = self._request("POST", self._url(f"/teams/{team_id}/leave"),
                             headers=self._headers())
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "leave_team", resp)
        return resp.json()

    def remove_team_member(self, team_id: int, member_id: int) -> dict:
        """Remove a member from your team (leader only).

        Mirrors ``DELETE /api/teams/{team_id}/members/{member_id}``.
        ``member_id`` is the membership row id — ``member["id"]`` of a team's
        ``members``, not ``member["user_id"]``.
        """
        resp = self._request("DELETE", self._url(f"/teams/{team_id}/members/{member_id}"),
                             headers=self._headers())
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "remove_team_member", resp)
        return resp.json()

    def search_teams(self, challenge_id: int, q: str) -> dict:
        """Users you could invite: up to 5 whose username contains ``q``.

        Mirrors ``GET /api/teams/search``. Scoped to one challenge on purpose
        — only users who already have a submission on it are listed, so this
        is not an open username-enumeration oracle. Returns
        ``{"users": [{"id", "username", "avatar_key"}, ...]}``.
        """
        resp = self._request("GET", self._url("/teams/search"),
                             params={"q": q, "challenge_id": challenge_id},
                             headers=self._headers())
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "search_teams", resp)
        return resp.json()

    def invite_to_team(self, team_id: int, user_id: int) -> dict:
        """Invite a user onto your team (leader only).

        Mirrors ``POST /api/teams/{team_id}/invitations`` and returns the
        created invitation. 409 if the recipient is already on a team for the
        challenge, 400 if an invitation is already pending for them.
        """
        resp = self._request("POST", self._url(f"/teams/{team_id}/invitations"),
                             headers=self._headers(json_body=True),
                             json={"user_id": user_id})
        self._handle_response(resp)
        if resp.status_code != 201:
            raise _failed(MLArenaError, "invite_to_team", resp)
        return resp.json()

    def pending_invitations(self, team_id: int) -> list:
        """The invitations your team has out (leader only).

        Mirrors ``GET /api/teams/{team_id}/invitations/pending``.
        """
        resp = self._request("GET", self._url(f"/teams/{team_id}/invitations/pending"),
                             headers=self._headers())
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "pending_invitations", resp)
        return resp.json()

    def received_invitations(self) -> list:
        """The pending team invitations addressed to you.

        Mirrors ``GET /api/teams/invitations/received``.
        """
        resp = self._request("GET", self._url("/teams/invitations/received"),
                             headers=self._headers())
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "received_invitations", resp)
        return resp.json()

    def respond_to_invitation(self, invitation_id: int, accept: bool) -> dict:
        """Accept or decline an invitation addressed to you.

        Mirrors ``POST /api/teams/invitations/{id}/respond`` and returns the
        settled invitation (its ``status`` is now "accepted" or "declined").

        Accepting is refused with 409 when the team would go over the
        challenge's active-submission limit; that body carries
        ``combined_active``, ``max_active_submissions`` and ``excess`` — read
        them off ``err.body``.
        """
        resp = self._request("POST", self._url(f"/teams/invitations/{invitation_id}/respond"),
                             headers=self._headers(json_body=True),
                             json={"accept": accept})
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "respond_to_invitation", resp)
        return resp.json()

    def cancel_invitation(self, invitation_id: int) -> dict:
        """Cancel a pending invitation your team sent (leader only).

        Mirrors ``DELETE /api/teams/invitations/{id}``.
        """
        resp = self._request("DELETE", self._url(f"/teams/invitations/{invitation_id}"),
                             headers=self._headers())
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "cancel_invitation", resp)
        return resp.json()

    def link_datagouv_reuse(self, team_id: int, url: str) -> dict:
        """Link your team's data.gouv.fr reuse to the challenge (leader only).

        Mirrors ``PUT /api/teams/{team_id}/datagouv_reuse``. ``url`` is the
        reuse's data.gouv.fr page (``https://www.data.gouv.fr/.../reuses/<slug>/``).
        Replaces the team's previous link. Returns the link: ``datagouv_reuse_id``,
        ``title``, ``page_url``, ``owner_name``, ``status`` (``"published"`` |
        ``"validated"`` | ``"unavailable"``), ``linked_at_ts`` and
        ``checked_at_ts``.

        400 when the challenge has no data.gouv.fr partnership, the URL is not
        a data.gouv.fr reuse, or the reuse is not public / uses no dataset;
        403 (``PermissionDeniedError``) when you do not lead the team; 409
        when another team already linked that reuse; 502 when data.gouv.fr
        does not answer.
        """
        resp = self._request("PUT", self._url(f"/teams/{team_id}/datagouv_reuse"),
                             headers=self._headers(json_body=True),
                             json={"url": url})
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "link_datagouv_reuse", resp)
        return resp.json()

    def unlink_datagouv_reuse(self, team_id: int) -> dict:
        """Remove your team's data.gouv.fr reuse link (leader only).

        Mirrors ``DELETE /api/teams/{team_id}/datagouv_reuse``. 404 when the
        team has no link.
        """
        resp = self._request("DELETE", self._url(f"/teams/{team_id}/datagouv_reuse"),
                             headers=self._headers())
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "unlink_datagouv_reuse", resp)
        return resp.json()

    # ---- Challenges (public read; create/update via author scope) ----

    def challenges(
        self,
        *,
        q: str | None = None,
        tags: list[str] | str | None = None,
        status: str = "started",
        page: int | None = None,
        per_page: int | None = None,
    ):
        """List the challenges with no course. Public; no
        auth required.

        A challenge of a course is never listed here: it is reached from its
        course (`list_courses`, `course(slug)`, `workspace()` for its staff).
        Anonymous callers get the public ones; a signed-in caller also
        gets the ones they own or assist on.

        With no args, returns all matching challenges as a DataFrame
        (auto-paginates internally). Pass ``page``/``per_page`` to fetch a
        single page; pass ``q``/``tags``/``status`` to filter server-side.
        ``status`` is ``"started"`` (the open challenges, the default),
        ``"ended"`` (the past ones, most recently ended first) or ``"all"``
        (every status) — the route knows no other value and answers 400.

        Each row carries ``id``, ``name``, ``description``, ``miniature``
        (the image route, None when the challenge has no miniature),
        ``status`` (``"draft"`` | ``"started"`` | ``"stopped"`` |
        ``"ended"``), ``end_date_ts`` (the scheduled end while started, the
        date it ended once ended; None = open-ended),
        ``is_award_points``, ``number_of_agents``, ``tags``,
        ``statistics`` (``running_submissions`` and
        ``unique_participants``: a team counts once, a user in no team
        once), ``kernel_version`` and ``kind_label`` (the kind and
        its one label, ``"Python agent"`` | ``"Upload a file"`` |
        ``"Chat with an AI agent"`` | ``"Exercises"``; None without a
        configuration) and ``my_rank``
        (your participant's rank on the challenge's board; None when
        anonymous or with nothing on the board). Call
        ``challenge(id)`` for the machine and the limits.
        """
        base_params: dict = {}
        if q:
            base_params["q"] = q
        if status:
            base_params["status"] = status
        if tags:
            base_params["tags"] = ",".join(tags) if isinstance(tags, list) else tags

        # Public route, but it answers differently to the caller it can
        # identify (non-public challenges they own, assist on or reach through
        # a course; every challenge for an admin): send the token.
        if page is not None or per_page is not None:
            params = {**base_params, "page": page or 1, "per_page": per_page or 24}
            resp = self._request("GET", self._url("/challenges/"), params=params,
                                 headers=self._headers(), timeout=30)
            self._handle_response(resp)
            if resp.status_code != 200:
                raise _failed(MLArenaError, "challenges", resp)
            return _to_dataframe(resp.json().get("items", []))

        items: list = []
        page_n = 1
        per_page_n = 100
        while True:
            params = {**base_params, "page": page_n, "per_page": per_page_n}
            resp = self._request("GET", self._url("/challenges/"), params=params,
                                 headers=self._headers(), timeout=30)
            self._handle_response(resp)
            if resp.status_code != 200:
                raise _failed(MLArenaError, "challenges", resp)
            body = resp.json()
            items.extend(body.get("items", []))
            if not body.get("metadata", {}).get("has_next"):
                break
            page_n += 1
        return _to_dataframe(items)

    def challenge(self, challenge_id: int) -> dict:
        """Fetch a single challenge's detail (settings, limits, status).

        Useful for client-side preflight against `max_upload_size_bytes` and
        `max_upload_files` before calling `upload_submission_file` / `submit`.
        Mirrors `GET /api/challenges/{id}` (`challenges.py`, `get_challenge`).

        Includes a `machine` sub-object — `{id, name, runtime,
        vm_health_ok, vm_health_checked_at_ts}` — naming where the challenge
        runs; `runtime` is `"k8s"` or `"vm"`, and the two health keys are set
        only on a `vm` machine (polled every minute by simulationmanager).
        `vm_health_ok=False` means the GPU VM is currently unreachable and
        submissions will queue rather than run. The agent sizing a submission
        runs under is `number_of_agents`, `agent_cpu_limit`,
        `agent_memory_limit`, `gpu_memory_limit` and `has_gpu`.

        Where it sits: `course_id`, `module_id` and `pass_threshold` (the
        course's bar on its score, None = no bar), all None for a
        challenge with no course; `course` is the course a course challenge belongs to
        (`{id, slug, name, visibility, single_challenge_id,
        example_listed_at_ts, enrollment_state, is_enrolled, join_code}`,
        None for a challenge with no course): what a reader who is not enrolled needs to
        enroll (`enroll_in_course(join_code)`) while `enrollment_state` is
        `open` (`open` | `archived` | `ended`). `single_challenge_id` is
        this challenge's id when it is the course's only content (the course
        is then shown as the challenge), else None; `example_listed_at_ts`
        is when the course was listed as an example (None = not an example).
        `copied_from_challenge_id` is the challenge it was copied from (None
        when it was not); `copied_from_challenge` (`{id, name, slug}`) names
        that source when it is a public challenge with no course ("Based on
        the challenge X"), None otherwise.

        You on it: `role` (`"owner"`, `"assistant"` or None; an admin who is
        neither reads None), `can_edit` (you may open its workspace: its
        staff, an admin), `can_participate` (you may submit, deploy,
        chat, check an exercise or team up: on a challenge with no course any signed-in
        user who sees it, on a course challenge the course's staff and the
        users enrolled in it; a refusal raises `EnrollmentRequiredError`)
        and `can_copy` (you may copy it into a course of yours,
        `use_example_challenge` / `add_course_challenge(
        copy_from_challenge_id=…)`: a started, stopped or ended challenge
        in a published module of an example course for any signed-in user,
        any challenge for its staff and for an admin; False without a
        credential. Read this flag, not the course's `example_listed_at_ts`:
        a draft of an example course is not copied. Where it is False the
        copy methods answer 404).

        `kind_label` is the kind's one label (`"Python agent"` |
        `"Upload a file"` | `"Chat with an AI agent"` | `"Exercises"`).
        `agent_template` (the code
        a participant starts `agent.py` from) is served for code challenges
        only, None for a file or chat challenge. `submission_layout` is the
        CSV a file challenge's submission must follow — `columns`,
        `separator`, `id_column`, `target_column`, `row_count` (None when
        unknown) — read from the challenge's ground truth; None for any other
        kind, a non-CSV submission file, or before a `y_test.csv` is parsed.
        `miniature` is None when the challenge has no miniature.

        `runtime_options` is the set of agent runtime images the challenge
        offers (ordered by id; `[]` for a file or chat challenge) and
        `default_agent_runtime_id` the one a new submission gets (None for a
        file or chat challenge) — both written by the challenge's staff
        (`update_settings(agent_runtime_ids=…, default_agent_runtime_id=…)`).
        """
        # Public route, but a non-public course challenge is only visible to a
        # caller the backend can identify: send the token.
        resp = self._request("GET", self._url(f"/challenges/{challenge_id}"),
                             headers=self._headers(), timeout=30)
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "challenge", resp)
        return resp.json()

    def challenge_by_slug(self, slug: str) -> dict:
        """`challenge()`, looked up by the challenge's URL slug: the
        `<slug>` of `https://ml-arena.com/challenges/<slug>`.

        Returns the same detail (its `id` is what every other method takes).
        A hidden challenge raises like a missing one. Mirrors
        `GET /api/challenges/slug/{slug}` (`challenges.py`,
        `get_challenge_by_slug`). The slug follows the challenge's name: a
        rename (allowed only while the challenge is not started) changes it.
        """
        resp = self._request("GET", self._url(f"/challenges/slug/{quote(slug, safe='')}"),
                             headers=self._headers(), timeout=30)
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "challenge_by_slug", resp)
        return resp.json()

    def machines(self, kernel_version: str | None = None) -> list:
        """The machines a challenge may run on, with their queues.

        Mirrors `GET /api/machines[?kernel_version=]` (`machines.py`,
        `list_machines`) — the list the console's Runtime section offers.
        Each entry is a `MachineOut`: `id`, `name`, `runtime` (`"k8s"` |
        `"vm"`), `node_pool` (`"autoscaling"` | `"large-autoscaling"`;
        `None` on a vm machine), `vm_health_ok`,
        `vm_health_checked_at_ts`, `gpu_device_ids`, `gpu_count`,
        `image_pull_policy` (`"IfNotPresent"` | `"Always"` | `"Never"`),
        `job_cpu_max`, `job_memory_max`,
        `job_timeout_max_sec` (the longest `simulation_timeout_sec` a
        challenge on it may ask for), `max_concurrent_jobs`, the replay caps
        of a run on it (`replay_frames_max`, `replay_video_second_max`,
        `replay_frames_bytes_max`, `replay_video_bytes_max`,
        `replay_image_bytes_max`), `is_enabled`,
        `created_at_ts`, `kernels`,
        its queues (`MachineKernelOut`: `id`, `machine_id`, `kernel_version`,
        `docker_image_worker_envagent_id`, `worker_replicas`, `is_public`,
        `is_default`, `created_at_ts`), `energy_profile` (the coefficients a
        run's footprint is computed with: `id`, `valid_from_ts`,
        `cpu_watts_per_vcpu`, `ram_watts_per_gib`, `pue`,
        `carbon_intensity_g_per_kwh`, `source_note`; None on a machine whose
        runs are not metered) and `rate`, what time on the machine costs
        (`MachineRateOut`: `id`, `valid_from_ts`, `resource` — `"cpu"` |
        `"gpu"`, the balance of the challenge's owner it is taken from,
        `credits()` —, `credits_per_minute`, a four-decimal string like
        every amount of credits, and `source_note`; None on a machine
        without a rate: nothing that runs there is charged).

        A non-admin sees only the machines with a queue visible to them
        (public, or granted to them) and only those queues; an admin sees
        every machine. `kernel_version` keeps the queues of that kernel —
        pass a challenge's `creator_challenge(id)["configuration"]
        ["kernel_version"]` to list the machines `update_settings(
        machine_id=…)` accepts for it. An unknown kernel answers 400.

        Creating, editing and granting machines is admin cluster-ops,
        console-only (no SDK method, by decision).

        Requires an `author`-scope token.
        """
        params = {} if kernel_version is None else {"kernel_version": kernel_version}
        resp = self._request("GET", self._url("/machines/"),
                             params=params, headers=self._headers(), timeout=30)
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "machines", resp)
        return resp.json()

    def workspace(self) -> dict:
        """Everything you author, in one read: the Studio home's payload.

        Mirrors `GET /api/author/workspace`. Needs an `author` key (a
        `creator` or `teacher` key minted before 4.23.0 is one). A `user` key
        is refused (`PermissionDeniedError`). An account that
        owns and assists nothing gets empty lists and its quotas. Returns
        `{"scope", "challenges", "courses", "attention", "attention_overflow",
        "quotas"}`; `scope` is `"mine"`: everyone's challenges and courses
        (`scope=all`) is an admin's listing in the console, which no key
        reads.

        `challenges`: the challenges you own or assist on (archived ones
        included), your own first, each group newest first. A row is
        `{"challenge", "counters", "created_at_ts", "archived_at_ts",
        "start_blockers", "start_blockers_error"}` (`archived_at_ts` is null
        unless `archive_challenge` stamped it):
        `challenge` and `counters` as `challenge_overview` returns them
        (`challenge["role"]` is `"owner"` or `"assistant"`;
        `challenge["can_edit"]` says whether you may open its workspace;
        `challenge["kernel_version"]` and `["kind_label"]` are None when the
        platform cannot tell the challenge's kind). `start_blockers` is
        served for a `draft`: every reason `start_challenge` would refuse
        now (the list `creator_challenge()` serves; `[]` = ready), else
        None; `start_blockers_error` says why it is unknown when the gate
        could not answer for the draft (its files could not be read, it is
        incomplete, or computing the list failed).

        `courses`: the courses you own or assist on, archived ones included.
        A row carries `id`, `name`, `code`, `slug`, `visibility`,
        `start_date`, `end_date`, `archived_at_ts`, `single_challenge_id`
        (the id of the course's only challenge when the course holds exactly
        one module, one challenge and no lesson, else None: the course is
        then shown as that challenge, one of `challenges` and of the
        challenge rows above), `enrollment_state`
        (`open`, `archived` or `ended`: whether the course takes a new
        enrollment now), `join_code`, `role`, `can_edit` (you may open its
        workspace: true on every row), `owner_username`, `enrollment_count`,
        `active_enrollment_count_7d`
        (enrollments active in the last 7 days), `last_active_at_ts` (the
        last time a student opened a lesson, deployed, sent a chat message or
        attempted an exercise in the course), `challenges` (the course's
        challenges as references, entries hidden from students included, in
        the course's order), `hidden_challenge_ids`, and
        `enrolled_without_submission`: enrolled users who handed in nothing
        on the challenges students see, served while the course is not
        archived and ends within 7 days, else None.

        `attention`: what needs you, about the rows you have a relation to;
        each item has a `kind`. Listed first, the credits: `credits_used`
        (a balance of yours reached an alert level: `resource`, `alert_pct`,
        `pct_used`, `balance`, `full_grant` and `unattended`, the
        challenges whose charges no participant caused, each
        `{challenge_id, challenge_name, amount, share_pct}`) and
        `scheduled_work_paused` (a started challenge whose owner cannot
        spend on `resource`: its matchmaking and data-feed batches are
        skipped; `challenge_id`, `challenge_name`, `owner_username`,
        `resource`). Then `deploy_failed_submissions` (a `started`
        challenge with submissions whose deploy failed: the count and its
        split by owner), `draft_blocked` (a draft whose `start_blockers`
        include a platform conflict — `platform_conflict` true: its machine
        or queue, its size against the machine's ceilings, its image rows,
        its owner's limit of started challenges or exercise sets, its
        owner's credits;
        the item's `start_blockers` are those conflicts only, the row has
        the full list; a draft that only lacks authoring raises nothing),
        `draft_unreadable` (the gate could not run: the row's
        `start_blockers_error`), `course_ending` (a course ending within 7 days
        with `enrolled_without_submission` > 0), `course_challenge_closed`
        (a challenge students see in a course of yours that is still
        running, not your own, that is `stopped` or `ended`). Items name
        their object by id (`challenge_id`, `course_id`) and name. At most
        20 items of each kind are listed, in the rows' order;
        `attention_overflow` is `{kind: how many more}` (0 when all are
        there). The rows carry every fact.

        `quotas`: `owned_challenge_count` (the challenges you own that are
        not archived; no limit on challenges yet),
        `max_courses_limit`, `active_course_count`, `can_create_course`.
        """
        return self._json_call("GET", "/author/workspace", error_label="workspace")

    def creator_challenges(self) -> list:
        """Deprecated: use `workspace()["challenges"]`.

        The challenges you own or assist on, built from `workspace()`: each
        item is the row's challenge reference (`id`, `name`, `slug`,
        `status`, `end_date_ts`, `is_public`, `kernel_version`, `kind_label`,
        `user_id`, `owner_username`, `role`, `can_edit`, `can_copy`,
        `statistics`) plus `created_at_ts`. The list route this method called
        (`GET /api/creator_challenge/challenges`) is gone, and with it
        `description`, `has_configuration`, `has_environment` and
        `machine_name` (read them from `creator_challenge(id)`). An admin
        gets their own challenges; everyone's is the console's. Needs an
        `author` key.
        """
        warnings.warn(
            "MLArenaClient.creator_challenges() is deprecated, "
            "use .workspace()[\"challenges\"] instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return [
            {**row["challenge"], "created_at_ts": row["created_at_ts"]}
            for row in self.workspace()["challenges"]
        ]

    def creator_challenge(self, challenge_id: int) -> dict:
        """The challenge as its editor sees it (author scope).

        The challenge's staff read it: its owner and assistants (for a course
        challenge, the course's owner and assistants), an admin.

        Mirrors `GET /api/creator_challenge/challenge/{id}`
        (`lifecycle.py`, `get_creator_challenge`). Returns the challenge plus
        its three sibling rows as three objects, each under its own column
        names: `configuration` (`machine_id` and the runtime sizing, the env
        runtime image, `number_of_agents`, step deadlines, upload limits,
        `submission_filename`, the GitHub sync), `evaluation`
        (`metrics` — the MetricSpec declaration —, `window_days`, the ELO
        parameters, the episode budget brackets) and `environment` (`id`,
        `challenge_id`, `created_at_ts` and the latest benchmark run's
        `benchmark_simulation_result_id`; the env folder's files are
        `list_env_files`). The configuration's
        `kernel_version` is the env runtime's kernel. Also `machine_name`,
        `machine` (`{id, name, runtime, vm_health_ok,
        vm_health_checked_at_ts}`), `role` — what you are to the challenge:
        `"owner"`, `"assistant"`, or None for an admin who is neither —,
        `has_owner_rights` (whether you may do what only its owner may, the
        assistant roster: the owner and an admin), `owner_username`, whose
        challenge it is, and `kind_label`, the kind's name as every challenge
        reference carries it.

        `config` is the challenge config as one flat document (the platform's
        modelmanager/PROCESS.md "Challenge config"): `values` — every config key the kind
        accepts, as stored (never `llm_api_key`) —, `origin` — per key,
        `"file"` (the applied `challenge.toml` holds it: `update_settings`
        refuses it), `"console"` (set by `update_settings` or the console) or
        `"default"` —, `file_lines` (the line of each file-owned key),
        `file_applied_at_ts` (None without a file) and `file_in_sync` (False
        when the env folder's `challenge.toml` is not the text that was
        applied). `config_fields()` describes every key.

        `start_blockers` lists every reason `start_challenge` would refuse
        now, each `{code, message, fix_by, can_fix, place,
        platform_conflict}`: `fix_by` is who fixes it (`"author"`: any editor
        of the challenge; `"admin"`), `can_fix` whether you can, `place` the
        part of the challenge workspace where (`files`, `benchmark`,
        `chat_agent`, `scoring_settings` under Scoring; `advanced`,
        `platform` under Settings; None when no part does) and
        `platform_conflict` whether it is a conflict with the platform's
        state rather than authoring still to do (`start_challenge` answers
        409 when the first blocker is one, 400 otherwise).
        Codes, in the order the checks run (`StartBlockerCode`):
        `env_missing`, `configuration_missing`,
        `image_inconsistent`, `config_not_applied`, `required_file_missing`,
        `runtime` — the size, the run timeout, the agents per run or where
        the challenge is placed —,
        `env_invalid`, `chat_missing`, `exercise_missing` (an exercise set
        without its row: an admin's repair), `started_sets_cap` (the owner
        already has 15 exercise sets started, the limit: stop one),
        `out_of_credits` (the challenge's owner has no credits left on the
        balance its machine draws from, `credits()`: more come from an
        admin; never while credits are not enforced),
        `started_challenges_cap` (a course challenge that is not an
        exercise set: its owner already has `max_started_challenges_limit`
        of them started, 10 unless an admin changed it; "Limit of started
        challenges reached: this challenge's owner has 10 started and may
        have 10 (exercise sets are not counted). Stop one to start
        another."; challenges with no course and an admin's are not limited),
        `llm_not_configured`,
        `ground_truth_missing`, `benchmark_missing`, `benchmark_unscored`;
        `[]` means ready, None while started, and None with
        `start_blockers_error` set (a sentence) when the challenge's files
        could not be read, it is incomplete or computing the list failed, so
        the checklist is unknown. A file you got wrong
        is a blocker, not such a fault: an `env.py` that is not UTF-8 text is
        `env_invalid`, a `challenge.toml` that is not is `config_not_applied`.
        A `runtime` message is worded
        for the caller: what to change (or "ask an admin") for an author,
        the machine, its limits and the rule that failed for an admin key. `start_recommendations` lists,
        the same way, what participants would miss if it started now — advice
        `start_challenge` never checks: `description_missing` (no card
        blurb), `overview_missing` (no overview, or only whitespace),
        `cover_missing` (no cover image); `[]` when nothing is missing, None
        while started. `running_editable_fields` names
        what stays editable once started, for you and this kind (among
        `description`, `is_public`, `metrics`, a code challenge's
        `is_stop_after_deployment` and `render_delay_second`, the upload and
        active-submission limits, the chat LLM keys and limits, and
        `machine_id` for an admin); anything else needs stop → edit →
        restart. `miniature` is the image route, None when none was
        uploaded.

        Whoever reads this payload may also write the challenge's code and
        files (env files, bundle, scoring form, datasets, benchmark files),
        whatever its kind: every member of a challenge's staff is its
        author.

        Unlike `challenge()` this works on your own hidden challenges and
        shows the authoring state; `challenge()` is the participant view.

        Requires an `author`-scope token and ownership (or admin) of the
        challenge.
        """
        return self._creator_get(challenge_id, "", "creator_challenge")

    def config_fields(self) -> list:
        """Every challenge config key, in display order (public read).

        Mirrors `GET /api/creator_challenge/config_fields` (`config.py`) — the
        registry the console's Config tab is generated from. Each entry:
        `key` (the column, `update_settings` keyword and `challenge.toml`
        key), `group` (`evaluation` | `participation` | `data_feed` |
        `resources` | `chat` | `exercise` | `partner`), `writer`
        (`creator`: the challenge's staff; `admin`: admins only),
        `capability` (the `available_kinds()`
        capability a kind needs for it, None = every kind),
        `fixed_capability` (the capability of the kinds that hold the key
        fixed: served read-only at the kind's value and refused on every
        write, None = no kind; `"chat"` on `metrics` and `window_days`),
        `running_editable` (still writable once started), `in_file` (may be
        written in `challenge.toml`) and `help`.
        """
        resp = self._request("GET", self._url("/creator_challenge/config_fields"),
                             headers=self._headers(), timeout=30)
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "config_fields", resp)
        return resp.json()["fields"]

    def config_schema(self) -> dict:
        """The JSON Schema of `challenge.toml` (public read).

        Mirrors `GET /api/creator_challenge/config_schema.json`: the
        `#:schema` line an exported file starts with, for editor completion
        (Taplo / VS Code Even Better TOML). Keys a file may hold only.
        """
        resp = self._request("GET", self._url("/creator_challenge/config_schema.json"),
                             headers=self._headers(), timeout=30)
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "config_schema", resp)
        return resp.json()

    def export_config(self, challenge_id: int, path: str | None = None) -> str:
        """The challenge's resolved config as a `challenge.toml` (author scope).

        Mirrors `GET /api/creator_challenge/challenge/{id}/config.toml` — the
        console's *Export challenge.toml*. Every key the kind accepts and a
        file may hold: the file's and the console's as values, the defaults
        commented out with their help. Uploading it back
        (`upload_env_file(cid, "challenge.toml")`) makes the file own every
        written key. Returns the text; writes it to `path` when given.
        """
        resp = self._request("GET",
            self._url(f"/creator_challenge/challenge/{challenge_id}/config.toml"),
            headers=self._headers(),
            timeout=30,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "export_config", resp)
        text = resp.text
        if path is not None:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
        return text

    def available_kinds(self) -> list:
        """The challenge kinds `create_challenge` accepts (author scope).

        Mirrors `GET /api/creator_challenge/available_kinds` (`kinds.py`).
        Each entry carries `kernel_version` (what `create_challenge` takes),
        `label`, `description`, `protocol`, `isolation`, `has_machine`,
        `capabilities` (`agent_template`, `benchmark`,
        `env_structural_check`, `runs`, `chat`, `submission_upload`,
        `submission_filename`, `elo`, `episode_budget_brackets`,
        `run_limits`, `agent_containers`, `multi_agent` — the
        tabs, checks and `update_settings` keywords the kind has),
        `runtime_defaults` (the nine sizing keys a new challenge of the kind
        starts with, as K8s quantity strings), `runtime_presets`
        (`{"small" | "medium" | "large": {sizing key: quantity}}`) and
        `minimal_loop_snippet`. `has_machine` is False when the deployment
        has no default machine queue (or no base env runtime) for that
        kernel, and creating one then answers 503.
        """
        resp = self._request("GET",
            self._url("/creator_challenge/available_kinds"),
            headers=self._headers(),
            timeout=30,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "available_kinds", resp)
        return resp.json()

    def datasets(self, challenge_id: int) -> dict:
        """List the public datasets attached to a challenge.

        Mirrors `GET /api/challenges/{id}/datasets` (`challenges.py`,
        `get_challenge_datasets`, `@auth_required('user')`). Returns
        ``{"datasets": [{"id", "label", "description",
        "files": [{"id", "label", "download_url", "file_size_bytes", ...}]}]}``
        where each ``download_url`` is a short-lived (60 min) signed URL the
        caller can GET directly. This is how a participant pulls the training /
        test data an author published for a file challenge. Requires any
        valid token scope.
        """
        resp = self._request("GET",
            self._url(f"/challenges/{challenge_id}/datasets"),
            headers=self._headers(),
            timeout=30,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "datasets", resp)
        return resp.json()

    def challenge_footprint(self, challenge_id: int) -> dict:
        """The challenge's evaluation footprint: what its runs consumed on
        the platform's machines.

        Mirrors `GET /api/challenges/{id}/footprint` — the console challenge
        page's footprint card; open to anyone who can see the challenge.
        Returns `ChallengeFootprint` `{"last_30d", "since_start"}`, each
        `{"runs", "metered_runs", "agent_energy_wh", "env_energy_wh",
        "agent_co2e_g", "env_co2e_g"}`: the sums over the challenge's runs
        of the window, agents and env apart (energy in Wh, CO₂e in g;
        `mlarena.format_metric(v, {"format": "energy", "precision": 2,
        "unit": None})` shows one, `"mass"` for CO₂e). Only `metered_runs` of the `runs` count in the sums (a
        run before the platform measured it, or on an unmetered machine,
        has no figure). A challenge you cannot see raises
        `ChallengeNotFoundError` (404).
        """
        resp = self._request("GET",
            self._url(f"/challenges/{challenge_id}/footprint"),
            headers=self._headers(),
            timeout=30,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "challenge_footprint", resp)
        return resp.json()

    def download_dataset(self, challenge_id: int, dest_dir: str = ".") -> list[str]:
        """Download every published dataset file for a challenge into `dest_dir`.

        Convenience composition over `datasets()`: resolves the signed
        `download_url`s and streams each file to `<dest_dir>/<label>`. The
        signed URLs are pre-authenticated R2 links, so they are fetched
        WITHOUT the bearer token (and redirects are followed, unlike the API
        calls). Returns the list of written file paths. This is the call a
        starter notebook makes to get the challenge data.
        """
        os.makedirs(dest_dir, exist_ok=True)
        written: list[str] = []
        payload = self.datasets(challenge_id)
        for ds in payload.get("datasets", []):
            for f in ds.get("files", []):
                url = f.get("download_url")
                if not url:
                    continue
                # Presigned R2 URL: no auth header, follow redirects.
                resp = self._request("GET", url, allow_redirects=True, timeout=300, stream=True)
                resp.raise_for_status()
                out_path = os.path.join(dest_dir, f["label"])
                with open(out_path, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        if chunk:
                            fh.write(chunk)
                written.append(out_path)
        return written

    def create_challenge(self, name: str | None = None,
                         kernel_version: str | None = None,
                         description: str | None = None,
                         copy_from_challenge_id: int | None = None,
                         tag_names: list[str] | None = None,
                         is_public: bool | None = None) -> dict:
        """Create a challenge with no course, for everyone (author scope,
        and the publishing right: an admin, for now).

        It is open to every signed-in user and awards medals and
        ranking points by default (`is_award_points`, `update_challenge`).
        It is the platform's: a sponsored challenge or a hard benchmark. A
        challenge for a class is created in its course instead
        (`add_course_challenge`, or `create_challenge_for_class` for a new
        course around it); anyone else is refused here with
        `PermissionDeniedError` (403). Whoever may publish has accepted the
        terms of use first, an admin included (`accept_terms`, once per
        account): 403 with `.body["reason"] == "terms_required"` otherwise,
        and nothing is created.

        Mirrors `POST /api/creator_challenge/challenge`. `kernel_version` is
        a kind: the keys `available_kinds()` serves (call it for the live
        list and whether each has a machine, `has_machine`). Gymnasium,
        PettingZoo and the other env-image families are not kinds: they are
        flex_v1 challenges whose env runtime an admin picks. `name` is
        required for a new challenge.

        A copy (`copy_from_challenge_id`, a challenge you may copy: one of
        an example course, one you are staff of, any for an admin) has its
        source's kind (`kernel_version` is not needed), takes its env
        folder, `challenge.toml`, every config key, its description and its
        tags unless you send your own, and is named after its source unless
        you send `name` (` (2)`, ` (3)`… when taken). Every copy starts
        with the same data and the same scoring files as its source: change
        the seed or the data before you start it.

        The challenge runs on its kind's default machine, for a copy too;
        the reply carries `message`, `challenge_id`, `name`, `slug`,
        `machine_name` and `copied_from_challenge_id`. To move it, call
        `update_settings(challenge_id, machine_id=…)` with an id from
        `machines(kernel_version)`. A new challenge is born with its kind's
        sample `env.py` and no benchmark file (`run_benchmark`); an exercise
        set (`exercise_v1`) is born data-only (a two-line `env.py` and an
        empty `exercises.json`, written with `update_exercise_definition`).

        If `tag_names` is given, each name is resolved against the public
        tag catalog (`GET /api/challenge_tags/tags`) before the create
        call; on a copy they replace the source's tags. Unknown names raise
        `MLArenaError` (fail fast — the catalog is admin-curated; new tags
        are not auto-created).

        A new challenge with no course is private unless `is_public=True` is passed:
        hidden from listings and direct URLs, seen by its owner, its
        assistants and admins only. `update_challenge(is_public=...)`
        changes it later.
        """
        if copy_from_challenge_id is None and (name is None or kernel_version is None):
            raise TypeError(
                "create_challenge needs name and kernel_version for a new "
                "challenge with no course (or copy_from_challenge_id for a copy)"
            )
        body: dict = {}
        if name is not None:
            body["name"] = name
        if kernel_version is not None:
            body["kernel_version"] = kernel_version
        if description is not None:
            body["description"] = description
        if copy_from_challenge_id is not None:
            body["copy_from_challenge_id"] = copy_from_challenge_id
        if tag_names:
            body["tag_ids"] = self._resolve_tag_names(tag_names)
        if is_public is not None:
            body["is_public"] = is_public
        resp = self._request("POST",
            self._url("/creator_challenge/challenge"),
            headers=self._headers(json_body=True),
            json=body,
            timeout=60,
        )
        self._handle_response(resp)
        if resp.status_code not in (200, 201):
            raise _failed(MLArenaError, "create_challenge", resp)
        return resp.json()

    def list_tags(self):
        """Return the tag catalog (`GET /api/challenge_tags/tags`).

        Each row carries `id`, `name`, `description` and `challenge_count`:
        how many of the challenges `challenges()` lists for you (started ones)
        carry the tag. A public route; the bearer token travels anyway, as on
        every read, and the counts follow what you may see.
        """
        resp = self._request("GET", self._url("/challenge_tags/tags"),
                             headers=self._headers(), timeout=30)
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "list_tags", resp)
        return _to_dataframe(resp.json())

    def public_stats(self) -> dict:
        """The platform's three counters, as the landing page shows them.

        Mirrors `GET /api/public/stats/` (`LandingStatsOut`): returns
        `{"users": int, "challenges": int, "submissions": int}`. `users` and
        `submissions` count every row; `challenges` counts what `challenges()`
        lists for you. A public route; the bearer token travels anyway, as on
        every read.
        """
        resp = self._request("GET", self._url("/public/stats/"),
                             headers=self._headers(), timeout=30)
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "public_stats", resp)
        return resp.json()

    def set_challenge_tags(self, challenge_id: int,
                             tag_names: list[str] | None = None,
                             tag_ids: list[int] | None = None) -> dict:
        """Replace the tag set on a challenge.

        Pass exactly one of `tag_names` or `tag_ids`. Names are resolved
        against the catalog; unknown names raise `MLArenaError`. Pass an
        empty list to clear all tags.

        Requires an `author`-scope token and ownership (or admin) of the
        target challenge.
        """
        if (tag_names is None) == (tag_ids is None):
            raise MLArenaError("Provide exactly one of tag_names= or tag_ids=")
        ids = tag_ids if tag_ids is not None else self._resolve_tag_names(tag_names)
        resp = self._request("PUT",
            self._url(f"/creator_challenge/challenge/{challenge_id}/tags"),
            headers=self._headers(json_body=True),
            json={"tag_ids": ids},
            timeout=30,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "set_challenge_tags", resp)
        return resp.json()

    def challenge_tags(self, challenge_id: int) -> list:
        """The tags currently on a challenge (author scope).

        Mirrors `GET /api/creator_challenge/challenge/{id}/tags`
        (`tags.py`). Returns the tag rows (`id`, `name`, `description`);
        `set_challenge_tags()` replaces the set.
        """
        return self._creator_get(challenge_id, "/tags", "challenge_tags")

    def _creator_get(self, challenge_id: int, suffix: str, label: str,
                     params: dict | None = None):
        """GET one creator_challenge sub-resource of a challenge.

        Every authoring read answers the same way — 404 for a challenge you
        cannot edit, the handler's own `error` otherwise — so they share one
        call site instead of fifteen copies of it.
        """
        resp = self._request("GET",
            self._url(f"/creator_challenge/challenge/{challenge_id}{suffix}"),
            headers=self._headers(),
            params=params,
            timeout=30,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, label, resp)
        return resp.json()

    def _creator_delete(self, challenge_id: int, suffix: str, label: str):
        """DELETE one creator_challenge sub-resource of a challenge."""
        resp = self._request("DELETE",
            self._url(f"/creator_challenge/challenge/{challenge_id}{suffix}"),
            headers=self._headers(),
            timeout=30,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, label, resp)
        return resp.json()

    def _resolve_tag_names(self, tag_names: list[str]) -> list[int]:
        """Resolve a list of tag names to ids via the public catalog.

        Matching is case-insensitive: prod seeds tags as "GYMNASIUM" while
        local seeds them as "gymnasium", and callers shouldn't have to know
        which environment they hit.
        """
        catalog_resp = self._request("GET", self._url("/challenge_tags/tags"),
                                     headers=self._headers(), timeout=30)
        self._handle_response(catalog_resp)
        if catalog_resp.status_code != 200:
            raise _failed(MLArenaError, "list_tags", catalog_resp)
        catalog = catalog_resp.json()
        by_name_ci = {t["name"].casefold(): t["id"] for t in catalog}
        unknown = [n for n in tag_names if n.casefold() not in by_name_ci]
        if unknown:
            raise MLArenaError(
                f"Unknown tag name(s): {unknown}. "
                f"Known tags: {sorted(t['name'] for t in catalog)}"
            )
        return [by_name_ci[n.casefold()] for n in tag_names]

    def update_challenge(self, challenge_id: int, *,
                           name: str | None = None,
                           description: str | None = None,
                           is_public: bool | None = None,
                           is_award_points: bool | None = None,
                           end_date_ts: str | None | _Unset = _UNSET) -> dict:
        """Patch top-level challenge fields (author scope).

        Mirrors `PUT /api/creator_challenge/challenge/{id}`. Only fields
        explicitly passed are sent — everything else is left untouched.

        ``name`` is locked once the challenge has started — a rename strands
        the links and course material already pointing at it.
        ``description`` stays editable while it runs.

        ``is_public`` and ``is_award_points`` are a challenge's with no
        course: a course challenge is seen by whoever
        reads its course and never awards points, so ``True`` on one is a
        400. Changing either needs the owner's rights (403 for an
        assistant), and making such a challenge public needs the publishing
        right as well (an admin, for now); ``is_public=False`` is always the
        owner's to set, so an owner can withdraw their challenge. A
        private one is seen by its owner, its assistants and admins.

        ``end_date_ts`` schedules the end: an ISO-8601 instant in the future
        (``"2026-12-01T18:00:00Z"``; an offset is converted to UTC, a bare
        value is taken as UTC). The platform moves the started challenge to
        ``"ended"`` within a minute of it. ``end_date_ts=None`` removes the
        schedule. Editable while the challenge runs; 400 for a past date and on
        an ended challenge (its end date is the date it ended).
        """
        body: dict = {}
        if name is not None:
            body["name"] = name
        if description is not None:
            body["description"] = description
        if is_public is not None:
            body["is_public"] = is_public
        if is_award_points is not None:
            body["is_award_points"] = is_award_points
        if end_date_ts is not _UNSET:
            body["end_date_ts"] = end_date_ts
        if not body:
            raise MLArenaError("update_challenge requires at least one field")
        resp = self._request("PUT",
            self._url(f"/creator_challenge/challenge/{challenge_id}"),
            headers=self._headers(json_body=True),
            json=body,
            timeout=30,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "update_challenge", resp)
        return resp.json()

    @_refuses_removed_score_kwargs
    def update_settings(self, challenge_id: int, *,
                        simulation_timeout_sec: int | None = None,
                        max_upload_size_bytes: int | None = None,
                        max_upload_files: int | None = None,
                        max_active_submissions_per_participant: int | None = None,
                        submission_filename: str | None = None,
                        agent_max_time_per_step_second: float | None = None,
                        simulation_max_steps: int | None = None,
                        required_files: list[str] | None = None,
                        metrics: list[dict] | None = None,
                        window_days: int | None | _Unset = _UNSET,
                        is_stop_after_deployment: bool | None = None,
                        deployment_nb_constraint_run: int | None = None,
                        deployment_nb_initial_score_run: int | None = None,
                        episode_budget_brackets: list | None = None,
                        data_source_enabled: bool | _Unset = _UNSET,
                        data_source_url: str | None | _Unset = _UNSET,
                        data_source_asset: str | None | _Unset = _UNSET,
                        data_source_filter: dict[str, str] | None | _Unset = _UNSET,
                        data_source_history_hours: int | _Unset = _UNSET,
                        batch_cron: str | _Unset = _UNSET,
                        datagouv_topic_id: str | None | _Unset = _UNSET,
                        machine_id: int | None = None,
                        env_cpu_request: str | None = None,
                        env_cpu_limit: str | None = None,
                        env_memory_request: str | None = None,
                        env_memory_limit: str | None = None,
                        agent_cpu_request: str | None = None,
                        agent_cpu_limit: str | None = None,
                        agent_memory_request: str | None = None,
                        agent_memory_limit: str | None = None,
                        agent_ephemeral_storage_limit: str | None = None,
                        env_gpu_count: int | None = None,
                        agent_gpu_count: int | None = None,
                        gpu_memory_limit: str | None | _Unset = _UNSET,
                        number_of_agents: int | None = None,
                        docker_image_env_runtime_id: int | None = None,
                        agent_runtime_ids: list[int] | None = None,
                        default_agent_runtime_id: int | None = None,
                        render_delay_second: float | None = None,
                        time_limit_sec: int | None = None,
                        memory_limit_mb: int | None = None,
                        max_concurrent_attempts: int | None = None,
                        max_attempts_per_minute: int | None = None,
                        ) -> dict:
        """Update challenge settings + evaluation parameters.

        Mirrors `PUT /api/creator_challenge/challenge/{id}/settings` —
        the same call the console's Config tab makes. Only fields explicitly
        passed are sent; everything else is left untouched. Every keyword is a
        config key (`config_fields()`), the name of the column that holds it
        and the key a `challenge.toml` would use.

        The config has two editors: this method (and the console's form) and
        the env folder's `challenge.toml`. A key the applied file holds is
        owned by the file: sending it here raises `MLArenaError` (400)
        `"<key> is set in challenge.toml (line N); edit the file"` — change
        the file (`upload_env_file`) instead. `creator_challenge(id)
        ["config"]["origin"]` says who owns each key. On
        a file challenge whose `env.py` the scoring form generated,
        `submission_filename` and what `metrics` ranks (the ranking
        descriptor's `key`, `source`, `order`) are the form's: a value that
        would change them away from what the file states is a 409 naming
        the form, since the generated file would keep the old ones; change
        them with `set_scoring_form` (label, precision and visibility stay
        settable here). The same rule holds for a `challenge.toml` and for a
        bundle that keeps the generated `env.py`; one that brings its own
        `env.py` takes the scoring over. Once
        the challenge has started only the fields that cannot rescore an
        existing run still apply — the presentation fields of `metrics`
        (`label`, `unit`, `precision`, `format`, `visible`) and
        `is_stop_after_deployment` (and `machine_id` and `datagouv_topic_id`
        for an admin); everything
        else is rejected with 400 until the challenge is stopped.

        SDK 4.0.0 removed the keywords `metric`, `metric2`, `is_elo_score`,
        `metric_order`, `frontend_precision` and `metrics_schema`: passing
        one raises `TypeError` pointing at `metrics`.

        A kind accepts only the settings it has: the keywords of a
        capability flag that is false on the challenge's kind
        (`available_kinds()[i]["capabilities"]`) are refused with 400 —
        `runs` (simulation_timeout_sec, the
        deployment_nb_* runs), `submission_upload` (max_upload_size_bytes,
        max_upload_files, max_active_submissions_per_participant),
        `submission_filename`, `elo` (a `metrics` spec with
        `agg: "rating"`),
        `episode_budget_brackets`, `run_limits`, `agent_containers`
        (the agent_* sizing, agent_gpu_count, is_stop_after_deployment,
        render_delay_second: a file challenge has neither), `multi_agent`
        (number_of_agents) and `exercises` (time_limit_sec, memory_limit_mb,
        max_concurrent_attempts, max_attempts_per_minute). The others apply to every kind, the runtime
        included (machine_id, the env_* sizing, env_gpu_count,
        gpu_memory_limit).

        Returns `{"configuration": {...}, "evaluation": {...}, "config":
        {...}}` — the two rows as the update left them, each under its own
        column names (`number_of_agents` is the configuration's), and the
        flat config document of `creator_challenge`.

        Notable parameters:
            submission_filename: file_v1 only — the one file a participant
                uploads, e.g. "submission.csv" (the default), "pitch.txt" or
                "submission.csv.gz". A bare file name: letters, digits, '.',
                '_' and '-', starting with a letter or digit, at most 128
                characters. A name not ending in ".csv" skips the y_test.csv
                column check at upload and the y_test.csv start requirement,
                so env.py must validate the file itself. Frozen once the
                challenge has started.
            metrics: the challenge's score declaration, the whole list (it
                replaces the stored one). Each entry is a MetricSpec dict
                with every key: `key`, `label`, `source` ("score" | "env" |
                "platform"), `agg` ("mean" | "sum" | "max" | "min" |
                "last" | "rating"), `order` ("desc" | "asc" | None),
                `format` ("number" | "integer" | "percent" | "seconds" |
                "bytes" | "currency" | "energy" (a value in Wh, shown
                mWh / Wh / kWh) | "mass" (a value in g, shown mg / g / kg);
                `unit` must be None for the last two), `unit`, `precision`
                (0..10), `is_ranking`, `visible`. A `platform` key is one of
                `elo`, `action_time_max`, `ram_max`, `vram_max`, `steps`,
                `n_episodes`, and the agent's footprint `energy` (Wh / run),
                `co2e` (g CO₂e / run), `cpu_time`, `gpu_time`, `busy_time`
                (seconds) — the footprint keys only on a kind with agent
                containers (flex_v1; 400 elsewhere). `mlarena.format_metric`
                renders a value as the console does. Exactly one entry has
                `is_ranking: True` with an `order`: its `order` decides who
                wins each run, the episode budget tiers, the leaderboard
                order and course pass verdicts. `agg: "rating"` (only on the
                platform key `elo`, `order: "desc"`, ranking) makes the board
                ELO-rated; refused unless the kind has `elo` and the challenge
                seats at least 2 agents (`number_of_agents`). Validated on the merged state (400
                names the rule); once started only the presentation fields
                may change. On flex_v1 the server appends the platform
                footprint descriptors (`energy`, `co2e`) the list leaves
                out: list them to relabel or hide them; they cannot be
                removed. Not editable on a chat challenge. See the README
                "Metrics" section.
            window_days: when set, every declared key is also folded over
                the runs of the last `window_days` days (the leaderboard's
                `metrics_window`). >= 1; `None` sends null and removes the
                window; not sent when omitted. Frozen once started.
            is_stop_after_deployment: True (the creation default) means an
                agent runs only its deployment runs and is never matched
                again, which pins an ELO leaderboard to its bootstrap ratings
                forever. Set False to let the matchmaker keep pairing agents
                (~10 matches per challenge every 2h). Editable on a running
                challenge. Code challenges only: a file is scored once.
            deployment_nb_constraint_run, deployment_nb_initial_score_run:
                the unscored check runs, then the scored runs, of each
                deploy. A new code challenge has 2 and 10, a new file
                challenge 0 and 1 (scoring the same file again gives the
                same score; raise them when your env draws at random).
            episode_budget_brackets: episode budget tiers, a list of
                `[threshold, n_episodes]` pairs from the worst threshold to
                the best. A submission whose running mean is worse than a
                tier's threshold gets that tier's episodes; one that beats
                every threshold gets the last tier's. Thresholds strictly
                ascending under a ranking `order` "desc", strictly
                descending under "asc"; n_episodes non-decreasing; max 100
                episodes per bracket. A single pair is a fixed budget.

        The Elo parameters of a rated board (K factor, scale, pairing
        weights, initial rating and variance) are platform constants: the
        `elo_*` keywords left in SDK 4.12.0.

        Run budgets (author keys; strictly positive; frozen while the
        challenge runs):
            simulation_timeout_sec: the wallclock of one run, in seconds, at
                most the challenge's machine's `job_timeout_max_sec`
                (`machines()`); above it the backend answers 400 naming the
                cap.
            agent_max_time_per_step_second: deadline of one agent call
                (`AgentProxy.call`), in seconds.
            simulation_max_steps: the step budget of one episode (flexkit
                loops).
            required_files: names that must exist in the env folder before
                the challenge can start (e.g. `["y_test.csv"]`); plain file
                names. A missing one is the `required_file_missing` start
                blocker.
        SDK 4.3.0 removed `env_max_time_per_step_second` (the platform
        dropped it: it was measured and never enforced).

        Data feed (admin only — anyone else gets `PermissionDeniedError`,
        403; frozen while the challenge runs). Each is sent only when passed,
        and `None` is sent as null where the column allows it:
            data_source_enabled: stage `{data_source_url}/{data_source_asset}`
                for env.py on every `batch_cron` tick (a continuous
                challenge). Enabling needs a URL and an asset.
            data_source_url / data_source_asset: the DC API base URL and
                the asset path under it (e.g. ".../api-dc", "weather");
                None clears one on a disabled feed.
            data_source_filter: extra query parameters, string values only,
                e.g. `{"cities": "Paris:FR,Berlin:DE"}`; None clears it.
            data_source_history_hours: hours of history per batch (1..9600).
            batch_cron: the UTC cron schedule of fetches and runs.

        data.gouv.fr partnership (admin only — anyone else gets
        `PermissionDeniedError`, 403; editable while the challenge runs,
        sent only when passed):
            datagouv_topic_id: the 24-hex id of the data.gouv.fr topic of
                the challenge's défi; teams may then link a reuse
                (`link_datagouv_reuse`). None clears it.

        Runtime (the Runtime section of the console; frozen while the
        challenge runs, except `machine_id` for an admin — pending jobs move
        with it, running ones finish where they are). Quantities are K8s
        strings (`"500m"`, `"2"`, `"1024Mi"`, `"4Gi"`); a kind's defaults
        and its `small` / `medium` / `large` presets are
        `available_kinds()[i]["runtime_defaults"]` / `["runtime_presets"]`.
        The backend checks the merged runtime against the machine (request
        ≤ limit, one job within the machine's `job_cpu_max` /
        `job_memory_max`, GPUs within its inventory) and answers 400. The
        message is worded for the caller: what to change (a smaller size, a
        shorter run timeout, the agents per run, or "ask an admin") for an
        author, the machine, its limit and the rule that failed for an
        admin key; `.body["details"]` lists each reason. A refused
        `challenge.toml` adds the lines of the keys concerned:
            machine_id: a machine from `machines(kernel_version)` whose queue
                for the challenge's kernel the owner may use.
            env_cpu_request / env_cpu_limit / env_memory_request /
                env_memory_limit: the env container's size.
            agent_cpu_request / agent_cpu_limit / agent_memory_request /
                agent_memory_limit / agent_ephemeral_storage_limit: each agent
                container's size (kinds with `agent_containers`).
            number_of_agents: agents seated per match, >= 1 (kinds with
                `multi_agent`; an ELO board needs >= 2).
            env_gpu_count / agent_gpu_count / gpu_memory_limit: admin only —
                GPU devices from the machine's inventory, and the GPU memory
                cap (`None` sends null and clears it).

        Infrastructure (admin only — anyone else gets
        `PermissionDeniedError`, 403):
            docker_image_env_runtime_id: the env runtime image, of the
                challenge's own kernel.

        Agent runtimes (author keys, kinds with `agent_containers`;
        editable while the challenge runs — an offer changes no score):
            agent_runtime_ids: the agent runtime images participants may pick,
                the whole set (it replaces the stored one): non-empty, unique
                ids from `agent_runtimes(kernel_version)`. A submission keeps
                the image it already pins when that image leaves the set; its
                next deploy is refused (409) until its owner picks an offered
                runtime (`set_agent_runtime`).
            default_agent_runtime_id: the runtime a new submission gets, one
                of the set. Removing the current default from the set needs a
                new default in the same call.
            The backend checks the merged pair and answers 400 naming the
            rule: an id of another kernel, an empty set, a default outside
            the set. `challenge(id)["runtime_options"]` /
            `["default_agent_runtime_id"]` serve them back.

        Replay (an author key, editable while started):
            render_delay_second: the default frame duration (s, >= 0): how
                long a replay frame that sets no `duration_s` of its own is
                shown. Editing it changes the playback of stored replays too.

        Exercise limits (author keys, kinds with `exercises`; editable while
        the challenge runs except `memory_limit_mb`, which sizes the checker):
            time_limit_sec: the CPU time one attempt may use, 1..30 s
                (default 5); past it the attempt fails `timeout`.
            memory_limit_mb: the memory of one attempt, 64..2048 MB (default
                512); past it the attempt fails `memory`.
            max_concurrent_attempts: attempts the checker grades at once,
                1..16 (default 8).
            max_attempts_per_minute: attempts one group may send per minute,
                1..600 (default 20); past it `submit_exercise` gets a 429.

        Requires an `author`-scope token and the challenge's staff (its
        owner, an assistant — a course challenge's: the course's owner and
        assistants —, an admin).
        """
        sent = {
            "simulation_timeout_sec": simulation_timeout_sec,
            "max_upload_size_bytes": max_upload_size_bytes,
            "max_upload_files": max_upload_files,
            "max_active_submissions_per_participant":
                max_active_submissions_per_participant,
            "submission_filename": submission_filename,
            "agent_max_time_per_step_second": agent_max_time_per_step_second,
            "simulation_max_steps": simulation_max_steps,
            "required_files": required_files,
            "metrics": metrics,
            "is_stop_after_deployment": is_stop_after_deployment,
            "deployment_nb_constraint_run": deployment_nb_constraint_run,
            "deployment_nb_initial_score_run": deployment_nb_initial_score_run,
            "episode_budget_brackets": episode_budget_brackets,
            "machine_id": machine_id,
            "env_cpu_request": env_cpu_request,
            "env_cpu_limit": env_cpu_limit,
            "env_memory_request": env_memory_request,
            "env_memory_limit": env_memory_limit,
            "agent_cpu_request": agent_cpu_request,
            "agent_cpu_limit": agent_cpu_limit,
            "agent_memory_request": agent_memory_request,
            "agent_memory_limit": agent_memory_limit,
            "agent_ephemeral_storage_limit": agent_ephemeral_storage_limit,
            "env_gpu_count": env_gpu_count,
            "agent_gpu_count": agent_gpu_count,
            "number_of_agents": number_of_agents,
            "docker_image_env_runtime_id": docker_image_env_runtime_id,
            "agent_runtime_ids": agent_runtime_ids,
            "default_agent_runtime_id": default_agent_runtime_id,
            "render_delay_second": render_delay_second,
            "time_limit_sec": time_limit_sec,
            "memory_limit_mb": memory_limit_mb,
            "max_concurrent_attempts": max_concurrent_attempts,
            "max_attempts_per_minute": max_attempts_per_minute,
        }
        body = {key: value for key, value in sent.items() if value is not None}
        # Keys whose None is an explicit null (the backend clears the
        # column): sent only when passed.
        nullable = {
            "window_days": window_days,
            "data_source_enabled": data_source_enabled,
            "data_source_url": data_source_url,
            "data_source_asset": data_source_asset,
            "data_source_filter": data_source_filter,
            "data_source_history_hours": data_source_history_hours,
            "batch_cron": batch_cron,
            "datagouv_topic_id": datagouv_topic_id,
            "gpu_memory_limit": gpu_memory_limit,
        }
        body.update({key: value for key, value in nullable.items() if value is not _UNSET})
        if not body:
            raise MLArenaError("update_settings requires at least one field")
        resp = self._request("PUT",
            self._url(f"/creator_challenge/challenge/{challenge_id}/settings"),
            headers=self._headers(json_body=True),
            json=body,
            timeout=30,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "update_settings", resp)
        return resp.json()

    def list_env_files(self, challenge_id: int) -> dict:
        """The challenge's env folder, as the editor lists it (author scope).

        Mirrors `GET /api/creator_challenge/challenge/{id}/env/files`
        (`env_files.py`). Returns `{"files": {"<name>": {"content",
        "language", "is_binary", "size"}}}`; a binary file has `content: null`
        and is reachable only through the console's download.
        """
        return self._creator_get(challenge_id, "/env/files", "list_env_files")

    def delete_env_file(self, challenge_id: int, filename: str) -> dict:
        """Delete one file from the challenge's env folder (author scope).

        Mirrors `DELETE /api/creator_challenge/challenge/{id}/env/files/{name}`
        (`env_files.py`). Accepted on a started challenge too: the delete is
        an update of the env folder (`env_update`), put in place once the
        running runs have ended, and the reply carries its state under
        `env_update`. On a started challenge whose owner has no compute
        credits left it is refused like every update of its files (409,
        `.body["reason"] == "out_of_credits"`, see `upload_env_file`).
        """
        return self._creator_delete(
            challenge_id, f"/env/files/{filename}", "delete_env_file"
        )

    def check_env(self, challenge_id: int, content: str = "") -> dict:
        """Validate an `env.py` buffer without saving it (author scope).

        Mirrors `POST /api/creator_challenge/challenge/{id}/env/check`
        (`env_files.py`) — the structural check the console runs as you type:
        `class Env`, its `evaluate` signature, the returned `agent_results`
        keys. Pass the buffer you are about to save; the empty default reports
        the missing-source case. Returns the validator's findings, not a
        pass/fail HTTP status.
        """
        resp = self._request("POST",
            self._url(f"/creator_challenge/challenge/{challenge_id}/env/check"),
            headers=self._headers(json_body=True),
            json={"content": content},
            timeout=30,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "check_env", resp)
        return resp.json()

    def upload_env_file(self, challenge_id: int, file_path: str) -> dict:
        """Upload a single environment file (env.py, requirements.txt, …).

        Multipart PUT to `/api/creator_challenge/challenge/{id}/env/files`.
        Use this for binary files; for text content driven by a template,
        prefer `update_env_file_content`.

        Accepted on a started challenge too (4.19.0): the write is an update
        of the env folder, recorded as a revision. The reply's `env_update`
        (the shape of `env_update(id)`) says whether it is in place or
        waiting for the challenge's running runs; on a started file or code
        challenge the benchmark then runs before any participant's run, and
        a failure puts the previous files back (`wait_for_env_update`). That
        check is a run (or, for a chat challenge or an exercise set, a new
        pod) the challenge's owner pays: when the owner has no compute
        credits left on the balance the challenge's machine draws from
        (`credits()`), an update of a started challenge's files is refused
        with a 409 whose `.body` carries `reason: "out_of_credits"` beside
        `error`, and nothing is written. The same refusal holds for
        `update_env_file_content`, `delete_env_file`, `push_challenge`,
        `import_challenge_from_github`, `set_scoring_form`,
        `update_exercise_definition` and `revert_env`; a challenge that is
        not started is never refused for credits.

        `challenge.toml` is the challenge config's file editor: it is applied
        when it is saved — validated like `update_settings`, with the same
        permissions — and written only if it applies. A refused file raises
        `MLArenaError` (400, or 403 for an admin-only key) whose message names
        each key's line (`challenge.toml:3: unknown key 'x'`), and nothing
        changes. Accepted, it owns every key it holds; the reply carries the
        new `config` (as in `creator_challenge`). Once started, a file is
        accepted only when it changes keys a started challenge still accepts.
        `export_config()` writes a starting file.
        """
        if not os.path.isfile(file_path):
            raise MLArenaError(f"File not found: {file_path}")
        with open(file_path, "rb") as fh:
            resp = self._request("PUT",
                self._url(
                    f"/creator_challenge/challenge/{challenge_id}/env/files"
                ),
                headers=self._headers(),
                files={"file": (os.path.basename(file_path), fh)},
                timeout=120,
            )
        self._handle_response(resp)
        if resp.status_code not in (200, 201):
            raise _failed(MLArenaError, "upload_env_file", resp)
        return resp.json()

    def push_challenge(self, challenge_id: int, path: str, *,
                       subpath: str | None = None,
                       dry_run: bool = False) -> dict:
        """Replace the challenge's files with a directory, in one step
        (author scope).

        Mirrors `PUT /api/creator_challenge/challenge/{id}/bundle` (multipart
        field `archive`, query `path` / `dry_run`). `path` is the challenge
        directory (`env.py`, `challenge.toml`, data files…), zipped by
        `mlarena.bundle.make_bundle` (`.git`, caches and virtualenvs left
        out, symbolic links refused), or an existing `.zip` / `.tar.gz` of
        it. `subpath` names the folder inside `path` that holds the
        challenge (one repository, several challenges): of a directory, only
        that folder is bundled; of an archive, the server reads only that
        folder (the route's `path` query).

        The bundle is applied all or nothing: `challenge.toml` is validated
        like `upload_env_file` does, `env.py` is checked, and a refused
        bundle raises `MLArenaError` with the server's message (its details
        on `.body`) and changes nothing. `dry_run=True` runs the same checks
        and writes nothing. Accepted on a started challenge too, as the file
        routes are: the env files become the folder's pending update
        (`env_update`), `challenge.toml` applies the keys a started challenge
        still accepts and an `agent.py` that differs from the stored agent
        template is a 400. On a started
        challenge whose owner has no compute credits left the push is a 409
        with `.body["reason"] == "out_of_credits"` (see `upload_env_file`).

        Returns the server's report: what was written, the config result,
        the `env.py` validation and the entries it ignored. From a GitHub
        Actions run the commit is recorded as the environment's
        `source_revision`.
        """
        params: dict = {}
        if os.path.isdir(path):
            # The sub-folder is what gets bundled: nothing else is uploaded.
            path = bundle_root(path, subpath)
        elif subpath is not None:
            params["path"] = subpath
        filename, archive = read_bundle(path)
        if dry_run:
            params["dry_run"] = "true"
        resp = self._request("PUT",
            self._url(f"/creator_challenge/challenge/{challenge_id}/bundle"),
            headers=self._headers(),
            params=params or None,
            files={"archive": (filename, archive)},
            timeout=300,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "push_challenge", resp)
        return resp.json()

    def import_challenge_from_github(self, challenge_id: int, repository: str, *,
                                     ref: str | None = None,
                                     path: str | None = None,
                                     dry_run: bool = False) -> dict:
        """Replace the challenge's files with a public GitHub repository's
        (author scope).

        Mirrors `POST /api/creator_challenge/challenge/{id}/import-github`
        `{"repository", "ref", "path", "dry_run"}`. `repository` is
        `owner/name` or the repository's URL; `ref` a branch, tag or commit
        (default: the default branch); `path` the folder of the repository
        that holds the challenge. The server downloads that commit and
        applies it as `push_challenge` does, with the same report. A private
        repository is pushed from its own GitHub Actions workflow instead
        (`mlarena challenge push`).
        """
        resp = self._request("POST",
            self._url(f"/creator_challenge/challenge/{challenge_id}/import-github"),
            headers=self._headers(json_body=True),
            json={"repository": repository, "ref": ref, "path": path,
                  "dry_run": dry_run},
            timeout=300,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "import_challenge_from_github", resp)
        return resp.json()

    def update_env_file_content(self, challenge_id: int, filename: str,
                                content: str) -> dict:
        """Write a text env file by content (e.g. a rendered env.py template).

        JSON PUT to the same endpoint as `upload_env_file`. The backend runs
        the kernel-specific AST validator on env.py and returns a
        `validation` block in the response. `challenge.toml` is applied on
        save, as `upload_env_file` describes; so are the write on a started
        challenge and its 409 `reason: "out_of_credits"`.
        """
        resp = self._request("PUT",
            self._url(
                f"/creator_challenge/challenge/{challenge_id}/env/files"
            ),
            headers=self._headers(json_body=True),
            json={"filename": filename, "content": content},
            timeout=60,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "update_env_file_content", resp)
        return resp.json()

    def set_challenge_image(self, challenge_id: int, image_path: str) -> dict:
        """Upload (or replace) the challenge thumbnail (`miniature.png`).

        Multipart POST to `/api/creator_challenge/challenge/{id}/image`.
        """
        if not os.path.isfile(image_path):
            raise MLArenaError(f"Image file not found: {image_path}")
        with open(image_path, "rb") as fh:
            resp = self._request("POST",
                self._url(
                    f"/creator_challenge/challenge/{challenge_id}/image"
                ),
                headers=self._headers(),
                files={"file": (os.path.basename(image_path), fh)},
                timeout=120,
            )
        self._handle_response(resp)
        if resp.status_code not in (200, 201):
            raise _failed(MLArenaError, "set_challenge_image", resp)
        return resp.json()

    def challenge_image(self, challenge_id: int, dest_dir: str = ".") -> str:
        """Download the challenge thumbnail (`miniature.png`); returns the path.

        Mirrors `GET /api/creator_challenge/challenge/{id}/image`
        (`assets.py`), the author-scoped read of the same file the public
        `/api/challenge_asset/{id}/image/miniature` serves. Raises
        `ChallengeNotFoundError` when the challenge has no image yet.
        """
        os.makedirs(dest_dir, exist_ok=True)
        resp = self._request("GET",
            self._url(f"/creator_challenge/challenge/{challenge_id}/image"),
            headers=self._headers(),
            timeout=120,
            stream=True,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "challenge_image", resp)
        out_path = os.path.join(dest_dir, "miniature.png")
        with open(out_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    fh.write(chunk)
        return out_path

    def delete_challenge_image(self, challenge_id: int) -> dict:
        """Remove the challenge thumbnail (`miniature.png`).

        Mirrors `DELETE /api/creator_challenge/challenge/{id}/image`
        (`assets.py`). The challenge card then renders without one.
        """
        return self._creator_delete(challenge_id, "/image", "delete_challenge_image")

    def challenge_markdown(self, challenge_id: int) -> dict:
        """The challenge overview markdown (`overview.md`), author scope.

        Mirrors `GET /api/creator_challenge/challenge/{id}/markdown`
        (`assets.py`). Returns `{"content": "..."}` — the empty string when
        none was ever written.
        """
        return self._creator_get(challenge_id, "/markdown", "challenge_markdown")

    def set_challenge_markdown(self, challenge_id: int, content: str) -> dict:
        """Replace the challenge overview markdown (`overview.md`)."""
        resp = self._request("PUT",
            self._url(
                f"/creator_challenge/challenge/{challenge_id}/markdown"
            ),
            headers=self._headers(json_body=True),
            json={"content": content},
            timeout=30,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "set_challenge_markdown", resp)
        return resp.json()

    def upload_benchmark_file(self, challenge_id: int, file_path: str,
                              filename: str | None = None) -> dict:
        """Upload a benchmark file (`agent.py`, or a file_v1 submission file).

        Backend reads the benchmark folder when `run_benchmark` fires; you
        must upload the benchmark file before kicking the run. The upload is
        multipart and byte-exact, so use it (not
        `update_benchmark_file_content`, which writes text) for a binary
        submission such as `submission.csv.gz`.

        `filename` stores the file under another name — e.g. a local
        `benchmark_submission.csv.gz` as the challenge's `submission_filename`
        — the same rename the console's benchmark drop zone does. Defaults to
        the local file's base name.
        """
        if not os.path.isfile(file_path):
            raise MLArenaError(f"File not found: {file_path}")
        upload_name = filename if filename is not None else os.path.basename(file_path)
        with open(file_path, "rb") as fh:
            resp = self._request("PUT",
                self._url(
                    f"/creator_challenge/challenge/{challenge_id}/benchmark/files"
                ),
                headers=self._headers(),
                files={"file": (upload_name, fh)},
                timeout=120,
            )
        self._handle_response(resp)
        if resp.status_code not in (200, 201):
            raise _failed(MLArenaError, "upload_benchmark_file", resp)
        return resp.json()

    def update_benchmark_file_content(self, challenge_id: int, filename: str,
                                      content: str) -> dict:
        """Write a benchmark file by content (templated baseline agent)."""
        resp = self._request("PUT",
            self._url(
                f"/creator_challenge/challenge/{challenge_id}/benchmark/files"
            ),
            headers=self._headers(json_body=True),
            json={"filename": filename, "content": content},
            timeout=60,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "update_benchmark_file_content", resp)
        return resp.json()

    def list_benchmark_files(self, challenge_id: int) -> dict:
        """The challenge's benchmark folder (author scope).

        Mirrors `GET /api/creator_challenge/challenge/{id}/benchmark/files`
        (`benchmark.py`). Same shape as `list_env_files()`: what is stored.
        The benchmark `agent.py` on a code challenge once you uploaded one or
        a first `run_benchmark` stored the agent template; the benchmark
        submission file on a file challenge once you uploaded it
        (`upload_benchmark_file`) or generated the baseline
        (`generate_scoring_baseline`). A read: it writes nothing, and
        `{"files": {}}` is an empty folder.
        """
        return self._creator_get(
            challenge_id, "/benchmark/files", "list_benchmark_files"
        )

    def delete_benchmark_file(self, challenge_id: int, filename: str) -> dict:
        """Delete one benchmark file (author scope).

        Mirrors `DELETE
        /api/creator_challenge/challenge/{id}/benchmark/files/{name}`
        (`benchmark.py`). Refused once the challenge has started.
        """
        return self._creator_delete(
            challenge_id, f"/benchmark/files/{filename}", "delete_benchmark_file"
        )

    def run_benchmark(self, challenge_id: int) -> dict:
        """Kick off the benchmark simulation. Returns the new run, in the
        shape `benchmark_status()["run"]` serves it (`job_status` `pending`).

        Requires that env.py has been uploaded. On a code challenge with no
        benchmark `agent.py` stored, the challenge's current agent template
        is stored as `agent.py` and runs (no upload and no earlier read
        needed). A file challenge needs its submission file (e.g.
        submission.csv / pitch.txt); while the challenge is not started, one
        with nothing to score is refused (400) before a run is queued: an
        empty file, or a CSV holding only the ground truth's header line.
        The call itself is fire-and-forget; poll `benchmark_status` for
        completion.

        The run is the challenge's owner's to pay, whoever asks for it: a 409
        whose `.body` carries `reason: "out_of_credits"` beside `error` when
        the owner has no compute credits left on the balance the challenge's
        machine draws from (`credits()`), started or not; no run is queued.
        A 409 with `reason: "test_runs_cap"` when the owner already has 3
        test runs queued or running, all their challenges together
        (benchmark runs and exercise self-tests, whoever of the staff asked;
        an admin owner is exempt): nothing is queued, wait for one to finish
        (`benchmark_status`) and call again.
        """
        resp = self._request("POST",
            self._url(
                f"/creator_challenge/challenge/{challenge_id}/benchmark/run"
            ),
            headers=self._headers(),
            timeout=60,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "run_benchmark", resp)
        return resp.json()

    def benchmark_status(self, challenge_id: int) -> dict:
        """The latest benchmark run and the config's budgets checked against it.

        Mirrors `GET /api/creator_challenge/challenge/{id}/benchmark/status`
        (`benchmark.py`). Returns `{"run": …, "checks": […]}` (SDK 4.3.0; the
        run alone before). `run` is None before the first run.

        `checks` is empty until the run completes; then one entry per budget:
        `key` (the config key), `label`, `configured`, `measured`, `ratio`
        (measured / configured, None for a non-numeric check), `status`
        (`ok`, `warn` above 0.8, `fail` above 1, `unmeasured`) and
        `message` — the run's wallclock against `simulation_timeout_sec`,
        the longest agent call against `agent_max_time_per_step_second`,
        memory peaks against the limits, episodes against the budget, and
        the env metric keys returned against `metrics`.

        `run` is the same dict `creator_runs()` lists
        (`RunResult`): `job_status` (`pending`, `running`, then `completed`,
        `failed` or `cancelled`), `job_error_type` (why the platform lost
        the run), `env_error_type` / `env_error_message` / `env_stdout_logs`, the timestamps, and
        `submission_results` — the benchmark submission's row, its score
        under `score`. When the run completes cleanly the backend
        scores the benchmark submission (`active`, `score`), which is
        what `start_challenge` requires.
        """
        resp = self._request("GET",
            self._url(
                f"/creator_challenge/challenge/{challenge_id}/benchmark/status"
            ),
            headers=self._headers(),
            timeout=30,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "benchmark_status", resp)
        return resp.json()

    def start_challenge(self, challenge_id: int, *,
                        clean_redeploy_all: bool = False) -> dict:
        """Move a challenge to ``status="started"``.

        Mirrors `PUT /api/creator_challenge/challenge/{id}/start`. Accepted
        from ``draft``, ``stopped`` or ``ended`` (400 when already started;
        409 for an archived challenge, `unarchive_challenge` it first).
        Returns `{"message", "clean_redeploy"}`.

        Any of the challenge's staff starts it (a course challenge's: the
        course's owner and assistants). A public challenge with no course
        is started only by an admin (the publishing right); a stopped one too.

        `clean_redeploy_all=True` restarts a stopped or ended challenge and
        re-scores every eligible submission in one transaction (the last step
        of an update of the scoring: `stop_challenge`, edit, benchmark, then
        this), all or nothing: no deploy and no scheduled run lands between
        the restart and the re-score, and `clean_redeploy` then holds what
        `clean_redeploy_all` answers (`total`, `redeployed`; `failed` is
        empty). Stopping cancels no run, so while a submission to re-score
        still has a run pending or running the call is a 409
        (`.body["runs_in_flight"]`) and nothing is changed: wait for
        `scoring_update(id)["rescore"]["runs_in_flight"]` to reach 0, then
        call again. Read the count of submissions first with
        `scoring_update(id)["rescore"]["submissions"]`. 400 on a draft and on
        a kind without runs. Without it `clean_redeploy` is None.

        The first start sets ``start_date_ts``; a restart keeps it, and clears
        an ``end_date_ts`` already past (the restarted challenge is open-ended
        until `update_challenge(end_date_ts=…)` sets a new end).
        Backend gates this behind: env.py uploaded, benchmark submission
        ACTIVE with a non-null `score`, and the runtime gate (the machine
        is enabled and usable by the owner, the sizing fits it, the seat
        count suits the kind and the board). These are the checks
        `creator_challenge(id)["start_blockers"]` lists ahead of time. A
        refusal bubbles up as MLArenaError whose message is the first
        blocker's (409 when that blocker is a `platform_conflict` — image
        consistency, a runtime rule measured against the machine: the
        machine, its queue, its GPUs, a valid size or timeout above its
        ceiling, the owner's limit of started exercise sets
        (`started_sets_cap`: 15), the owner's limit of started course
        challenges that are not exercise sets (`started_challenges_cap`:
        `max_started_challenges_limit`, 10 by default, admins exempt; stop
        one to start another), or an owner with no compute credits left on
        the balance the challenge's machine draws from (`out_of_credits`:
        the code of the blocker, in `.body["start_blockers"]`; this refusal
        has no `reason` key) —, 400 for missing authoring, which includes a size that is
        invalid on any machine and a seat count the challenge's own settings
        refuse) and whose
        `.body["details"]` lists every blocker's message, each worded for
        the caller as in `start_blockers`; `.body["start_blockers"]` is that
        list itself (`code`, `message`, `fix_by`, `can_fix`, `place`,
        `platform_conflict`). A 503 whose `.body["reason"]` is
        `"files_unreadable"` means the challenge's files could not be read:
        nothing was changed, try again (the `reason` tells it from a
        maintenance 503, which is a `MaintenanceError`).
        """
        resp = self._request("PUT",
            self._url(
                f"/creator_challenge/challenge/{challenge_id}/start"
            ),
            params={"clean_redeploy_all": "true"} if clean_redeploy_all else None,
            headers=self._headers(),
            timeout=300 if clean_redeploy_all else 30,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "start_challenge", resp)
        return resp.json()

    def stop_challenge(self, challenge_id: int) -> dict:
        """Move a started challenge to ``status="stopped"``.

        Mirrors `PUT /api/creator_challenge/challenge/{id}/stop`. Accepted
        only from ``started`` (400 otherwise). Stopping leaves submissions,
        results, the benchmark record and ``start_date_ts`` untouched, and
        unlocks the authoring fields like a draft, so the challenge can be
        fixed and restarted with `start_challenge()`. It changes the status
        and nothing else: runs already pending or running still finish and
        count, no new deploy is accepted, and the challenge leaves the
        catalog's started list while its page and results stay
        (`scoring_update(id)["stop"]` counts what is in flight).
        """
        resp = self._request("PUT",
            self._url(
                f"/creator_challenge/challenge/{challenge_id}/stop"
            ),
            headers=self._headers(),
            timeout=30,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "stop_challenge", resp)
        return resp.json()

    def end_challenge(self, challenge_id: int) -> dict:
        """Move a started challenge to ``status="ended"`` now.

        Mirrors `PUT /api/creator_challenge/challenge/{id}/end`. Accepted only
        from ``started`` (400 otherwise). Submissions close, the leaderboard
        is final, ``end_date_ts`` becomes now, and the challenge is listed
        with the past ones (`challenges(status="ended")`). Unlike
        `stop_challenge()` (paused or withdrawn, listed nowhere), an ended
        challenge stays on show. `start_challenge()` reopens it. To end at a
        set date instead, `update_challenge(cid, end_date_ts=…)`.
        """
        resp = self._request("PUT",
            self._url(
                f"/creator_challenge/challenge/{challenge_id}/end"
            ),
            headers=self._headers(),
            timeout=30,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "end_challenge", resp)
        return resp.json()

    def scoring_update(self, challenge_id: int) -> dict:
        """What stopping a challenge and re-scoring its submissions would do now.

        Mirrors `GET /api/creator_challenge/challenge/{id}/scoring_update`:
        the facts to read before `stop_challenge` → edit → `run_benchmark` →
        `start_challenge(clean_redeploy_all=True)`. Returns

        - `status`;
        - `can_cancel_runs_in_flight`: the challenge is stopped or ended and has
          runs `cancel_runs_in_flight` would cancel (`stop.cancellable_runs`);
        - `stop`: `pending_runs`, `running_runs`, `deploys_in_flight`. Stop
          cancels none of them: the runs still finish and count, the deploy
          attempts still settle. After it no deploy, chat message or exercise
          check is accepted, no further run is scheduled (a run queued before
          the stop still plays, with the env folder as it is when its pod
          starts), and the challenge
          leaves the catalog's started list (only `challenges(status="all")`
          still lists it) while its page and results stay;
        - `rescore` (None on a chat challenge or an exercise set, which have
          no runs): `submissions` (how many a re-score takes: the active and
          the deploy-failed ones), `includes_benchmark`, `deleted_results`
          (their result rows, all deleted), `runs_in_flight` (runs of theirs
          still pending or running: the restart that re-scores is refused
          until it is 0). Each of them gets a fresh deploy on the staff's
          account and has no score until it settles;
        - `env_takes_effect`: `next_run` (every run claimed after an edit of
          the env folder uses it) or `pod_restart` (a chat agent or exercise
          checker loads env.py when its pod is created, and every start
          replaces the pod).
        """
        return self._json_call(
            "GET", f"/creator_challenge/challenge/{challenge_id}/scoring_update",
            error_label="scoring_update",
        )

    def env_update(self, challenge_id: int) -> dict:
        """The state of a challenge's environment files and of an update on its way.

        Mirrors `GET /api/creator_challenge/challenge/{id}/env/update`. Every
        change of the env folder (`upload_env_file`, `delete_env_file`,
        `set_scoring_form`, `update_exercise_definition`, `push_challenge`,
        `import_challenge_from_github`, `revert_env`) builds the new folder
        beside the current one and swaps it in, started or not. While runs of
        the challenge are on their way the swap waits for them, and until it
        is done no queued run starts: the challenge stays started and
        accepts submissions, which queue. Returns

        - `step`: `live` (nothing on its way), `waiting_runs`, `settling`,
          `checking` or `check_failed`: the one step the update is at.
          `check_failed`: the files failed their check and there are no
          checked files to return to; on a file or code challenge no run
          starts until you upload a fix or call `run_benchmark` again;
        - `live`: the revision of the files runs are scored by (None before
          the first recorded change);
        - `pending`: the update waiting, with `waiting_runs` (the running
          runs it waits for) and `waiting_until_ts` (the latest instant the
          platform lets one of them last);
        - `settling_until_ts`: after the swap, the instant from which workers
          read the new files (about a minute: the storage's cache). Until it
          no run starts; a chat agent or an exercise checker keeps serving
          the previous files and is replaced then;
        - `verifying_simulation_result_id`: on a started file or code
          challenge the swap starts the benchmark; until its verdict it is
          the only run that starts, and a failure puts the previous files
          back by itself (`last_reverted["reverted_by"]` is then
          `benchmark_failed` or `benchmark_deadline` (a chat agent or an
          exercise checker that does not load the new files is reverted the
          same way), `"author"` after
          `revert_env`, and `last_reverted["benchmark_failure"]` says why);
        - `auto_reverted`: the files live now are the previous ones, put
          back by the platform after the update that replaced them failed
          its check or gave no verdict in time (`last_reverted` is that
          update). False after your own `revert_env` and once another update
          went live;
        - `swap_error`: why `pending` is not in place yet when something
          stops it (the storage refused a rename or a copy: retried; or the
          record of the swap cannot be read: `cancel_env_update`);
        - `claims_held`, `held_runs`, `message` (the state in one sentence,
          with what participants' runs are scored by);
        - `can_revert`, `can_cancel`.

        A revision is `{"id", "provider", "transport", "content_sha256",
        "user", "created_at_ts", "applied_at_ts", "reverted_at_ts",
        "benchmark_simulation_result_id", "benchmark_passed",
        "benchmark_failure", "is_live", "is_previous", "is_pending", …}`;
        `transport` says what changed the folder (`SourceTransport`:
        `file_write`, `file_delete`, `form`, `bundle`, `revert`, `baseline`
        — the files as they were when the folder's first revision was
        recorded —, or the GitHub ones, `actions_oidc` — a GitHub Actions
        run — and `public_fetch` — a public repository read by the
        platform).
        Reading the state also moves a pending update forward.
        """
        return self._json_call(
            "GET", f"/creator_challenge/challenge/{challenge_id}/env/update",
            error_label="env_update",
        )

    def wait_for_env_update(
        self, challenge_id: int, *, timeout: float = 600.0, poll_interval: float = 2.0,
    ) -> dict:
        """Poll `env_update` until the update is in place, settled and judged.

        That is: nothing pending, the settling time after the swap over
        (`settling_until_ts`: workers read the new files about a minute after
        a change), and no benchmark verdict awaited.

        A client-side composition of `env_update`. Returns the last state;
        raises `TimeoutError` after `timeout` seconds. After it,
        `state["live"]` is the revision in place: when the update's benchmark
        failed, that is a `revert` revision and `state["last_reverted"]` is
        the update that was taken out.
        """
        deadline = time.monotonic() + timeout
        while True:
            state = self.env_update(challenge_id)
            if state["step"] in ("live", "check_failed"):
                return state
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"env update of challenge {challenge_id} still on its way after "
                    f"{timeout:.0f} s: {state['message']}"
                )
            time.sleep(poll_interval)

    def cancel_env_update(self, challenge_id: int) -> dict:
        """Cancel the pending update of a challenge's environment files.

        Mirrors `DELETE /api/creator_challenge/challenge/{id}/env/update`:
        the staged files are dropped, and a swap of them that kept failing
        is undone. Cancelling releases nothing by itself: the queued runs
        start again only when the files in place were released to
        participants (after a failed check with nothing to return to, or
        when you cancel the platform's own revert, the queue stays held
        until you upload a fix or run the benchmark again:
        `state["claims_held"]`). 409 when none is pending, and when the
        update's swap stopped half way and the platform cannot tell how to
        put the folders back (nothing is changed then). Settings the request
        had already applied (a bundle's `challenge.toml`, an overview) stay.
        Returns the state (`env_update`).
        """
        return self._json_call(
            "DELETE", f"/creator_challenge/challenge/{challenge_id}/env/update",
            error_label="cancel_env_update",
        )

    def revert_env(self, challenge_id: int) -> dict:
        """Put the previous version of a challenge's environment files back.

        Mirrors `POST /api/creator_challenge/challenge/{id}/env/revert`. The
        files come back byte for byte (`challenge.toml` excepted: it stays
        the applied settings' file). A revert is an update like any other: it
        waits for the running runs and is recorded as a `revert` revision.
        On a started challenge the previous version is the last one that was
        released to participants, and going back to it is not checked again
        (those files served); files that were never released are checked
        like any update (the benchmark, or the new pod's load) before a
        participant's run starts. The files a revert replaces become the
        previous version when they are worth returning to (released, on a
        started challenge), so a second call then undoes the first. 409 when
        no previous version is kept, when it never served a started
        challenge, failed its check or is the current one, when its folder
        is no longer what was recorded, or while an update is pending; 409
        with `.body["reason"] == "out_of_credits"` on a started challenge whose owner has no compute
        credits left, as every update of its files (`upload_env_file`).
        Returns the state (`env_update`).
        """
        return self._json_call(
            "POST", f"/creator_challenge/challenge/{challenge_id}/env/revert",
            error_label="revert_env",
        )

    def env_revisions(self, challenge_id: int, *, offset: int = 0, limit: int = 30) -> dict:
        """The history of a challenge's environment files, newest first.

        Mirrors `GET /api/creator_challenge/challenge/{id}/env/revisions`.
        Returns `{"revisions": [...], "total", "offset", "limit"}`; each
        revision is the shape `env_update` documents: what changed the files
        (`transport`), who (`user`), when (`created_at_ts`, `applied_at_ts`),
        the benchmark's verdict, and whether it is the live, the previous or
        the pending version (`is_live`, `is_previous`, `is_pending`) or was
        reverted (`reverted_at_ts`).
        """
        return self._json_call(
            "GET", f"/creator_challenge/challenge/{challenge_id}/env/revisions",
            params={"offset": offset, "limit": limit}, error_label="env_revisions",
        )

    def cancel_runs_in_flight(self, challenge_id: int) -> dict:
        """Cancel the runs a stopped challenge still has on their way and nobody will play.

        Mirrors `POST
        /api/creator_challenge/challenge/{id}/runs/cancel_in_flight`. The way
        out when `start_challenge(id, clean_redeploy_all=True)` answers 409
        because of runs that will never end: a pending run on a queue with no
        poller or a disabled machine, a running row the platform lost track
        of. A stopped or ended challenge only (409 otherwise). Cancels every
        pending run and those lost running rows; a cancelled run scores
        nothing and moves no rating. A participant's deploy that was open
        fails as the platform's (`failure_owner: "platform"`, the submission
        `deploy_failed`) and stays charged to their daily deploy quota
        (`failed_participant_deploys` in the reply;
        `scoring_update(id)["stop"]["participant_deploys_in_flight"]`
        beforehand): follow with `start_challenge(id,
        clean_redeploy_all=True)`, which redeploys those submissions as
        staff, or the participant spends another deploy. A run a worker
        still holds is left to end. Returns `{"cancelled_pending_runs", "cancelled_running_runs",
        "failed_participant_deploys", "runs_in_flight"}` (the last: still on
        their way after the call).
        `scoring_update(id)["can_cancel_runs_in_flight"]` says whether the
        call would do something.
        """
        return self._json_call(
            "POST", f"/creator_challenge/challenge/{challenge_id}/runs/cancel_in_flight",
            error_label="cancel_runs_in_flight",
        )

    def archive_challenge(self, challenge_id: int) -> dict:
        """Archive a stopped or ended challenge — its owner or an admin.

        Mirrors `POST /api/creator_challenge/challenge/{id}/archive`. The
        challenge no longer counts among your challenges
        (`workspace()["quotas"]["owned_challenge_count"]`) and its workspace
        row carries `archived_at_ts`. Nothing else changes: every result, the
        leaderboard and the public URL stay, and its editors still open it.
        Returns the challenge (the `creator_challenge` shape, `archived_at_ts`
        set). 409 when it is already archived, started (stop or end it first)
        or a draft (`delete_challenge` instead). Archiving is about your
        side only: an ended challenge stays among the past challenges of the
        public catalog.
        """
        return self._json_call(
            "POST", f"/creator_challenge/challenge/{challenge_id}/archive",
            error_label="archive_challenge",
        )

    def unarchive_challenge(self, challenge_id: int) -> dict:
        """Take a challenge out of the archive — its owner or an admin.

        Mirrors `POST /api/creator_challenge/challenge/{id}/unarchive`. It
        counts among your challenges again and can be restarted. Returns the
        challenge; 409 when it is not archived.
        """
        return self._json_call(
            "POST", f"/creator_challenge/challenge/{challenge_id}/unarchive",
            error_label="unarchive_challenge",
        )

    def challenge_deletion(self, challenge_id: int) -> dict:
        """What `delete_challenge` would do, without doing it — owner or admin.

        Mirrors `GET /api/creator_challenge/challenge/{id}/deletion`
        (`ChallengeDeletionOut`). Returns `can_delete` and what decides it:
        `status` (only a `draft` is deleted), `submission_count` (submissions
        other than the benchmark and the deleted ones), `team_count`,
        `lesson_count` and `lessons` (`[{id, title, module_id, course_id}]`,
        in id order: the lessons, anyone's, whose text names the challenge in
        a directive; remove its card from each first), `runs_in_flight` (runs
        pending or running, a benchmark test on its way); and what goes with
        the challenge: `module_deleted` (a course challenge's module is
        deleted with it, because it then holds no lesson and no other
        challenge; False for a challenge with no course), `env_files` (names in its env
        folder), `benchmark_files` (names in its benchmark folder),
        `has_benchmark`, `run_count` (every run of its environment),
        `exercise_count`, `chat_session_count`, `deleted_datasets` (`[{id,
        label, file_count}]`, attached to this challenge alone: deleted with
        their stored files) and `kept_datasets` (attached to another
        challenge too). `replay_count` replay files of those runs are not
        removed: they expire from storage after 365 days. `storage_left` is
        `[]` here.
        """
        return self._json_call(
            "GET", f"/creator_challenge/challenge/{challenge_id}/deletion",
            error_label="challenge_deletion",
        )

    def delete_challenge(self, challenge_id: int) -> dict:
        """Delete your own draft challenge — its owner or an admin.

        Mirrors `DELETE /api/creator_challenge/challenge/{id}`. Only a draft
        with no submission but the benchmark, no team, no run on its way and
        no lesson naming it in a directive. 409 otherwise, with `status`,
        `submission_count`, `team_count`, `lesson_count`, `lessons` (the
        lessons to remove its card from) and `runs_in_flight` in `.body`; a
        challenge that ran is archived instead (`archive_challenge`). Deletes
        its rows (configuration, environment, evaluation, tags, assistants, a
        chat challenge's conversations, an exercise set's exercises and their
        attempts), its benchmark submission, every run of its environment,
        its folder (env files, overview, cover, benchmark files) and the
        datasets attached to it alone with their stored files. A course
        challenge's module is deleted with it when it then holds no lesson
        and no other challenge (`module_deleted`). Returns what
        `challenge_deletion` announced (`ChallengeDeletionOut`), with
        `storage_left`: what could not be removed from storage although the
        rows are gone (`[]` when all of it went; check it).
        """
        return self._json_call(
            "DELETE", f"/creator_challenge/challenge/{challenge_id}",
            error_label="delete_challenge",
        )

    def agent_template(self, challenge_id: int) -> dict:
        """The agent.py template handed to the challenge's participants.

        Mirrors `GET /api/creator_challenge/challenge/{id}/agent-template`
        (`AgentTemplateOut`): returns `{"agent_template": str}`, the stored
        template or, when the challenge has none, the platform's default one
        — what the console's template editor shows. `creator_challenge()
        ["configuration"]["agent_template"]` is the raw column (null when
        unset). Raises `ChallengeNotFoundError` for a challenge you cannot
        edit.
        """
        return self._creator_get(
            challenge_id, "/agent-template", "agent_template"
        )

    def update_agent_template(self, challenge_id: int,
                              agent_template: str) -> dict:
        """Set the default agent.py template handed to participants.

        Mirrors `PUT /api/creator_challenge/challenge/{id}/agent-template`.
        The backend rejects this while the challenge is started, so stop it
        first (`stop_challenge`) and re-start afterwards if needed.
        """
        resp = self._request("PUT",
            self._url(
                f"/creator_challenge/challenge/{challenge_id}/agent-template"
            ),
            headers=self._headers(json_body=True),
            json={"agent_template": agent_template},
            timeout=30,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "update_agent_template", resp)
        return resp.json()

    def csv_ground_truth(self, challenge_id: int) -> dict:
        """The parsed CSV ground-truth metadata of a file challenge.

        Mirrors `GET /api/creator_challenge/challenge/{id}/csv-ground-truth`
        (`agent_template.py`): the columns and separator the challenge's
        `agent_template` declares, as the console shows them beside the
        benchmark upload. Returns `{"ground_truth": {...}}`, or
        `{"ground_truth": null}` when the template declares none.
        """
        return self._creator_get(
            challenge_id, "/csv-ground-truth", "csv_ground_truth"
        )

    def scoring_form(self, challenge_id: int) -> dict:
        """The scoring form of a file challenge: everything the form shows.

        Mirrors `GET /api/creator_challenge/challenge/{id}/scoring-form`
        (`scoring_form.py`). The form generates a plain `env.py` from a
        metric, the scored column and the id column, so a file challenge
        needs no Python. Returns:

        - `metrics`: the metrics offered, each `{scoring_metric, label,
          order, prediction, prediction_help}`: `accuracy`, `f1_macro`,
          `rmse`, `mae`, `auc`, `log_loss`. `order` (`asc` / `desc`) is the
          board's direction and comes with the metric; `prediction` is what a
          participant writes in the scored column (`label`, `number`,
          `probability`).
        - `scoring_metric`, `scoring_target_column`, `scoring_id_column`: the
          last choice env.py was generated from (all None before the first),
          and `submission_filename`.
        - `env_state`: `absent`, `default` (the untouched starting template),
          `generated` (still the form's text) or `edited` (your own code).
          `can_generate` says whether `set_scoring_form` would be accepted;
          when not, `read_only_reason` (`env_edited`, `config_file`) and
          `read_only_message` say why.
        - `ground_truth`: `{filename, columns, separator,
          suggested_id_column}` of the uploaded `y_test.csv`
          (`ground_truth_filename`), None until it is uploaded.
          `suggested_id_column` is the id column to pass to
          `set_scoring_form` when it is obvious: the only column whose name
          says "id" (`id`, `row_id`, `PassengerId`) and whose values name
          each row once, else None;
          `ground_truth_error` when the file there cannot be read or does not
          fit the recorded choice. `ground_truth_max_rows`: the most rows the
          form reads (a larger ground truth is refused at upload).
        - `can_generate_baseline`: `generate_scoring_baseline` will succeed.
        - `participant_messages_notice`: what a refused submission is told
          (the row count, the first missing id, whether an id exists: so the
          ground truth's ids are visible to participants; no value of the
          scored column ever is).

        A 400 on a challenge that is not a file challenge.
        """
        return self._creator_get(challenge_id, "/scoring-form", "scoring_form")

    def set_scoring_form(self, challenge_id: int, scoring_metric: str,
                         scoring_target_column: str, scoring_id_column: str,
                         *, submission_filename: str | None = None) -> dict:
        """Generate a file challenge's `env.py` from a metric and two columns.

        Mirrors `PUT /api/creator_challenge/challenge/{id}/scoring-form`.
        Writes `env.py` into the env folder (standard library only: rows of
        the submission are matched to `y_test.csv` on `scoring_id_column`
        and `scoring_target_column` is scored with `scoring_metric`), and
        sets the board's label and direction for that metric. Returns the
        form's new state (`scoring_form`).

            client.set_scoring_form(cid, "rmse", "price", "house_id")
            client.upload_env_file(cid, "y_test.csv")
            client.generate_scoring_baseline(cid)
            client.run_benchmark(cid)

        What the generated file accepts, a participant's upload accepts: the
        upload is checked by the same reader. Each file's separator is
        detected (comma, semicolon, tab, pipe), a byte-order mark and other
        columns are ignored, rows come in any order, and a whole-number id
        or class label has one spelling (`1` = `1.0`; any other text must
        match exactly, so `007` is not `7`). A number is finite and at most
        1e100 in magnitude. A file keeps to fixed limits: a line of at most
        1 000 000 characters, 1 000 columns each named once (`id,y,y` is
        refused), 1 000 empty lines. A column name may hold any character: it reaches
        the file as a string literal only.

        `submission_filename` (a `.csv` name) is optional; the challenge
        keeps its own otherwise. The ground truth may come before or after:
        whichever comes second is checked against the other, and a missing
        column is a 400 that names it.

        The generated file is yours. Calling this again (another metric,
        another column) rewrites it only while it is byte-identical to what
        was generated: after a hand edit the call is a 409 (`MLArenaError`)
        and nothing changes; delete `env.py` to use the form again. While
        the challenge is started the columns can still be changed (an update
        of the environment files, checked before it serves: see
        `env_update`); another metric or submission file name is a 409 then
        (stop the challenge to change them). A 400 when `challenge.toml`
        holds `metrics`, or for a bad name. A 409 too when other writes of
        the environment files kept landing while this one was being
        prepared (it is prepared again up to three times first): call it
        again; and, on a started challenge whose owner has no compute
        credits left, the 409 of every update of its files
        (`.body["reason"] == "out_of_credits"`, see `upload_env_file`).
        """
        body = {
            "scoring_metric": scoring_metric,
            "scoring_target_column": scoring_target_column,
            "scoring_id_column": scoring_id_column,
        }
        if submission_filename is not None:
            body["submission_filename"] = submission_filename
        resp = self._request("PUT",
            self._url(
                f"/creator_challenge/challenge/{challenge_id}/scoring-form"
            ),
            headers=self._headers(json_body=True),
            json=body,
            timeout=30,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "set_scoring_form", resp)
        return resp.json()

    def generate_scoring_baseline(self, challenge_id: int) -> dict:
        """Write the scoring form's baseline into the benchmark files.

        Mirrors `POST /api/creator_challenge/challenge/{id}/scoring-form/baseline`.
        The baseline is a submission that predicts one constant on every row:
        the most frequent class (`accuracy`, `f1_macro`), the mean (`rmse`,
        `mae`) or the share of the positive class (`auc`, `log_loss`). It is
        never the ground truth. It replaces the benchmark's submission file;
        `run_benchmark` then scores it, which is what Start asks for.

        Returns `{message, filename, prediction, row_count,
        replaced_previous}`: `replaced_previous` is True when a benchmark
        file with rows was there and was overwritten. Needs the form
        applied (`set_scoring_form`) and `y_test.csv` uploaded (400
        otherwise); a 409 once `env.py` was edited by hand.

        The benchmark is your reference, not a participant: it is on no
        leaderboard. Read its score with `benchmark_status` or
        `creator_submissions`.
        """
        resp = self._request("POST",
            self._url(
                f"/creator_challenge/challenge/{challenge_id}/scoring-form/baseline"
            ),
            headers=self._headers(),
            timeout=60,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "generate_scoring_baseline", resp)
        return resp.json()

    def create_dataset(self, challenge_id: int, label: str,
                       description: str | None = None) -> dict:
        """Create a public dataset bucket on a challenge (author scope).

        Mirrors `POST /api/creator_challenge/challenge/{id}/datasets`
        (`datasets.py`, `create_creator_dataset`). Returns
        `{"id", "label", "description"}`; use the returned `id` with
        `upload_dataset_file`. Accepted on a started challenge too, except
        one that takes no update in place (409: a started challenge on a
        `vm` machine). Files uploaded here are served to participants via `datasets()`.

        Requires an `author`-scope token.
        """
        body: dict = {"label": label}
        if description is not None:
            body["description"] = description
        resp = self._request("POST",
            self._url(f"/creator_challenge/challenge/{challenge_id}/datasets"),
            headers=self._headers(json_body=True),
            json=body,
            timeout=30,
        )
        self._handle_response(resp)
        if resp.status_code not in (200, 201):
            raise _failed(MLArenaError, "create_dataset", resp)
        return resp.json()

    def creator_datasets(self, challenge_id: int) -> dict:
        """List a challenge's datasets + files from the authoring side.

        Mirrors `GET /api/creator_challenge/challenge/{id}/datasets`
        (`datasets.py`, `get_creator_datasets`). Like `datasets()` but
        author-scoped (works on your own hidden challenges). Returns
        `{"datasets": [{"id","label","files":[…]}]}`.

        Requires an `author`-scope token.
        """
        resp = self._request("GET",
            self._url(f"/creator_challenge/challenge/{challenge_id}/datasets"),
            headers=self._headers(),
            timeout=30,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "creator_datasets", resp)
        return resp.json()

    def upload_dataset_file(self, challenge_id: int, dataset_id: int,
                            file_path: str) -> dict:
        """Upload one file into a challenge dataset (author scope).

        Multipart POST to
        `/api/creator_challenge/challenge/{id}/datasets/{dataset_id}/files`
        (`datasets.py`, `upload_dataset_file`); the file is stored in R2 and
        exposed to participants through `datasets()` as a signed
        `download_url`, started or not (409 on a started challenge that
        takes no update in place). Returns `{"id", "label",
        "file_size_bytes"}`. A file uploaded under a name the dataset
        already holds replaces that file (same `id`, new size).

        Refused on the file's size before it is stored: 400 above 1 GiB a
        file (`.body["size_bytes"]`, `["max_bytes"]`); 409 with `reason:
        "datasets_total_cap"` when it would take the datasets of the
        challenge's owner over 20 GiB in total, all their challenges together
        (`.body["used_bytes"]`, `["size_bytes"]`, `["max_bytes"]`; a dataset a
        copy shares with its source counts on the source's owner; an admin
        owner is exempt): remove a dataset, or ask for more.

        Requires an `author`-scope token.
        """
        if not os.path.isfile(file_path):
            raise MLArenaError(f"File not found: {file_path}")
        with open(file_path, "rb") as fh:
            resp = self._request("POST",
                self._url(
                    f"/creator_challenge/challenge/{challenge_id}/datasets/{dataset_id}/files"
                ),
                headers=self._headers(),
                files={"file": (os.path.basename(file_path), fh)},
                timeout=300,
            )
        self._handle_response(resp)
        if resp.status_code not in (200, 201):
            raise _failed(MLArenaError, "upload_dataset_file", resp)
        return resp.json()

    def update_dataset(self, challenge_id: int, dataset_id: int, *,
                       label: str | None = None,
                       description: str | None = None) -> dict:
        """Update a dataset's label / description (author scope).

        PUT to `/api/creator_challenge/challenge/{id}/datasets/{dataset_id}`
        (`datasets.py`, `update_creator_dataset`). Only fields explicitly
        passed are sent. Allowed while the challenge is started (text only; its files stay locked). Returns `{"id", "label", "description"}`.

        Requires an `author`-scope token.
        """
        body: dict = {}
        if label is not None:
            body["label"] = label
        if description is not None:
            body["description"] = description
        if not body:
            raise MLArenaError("update_dataset requires at least one field")
        resp = self._request("PUT",
            self._url(
                f"/creator_challenge/challenge/{challenge_id}/datasets/{dataset_id}"
            ),
            headers=self._headers(json_body=True),
            json=body,
            timeout=30,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "update_dataset", resp)
        return resp.json()

    def delete_dataset(self, challenge_id: int, dataset_id: int) -> dict:
        """Delete a dataset and every file in it (author scope).

        Mirrors `DELETE
        /api/creator_challenge/challenge/{id}/datasets/{dataset_id}`
        (`datasets.py`). Refused once the challenge has started: participants
        may already have trained on it.
        """
        return self._creator_delete(
            challenge_id, f"/datasets/{dataset_id}", "delete_dataset"
        )

    def delete_dataset_file(self, challenge_id: int, dataset_id: int,
                            file_id: int) -> dict:
        """Delete one file from a challenge dataset (author scope).

        DELETE to
        `/api/creator_challenge/challenge/{id}/datasets/{dataset_id}/files/{file_id}`
        (`datasets.py`, `delete_dataset_file`); removes the DB row and its
        R2 object. Rejected once the challenge is started. Use this before re-uploading a file with the
        same name — `upload_dataset_file` always adds a new row, so replacing a
        file means delete-then-upload. Get `file_id` from `creator_datasets()`.

        Requires an `author`-scope token.
        """
        resp = self._request("DELETE",
            self._url(
                f"/creator_challenge/challenge/{challenge_id}/datasets/{dataset_id}/files/{file_id}"
            ),
            headers=self._headers(),
            timeout=60,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "delete_dataset_file", resp)
        return resp.json()

    # ---- Authoring: runs, submissions and assistants (author scope) --------

    def creator_runs(self, challenge_id: int, offset: int = 0,
                     limit: int = 30) -> dict:
        """One page of the challenge's runs, newest first, with the env side's
        diagnostics.

        Mirrors `GET /api/creator_challenge/challenge/{id}/runs` (`runs.py`).
        Returns `{"runs": [...], "total": N, "offset": offset, "limit":
        limit}`: `total` is every run of the challenge, `offset` / `limit`
        (at most 100) the page — `creator_runs(cid, offset=30)` is the page
        after the 30 most recent. The rows are the same `RunResult` the
        submission status and the admin runs page serve, but served to the
        challenge's staff with the environment's error, its stdout and every agent's row
        inline. Includes benchmark tests, deployment games and participant runs.
        """
        return self._creator_get(
            challenge_id, "/runs", "creator_runs",
            params={"offset": offset, "limit": limit},
        )

    def creator_submissions(self, challenge_id: int) -> dict:
        """Every non-deleted submission on a challenge you own.

        Mirrors `GET /api/creator_challenge/challenge/{id}/submissions`
        (`submissions.py`). Returns `{"submissions": [...], "metrics":
        [MetricSpec, ...]}` — `metrics` is the challenge's declaration, and
        `rank` follows its ranking spec's `order`; each row carries
        `submission_id`, `submission_name`, `user_id`, `username`,
        `team_id` and `team_name` (the owner's team on the challenge, None
        for a solo participant), `failure_owner` and `error_type` (the cause
        of a `deploy_failed` row, the pair `challenge_deploy_failures()`
        groups by: filter the rows on them to list a group; both None on a
        row that is not `deploy_failed` and on the benchmark),
        `created_at_ts`, `last_end_run_ts`, `score` (the ranking value — the
        rating on a rated board), `metrics` (`{key: value}` for every
        declared key), `number_of_runs`, `rank`, `restart_allowed` and the
        status block.
        `rank` is the place of the submission's row on the challenge
        leaderboard, one row per participant (the team's or your best active
        submission, as `leaderboard()` ranks it by default); None for a
        submission that has no row there. A course challenge's submissions
        are its class's: there is no course filter (4.23.0 removed
        `course_id`).
        """
        return self._creator_get(challenge_id, "/submissions", "creator_submissions")

    def challenge_overview(self, challenge_id: int) -> dict:
        """The numbers of a challenge you own or assist on.

        Mirrors `GET /api/creator_challenge/challenge/{id}/overview`
        (`follow.py`). Returns `{"challenge": {...}, "counters": {...}}`.

        `challenge` is the challenge's reference, the shape every payload
        that names a challenge carries: `id`, `name`, `slug`, `status`,
        `end_date_ts`, `is_public`, `kernel_version`, `kind_label`,
        `user_id`, `owner_username`, `role` (yours on it: `"owner"`,
        `"assistant"` or None), `can_edit` (whether you may open its
        workspace: its owner, an assistant, or an admin), `can_copy`
        (whether you may copy it into a course of yours, as on
        `challenge()`: where it is False, `add_course_challenge(
        copy_from_challenge_id=…)` and `use_example_challenge` answer 404)
        and `statistics` (`unique_participants`: teams
        and solo users with an `active` submission; `running_submissions`).

        `counters` (a submission here is neither deleted nor the benchmark):
        `team_count` (teams on the challenge, with or without a submission:
        not a participant count); `submissions_created_7d`, the submissions created in the
        last 7 UTC days, today included (new submissions, not deploys), and
        `submissions_created_daily`, the same per day, oldest first
        (`[{"day", "submissions_created"}]`, 7 entries);
        `deploy_failed_submissions`, the submissions whose status is
        `deploy_failed` now, and `deploy_failed_submissions_by_owner`
        (`participant`, `challenge`, `platform`, `unattributed`), both None
        on a kind that never deploys (chat, exercises: not "zero failures");
        `last_submission_created_at_ts`. A course challenge's course is
        `challenge(id)["course_id"]`: a challenge belongs to one course.
        """
        return self._creator_get(challenge_id, "/overview", "challenge_overview")

    def challenge_deploy_failures(self, challenge_id: int) -> dict:
        """The challenge's deploy-failed submissions, grouped by cause.

        Mirrors `GET /api/creator_challenge/challenge/{id}/deploy-failures`
        (`follow.py`). Returns `{"deploy_failed_submissions": N, "groups": [...]}`,
        largest group first. Listed: the submissions whose status is
        `deploy_failed` now (its latest deploy failed and it was not deployed
        again); the benchmark and deleted submissions are left out, and so is
        an `active` submission whose recent runs fail. On a kind that never
        deploys (chat, exercises) the count is None and `groups` is empty:
        the notion does not apply there.

        Each group: `failure_owner` (who the latest deploy's verdict blames:
        `participant`, `challenge`, `platform`; None when it carries no
        verdict), `error_type` (what the first run of that deploy recorded on
        the blamed side, e.g. `code_error`, `timeout`, `oom_killed`, `crash`,
        or a platform cause such as `pod_failed`; None when no run did),
        `deploy_failed_submissions` (the count), `sample_failure_message` (the
        first 300 characters of the most recent one) and `submission_ids`
        (at most 20, newest deploy first).
        """
        return self._creator_get(challenge_id, "/deploy-failures", "challenge_deploy_failures")

    def clean_redeploy_submission(self, challenge_id: int,
                                  submission_id: int) -> dict:
        """Wipe one submission's results and queue a fresh deployment.

        Mirrors `POST
        /api/creator_challenge/challenge/{cid}/submissions/{sid}/clean_redeploy`
        (`submissions.py`) — the Re-score action of the Studio's submissions
        tab. Allowed for an `active` or `deploy_failed` submission; anything
        else is a 400 naming the status. Returns `submission_id`,
        `deleted_results`, `deployment_id` and the submission's status block.
        The runs of a re-score are the challenge's owner's to pay: a 409
        whose `.body` carries `reason: "out_of_credits"` beside `error` when
        the owner has no compute credits left, and nothing is wiped.

        Use it after changing how a run is scored: the old results were
        produced under the old rules, and this is what makes the leaderboard
        comparable again.
        """
        resp = self._request("POST",
            self._url(
                f"/creator_challenge/challenge/{challenge_id}"
                f"/submissions/{submission_id}/clean_redeploy"
            ),
            headers=self._headers(),
            timeout=60,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "clean_redeploy_submission", resp)
        return resp.json()

    def clean_redeploy_all(
        self, challenge_id: int, *, dry_run: bool = False,
        scored_by_another_revision: bool = False,
    ) -> dict:
        """Re-score every eligible submission of a challenge (wipe its runs, deploy it again).

        Mirrors `POST
        /api/creator_challenge/challenge/{id}/submissions/clean_redeploy_all`
        (`submissions.py`). Returns `{"dry_run", "total", "runs_in_flight",
        "redeployed": [...], "failed": [{"submission_id", "error"}]}` — a
        per-submission failure does not stop the others, so retry only what
        `failed` names. `runs_in_flight` is how many runs of those
        submissions were pending or running when the call began: a
        submission with one is not re-scored (it is under `failed`).

        `dry_run=True` writes nothing and returns the counts: `total` is the
        number of submissions the call would re-score now (both lists empty),
        in any status. The real call needs a started challenge (409
        otherwise): to restart a stopped one and re-score it in one step use
        `start_challenge(id, clean_redeploy_all=True)`. It is also a 409,
        with `.body["reason"] == "out_of_credits"` beside `error`, when the
        challenge's owner has no compute credits left (the runs of a
        re-score are the owner's to pay): nothing is re-scored. A dry run is
        never refused for credits.

        `scored_by_another_revision=True` takes only the active submissions
        whose last scored run was scored by other environment files than the
        current ones: what an update of the env folder left on the old
        scoring (a submission scored before a change that was reverted is on
        the current files again and is left out). With `dry_run=True` this is
        the count behind "re-score N" after an update. The reply echoes the
        flag as `scored_by_another_revision`.
        """
        resp = self._request("POST",
            self._url(
                f"/creator_challenge/challenge/{challenge_id}"
                "/submissions/clean_redeploy_all"
            ),
            params={
                key: "true" for key, on in (
                    ("dry_run", dry_run),
                    ("scored_by_another_revision", scored_by_another_revision),
                ) if on
            } or None,
            headers=self._headers(),
            timeout=300,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "clean_redeploy_all", resp)
        return resp.json()

    def soft_delete_submission(self, challenge_id: int,
                               submission_id: int) -> dict:
        """Delete a participant's submission from a challenge you own.

        Mirrors `DELETE
        /api/creator_challenge/challenge/{cid}/submissions/{sid}`
        (`submissions.py`). Removes the submitter's files, cancels any deploy
        attempt still open and flips the row to `deleted` — the same effect as
        the participant's own `delete_submission()`, from the staff's side.
        Returns `submission_id` and the status block.
        """
        return self._creator_delete(
            challenge_id, f"/submissions/{submission_id}",
            "soft_delete_submission",
        )

    def challenge_assistants(self, challenge_id: int) -> list:
        """The assistants of a challenge.

        Mirrors `GET /api/creator_challenge/challenge/{id}/assistants`
        (`assistants.py`). Each entry carries `user_id`, `username` and
        `created_at_ts` (no email). An assistant has the owner's editing
        rights on this challenge. Every editor of the challenge (its owner,
        an assistant, an admin) may read the list; only the owner (or an
        admin) may change it. The owner is not an entry:
        `creator_challenge(id)["owner_username"]` names them. A challenge with no course
        only: a course challenge's staff are its course's (`course_assistants`),
        and this read, like the add, answers 400 on one.
        """
        return self._creator_get(
            challenge_id, "/assistants", "challenge_assistants"
        )

    def add_challenge_assistant(self, challenge_id: int, username: str) -> dict:
        """Grant a user the editing rights on this challenge — owner only.

        Mirrors `POST /api/creator_challenge/challenge/{id}/assistants`
        (`assistants.py`). Returns `message`, `user_id` and `username`.
        """
        resp = self._request("POST",
            self._url(f"/creator_challenge/challenge/{challenge_id}/assistants"),
            headers=self._headers(json_body=True),
            json={"username": username},
            timeout=30,
        )
        self._handle_response(resp)
        if resp.status_code not in (200, 201):
            raise _failed(MLArenaError, "add_challenge_assistant", resp)
        return resp.json()

    def remove_challenge_assistant(self, challenge_id: int,
                                   user_id: int) -> dict:
        """Remove an assistant from a challenge — owner only.

        Mirrors `DELETE
        /api/creator_challenge/challenge/{id}/assistants/{user_id}`
        (`assistants.py`).
        """
        return self._creator_delete(
            challenge_id, f"/assistants/{user_id}", "remove_challenge_assistant"
        )

    def search_challenge_assistant_candidates(self, challenge_id: int, q: str) -> list:
        """Accounts you may add as assistants of this challenge, by username.

        Mirrors `GET /api/creator_challenge/challenge/{id}/assistants/candidates`. `q` (two characters at least) is
        looked for anywhere in the username; at most six rows, those starting
        with `q` first, each `user_id`, `username`, `avatar_key`. The owner
        and the assistants already there are left out. Owner only.
        """
        return self._json_call(
            "GET", f"/creator_challenge/challenge/{challenge_id}/assistants/candidates", params={"q": q},
            error_label="search_challenge_assistant_candidates",
        )

    def challenge_assistant_invitations(self, challenge_id: int) -> list:
        """The assistant invitations mailed and not answered yet, newest first.

        Mirrors `GET /api/creator_challenge/challenge/{id}/assistants/invitations`. Each row carries `id`, `email`,
        `sender_username`, `created_at_ts`, `expires_at_ts` and `is_expired`.
        Owner only.
        """
        return self._json_call(
            "GET", f"/creator_challenge/challenge/{challenge_id}/assistants/invitations",
            error_label="challenge_assistant_invitations",
        )

    def invite_challenge_assistant(self, challenge_id: int, email: str) -> dict:
        """Mail `email` a link that makes whoever opens it an assistant of
        this challenge, whether or not the address has an account yet.

        Mirrors `POST /api/creator_challenge/challenge/{id}/assistants/invitations`. The link works once, for 14 days.
        Inviting an address that is already waiting renews its invitation
        (same `id`): a new link is mailed and the earlier one stops working.
        Returns the invitation row. Owner only.
        """
        return self._json_call(
            "POST", f"/creator_challenge/challenge/{challenge_id}/assistants/invitations",
            json_body={"email": email}, ok=(201,),
            error_label="invite_challenge_assistant",
        )

    def cancel_challenge_assistant_invitation(self, challenge_id: int,
                                           invitation_id: int) -> dict:
        """Cancel a pending assistant invitation: its link stops working.

        Mirrors `DELETE /api/creator_challenge/challenge/{id}/assistants/invitations/{invitation_id}`. Owner only.
        """
        return self._json_call(
            "DELETE", f"/creator_challenge/challenge/{challenge_id}/assistants/invitations/{invitation_id}",
            error_label="cancel_challenge_assistant_invitation",
        )

    def assistant_invitation(self, token: str) -> dict:
        """What an assistant invitation link invites to.

        Mirrors `GET /api/author/assistant_invitations/{token}` (`token` is
        the last part of the mailed link). Returns `challenge_id` / `course_id`
        (one of the two), `name`, `sender_username`, `expires_at_ts` and
        `already_editor` (you already edit it). A link that was used,
        cancelled or replaced raises `ChallengeNotFoundError` (404); an
        expired one `MLArenaError` (410).
        """
        return self._json_call(
            "GET", f"/author/assistant_invitations/{token}",
            error_label="assistant_invitation",
        )

    def accept_assistant_invitation(self, token: str) -> dict:
        """Accept an assistant invitation: you become an assistant of its
        challenge or course, and the link is used up.

        Mirrors `POST /api/author/assistant_invitations/{token}/accept`; a key
        of any scope. Returns `challenge_id` / `course_id` (one of the two)
        and `name`. Note that a `user` key does not reach the authoring
        routes afterwards: mint an `author` key.
        """
        return self._json_call(
            "POST", f"/author/assistant_invitations/{token}/accept",
            error_label="accept_assistant_invitation",
        )

    # ---- Submissions (user scope) ----

    def create_submission(self, challenge_id: int, submission_name: str,
                          copy_from_submission_id: int | None = None, *,
                          allow_same_commit: bool = False) -> dict:
        """Create a new submission for `challenge_id`.

        Mirrors `POST /api/submissions/challenge/{cid}` (`create_delete.py`).
        With `copy_from_submission_id`, the files of one of your existing
        submissions are copied into the new one, with its agent runtime when
        this challenge offers it (else the challenge's default). Returns
        `{"message", "submission_id", "status", "validation_message",
        "template_source", ...}`.

        A 404 is `ChallengeNotFoundError`, or `SubmissionNotFoundError` when
        `copy_from_submission_id` is given: the source ("Source submission not
        found", also for a submission that is not yours) is the id most
        likely to be wrong on that call. Requires a `user`-scope token.

        From a GitHub Actions run the submission records the commit it was
        built from (`submission()["source_revision"]`), and a second create
        for the same commit on the same challenge is refused: 409, a
        `SubmissionError` whose `.body["submission_id"]` is the submission
        that commit already made. `allow_same_commit=True` creates another
        one anyway (a training run that is not deterministic).
        """
        body: dict = {"submission_name": submission_name}
        if copy_from_submission_id is not None:
            body["copy_from_submission_id"] = copy_from_submission_id
        if allow_same_commit:
            body["allow_same_commit"] = True
        resp = self._request("POST",
            self._url(f"/submissions/challenge/{challenge_id}"),
            headers=self._headers(json_body=True),
            json=body,
            timeout=60,
        )
        self._handle_response(
            resp,
            not_found=(SubmissionNotFoundError if copy_from_submission_id is not None
                       else ChallengeNotFoundError),
        )
        if resp.status_code not in (200, 201):
            raise _failed(SubmissionError, "create_submission", resp)
        return resp.json()

    def upload_submission_file(self, challenge_id: int, submission_id: int,
                               file_path: str) -> dict:
        """Upload (or overwrite) a single file in an existing submission.

        The file is stored under its basename, so the name matters: a
        ``file_v1`` challenge accepts exactly the file it configured, and
        anything else is rejected with ``"<name> file not found"``. Read the
        name from ``challenge(challenge_id)["entry_filename"]`` rather than
        assuming ``submission.csv`` — a challenge may ask for ``pitch.txt`` or
        ``submission.csv.gz``; on a code challenge it is ``agent.py``, plus any
        helper files.

        Returns the server response: ``message``, ``validation_message`` and the
        status block (see ``submission()``).
        Note: the file route returns HTTP 200 even when validation *rejects* the
        submission — the file is saved but the whole submission is re-validated and
        may come back ``status == "upload_failed"`` with an actionable
        ``validation_message``. A 2xx here is not proof the submission is deployable;
        read ``is_deployable`` (``submit()`` checks it once, on
        ``submission()``, after every pre-deploy step and before deploying)."""
        if not os.path.isfile(file_path):
            raise SubmissionError(f"File not found: {file_path}")
        with open(file_path, "rb") as fh:
            resp = self._request("PUT",
                self._url(
                    f"/submissions/challenge/{challenge_id}/{submission_id}/file"
                ),
                headers=self._headers(),
                files={"file": (os.path.basename(file_path), fh)},
                timeout=120,
            )
        self._handle_response(resp, not_found=SubmissionNotFoundError)
        if resp.status_code not in (200, 201):
            raise _failed(SubmissionError, "upload_submission_file", resp)
        return resp.json()

    def deploy_submission(self, challenge_id: int, submission_id: int) -> dict:
        """Trigger deployment of an uploaded submission.

        Mirrors `PUT /api/submissions/challenge/{cid}/{sid}/deploy`
        (`deploy.py`). The backend answers 202 with `message`,
        `deployment_id` and the status block as the transaction left it
        (`status == "deploy_queue"`), so no second read is needed.

        A refusal is a 409 raised as `SubmissionError`: the submission is not
        in a deployable state (`.body` is `{"error"}`), the daily deploy
        quota is used up, the participant already holds
        `max_active_submissions` slots, or the challenge's owner has no
        compute credits left. For a quota or a credits refusal the body, on
        the error's `.body`, is `{"error", "reason", "deployment_limits",
        "active_submission_limits"}` — the same two blocks
        `submission_deploy_status()` returns, so the caller can read
        `next_deploy_available_at` or `active_submissions_remaining` off the
        exception without another call. `reason` is always there: None on a
        quota refusal, which the two blocks describe, and
        `"out_of_credits"` on a credits refusal: the owner of the challenge
        pays for the runs a deploy queues, so the deploy waits for their
        credits. That refusal uses none of your daily deploys, leaves the
        submission and its files as they are and names no figure (`error`
        is the sentence: "… your submission cannot be scored yet. It is
        saved; deploy it again later."). `EnrollmentRequiredError` on a
        course challenge whose course you are not enrolled in.
        """
        resp = self._request("PUT",
            self._url(
                f"/submissions/challenge/{challenge_id}/{submission_id}/deploy"
            ),
            headers=self._headers(),
            timeout=60,
        )
        self._handle_response(resp, not_found=SubmissionNotFoundError)
        if resp.status_code not in (200, 201, 202):
            raise _failed(SubmissionError, "deploy_submission", resp)
        return resp.json()

    def submission_deploy_status(self, challenge_id: int, submission_id: int) -> dict:
        """Read the two deploy quotas of a submission's owner.

        Mirrors `GET /api/submissions/challenge/{cid}/{sid}/deploy`
        (`deploy.py`).

        - `deployment_limits` — `daily_deploy_limit`, `daily_deploys_used`,
          `daily_deploys_remaining`, `next_deploy_available_at`, `can_deploy`.
          The same block `my_submissions()` returns.
        - `active_submission_limits` — `max_active_submissions`,
          `active_submissions_count`, `active_submissions_remaining`.
          `active_submissions_count` counts the participant's (or their
          team's) submissions on this challenge that are active *or*
          deploying (`deploy_queue` / `deploy_run`): a deploy in flight
          already holds one of the slots.

        The latest attempt is the submission's own:
        `submission(submission_id)["latest_deploy"]` (4.6.0 moved it there).
        """
        resp = self._request("GET",
            self._url(
                f"/submissions/challenge/{challenge_id}/{submission_id}/deploy"
            ),
            headers=self._headers(),
            timeout=30,
        )
        self._handle_response(resp, not_found=SubmissionNotFoundError)
        if resp.status_code != 200:
            # `raise_for_status()` used to end this call, which threw away the
            # server's reason with the body.
            raise _failed(SubmissionError, "submission_deploy_status", resp)
        return resp.json()

    def delete_submission(self, challenge_id: int, submission_id: int) -> dict:
        """Delete a submission (`DELETE /api/submissions/challenge/{cid}/{sid}`).

        The row is soft-deleted, its files are removed from storage and any
        deploy attempt still open is cancelled — one route, one effect. (The
        console used to have a second delete route of its own that left the
        files behind.)

        A chat challenge's submission cannot be deleted: it is the team's
        score line, and the backend answers 409 (`SubmissionError`, with
        `kernel_version` on `.body`).
        """
        resp = self._request("DELETE",
            self._url(
                f"/submissions/challenge/{challenge_id}/{submission_id}"
            ),
            headers=self._headers(),
            timeout=30,
        )
        self._handle_response(resp, not_found=SubmissionNotFoundError)
        if resp.status_code != 200:
            raise _failed(SubmissionError, "delete_submission", resp)
        return resp.json()

    # ---- Runner / runtime selection (DockerImageAgentRuntime) ----

    def agent_runtimes(self, kernel_version: str = "flex_v1"):
        """The agent runtime images of a kernel: the catalogue an author picks
        a challenge's offered set from (`update_settings(agent_runtime_ids=…,
        default_agent_runtime_id=…)`).

        Mirrors `GET /api/agent-runtimes?kernel_version=` — the list behind
        the console's Agent runtimes setting. One row per image, the entry
        shape `runtime_options()` lists: `{id, name, language, language_version,
        framework, framework_version, custom_pkg, requirement, display_name,
        packages}` (`custom_pkg` is "none" on a generic image, an env family's
        name such as "v1kart" on one built for that family),
        as a DataFrame when pandas is installed (a list of dicts otherwise).
        A runtime synced after the challenge was created is offered only once
        an author adds it to the set. An unknown kernel answers 400.

        Requires an `author`-scope token.
        """
        resp = self._request("GET", self._url("/agent-runtimes"),
                             params={"kernel_version": kernel_version},
                             headers=self._headers(), timeout=30)
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "agent_runtimes", resp)
        return _to_dataframe(resp.json())

    def runtime_options(self, challenge_id: int):
        """The agent runtimes (language × framework × version) this challenge offers.

        The challenge's own `runtime_options` (`challenge(challenge_id)`,
        `GET /api/challenges/{id}`): the set its staff chose
        (`update_settings(agent_runtime_ids=…)`), ordered by id — the only
        runtimes `set_agent_runtime` accepts on it. The one a new submission
        gets is `challenge(challenge_id)["default_agent_runtime_id"]`. Each
        entry: `{id, name, language,
        language_version, framework, framework_version, custom_pkg, requirement,
        display_name, packages}` — `name` is what `submit(..., runtime=...)`
        takes (`"python"`, `"sklearn"`, `"torch"`; distinct on a challenge),
        `display_name` how it reads
        (`"PyTorch"`, `"scikit-learn · XGBoost · LightGBM"`, `"No
        framework"`), `packages` its headline packages as `{name, version}`
        (`[]` for no framework); both None only on a runtime no image sync
        has written since the platform added them.
        """
        return _to_dataframe(self.challenge(challenge_id)["runtime_options"])

    def agent_runtime(self, submission_id: int) -> dict | None:
        """Agent runtime currently pinned to a submission, or None when it pins
        none: a file or chat challenge's submission runs no agent container.

        The submission's own `agent_runtime` (`submission(submission_id)`),
        the entry shape `runtime_options()` lists.
        """
        return self.submission(submission_id)["agent_runtime"]

    def set_agent_runtime(self, submission_id: int, runtime_id: int) -> dict:
        """Pin a `DockerImageAgentRuntime` onto a submission.

        Mirrors `PUT /api/submissions/agent_runtime/{sid}` (`runtime.py`,
        `update_agent_runtime`). The runtime must be one the challenge offers
        (`runtime_options(challenge_id)`): any other is refused with
        `SubmissionError` (400); an unknown id is `SubmissionNotFoundError`
        (404). Only while the submission is uploadable (409 after a deploy).
        """
        resp = self._request("PUT",
            self._url(f"/submissions/agent_runtime/{submission_id}"),
            headers=self._headers(json_body=True),
            json={"docker_image_agent_runtime_id": runtime_id},
            timeout=30,
        )
        self._handle_response(resp, not_found=SubmissionNotFoundError)
        if resp.status_code != 200:
            raise _failed(SubmissionError, "set_agent_runtime", resp)
        return resp.json()

    def resolve_runtime(self, challenge_id: int, name: str | None = None, *,
                        language: str | None = None,
                        framework: str | None = None,
                        framework_version: str | None = None) -> dict:
        """The runtime of this challenge named `name` ("python", "sklearn",
        "torch", ...): a challenge's offered runtimes have distinct names.

        Convenience over `runtime_options`; raises `MLArenaError` listing the
        offered names when none matches. What `submit(..., runtime="sklearn")`
        calls. The keyword form (`language=` / `framework=` /
        `framework_version=`, first match) is deprecated (4.8.0).
        """
        if name is None:
            warnings.warn(
                "resolve_runtime(language=, framework=, framework_version=) is deprecated, "
                "pass the runtime's name: resolve_runtime(challenge_id, \"sklearn\")",
                DeprecationWarning,
                stacklevel=2,
            )
        return _match_runtime(
            challenge_id, self.challenge(challenge_id)["runtime_options"], name,
            language=language, framework=framework,
            framework_version=framework_version,
        )

    # ---- File listing / reading / editing / deleting ----

    def list_submission_files(self, challenge_id: int, submission_id: int) -> dict:
        """List the files of a submission: metadata only.

        Mirrors `GET /api/submissions/challenge/{cid}/{sid}/file` (`file.py`,
        `get_submission_files`). Returns `{"files": [{name, size, role,
        language}, ...]}`, the entry file first, then the reports, then the
        rest, each group by name:

        - `role` — `entry` (the file the challenge runs or scores: `agent.py`,
          or a file challenge's `submission_filename`), `report` (a
          `README.md` in any case, rendered on the submission's page, or any
          `.pdf`), or `other`.
        - `language` — how the console highlights it (`python`, `markdown`,
          `plaintext`, …), or None when the file is not served as text (a
          binary or PDF, over 1 MB, or not UTF-8): download it with
          `download_submission_file()`.

        A text file's content is `get_submission_file_content()`. Readable by
        every reader of the submission (see `submission()`).
        """
        resp = self._request("GET",
            self._url(
                f"/submissions/challenge/{challenge_id}/{submission_id}/file"
            ),
            headers=self._headers(),
            timeout=30,
        )
        self._handle_response(resp, not_found=SubmissionNotFoundError)
        if resp.status_code != 200:
            raise _failed(SubmissionError, "list_submission_files", resp)
        return resp.json()

    def get_submission_file_content(self, challenge_id: int, submission_id: int,
                                    filename: str) -> str:
        """Fetch the raw text content of one file in a submission.

        Mirrors `GET /api/submissions/challenge/{cid}/{sid}/file/{filename}/content`
        (`file.py`, `get_file_content`). A file listed with `language: None`
        is not text: the backend answers 400 (`SubmissionError`); use
        `download_submission_file()`.
        """
        resp = self._request("GET",
            self._url(
                f"/submissions/challenge/{challenge_id}/{submission_id}/file/{filename}/content"
            ),
            headers=self._headers(),
            timeout=30,
        )
        self._handle_response(resp, not_found=SubmissionNotFoundError)
        if resp.status_code != 200:
            raise _failed(SubmissionError, "get_submission_file_content", resp)
        return resp.json()["content"]

    def update_submission_file_content(self, challenge_id: int, submission_id: int,
                                       filename: str, content: str) -> dict:
        """Write a text file by content (template render → upload).

        Multipart PUT to the same upload endpoint as `upload_submission_file`,
        carrying the content as the `file` part. Sibling of
        `update_env_file_content` on the authoring side. The backend validates
        the resulting submission folder and returns `{status, validation_message}`.
        """
        resp = self._request("PUT",
            self._url(
                f"/submissions/challenge/{challenge_id}/{submission_id}/file"
            ),
            headers=self._headers(),
            files={"file": (filename, io.BytesIO(content.encode("utf-8")))},
            timeout=120,
        )
        self._handle_response(resp, not_found=SubmissionNotFoundError)
        if resp.status_code not in (200, 201):
            raise _failed(SubmissionError, "update_submission_file_content", resp)
        return resp.json()

    def delete_submission_file(self, challenge_id: int, submission_id: int,
                               filename: str) -> dict:
        """Delete one file from a submission.

        The backend overloads the upload PUT with a `delete_file=<name>` form
        field — same endpoint, different verb-encoding (`file.py`,
        `update_attachment_file`).
        """
        resp = self._request("PUT",
            self._url(
                f"/submissions/challenge/{challenge_id}/{submission_id}/file"
            ),
            headers=self._headers(),
            data={"delete_file": filename},
            timeout=60,
        )
        self._handle_response(resp, not_found=SubmissionNotFoundError)
        if resp.status_code != 200:
            raise _failed(SubmissionError, "delete_submission_file", resp)
        return resp.json()

    def download_submission_file(self, challenge_id: int, submission_id: int,
                                 filename: str, dest_dir: str = ".") -> str:
        """Download one file of a submission to `dest_dir`; returns the path.

        Mirrors `GET /api/submissions/challenge/{cid}/{sid}/file/{filename}`
        (`file.py`, `download_submission_file`) — the console's download
        button. Unlike `get_submission_file_content()`, which decodes text,
        this writes the bytes as they are, so it is the only way to get model
        weights (`.pt`, `.pkl`, …) back out of a submission.

        Readable by every reader of the submission (see `submission()`).
        """
        os.makedirs(dest_dir, exist_ok=True)
        resp = self._request("GET",
            self._url(
                f"/submissions/challenge/{challenge_id}/{submission_id}/file/{filename}"
            ),
            headers=self._headers(),
            timeout=300,
            stream=True,
        )
        self._handle_response(resp, not_found=SubmissionNotFoundError)
        if resp.status_code != 200:
            raise _failed(SubmissionError, "download_submission_file", resp)
        out_path = os.path.join(dest_dir, os.path.basename(filename))
        with open(out_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    fh.write(chunk)
        return out_path

    def copyable_submissions(self) -> dict:
        """Your submissions that can seed a new one, across every challenge.

        Mirrors `GET /api/submissions/copyable_submissions`
        (`create_delete.py`) — what the console's "start from an existing
        submission" picker lists. Each entry carries `id`, `submission_name`,
        `challenge_id`, `challenge_name` and the status block; the `id` is what
        `create_submission(..., copy_from_submission_id=...)` takes, which
        until now the SDK accepted without being able to list the candidates.
        """
        resp = self._request("GET",
            self._url("/submissions/copyable_submissions"),
            headers=self._headers(),
            timeout=30,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(SubmissionError, "copyable_submissions", resp)
        return resp.json()

    # ---- The submission's page: the submission, its runs, its attempts ----

    def submission(self, submission_id: int) -> dict:
        """The submission, as every reader of its page sees it.

        Mirrors `GET /api/submissions/submission/{sid}` (`read.py`,
        `SubmissionOut`). Readable by its owner, a member of the owner's team
        on the challenge, the challenge's staff (its owner, its assistants,
        an admin), the owner's teacher, and anyone signed in when it is
        public; a 404 (`SubmissionNotFoundError`) for everyone else.

        Returns `id`, `challenge_id`, `submission_name`, `created_at_ts`,
        `is_public`, and:

        - the status block every submission payload carries: `status` (one
          of `created`, `upload_failed`, `upload_validated`, `deploy_queue`,
          `deploy_run`, `deploy_failed`, `active`, `deleted`), `phase`
          (`upload` | `deployment` | `active` | `terminal`),
          `last_status_message`, `status_update_ts`, and `is_uploadable` /
          `is_deployable` / `is_settled` — the server's own answer to "may I
          upload / deploy / stop polling". Read these instead of comparing
          `status` against a set of your own.
        - `owner` — `{user_id, username, avatar_key}`; `team` — the owner's
          team on the challenge, `{id, name, members: [{id, username,
          avatar_key}]}`, or None.
        - `viewer_role` — your standing on it: `owner` | `teammate` | `staff`
          | `public` (`staff`: the challenge's, a course challenge's being its
          course's owner and assistants). Only the owner writes (`update_submission`,
          uploads, deploy, delete).
        - `agent_runtime` — the pinned runtime (the `runtime_options()` entry
          shape), None on a kernel without agent containers.
        - `latest_deploy` — the newest attempt (`submission_deploys()` row:
          `id`, `created_at_ts`, `status` — `queued` | `running` |
          `succeeded` | `failed` | `cancelled` —, `finished_at_ts`,
          `failure_message`, `failure_owner`, `initiated_by`), None before the
          first deploy. `initiated_by` says who started it: `participant`
          (you, `deploy_submission`), `staff` (the challenge staff's
          `clean_redeploy_submission`) or `operator`; only `participant`
          attempts spend your daily deploy quota.
          `failure_owner` says who a `failed` attempt's verdict blames:
          `participant` (your code or file), `challenge` (the challenge's
          env, or the challenge changed under the attempt) or `platform`
          (the runners lost the runs).
        - `queue_info` — `queue_position`, `queue_total`, `created_at_ts`,
          `in_queue_for_second`; all-null when nothing is queued.
        - the numbers of its board line, under the leaderboard row's names:
          `score`, `score_ci95`, `metrics`, `metrics_window`,
          `number_of_runs`, `last_end_run_ts`, `action_time_max_sec`,
          `agent_metric_total_ram_max_bytes`, `agent_metric_vram_max_bytes`.
          Its rank is the board's: `leaderboard(cid, around=submission_id)`.
        - `latest_run_failure` — the newest scored run when it failed
          (`simulation_result_id`, `created_at_ts`, `job_status`,
          `job_error_type`, `env_error_type`, `agent_error_type`,
          `agent_error_message`), else None.

        `last_status_message`, `latest_deploy.failure_message` and
        `latest_run_failure.agent_error_message` quote the agent's own
        traceback: they are None for a `public` reader.
        """
        resp = self._request("GET",
            self._url(f"/submissions/submission/{submission_id}"),
            headers=self._headers(),
            timeout=30,
        )
        self._handle_response(resp, not_found=SubmissionNotFoundError)
        if resp.status_code != 200:
            raise _failed(SubmissionError, "submission", resp)
        return resp.json()

    def update_submission(self, submission_id: int, *,
                          submission_name: str | None = None,
                          is_public: bool | None = None) -> dict:
        """Rename your submission, publish or unpublish it, or both.

        Mirrors `PATCH /api/submissions/submission/{sid}` (`read.py`). Only
        the keywords passed are sent; passing neither is a 400. A name is 1 to
        40 characters on one line, without `<` or `>`. `is_public=True` opens
        every read of the submission — its code, runs and replays — to
        anyone signed in; its error messages stay yours. Returns the updated
        submission (the `submission()` shape).

        Owner only: anyone else gets a 404 (`SubmissionNotFoundError`). A
        deleted submission cannot be changed (409, `SubmissionError`).
        """
        body: dict = {}
        if submission_name is not None:
            body["submission_name"] = submission_name
        if is_public is not None:
            body["is_public"] = is_public
        resp = self._request("PATCH",
            self._url(f"/submissions/submission/{submission_id}"),
            headers=self._headers(json_body=True),
            json=body,
            timeout=30,
        )
        self._handle_response(resp, not_found=SubmissionNotFoundError)
        if resp.status_code != 200:
            raise _failed(SubmissionError, "update_submission", resp)
        return resp.json()

    def rename_submission(self, submission_id: int, name: str) -> dict:
        """Rename your submission: `update_submission(submission_id,
        submission_name=name)`."""
        return self.update_submission(submission_id, submission_name=name)

    def submission_runs(self, submission_id: int, *,
                        deploy_id: int | None = None,
                        is_test: bool | None = None,
                        offset: int = 0, limit: int = 20) -> dict:
        """The submission's runs, newest first, one page at a time.

        Mirrors `GET /api/submissions/submission/{sid}/runs` (`read.py`).
        Returns `{"runs", "total", "retention_days", "challenge_context"}`:

        - `runs` — the page (`offset`, `limit` 1..100) of the runs that
          match, `total` the count of every run that matches. Both cover the
          last `retention_days` days only (the replay bucket's retention).
        - `deploy_id` keeps one deploy attempt's runs (its validation and
          first scored runs; the ids are `submission_deploys()`'s); an
          attempt of another submission is a 404. `is_test=True` keeps the
          validation runs, `False` the others.
        - Each run is a `RunResult`, the one run model every run list on the
          API serves: the job's own `job_status` (`pending` | `running` |
          `completed` | `failed` | `cancelled`), `job_started_at_ts`,
          `job_completed_at_ts`, `job_error_type` (the platform's cause, set
          on a `failed` run and on a `pending` retry for its last lost
          attempt: `pod_failed` | `pod_deadline` | `worker_error` |
          `worker_shutdown` | `wire_mismatch` | `spec_error` | `lost`) +
          `job_error_message` (served to the challenge's staff only),
          `credits` (what the run was charged, `{amount, seconds, kind,
          resource}`,
          for the challenge's staff; None for anyone else and for a run
          that was not charged),
          `job_retry_count`, `simulation_result_id`, `submission_deploy_id`
          (the attempt that queued it), `created_at_ts`,
          `is_test`, `is_deployment`,
          `env_nb_steps`, the env's `step_time_*_sec` / `env_metric_*`
          columns and, when the challenge's own code is why the run ended,
          `env_error_type` (`code_error` | `timeout` | `oom_killed` |
          `crash`; the legacy `simulation_error` | `pod_crash` on older
          runs). `completed` means an outcome arrived: it can carry an
          `env_error_type`. `replay` = `{render_format, signed_url,
          render_size_bytes}` (None when the run recorded none;
          `download_replay` saves it).
        - `submission_results` lists one `RunSubmissionResult` per agent in
          the run — **this submission's is the row whose `submission_id` is
          its id**, the others are the opponents: `submission_id`,
          `submission_name`, `user_name`, `submission_status` (that
          submission's status now: `deleted` when it was deleted since),
          `agent_attached_player_id`, `env_player_name`, `score`,
          `score_variance`, `n_episodes`, `score_ci95`, `agent_nb_steps`,
          `game_outcome` and `final_rank` (both None on a solo run),
          `score_elo_before`, `score_elo_delta`, `metrics` (the env's
          per-run keys, plus the optional secondary `score2`),
          `info_message`, the agent's `action_time_*_sec` /
          `agent_metric_*` columns, its footprint (`agent_busy_time_sec`:
          `__init__` + every call; `agent_call_count`,
          `agent_call_time_total_sec`, `agent_metric_cpu_time_total_sec`,
          `agent_metric_ram_gib_sec`, `agent_metric_gpu_energy_j`,
          `agent_metric_gpu_busy_sec`, `agent_metric_vram_max_bytes`,
          `agent_energy_wh`, `agent_co2e_g`) and, when that agent is why the
          run ended, `agent_error_type` with `agent_error_message`. A None
          footprint value means the run was not metered (before the
          platform measured it, an older runtime image, an unmetered
          machine), never zero; the run's env side carries
          `env_metric_ram_gib_sec`, `env_metric_gpu_energy_j`,
          `env_metric_gpu_busy_sec`, `env_energy_wh`, `env_co2e_g` the same
          way.
        - `challenge_context` — the `RunsContext` (`metrics`,
          `number_of_agents`, `render_delay_second`).

        `agent_error_message` and `agent_stdout_logs` are None on a row that
        is not this submission's, and for a `public` reader: they quote the
        owner's own traceback and stdout. `env_error_message` and
        `env_stdout_logs` belong to the challenge's staff and are None
        here. Timestamps are ISO-8601 UTC with a `Z`.
        """
        params: dict = {"offset": offset, "limit": limit}
        if deploy_id is not None:
            params["deploy_id"] = deploy_id
        if is_test is not None:
            params["is_test"] = "true" if is_test else "false"
        resp = self._request("GET",
            self._url(f"/submissions/submission/{submission_id}/runs"),
            headers=self._headers(),
            params=params,
            timeout=30,
        )
        self._handle_response(resp, not_found=SubmissionNotFoundError)
        if resp.status_code != 200:
            raise _failed(SubmissionError, "submission_runs", resp)
        return resp.json()

    def submission_deploys(self, submission_id: int) -> dict:
        """Every deploy attempt of the submission, newest first.

        Mirrors `GET /api/submissions/submission/{sid}/deploys` (`read.py`).
        Returns `{"deploys": [...]}`, each row the `latest_deploy` shape of
        `submission()` (`id`, `created_at_ts`, `status`, `finished_at_ts`,
        `failure_message`, `failure_owner`, `initiated_by`). An attempt's runs are
        `submission_runs(submission_id, deploy_id=row["id"])`.
        """
        resp = self._request("GET",
            self._url(f"/submissions/submission/{submission_id}/deploys"),
            headers=self._headers(),
            timeout=30,
        )
        self._handle_response(resp, not_found=SubmissionNotFoundError)
        if resp.status_code != 200:
            raise _failed(SubmissionError, "submission_deploys", resp)
        return resp.json()

    def my_submissions(self) -> dict:
        """Every submission you have made, across every challenge.

        Mirrors `GET /api/submissions/mine` (`my_submissions.py`) — what the
        console's My Submissions page lists.

        Lists your own submissions and, on a challenge where you are in a
        team, your teammates' (a team is one participant). Returns
        `submissions`, `started_submissions_count` (how many of *your own*
        submissions are `active`; a teammate's is not counted) and
        `deployment_limits` (the same quota block `submission_deploy_status`
        serves). Each row carries `id`, `submission_name`, `challenge_id`,
        `challenge_name`, `rank`, `created_at_ts`, `challenge_start_date_ts`
        (the challenge's own `start_date_ts` column), `last_end_run_ts`,
        `owner` (`{user_id, username, avatar_key}`), `viewer_role` (`owner` |
        `teammate`: only the owner writes) and the status block. Deleted
        submissions are not listed: a row you deleted is gone from here, as
        from the console.
        `rank` is the place of the submission's row on the challenge
        leaderboard, one row per participant (the team's or your best active
        submission, as `leaderboard()` ranks it by default); None for a
        submission that has no row there.
        """
        resp = self._request("GET",
            self._url("/submissions/mine"),
            headers=self._headers(),
            timeout=30,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(SubmissionError, "my_submissions", resp)
        return resp.json()

    def set_submission_visibility(self, submission_id: int, is_public: bool) -> dict:
        """Publish or unpublish your submission: `update_submission(
        submission_id, is_public=is_public)`, which says what public opens."""
        return self.update_submission(submission_id, is_public=is_public)

    def replays(self, challenge_id: int, limit: int = 12,
                before_simulation_result_id: int | None = None,
                mine: bool = False, order: str = "recent") -> dict:
        """The challenge's runs that recorded a replay, newest first
        (`order="recent"`) or best first (`order="score"`).

        Mirrors `GET /api/challenges/{id}/replays` — the console's Overview
        "Replays" section. Returns `{"replays", "challenge_context"}`:
        `replays` are public `ReplayRun` rows, the public subset of the
        `RunResult` that `submission_runs()` serves,
        under the same keys: `simulation_result_id`, `created_at_ts`,
        `env_nb_steps`, `replay` = `{render_format, signed_url,
        render_size_bytes}` and `submission_results`, one per seat with
        `id`, `submission_id`, `submission_name`, `user_name`,
        `agent_attached_player_id`, `env_player_name`, `score`,
        `game_outcome` and `final_rank`. No metrics, info messages, error
        kinds, resource figures or Elo: the gallery is every viewer's, so a
        run's details stay with its owner (`submission_runs()`) and the
        challenge's staff (`creator_runs()`). `challenge_context`
        is the `RunsContext` (`metrics`, `number_of_agents`,
        `render_delay_second`, the default frame duration). Only completed,
        non-test runs of the last 364 days are listed (the replay bucket
        deletes objects after 365 days), and none a since-deleted submission
        played in. A row's participants (`submission_results`) come in seat
        order (`agent_attached_player_id`).

        `limit` is 1..50. Page with `before_simulation_result_id` = the last
        row's `simulation_result_id`. `order="score"` ranks the runs of a
        single-agent challenge by their score, in the direction that wins a
        run, unscored runs last (the console's "Best"); it is one page of
        `limit` rows, takes no `before_simulation_result_id`, and a
        multi-agent challenge answers 400. `mine=True` keeps the runs one of
        your (or your team's) submissions played in. The route is open to
        anyone who can see the challenge. Pass a row to
        `download_replay` to save its file.
        """
        params: dict = {"limit": limit}
        if before_simulation_result_id is not None:
            params["before_simulation_result_id"] = before_simulation_result_id
        if mine:
            params["mine"] = "true"
        if order != "recent":
            params["order"] = order
        resp = self._request("GET",
            self._url(f"/challenges/{challenge_id}/replays"),
            headers=self._headers(),
            params=params,
            timeout=30,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "replays", resp)
        return resp.json()

    def replay(self, challenge_id: int, simulation_result_id: int) -> dict:
        """One run of the challenge with its replay, freshly signed.

        Mirrors `GET /api/challenges/{id}/replays/{simulation_result_id}` —
        the console's `?replay=<id>` deep link. Returns `{"run",
        "challenge_context"}` (`run` a public `ReplayRun`, the row shape of
        `replays()`, for the staff too). A test run
        is served to the challenge's staff only (403 otherwise); a run of
        another challenge, or one with no replay, is a 404. Call it again
        for a new `signed_url` once one has expired (6 h).
        """
        resp = self._request("GET",
            self._url(f"/challenges/{challenge_id}/replays/{simulation_result_id}"),
            headers=self._headers(),
            timeout=30,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "replay", resp)
        return resp.json()

    def download_replay(self, run: dict, dest_dir: str = ".") -> Path:
        """Save a run's replay file; returns its path.

        `run` is any row carrying `simulation_result_id` and `replay`: a
        public `ReplayRun` (from `replays()`, `replay()["run"]`) or a
        `RunResult` (`submission_runs()["runs"]`, `creator_runs()` or
        `run_benchmark()`). The file is the stored object, streamed as is to
        `<dest_dir>/<simulation_result_id><suffix>`, the suffix from
        `replay.render_format`: `frames_v2` → `.frames.mp` (msgpack frames),
        `video_mp4` → `.mp4`, `image_png` → `.png`. The console's "Open file". A composition: no endpoint; the
        signed URL is a pre-authenticated R2 link, fetched without the token.

        Raises:
            ValueError: the run has no replay (`replay` is None).
            MLArenaError: the storage refused the URL (403 once it expired:
                re-read the row with `replay()`), with its status code.
        """
        replay = run["replay"]
        if replay is None:
            raise ValueError(
                f"run {run['simulation_result_id']} has no replay"
            )
        suffix = _REPLAY_SUFFIXES[replay["render_format"]]
        os.makedirs(dest_dir, exist_ok=True)
        out_path = Path(dest_dir) / f"{run['simulation_result_id']}{suffix}"
        # Presigned R2 URL: no auth header, follow redirects.
        resp = self._request("GET", replay["signed_url"], allow_redirects=True,
                             timeout=300, stream=True)
        if resp.status_code != 200:
            raise MLArenaError(
                f"download_replay failed: storage answered {resp.status_code}",
                status_code=resp.status_code,
            )
        tmp_path = out_path.with_name(out_path.name + ".partial")
        with open(tmp_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    fh.write(chunk)
        os.replace(tmp_path, out_path)
        return out_path

    def latest_replays(self, limit: int = 1, one_per_challenge: bool = False) -> dict:
        """The newest replays across the public challenges, newest first.

        Mirrors `GET /api/challenges/replays` — the landing page's "Latest
        replays". Returns `{"replays": [...]}`, each row `{"challenge_id",
        "challenge_name", "run", "challenge_context"}`: `run` is a public
        `ReplayRun` (the row shape of `replays()`, so `download_replay(row[
        "run"])` saves its file) and `challenge_context` its challenge's
        `RunsContext`. The runs are the ones each challenge's `replays()`
        lists, taken over the public challenges only, whoever calls: a
        private challenge you can see is not in it. `limit` is 1..12.
        `one_per_challenge=True` keeps each challenge's newest replay only,
        so `limit` rows are `limit` different challenges.
        """
        params: dict = {"limit": limit}
        if one_per_challenge:
            params["one_per_challenge"] = "true"
        resp = self._request("GET",
            self._url("/challenges/replays"),
            headers=self._headers(),
            params=params,
            timeout=30,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "latest_replays", resp)
        return resp.json()

    def recent_replays(self, challenge_id: int, limit: int = 12) -> dict:
        """Deprecated alias of :meth:`replays` (4.4.0).

        The `/recent-replays` route is gone: this calls `replays()` and
        returns its reply, `RunResult` rows with `replay`, not the old
        `simulation_id` / `participants` rows.
        """
        warnings.warn(
            "MLArenaClient.recent_replays() is deprecated, use .replays() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.replays(challenge_id, limit=limit)

    # ---- Status / logs ----

    def tail_logs(self, challenge_id: int, submission_id: int, *,
                  follow: bool = False, poll_sec: float = 5.0,
                  timeout_sec: float | None = None) -> Iterator[str]:
        """Yield human-readable status / log lines for a submission.

        Polls `submission()` and emits one line per status transition,
        including queue position for `deploy_queue` and, for each run of the
        latest deploy attempt (`submission_runs(submission_id,
        deploy_id=latest_deploy["id"])`, oldest first), the job's `job_status` with the steps / reward / outcome of
        **your own row** of that run (the `submission_results` entry whose
        `submission_id` is this submission's), then an
        `agent error[<agent_error_type>]: <agent_error_message>` line when
        your agent is why the run ended, an `env error[<env_error_type>]`
        line when the challenge's code is (its message is the staff's, so
        it is not printed), and a `job error[<job_error_type>]` line when
        the platform lost a `failed` run, followed by `: <job_error_message>`
        only for the challenge's staff (the server serves the message to
        them alone). `outcome=None` on a run line is a solo run, which has
        no winner or loser. A line is emitted only when what it says changed:
        a deploy that takes twenty polls prints each run once per state, not
        twenty times.

        With `follow=False` (default), returns as soon as the server reports
        the submission as settled (`is_settled`: nothing is running on it, so
        its status is the answer) — including for a submission that was never
        deployed, which simply yields its one current line and returns.
        With `follow=True`, keeps polling until the caller breaks out.

        `timeout_sec` caps the total wait and raises `SubmissionError` when it
        runs out, naming the status the submission was left in. It used to
        return silently, which was indistinguishable from "finished".
        A `submission_id` of another challenge than `challenge_id` raises
        `SubmissionError` on the first poll.

        Note: this does NOT stream pod stdout. Your agent's stdout of a run
        is `agent_stdout_logs` on your `submission_results` row (from
        `submission_runs()`). A run's `replay`
        holds its replay (frames, a video or a still), not logs: save it
        with `download_replay(run, dest_dir)`.

        Raises:
            SubmissionError: on `timeout_sec` elapsing, or on a submission
                of another challenge.
        """
        last_signature = None
        last_queue_line = None
        # Per run: the last line emitted for it. A run's line repeats across
        # polls until its steps / reward / outcome move.
        last_run_lines: dict[int, tuple] = {}
        deadline = (time.monotonic() + timeout_sec) if timeout_sec else None

        while True:
            status = self.submission(submission_id)
            if status["challenge_id"] != challenge_id:
                raise SubmissionError(
                    f"submission {submission_id} belongs to challenge "
                    f"{status['challenge_id']}, not {challenge_id}",
                    body=status,
                )
            sig = (status.get("status"), status.get("last_status_message"))
            if sig != last_signature:
                yield f"[{status.get('status')}] {status.get('last_status_message') or ''}".rstrip()
                last_signature = sig

            if status.get("status") == "deploy_queue":
                qi = status.get("queue_info") or {}
                if qi.get("queue_position") is not None:
                    queue_line = (f"  queued: position={qi.get('queue_position')}"
                                  f"/{qi.get('queue_total')} "
                                  f"waiting={qi.get('in_queue_for_second')}s")
                    if queue_line != last_queue_line:
                        yield queue_line
                        last_queue_line = queue_line

            # The latest attempt's runs, in every post-deploy state: the runs
            # of a failed attempt are still there to print. The route pages
            # newest first; they are printed oldest first.
            latest = status["latest_deploy"]
            attempt_runs = [] if latest is None else self.submission_runs(
                submission_id, deploy_id=latest["id"], limit=100)["runs"]
            for run in reversed(attempt_runs):
                # A run lists every agent's row under `submission_results`;
                # ours is the row whose `submission_id` is this submission's.
                # Direct key access on the shape: a backend serving another
                # run model fails here instead of printing `None` forever.
                mine = next((row for row in run["submission_results"]
                             if row["submission_id"] == submission_id), {})
                run_line = (f"  run: job_status={run.get('job_status')} "
                            f"steps={mine.get('agent_nb_steps')} "
                            f"score={mine.get('score')} "
                            f"outcome={mine.get('game_outcome')}")
                error_lines = tuple(line for line in (
                    _error_line("agent", mine.get("agent_error_type"),
                                mine.get("agent_error_message")),
                    _error_line("env", run.get("env_error_type"),
                                run.get("env_error_message")),
                    _error_line("job", run["job_error_type"],
                                run.get("job_error_message"))
                    if run.get("job_status") == "failed" else None,
                ) if line is not None)
                run_id = run["simulation_result_id"]
                if last_run_lines.get(run_id) == (run_line, error_lines):
                    continue
                last_run_lines[run_id] = (run_line, error_lines)
                yield run_line
                yield from error_lines

            # The server decides what "done" means; direct key access so a
            # backend that does not send the block fails loudly here.
            if status["is_settled"] and not follow:
                return

            if deadline and time.monotonic() > deadline:
                raise WaitTimeoutError(
                    f"tail_logs timed out after {timeout_sec}s: submission "
                    f"{submission_id} is still {status.get('status')!r}. Poll "
                    f"again with submission({submission_id}) or raise "
                    f"timeout_sec.",
                    body=status,
                )

            time.sleep(poll_sec)

    # ---- One-shot submit helper (still used by the notebook flow) ----

    def submit(self, challenge_id: int, agent=None, files=None,
               submission_name: str | None = None,
               runtime_id: int | None = None,
               runtime: str | dict | None = None,
               wait: bool = False,
               timeout_sec: float | None = None,
               poll_sec: float = 5.0, *,
               deploy: bool = True,
               allow_same_commit: bool = False) -> dict:
        """Convenience: check → create submission → (pick runner) → upload
        files → deploy.

        Either pass `agent=<class>` or `files=[...]` (a list of paths, each
        uploaded under its base name). `agent=` takes a class (or a module),
        never a path: the class's source, `inspect.getsource(agent)`, is
        uploaded as `agent.py`, and that is the class body only — imports,
        constants and helpers outside the class are not in it. An agent that
        needs them is a file: pass `files=["agent.py", ...]`.
        `submission_name` defaults to the class name, or to the first file's
        base name without extension.

        Optional runner selection:
            runtime:    the runtime's name, one the challenge offers
                        (`runtime_options(challenge_id)["name"]`, listed on the
                        challenge page): "python", "sklearn", "torch", ...
                        Resolved like `resolve_runtime()`. A dict spec
                        (`{"framework": "torch"}`) is deprecated (4.8.0).
            runtime_id: the runtime's numeric id, one of the challenge's
                        `runtime_options`.
        Pass at most one of `runtime_id` / `runtime`. With neither, the
        submission runs on the challenge's default runtime
        (`challenge(challenge_id)["default_agent_runtime_id"]`).

        Before anything is created, one `challenge()` read checks what needs
        no submission, and raises `SubmissionError` (`MLArenaError` for an
        unknown runtime name) with no submission left behind: the challenge's
        `status` (with `deploy=True`, a challenge that is not `started` takes
        no deploy: `draft`, `stopped` or `ended`), the runtime
        (a name or an id the challenge does not offer), the `agent=`
        argument (a path, an instance, a class without readable source, a
        challenge whose entry file is not `agent.py`), the files (a missing
        path, two files with one base name, more files than
        `max_upload_files` or more bytes than `max_upload_size_bytes`).
        The server still validates every upload.

        After the uploads, the submission's `is_deployable` is checked (one
        `submission()` call) and a `SubmissionError` carrying the
        server's `last_status_message` (and, on `.body`, that status payload)
        is raised rather than deploying files the backend already rejected.

        With `wait=True`, the call returns only once the deploy has settled:
        it drains `tail_logs()` (a client-side composition of the same public
        routes — there is no wait endpoint) and adds the final status block
        under `"status"`. An attempt that ends `deploy_failed` raises
        `SubmissionError` whose message is `latest_deploy.failure_message`
        (or `last_status_message`), followed by `(failure_owner: <owner>)`
        when the attempt's verdict names who it blames (`participant` |
        `challenge` | `platform`), and whose `.body` is that final status
        payload — the same contract as a rejected upload, so a `wait=True`
        call that returns is a submission that is `active`. `timeout_sec`
        caps the wait and raises `SubmissionError` when it runs out;
        `poll_sec` is the polling interval. The default `wait=False` returns
        as soon as the deploy is accepted (202).

        Any error raised once the submission exists carries its id as
        `err.submission_id`; nothing is deleted for you. An error raised
        before the deploy (a refused upload or runtime pin, files the server
        rejected, a refused deploy: 409 — quota, active-submission limit,
        state, with the server's limits on `.body`) also names it in the
        message, with the `deploy_submission(...)` / `delete_submission(...)`
        calls that deploy it later or delete it. The submission is also
        remembered for `status()` as soon as it is created.

        Returns `{"submission_id": <id>, "deploy": <deploy response>}`, plus
        `"status"` when `wait=True` (a wait that runs out of `timeout_sec`
        raises `WaitTimeoutError`, a `SubmissionError`). With `deploy=False` the call stops after
        the upload check (`"deploy"` is None): deploy it later with
        `deploy_submission()` or from the console.

        From a GitHub Actions run, a commit that already made a submission on
        this challenge is not submitted twice (a re-run, a second tag on the
        same commit): nothing is created, uploaded or deployed, and the call
        returns `{"submission_id": <the existing id>, "deploy": None,
        "already_submitted": True, "status": <submission()>}`. Pass
        `allow_same_commit=True` to submit it again.
        """
        if (agent is None) == (files is None):
            raise SubmissionError("Provide exactly one of agent= or files=")
        if files is not None and isinstance(files, (str, bytes, os.PathLike)):
            raise SubmissionError(
                f"files= takes a list of paths: files=[{files!r}]")
        if runtime_id is not None and runtime is not None:
            raise SubmissionError("Provide at most one of runtime_id= or runtime=")
        if wait and not deploy:
            raise SubmissionError("wait=True needs deploy=True")
        if agent is not None:
            _check_agent_argument(agent)

        # ---- Everything that needs no submission, before the create -------
        challenge = self.challenge(challenge_id)
        if deploy and challenge["status"] != "started":
            # The deploy route's own 409, said before a submission exists.
            raise SubmissionError(
                f"Challenge {challenge_id} is {challenge['status']}: deploys are closed. "
                "Nothing was created; submit(..., deploy=False) creates the "
                "submission and uploads its files without deploying."
            )
        offered = challenge["runtime_options"]
        if isinstance(runtime, dict):
            warnings.warn(
                "submit(runtime={...}) is deprecated, pass the runtime's name: "
                "runtime=\"sklearn\" (runtime_options(challenge_id) lists them)",
                DeprecationWarning,
                stacklevel=2,
            )
            runtime_id = _match_runtime(challenge_id, offered, None, **runtime)["id"]
        elif runtime is not None:
            runtime_id = _match_runtime(challenge_id, offered, runtime)["id"]
        elif runtime_id is not None and runtime_id not in {r["id"] for r in offered}:
            listed = ", ".join(f"{r['id']} ({r['name']})" for r in offered)
            raise SubmissionError(
                f"Challenge {challenge_id} offers no runtime with id {runtime_id}; "
                f"its runtimes: {listed or 'none'}"
            )

        if agent is not None:
            agent_source = _agent_source(agent, challenge["entry_filename"], challenge_id)
            uploads = [("agent.py", len(agent_source.encode("utf-8")))]
        else:
            agent_source = None
            for path in files:
                if not os.path.isfile(path):
                    raise SubmissionError(f"File not found: {path}")
            uploads = [(os.path.basename(path), os.path.getsize(path)) for path in files]
        _check_upload_limits(challenge_id, challenge, uploads)

        # ---- Create -------------------------------------------------------
        try:
            created = self.create_submission(
                challenge_id,
                submission_name or _default_submission_name(agent, files),
                allow_same_commit=allow_same_commit,
            )
        except SubmissionError as err:
            existing_id = (err.body or {}).get("submission_id")
            if err.status_code != 409 or existing_id is None:
                raise
            # This commit already made a submission (create_submission).
            self._last_submission_id = existing_id
            self._last_challenge = challenge_id
            return {
                "submission_id": existing_id,
                "deploy": None,
                "already_submitted": True,
                "status": self.submission(existing_id),
            }
        submission_id = created["submission_id"]
        # Remembered now, not after the deploy: `status()` must find the
        # submission that a rejected upload or a refused deploy leaves behind.
        self._last_submission_id = submission_id
        self._last_challenge = challenge_id

        # ---- From here the submission exists: every error names it --------
        tmp_dir = None
        try:
            # Pin runner before deploy so the JobPod uses the right image.
            if runtime_id is not None:
                self.set_agent_runtime(submission_id, runtime_id)
            if agent_source is not None:
                tmp_dir = tempfile.mkdtemp()
                agent_py = os.path.join(tmp_dir, "agent.py")
                with open(agent_py, "w", encoding="utf-8") as f:
                    f.write(agent_source)
                self.upload_submission_file(
                    challenge_id, submission_id, agent_py)
            else:
                for path in files:
                    self.upload_submission_file(
                        challenge_id, submission_id, path)

            # Fail fast on a rejected upload. The file route returns HTTP 200 even
            # when validation rejects the *submission* (an actionable message, e.g.
            # "Rename your file to 'agent.py'"), so upload_submission_file() cannot
            # see it as an error. Ask the status route whether the submission may
            # deploy — that is the same fact the Deploy button reads, and it is
            # asked after every file is on disk, so a legitimately transient
            # mid-sequence failure (multi-file upload) isn't misreported.
            status = self.submission(submission_id)
            if not status["is_deployable"]:
                raise SubmissionError(
                    f"Submission {submission_id} did not pass upload validation: "
                    f"{status['last_status_message'] or 'files rejected'} "
                    f"(fix the files and re-upload, or delete the submission with "
                    f"delete_submission({challenge_id}, {submission_id})).",
                    body=status,
                    submission_id=submission_id,
                )

            if not deploy:
                return {"submission_id": submission_id, "deploy": None}
            deployed = self.deploy_submission(challenge_id, submission_id)
        except MLArenaError as err:
            if err.submission_id is not None:  # already names it
                raise
            raise _left_behind(err, challenge_id, submission_id) from err
        finally:
            if tmp_dir:
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)

        result = {"submission_id": submission_id, "deploy": deployed}
        if not wait:
            return result
        try:
            # Drain the generator: tail_logs() stops on `is_settled` and
            # raises on timeout, so this is the whole wait.
            for _ in self.tail_logs(challenge_id, submission_id,
                                    poll_sec=poll_sec,
                                    timeout_sec=timeout_sec):
                pass
            final = self.submission(submission_id)
        except MLArenaError as err:
            # Deployed: the message stays the wait's own, the id rides along.
            err.submission_id = submission_id
            raise
        if final["status"] == "deploy_failed":
            # Same contract as the rejected upload above: the server's
            # reason is the message, the payload rides on `.body`.
            # A deploy_failed submission has deployed: the attempt is
            # there.
            latest = final["latest_deploy"]
            reason = (latest.get("failure_message")
                      or final.get("last_status_message")
                      or f"Submission {submission_id} ended in "
                         f"deploy_failed without a message")
            # Who the verdict blames (participant | challenge |
            # platform), when the server's verdict said.
            owner = latest.get("failure_owner")
            if owner is not None:
                reason = f"{reason} (failure_owner: {owner})"
            raise SubmissionError(reason, body=final, submission_id=submission_id)
        result["status"] = final
        return result

    def status(self, submission_id: int | None = None) -> dict:
        """The submission (`submission()`), by default the last one made
        through this client (`submit()`, which remembers it as soon as it is
        created)."""
        submission_id = submission_id or self._last_submission_id
        if submission_id is None:
            raise SubmissionError(
                "submission_id is required (no previous submission found)"
            )
        return self.submission(submission_id)

    # ---- Leaderboard (public read) ----

    def leaderboard(self, challenge_id: int | None = None, top: int | None = None, *,
                    aggregate: str | None = None,
                    me: bool = False, q: str | None = None,
                    window: int | None = None, offset: int | None = None,
                    around: int | None = None):
        """Get the leaderboard for a challenge.

        Mirrors `GET /api/leaderboard/challenge/{id}` (`leaderboard.py`) with
        the query keys the console sends — `limit` (here `top`), `offset`,
        `aggregate`, `me`, `around`, `q`, `window` — each sent only when
        passed.

        - ``aggregate="user"`` (the default, as on the console): one row per
          participant — their best active submission, a team's best for a
          team, ranked among participants. ``"submission"``: every active
          submission is a row.
        - One board per challenge. A course challenge's board is its class:
          its participants, with no filter (4.23.0 removed `course_id`).
          Its rows carry ``passed`` (tri-state: None while the row has no
          ranked value, or when the course sets no bar) and the envelope a
          ``course_context`` block (`course_id`, `pass_threshold`,
          `names_withheld`); a board with no course has neither. Names go to
          the course's members only: unless you are enrolled in the course,
          own it, assist it or are an admin,
          ``course_context["names_withheld"]`` is True and every row comes
          with its rank, score and metrics but with `username`,
          `avatar_key`, `submission_id`, `submission_name`, `team_id`,
          `team_name`, `team_members`, `datagouv_reuse`, `created_at_ts`,
          `last_end_run_ts`, `number_of_runs`, `action_time_max_sec`,
          `agent_metric_total_ram_max_bytes` and
          `agent_metric_vram_max_bytes` set to None and `can_open` False
          (the keys stay). On such a read ``q`` and ``around`` are refused
          (`MLArenaError`, 400): each would say whether a given person is
          in the course. That is a public course: a private course's
          challenge, and its board, are its members' only
          (`NotFoundError`). An exercise set's board is served to its
          course's staff only (`NotFoundError` otherwise), and so is the
          board of a draft challenge the caller does not edit.
        - ``top=N``: the first N rows (every ranked row without it);
          ``offset=K`` starts the page at the K-th row (0-based), so
          ``top=50, offset=100`` is rows 101-150 — the console's page 3.
          ``me=True`` adds your own row with `rank`, `position` (its
          1-based place in board order; ties share a rank, not a position),
          `percentile` and its
          ``window`` neighbours (default 3, at most 25) under `me`;
          ``around=<submission_id>`` adds that submission's row, in the same
          shape, under `around` (None when the submission has no row at this
          ``aggregate``: not active, or shadowed by its owner's better
          submission on the per-participant board); ``q``
          adds the rows whose username contains it under `matches` (at most
          50).

        **Shape.** The backend serves one envelope: `challenge`, `total`,
        `leaders`, `me`, `around`, `matches` and, on a course challenge,
        `course_context`. With pandas, the call returns `leaders` as a
        DataFrame and every other envelope key on ``df.attrs``
        (``df.attrs["challenge"]["window_days"]``, ``df.attrs["me"]``, …),
        plus ``df.attrs["metrics"]``, the declaration; without pandas it
        returns the envelope dict as served.

        **The `challenge` block** — what every row shares, served once:
        `challenge_id`, `metrics` (the MetricSpec declaration, see the README
        "Metrics" section), `window_days` (None unless the board also folds
        a rolling window), `has_gpu` and `kernel_version`. The spec with `is_ranking` is what
        the board ranks on; its ``order`` says which way (``"desc"``: higher
        is better, ``"asc"``: lower is better).

        **Columns** — the backend's own names, one set with the console:
        `rank`, `username`, `avatar_key`, `submission_id`,
        `submission_name`, `score` (the ranking spec's value; None =
        unranked), `score_ci95`, `n_episodes_total`, `elo_variance`,
        `number_of_runs`, `created_at_ts` and `last_end_run_ts` (ISO-8601
        UTC with a `Z`), `is_my_submission` (your row: your submission or
        your team's), `can_open` (whether you may read the row's submission
        with `submission()`), `team_id`, `team_name`,
        `team_members`, `datagouv_reuse` (the team's linked data.gouv.fr
        reuse, see `link_datagouv_reuse`, or None), `action_time_max_sec`,
        `agent_metric_total_ram_max_bytes`, `agent_metric_vram_max_bytes`,
        `is_public` (None when the row is not yours to know) and, through a
        course with a bar, `passed`. `username`, `submission_id`,
        `submission_name` and `team_members` are None only under
        ``course_context["names_withheld"]`` (see above). With pandas, each row's `metrics`
        (`{key: value}`) becomes one column per declared key, named by the
        key, and — when `window_days` is set — its `metrics_window` becomes
        one `metrics_window_<key>` column per key. Rows come in server rank
        order.
        """
        challenge_id = challenge_id or self._last_challenge
        if challenge_id is None:
            raise SubmissionError("No challenge specified and no previous submission found")
        params: dict = {}
        if top is not None:
            params["limit"] = top
        if offset is not None:
            params["offset"] = offset
        if aggregate is not None:
            params["aggregate"] = aggregate
        if window is not None:
            params["window"] = window
        if me:
            params["me"] = "true"
        if around is not None:
            params["around"] = around
        if q:
            params["q"] = q
        params = params or None
        # The route is public, but it answers differently to the caller it
        # can identify: `is_my_submission`, the names of a course's board to
        # its members, and the challenges of a private course. Without the bearer token every row came back
        # as somebody else's.
        resp = self._request("GET",
            self._url(f"/leaderboard/challenge/{challenge_id}"),
            params=params,
            headers=self._headers(),
            timeout=30,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            # `raise_for_status()` used to end this call, which threw away the
            # server's reason with the body.
            raise _failed(MLArenaError, "leaderboard", resp)
        envelope = resp.json()
        try:
            import pandas as pd
        except ImportError:
            return envelope
        rows = pd.DataFrame(_leaderboard_records(envelope))
        rows.attrs.update({k: v for k, v in envelope.items() if k != "leaders"})
        rows.attrs["metrics"] = envelope["challenge"]["metrics"]
        return rows

    def global_ranking(self, search: str | None = None, page: int = 1, per_page: int = 100):
        """Global user ranking across all challenges (points + medals).

        How medals and points are awarded (the /ranking page's text, word for
        word):

        Every hour, each challenge that awards points ranks its
        participants (a team, or a user on their own) by their best
        submission on its leaderboard. Ranks 1, 2 and 3 earn a gold,
        silver and bronze medal. Every rank earns 100,000 × rank^-0.75 ×
        log10(1 + log10(N)) ÷ √(team size) points, where N is the number
        of ranked participants; each team member gets the team's medal
        and points. When a challenge stops or ends, its last standings
        stay. A course's challenges award no points. Your total is the
        sum of your points over all challenges.

        Mirrors `GET /api/ranking/` (`ranking.py`). Returns a DataFrame of the
        requested page (default: top 100) with the server's own column names:
        `user_id`, `username`, `avatar_key`, `rank`, `current_points`,
        `medals_gold`, `medals_silver`, `medals_bronze`. Pass ``search`` to
        filter by username; matched rows carry their true global rank. Ties
        share a rank (competition ranking: 1, 1, 3); a user with no points
        holds no rank — `rank` is None on their row, which sits at the tail
        in alphabetical order.

        The envelope's pagination block rides along as
        ``df.attrs["metadata"]`` — `total_pages`, `current_page`,
        `total_users`, `has_next`, `has_prev` — the same way `leaderboard()`
        carries its envelope blocks (a plain list, without pandas, cannot).
        It used to be dropped, so a caller could not tell a full page from
        the last one.
        """
        params: dict = {"page": page, "per_page": per_page}
        if search:
            params["search"] = search
        resp = self._request("GET", self._url("/ranking/"), params=params,
                             headers=self._headers(), timeout=30)
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "global_ranking", resp)
        data = resp.json()
        rows = _to_dataframe(data["rankings"])
        if hasattr(rows, "attrs"):
            rows.attrs["metadata"] = data["metadata"]
        return rows

    def user_global_rank(self, user_id: int) -> dict:
        """A user's global rank, percentile and medal counts.

        Mirrors `GET /api/ranking/user/{id}` (`ranking.py`, login required:
        the bearer token is what identifies you).

        Flat, under the same names a `global_ranking()` row uses: `user_id`,
        `username`, `rank`, `current_points`, `percentile`, `medals_gold`,
        `medals_silver`, `medals_bronze`, plus `position`: the user's 1-based
        place in `global_ranking()`'s order (``page = (position - 1) //
        per_page + 1``), None when `rank` is. The route used to nest all but the
        first two under a `stats` key that this method silently unwrapped;
        both the wrapper and the unwrap are gone.

        A user who is not ranked (no points yet) is a normal answer: `rank`
        and `percentile` are None (until 4.1.0 the route answered 404). A
        404 — raised as `ChallengeNotFoundError`, a `NotFoundError` — now
        means the user does not exist. Ties share a rank, not a position.
        """
        resp = self._request("GET", self._url(f"/ranking/user/{user_id}"),
                             headers=self._headers(), timeout=30)
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "user_global_rank", resp)
        return resp.json()

    def user_profile(self, username: str) -> dict:
        """A user's public profile, as anyone may read it.

        Mirrors `GET /api/users/{username}` (`users/profile.py`, public: works
        with or without a key; the lists are what the caller may see). The
        name is matched whatever its case. Returns `user_id`, `username`,
        `avatar_key`, `bio`, `personal_website`, `created_at_ts`, `ranking`
        (`user_global_rank()`'s dict), `organized_challenges` (the started
        or ended competitions they own, as `challenges()` rows: running
        first, then the most recently ended) and `taught_courses` (the courses
        they own that the course catalog lists, as `course_catalog()` cards).
        Never the email, the student identity or the linked GitHub account.

        A 404 — a name no active account holds — raises `NotFoundError`.
        """
        resp = self._request("GET", self._url(f"/users/{quote(username, safe='')}"),
                             headers=self._headers(), timeout=30)
        self._handle_response(resp, not_found=NotFoundError)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "user_profile", resp)
        return resp.json()

    # ---- Chat challenges (`chat_v1`): one shared JSON-call helper -----------

    def _chat_call(self, method: str, path: str, *,
                   json_body: dict | None = None,
                   params: dict | None = None,
                   ok: tuple[int, ...] = (200,),
                   error_label: str,
                   not_found: type = ChallengeNotFoundError,
                   timeout: int = 30) -> requests.Response:
        """Single chokepoint for the `/api/chat/*` routes (`_json_call`'s
        sibling). `not_found` names the class a 404 raises:
        `ChallengeNotFoundError` on a challenge-scoped route,
        `ChatSessionNotFoundError` on a session-scoped one. Any other
        non-`ok` status — the 409s of the chat rules (challenge not started, agent offline, closed or busy session), a 400 — raises `MLArenaError`
        with the server's reason. Returns the response: `export_chat_session`
        reads `text`, everything else `.json()`."""
        resp = self._request(
            method, self._url(path),
            headers=self._headers(json_body=json_body is not None),
            json=json_body, params=params, timeout=timeout,
        )
        self._handle_response(resp, not_found=not_found)
        if resp.status_code not in ok:
            raise _failed(MLArenaError, error_label, resp)
        return resp

    # ---- Chat challenges: participant routes (user scope) --------------------

    def chat_challenge(self, challenge_id: int) -> dict:
        """The participant view of a chat challenge — `ChatChallengeView`.

        Mirrors `GET /api/chat/challenge/{cid}` (`backend/app/views/chat/`). Returns `challenge_name`, `challenge_slug`, `status`
        (`draft` | `started` | `stopped` | `ended`),
        `manifest` (what the agent registered: `bot_name`, `tagline`,
        `welcome_message`, `charter_md`, `rules_public_md`, the public
        `scoring_rules`, its `tools`; None until the ChatPod is up),
        `env_status` (the error text behind an `"error"` is the staff's:
        `chat_admin()`), `agent_online`, the limits
        (`max_turns_per_session`, `max_sessions_per_participant`; a None limit
        is unlimited), `participant` (your group on this challenge:
        `submission_id`, `team_name`, `members`, `total_amount_eur` — the
        group's cumulative euros —, `scoreboard` (one
        `{"rule_key", "count", "total_amount_eur"}` per rule you have fired),
        `session_count` — the non-voided sessions, the leaderboard's
        `number_of_runs` —, `rank` — the group's place on `leaderboard()`,
        whose `score` is the same euro total under the fixed `loot` spec,
        `format: "currency"`, `unit: "€"`), `sessions` (yours and your teammates',
        newest first, without those closed before their first message;
        `session_count` still counts them) and `open_session_id` (your open
        session, or None).

        Raises `ChallengeNotFoundError` when the challenge is not a chat
        challenge or is hidden from you.
        """
        return self._chat_call(
            "GET", f"/chat/challenge/{challenge_id}",
            error_label="chat_challenge",
        ).json()

    def wait_for_chat_agent(self, challenge_id: int, *, timeout: float = 180,
                            poll_interval: float = 3) -> dict:
        """Wait until the challenge's agent is online. Returns the
        `ChatChallengeView` that shows it (`agent_online: true`).

        A chat challenge nobody used for 30 minutes has no agent running: it
        sleeps, and reading the challenge (`chat_challenge()`, which this
        polls) wakes it, usually within a minute. Until then
        `open_chat_session` answers 409. A client-side composition of
        `chat_challenge` — no endpoint of its own; `chat().say()` calls it
        before it opens a session.

        Raises `MLArenaError` at once when waiting cannot help (the challenge
        is not `started`, or its environment failed to load), and after
        `timeout` seconds without the agent coming online.
        """
        deadline = time.monotonic() + timeout
        while True:
            view = self.chat_challenge(challenge_id)
            if view["status"] != "started":
                raise MLArenaError(
                    f"wait_for_chat_agent: challenge {challenge_id} is "
                    f"{view['status']!r}, not started — its agent does not run"
                )
            if view["env_status"] == "error":
                raise MLArenaError(
                    f"wait_for_chat_agent: the environment of challenge "
                    f"{challenge_id} failed to load; the challenge team has to fix it"
                )
            if view["agent_online"]:
                return view
            if time.monotonic() > deadline:
                raise MLArenaError(
                    f"wait_for_chat_agent timed out after {timeout}s: the agent "
                    f"of challenge {challenge_id} is still not online "
                    f"(env_status {view['env_status']!r}). Try again or raise timeout."
                )
            time.sleep(poll_interval)

    def open_chat_session(self, challenge_id: int) -> dict:
        """Open a conversation with the agent. Returns the new `ChatSessionView`.

        Mirrors `POST /api/chat/challenge/{cid}/sessions` with
        `{"charter_accepted": true}`: calling this **is** accepting the
        challenge's charter (`chat_challenge(cid)["manifest"]["charter_md"]`)
        — the server keeps no session without it. Your group's submission is
        created on the first session (born active: nothing to upload or
        deploy) and your previous open session is closed — one open session
        per user per challenge.

        The reply is the same shape `chat_session()` returns: `session`,
        `messages` (empty), `tool_calls`, `scoring_events`, `turn` (None),
        `participant`, `can_send`, `can_send_reason`.

        Raises `ChallengeNotFoundError` on a non-chat or hidden challenge, and
        `MLArenaError` with the server's reason when it refuses (409: the
        challenge is not started, the agent is offline or asleep — it sleeps
        after 30 idle minutes, see `wait_for_chat_agent` —,
        `max_sessions_per_participant` is reached, the agent is still
        answering a message of your open session — wait for it first —, or
        the challenge's owner has no compute credits left: that 409 carries
        `.body["reason"] == "out_of_credits"` beside `error`, and no
        conversation is opened). `EnrollmentRequiredError` when you are not
        enrolled in the challenge's course.
        """
        return self._chat_call(
            "POST", f"/chat/challenge/{challenge_id}/sessions",
            json_body={"charter_accepted": True}, ok=(200, 201),
            error_label="open_chat_session",
        ).json()

    def chat_session(self, session_id: int) -> dict:
        """One conversation with its evidence — `ChatSessionView`.

        Mirrors `GET /api/chat/sessions/{sid}`. Returns `session`
        (`ChatSessionSummary`: `status` `open` | `closed` | `voided`, `title`,
        `total_amount_eur`, counts, timestamps), `messages` (ordered),
        `tool_calls` (what the agent called, with arguments and results),
        `scoring_events` (each breach: `rule_key`, `label`, `amount_eur`; the
        `evidence` blob behind it is in the export), `turn` (the in-flight or
        latest turn: `status` `pending` | `running` | `completed` | `failed`,
        `progress` while running, `queue_position` while pending — how many
        conversations the agent answers before this one, 0 = next, None in
        any other status; when failed, `error_type` — the cause
        (`llm_unavailable` | `llm_timeout` | `context_full` — the conversation
        outgrew the model's context window and was closed: open a new one —
        | `turn_timeout` | `env_error` | `agent_offline` | `platform_error`;
        always set on a failed turn, None otherwise) — and `error_message`,
        the raw text behind it, served to the challenge's owner,
        assistants and admins only (None for everyone else); None before the
        first message), `participant` (your group's `total_amount_eur` and
        `scoreboard`, the same keys `chat_challenge()` serves) and `can_send`
        with `can_send_reason` (`ChatCannotSendReason`: `session_voided` |
        `session_closed` | `turn_in_flight` | `challenge_not_started` |
        `challenge_stopped` | `challenge_ended` | `enrollment_required` —
        the session's owner is no longer enrolled in the challenge's course
        — | `out_of_credits` — the challenge's owner has no compute credits
        left: nothing more is sent until they have some — | `agent_offline`
        | `turn_limit_reached`, None exactly when `can_send` is true). Readable
        by the session's user, their teammates, the challenge's owner,
        assistants and admins.

        Cheap enough to poll — `send_chat_message(wait=True)` does.

        Raises `ChatSessionNotFoundError` when the session does not exist or
        is not yours to read.
        """
        return self._chat_call(
            "GET", f"/chat/sessions/{session_id}",
            error_label="chat_session", not_found=ChatSessionNotFoundError,
        ).json()

    def send_chat_message(self, session_id: int, content: str, *,
                          wait: bool = True, timeout: float = 180,
                          poll_interval: float = 0.7) -> dict:
        """Send a message and, by default, wait for the agent's answer.

        Mirrors `POST /api/chat/sessions/{sid}/messages` `{"content": ...}`
       , which answers 202 `{"turn", "message"}` as soon as the
        turn is queued for the agent — `turn` is the `ChatTurnView` the session
        poll will serve, with its `queue_position` (0 = the agent takes it
        next). `content` is 1..4000 characters.

        With `wait=True` (default) the call then polls `chat_session()` every
        `poll_interval` seconds — a client-side composition of the two public
        routes, the console's own cadence; there is no wait endpoint — until
        **that** turn's `status` is `completed` or `failed`, and returns the
        final `ChatSessionView`: the assistant's reply is the message whose
        `id` is `turn["assistant_message_id"]`, and the events this turn
        earned are the `scoring_events` with `turn_id == turn["id"]`. A
        `failed` turn raises `MLArenaError` naming the turn's `error_type`,
        with its `error_message` appended when the server serves it (to the
        challenge's staff only); the session stays open, so you may send again. Not
        settled after `timeout` seconds raises `MLArenaError` too, naming the
        state the turn was left in. The `timeout` budget restarts each time
        the turn moves up the queue (`queue_position` decreases): a busy
        challenge serves the conversations in order, and waiting your place
        is not the agent being stuck.

        With `wait=False` the 202 body is returned as-is.

        Raises `ChatSessionNotFoundError` when the session is not yours, and
        `MLArenaError` with the server's reason when it refuses (409: the
        challenge is not started, a turn is already in flight on this
        session, the session is not `open`, `max_turns_per_session` is
        reached, or the challenge's owner has no compute credits left: that
        409 carries `.body["reason"] == "out_of_credits"` beside `error`,
        the `can_send_reason` of the session). `EnrollmentRequiredError`
        when you are no longer enrolled in the challenge's course.
        """
        accepted = self._chat_call(
            "POST", f"/chat/sessions/{session_id}/messages",
            json_body={"content": content}, ok=(200, 201, 202),
            error_label="send_chat_message", not_found=ChatSessionNotFoundError,
            timeout=60,
        ).json()
        if not wait:
            return accepted
        return self._wait_for_chat_turn(
            session_id, accepted["turn"]["id"],
            timeout=timeout, poll_interval=poll_interval,
        )

    def _wait_for_chat_turn(self, session_id: int, turn_id: int, *,
                            timeout: float, poll_interval: float) -> dict:
        """Poll `chat_session` until turn `turn_id` settles (see
        `send_chat_message`). A view whose `turn` is another turn — the
        previous one, still the latest for a moment — is not that turn
        settling, so it is skipped rather than misread. The deadline restarts
        whenever the turn's `queue_position` goes down."""
        deadline = time.monotonic() + timeout
        last_position = None
        while True:
            view = self.chat_session(session_id)
            turn = view["turn"]
            if turn is not None and turn["id"] == turn_id and turn["status"] == "pending":
                position = turn["queue_position"]
                if last_position is not None and position < last_position:
                    deadline = time.monotonic() + timeout
                last_position = position
            if (turn is not None and turn["id"] == turn_id
                    and turn["status"] in ("completed", "failed")):
                if turn["status"] == "failed":
                    # `error_type` is what a participant is told; the raw
                    # `error_message` is served to the challenge's staff only
                    # (None for everyone else), so it is appended when present.
                    reason = f"chat turn {turn_id} failed ({turn['error_type']})"
                    if turn["error_message"] is not None:
                        reason += f": {turn['error_message']}"
                    raise MLArenaError(reason)
                return view
            if time.monotonic() > deadline:
                left_in = (turn["status"] if turn is not None
                           and turn["id"] == turn_id else "not yet visible")
                raise MLArenaError(
                    f"send_chat_message timed out after {timeout}s: turn "
                    f"{turn_id} of session {session_id} is still {left_in!r}. "
                    f"Poll chat_session({session_id}) or raise timeout."
                )
            time.sleep(poll_interval)

    def close_chat_session(self, session_id: int) -> dict:
        """End a conversation: `open` → `closed`. Idempotent.

        Mirrors `POST /api/chat/sessions/{sid}/close`. What the
        session earned stays banked; open a new one with
        `open_chat_session()`.

        Raises `ChatSessionNotFoundError` when the session is not yours, and
        `MLArenaError` (409) while the agent is still answering a message of
        this session.
        """
        return self._chat_call(
            "POST", f"/chat/sessions/{session_id}/close",
            error_label="close_chat_session", not_found=ChatSessionNotFoundError,
        ).json()

    def export_chat_session(self, session_id: int,
                            format: str = "json") -> dict | str:
        """The evidence file of one session — « la pièce à conviction ».

        Mirrors `GET /api/chat/sessions/{sid}/export?format=json|md`:
        transcript, tool log, scoring events, totals and session
        metadata — including what the session view leaves out, because the
        export is the record: each event's `evidence` blob and
        `created_at_ts`, each tool call's `created_at_ts`, each turn's
        `claimed_at_ts` / `completed_at_ts` / `tool_round_count`, and, on a
        voided session, `voided_by_username` / `voided_at_ts` next to the
        `void_reason`. `format="json"` (default) returns the parsed document;
        `format="md"` returns the Markdown text. Same readers as
        `chat_session()`.

        Raises `MLArenaError` on any other `format`, and
        `ChatSessionNotFoundError` when the session is not yours.
        """
        if format not in ("json", "md"):
            raise MLArenaError(
                f"export_chat_session: format must be 'json' or 'md', got {format!r}"
            )
        resp = self._chat_call(
            "GET", f"/chat/sessions/{session_id}/export",
            params={"format": format}, error_label="export_chat_session",
            not_found=ChatSessionNotFoundError, timeout=60,
        )
        return resp.json() if format == "json" else resp.text

    def chat(self, challenge_id: int, *, timeout: float = 180,
             poll_interval: float = 0.7, echo: bool = True) -> ChatConversation:
        """A conversation object for a chat challenge (user scope).

        A client-side composition of `open_chat_session`, `send_chat_message`
        and `chat_session` — no endpoint of its own (the `submit()` idiom):

            chat = client.chat(42)
            reply = chat.say("Bonjour, j'ai perdu ma réservation")
            chat.total_amount_eur   # Decimal, this session's euros
            chat.reset()            # close it, start over

        `say()` opens the session on its first call (which accepts the
        charter, after `wait_for_chat_agent` when the agent was asleep), waits for the answer, prints the scoring events it earned
        and the reply, and returns the reply text. `timeout` and
        `poll_interval` are passed to every `send_chat_message`; `echo=False`
        prints nothing.
        """
        return ChatConversation(self, challenge_id, timeout=timeout,
                                poll_interval=poll_interval, echo=echo)

    # ---- Chat challenges: staff routes (author scope + challenge access) -----

    def chat_admin(self, challenge_id: int) -> dict:
        """The staff view of a chat challenge — `ChatChallengeAdminView`.

        Mirrors `GET /api/chat/challenge/{cid}/admin`. Returns the
        LLM connection (`llm_base_url`, `llm_model`, `llm_api_key_set` — a
        boolean, the key itself is never served), the limits, the agent's
        health (`env_status`, `env_status_message`, `env_status_at_ts`,
        `worker_ref`, `last_seen_at_ts`, `agent_online`), the registered
        `manifest` and `counts` (`sessions`, `open_sessions`, `turns`,
        `scoring_events`, `total_amount_eur`, `participants`).

        Requires an `author`-scope token and access to the challenge.
        """
        return self._chat_call(
            "GET", f"/chat/challenge/{challenge_id}/admin",
            error_label="chat_admin",
        ).json()

    def update_chat_settings(self, challenge_id: int, *,
                             llm_base_url: str | _Unset = _UNSET,
                             llm_model: str | _Unset = _UNSET,
                             llm_api_key: str | _Unset = _UNSET,
                             turn_timeout_sec: int | _Unset = _UNSET,
                             max_turns_per_session: int | None | _Unset = _UNSET,
                             max_sessions_per_participant: int | None | _Unset = _UNSET,
                             max_concurrent_turns: int | _Unset = _UNSET,
                             ) -> dict:
        """Point the challenge at its LLM and set its limits.

        Mirrors `PUT /api/chat/challenge/{cid}/settings`
        (`UpdateChatSettingsRequest`) — the console's Chat tab.
        Only the keywords you pass are sent; the rest is left untouched.
        Allowed while the challenge is started (the agent reads the LLM
        config on every turn).

        - `llm_base_url` / `llm_model` — an OpenAI-compatible chat-completions
          endpoint (`…/v1`) and the model name. Both must be set before the
          challenge can start.
        - `llm_api_key` — write-only: `""` clears it, and `chat_admin()`
          reports only `llm_api_key_set`.
        - `turn_timeout_sec` — how long one answer may take.
        - `max_turns_per_session` / `max_sessions_per_participant` — pass
          `None` explicitly to lift a limit (unlimited).
        - `max_concurrent_turns` — how many participant messages the agent
          answers at the same time, 1..16 (default 1). A team never has two
          answers running at once, whatever the value. Above 1, one `Env`
          instance serves several turns in parallel: env.py must keep
          per-turn data in `session.state` / `session.participant_state` and
          treat its own attributes as read-only after `__init__`.

        Returns the updated `ChatChallengeAdminView`. Raises `MLArenaError`
        when called with no field at all.
        """
        passed = {
            "llm_base_url": llm_base_url,
            "llm_model": llm_model,
            "llm_api_key": llm_api_key,
            "turn_timeout_sec": turn_timeout_sec,
            "max_turns_per_session": max_turns_per_session,
            "max_sessions_per_participant": max_sessions_per_participant,
            "max_concurrent_turns": max_concurrent_turns,
        }
        body = {key: value for key, value in passed.items() if value is not _UNSET}
        if not body:
            raise MLArenaError(
                f"update_chat_settings requires at least one field from "
                f"{sorted(passed)}"
            )
        return self._chat_call(
            "PUT", f"/chat/challenge/{challenge_id}/settings",
            json_body=body, error_label="update_chat_settings",
        ).json()

    def chat_sessions(self, challenge_id: int,
                      status: str | None = None) -> dict:
        """Every session of the challenge, for its staff.

        Mirrors `GET /api/chat/challenge/{cid}/sessions`. Returns
        `{"sessions": [...]}` — each a `ChatSessionSummary` (user, team's
        submission, `status`, counts, `total_amount_eur`, and on a voided one
        `void_reason` / `voided_by_username` / `voided_at_ts`) plus
        `submission_name`. `status` filters on `open` | `closed` | `voided`;
        any other value is a 400.
        """
        params = {"status": status} if status is not None else None
        return self._chat_call(
            "GET", f"/chat/challenge/{challenge_id}/sessions",
            params=params, error_label="chat_sessions",
        ).json()

    def void_chat_session(self, session_id: int, reason: str) -> dict:
        """Annul a session — « l'annulation de la manche ».

        Mirrors `POST /api/chat/sessions/{sid}/void` `{"reason": ...}`:
        the session becomes `voided` and its scoring events leave the
        group's total (recomputed in the same transaction). Undo with
        `unvoid_chat_session()`. A 409 while a turn of the session is in
        flight, as on `close_chat_session()`: retry once the agent answered.

        Raises `ChatSessionNotFoundError` when the session is not on a
        challenge you may manage.
        """
        return self._chat_call(
            "POST", f"/chat/sessions/{session_id}/void",
            json_body={"reason": reason}, error_label="void_chat_session",
            not_found=ChatSessionNotFoundError,
        ).json()

    def unvoid_chat_session(self, session_id: int) -> dict:
        """Reinstate a voided session; its events count again.

        Mirrors `POST /api/chat/sessions/{sid}/unvoid`.

        Raises `ChatSessionNotFoundError` when the session is not on a
        challenge you may manage.
        """
        return self._chat_call(
            "POST", f"/chat/sessions/{session_id}/unvoid",
            error_label="unvoid_chat_session", not_found=ChatSessionNotFoundError,
        ).json()

    def export_chat_evidence(self, challenge_id: int) -> dict:
        """Every session's evidence in one document — the review dossier.

        Mirrors `GET /api/chat/challenge/{cid}/export`: the
        per-session export of `export_chat_session(format="json")` for every
        session of the challenge, in one JSON.
        """
        return self._chat_call(
            "GET", f"/chat/challenge/{challenge_id}/export",
            error_label="export_chat_evidence", timeout=120,
        ).json()

    # ---- Exercise challenges (`exercise_v1`) ---------------------------------

    def _exercise_call(self, method: str, path: str, *,
                       json_body: dict | None = None,
                       params: dict | None = None,
                       ok: tuple[int, ...] = (200,),
                       error_label: str,
                       not_found: type = ChallengeNotFoundError) -> dict:
        """Single chokepoint for the `/api/exercise/*` routes (`_chat_call`'s
        sibling). `not_found` names the class a 404 raises:
        `ChallengeNotFoundError` on a challenge-scoped route (an unknown or
        inactive exercise key included), `ExerciseAttemptNotFoundError` on an
        attempt-scoped one. Any other non-`ok` status — 409 (an attempt of
        this exercise is still in flight, the challenge is not started), 429
        (over the per-minute attempt cap), 400 — raises `MLArenaError` with
        the server's reason, `status_code` and `body`."""
        resp = self._request(
            method, self._url(path),
            headers=self._headers(json_body=json_body is not None),
            json=json_body, params=params, timeout=30,
        )
        self._handle_response(resp, not_found=not_found)
        if resp.status_code not in ok:
            raise _failed(MLArenaError, error_label, resp)
        return resp.json()

    def exercise_challenge(self, challenge_id: int) -> dict:
        """An exercise challenge and your progress on it — `ExerciseChallengeOut`.

        Mirrors `GET /api/exercise/challenge/{cid}` (`backend/app/views/exercise/`).
        Readable by whoever may see the challenge, and by anyone when a
        public, not archived course assigns the set (an exercise is part of
        its lesson); 404 otherwise.
        Returns `challenge_id`, `env_status` (`unknown` | `loading` | `ready`
        | `error`: the checker's load of the set's `env.py`, self-test
        included), `checker_online` (the resident checker claimed work within
        the last 30 s; a set idle for 30 min sleeps and wakes on the next
        attempt, which then takes 10–20 s), `time_limit_sec`, your group's
        `submission_id` (None before your first attempt), `score` (the
        percentage of points solved, None before your first attempt),
        `points_earned`, `points_available`, and `exercises` in order — each
        `ExerciseOut`: `key`, `position`, `kind` (`code` | `text` |
        `choice`), `title`, `points` (0 = practice: checked, never scored),
        `starter_code`, `examples`, `choices` (the options of a `choice`
        exercise, else None), and your state: `solved` (a passed attempt
        exists), `attempt_count`, `last_attempt` (`ExerciseAttemptOut` or
        None).

        Readable by anyone who may view the challenge; the per-exercise state
        is yours (your team's on a team challenge).

        Raises `ChallengeNotFoundError` when the challenge is not an exercise
        challenge or is hidden from you.
        """
        return self._exercise_call(
            "GET", f"/exercise/challenge/{challenge_id}",
            error_label="exercise_challenge",
        )

    def submit_exercise(self, challenge_id: int, key: str,
                        code: str | None = None, answer: str | None = None, *,
                        wait: bool = True, timeout: float = 60,
                        poll_interval: float = 0.4) -> dict:
        """Send one attempt at exercise `key` and, by default, wait for the verdict.

        Mirrors `POST /api/exercise/challenge/{cid}/exercises/{key}/attempts`
        (`CreateExerciseAttemptRequest`): `code` for a `code` exercise,
        `answer` for a `text` or `choice` one (a `choice` answer is one of
        the exercise's `choices`) — exactly one of the two, at most 64 KiB.
        The server answers 202 with the `pending` `ExerciseAttemptOut` as
        soon as the attempt is queued for the checker. Your group's
        submission is created on its first attempt (born active: nothing to
        upload or deploy).

        With `wait=True` (default) the call then polls `exercise_attempt()`
        every `poll_interval` seconds — a client-side composition of the two
        public routes, the console's own cadence; there is no wait endpoint —
        until the attempt is settled, and returns it:

        - `status="passed"`: solved (`error_type` None).
        - `status="failed"`: the verdict on your code or answer, with
          `error_type` `failed` (an assertion: its message is
          `feedback["assertion"]`), `syntax_error` / `runtime_error`
          (`feedback["line"]` is the line in your code, `traceback` the
          trimmed traceback), `timeout`, `memory` or `output_limit`;
          `feedback["stdout"]` is what your code printed.

        An attempt that ends `status="error"` is not a verdict on your code
        (`grader_error`: the checker failed, the exercise set's owner was told;
        `worker_offline`: no checker picked it up) and raises `MLArenaError`
        carrying the attempt as `.body` — send it again later. Not settled
        after `timeout` seconds raises `MLArenaError` naming the state the
        attempt was left in (the first attempt on a sleeping set waits for
        the checker to start).

        With `wait=False` the 202 body is returned as-is.

        Raises `MLArenaError` before any request unless exactly one of
        `code` / `answer` is given; `ChallengeNotFoundError` for an unknown or
        inactive key or a hidden challenge; `MLArenaError` with the server's
        reason on a refusal — 409 while an attempt of this exercise is still
        in flight for your group (`.body["attempt_id"]` names it), when the
        challenge is a draft (not started) or closed (stopped or ended), or
        when the set's owner has no compute credits left (`.body["reason"]
        == "out_of_credits"` beside `error`: the Check is paused, nothing is
        written; an attempt already queued is still checked); 429
        over the challenge's per-minute attempt cap (`.body
        ["max_attempts_per_minute"]`, per group, last 60 s); 400 for `code`
        on a text / choice exercise, `answer` on a code one, or a `choice`
        answer that is not one of its `choices`. `EnrollmentRequiredError`
        when you are not enrolled in the set's course.
        """
        if (code is None) == (answer is None):
            raise MLArenaError(
                "submit_exercise takes exactly one of code= (a code exercise) "
                "or answer= (a text or choice exercise)"
            )
        accepted = self._exercise_call(
            "POST", f"/exercise/challenge/{challenge_id}/exercises/{quote(key, safe='')}/attempts",
            json_body={"code": code, "answer": answer}, ok=(202,),
            error_label="submit_exercise",
        )
        if not wait:
            return accepted
        return self._wait_for_exercise_attempt(
            accepted["id"], timeout=timeout, poll_interval=poll_interval,
        )

    def _wait_for_exercise_attempt(self, attempt_id: int, *, timeout: float,
                                   poll_interval: float) -> dict:
        """Poll `exercise_attempt` until it settles (see `submit_exercise`)."""
        deadline = time.monotonic() + timeout
        while True:
            attempt = self.exercise_attempt(attempt_id)
            status = attempt["status"]
            if status in ("passed", "failed"):
                return attempt
            if status == "error":
                reason = f"exercise attempt {attempt_id} errored ({attempt['error_type']})"
                if attempt["message"]:
                    reason += f": {attempt['message']}"
                raise MLArenaError(reason, body=attempt)
            if time.monotonic() > deadline:
                raise MLArenaError(
                    f"submit_exercise timed out after {timeout}s: attempt "
                    f"{attempt_id} is still {status!r}. Poll "
                    f"exercise_attempt({attempt_id}) or raise timeout.",
                    body=attempt,
                )
            time.sleep(poll_interval)

    def exercise_attempt(self, attempt_id: int) -> dict:
        """One attempt — `ExerciseAttemptOut`.

        Mirrors `GET /api/exercise/attempts/{aid}`. Returns `id`,
        `challenge_id`, `exercise_key`, `user_id` (who sent it), `status`
        (`pending` | `running` | `passed` | `failed` | `error`), `error_type`
        (None unless failed or error; see `submit_exercise`), `message`,
        `feedback` (`ExerciseFeedback`: `assertion`, `traceback`, `stdout`,
        `line`, `expected_hint`; None until settled), `duration_ms`, the
        `code` or `answer` sent (the other is None), `created_at_ts` and
        `completed_at_ts`. The `message` of a `grader_error` is the
        checker's own text for the challenge's staff and a fixed sentence for
        everyone else.

        Readable by the attempt's group, the challenge's owner and assistants,
        admins and the student's teachers. Cheap enough to poll —
        `submit_exercise(wait=True)` does.

        Raises `ExerciseAttemptNotFoundError` when the attempt does not exist
        or is not yours to read.
        """
        return self._exercise_call(
            "GET", f"/exercise/attempts/{attempt_id}",
            error_label="exercise_attempt", not_found=ExerciseAttemptNotFoundError,
        )

    def exercise_attempts(self, challenge_id: int, user_id: int | None = None,
                          key: str | None = None) -> dict:
        """The attempts of an exercise challenge — `ExerciseAttemptListResponse`.

        Mirrors `GET /api/exercise/challenge/{cid}/attempts?user_id=&key=`.
        Returns `{"attempts": [ExerciseAttemptOut, ...]}`, the submitted
        `code` / `answer` included. A participant reads their own group's;
        the challenge's owner and assistants, admins and the teachers of a
        course the student is enrolled in read a student's (`user_id`).
        `key` keeps one exercise's. Only the filters passed are sent.
        """
        params = {name: value for name, value in (("user_id", user_id), ("key", key))
                  if value is not None}
        return self._exercise_call(
            "GET", f"/exercise/challenge/{challenge_id}/attempts",
            params=params or None, error_label="exercise_attempts",
        )

    def exercise_admin(self, challenge_id: int) -> dict:
        """The staff view of an exercise set — `ExerciseChallengeAdminOut`.

        Mirrors `GET /api/exercise/challenge/{cid}/admin`. Returns the
        checker's health (`env_status`, `env_status_message` — the load or
        self-test error, naming the exercise whose `solution` failed or whose
        `wrong` passed —, `env_status_at_ts`, `manifest_at_ts`,
        `checker_online`, `worker_ref`), the four limits (`time_limit_sec`,
        `memory_limit_mb`, `max_concurrent_attempts`,
        `max_attempts_per_minute`; written with `update_settings`) and
        `exercises`, every exercise the manifest ever posted — `key`,
        `position`, `kind`, `title`, `points`, `is_active` (False once a key
        left env.py; its attempts are kept), `attempt_count`, `solved_count`.

        Readable by the challenge's owner and assistants and admins; anyone
        else gets `ChallengeNotFoundError` (404), as for a non-exercise
        challenge.
        """
        return self._exercise_call(
            "GET", f"/exercise/challenge/{challenge_id}/admin",
            error_label="exercise_admin",
        )

    def exercise_list(self, challenge_id: int) -> dict:
        """An exercise set's exercises, for writing a lesson — `ExerciseListOut`.

        Mirrors `GET /api/exercise/challenge/{cid}/exercises`. Returns
        `exercises`, each `{key, title, kind, points}` (the `key` is what a
        lesson's `mlarena:exercise id=<cid> key=<key>` directive names; never
        a check, a solution or an accepted answer), `source` and `reason`:

        - `source="file"`: read from the set's `exercises.json` (its env.py
          loads it), whatever the set's status;
        - `source="manifest"`: the exercises its checker registered the last
          time the set was started (a hand-written env.py);
        - `source="none"`: nothing known yet; `reason` says why (start the
          set once).

        `reason` is also set when the file holds no exercise or does not fit
        the format. For whoever may edit the set: its staff (the owner and
        assistants of its course) and admins; `ChallengeNotFoundError` (404)
        otherwise. Reading it is not activity: it wakes no checker.
        """
        return self._exercise_call(
            "GET", f"/exercise/challenge/{challenge_id}/exercises",
            error_label="exercise_list",
        )

    def exercise_definition(self, challenge_id: int) -> dict:
        """An exercise set's exercises as data — `ExerciseDefinitionOut`.

        Mirrors `GET /api/exercise/challenge/{cid}/definition`. Returns
        `file` (the env folder's `exercises.json`, the document itself:
        `{"version": 1, "exercises": [...]}`; None when there is none or
        when it does not fit the format, `file_problem` then says why),
        `env_py` (`generated`: the two-line loader the platform wrote;
        `edited`: your own Python; `missing`), `loads_exercises_file`
        (whether env.py calls `load_exercises("exercises.json")`, that
        literal: when False the set serves what env.py declares, the file is
        not read and the form does not write it), `write_blocked_by`
        (None when `update_exercise_definition` is accepted now, else
        `env_py`: the hand-written env.py does not load the file; a started
        set is not a block, it takes the write as an update of its files)
        and `env_update` (the state of the env folder's update, the
        shape of `env_update(id)`: while one is pending, `file` is the
        pending one's). Whoever reads it writes every kind of exercise, a
        `code` exercise's Python `check` included.

        Each exercise has `key`, `kind`, `title` (the prompt), `points`
        (0 = practice) and `examples`, then by kind — `code`: `starter`,
        `check` (Python run with `student` and `forbid` defined; it passes
        when it raises nothing), `solution`, `wrong`; `text`: `accept` (the
        accepted answers), `case_sensitive`, `wrong`; `choice`: `choices`,
        `solution`. The format is documented in the platform's
        `workers/exercise_v1/PROCESS.md` ("exercises.json").

        For whoever may edit the set (owner, assistants, admins), with a
        session or an `author` key; anyone else gets
        `ChallengeNotFoundError` (404), as for a challenge of another kind.
        """
        return self._exercise_call(
            "GET", f"/exercise/challenge/{challenge_id}/definition",
            error_label="exercise_definition",
        )

    def update_exercise_definition(self, challenge_id: int, exercises: list[dict]) -> dict:
        """Write an exercise set's exercises (the whole list) — no Python file needed.

        Mirrors `PUT /api/exercise/challenge/{cid}/definition`: the body is
        the `exercises.json` document, `{"version": 1, "exercises":
        exercises}`, validated strictly (an unknown field, a `check` that
        does not compile, a `wrong` answer that is accepted, a repeated key
        are a 400 naming the exercise and the field) and written to the env
        folder; a folder without env.py gets the generated loader. Returns
        `ExerciseDefinitionOut` (see `exercise_definition`). The stored
        self-test is forgotten: run `self_test_exercises` again.

            client.update_exercise_definition(cid, [
                {"key": "bmi", "kind": "code", "title": "Body-mass index", "points": 1,
                 "starter": "def bmi(weight_kg, height_m):\n    return ...\n",
                 "check": "assert student.function('bmi')(70, 1.75) == 22.9\n",
                 "solution": "def bmi(w, h):\n    return round(w / h ** 2, 1)\n",
                 "wrong": ["def bmi(w, h):\n    return w / h ** 2\n"]},
                {"key": "floor-div", "kind": "text", "title": "What does 7 // 2 print?",
                 "points": 1, "accept": ["3"], "wrong": ["3.5"]},
                {"key": "mutable", "kind": "choice", "title": "Which one is mutable?",
                 "points": 1, "choices": ["tuple", "list"], "solution": "list"},
            ])

        Every kind is the set's staff's to write, a `code` exercise and its
        Python `check` included. Raises
        `MLArenaError` (409, `.body["write_blocked_by"]`: `env_py`) when its
        hand-written env.py does not load `exercises.json`. A started set
        takes the write too, as an update of its environment files: a new
        pod loads them about a minute later and its load is their check
        (`env_update`), a pod the set's owner pays, so the write is a 409
        with `.body["reason"] == "out_of_credits"` when the owner has no
        compute credits left; `MLArenaError` (400) for a
        document that does not fit the format, or whose file would be larger
        than 1 000 000 bytes (`.body["size"]`).
        """
        return self._exercise_call(
            "PUT", f"/exercise/challenge/{challenge_id}/definition",
            json_body={"version": 1, "exercises": exercises},
            error_label="update_exercise_definition",
        )

    def exercise_self_test(self, challenge_id: int) -> dict:
        """Where an exercise set's self-test stands — `ExerciseSelfTestOut`.

        Mirrors `GET /api/exercise/challenge/{cid}/self-test`. Returns
        `status` (`none` | `requested` | `running` | `done` | `timed_out`),
        `self_test_requested_at_ts`, `self_test_at_ts` and, once `done`:
        `passed` (the env loaded and every exercise's solution passed and
        every `wrong` failed), `self_test_error` (the first failure, or why
        the env did not load; None on a pass) and `self_test`, one `{key,
        passed, message}` per exercise (None when the env did not load; a
        check that does not finish in time fails with a timeout message). On
        a started set, `done` is its checker's last load.
        """
        return self._exercise_call(
            "GET", f"/exercise/challenge/{challenge_id}/self-test",
            error_label="exercise_self_test",
        )

    def self_test_exercises(self, challenge_id: int, *, wait: bool = True,
                            timeout: float = 300, poll_interval: float = 2) -> dict:
        """Self-test an exercise set before starting it.

        Mirrors `POST /api/exercise/challenge/{cid}/self-test` (202): the
        platform starts a checker for the set, which loads the env, runs
        every exercise's `solution` (it must pass) and `wrong` answers (each
        must fail), reports and goes away; it serves no participant and
        changes no score, no exercise and no status of the set. The checker
        has to start first: expect 20–60 s, more on a cold machine. For a
        draft or a stopped set.

        With `wait=True` (default) the call then polls `exercise_self_test`
        every `poll_interval` seconds (a client-side composition of the two
        public routes) until `status` is `done`, and returns that
        `ExerciseSelfTestOut`: read `passed` and `self_test`. `timed_out`,
        or no result after `timeout` seconds, raises `MLArenaError` with the
        last state as `.body`. With `wait=False` the 202 body is returned.

        The set's staff write it, self-test it and start it, with the
        Python it holds (a set copied from an example comes with its
        checks). Raises `MLArenaError`
        with the server's reason: 409 on a started set
        (its checker self-tests at load: stop the set to self-test an edit),
        on an ended one (restart it), and while a self-test of this set is
        already pending (one at a time; it settles or times out after 10
        minutes); the start gate's refusal, since a self-test starts a
        checker where a start would (400 for authoring still to do, such as
        an `exercises.json` that is missing, invalid or empty; 409 when the
        set cannot run on its machine or its owner has no compute credits
        left, the blocker `out_of_credits`; `.body["start_blockers"]` lists
        them);
        429 when you already have 2 self-tests pending, or the platform 4
        (`.body["limit"]`), and within 60 s of your previous self-test
        request, on any set (`.body["retry_after_sec"]`; an admin is exempt
        from that wait); 409 with `reason: "test_runs_cap"` when the set's
        owner already has 3 test runs queued or running (benchmark runs and
        self-tests, all their challenges together; an admin owner is
        exempt).
        """
        state = self._exercise_call(
            "POST", f"/exercise/challenge/{challenge_id}/self-test", ok=(202,),
            error_label="self_test_exercises",
        )
        if not wait:
            return state
        deadline = time.monotonic() + timeout
        while True:
            if state["status"] == "done":
                return state
            if state["status"] == "timed_out":
                raise MLArenaError(
                    f"self_test_exercises: no checker reported for challenge "
                    f"{challenge_id} (status 'timed_out'): its machine may have no "
                    f"room; ask again later.",
                    body=state,
                )
            if time.monotonic() > deadline:
                raise MLArenaError(
                    f"self_test_exercises timed out after {timeout}s: the self-test "
                    f"is still {state['status']!r}. Poll "
                    f"exercise_self_test({challenge_id}) or raise timeout.",
                    body=state,
                )
            time.sleep(poll_interval)
            state = self.exercise_self_test(challenge_id)

    def check(self, challenge_id: int, key: str, obj, *, timeout: float = 60,
              echo: bool = True) -> dict:
        """Check a function, a class or a str of code against a `code` exercise.

        A client-side composition — no endpoint of its own (the `submit()`
        idiom): sends `source_of(obj)` — the object's source through
        `inspect.getsource`, or the `str` as it is — with
        `submit_exercise(challenge_id, key, code=…, wait=True)`, prints the
        verdict line (`format_attempt`: `imc: passed (8 ms)`, or the status,
        the assertion, the line and the output) unless `echo=False`, and
        returns the settled `ExerciseAttemptOut`:

            def calculer_imc(poids_kg, taille_m):
                return round(poids_kg / taille_m ** 2, 1)

            client.check(42, "imc", calculer_imc)
            client.check(42, "patients", "import math\\n\\ndef ...")

        Only the object's own definition is sent: helpers it calls and the
        imports it relies on go in a `str`. A `text` or `choice` exercise
        takes `submit_exercise(…, answer=…)`. Raises as `submit_exercise`
        does.
        """
        attempt = self.submit_exercise(
            challenge_id, key, code=source_of(obj), wait=True, timeout=timeout,
        )
        if echo:
            print(format_attempt(attempt))
        return attempt

    # ---- Shared JSON-call helper ---------------------------------------------

    def _json_call(self, method: str, path: str, *,
                   json_body: dict | None = None,
                   params: dict | None = None,
                   ok: tuple[int, ...] = (200,),
                   error_label: str = "request",
                   timeout: int = 30) -> dict | list:
        """One authenticated JSON request: the course-content routes
        (`/teacher`, `/academic_courses`), `workspace()` and the challenge
        lifecycle routes of `/creator_challenge` that need nothing more than
        send, map errors, return `.json()`. (Named `_course_call` before 4.19.0, when
        the course methods were its only callers.)

        The bearer token is always sent, even on public reads: the consumption
        landing/lesson routes personalize their response when authenticated
        (own progress, ``can_manage`` drafts, gated bodies for the owner), and
        the public routes ignore it. ``_handle_response`` maps 401/403/404 to the
        SDK's auth/not-found exceptions; any other non-``ok`` status raises
        ``MLArenaError`` with the backend's error text.
        """
        resp = self._request(
            method, self._url(path),
            headers=self._headers(json_body=json_body is not None),
            json=json_body, params=params, timeout=timeout,
        )
        self._handle_response(resp)
        if resp.status_code not in ok:
            raise _failed(MLArenaError, error_label, resp)
        return resp.json()

    def _upload_course_file(self, method: str, path: str, file_path: str, *,
                            ok: tuple[int, ...] = (200, 201),
                            error_label: str = "upload",
                            timeout: int = 120) -> dict:
        """Multipart helper for the cover upload route (sibling of
        ``upload_env_file``)."""
        if not os.path.isfile(file_path):
            raise MLArenaError(f"File not found: {file_path}")
        with open(file_path, "rb") as fh:
            resp = self._request(
                method, self._url(path),
                headers=self._headers(),
                files={"file": (os.path.basename(file_path), fh)},
                timeout=timeout,
            )
        self._handle_response(resp)
        if resp.status_code not in ok:
            raise _failed(MLArenaError, error_label, resp)
        return resp.json()

    # ---- Compute credits (any token scope; the staff readings: author) -----

    def credits(self) -> dict:
        """Your two balances of compute credits, CPU then GPU.

        Mirrors `GET /api/credits`. A credit pays for time on a machine: the
        minutes a run, or a chat or exercise pod, spends there, at the
        machine's rate, whatever its size. The owner of a challenge pays for
        its compute (a course challenge's owner is the course's owner); a
        participant pays nothing. A machine draws from one balance: the GPU
        machine from `gpu`, every other from `cpu`.

        Returns `{"balances": [cpu, gpu]}`, each `{resource, balance, debt,
        full_grant, pct_used, alert_pct, grants}`: `balance` is negative
        while it owes (`debt`), `pct_used` the share of `full_grant` used
        (0 to 100), `alert_pct` the alert level reached (80, 100 or None),
        and `grants` (oldest first) each `{id, kind, amount, consumed,
        expires_at_ts, note, created_at_ts, is_active}`, `kind` being
        `entry` (the grant a new account starts with), `admin` or `refill`.
        Amounts are four-decimal strings (`"5000.0000"`): parse them with
        `decimal.Decimal`. The first read creates the two balances (the CPU
        one with its entry grant). More credits come from an admin.
        """
        return self._json_call("GET", "/credits", error_label="credits")

    def credit_usage(self, resource: str, group_by: str, since: str | None = None,
                     course_id: int | None = None,
                     challenge_id: int | None = None) -> dict:
        """What was charged on a balance, summed by course, challenge or kind.

        Mirrors `GET /api/credits/usage`. `resource` is `cpu` or `gpu` (the
        two kinds of credits are never added up); `group_by` is `course`,
        `challenge` or `kind` (`participant_run`, `scheduled_run`,
        `author_run`, `chat_pod`, `exercise_pod`); `since` (`YYYY-MM-DD`,
        00:00 UTC) keeps the charges from that day on, every charge without
        it. 400 on a bad value.

        Without `course_id` / `challenge_id` it reads your own account. With
        one of them (never both) it reads what that course or that challenge
        was charged, whoever paid, for its staff with an `author` key (a
        course you do not manage is a 403, a challenge you do not edit a
        404; both ids at once a 400). A run's own credits are on its row for
        staff (`submission_runs()` / `creator_runs()`: `credits`).

        Returns `{resource, group_by, since, course_id, challenge_id, amount,
        seconds, unattended_amount, rows}`: `amount` (a four-decimal string)
        and `seconds` the totals, `unattended_amount` the part no
        participant caused (scheduled runs, chat and exercise pods), `rows`
        largest amount first, keyed by `group_by` (`course_id` +
        `course_name`; `challenge_id`, `challenge_name`, `course_id`,
        `course_name`; or `kind`) with their `amount` and `seconds`. A null
        id groups what belongs to no course or to a deleted
        challenge or course. Writes nothing.
        """
        params: dict = {"resource": resource, "group_by": group_by}
        for key, value in (("since", since), ("course_id", course_id),
                           ("challenge_id", challenge_id)):
            if value is not None:
                params[key] = value
        return self._json_call(
            "GET", "/credits/usage", params=params, error_label="credit_usage",
        )

    # ---- Academic courses: create / enroll / list (author + user scope) -----

    def create_course(self, name: str, code: str | None = None,
                      start_date: str | None = None, end_date: str | None = None,
                      instructor_name: str | None = None,
                      *,
                      slug: str | None = None,
                      description: str | None = None,
                      visibility: str | None = None,
                      ask_student_identity: bool | None = None,
                      teacher_user_id: int | None = None) -> dict:
        """Create a course; you own it (any signed-in user).

        Mirrors `POST /api/academic_courses/` (`academic_courses/courses.py`).
        Creating a course works with a key of either scope. Each account may
        own at most `max_courses_limit` active (not archived) courses
        (default 10); admins are exempt. At the limit the 403 states both
        numbers; `archive_course` or `delete_course` frees a place. Authoring
        its content (modules, lessons, challenges) takes an `author`-scope
        key from your Profile page.

        `start_date` / `end_date` are optional ISO date strings (`YYYY-MM-DD`) —
        a name is the only required field, so a course can be created without
        term dates (a course with no end date never closes). They are kept
        positional for backwards compatibility with the original signature. The
        backend mints the course's `join_code` — the single enrollment token —
        and returns it in the course dict; share that, or the
        `/enroll/<join_code>` link built from it, with students.

        Keywords:
            slug:        kebab URL slug (`^[a-z0-9-]+$`); auto-derived from the
                         name when omitted.
            description: markdown landing-page intro.
            visibility:  `private` (default) or `public`: a course is one or
                         the other, and so is everything in it (a public
                         course, its challenges and their boards are read by
                         everyone; a private one by its members). Any other
                         value is a 400. Only `public` courses appear in
                         `course_catalog()`.
            ask_student_identity: whether enrolling asks for the student
                         number and email; False (the default) unless you
                         turn it on, here or with `update_course`.
            teacher_user_id: an admin may name the owning teacher; otherwise the
                         caller owns the course.

        The course starts empty: build a challenge in it with
        `add_course_challenge`, or copy one; `create_challenge_for_class`
        makes a course and its challenge in one call.
        """
        body: dict = {"name": name}
        if start_date is not None:
            body["start_date"] = start_date
        if end_date is not None:
            body["end_date"] = end_date
        if code is not None:
            body["code"] = code
        if instructor_name is not None:
            body["instructor_name"] = instructor_name
        if slug is not None:
            body["slug"] = slug
        if description is not None:
            body["description"] = description
        if visibility is not None:
            body["visibility"] = visibility
        if ask_student_identity is not None:
            body["ask_student_identity"] = ask_student_identity
        if teacher_user_id is not None:
            body["teacher_user_id"] = teacher_user_id
        return self._json_call(
            "POST", "/academic_courses/", json_body=body,
            ok=(200, 201), error_label="create_course",
        )

    def enrollment_info(self, join_code: str) -> dict:
        """Preview a course from its join code (without enrolling).

        Mirrors `GET /api/academic_courses/enroll/<join_code>` — the same public
        lookup the console's EnrollPage makes. Returns
        `{course, challenge_names, lesson_count, already_enrolled,
        ask_student_identity, has_student_info}`; the course sits under
        `course` with its own field names (`id`, `name`, `code`, `slug`, ...,
        `enrollment_state`).
        `ask_student_identity` is the course's setting: when it is False,
        `enroll_in_course` needs no student number or email.
        `already_enrolled` and `has_student_info` stay false for an anonymous
        caller. An archived or ended course answers 410, except to a caller
        who is already enrolled in it (`already_enrolled: true`).
        """
        return self._json_call(
            "GET", f"/academic_courses/enroll/{join_code}",
            error_label="enrollment_info",
        )

    def enroll_in_course(self, join_code: str | None = None,
                         student_email: str | None = None,
                         student_number: str | None = None) -> dict:
        """Enroll the authenticated user in a course.

        Mirrors `POST /api/academic_courses/enroll/<join_code>`. A course has a
        single enrollment token, its `join_code` — pass it positionally or as
        `join_code=`. A full `/enroll/<join_code>` URL is not accepted; pass the
        code itself.

        `student_email` / `student_number` fill the caller's global student
        identity when it isn't set yet (they never overwrite values already on
        the profile — set those via the Profile page / profile update). The
        backend requires both to be present before enrolling, unless the course
        does not ask for them (`enrollment_info()["ask_student_identity"]`,
        a course setting). `student_email` must be an email address: any
        other value is a 400 naming the field, nothing is stored and the call
        can be made again (the same rule as `update_profile` and a course's
        `report_recipients`). An archived or ended course answers 410. Returns
        `{message, challenge_ids, course_slug}`.
        """
        if join_code is None:
            raise MLArenaError("enroll_in_course requires a join code")
        body: dict = {}
        if student_email is not None:
            body["student_email"] = student_email
        if student_number is not None:
            body["student_number"] = student_number
        return self._json_call(
            "POST", f"/academic_courses/enroll/{join_code}", json_body=body,
            ok=(200, 201), error_label="enroll_in_course",
        )

    def list_courses(self, *, show_all: bool = False):
        """List courses (enrolled + currently-active by default).

        Mirrors `GET /api/academic_courses/`. With `show_all=True` returns every
        course. Each row carries `is_enrolled` for the caller,
        `enrollment_state`: `open`, or why the course takes no new enrollment
        (`archived`, `ended`: its end date is past), decided by the server
        with the rule `enroll_in_course` refuses by, `single_challenge_id`
        (the id of the course's only challenge when the course is shown as
        that challenge: one module, one challenge, no lesson; else None) and
        `example_listed_at_ts` (set on an example course). A private course
        is listed only to its members and staff, and `join_code` is set only
        on the courses the caller belongs to or manages (None elsewhere). An
        enrolled row also carries the caller's `progress`, `next_lesson`,
        `progress_state` and `last_active_at_ts` (None on the others).
        Returns a DataFrame if pandas is installed, else a list of dicts.

        A challenge's course is `challenge(id)["course_id"]` (4.23.0 removed
        `challenge_id=`: a challenge belongs to one course).
        """
        params: dict = {}
        if show_all:
            params["show_all"] = "true"
        return _to_dataframe(self._json_call(
            "GET", "/academic_courses/", params=params or None,
            error_label="list_courses",
        ))

    # ---- Course consumption (learner-facing reads + progress) ---------------

    def course_catalog(self, search: str | None = None,
                       limit: int | None = None, offset: int | None = None,
                       *, example: bool = False) -> dict:
        """List public courses (search / paginate). Public read.

        Mirrors `GET /api/academic_courses/catalog`. Returns
        `{courses, total, limit, offset}` — only public courses appear.
        `search` matches name + description. A card carries `has_cover`
        (fetch the image with `course_cover(card["id"])`),
        `example_listed_at_ts`, `single_challenge_id` and, for a course
        shown as its one challenge, `single_challenge` (the challenge's
        reference with its `description` and `miniature`; its `can_copy`
        says whether you may copy it into a course of yours).

        `example=True` lists the example courses only, newest listing
        first: public courses an admin offers to teachers as starting
        points (the console's teacher view of the courses page, titled
        Catalog). Start from one with `start_from_example_course`, or take
        one of its challenges into a course of yours with
        `use_example_challenge`. A copy is a challenge you create: accept
        the terms of use once first (`accept_terms`). Every copy starts with
        the same data and the same scoring files: change the seed or the
        data before you start a challenge.
        """
        params: dict = {}
        if search is not None:
            params["search"] = search
        if example:
            params["example"] = "true"
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset
        return self._json_call(
            "GET", "/academic_courses/catalog", params=params or None,
            error_label="course_catalog",
        )

    def course(self, slug: str, as_student: bool = False) -> dict:
        """Course landing: meta + ordered modules + lesson TOC + own progress.

        Mirrors `GET /api/academic_courses/{slug}`. Lesson bodies are omitted
        here (use `lesson(...)`). When authenticated, the response includes
        `is_enrolled`, `can_manage`, and a `progress` summary; managers
        (teacher/TA/admin) additionally see unpublished draft lessons.
        `has_cover` says whether there is a cover image — fetch it with
        `course_cover(landing["id"])`. `enrollment_state` (`open` |
        `archived` | `ended`) says whether the course takes a new enrollment;
        `join_code` is its enrollment token for its members and, on a public
        course that takes enrollment, for anyone (None otherwise): what
        `enroll_in_course` takes. `single_challenge_id` is set when the
        course is shown as its one challenge; `example_listed_at_ts` when it
        is an example course. An archived public course is a 404 to anyone
        who is not its member.
        `show_module_headings` is False when you are served a single module:
        the course is then its content, with no module heading. Each lesson
        of the TOC carries `visible_to_students` (it is published and so is
        its module), next to its own `is_published`.

        A module's `challenges` are the course's challenges it holds, in
        order. A challenge that is a draft you do not edit is not open yet:
        its `challenge` and `metric` are None, and it carries only its
        `pass_threshold`. No field of the draft is served before its first
        Start (`my_progress` has no cell for it, `enrollment_info` does not
        name it). A challenge has one title, its name. Each `challenge` is
        the challenge's reference (see `challenge_overview`); its `can_copy`
        says whether you may copy that challenge into a course of yours
        (`use_example_challenge`; a 404 where it is False), which is how an
        example course offers its challenges.

        `as_student=True` ("View as student", `?as_student=true`) serves a
        manager of the course what an enrolled student gets: no module or
        lesson hidden from students, gated lessons open, `can_manage` False
        and `as_student` True in the answer. The answer describes an
        enrolled student who has done nothing yet: `is_enrolled` is True and
        `progress` is the empty summary (`completed` 0 of the lessons
        students see), whatever your own enrollment and progress. Nothing is
        written (a progress write still needs your own enrollment) and it
        changes nothing for anyone who does not manage the course. The
        same flag is on `module_overview` and `lesson`.
        """
        return self._json_call(
            "GET", f"/academic_courses/{slug}", params=_as_student(as_student),
            error_label="course",
        )

    def module_overview(self, slug: str, module_slug: str,
                        as_student: bool = False) -> dict:
        """A course module's overview: lesson list + its challenge cards.

        Mirrors `GET /api/academic_courses/{slug}/modules/{module_slug}`.
        Each challenge card's `challenge` is the challenge's reference, with
        `can_copy` as on `course`. `as_student=True`: as on `course` (a
        module hidden from students is then a 404).
        """
        return self._json_call(
            "GET", f"/academic_courses/{slug}/modules/{module_slug}",
            params=_as_student(as_student), error_label="module_overview",
        )

    def lesson(self, slug: str, module_slug: str, lesson_slug: str,
               as_student: bool = False) -> dict:
        """Full lesson body + server-resolved `mlarena:` directive payloads.

        Mirrors
        `GET /api/academic_courses/{slug}/modules/{module_slug}/lessons/{lesson_slug}`.
        A gated lesson requires enrollment (or manage rights) — otherwise the
        backend returns 403, surfaced here as `AuthenticationError`. The
        returned dict carries `body_md`, `directives`, `directive_warnings`,
        `is_enrolled` and `enrollment_state` (what a reader who is not
        enrolled needs to be offered Enroll).
        A directive naming a challenge you may not see is left out, with the
        warning of a challenge that does not exist. A `challenge` /
        `competition` directive's payload is `{challenge_id, challenge,
        description}`, `challenge` being the challenge's reference (see
        `challenge_overview`; its `can_copy` says whether you may copy the
        challenge into a course of yours, a 404 from the copy methods where
        it is False). `as_student=True`: as on `course` (a lesson
        hidden from students is then a 404, and a block whose challenge the
        course's students do not see is left out, with its warning).
        """
        return self._json_call(
            "GET",
            f"/academic_courses/{slug}/modules/{module_slug}/lessons/{lesson_slug}",
            params=_as_student(as_student), error_label="lesson",
        )

    def mark_lesson_viewed(self, lesson_id: int) -> dict:
        """Mark a lesson in-progress + bump last-viewed/last-active (idempotent).

        Mirrors `POST /api/academic_courses/lessons/{id}/view`. A lesson has
        one course (its module's): you must be enrolled in it (403 "Not
        enrolled in this lesson's course" otherwise). Returns the progress
        row: `id`, `user_id`, `lesson_id`, `status`, `completed_at_ts`.
        """
        return self._json_call(
            "POST", f"/academic_courses/lessons/{lesson_id}/view",
            error_label="mark_lesson_viewed",
        )

    def mark_lesson_complete(self, lesson_id: int) -> dict:
        """Mark a lesson completed (upsert progress, bump last-active).

        Mirrors `POST /api/academic_courses/lessons/{id}/complete`. Needs your
        enrollment in the lesson's course, as `mark_lesson_viewed`.
        """
        return self._json_call(
            "POST", f"/academic_courses/lessons/{lesson_id}/complete",
            error_label="mark_lesson_complete",
        )

    def mark_lesson_incomplete(self, lesson_id: int) -> dict:
        """Undo a completion — back to in-progress, `completed_at_ts` cleared.

        Mirrors `POST /api/academic_courses/lessons/{id}/uncomplete`. The tick
        is student-self-reported, so it is reversible. Needs your enrollment
        in the lesson's course, as `mark_lesson_viewed`.
        """
        return self._json_call(
            "POST", f"/academic_courses/lessons/{lesson_id}/uncomplete",
            error_label="mark_lesson_incomplete",
        )

    def my_progress(self, course_id: int) -> dict:
        """Own progress across a course: content % + next lesson + progress
        state + challenge results.

        Mirrors `GET /api/academic_courses/{course_id}/progress/me`. Requires
        enrollment (or manage rights). `progress_state` is `not_started`,
        `in_progress` (a lesson opened, the course not done) or `done` (every
        published lesson completed) — the same value `list_courses()` serves
        on the caller's enrolled row. Each challenge cell carries `value`,
        `pass_threshold`, `passed` (null when there is nothing to judge) and
        `metric`, the challenge's ranking MetricSpec (its `order` "desc":
        `passed` means value >= bar, "asc": <=; its `label`, `format` and
        `precision` say how to read `value`, the best submission's `score`).
        `best_submission_id` is the submission behind `value` when the caller
        may open it (their own, or a teammate's). `status` says where you
        stand on the challenge (`not_submitted` | `failed` | `scoring` |
        `scored`) and `team_name` names your team on it (else None).
        """
        return self._json_call(
            "GET", f"/academic_courses/{course_id}/progress/me",
            error_label="my_progress",
        )

    # ---- Course authoring: modules (author scope) ---------------------------

    def get_module(self, module_id: int) -> dict:
        """Module detail + lesson TOC + its challenges.

        Mirrors `GET /api/teacher/modules/{id}`. A module is a section of one
        course (`course_id`); its course's staff read it (its owner, its
        assistants, an admin). Carries `id`, `course_id`, `title`, `slug`,
        `summary`, `icon`, `is_published`, `lesson_count`,
        `challenge_count`, `content_origin`, `source_revision`, `lessons`
        and `challenges` (each `challenge_id`, `challenge`, `position`,
        `pass_threshold`, `metric`; `challenge` is the challenge's
        reference, `can_copy` included: see `challenge_overview`).
        """
        return self._json_call(
            "GET", f"/teacher/modules/{module_id}", error_label="get_module",
        )

    def update_module(self, module_id: int, **fields) -> dict:
        """Update module meta. Mirrors `PUT /api/teacher/modules/{id}`.

        A module is edited by its course's staff: the course's owner, its
        assistants, an admin. Accepts `title`, `slug`, `summary`, `icon`,
        `is_published`. Only the fields you pass are changed, so
        `update_module(id, is_published=False)` sends a module back to draft
        (hidden from students, with its lessons and challenges) without
        touching anything else. A new `slug` changes the module's URLs (409
        when its course already has a module with it); its lessons and their
        progress are kept. A module managed by a bundle
        (`content_origin == "bundle"`) refuses edits with a 409 naming its
        source: change the files and push again, or `detach_module`.
        """
        allowed = {"title", "slug", "summary", "icon", "is_published"}
        body = _filtered_fields(fields, allowed, "update_module")
        return self._json_call(
            "PUT", f"/teacher/modules/{module_id}", json_body=body,
            error_label="update_module",
        )

    def delete_module(self, module_id: int) -> dict:
        """Delete a module with its lessons and its assets — the course's
        owner or an admin (403 for an assistant).

        Mirrors `DELETE /api/teacher/modules/{id}`. Refused (409, with
        `challenge_count` in `.body`) while the module holds a challenge: a
        challenge never leaves its course, so delete or archive it
        (`delete_challenge`, `archive_challenge`) or move it to another
        module of the course (`update_course_challenge(..., module_id=…)`)
        first.
        """
        return self._json_call(
            "DELETE", f"/teacher/modules/{module_id}",
            error_label="delete_module",
        )

    def detach_module(self, module_id: int) -> dict:
        """Hand a bundle-managed module back to the editor — the course's
        owner or an admin (403 for an assistant).

        Mirrors `POST /api/teacher/modules/{id}/detach`. A module last
        written by a course bundle (`push_course`, `mlarena course push`) has
        `content_origin == "bundle"` and refuses editor writes; detaching
        sets it to `editor`. The next push of a bundle that names the module
        takes it over again. Returns the module detail.
        """
        return self._json_call(
            "POST", f"/teacher/modules/{module_id}/detach",
            error_label="detach_module",
        )

    def update_course_challenge(self, module_id: int, challenge_id: int, /,
                                **fields) -> dict:
        """Set the pass bar of a course challenge, or move it to another
        module of its course, or both.

        Mirrors `PUT /api/teacher/modules/{module_id}/challenges/{challenge_id}`
        (`module_id` is the module the challenge is in now). Keywords, the
        route's keys: `pass_threshold` (the course's validation bar: a
        student validates the challenge when their best leaderboard value
        meets it in the direction the challenge ranks, `>=` for an `order`
        "desc", `<=` for "asc"; `None` removes the bar; an exercise set's is
        0 to 100) and `module_id` (a module of the same course: the
        challenge goes after its last challenge; another course's module is
        a 400, since a challenge never leaves its course). 409 when the
        challenge's current module is bundle-managed (its pass bar and its
        move out are the bundle's); moving it into a bundle-managed module is
        allowed. Partial: a key left out is left alone. Returns the
        challenge's row as the module lists it (`challenge_id`, `challenge`,
        `position`, `pass_threshold`, `metric`). The course's staff.
        """
        unknown = set(fields) - {"pass_threshold", "module_id"}
        if unknown:
            raise MLArenaError(
                f"update_course_challenge: unknown field(s) {sorted(unknown)}. "
                "Allowed: ['module_id', 'pass_threshold']"
            )
        if not fields:
            raise MLArenaError(
                "update_course_challenge requires pass_threshold= or module_id="
            )
        return self._json_call(
            "PUT",
            f"/teacher/modules/{module_id}/challenges/{challenge_id}",
            json_body=fields, error_label="update_course_challenge",
        )

    def reorder_module_challenges(self, module_id: int,
                                    ordered_challenge_ids: list[int]) -> list:
        """Reorder the challenges of a module.

        Mirrors `PUT /api/teacher/modules/{id}/challenges/reorder`.
        `ordered_challenge_ids` must be exactly the ids of the module's
        challenges, in the new order (400 otherwise, `expected` in `.body`).
        Returns the module's challenge rows in that order. 409 while a
        course bundle manages the module.
        """
        return self._json_call(
            "PUT", f"/teacher/modules/{module_id}/challenges/reorder",
            json_body={"ordered_ids": ordered_challenge_ids},
            error_label="reorder_module_challenges",
        )

    # ---- Course authoring: lessons (author scope) ---------------------------

    def create_lesson(self, module_id: int, title: str, kind: str = "lesson",
                      slug: str | None = None,
                      parent_lesson_id: int | None = None,
                      body_md: str = "", gated: bool = False) -> dict:
        """Create a lesson inside an owned module (starts as an unpublished draft).

        Mirrors `POST /api/teacher/modules/{id}/lessons`. `kind` is `lesson`
        (default) | `exercise`. `parent_lesson_id` nests it one level under
        another lesson in the same module. Publish it later via
        `update_lesson(id, is_published=True)`. `slug` is derived from the
        title when omitted (suffixed until free in the module); one you pass
        is kept as written, or refused with a 409 when the module has it.
        """
        body: dict = {"title": title, "kind": kind, "body_md": body_md,
                      "gated": gated}
        if slug is not None:
            body["slug"] = slug
        if parent_lesson_id is not None:
            body["parent_lesson_id"] = parent_lesson_id
        return self._json_call(
            "POST", f"/teacher/modules/{module_id}/lessons", json_body=body,
            ok=(200, 201), error_label="create_lesson",
        )

    def get_lesson(self, lesson_id: int) -> dict:
        """Full lesson detail incl. body — owner only.

        Mirrors `GET /api/teacher/lessons/{id}` (the authoring view; use
        `lesson(slug, module_slug, lesson_slug)` for the learner view with
        resolved directives).
        """
        return self._json_call(
            "GET", f"/teacher/lessons/{lesson_id}", error_label="get_lesson",
        )

    def update_lesson(self, lesson_id: int, **fields) -> dict:
        """Partial update of a lesson — owner only.

        Mirrors `PUT /api/teacher/lessons/{id}`. Accepts `title`, `slug`,
        `body_md`, `is_published`, `gated`, `estimated_minutes`. Only the
        fields you pass are changed. A new `slug` changes the lesson's URL
        and keeps its id, so the students' progress on it is kept.
        """
        allowed = {"title", "slug", "body_md", "is_published", "gated",
                   "estimated_minutes"}
        body = _filtered_fields(fields, allowed, "update_lesson")
        return self._json_call(
            "PUT", f"/teacher/lessons/{lesson_id}", json_body=body,
            error_label="update_lesson",
        )

    def delete_lesson(self, lesson_id: int) -> dict:
        """Delete a lesson and its descendants — owner only.

        Mirrors `DELETE /api/teacher/lessons/{id}`.
        """
        return self._json_call(
            "DELETE", f"/teacher/lessons/{lesson_id}", error_label="delete_lesson",
        )

    def reorder_lessons(self, module_id: int,
                        ordered_lesson_ids: list[int]) -> list:
        """Reorder a module's lessons.

        Mirrors `PUT /api/teacher/modules/{id}/lessons/reorder`.
        `ordered_lesson_ids` must be exactly this module's lesson ids in the
        desired display order (`position` becomes the index).
        """
        return self._json_call(
            "PUT", f"/teacher/modules/{module_id}/lessons/reorder",
            json_body={"ordered_ids": ordered_lesson_ids},
            error_label="reorder_lessons",
        )

    # ---- Course authoring: module assets (author scope) ---------------------
    #
    # A module has one asset store: the images and files its lessons link.
    # A lesson body links an asset by its path in the store, as a relative
    # link (`![fig](media/fig.png)`, `[data](data/week1.csv)`); the lesson's
    # `asset_base_url` is what such a link resolves against.

    def upload_module_asset(self, module_id: int, file_path: str, *,
                            path: str | None = None) -> dict:
        """Upload a file to a module's asset store; returns `{path, url}`.

        Mirrors `POST /api/teacher/modules/{id}/assets` (multipart `file`,
        form field `path`). `path` is where it is stored, relative to the
        store (`data/week1/x.csv`: letters, digits, `.`, `_` and `-` in each
        segment); left out, `media/<the file's name>`. A file already at that
        path is replaced. Any file type: write the returned `path` in a
        lesson body as `![alt](path)` for an image, or as `[label](path)` for
        anything else (notebook, dataset, handout), which downloads when a
        student clicks it. Refused (409) on a module managed by a bundle.
        """
        if not os.path.isfile(file_path):
            raise MLArenaError(f"File not found: {file_path}")
        with open(file_path, "rb") as fh:
            resp = self._request(
                "POST", self._url(f"/teacher/modules/{module_id}/assets"),
                headers=self._headers(),
                files={"file": (os.path.basename(file_path), fh)},
                data={"path": path} if path is not None else None,
                timeout=120,
            )
        self._handle_response(resp)
        if resp.status_code not in (200, 201):
            raise _failed(MLArenaError, "upload_module_asset", resp)
        return resp.json()

    def list_module_assets(self, module_id: int) -> list:
        """The files of a module's asset store: `[{path, url, size_bytes}]`.

        Mirrors `GET /api/teacher/modules/{id}/assets` (owner). Use it to
        find a file whose link is no longer in any body.
        """
        return self._json_call(
            "GET", f"/teacher/modules/{module_id}/assets",
            error_label="list_module_assets",
        )

    def delete_module_asset(self, module_id: int, path: str) -> dict:
        """Delete one file of a module's asset store, by its `path`.

        Mirrors `DELETE /api/teacher/modules/{id}/assets/{path}`.
        """
        return self._json_call(
            "DELETE", f"/teacher/modules/{module_id}/assets/{quote(path)}",
            error_label="delete_module_asset",
        )

    def download_module_asset(self, module_id: int, path: str,
                              dest_dir: str = ".") -> str:
        """Download one module asset to `dest_dir/<path>`; returns the
        written path.

        Mirrors `GET /api/academic_courses/assets/modules/{id}/{path}` — the
        *consumption* route, so this is the call a student (not just the
        author) makes to fetch a lab notebook or dataset a lesson links:
        `path` is the relative link as the lesson body writes it. A module
        whose lessons are gated requires enrollment; the bearer token carries
        that.
        """
        parts = path.split("/")
        if path.startswith("/") or "\\" in path or any(
                part in ("", ".", "..") for part in parts):
            raise MLArenaError(f"Not a module asset path: {path!r}")
        return self._download_course_file(
            f"/academic_courses/assets/modules/{module_id}/{quote(path)}",
            os.path.join(dest_dir, *parts),
            error_label="download_module_asset",
        )

    def preview_lesson(self, lesson_id: int, body_md: str | None = None) -> dict:
        """Server-validate a lesson body's `mlarena:` directives (strict mode).

        Mirrors `POST /api/teacher/lessons/{id}/preview`. Pass `body_md` to
        validate an unsaved buffer; omit it to validate the saved body. An
        unknown/unresolvable directive fails loud (400 → `MLArenaError`) so it's
        caught before publishing: the message ends with the server's
        `directive_error` (which block, and why), also on `err.body`; a
        challenge you may not see is refused as
        one that does not exist. Returns `{directives, warnings}`: the
        payloads as you would read them, and one warning per block and per
        course linking the lesson's module whose students would not see the
        block's challenge (it is private and that course does not assign it).
        """
        body = {"body_md": body_md} if body_md is not None else {}
        try:
            return self._json_call(
                "POST", f"/teacher/lessons/{lesson_id}/preview", json_body=body,
                error_label="preview_lesson",
            )
        except MLArenaError as err:
            detail = (err.body or {}).get("directive_error")
            if detail is None:
                raise
            # The reply's `error` is only "Invalid directive": the reason is
            # the `directive_error` key beside it.
            raise type(err)(
                f"{err}: {detail}", status_code=err.status_code, body=err.body,
            ) from None

    # ---- Course authoring: course composition (author scope) ----------------

    def update_course(self, course_id: int, *,
                      start_date: str | None | _Unset = _UNSET,
                      end_date: str | None | _Unset = _UNSET,
                      **fields) -> dict:
        """Update course meta and settings — owner, assistant or admin.

        Mirrors `PUT /api/teacher/course/{id}`. Accepts `name`, `code`,
        `description`, `visibility`, `instructor_name`, `start_date`, `end_date`
        (ISO date strings), and the settings `ask_student_identity` (bool:
        whether enrolling asks for the student number and email),
        `report_recipients` (list of extra email addresses the course report
        goes to, at most 20, one plain address per entry; the whole list is
        replaced) and `report_to_assistants` (bool). The two report settings
        are the owner's (or an admin's): an assistant sending either gets a
        403 and nothing is written. `name` and `instructor_name` are one line
        of text (a line break or a tab is a 400). With
        `ask_student_identity` off, the identity a student already has on
        their profile is still shown and exported; turning it on later does
        not ask the students who are already enrolled. Only the fields you
        pass are changed; a keyword passed as None is not sent, except the
        two dates: `start_date=None` / `end_date=None` send `null`, which
        clears that date, and leaving one out leaves it unchanged. Returns the
        course as `teacher_course(course_id)` does.
        """
        allowed = {"name", "code", "description", "visibility",
                   "instructor_name",
                   "ask_student_identity", "report_recipients",
                   "report_to_assistants"}
        body = _filtered_fields(fields, allowed, "update_course",
                                nullable={"start_date": start_date,
                                          "end_date": end_date})
        return self._json_call(
            "PUT", f"/teacher/course/{course_id}", json_body=body,
            error_label="update_course",
        )

    def set_course_cover(self, course_id: int, image_path: str) -> dict:
        """Upload (or replace) a course cover image.

        Mirrors `POST /api/teacher/course/{id}/cover`. Returns `{has_cover}` —
        the same flag every course payload carries. The image itself comes back
        from `course_cover(course_id)`; the stored path stays server-side.
        """
        return self._upload_course_file(
            "POST", f"/teacher/course/{course_id}/cover", image_path,
            error_label="set_course_cover",
        )

    def list_course_modules(self, course_id: int) -> list:
        """List a course's modules in order, each with its `lessons` (the
        lesson summaries of `get_module`, in order, each with
        `visible_to_students`) and its `challenges` (each `challenge_id`,
        `challenge`, `position`, `pass_threshold`, `metric`; `challenge` is
        the challenge's reference, `can_copy` included: see
        `challenge_overview`).

        Mirrors `GET /api/teacher/course/{id}/modules`. A row's one id is
        `module_id`: the id `reorder_modules`, `update_course_challenge` and
        every module method (`get_module`, `create_lesson`, …) take. Its
        course's staff read it (owner, assistant, admin).
        """
        return self._json_call(
            "GET", f"/teacher/course/{course_id}/modules",
            error_label="list_course_modules",
        )

    def create_course_module(self, course_id: int, title: str, *,
                             slug: str | None = None,
                             summary: str | None = None,
                             icon: str | None = None,
                             is_published: bool | None = None) -> dict:
        """Create a module at the end of a course: a section of that course.

        Mirrors `POST /api/teacher/course/{id}/modules/create`. Its slug is
        one no module of the course uses: a `slug` you pass that is taken is
        a 409 (a slug that is not one as the platform writes it — a leading,
        trailing or doubled hyphen, over 100 characters — a 400); without one
        it is derived from the title and suffixed until free.
        `is_published=False` creates it as a draft, hidden from students.
        Returns the course's new module row (the `list_course_modules`
        shape). The course's staff (owner, assistant, admin).
        """
        body: dict = {"title": title}
        for key, value in (("slug", slug), ("summary", summary), ("icon", icon),
                           ("is_published", is_published)):
            if value is not None:
                body[key] = value
        return self._json_call(
            "POST", f"/teacher/course/{course_id}/modules/create", json_body=body,
            ok=(201,), error_label="create_course_module",
        )

    def copy_module_into_course(self, course_id: int, module_id: int) -> dict:
        """Copy a module of another course of yours into this one.

        Mirrors `POST /api/teacher/course/{id}/modules/copy`: "Copy a module
        from another course". `module_id` is a module of a course you are on
        the staff of (404 otherwise). The copy goes at the end of this
        course, with a slug no module of it uses, and takes the module's
        lessons, assets and challenges: each challenge becomes a draft copy
        of this course with an empty board, on its kind's default machine,
        keeping its pass bar, and every `mlarena:` directive of the copied
        lessons that names a copied challenge names its copy. Every copy
        starts with the same data and the same scoring files as its source:
        change the seed or the data before you start it. The source is
        untouched. Returns the course's new module row (the
        `list_course_modules` shape). The staff of this course; 409 when a
        challenge's files are being updated at this instant, 503 when no
        machine runs a challenge's kind, 403 with `.body["reason"] ==
        "terms_required"` when the module holds a challenge and you have not
        accepted the terms of use (`accept_terms`; a module of lessons only
        needs no terms). Nothing is copied then.
        """
        return self._json_call(
            "POST", f"/teacher/course/{course_id}/modules/copy",
            json_body={"module_id": module_id},
            ok=(201,), error_label="copy_module_into_course",
        )

    def list_copyable_modules(self, course_id: int) -> list:
        """The modules `copy_module_into_course(course_id, …)` is offered:
        those of the other courses you are the owner or an assistant of
        (archived ones included), by course name then module order.

        Mirrors `GET /api/teacher/course/{id}/copyable-modules`. Each row is
        `{module_id, title, summary, lesson_count, challenge_count,
        course: {id, name}}`.
        """
        return self._json_call(
            "GET", f"/teacher/course/{course_id}/copyable-modules",
            error_label="list_copyable_modules",
        )

    def add_course_challenge(self, course_id: int, name: str | None = None, *,
                             kernel_version: str | None = None,
                             copy_from_challenge_id: int | None = None,
                             module_id: int | None = None,
                             pass_threshold: float | None | _Unset = _UNSET) -> dict:
        """Create a challenge in a course, or copy one into it.

        Mirrors `POST /api/teacher/course/{id}/challenges` — the console's
        Add › Challenge. A challenge belongs to one course: it is built
        there or copied there, owned by the course's owner (whoever of its
        staff calls), and a draft.

        - **Build**: `name` (unique in the course) and `kernel_version`,
          any kind of `available_kinds()`: `"file_v1"`, `"flex_v1"`,
          `"chat_v1"` or `"exercise_v1"` (an exercise set). Every member of
          the course's staff builds every kind and writes its code and
          files.
        - **Copy**: `copy_from_challenge_id`, a challenge you may copy: a
          started, stopped or ended challenge of an example course
          (`course_catalog(example=True)`), one you are staff of, any for an
          admin. That rule is served as `can_copy` on the challenge
          (`challenge(id)["can_copy"]`) and on every reference to it in a
          course, module or lesson payload; where it is false the copy is a
          404. It takes the source's kind, evaluation, config, env folder,
          benchmark, datasets, description and tags; not its submissions,
          runs, teams, assistants or dates; the data-feed keys only from an
          example course's challenge, the chat LLM key only when you have
          the owner's rights on the source. `name` defaults to the source's,
          suffixed ` (2)`, ` (3)`… when the course has it. Every copy starts
          with the same data and the same scoring files as its source:
          change the seed or the data before you start it.

        `module_id` places it at the end of that module of the course (404
        for another course's module; a bundle-managed module takes it too); without
        it a new module titled with the challenge's name is added at the end
        of the course. `pass_threshold` left out is 70 for an exercise set
        and no bar otherwise; `None` sends no bar (null); an exercise set's
        is 0 to 100. The challenge runs on its kind's default machine.

        Returns `{message, challenge_id, name, slug, course_id, module_id,
        module_created, position, pass_threshold, copied_from_challenge_id,
        machine_name}` (201). 409 when the course already has a challenge of
        that `name`, 503 when no machine runs the kind: nothing is created
        then. The course's staff (owner, assistant, admin).

        A build and a copy alike need the terms of use accepted, once per
        account (`accept_terms`): until then the call is a
        `PermissionDeniedError` (403) with `.body["reason"] ==
        "terms_required"`, whose message names the page of the terms and
        `accept_terms()`; nothing is created. The caller accepts for
        themselves: an assistant who creates a challenge has accepted,
        whether or not the course's owner has. No limit on the drafts of a
        course; what is limited is what is started (`start_challenge`).

        Before 4.23.0 this method linked an existing challenge
        (`add_course_challenge(course_id, challenge_id)`): that is gone, and
        a challenge id in place of `name` raises `TypeError`.
        """
        if not isinstance(name, (str, type(None))):
            raise TypeError(
                "add_course_challenge no longer links an existing challenge: "
                "a challenge now belongs to its course. Build one "
                "(name=, kernel_version=) or copy one (copy_from_challenge_id=)"
            )
        if (kernel_version is None) == (copy_from_challenge_id is None):
            raise TypeError(
                "add_course_challenge needs exactly one of kernel_version= "
                "(a new challenge) and copy_from_challenge_id= (a copy)"
            )
        if kernel_version is not None and name is None:
            raise TypeError("add_course_challenge needs a name for a new challenge")
        body: dict = {}
        for key, value in (("name", name), ("kernel_version", kernel_version),
                           ("copy_from_challenge_id", copy_from_challenge_id),
                           ("module_id", module_id)):
            if value is not None:
                body[key] = value
        if pass_threshold is not _UNSET:
            body["pass_threshold"] = pass_threshold
        return self._json_call(
            "POST", f"/teacher/course/{course_id}/challenges", json_body=body,
            ok=(201,), error_label="add_course_challenge", timeout=120,
        )

    def create_challenge_for_class(self, name: str, kernel_version: str | None = None, *,
                                   copy_from_challenge_id: int | None = None,
                                   end_date: str | None = None,
                                   pass_threshold: float | None | _Unset = _UNSET) -> dict:
        """A challenge for my class: a new private course named after the
        challenge, and the challenge in it.

        The console's New › "Challenge for my class", composed client-side
        from two public routes (no route of its own): `create_course(name,
        end_date=…)` (private; `end_date` optional, `YYYY-MM-DD`), then
        `add_course_challenge` in it, built (`kernel_version`) or copied
        (`copy_from_challenge_id`, e.g. a challenge of an example course).
        A course holding one challenge and no lesson is shown as that
        challenge: one page for students, one workspace for you.

        Creating a challenge needs the terms of use accepted
        (`accept_terms`, once per account): `me()["terms_accepted_at_ts"]`
        is read first, and an account that has not accepted is refused
        before the course is created, with the error the challenge route
        gives (`PermissionDeniedError`, 403, `.body["reason"] ==
        "terms_required"`). Nothing is created then.

        If the second call is refused for another cause (a name the course
        cannot take, no machine for the kind), the course stays (empty: add
        a challenge to it, or `delete_course` it) and the refusal is raised
        with a message naming the course, its id in `err.body["course_id"]`
        when the server's body had none.

        Returns `{"course": <the create_course answer>, "challenge": <the
        add_course_challenge answer>}`: share `course["join_code"]` (or the
        `/enroll/<join_code>` link) with students.
        """
        self._refuse_without_terms()
        course = self.create_course(name, end_date=end_date)
        try:
            challenge = self.add_course_challenge(
                course["id"], name, kernel_version=kernel_version,
                copy_from_challenge_id=copy_from_challenge_id,
                pass_threshold=pass_threshold,
            )
        except MLArenaError as err:
            raise _course_left_behind(err, course) from None
        return {"course": course, "challenge": challenge}

    def use_example_challenge(self, challenge_id: int, course_id: int | None = None, *,
                              name: str | None = None,
                              module_id: int | None = None) -> dict:
        """Use a challenge of an example course in a course of mine: a copy
        of it, yours to change.

        The console's "Use in my course", composed client-side from public
        routes. `challenge_id` is a challenge of an example course
        (`course_catalog(example=True)`; any challenge you may copy works).

        - `course_id` given: `add_course_challenge(course_id,
          copy_from_challenge_id=challenge_id, name=…, module_id=…)`.
        - `course_id` None: a new private course named `name` or, without
          one, after the challenge (`challenge(challenge_id)["name"]`), and
          the copy in it, as `create_challenge_for_class` does (the terms of
          use are checked before the course is created; if the copy is
          refused for another cause the empty course stays and the error
          says so).

        A copy is a challenge you create: it needs the terms of use accepted
        (`accept_terms`; 403 `.body["reason"] == "terms_required"`
        otherwise, and nothing is created). `challenge(challenge_id)
        ["can_copy"]` says whether the challenge is yours to copy (404
        where it is false).

        Every copy starts with the same data and the same scoring files as
        its source: change the seed or the data before you start a
        challenge. The copy is yours to change like a challenge you built:
        its files, its scoring, its datasets, its benchmark. A copy
        of a chat challenge has no LLM key: set yours before Start
        (`update_chat_settings`). The copy's compute is paid from your
        credits (`credits()`).

        Returns `{"course": <the course>, "challenge": <the
        add_course_challenge answer>}`; `course` is `teacher_course(id)`
        for an existing course, the `create_course` answer for a new one.
        """
        if course_id is not None:
            challenge = self.add_course_challenge(
                course_id, name, copy_from_challenge_id=challenge_id,
                module_id=module_id,
            )
            return {"course": self.teacher_course(course_id), "challenge": challenge}
        if module_id is not None:
            raise TypeError("use_example_challenge: module_id= needs course_id=")
        if name is None:
            name = self.challenge(challenge_id)["name"]
        return self.create_challenge_for_class(name, copy_from_challenge_id=challenge_id)

    def start_from_example_course(self, course: int | str, name: str | None = None) -> dict:
        """Start from an example course: your own private copy of its
        published modules, lessons and challenges.

        The console's "Start from this course". `course` is the example
        course's id or its slug (the `/courses/<slug>` of its page; a slug
        is resolved with `course(slug)`). Then `duplicate_course(id, name)`,
        which any author may call on an example course: the copy counts in
        your `max_courses_limit`, its challenges are drafts with empty
        boards on their kind's default machine, paid from your credits, and
        every `mlarena:` directive of its lessons that names a copied
        challenge names the copy. Every copy starts with the same data and
        the same scoring files as its source: change the seed or the data
        before you start a challenge. Returns the new course
        (`teacher_course` shape).

        A course that holds a challenge you would copy needs the terms of
        use accepted (`accept_terms`): the one route refuses otherwise (403,
        `.body["reason"] == "terms_required"`) and copies nothing, so no
        course is left behind. A course of lessons only needs no terms.
        """
        course_id = self.course(course)["id"] if isinstance(course, str) else course
        return self.duplicate_course(course_id, name)

    def reorder_modules(self, course_id: int,
                        ordered_module_ids: list[int]) -> list:
        """Reorder a course's linked modules.

        Mirrors `PUT /api/teacher/course/{id}/modules/reorder`.
        `ordered_module_ids` must be exactly this course's linked module ids.
        """
        return self._json_call(
            "PUT", f"/teacher/course/{course_id}/modules/reorder",
            json_body={"ordered_ids": ordered_module_ids},
            error_label="reorder_modules",
        )

    def course_progress(self, course_id: int) -> dict:
        """Teacher follow dashboard: per enrolled student × (content %, per-challenge
        best result).

        Mirrors `GET /api/teacher/course/{id}/progress`. Requires teacher/TA/admin.
        Challenge metas and cells carry `metric`, the ranking MetricSpec,
        next to `pass_threshold` (see `my_progress`). Each challenge meta
        carries `is_published`: False for a challenge of an entry hidden from
        students, which only the course's staff follow. A cell's
        `best_submission_id` is served only when the caller may open that
        submission (`submission()`): it is None, next to a `value`, on an
        entry hidden from students (unless the caller is an admin, staff of
        the challenge, or the submission is public) and for a team's line
        held by a teammate the caller does not teach.

        A cell also carries `status` (`not_submitted` | `failed` | `scoring` |
        `scored`: nothing handed in, a failed upload or deploy, a deploy or a
        first run on its way, a scored line), `team_id` and `team_name` (the
        student's team on that challenge, else None). A student row carries
        `student_email`, `enrolled_at_ts` and `passed_count` (pass bars
        cleared, out of the payload's `pass_bar_count`).
        `export_course_grades_csv` is this payload as a CSV.

        An exercise set (the in-lesson exercises) is an entry like the
        others: its `value` is the % of points solved. Its cells also carry
        `exercises`: `{solved, attempted, total}` counted over the set's
        active exercises (the student's, or their team's; zeros before the
        first Check), and its challenge meta `exercise_count`. Both are None
        for every other kind of challenge.
        """
        return self._json_call(
            "GET", f"/teacher/course/{course_id}/progress",
            error_label="course_progress",
        )

    def export_course_grades_csv(self, course_id: int, dest_dir: str = ".") -> str:
        """Download the course-wide grade sheet as CSV; returns the written path.

        Mirrors `GET /api/teacher/course/{id}/progress.csv` — `course_progress`
        as a file: one row per enrolled student; lessons completed, then one
        column per challenge or exercise set of the course (entries hidden
        from students included, headed `<name> (hidden)`) holding the score,
        or `Not submitted` / `Scoring` / `Failed`; a `passed` column per
        challenge with a pass bar, `solved` and `attempted` columns
        (`n/total` exercises) per exercise set, a `team` column per challenge
        with teams, and `Passed` (`cleared/total` pass bars). Writes
        `grades_course_<id>.csv`: UTF-8 with a byte-order mark, formula-like
        text cells prefixed with an apostrophe (as `export_course_csv`).
        Owner, assistant or admin.
        """
        return self._download_course_file(
            f"/teacher/course/{course_id}/progress.csv",
            os.path.join(dest_dir, f"grades_course_{course_id}.csv"),
            error_label="export_course_grades_csv",
        )

    # ---- Course administration: roster, assistants, exports ----------------

    def teacher_courses(self):
        """Deprecated: use `workspace()["courses"]`.

        The courses you own or assist on (archived ones included), built
        from `workspace()`: each row keeps the columns the old list had that
        the workspace serves (`id`, `name`, `code`, `slug`, `visibility`,
        `join_code`, `start_date`, `end_date`, `archived_at_ts`, `role`). The
        list route this method called (`GET /api/teacher/courses`) is gone,
        and with it `description`, `has_cover`, `instructor_name` and
        `is_enrolled` (read them from `teacher_course(id)`). An admin gets
        their own courses; everyone's is the console's. Works with the
        `author` key.
        Returns a DataFrame if pandas is installed, else a list of dicts.
        """
        warnings.warn(
            "MLArenaClient.teacher_courses() is deprecated, "
            "use .workspace()[\"courses\"] instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return _to_dataframe([
            {key: row[key] for key in _TEACHER_COURSE_KEYS}
            for row in self.workspace()["courses"]
        ])

    def teacher_course(self, course_id: int) -> dict:
        """One course you own or assist on, as its workspace reads it.

        Mirrors `GET /api/teacher/course/{id}`. The course row (`id`, `name`,
        `code`, `slug`, `description`, `visibility`, `has_cover`,
        `instructor_name`, `join_code`, `start_date`, `end_date`,
        `archived_at_ts`, `enrollment_state`, `is_enrolled`,
        `single_challenge_id`: the course's one challenge when the course is
        shown as it, `example_listed_at_ts`: set while it is listed as an
        example) plus `role`
        (what you are to it: `owner`, `assistant`, or None for an admin who
        is neither), `can_edit` (true: you may open its workspace),
        `has_owner_rights` (whether you may do what only its owner may: the
        assistant roster, archive, delete, unlist from the examples — the
        owner and an admin), `may_list_as_example` (whether you may list it
        as an example, `list_course_as_example`: the publishing right with
        the owner's rights; that the course must be public and not archived
        is the route's 409, not this flag), `owner_username`,
        `created_at_ts`, the settings
        (`ask_student_identity`, `report_recipients`, `report_to_assistants`,
        `report_last_sent_at_ts`) and the counts `enrollment_count`,
        `assistant_count`, `module_count`, `challenge_count` (entries hidden
        from students included), `show_module_headings` (False for a course
        with a single module: its entries are then the course's, with no
        module heading) and `exercise_default_pass_threshold` (the pass bar
        `add_course_challenge` gives a new exercise set when you name none). It also carries what `workspace()` follows
        about the course, under the same names and from the same computation:
        `active_enrollment_count_7d`, `last_active_at_ts` and
        `enrolled_without_submission` (served while the course is not
        archived, ends within 7 days and assigns a challenge, else None).
        Owner, assistant or admin; 403 otherwise.
        """
        return self._json_call(
            "GET", f"/teacher/course/{course_id}", error_label="teacher_course",
        )

    def archive_course(self, course_id: int) -> dict:
        """Archive a course — its owner or an admin.

        Mirrors `POST /api/teacher/course/{id}/archive`. An archived course
        takes no enrollment (its join code answers 410), leaves the catalog and
        the listing of non-members, is skipped by the end-of-course report and
        no longer counts in your limit of active courses. Nothing is deleted:
        its students keep reading it and you keep following it. Returns the
        course (`archived_at_ts` set); 409 when it is already archived or is
        listed as an example (`unlist_course_example` it first).
        """
        return self._json_call(
            "POST", f"/teacher/course/{course_id}/archive",
            error_label="archive_course",
        )

    def unarchive_course(self, course_id: int) -> dict:
        """Take a course out of the archive — its owner or an admin.

        Mirrors `POST /api/teacher/course/{id}/unarchive`. The course counts
        in its owner's limit again: 403 when the owner has no place left, 409
        when the course is not archived. Returns the course.
        """
        return self._json_call(
            "POST", f"/teacher/course/{course_id}/unarchive",
            error_label="unarchive_course",
        )

    def duplicate_course(self, course_id: int, name: str | None = None) -> dict:
        """Copy a course into a new private course of yours: "Duplicate for
        next term" for its staff, "Start from this course" for any author on
        an example course (`start_from_example_course` takes its slug).

        Mirrors `POST /api/teacher/course/{id}/duplicate`. 403 for a course
        you may not copy (not on its staff, not an example) and at your
        `max_courses_limit`. The copy gets the settings (code, description,
        cover, enrollment settings; the instructor and the report settings
        when you hold the owner's rights on the source) and a copy of each
        module with its lessons, its assets and its challenges: each
        challenge becomes a draft of the new course with an empty board, on
        its kind's default machine, and the lessons' `mlarena:` directives
        that name a copied challenge name the copy. Staff copy the course as
        it is; anyone else gets what its students see (its published
        modules and lessons). Every copy starts with the same data and the
        same scoring files as its source: change the seed or the data before
        you start a challenge. Not copied: enrollments, progress, dates,
        assistants; the copy is `private` and has a fresh `join_code`.
        `name` defaults to the source's name followed by " (copy)". Returns
        the new course (the `teacher_course` shape). 409 when another course
        took the copy's address at the same moment (call it again), when the
        source's cover file is missing from storage or a challenge's files
        are being updated; 503 when no machine runs a challenge's kind; 403
        with `.body["reason"] == "terms_required"` when the course holds a
        challenge you would copy and you have not accepted the terms of use
        (`accept_terms`; a course without one needs no terms). Nothing is
        copied then, and no course is left behind.
        """
        body: dict = {}
        if name is not None:
            body["name"] = name
        return self._json_call(
            "POST", f"/teacher/course/{course_id}/duplicate", json_body=body,
            ok=(201,), error_label="duplicate_course", timeout=300,
        )

    def list_course_as_example(self, course_id: int) -> dict:
        """List a course as an example: a starting point offered to teachers,
        who copy it whole (`start_from_example_course`) or take one of its
        challenges into a course of theirs (`use_example_challenge`).

        Mirrors `POST /api/teacher/course/{id}/example`. It needs the
        publishing right (an admin, for now; 403 otherwise) and the owner's
        rights on the course; 409 when the course is not public, is archived
        or is listed already. Returns the course (`teacher_course` shape)
        with `example_listed_at_ts` set.

        What listing gives away: every copy takes the challenges' scoring
        files, their data, the exercise solutions and the data-feed keys,
        and every copy starts with the same data and the same scoring files
        (a teacher should change the seed or the data before starting a
        challenge). List a course whose challenges a full copy does not
        spoil: scored on data that does not exist yet (a live feed),
        played against other participants (multiplayer), or generated from
        a configuration the teacher changes (a seed, a base to extend); not
        a challenge scored against a fixed file of answers. Whoever lists is
        responsible for the choice. While listed, making the course
        private, archiving it and deleting it are refused (409).
        """
        return self._json_call(
            "POST", f"/teacher/course/{course_id}/example",
            error_label="list_course_as_example",
        )

    def unlist_course_example(self, course_id: int) -> dict:
        """Take a course out of the examples (the owner's rights are enough).

        Mirrors `DELETE /api/teacher/course/{id}/example`. The copies made
        from it are untouched: each was its maker's from the start, with the
        data and the scoring files it took (every copy starts with the same
        data and the same scoring files as its source: its teacher should
        change the seed or the data before starting a challenge). 409 when
        the course is not listed. Returns the course with `example_listed_at_ts` None.
        """
        return self._json_call(
            "DELETE", f"/teacher/course/{course_id}/example",
            error_label="unlist_course_example",
        )

    def delete_course(self, course_id: int) -> dict:
        """Delete a course with everything in it — its owner or an admin.

        Mirrors `DELETE /api/teacher/course/{id}`. A course goes with its
        modules, their lessons and assets, and its challenges, so it is
        deleted only when nobody is enrolled (a student, or yourself:
        `remove_student` them first), it is not listed as an example, and
        every one of its challenges may be deleted (a draft with no
        submission, no team and no run on its way). Otherwise 409, with the
        plan's numbers in `.body`: `archive_course` it instead. Returns what
        `course_deletion` announced, with `storage_left` (what could not be
        removed from storage although the rows are gone; `[]` when all of it
        went).
        """
        return self._json_call(
            "DELETE", f"/teacher/course/{course_id}", error_label="delete_course",
        )

    def course_deletion(self, course_id: int) -> dict:
        """What `delete_course` would do, without doing it — owner or admin.

        Mirrors `GET /api/teacher/course/{id}/deletion` (the console's
        confirmation dialog). Returns `can_delete`, `enrollment_count`,
        `example_listed_at_ts`, `deleted_modules` (`[{id, title}]`, every
        module), `deleted_challenges` (`[{id, name}]`, every challenge, in
        course order), `blocking_challenges` (`[{id, name, reason}]`, those
        that may not be deleted now) and `storage_left` (`[]` here).
        """
        return self._json_call(
            "GET", f"/teacher/course/{course_id}/deletion",
            error_label="course_deletion",
        )

    def send_course_report(self, course_id: int) -> dict:
        """Mail the course report now — owner, assistant or admin.

        Mirrors `POST /api/teacher/course/{id}/report/send` (the console's
        "Send now"). The report is the end-of-course mail: two leaderboard
        CSVs (by participant, all submissions) per challenge of the course,
        entries hidden from students included. It goes to the owner, to the
        assistants while `report_to_assistants`, and to `report_recipients`
        (the owner sets both with `update_course`), one message per
        recipient: none of them sees the others. Returns `{recipients,
        challenge_count, attachment_count, report_last_sent_at_ts}`. 400 when
        the course has no challenge or no enrolled student; 429 when the
        report went out less than 10 minutes ago (`retry_after_sec` in the
        error body). The mail the platform sends by itself on the course's
        `end_date` needs that date; this call does not.
        """
        return self._json_call(
            "POST", f"/teacher/course/{course_id}/report/send",
            error_label="send_course_report",
        )

    def course_students(self, course_id: int,
                        challenge_id: int | None = None) -> dict:
        """Deprecated: use `course_progress(course_id)`.

        The course roster, with team membership for one attached challenge,
        built from `course_progress()` in the shape this method always
        returned: `{challenges, selected_challenge_id, students}`. The route
        it called (`GET /api/teacher/students/{course_id}`) is gone: the
        grade sheet is the roster. `challenges` are the course's challenge
        references; each student carries `user_id`, `username`,
        `student_number`, `student_email`, `enrolled_at_ts`, and `team_id` /
        `team_name` on the challenge named by `challenge_id` (None without
        one; a challenge not attached to the course raises). Students are
        ordered by enrollment date.
        """
        warnings.warn(
            "MLArenaClient.course_students() is deprecated, "
            "use .course_progress() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        progress = self.course_progress(course_id)
        challenges = [column["challenge"] for column in progress["challenges"]]
        index = None
        if challenge_id is not None:
            ids = [challenge["id"] for challenge in challenges]
            if challenge_id not in ids:
                raise MLArenaError(
                    "course_students failed: challenge_id is not attached to this course")
            index = ids.index(challenge_id)

        def team(student: dict, key: str):
            return None if index is None else student["challenges"][index][key]

        return {
            "challenges": challenges,
            "selected_challenge_id": challenge_id,
            "students": [
                {
                    "user_id": student["user_id"],
                    "username": student["username"],
                    "student_number": student["student_number"],
                    "student_email": student["student_email"],
                    "team_id": team(student, "team_id"),
                    "team_name": team(student, "team_name"),
                    "enrolled_at_ts": student["enrolled_at_ts"],
                }
                for student in sorted(
                    progress["students"], key=lambda student: student["enrolled_at_ts"] or "")
            ],
        }

    def remove_student(self, course_id: int, user_id: int) -> dict:
        """Remove a student's enrollment from a course.

        Mirrors `DELETE /api/teacher/student/{course_id}/{user_id}`. Only the
        enrollment row is deleted: the student leaves the roster, the progress
        table, the exports and the course view of the leaderboards. Their
        account, student number and email, submissions, teams and lesson
        progress are kept, and they can enroll again with the join code (their
        completed lessons then count again).
        """
        return self._json_call(
            "DELETE", f"/teacher/student/{course_id}/{user_id}",
            error_label="remove_student",
        )

    def course_assistants(self, course_id: int):
        """The assistants on a course.

        Mirrors `GET /api/teacher/course/{id}/assistants`. Each row carries
        `user_id`, `username` and `created_at_ts` (no email). Everyone who
        manages the course (its owner, an assistant, an admin) may read the
        list; only the owner (or an admin) may change it. The owner is not a
        row: `teacher_course(id)["owner_username"]` names them. Returns a DataFrame if
        pandas is installed, else a list of dicts.
        """
        return _to_dataframe(self._json_call(
            "GET", f"/teacher/course/{course_id}/assistants",
            error_label="course_assistants",
        ))

    def add_course_assistant(self, course_id: int, username: str) -> dict:
        """Add an assistant by username (course owner or admin).

        Mirrors `POST /api/teacher/course/{id}/assistants`. A TA gets the same
        teacher view of the course, but cannot manage the assistant list.
        """
        return self._json_call(
            "POST", f"/teacher/course/{course_id}/assistants",
            json_body={"username": username}, ok=(200, 201),
            error_label="add_course_assistant",
        )

    def remove_course_assistant(self, course_id: int, user_id: int) -> dict:
        """Remove an assistant (course owner or admin).

        Mirrors `DELETE /api/teacher/course/{id}/assistants/{user_id}`.
        """
        return self._json_call(
            "DELETE", f"/teacher/course/{course_id}/assistants/{user_id}",
            error_label="remove_course_assistant",
        )

    def search_course_assistant_candidates(self, course_id: int, q: str) -> list:
        """Accounts you may add as assistants of this course, by username.

        Mirrors `GET /api/teacher/course/{id}/assistants/candidates`. `q` (two characters at least) is
        looked for anywhere in the username; at most six rows, those starting
        with `q` first, each `user_id`, `username`, `avatar_key`. The owner
        and the assistants already there are left out. Course owner or admin.
        """
        return self._json_call(
            "GET", f"/teacher/course/{course_id}/assistants/candidates", params={"q": q},
            error_label="search_course_assistant_candidates",
        )

    def course_assistant_invitations(self, course_id: int) -> list:
        """The assistant invitations mailed and not answered yet, newest first.

        Mirrors `GET /api/teacher/course/{id}/assistants/invitations`. Each row carries `id`, `email`,
        `sender_username`, `created_at_ts`, `expires_at_ts` and `is_expired`.
        Course owner or admin.
        """
        return self._json_call(
            "GET", f"/teacher/course/{course_id}/assistants/invitations",
            error_label="course_assistant_invitations",
        )

    def invite_course_assistant(self, course_id: int, email: str) -> dict:
        """Mail `email` a link that makes whoever opens it an assistant of
        this course, whether or not the address has an account yet.

        Mirrors `POST /api/teacher/course/{id}/assistants/invitations`. The link works once, for 14 days.
        Inviting an address that is already waiting renews its invitation
        (same `id`): a new link is mailed and the earlier one stops working.
        Returns the invitation row. Course owner or admin.
        """
        return self._json_call(
            "POST", f"/teacher/course/{course_id}/assistants/invitations",
            json_body={"email": email}, ok=(201,),
            error_label="invite_course_assistant",
        )

    def cancel_course_assistant_invitation(self, course_id: int,
                                           invitation_id: int) -> dict:
        """Cancel a pending assistant invitation: its link stops working.

        Mirrors `DELETE /api/teacher/course/{id}/assistants/invitations/{invitation_id}`. Course owner or admin.
        """
        return self._json_call(
            "DELETE", f"/teacher/course/{course_id}/assistants/invitations/{invitation_id}",
            error_label="cancel_course_assistant_invitation",
        )

    def export_course_csv(self, course_id: int, challenge_id: int, *,
                          by_participant: bool = False,
                          dest_dir: str = ".") -> str:
        """Download the leaderboard of one of the course's challenges as
        CSV; returns the written path.

        Mirrors `GET /api/teacher/export-csv/{course_id}` — the console's
        Students-tab export. `challenge_id` (required) is one of the course's
        challenges; the server answers 400 for any other.
        `by_participant=True` writes one row per enrolled student (their team's
        best submission) instead of one row per submission. The file is UTF-8
        with a byte-order mark (read it with `encoding="utf-8-sig"`, or
        `pandas.read_csv`, which skips the mark); a text cell that a
        spreadsheet would run as a formula (it starts with `=`, `+`, `-`,
        `@`) is written with a leading apostrophe; a student without a
        submission has rank `N/A`.
        """
        params: dict = {
            "by_participant": str(bool(by_participant)).lower(),
            "challenge_id": challenge_id,
        }
        return self._download_course_file(
            f"/teacher/export-csv/{course_id}",
            os.path.join(
                dest_dir,
                f"leaderboard_course_{course_id}_challenge_{challenge_id}.csv",
            ),
            params=params, error_label="export_course_csv",
        )

    def course_cover(self, course_id: int, dest_dir: str = ".") -> str:
        """Download a course's cover image; returns the written path.

        Mirrors `GET /api/academic_courses/assets/courses/{id}/cover` — the same
        URL the console builds from the course id. Course payloads say only
        whether there is one (`has_cover`), never where it is stored. 404s when
        the course has no cover.
        """
        return self._download_course_file(
            f"/academic_courses/assets/courses/{course_id}/cover",
            os.path.join(dest_dir, f"course_{course_id}_cover"),
            error_label="course_cover",
        )

    def _download_course_file(self, path: str, out_path: str, *,
                              params: dict | None = None,
                              error_label: str = "download") -> str:
        """Stream a course route's file body to ``out_path``; returns the
        written path."""
        directory = os.path.dirname(out_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        resp = self._request(
            "GET", self._url(path), headers=self._headers(),
            params=params, timeout=300, stream=True,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, error_label, resp)
        with open(out_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    fh.write(chunk)
        return out_path

    # ---- High-level course helpers (client-side compositions) ---------------

    def push_course(self, path: str, *, slug: str | None = None,
                    subpath: str | None = None,
                    dry_run: bool = False, prune: bool = False) -> dict:
        """Create or update a whole course from a directory, in one step
        (author scope).

        Mirrors `PUT /api/teacher/courses/{slug}/bundle` (multipart field
        `archive`, query `dry_run` / `prune`). `path` is the course directory
        (`course.yaml`, the lessons' markdown, their images and files),
        zipped by `mlarena.bundle.make_bundle`, or an existing `.zip` /
        `.tar.gz` of it. `slug` is the course's; by default it is read from
        the manifest's `course.slug`. `subpath` names the folder inside the
        directory `path` that holds the course (one repository, several
        courses): only that folder is bundled, as `push_challenge` does. The
        course route takes no sub-folder, so an archive is pushed whole and
        `subpath` with one raises.

        Directory layout::

            my-course/
              course.yaml            # or course.json: the manifest
              cover.png              # optional, named by the manifest
              foundations/
                what-is-rl.md        # a lesson body, named by the manifest
                images/loop.png      # linked from it as ![](images/loop.png)

        A relative path is relative to the folder of the file that writes
        it, as on GitHub: a path of the manifest (`file`, `cover`) from the
        directory's root, a link of a lesson body from the lesson file's
        folder (an inline `body_md` from the root). A link with `..` is
        refused: a lesson's files sit in its folder or below it.

        Manifest (`course.yaml`)::

            course:
              name: "Intro to RL"
              slug: intro-to-rl
              code: "RL101"          # optional
              description: "..."     # optional markdown
              visibility: public     # private | public
              instructor_name: "Prof X"
              start_date: 2026-01-01
              end_date: 2026-06-01
              cover: cover.png       # optional, from the directory's root
            modules:
              - title: "Foundations"
                slug: foundations
                summary: "..."       # optional
                icon: book           # optional
                is_published: true   # optional (default true): false = hidden from students
                challenges:          # optional: challenges of this course, placed here
                  - slug: cartpole-v1       # the challenge's URL slug (or challenge_id:)
                    pass_threshold: 195.0   # optional: the score that validates it; left
                                            # out keeps the bar, null clears it
                lessons:
                  - title: "What is RL?"
                    slug: what-is-rl
                    kind: lesson           # lesson | exercise
                    file: foundations/what-is-rl.md   # or inline body_md:
                    gated: false           # optional
                    is_published: true     # optional (default false: a draft)
                    estimated_minutes: 15  # optional
                    renamed_from: intro    # optional: the lesson's previous slug
                    children: []           # optional: sub-lessons, one level

        The server applies the bundle; this method only uploads it. It is
        idempotent: the course, each module and each lesson are matched by
        slug (required on all three; `renamed_from` renames a module or a
        lesson in place), so pushing twice changes nothing the second time.
        The manifest is declarative for what it lists: a key left out takes
        its default, except in the `course` block. A `challenges` entry
        places a challenge of this course in its module, in the manifest's
        order (a challenge of another course, or one with no course, is refused:
        a challenge belongs to one course and is created there, with
        `add_course_challenge`); a challenge no entry names is left where it
        is, and `challenges: []` is the same as leaving the key out. A
        module's `visibility` and an entry's `label` are no longer read:
        warned and ignored. `course.visibility: unlisted` is refused. A
        lesson that left the manifest is unpublished, not deleted (the students' progress on it is
        kept) and reported as orphaned; `prune=True` deletes it. Unknown
        keys, a missing file or an invalid lesson directive refuse the whole
        bundle: `MLArenaError` with the server's message, nothing written.
        `dry_run=True` returns the same plan and writes nothing.

        A module written by a bundle is managed by it: the console's editor
        and the module / lesson write methods answer 409 until
        `detach_module`. Relative links in a lesson are kept as written; the
        files they name become the module's assets, keyed by the link (the
        lessons of a module share one store: two lessons that link one key
        to different files are refused).

        Returns the server's plan: `dry_run`, `course` (`id`, `slug`,
        `action`, `changes`), `modules` (each `slug`, `id`, `action` —
        `created` | `updated` | `unchanged` | `adopted` —, `changes`,
        `renamed_from`, `lessons` (each with `action` and `changes`, the
        fields an `updated` lesson changed), `challenges` (each entry's
        `name`, `id`, `action` — `updated` | `unchanged` — and `changes`
        among `module_id`, `position`, `pass_threshold`),
        `orphaned_lessons`, `pruned_lessons`, `assets`: `written` /
        `removed` and their store keys `written_paths` / `removed_paths`),
        `unmanaged_modules`, `unmanaged_challenges` (the names of the
        course's challenges no entry names, in course order),
        `source_revision`, the course's `url`, and `warnings`: first one
        sentence per manifest key that is no longer used, then one per
        `mlarena:` block whose challenge the students of this course will
        not see, naming its module and lesson; `[]` when there is nothing to
        say. A push with warnings succeeds. Manifests written before
        SDK 2.0 (`competitions:` / `competition_id:`) are still read by the
        server.
        """
        if os.path.isdir(path):
            path = bundle_root(path, subpath)
        elif subpath is not None:
            raise MLArenaError(
                "push_course takes subpath= only with a directory: an archive "
                "is pushed whole, with the manifest at its root"
            )
        if slug is None:
            if not os.path.isdir(path):
                raise MLArenaError(
                    "push_course needs slug= when path is an archive (the "
                    "manifest inside it is read by the server only)"
                )
            slug = _manifest_course_slug(os.path.abspath(path))
        filename, archive = read_bundle(path)
        params: dict = {}
        if dry_run:
            params["dry_run"] = "true"
        if prune:
            params["prune"] = "true"
        resp = self._request(
            "PUT", self._url(f"/teacher/courses/{quote(slug, safe='')}/bundle"),
            headers=self._headers(),
            params=params or None,
            files={"archive": (filename, archive)},
            timeout=300,
        )
        self._handle_response(resp)
        if resp.status_code not in (200, 201):
            raise _failed(MLArenaError, "push_course", resp)
        return resp.json()

    def import_course_from_github(self, slug: str, repository: str, *,
                                  ref: str | None = None,
                                  path: str | None = None,
                                  dry_run: bool = False,
                                  prune: bool = False) -> dict:
        """Create or update a course from a public GitHub repository
        (author scope).

        Mirrors `POST /api/teacher/courses/{slug}/import-github`
        `{"repository", "ref", "path", "dry_run", "prune"}`. `repository` is
        `owner/name` or the repository's URL; `ref` a branch, tag or commit
        (default: the default branch); `path` the folder of the repository
        that holds `course.yaml`. The server downloads that commit and
        applies it as `push_course` does, with the same plan. A private
        repository is pushed from its own GitHub Actions workflow instead
        (`mlarena course push`).
        """
        return self._json_call(
            "POST", f"/teacher/courses/{quote(slug, safe='')}/import-github",
            json_body={"repository": repository, "ref": ref, "path": path,
                       "dry_run": dry_run, "prune": prune},
            ok=(200, 201), error_label="import_course_from_github",
            timeout=300,
        )

    def author_course_from_dir(self, path: str) -> dict:
        """Deprecated alias of `push_course(path)`.

        It uploads the directory as a course bundle and returns the server's
        plan. The course is matched by the manifest's `course.slug`, which is
        therefore required; the manifest's `course_id` / `module_id` keys are
        not part of the bundle format.
        """
        warnings.warn(
            "author_course_from_dir() is deprecated, use push_course()",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.push_course(path)

    def export_course_to_dir(self, slug: str, path: str, *,
                             overwrite: bool = False) -> dict:
        """Write a course as a folder that `push_course` (or a GitHub
        repository's workflow) pushes back unchanged; returns its manifest.

        Mirrors `GET /api/academic_courses/{slug}/bundle`, the console's
        *Download as a folder (.zip)*: the server builds the zip and this
        method unpacks it into `path` (`mlarena.bundle.extract_bundle`). The
        folder holds `course.yaml`, each lesson body at
        `<module-slug>/<lesson-slug>.md` with the files it links beside it
        (a link is relative to the lesson file's folder, as on GitHub), and
        the cover. For a course you manage it also holds
        `.github/workflows/push.yml` (the course template's workflow: commit
        the folder to a GitHub repository and each push to `main` updates
        the course) and a `README.md`, which lists under "Fix before your
        first push" any link of a lesson to a file the course does not hold.

        `path` is created when missing. An existing folder must be empty, so
        an export never mixes with other files by accident; `overwrite=True`
        writes the course's files over the ones there and keeps every other
        file (to refresh a repository's checkout: its `.git` stays). The
        archive is checked before anything is written: an entry outside the
        folder (absolute, `..`), a symbolic link or a duplicate raises
        `MLArenaError` and writes nothing.

        Returns the parsed `course.yaml`, the dict `push_course` reads.
        Next step: `push_course(path, dry_run=True)` reports every lesson
        *unchanged*, and a module written in the Studio *adopted* (it moves
        to the repository; only the module's owner's push does that).
        """
        # Refused before the download: a folder that already holds files.
        check_extract_target(path, overwrite)
        resp = self._request(
            "GET", self._url(f"/academic_courses/{quote(slug, safe='')}/bundle"),
            headers=self._headers(), timeout=300,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "export_course_to_dir", resp)
        extract_bundle(resp.content, path, overwrite=overwrite)
        return _load_course_manifest(os.path.abspath(path))

    def __repr__(self) -> str:
        # mlk_<scope>_<lookup>_<secret> / mlo_<lookup>_<secret>: never the secret.
        head = self._token.split("_")
        masked = "_".join(head[:-1] + ["…"]) if len(head) in (3, 4) else "…"
        return f"MLArenaClient(base_url='{self._base_url}', token='{masked}')"


def _as_student(as_student: bool) -> dict | None:
    """The query string of a learner read: ``?as_student=true`` when asked,
    nothing otherwise."""
    return {"as_student": "true"} if as_student else None


def _filtered_fields(fields: dict, allowed: set, method_name: str, *,
                     nullable: dict | None = None) -> dict:
    """Validate a ``**fields`` partial-update kwargs dict against the route's
    editable set, failing loud (Fail-Fast) on an unknown field rather than
    silently dropping it, and requiring at least one field. A ``None`` in
    ``fields`` is not sent; ``nullable`` holds the keywords where ``None`` is
    a value (sent as null) and ``_UNSET`` means not passed."""
    nullable = nullable or {}
    unknown = set(fields) - allowed
    if unknown:
        raise MLArenaError(
            f"{method_name}: unknown field(s) {sorted(unknown)}. "
            f"Allowed: {sorted(allowed | set(nullable))}"
        )
    body = {k: v for k, v in fields.items() if v is not None}
    body.update({k: v for k, v in nullable.items() if v is not _UNSET})
    if not body:
        raise MLArenaError(
            f"{method_name} requires at least one field from "
            f"{sorted(allowed | set(nullable))}"
        )
    return body


def _load_course_manifest(base: str) -> dict:
    """Read ``course.yaml`` / ``course.yml`` or ``course.json`` from a course
    directory. Fails loud if none is found."""
    import json

    yaml_path = next(
        (os.path.join(base, name)
         for name in ("course.yaml", "course.yml")
         if os.path.isfile(os.path.join(base, name))),
        None,
    )
    json_path = os.path.join(base, "course.json")

    if yaml_path is not None:
        with open(yaml_path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    if os.path.isfile(json_path):
        with open(json_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    raise MLArenaError(
        f"No course manifest found in {base!r} (expected course.yaml or course.json)"
    )


def _manifest_course_slug(base: str) -> str:
    """`course.slug` of the manifest in a course directory: the course a
    bundle of it is pushed to."""
    course = _load_course_manifest(base).get("course") or {}
    slug = course.get("slug")
    if not slug:
        raise MLArenaError(
            f"The manifest in {base!r} has no course.slug: add it (the course "
            f"is matched by its slug), or pass slug="
        )
    return str(slug)


def _default_submission_name(agent, files) -> str:
    if agent is not None:
        return getattr(agent, "__name__", "submission")
    if files:
        return os.path.splitext(os.path.basename(files[0]))[0]
    return "submission"


def _match_runtime(challenge_id: int, rows: list, name: str | None, *,
                   language: str | None = None,
                   framework: str | None = None,
                   framework_version: str | None = None) -> dict:
    """The row of `rows` (a challenge's `runtime_options`) named `name`, or
    the first matching the deprecated keywords; `MLArenaError` when none."""
    if name is not None:
        if (language, framework, framework_version) != (None, None, None):
            raise MLArenaError("Pass a runtime name or the deprecated keywords, not both")
        for r in rows:
            if r["name"] == name:
                return r
        raise MLArenaError(
            f"Challenge {challenge_id} offers no runtime named {name!r}; "
            f"its runtimes: {', '.join(sorted(r['name'] for r in rows))}"
        )
    for r in rows:
        if language is not None and r.get("language") != language:
            continue
        if framework is not None and r.get("framework") != framework:
            continue
        if framework_version is not None and r.get("framework_version") != framework_version:
            continue
        return r
    raise MLArenaError(
        f"No runtime found for language={language!r} framework={framework!r} "
        f"framework_version={framework_version!r}. Available: {rows}"
    )


def _check_agent_argument(agent) -> None:
    """`submit(agent=…)` takes a class or a module: refuse a path (that is
    `files=`) and an instance before any request."""
    if isinstance(agent, (str, bytes, os.PathLike)):
        raise SubmissionError(
            f"agent= takes a class (or a module), not a path ({agent!r}): "
            f"to upload a file, pass files=[{agent!r}]"
        )
    if not (inspect.isclass(agent) or inspect.ismodule(agent)):
        raise SubmissionError(
            f"agent= takes a class (or a module), got an instance of "
            f"{type(agent).__name__}: pass the class itself "
            f"(agent={type(agent).__name__}), or the file with files=[...]"
        )


def _agent_source(agent, entry_filename: str | None, challenge_id: int) -> str:
    """The text `submit(agent=…)` uploads as `agent.py`: the source of a
    class (that class only: imports and helpers outside it are not in it)
    or of a module (the whole file). Refused before anything is created on
    a challenge whose entry file is not `agent.py`, or when the source
    cannot be read."""
    if entry_filename != "agent.py":
        raise SubmissionError(
            f"Challenge {challenge_id} does not take an agent.py "
            + (f"(its submission is {entry_filename!r}): pass files=[...]"
               if entry_filename is not None else "(it takes no upload)")
        )
    try:
        return inspect.getsource(agent)
    except (OSError, TypeError) as err:
        # A class defined in an interactive session has no source file.
        raise SubmissionError(
            f"agent=: cannot read the source of {agent!r} ({err}); write it "
            f"to agent.py and pass files=['agent.py']"
        ) from err


def _check_upload_limits(challenge_id: int, challenge: dict,
                         uploads: list[tuple[str, int]]) -> None:
    """Refuse, before the create, uploads the challenge's served limits
    (`max_upload_files`, `max_upload_size_bytes`) already refuse. `uploads`
    is `(file name, size in bytes)` per file. The server stays the judge;
    a None limit is one the challenge does not serve."""
    names = [name for name, _size in uploads]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise SubmissionError(
            f"Two files share the name {', '.join(duplicates)}: a submission "
            f"stores each file under its base name, so one would replace the other"
        )
    max_files = challenge["max_upload_files"]
    # A code challenge's new submission already holds `agent.py` (the
    # template), so it counts whether or not it is uploaded.
    stored = set(names) | ({"agent.py"} if challenge["entry_filename"] == "agent.py" else set())
    if max_files is not None and len(stored) > max_files:
        raise SubmissionError(
            f"Challenge {challenge_id} takes at most {max_files} file(s) per "
            f"submission; this one would hold {len(stored)} "
            f"({', '.join(sorted(stored))})"
        )
    max_bytes = challenge["max_upload_size_bytes"]
    total = sum(size for _name, size in uploads)
    if max_bytes is not None and total > max_bytes:
        raise SubmissionError(
            f"Challenge {challenge_id} takes at most {max_bytes} bytes per "
            f"submission ({max_bytes / (1024 * 1024):.0f} MB); these files "
            f"total {total} bytes"
        )


def _left_behind(err: MLArenaError, challenge_id: int,
                 submission_id: int) -> MLArenaError:
    """`err` (raised after `submit()` created a submission and before it
    deployed it) as the same class, naming the submission: the message says
    how to deploy or delete it, `submission_id` carries its id."""
    return type(err)(
        f"{err} (submission {submission_id} was created and is not deployed: "
        f"deploy it with deploy_submission({challenge_id}, {submission_id}) or "
        f"delete it with delete_submission({challenge_id}, {submission_id}))",
        status_code=err.status_code, body=err.body,
        submission_id=submission_id,
    )


def _course_left_behind(err: MLArenaError, course: dict) -> MLArenaError:
    """The refusal of the second call of a composition that created a course
    first (`create_challenge_for_class`): the course stays, empty, and the
    message says so. Same class, status and body; `course_id` is added to the
    body when the server's had none."""
    body = dict(err.body or {})
    body.setdefault("course_id", course["id"])
    return type(err)(
        f"{err} (course {course['id']} {course['name']!r} was created and holds "
        f"no challenge: add one with add_course_challenge({course['id']}, ...) or "
        f"delete it with delete_course({course['id']}))",
        status_code=err.status_code, body=body,
    )


def _safe_error(resp: requests.Response, fallback: str = "request failed") -> str:
    """The server's reason for refusing, as it always sends it.

    One key: every error body the backend produces — a view's own reply, the
    body-validation 400, and every `abort(...)` — puts the reason under
    `error`. A response with no JSON body at all (a proxy's HTML 502, say)
    falls back to its text.
    """
    try:
        reason = resp.json().get("error")
    except Exception:
        return resp.text or fallback
    return reason or fallback


def _json_body(resp: requests.Response) -> dict | None:
    """The reply's JSON object, or None when it has none (a proxy's HTML 502).

    What `MLArenaError.body` carries: a deploy 409's `deployment_limits` /
    `active_submission_limits`, a 400's `error` — the server's own keys,
    unchanged.
    """
    try:
        body = resp.json()
    except Exception:
        return None
    return body if isinstance(body, dict) else None


def _error(cls, resp: requests.Response, message: str):
    """Build `cls(message)` carrying the reply it was raised for."""
    return cls(message, status_code=resp.status_code, body=_json_body(resp))


def _failed(cls, label: str, resp: requests.Response):
    """`<label> failed: <server reason>` — the message every method has
    always raised on an unexpected status, now carrying the reply
    (`.status_code`, `.body`)."""
    return _error(cls, resp, f"{label} failed: {_safe_error(resp)}")


def _to_dataframe(data):
    """A list of rows as a pandas DataFrame if pandas is installed, else as
    is. Anything that is not a list (an envelope dict) is returned untouched."""
    if not isinstance(data, list):
        return data
    rows = data
    try:
        import pandas as pd
        return pd.DataFrame(rows)
    except ImportError:
        return rows


def _leaderboard_records(envelope: dict) -> list[dict]:
    """The board's rows with `metrics` spread into one column per declared
    key and, on a windowed board, `metrics_window` into one
    `metrics_window_<key>` column per key.

    A row's `metrics` is None while it has no scored run; otherwise it holds
    every declared key (the backend's contract), so a missing key is a drift
    and raises KeyError. A declared key that equals a row column would
    silently overwrite it, so it raises instead.
    """
    challenge = envelope["challenge"]
    keys = [spec["key"] for spec in challenge["metrics"]]
    windowed = challenge["window_days"] is not None
    records = []
    for leader in envelope["leaders"]:
        record = {k: v for k, v in leader.items()
                  if k not in ("metrics", "metrics_window")}
        values = leader["metrics"]
        window = leader["metrics_window"]
        for key in keys:
            if key in record:
                raise MLArenaError(
                    f"leaderboard: declared metric key {key!r} collides with "
                    "the row column of the same name"
                )
            record[key] = None if values is None else values[key]
            if windowed:
                record[f"metrics_window_{key}"] = (
                    None if window is None else window[key]
                )
        records.append(record)
    return records


# --- Backwards compatibility for the Python names ----------------------------
# 1.0.0 renamed "competition" to "challenge"; 2.0.0 renamed the participant's
# entry ("attached agent") to "submission". Existing notebooks and scripts are
# not expected to change with either: every old method name still resolves,
# and every old keyword argument is still accepted. Both paths emit a
# DeprecationWarning and forward to the new spelling. This covers the Python
# API only — the client speaks only the current REST routes and payload keys,
# so the values these methods return carry the new keys (`submission_id`, ...).

def _error_line(side: str, error_type, message) -> str | None:
    """One `tail_logs` error line: `    agent error[code_error]: boom`.

    `side` is `agent` (the caller's own `RunSubmissionResult`), `env` (the
    run) or `job` (the platform's cause on a failed run). The message is appended only when served — it is None on a row
    that is not the caller's, on the env side of a participant read, and on
    the job side for anyone but the challenge's staff.
    """
    if not error_type:
        return None
    line = f"    {side} error[{error_type}]"
    return f"{line}: {message}" if message else line


_LEGACY_KWARGS = {
    "competition_id": "challenge_id",
    "copy_from_competition_id": "copy_from_challenge_id",
    "ordered_competition_ids": "ordered_challenge_ids",
    "agent_name": "submission_name",
    "attache_agent_id": "submission_id",
    "agent_id": "submission_id",
    "copy_from_agent_id": "copy_from_submission_id",
    "max_active_agents_per_participant": "max_active_submissions_per_participant",
}

# `download_replay`: the file suffix of each `render_format` (the stored
# object's own key suffix, the backend's `RENDER_FORMATS`).
_REPLAY_SUFFIXES = {
    "frames_v2": ".frames.mp",
    "video_mp4": ".mp4",
    "image_png": ".png",
}

# `teacher_courses()` (deprecated): the columns of the removed list route
# that a `workspace()` course row carries.
_TEACHER_COURSE_KEYS = (
    "id", "name", "code", "slug", "visibility", "join_code",
    "start_date", "end_date", "archived_at_ts", "role",
)

_RENAMED_METHODS = {
    "competitions": "challenges",
    "competition": "challenge",
    "creator_competitions": "creator_challenges",
    "create_competition": "create_challenge",
    "set_competition_tags": "set_challenge_tags",
    "update_competition": "update_challenge",
    "set_competition_image": "set_challenge_image",
    "set_competition_markdown": "set_challenge_markdown",
    "start_competition": "start_challenge",
    "stop_competition": "stop_challenge",
    "update_competition_link": "update_course_challenge",
    "reorder_module_competitions": "reorder_module_challenges",
    "create_attached_agent": "create_submission",
    "upload_agent_file": "upload_submission_file",
    "deploy_agent": "deploy_submission",
    "agent_deploy_status": "submission_deploy_status",
    "delete_agent": "delete_submission",
    "list_agent_files": "list_submission_files",
    "get_agent_file_content": "get_submission_file_content",
    "update_agent_file_content": "update_submission_file_content",
    "delete_agent_file": "delete_submission_file",
    "agent_games": "submission_runs",
    # 4.6.0: the submission's page routes replaced the games route.
    "submission_games": "submission_runs",
}

# 4.23.0: linking an existing challenge into a course is gone (a challenge
# belongs to one course). The frozen aliases of the removed methods raise the
# sentence that names what replaces them.
_GONE_COMPETITION_METHODS = ("attach_competition", "detach_competition")
COURSE_CHALLENGE_SENTENCE = (
    "a challenge now belongs to its course: create it with `add_course_challenge`"
)

# Old methods that took `(challenge_id, submission_id)` where their
# replacement takes the submission alone (4.6.0: one read of the submission,
# `GET /api/submissions/submission/{sid}`). The alias drops the challenge id.
_KEYED_BY_SUBMISSION_METHODS = {
    "agent_status": "submission",
    "submission_status": "submission",
    "submission_overview": "submission",
}


def _accepts_legacy_kwargs(fn):
    """Let a renamed keyword argument still be passed by its old name."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        for old, new in _LEGACY_KWARGS.items():
            if old in kwargs:
                if new in kwargs:
                    raise TypeError(
                        f"{fn.__name__}() got both {old}= and {new}=; "
                        f"pass only {new}="
                    )
                warnings.warn(
                    f"{old}= is deprecated, use {new}= instead",
                    DeprecationWarning,
                    stacklevel=2,
                )
                kwargs[new] = kwargs.pop(old)
        return fn(*args, **kwargs)
    return wrapper


def _positional_only_names(method) -> tuple[str, ...]:
    """The positional-only parameters of `method`, `self` left out."""
    return tuple(
        name for name, parameter in inspect.signature(method).parameters.items()
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY and name != "self"
    )


def _deprecated_alias(old_name, new_name, *, drops_challenge_id=False):
    # The old method took every parameter by position or by keyword. Where
    # its replacement takes some by position only (`update_course_challenge
    # (module_id, challenge_id, /, **fields)`, so that `module_id=` can be a
    # body key), the alias keeps the old signature: it binds those names
    # itself and hands them over by position. The old keyword spelling
    # (`competition_id=`) was renamed before this runs (`_accepts_legacy_kwargs`).
    positional = _positional_only_names(getattr(MLArenaClient, new_name))

    def alias(self, *args, **kwargs):
        warnings.warn(
            f"MLArenaClient.{old_name}() is deprecated, "
            f"use .{new_name}() instead",
            DeprecationWarning,
            # The caller's line: every alias is called through
            # `_accepts_legacy_kwargs`, one frame above this one.
            stacklevel=3,
        )
        if drops_challenge_id:
            if args:
                args = args[1:]
            elif "challenge_id" in kwargs:
                del kwargs["challenge_id"]
            else:
                raise TypeError(
                    f"{old_name}() missing required argument: 'challenge_id'"
                )
        for name in positional[:len(args)]:
            if name in kwargs:
                raise TypeError(
                    f"{old_name}() got multiple values for argument {name!r}"
                )
        for name in positional[len(args):]:
            if name not in kwargs:
                raise TypeError(
                    f"{old_name}() missing required argument: {name!r}"
                )
            args += (kwargs.pop(name),)
        return getattr(self, new_name)(*args, **kwargs)
    alias.__name__ = old_name
    alias.__qualname__ = f"MLArenaClient.{old_name}"
    alias.__doc__ = f"Deprecated alias for :meth:`{new_name}`."
    return alias


for _name, _member in list(vars(MLArenaClient).items()):
    if not _name.startswith("_") and inspect.isfunction(_member):
        setattr(MLArenaClient, _name, _accepts_legacy_kwargs(_member))

for _old, _new in _RENAMED_METHODS.items():
    if not hasattr(MLArenaClient, _new):
        raise AttributeError(
            f"compat map is stale: MLArenaClient has no .{_new}()"
        )
    # Wrapped so an old keyword (`competition_id=`) is renamed before the
    # alias binds it.
    setattr(MLArenaClient, _old, _accepts_legacy_kwargs(
        _deprecated_alias(_old, _new)))

def _gone_alias(old_name):
    def gone(self, *args, **kwargs):
        raise MLArenaError(
            f"MLArenaClient.{old_name}() is gone: {COURSE_CHALLENGE_SENTENCE}"
        )
    gone.__name__ = old_name
    gone.__qualname__ = f"MLArenaClient.{old_name}"
    gone.__doc__ = f"Removed: {COURSE_CHALLENGE_SENTENCE}."
    return gone


for _old in _GONE_COMPETITION_METHODS:
    setattr(MLArenaClient, _old, _gone_alias(_old))

for _old, _new in _KEYED_BY_SUBMISSION_METHODS.items():
    if not hasattr(MLArenaClient, _new):
        raise AttributeError(
            f"compat map is stale: MLArenaClient has no .{_new}()"
        )
    # Wrapped so an old keyword (`competition_id=`) is renamed before the
    # challenge id is dropped.
    setattr(MLArenaClient, _old, _accepts_legacy_kwargs(
        _deprecated_alias(_old, _new, drops_challenge_id=True)))
