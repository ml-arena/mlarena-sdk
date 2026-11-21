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
import re
import tempfile
import time
import warnings
from pathlib import Path
from typing import Iterator
from urllib.parse import quote, unquote

import requests
import yaml

from mlarena.bundle import bundle_root, read_bundle
from mlarena.chat import ChatConversation
from mlarena.exercise import format_attempt, source_of
from mlarena.exceptions import (
    AuthenticationError,
    ChallengeNotFoundError,
    ChatSessionNotFoundError,
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

    Pass the full bearer token returned by your Profile page (`mlk_user_...`,
    `mlk_creator_...`, or `mlk_teacher_...`). The token's scope segment dictates
    which endpoints are reachable — a `user`-scope token cannot call a
    `creator`-required route, and vice versa. A GitHub Actions run sends the
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
            raise _error(PermissionDeniedError, resp,
                         _safe_error(resp, "Access denied"))
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
        course-assistant row), ``has_creator_access``
        (the real creator rule — the flag *or* admin *or* an assistant row
        on a challenge), ``can_create_course``, ``max_courses_limit`` and
        ``active_course_count`` (the limit behind it and the active, not
        archived, courses you own; an admin is exempt from the limit),
        and ``github_login`` (the GitHub account
        linked to this one, None when there is none; it is linked from the
        console's Profile page, a browser flow). Every key is always present.

        Note the route answers to any scope, but the *flags* describe the
        account, not the token: a ``mlk_user_…`` token on an admin account
        still reports ``is_admin: true`` while being refused admin routes.

        Listing or rotating API keys is deliberately **not** offered here:
        ``/api/auth/api_keys`` and ``/api/auth/api_keys/<scope>/rotate``
        answer 403 to a bearer caller, so that a leaked ``user`` key cannot
        mint a ``creator`` one. Rotate keys from the console's Profile page.
        """
        resp = self._request("GET", self._url("/auth/current_user"),
                             headers=self._headers())
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "me", resp)
        return resp.json()

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

    # ---- Challenges (public read; create/update via creator scope) ----

    def challenges(
        self,
        *,
        q: str | None = None,
        tags: list[str] | str | None = None,
        status: str = "started",
        page: int | None = None,
        per_page: int | None = None,
    ):
        """List the challenges the caller may see. Public; no auth required.

        Anonymous callers get the public challenges. A signed-in caller also
        gets the ones they own or assist on, and the ones taught by a course
        they are enrolled in, teach or assist (published modules only) — the
        same rule every challenge route applies.

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
        configuration), ``my_rank``
        (your participant's rank on the challenge's board; None when
        anonymous or with nothing on the board) and ``course``
        (``{id, name, code}`` of the course you are enrolled in that holds
        the challenge, None otherwise; those rows are listed first, and
        ``q`` also matches them on the course's name or code). Call
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

        `can_reuse` says whether you may add the challenge to a course of
        yours (`attach_challenge`, `add_course_challenge`): a public
        challenge, one you own or assist, any for an admin; False for a
        private challenge you only see through a course.

        `kind_label` is the kind's one label (`"Python agent"` |
        `"Upload a file"` | `"Chat with an AI agent"` | `"Exercises"`).
        `agent_template` (the code
        a participant starts `agent.py` from) is served for code challenges
        only, None for a file or chat challenge. `submission_layout` is the
        CSV a file challenge's submission must follow — `columns`,
        `separator`, `id_column`, `target_column`, `row_count` (None when
        unknown) — read from the creator's ground truth; None for any other
        kind, a non-CSV submission file, or before a `y_test.csv` is parsed.
        `miniature` is None when the challenge has no miniature.

        `runtime_options` is the set of agent runtime images the challenge
        offers (ordered by id; `[]` for a file or chat challenge) and
        `default_agent_runtime_id` the one a new submission gets (None for a
        file or chat challenge) — both written by the creator
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
        `"vm"`), `node_pool` (`"autoscaling"` | `"large-autoscaling"` |
        `"sandbox"`, the gVisor pool; `None` on a vm machine), `vm_health_ok`,
        `vm_health_checked_at_ts`, `gpu_device_ids`, `gpu_count`,
        `image_pull_policy`, `job_cpu_max`, `job_memory_max`,
        `job_timeout_max_sec` (the longest `simulation_timeout_sec` a
        challenge on it may ask for), `max_concurrent_jobs`, `is_enabled`,
        `created_at_ts` and `kernels`,
        its queues (`MachineKernelOut`: `id`, `machine_id`, `kernel_version`,
        `docker_image_worker_envagent_id`, `worker_replicas`, `is_public`,
        `is_default`, `created_at_ts`).

        A non-admin sees only the machines with a queue visible to them
        (public, or granted to them) and only those queues; an admin sees
        every machine. `kernel_version` keeps the queues of that kernel —
        pass a challenge's `creator_challenge(id)["configuration"]
        ["kernel_version"]` to list the machines `update_settings(
        machine_id=…)` accepts for it. An unknown kernel answers 400.

        Creating, editing and granting machines is admin cluster-ops,
        console-only (no SDK method, by decision).

        Requires a `creator`- or `teacher`-scope token.
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

        Mirrors `GET /api/author/workspace`. Needs a `creator` or a `teacher`
        key, either one: the route replaced the list of each scope, and one
        key reads the whole payload. A `user` key is refused
        (`PermissionDeniedError`), as it was by both lists. An account that
        owns and assists nothing gets empty lists and its quotas. Returns
        `{"scope", "challenges", "courses", "attention", "attention_overflow",
        "quotas"}`; `scope` is `"mine"`: everyone's challenges and courses
        (`scope=all`) is an admin's listing in the console, which no key
        reads.

        `challenges`: the challenges you own or assist on, your own first,
        each group newest first. A row is `{"challenge", "counters",
        "created_at_ts", "start_blockers", "start_blockers_error"}`:
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
        `start_date`, `end_date`, `archived_at_ts`, `enrollment_state`
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
        each item has a `kind`: `deploy_failed_submissions` (a `started`
        challenge with submissions whose deploy failed: the count and its
        split by owner), `draft_blocked` (a draft with `start_blockers`),
        `draft_unreadable`, `course_ending` (a course ending within 7 days
        with `enrolled_without_submission` > 0), `course_challenge_closed`
        (a challenge students see in a course of yours that is still
        running, not your own, that is `stopped` or `ended`). Items name
        their object by id (`challenge_id`, `course_id`) and name. At most
        20 items of each kind are listed, in the rows' order;
        `attention_overflow` is `{kind: how many more}` (0 when all are
        there). The rows carry every fact.

        A challenge row carries `archived_at_ts` (null unless
        `archive_challenge` stamped it).

        `quotas`: `owned_challenge_count` (the challenges you own that are
        not archived; no limit on challenges yet),
        `max_courses_limit`, `active_course_count`, `can_create_course`.
        """
        return self._course_call("GET", "/author/workspace", error_label="workspace")

    def creator_challenges(self) -> list:
        """Deprecated: use `workspace()["challenges"]`.

        The challenges you own or assist on, built from `workspace()`: each
        item is the row's challenge reference (`id`, `name`, `slug`,
        `status`, `end_date_ts`, `is_public`, `kernel_version`, `kind_label`,
        `user_id`, `owner_username`, `role`, `can_edit`, `statistics`) plus
        `created_at_ts`. The list route this method called
        (`GET /api/creator_challenge/challenges`) is gone, and with it
        `description`, `has_configuration`, `has_environment` and
        `machine_name` (read them from `creator_challenge(id)`). An admin
        gets their own challenges; everyone's is the console's. Works with
        the `creator` key it always needed (a `teacher` key reads it too).
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
        """The challenge as its editor sees it (creator scope).

        An exercise set you may edit (the one
        `add_course_exercise_set` made for your course) also takes a
        `teacher` key; every other kind takes the `creator` scope.

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
        now, each `{code, message, fix_by, can_fix, place}`: `fix_by` is who
        fixes it (`"author"`: any editor of the challenge; `"admin"`),
        `can_fix` whether you can, and `place` the part of the challenge
        workspace where (`files`, `benchmark`, `chat_agent`,
        `scoring_settings` under Scoring; `advanced`, `platform` under
        Settings; None when no part does).
        Codes: (`env_missing`, `configuration_missing`,
        `image_inconsistent`, `config_not_applied`, `required_file_missing`,
        `runtime` — the size, the run timeout, the agents per run or where
        the challenge is placed —,
        `env_invalid`, `chat_missing`, `llm_not_configured`,
        `ground_truth_missing`, `benchmark_missing`, `benchmark_unscored`);
        `[]` means ready, None while started, and None with
        `start_blockers_error` set (a sentence) when the challenge's files
        could not be read, it is incomplete or computing the list failed, so
        the checklist is unknown. A file you got wrong
        is a blocker, not such a fault: an `env.py` that is not UTF-8 text is
        `env_invalid`, a `challenge.toml` that is not is `config_not_applied`.
        A `runtime` message is worded
        for the caller: what to change (or "ask an admin") for a creator,
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

        Unlike `challenge()` this works on your own hidden challenges and
        shows the authoring state; `challenge()` is the participant view.

        Requires a `creator`-scope token and ownership (or admin) of the
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
        (`creator` | `admin`), `capability` (the `available_kinds()`
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
        """The challenge's resolved config as a `challenge.toml` (creator scope).

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
        """The challenge kinds `create_challenge` accepts (creator scope).

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

    def copyable_challenges(self) -> dict:
        """The challenges `create_challenge(copy_from_challenge_id=…)` accepts.

        Mirrors `GET /api/creator_challenge/copyable_challenges`
        (`lifecycle.py`). Returns `{"challenges": [...]}` — the ones you own
        (a challenge you only assist on is not a copy source), by name: each
        row is the challenge's reference (see `challenge_overview`) plus
        `machine_name`.
        """
        resp = self._request("GET",
            self._url("/creator_challenge/copyable_challenges"),
            headers=self._headers(),
            timeout=30,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "copyable_challenges", resp)
        return resp.json()

    def datasets(self, challenge_id: int) -> dict:
        """List the public datasets attached to a challenge.

        Mirrors `GET /api/challenges/{id}/datasets` (`challenges.py`,
        `get_challenge_datasets`, `@auth_required('user')`). Returns
        ``{"datasets": [{"id", "label", "description",
        "files": [{"id", "label", "download_url", "file_size_bytes", ...}]}]}``
        where each ``download_url`` is a short-lived (60 min) signed URL the
        caller can GET directly. This is how a participant pulls the training /
        test data a creator published for a file challenge. Requires any
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

    def create_challenge(self, name: str, kernel_version: str,
                           description: str | None = None,
                           copy_from_challenge_id: int | None = None,
                           tag_names: list[str] | None = None,
                           is_public: bool | None = None) -> dict:
        """Create a new challenge via the creator-scope authoring flow.

        `kernel_version` is a kind, and a kind is a kernel: the keys of
        `KINDS` in `backend/app/views/creator_challenge/_helpers.py`, served
        by `available_kinds()` — call it for the live list and whether each
        has a machine (`has_machine`). Gymnasium, PettingZoo and the other
        env-image families are not kinds: they are flex_v1 challenges whose
        env runtime an admin picks.

        The backend resolves the machine (the kernel's default queue; a
        copy keeps its source's machine when you may use it) + default
        evaluation + env-image family from the kind. The reply carries
        `machine_name` and `machine_note` (why a copy did not keep its
        source's machine, else None). To move the challenge to another
        machine, call `update_settings(challenge_id, machine_id=…)` with an
        id from `machines(kernel_version)`.

        A copy (`copy_from_challenge_id`) takes its source's env folder,
        `challenge.toml`, every config key (data feed and participation
        limits included) and tags; not its datasets.

        If `tag_names` is given, each name is resolved against the public
        tag catalog (`GET /api/challenge_tags/tags`) before the create
        call; on a copy they replace the source's tags. Unknown names raise
        `MLArenaError` (fail fast — the catalog is admin-curated; new tags
        are not auto-created).

        A new challenge is private unless `is_public=True` is passed: hidden
        from public listings, search and direct URLs. Only the owner,
        creator assistants, admins, and the students, teachers and TAs of a
        course that teaches it (through a published module) can view or
        interact with a hidden challenge; everyone else gets 404.
        Visibility can be toggled later via
        `update_challenge(is_public=...)`.

        Requires a `creator`-scope token.
        """
        body = {"name": name, "kernel_version": kernel_version}
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

        Requires a `creator`-scope token and ownership (or admin) of the
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
        """The tags currently on a challenge (creator scope).

        Mirrors `GET /api/creator_challenge/challenge/{id}/tags`
        (`tags.py`). Returns the tag rows (`id`, `name`, `description`);
        `set_challenge_tags()` replaces the set.
        """
        return self._creator_get(challenge_id, "/tags", "challenge_tags")

    def _creator_get(self, challenge_id: int, suffix: str, label: str,
                     params: dict | None = None):
        """GET one creator_challenge sub-resource of a challenge.

        Every creator read answers the same way — 404 for a challenge you
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
                           end_date_ts: str | None | _Unset = _UNSET) -> dict:
        """Patch top-level challenge fields (creator scope).

        Mirrors `PUT /api/creator_challenge/challenge/{id}`. Only fields
        explicitly passed are sent — everything else is left untouched.

        ``name`` is locked once the challenge has started — a rename strands
        the links and course material already pointing at it. ``description``
        and ``is_public`` stay editable while it runs, so a creator can fix the
        blurb or hide a started challenge without stopping it. When
        ``is_public=False``, only the owner, creator assistants, admins, and
        the members (students, teachers, TAs) of a course that teaches it can view or interact with the challenge.

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
        ["config"]["origin"]` says who owns each key. Once
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

        Run budgets (creator keys; strictly positive; frozen while the
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
        shorter run timeout, the agents per run, or "ask an admin") for a
        creator, the machine, its limit and the rule that failed for an
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

        Agent runtimes (creator keys, kinds with `agent_containers`;
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

        Replay (a creator key, editable while started):
            render_delay_second: the default frame duration (s, >= 0): how
                long a replay frame that sets no `duration_s` of its own is
                shown. Editing it changes the playback of stored replays too.

        Exercise limits (creator keys, kinds with `exercises`; editable while
        the challenge runs except `memory_limit_mb`, which sizes the checker):
            time_limit_sec: the CPU time one attempt may use, 1..30 s
                (default 5); past it the attempt fails `timeout`.
            memory_limit_mb: the memory of one attempt, 64..2048 MB (default
                512); past it the attempt fails `memory`.
            max_concurrent_attempts: attempts the checker grades at once,
                1..16 (default 8).
            max_attempts_per_minute: attempts one group may send per minute,
                1..600 (default 20); past it `submit_exercise` gets a 429.

        Requires a `creator`-scope token and ownership (or admin) of the
        target challenge; an exercise set you may edit also takes a `teacher`
        key.
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
        """The challenge's env folder, as the editor lists it (creator scope).

        Mirrors `GET /api/creator_challenge/challenge/{id}/env/files`
        (`env_files.py`). Returns `{"files": {"<name>": {"content",
        "language", "is_binary", "size"}}}`; a binary file has `content: null`
        and is reachable only through the console's download.
        """
        return self._creator_get(challenge_id, "/env/files", "list_env_files")

    def delete_env_file(self, challenge_id: int, filename: str) -> dict:
        """Delete one file from the challenge's env folder (creator scope).

        Mirrors `DELETE /api/creator_challenge/challenge/{id}/env/files/{name}`
        (`env_files.py`). Refused once the challenge has started, and `env.py`
        itself cannot be deleted — the challenge would have nothing to run.
        """
        return self._creator_delete(
            challenge_id, f"/env/files/{filename}", "delete_env_file"
        )

    def check_env(self, challenge_id: int, content: str = "") -> dict:
        """Validate an `env.py` buffer without saving it (creator scope).

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
        (creator scope).

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
        and writes nothing. Refused once the challenge has started, as the
        file routes are.

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
        (creator scope).

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
        save, as `upload_env_file` describes.
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
        (`assets.py`), the creator-scoped read of the same file the public
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
        """The challenge overview markdown (`overview.md`), creator scope.

        Mirrors `GET /api/creator_challenge/challenge/{id}/markdown`
        (`assets.py`). Returns `{"content": "..."}` — the empty string when
        the creator has never written one.
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
        """The challenge's benchmark folder (creator scope).

        Mirrors `GET /api/creator_challenge/challenge/{id}/benchmark/files`
        (`benchmark.py`). Same shape as `list_env_files()`: the benchmark
        `agent.py` on a code challenge, the benchmark submission file on a
        file challenge.
        """
        return self._creator_get(
            challenge_id, "/benchmark/files", "list_benchmark_files"
        )

    def delete_benchmark_file(self, challenge_id: int, filename: str) -> dict:
        """Delete one benchmark file (creator scope).

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

        Requires that env.py has been uploaded and a benchmark `agent.py`
        (or the challenge's submission file for file_v1, e.g. submission.csv /
        pitch.txt) is already present. The call itself is fire-and-forget; poll
        `benchmark_status` for completion.
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
        """Move a creator challenge to ``status="started"``.

        Mirrors `PUT /api/creator_challenge/challenge/{id}/start`. Accepted
        from ``draft``, ``stopped`` or ``ended`` (400 when already started;
        409 for an archived challenge, `unarchive_challenge` it first).
        Returns `{"message", "clean_redeploy"}`.

        An exercise set you may edit (the one
        `add_course_exercise_set` made for your course) also takes a
        `teacher` key; every other kind takes the `creator` scope. The start gate is the same for both.

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
        consistency, or a runtime rule measured against the machine: the
        machine, its queue, its GPUs, a valid size or timeout above its
        ceiling —, 400 for missing authoring, which includes a size that is
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
        """Move a started creator challenge to ``status="stopped"``.

        Mirrors `PUT /api/creator_challenge/challenge/{id}/stop`. Accepted
        only from ``started`` (400 otherwise). Stopping leaves submissions,
        results, the benchmark record and ``start_date_ts`` untouched, and
        unlocks the authoring fields like a draft, so the challenge can be
        fixed and restarted with `start_challenge()`. It changes the status
        and nothing else: runs already pending or running still finish and
        count, no new deploy is accepted, and the challenge leaves the
        catalog's started list while its page and results stay
        (`scoring_update(id)["stop"]` counts what is in flight).

        An exercise set you may edit (the one
        `add_course_exercise_set` made for your course) also takes a
        `teacher` key; every other kind takes the `creator` scope.
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
        """Move a started creator challenge to ``status="ended"`` now.

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
        return self._course_call(
            "GET", f"/creator_challenge/challenge/{challenge_id}/scoring_update",
            error_label="scoring_update",
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
        return self._course_call(
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
        return self._course_call(
            "POST", f"/creator_challenge/challenge/{challenge_id}/archive",
            error_label="archive_challenge",
        )

    def unarchive_challenge(self, challenge_id: int) -> dict:
        """Take a challenge out of the archive — its owner or an admin.

        Mirrors `POST /api/creator_challenge/challenge/{id}/unarchive`. It
        counts among your challenges again and can be restarted. Returns the
        challenge; 409 when it is not archived.
        """
        return self._course_call(
            "POST", f"/creator_challenge/challenge/{challenge_id}/unarchive",
            error_label="unarchive_challenge",
        )

    def challenge_deletion(self, challenge_id: int) -> dict:
        """What `delete_challenge` would do, without doing it — owner or admin.

        Mirrors `GET /api/creator_challenge/challenge/{id}/deletion`. Returns
        `can_delete` and what decides it: `status` (only a `draft` is
        deleted), `submission_count` (submissions other than the benchmark),
        `team_count`, `module_link_count` (course modules linking it),
        `lesson_count` (lessons naming it in a directive), `runs_in_flight`
        (a benchmark test on its way); and what goes with the challenge:
        `env_files` (names in its env folder), `benchmark_files` (names in
        its benchmark folder), `has_benchmark`, `run_count`
        (every run of its environment), `exercise_count`,
        `chat_session_count`, `deleted_datasets` (`[{id, label,
        file_count}]`, attached to this challenge alone: deleted with their
        stored files) and `kept_datasets` (attached to another challenge
        too). `replay_count` replay files of those runs are not removed:
        they expire from storage after 365 days. `storage_left` is `[]` here.
        """
        return self._course_call(
            "GET", f"/creator_challenge/challenge/{challenge_id}/deletion",
            error_label="challenge_deletion",
        )

    def delete_challenge(self, challenge_id: int) -> dict:
        """Delete your own draft challenge — its owner or an admin.

        Mirrors `DELETE /api/creator_challenge/challenge/{id}`. Only a draft
        with no submission but the benchmark, no team, no course module
        linking it, no lesson naming it and no run on its way (409 otherwise,
        the counts in `.body`; a challenge that ran is archived instead,
        `archive_challenge`). Deletes its rows (configuration, environment,
        evaluation, tags, assistants, an exercise set's exercises), its
        benchmark submission, every run of its environment, its folder (env
        files, overview, cover, benchmark files) and the datasets attached
        to it alone with their stored files. Returns what
        `challenge_deletion` announced, with `storage_left`: what could not
        be removed from storage although the rows are gone (`[]` when all of
        it went; check it).
        """
        return self._course_call(
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
          when not, `read_only_reason` (`started`, `env_edited`,
          `config_file`) and `read_only_message` say why.
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
        and nothing changes; delete `env.py` to use the form again. A 400
        while the challenge is started (stop it first), when `challenge.toml`
        holds `metrics`, or for a bad name.
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
        """Create a public dataset bucket on a challenge (creator scope).

        Mirrors `POST /api/creator_challenge/challenge/{id}/datasets`
        (`datasets.py`, `create_creator_dataset`). Returns
        `{"id", "label", "description"}`; use the returned `id` with
        `upload_dataset_file`. The backend rejects this once
        the challenge is started, so call it before `start_challenge`.
        Files uploaded here are served to participants via `datasets()`.

        Requires a `creator`-scope token.
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
        """List a challenge's datasets + files from the creator side.

        Mirrors `GET /api/creator_challenge/challenge/{id}/datasets`
        (`datasets.py`, `get_creator_datasets`). Like `datasets()` but
        creator-scoped (works on your own hidden challenges). Returns
        `{"datasets": [{"id","label","files":[…]}]}`.

        Requires a `creator`-scope token.
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
        """Upload one file into a challenge dataset (creator scope).

        Multipart POST to
        `/api/creator_challenge/challenge/{id}/datasets/{dataset_id}/files`
        (`datasets.py`, `upload_dataset_file`); the file is stored in R2 and
        exposed to participants through `datasets()` as a signed
        `download_url`. Rejected once the challenge is started. Returns
        `{"id", "label", "file_size_bytes"}`.

        Requires a `creator`-scope token.
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
        """Update a dataset's label / description (creator scope).

        PUT to `/api/creator_challenge/challenge/{id}/datasets/{dataset_id}`
        (`datasets.py`, `update_creator_dataset`). Only fields explicitly
        passed are sent. Allowed while the challenge is started (text only; its files stay locked). Returns `{"id", "label", "description"}`.

        Requires a `creator`-scope token.
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
        """Delete a dataset and every file in it (creator scope).

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
        """Delete one file from a challenge dataset (creator scope).

        DELETE to
        `/api/creator_challenge/challenge/{id}/datasets/{dataset_id}/files/{file_id}`
        (`datasets.py`, `delete_dataset_file`); removes the DB row and its
        R2 object. Rejected once the challenge is started. Use this before re-uploading a file with the
        same name — `upload_dataset_file` always adds a new row, so replacing a
        file means delete-then-upload. Get `file_id` from `creator_datasets()`.

        Requires a `creator`-scope token.
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

    # ---- Creator: runs, submissions and assistants (creator scope) ---------

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
        creator with the environment's error, its stdout and every agent's row
        inline. Includes benchmark tests, deployment games and participant runs.
        """
        return self._creator_get(
            challenge_id, "/runs", "creator_runs",
            params={"offset": offset, "limit": limit},
        )

    def creator_submissions(self, challenge_id: int,
                            course_id: int | None = None) -> dict:
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
        submission that has no row there.

        `course_id` keeps the submissions of that course's students (their
        owner is enrolled in it: the population of `leaderboard(course_id=…)`).
        It must be a course holding the challenge that you manage (you own
        or assist it, or you are an admin: the rows of `challenge_courses()`
        with `can_edit` true); any other id raises (400). `rank` stays
        the place on the whole board.
        """
        return self._creator_get(
            challenge_id, "/submissions", "creator_submissions",
            params=None if course_id is None else {"course_id": course_id},
        )

    def challenge_overview(self, challenge_id: int) -> dict:
        """The numbers of a challenge you own or assist on.

        Mirrors `GET /api/creator_challenge/challenge/{id}/overview`
        (`follow.py`). Returns `{"challenge": {...}, "counters": {...}}`.

        `challenge` is the challenge's reference, the shape every payload
        that names a challenge carries: `id`, `name`, `slug`, `status`,
        `end_date_ts`, `is_public`, `kernel_version`, `kind_label`,
        `user_id`, `owner_username`, `role` (yours on it: `"owner"`,
        `"assistant"` or None), `can_edit` (whether you may open its
        workspace: its owner, an assistant, or an admin) and `statistics` (`unique_participants`: teams
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
        `last_submission_created_at_ts`; `course_count`, how many courses
        hold the challenge, archived ones included (see `challenge_courses`).
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

    def challenge_courses(self, challenge_id: int) -> dict:
        """The courses that use a challenge you own or assist on.

        Mirrors `GET /api/creator_challenge/challenge/{id}/courses`
        (`follow.py`). Returns `{"courses": [...], "other_course_count": N}`.
        `courses` are the ones you may see, those you can open (`can_edit`)
        first, then by name: a course you own or assist on
        (an admin: every course), and a course you are enrolled in, or a
        public course that is not archived, where students see the
        challenge. Each carries `id`, `name`, `code`, `slug`, `visibility`,
        `archived_at_ts` (None = active), `owner_username` (the course
        owner's account name on a course you manage or are enrolled in, None
        on any other: a public course you only read), `role` (what you
        are to the course: `"owner"`, `"assistant"` or None), `can_edit`
        (whether you may open the course's workspace: its owner, an
        assistant, or an admin), `enrollment_count` (None unless `can_edit`)
        and `is_published` (whether students
        see the challenge there). Every other course holding the challenge —
        another teacher's private, unlisted or archived course — is only
        counted in `other_course_count`; archived courses count like the
        others, here and in `challenge_overview()["counters"]["course_count"]`.
        """
        return self._creator_get(challenge_id, "/courses", "challenge_courses")

    def clean_redeploy_submission(self, challenge_id: int,
                                  submission_id: int) -> dict:
        """Wipe one submission's results and queue a fresh deployment.

        Mirrors `POST
        /api/creator_challenge/challenge/{cid}/submissions/{sid}/clean_redeploy`
        (`submissions.py`) — the Re-score action of the Studio's submissions
        tab. Allowed for an `active` or `deploy_failed` submission; anything
        else is a 400 naming the status. Returns `submission_id`,
        `deleted_results`, `deployment_id` and the submission's status block.

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

    def clean_redeploy_all(self, challenge_id: int, *, dry_run: bool = False) -> dict:
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
        `start_challenge(id, clean_redeploy_all=True)`.
        """
        resp = self._request("POST",
            self._url(
                f"/creator_challenge/challenge/{challenge_id}"
                "/submissions/clean_redeploy_all"
            ),
            params={"dry_run": "true"} if dry_run else None,
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
        the participant's own `delete_submission()`, from the creator's side.
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
        `created_at_ts` (no email). An assistant has the creator's editing
        rights on this challenge. Every editor of the challenge (its owner,
        an assistant, an admin) may read the list; only the owner (or an
        admin) may change it. The owner is not an entry:
        `creator_challenge(id)["owner_username"]` names them.
        """
        return self._creator_get(
            challenge_id, "/assistants", "challenge_assistants"
        )

    def add_challenge_assistant(self, challenge_id: int, username: str) -> dict:
        """Grant a user the creator's rights on this challenge — owner only.

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
        in a deployable state, the daily deploy quota is used up, or the
        participant already holds `max_active_submissions` slots. Its body,
        on the error's `.body`, is `{"error", "deployment_limits",
        "active_submission_limits"}` — the same two blocks
        `submission_deploy_status()` returns, so the caller can read
        `next_deploy_available_at` or `active_submissions_remaining` off the
        exception without another call.
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
        """The agent runtime images of a kernel: the catalogue a creator picks
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
        a creator adds it to the set. An unknown kernel answers 400.

        Requires a `creator`-scope token.
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
        `GET /api/challenges/{id}`): the set its creator chose
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
        opts = self.runtime_options(challenge_id)
        # Normalize back to list[dict] in case pandas converted it.
        rows = opts.to_dict("records") if hasattr(opts, "to_dict") else list(opts)
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
        warnings.warn(
            "resolve_runtime(language=, framework=, framework_version=) is deprecated, "
            "pass the runtime's name: resolve_runtime(challenge_id, \"sklearn\")",
            DeprecationWarning,
            stacklevel=2,
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
        `update_env_file_content` on the creator side. The backend validates
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
        on the challenge, the challenge's staff (admin, creator, creator
        assistant), the owner's teacher, and anyone signed in when it is
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
          | `teacher` | `public`. Only the owner writes (`update_submission`,
          uploads, deploy, delete).
        - `agent_runtime` — the pinned runtime (the `runtime_options()` entry
          shape), None on a kernel without agent containers.
        - `latest_deploy` — the newest attempt (`submission_deploys()` row:
          `id`, `created_at_ts`, `status` — `queued` | `running` |
          `succeeded` | `failed` | `cancelled` —, `finished_at_ts`,
          `failure_message`, `failure_owner`, `initiated_by`), None before the
          first deploy. `initiated_by` says who started it: `participant`
          (you, `deploy_submission`), `staff` (a creator's
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
        `env_stdout_logs` belong to the challenge's creator and are None
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
        line when the challenge's code is (its message is the creator's, so
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
        """Convenience: create submission → (pick runner) → upload files → deploy.

        Either pass `agent=<class>` (its source is uploaded as `agent.py`) or
        `files=[...]` (each path is uploaded under its basename).
        `submission_name` defaults to the class name, or to the first file's
        base name without extension.

        Optional runner selection (between create and deploy):
            runtime:    the runtime's name, one the challenge offers
                        (`runtime_options(challenge_id)["name"]`, listed on the
                        challenge page): "python", "sklearn", "torch", ...
                        Resolved by `resolve_runtime()`. A dict spec
                        (`{"framework": "torch"}`) is deprecated (4.8.0).
            runtime_id: the runtime's numeric id.
        Pass at most one of `runtime_id` / `runtime`. With neither, the
        submission runs on the challenge's default runtime
        (`challenge(challenge_id)["default_agent_runtime_id"]`).

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
        as soon as the deploy is accepted (202), exactly as before.

        The submission is remembered for `status()` as soon as it is created,
        so after any of those errors `status()` reads the submission that was
        left behind; a refused deploy (409 — quota, active-submission limit,
        state) propagates as the `SubmissionError` `deploy_submission()`
        raises, with the server's limits on `.body`.

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
        if runtime_id is not None and runtime is not None:
            raise SubmissionError("Provide at most one of runtime_id= or runtime=")
        if wait and not deploy:
            raise SubmissionError("wait=True needs deploy=True")

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

        # Pin runner before deploy so the JobPod uses the right image.
        if isinstance(runtime, dict):
            warnings.warn(
                "submit(runtime={...}) is deprecated, pass the runtime's name: "
                "runtime=\"sklearn\" (runtime_options(challenge_id) lists them)",
                DeprecationWarning,
                stacklevel=2,
            )
            with warnings.catch_warnings():  # warned once, just above
                warnings.simplefilter("ignore", DeprecationWarning)
                runtime_id = self.resolve_runtime(challenge_id, **runtime)["id"]
        elif runtime is not None:
            runtime_id = self.resolve_runtime(challenge_id, runtime)["id"]
        if runtime_id is not None:
            self.set_agent_runtime(submission_id, runtime_id)

        tmp_dir = None
        try:
            if agent is not None:
                tmp_dir = tempfile.mkdtemp()
                agent_py = os.path.join(tmp_dir, "agent.py")
                with open(agent_py, "w") as f:
                    f.write(inspect.getsource(agent))
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
                )

            if not deploy:
                return {"submission_id": submission_id, "deploy": None}
            result = {
                "submission_id": submission_id,
                "deploy": self.deploy_submission(challenge_id, submission_id),
            }
            if wait:
                # Drain the generator: tail_logs() stops on `is_settled` and
                # raises on timeout, so this is the whole wait.
                for _ in self.tail_logs(challenge_id, submission_id,
                                        poll_sec=poll_sec,
                                        timeout_sec=timeout_sec):
                    pass
                final = self.submission(submission_id)
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
                    raise SubmissionError(reason, body=final)
                result["status"] = final
            return result
        finally:
            if tmp_dir:
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)

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
                    aggregate: str | None = None, course_id: int | None = None,
                    me: bool = False, q: str | None = None,
                    window: int | None = None, offset: int | None = None,
                    around: int | None = None):
        """Get the leaderboard for a challenge.

        Mirrors `GET /api/leaderboard/challenge/{id}` (`leaderboard.py`) with
        the query keys the console sends — `limit` (here `top`), `offset`,
        `aggregate`, `course_id`, `me`, `around`, `q`, `window` — each sent
        only when passed.

        - ``aggregate="user"`` (the default, as on the console): one row per
          participant — their best active submission, a team's best for a
          team, ranked among participants. ``"submission"``: every active
          submission is a row.
        - ``course_id``: read the board through a course. Only that course's
          students are listed, rows gain ``passed`` when the course sets a bar
          (tri-state: None while the row has no ranked value), and the
          envelope carries a ``course_context`` block (`course_id`,
          `pass_threshold`, `names_withheld`). Names go to the course's
          members only: unless you are enrolled in the course, own it,
          assist it or are an admin, ``course_context["names_withheld"]`` is
          True and every row comes with its rank, score and metrics but
          with `username`, `avatar_key`, `submission_id`,
          `submission_name`, `team_id`, `team_name`, `team_members` and
          `datagouv_reuse` set to None and `can_open` False (the keys
          stay). On such a read ``q`` and ``around`` are refused
          (`MLArenaError`, 400): each would say whether a given person is
          in the course. The board without ``course_id`` is unchanged.
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
        `leaders`, `me`, `around`, `matches` and, with ``course_id``,
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
        ``course_context["names_withheld"]`` (see ``course_id`` above). With pandas, each row's `metrics`
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
        if course_id is not None:
            params["course_id"] = course_id
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
        # can identify: `is_my_submission`, and non-public course challenges the
        # caller is enrolled in. Without the bearer token every row came back
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

        Every hour, each challenge that awards points & medals ranks
        its participants (a team, or a user on their own) by their
        best active submission, as its leaderboard does. Ranks 1, 2
        and 3 earn a gold, silver and bronze medal. Every rank earns
        100000 × rank^-0.75 × log10(1 + log10(N)) ÷ √(team size)
        points, where N is the number of ranked participants. Each
        team member receives the team's medal and points. When the
        challenge stops, its last standings stay. Your total is the
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

    # ---- Chat challenges (`chat_v1`): one shared JSON-call helper -----------

    def _chat_call(self, method: str, path: str, *,
                   json_body: dict | None = None,
                   params: dict | None = None,
                   ok: tuple[int, ...] = (200,),
                   error_label: str,
                   not_found: type = ChallengeNotFoundError,
                   timeout: int = 30) -> requests.Response:
        """Single chokepoint for the `/api/chat/*` routes (`_course_call`'s
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
        `env_status` (the error text behind an `"error"` is creator-only:
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
        `max_sessions_per_participant` is reached, or the agent is still
        answering a message of your open session — wait for it first).
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
        the raw text behind it, served to the challenge's creator /
        assistants and admins only (None for everyone else); None before the
        first message), `participant` (your group's `total_amount_eur` and
        `scoreboard`, the same keys `chat_challenge()` serves) and `can_send`
        with `can_send_reason` (`session_voided` | `session_closed` |
        `turn_in_flight` | `challenge_not_started` | `challenge_stopped` |
        `challenge_ended` | `agent_offline` |
        `turn_limit_reached`, None exactly when `can_send` is true). Readable
        by the session's user, their teammates, the challenge's creator /
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
        `MLArenaError` with the server's reason when it refuses (409: a turn
        is already in flight on this session, the session is not `open`, or
        `max_turns_per_session` is reached).
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

    # ---- Chat challenges: creator routes (creator scope + challenge access) --

    def chat_admin(self, challenge_id: int) -> dict:
        """The creator view of a chat challenge — `ChatChallengeAdminView`.

        Mirrors `GET /api/chat/challenge/{cid}/admin`. Returns the
        LLM connection (`llm_base_url`, `llm_model`, `llm_api_key_set` — a
        boolean, the key itself is never served), the limits, the agent's
        health (`env_status`, `env_status_message`, `env_status_at_ts`,
        `worker_ref`, `last_seen_at_ts`, `agent_online`), the registered
        `manifest` and `counts` (`sessions`, `open_sessions`, `turns`,
        `scoring_events`, `total_amount_eur`, `participants`).

        Requires a `creator`-scope token and access to the challenge.
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
        """Every session of the challenge, for the creator.

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
        | `error`: the checker's load of the creator's `env.py`, self-test
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
        challenge is a draft (not started) or closed (stopped or ended); 429
        over the challenge's per-minute attempt cap (`.body
        ["max_attempts_per_minute"]`, per group, last 60 s); 400 for `code`
        on a text / choice exercise, `answer` on a code one, or a `choice`
        answer that is not one of its `choices`.
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

        Readable by the attempt's group, the challenge's creator / assistants,
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
        the challenge's creator / assistants, admins and the teachers of a
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
        """The creator view of an exercise set — `ExerciseChallengeAdminOut`.

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

        Readable by the challenge's creator / assistants and admins; anyone
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
        the format. For whoever may edit the set, and for the owner and
        assistants of a course that links it; `ChallengeNotFoundError` (404)
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
        (None, `started` or `env_py`) and `can_write_code_exercises` (whether
        you may write `code` exercises, whose `check` is Python the checker
        runs: creator access, for now; `text` and `choice` exercises need
        none).

        Each exercise has `key`, `kind`, `title` (the prompt), `points`
        (0 = practice) and `examples`, then by kind — `code`: `starter`,
        `check` (Python run with `student` and `forbid` defined; it passes
        when it raises nothing), `solution`, `wrong`; `text`: `accept` (the
        accepted answers), `case_sensitive`, `wrong`; `choice`: `choices`,
        `solution`. The format is documented in the platform's
        `workers/exercise_v1/PROCESS.md` ("exercises.json").

        For whoever may edit the set (owner, assistants, admins), with a
        session or a creator / teacher key; anyone else gets
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

        Raises `PermissionDeniedError` (403) when the list has a `code`
        exercise and you have no creator access ("Exercises checked by
        Python code need creator access for now; short-answer and
        multiple-choice exercises do not");
        `MLArenaError` (409, `.body["write_blocked_by"]`) while the set
        is started (`started`: stop it first) or when its hand-written env.py
        does not load `exercises.json` (`env_py`); `MLArenaError` (400) for a
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

        Raises `PermissionDeniedError` (403) when the set holds Python
        checks (a `code` exercise, a hand-written env.py) and you have no
        creator access (the same holds for `start_challenge`); `MLArenaError`
        with the server's reason: 409 on a started set
        (its checker self-tests at load: stop the set to self-test an edit),
        on an ended one (restart it), and while a self-test of this set is
        already pending (one at a time; it settles or times out after 10
        minutes); the start gate's refusal, since a self-test starts a
        checker where a start would (400 for authoring still to do, such as
        an `exercises.json` that is missing, invalid or empty; 409 when the
        set cannot run on its machine; `.body["start_blockers"]` lists them);
        429 when you already have 2 self-tests pending, or the platform 4
        (`.body["limit"]`), and, without creator access, within 60 s of your
        previous self-test request (`.body["retry_after_sec"]`).
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

    # ---- Course content: shared JSON-call helper ----------------------------

    def _course_call(self, method: str, path: str, *,
                     json_body: dict | None = None,
                     params: dict | None = None,
                     ok: tuple[int, ...] = (200,),
                     error_label: str = "request",
                     timeout: int = 30) -> dict | list:
        """Single chokepoint for the course-content (`/teacher`, `/academic_courses`)
        JSON routes — mirrors the per-method boilerplate used above but keeps the
        ~30 course methods DRY.

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

    # ---- Academic courses: create / enroll / list (teacher + user scope) ----

    def create_course(self, name: str, code: str | None = None,
                      start_date: str | None = None, end_date: str | None = None,
                      instructor_name: str | None = None,
                      *,
                      slug: str | None = None,
                      description: str | None = None,
                      visibility: str | None = None,
                      teacher_user_id: int | None = None) -> dict:
        """Create an academic course — this is how you become a teacher.

        Mirrors `POST /api/academic_courses/` (`academic_courses/courses.py`).
        Creating a course is the self-serve teacher-onboarding action: it works
        with any valid token scope and flips the owning account to a teacher.
        Each account may own at most `max_courses_limit` active (not
        archived) courses (default 3); admins are exempt. At the limit the 403
        states both numbers; `archive_course` or `delete_course` frees a place. After creating, mint a `teacher`-scope key from your
        Profile page to author modules/lessons and attach challenges.

        `start_date` / `end_date` are optional ISO date strings (`YYYY-MM-DD`) —
        a name is the only required field, so a course can be created without
        term dates (a course with no end date never closes). They are kept
        positional for backwards compatibility with the original signature. The
        backend mints the course's `join_code` — the single enrollment token —
        and returns it in the course dict; share that, or the
        `/enroll/<join_code>` link built from it, with students.

        Course-content fields (added with the course-content model):
            slug:        kebab URL slug (`^[a-z0-9-]+$`); auto-derived from the
                         name when omitted.
            description: markdown landing-page intro.
            visibility:  `private` (default) | `unlisted` | `public`. Only
                         `public` courses appear in `course_catalog()`.
            teacher_user_id: an admin may name the owning teacher; otherwise the
                         caller owns the course.
        Challenges are attached afterwards: one step per challenge with
        `add_course_challenge`, or through modules you build and link (see
        `create_module` / `attach_challenge` / `link_module`).
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
        if teacher_user_id is not None:
            body["teacher_user_id"] = teacher_user_id
        return self._course_call(
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
        return self._course_call(
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
        a course setting). An archived or ended course answers 410. Returns
        `{message, challenge_ids, course_slug}`.
        """
        if join_code is None:
            raise MLArenaError("enroll_in_course requires a join code")
        body: dict = {}
        if student_email is not None:
            body["student_email"] = student_email
        if student_number is not None:
            body["student_number"] = student_number
        return self._course_call(
            "POST", f"/academic_courses/enroll/{join_code}", json_body=body,
            ok=(200, 201), error_label="enroll_in_course",
        )

    def list_courses(self, *, show_all: bool = False,
                     challenge_id: int | None = None):
        """List courses (enrolled + currently-active by default).

        Mirrors `GET /api/academic_courses/`. With `show_all=True` returns every
        course; with `challenge_id` returns courses attached to that
        challenge. Each row carries `is_enrolled` for the caller and
        `enrollment_state`: `open`, or why the course takes no new enrollment
        (`archived`, `ended`: its end date is past), decided by the server
        with the rule `enroll_in_course` refuses by. A private or
        unlisted course is listed only to its members and managers, and
        `join_code` is set only on the courses the caller belongs to or manages
        (None elsewhere). An enrolled row also carries the caller's
        `progress`, `next_lesson`, `progress_state` and `last_active_at_ts`
        (None on the others). Returns a
        DataFrame if pandas is installed, else a list of dicts.
        """
        params: dict = {}
        if show_all:
            params["show_all"] = "true"
        if challenge_id is not None:
            params["challenge_id"] = challenge_id
        return _to_dataframe(self._course_call(
            "GET", "/academic_courses/", params=params or None,
            error_label="list_courses",
        ))

    # ---- Course consumption (learner-facing reads + progress) ---------------

    def course_catalog(self, search: str | None = None,
                       limit: int | None = None, offset: int | None = None) -> dict:
        """List public courses (search / paginate). Public read.

        Mirrors `GET /api/academic_courses/catalog`. Returns
        `{courses, total, limit, offset}` — only `visibility=public` courses
        appear. `search` matches name + description. A card carries `has_cover`;
        fetch the image with `course_cover(card["id"])`.
        """
        params: dict = {}
        if search is not None:
            params["search"] = search
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset
        return self._course_call(
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
        `archived` | `ended`) says whether the course takes a new enrollment.
        `show_module_headings` is False when you are served a single module:
        the course is then its content, with no module heading. Each lesson
        of the TOC carries `visible_to_students` (it is published and so is
        its module), next to its own `is_published`.

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
        return self._course_call(
            "GET", f"/academic_courses/{slug}", params=_as_student(as_student),
            error_label="course",
        )

    def module_overview(self, slug: str, module_slug: str,
                        as_student: bool = False) -> dict:
        """A course module's overview: lesson list + attached challenge cards.

        Mirrors `GET /api/academic_courses/{slug}/modules/{module_slug}`.
        `as_student=True`: as on `course` (a module hidden from students is
        then a 404).
        """
        return self._course_call(
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
        returned dict carries `body_md`, `directives`, and `directive_warnings`.
        A directive naming a challenge you may not see is left out, with the
        warning of a challenge that does not exist. A `challenge` /
        `competition` directive's payload is `{challenge_id, challenge,
        description}`, `challenge` being the challenge's reference (see
        `challenge_overview`). `as_student=True`: as on `course` (a lesson
        hidden from students is then a 404, and a block whose challenge the
        course's students do not see is left out, with its warning).
        """
        return self._course_call(
            "GET",
            f"/academic_courses/{slug}/modules/{module_slug}/lessons/{lesson_slug}",
            params=_as_student(as_student), error_label="lesson",
        )

    def mark_lesson_viewed(self, lesson_id: int,
                           course_id: int | None = None) -> dict:
        """Mark a lesson in-progress + bump last-viewed/last-active (idempotent).

        Mirrors `POST /api/academic_courses/lessons/{id}/view`. Pass `course_id`
        when the lesson's module is reused by more than one course you're
        enrolled in (the backend fails loud on ambiguity otherwise).
        """
        body = {"course_id": course_id} if course_id is not None else {}
        return self._course_call(
            "POST", f"/academic_courses/lessons/{lesson_id}/view",
            json_body=body, error_label="mark_lesson_viewed",
        )

    def mark_lesson_complete(self, lesson_id: int,
                             course_id: int | None = None) -> dict:
        """Mark a lesson completed (upsert progress, bump last-active).

        Mirrors `POST /api/academic_courses/lessons/{id}/complete`. See
        `mark_lesson_viewed` for the `course_id` disambiguation rule.
        """
        body = {"course_id": course_id} if course_id is not None else {}
        return self._course_call(
            "POST", f"/academic_courses/lessons/{lesson_id}/complete",
            json_body=body, error_label="mark_lesson_complete",
        )

    def mark_lesson_incomplete(self, lesson_id: int,
                               course_id: int | None = None) -> dict:
        """Undo a completion — back to in-progress, `completed_at_ts` cleared.

        Mirrors `POST /api/academic_courses/lessons/{id}/uncomplete`. The tick
        is student-self-reported, so it is reversible. See `mark_lesson_viewed`
        for the `course_id` disambiguation rule.
        """
        body = {"course_id": course_id} if course_id is not None else {}
        return self._course_call(
            "POST", f"/academic_courses/lessons/{lesson_id}/uncomplete",
            json_body=body, error_label="mark_lesson_incomplete",
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
        return self._course_call(
            "GET", f"/academic_courses/{course_id}/progress/me",
            error_label="my_progress",
        )

    # ---- Course authoring: modules (teacher scope) --------------------------

    def create_module(self, title: str, slug: str | None = None,
                      summary: str | None = None, icon: str | None = None,
                      visibility: str = "private",
                      is_published: bool = True) -> dict:
        """Create an owned, reusable content module. Requires a `teacher` token.

        Mirrors `POST /api/teacher/modules`. `slug` (kebab, `^[a-z0-9-]+$`) is
        auto-derived from the title when omitted and is unique per owner.

        Two independent flags, easy to confuse:

        * `visibility` — *library sharing*: which other **teachers** may
          link/fork it. `private` (default) | `unlisted` | `public`; only
          `public` modules are forkable/linkable by other teachers.
        * `is_published` — the *student* gate. `False` makes the module a draft:
          it stays linked to its courses but disappears from every learner
          surface — lesson list, lesson bodies, attached challenges, progress
          denominators — until you publish it. Use it to keep a section you are
          still writing hidden inside an already-shared course.
        """
        body: dict = {
            "title": title, "visibility": visibility, "is_published": is_published,
        }
        if slug is not None:
            body["slug"] = slug
        if summary is not None:
            body["summary"] = summary
        if icon is not None:
            body["icon"] = icon
        return self._course_call(
            "POST", "/teacher/modules", json_body=body,
            ok=(200, 201), error_label="create_module",
        )

    def list_modules(self, library: str | None = None) -> list:
        """List the caller's own modules, or — with `library="public"` — other
        teachers' public modules for reuse browsing.

        Mirrors `GET /api/teacher/modules` (`?library=public`).
        """
        params = {"library": library} if library is not None else None
        return self._course_call(
            "GET", "/teacher/modules", params=params, error_label="list_modules",
        )

    def get_module(self, module_id: int) -> dict:
        """Module detail + lesson TOC + challenge links.

        Mirrors `GET /api/teacher/modules/{id}`. For a module you may read as
        an author: yours, a public one, or one linked into a course you are
        the owner or an assistant of. `can_edit` says whether you may edit it
        (see `update_module`), `is_owned` whether you hold its owner's rights
        (delete, `visibility`, `detach_module`).
        """
        return self._course_call(
            "GET", f"/teacher/modules/{module_id}", error_label="get_module",
        )

    def update_module(self, module_id: int, **fields) -> dict:
        """Update module meta. Mirrors `PUT /api/teacher/modules/{id}`.

        A module is edited by its owner, an admin, and the assistants of a
        course that links it and that the module's owner owns: a course's
        assistants edit the modules, lessons, assets and challenge
        attachments of the course's owner (`can_edit` on the module
        payloads). What is created from inside a course
        (`create_course_module`, `add_course_challenge`,
        `copy_module_into_course`, `push_course`) belongs to the course's
        owner, whoever of its staff creates it, so the owner edits what an
        assistant added. A module a course only borrows (another teacher's
        public module, or a module an assistant linked from their own
        library with `link_module`) is read-only there:
        `copy_module_into_course` it. `visibility` (library sharing) is the
        owner's: a 403 for another editor.

        Accepts `title`, `slug`, `summary`, `icon`, `visibility`,
        `is_published`. Only the fields you pass are changed, so
        `update_module(id, is_published=False)` sends a module back to draft
        without touching anything else. A new `slug` changes the module's
        URLs; its lessons and their progress are kept. A module managed by a
        bundle (`content_origin == "bundle"`) refuses edits with a 409 naming
        its source: change the files and push again, or `detach_module`.
        """
        allowed = {"title", "slug", "summary", "icon", "visibility",
                   "is_published"}
        body = _filtered_fields(fields, allowed, "update_module")
        return self._course_call(
            "PUT", f"/teacher/modules/{module_id}", json_body=body,
            error_label="update_module",
        )

    def delete_module(self, module_id: int, force: bool = False) -> dict:
        """Delete a module — its owner or an admin (403 for an editor who is
        neither: removing it from a course is `unlink_module`).

        Mirrors `DELETE /api/teacher/modules/{id}`. Blocked (409) if any course
        links the module; pass `force=True` (`?force`) to detach those links
        first and delete anyway.
        """
        params = {"force": "1"} if force else None
        return self._course_call(
            "DELETE", f"/teacher/modules/{module_id}", params=params,
            error_label="delete_module",
        )

    def fork_module(self, module_id: int) -> dict:
        """Deep-copy a module into a new module you own: a public module, or
        one of yours (an admin: any). A module you only read as the staff of
        a course is a 403 (the fork would be yours to share); to reuse it in
        another course of its owner, use `copy_module_into_course`.

        Mirrors `POST /api/teacher/modules/{id}/fork`. Copies lessons (nesting
        preserved) + challenge attachments; the fork starts `private` and
        records provenance via `forked_from_module_id`. An attachment of a
        challenge you may not attach yourself (`attach_challenge` would
        refuse it: not public, not yours, not one you assist) is left out;
        the answer names them in `dropped_challenges`
        (`[{challenge_id, name}]`, `name` None when you may not see it).
        """
        return self._course_call(
            "POST", f"/teacher/modules/{module_id}/fork",
            ok=(200, 201), error_label="fork_module",
        )

    def detach_module(self, module_id: int) -> dict:
        """Hand a bundle-managed module back to the editor — its owner or an
        admin (403 for an editor who is neither).

        Mirrors `POST /api/teacher/modules/{id}/detach`. A module last
        written by a course bundle (`push_course`, `mlarena course push`) has
        `content_origin == "bundle"` and refuses editor writes; detaching
        sets it to `editor`. The next push of a bundle that names the module
        takes it over again. Returns the module detail.
        """
        return self._course_call(
            "POST", f"/teacher/modules/{module_id}/detach",
            error_label="detach_module",
        )

    def attach_challenge(self, module_id: int, challenge_id: int,
                           label: str | None = None,
                           position: int | None = None,
                           pass_threshold: float | None = None) -> dict:
        """Attach a challenge to a module.

        Mirrors `POST /api/teacher/modules/{id}/challenges`. `position`
        defaults to the end of the module's challenge list.

        `pass_threshold` is the course's validation bar: a student validates the
        challenge when their best leaderboard value meets it in the direction
        the challenge ranks (`>=` when its ranking spec's `order` is "desc",
        `<=` when it is "asc"; the returned link carries that spec as
        `metric`). Omit it for the kind's default: 70 (% solved) for an
        exercise set, no pass/fail for every other kind
        (`challenges_for_course` lists it as `default_pass_threshold`). The
        default is applied when the link is created only: on a link that
        already exists (`update_challenge_link`, a `course.yaml` re-push) an
        omitted `pass_threshold` leaves the stored bar unchanged. To
        remove a bar, call `update_challenge_link(..., pass_threshold=None)`.
        Do not pass 0, which would validate every entrant.
        """
        body: dict = {"challenge_id": challenge_id}
        if label is not None:
            body["label"] = label
        if position is not None:
            body["position"] = position
        if pass_threshold is not None:
            body["pass_threshold"] = pass_threshold
        return self._course_call(
            "POST", f"/teacher/modules/{module_id}/challenges", json_body=body,
            ok=(200, 201), error_label="attach_challenge",
        )

    def update_challenge_link(self, module_id: int, challenge_id: int,
                                **fields) -> dict:
        """Update an existing attachment's `label` / `pass_threshold`.

        Mirrors `PUT /api/teacher/modules/{id}/challenges/{challenge_id}`.
        Partial: only the keys you pass are written, and an explicit ``None``
        clears that field — so ``update_challenge_link(m, c,
        pass_threshold=None)`` removes the bar, while omitting the key leaves
        it alone.
        """
        return self._course_call(
            "PUT",
            f"/teacher/modules/{module_id}/challenges/{challenge_id}",
            json_body=fields, error_label="update_challenge_link",
        )

    def detach_challenge(self, module_id: int, challenge_id: int) -> dict:
        """Detach a challenge from a module.

        Mirrors `DELETE /api/teacher/modules/{id}/challenges/{challenge_id}`.
        """
        return self._course_call(
            "DELETE",
            f"/teacher/modules/{module_id}/challenges/{challenge_id}",
            error_label="detach_challenge",
        )

    def reorder_module_challenges(self, module_id: int,
                                    ordered_challenge_ids: list[int]) -> list:
        """Reorder a module's attached challenges.

        Mirrors `PUT /api/teacher/modules/{id}/challenges/reorder`.
        `ordered_challenge_ids` must be exactly the module's attached
        challenge ids in the desired order.
        """
        return self._course_call(
            "PUT", f"/teacher/modules/{module_id}/challenges/reorder",
            json_body={"ordered_ids": ordered_challenge_ids},
            error_label="reorder_module_challenges",
        )

    # ---- Course authoring: lessons (teacher scope) --------------------------

    def create_lesson(self, module_id: int, title: str, kind: str = "lesson",
                      slug: str | None = None,
                      parent_lesson_id: int | None = None,
                      body_md: str = "", gated: bool = False) -> dict:
        """Create a lesson inside an owned module (starts as an unpublished draft).

        Mirrors `POST /api/teacher/modules/{id}/lessons`. `kind` is `lesson`
        (default) | `exercise`. `parent_lesson_id` nests it one level under
        another lesson in the same module. Publish it later via
        `update_lesson(id, is_published=True)`.
        """
        body: dict = {"title": title, "kind": kind, "body_md": body_md,
                      "gated": gated}
        if slug is not None:
            body["slug"] = slug
        if parent_lesson_id is not None:
            body["parent_lesson_id"] = parent_lesson_id
        return self._course_call(
            "POST", f"/teacher/modules/{module_id}/lessons", json_body=body,
            ok=(200, 201), error_label="create_lesson",
        )

    def get_lesson(self, lesson_id: int) -> dict:
        """Full lesson detail incl. body — owner only.

        Mirrors `GET /api/teacher/lessons/{id}` (the authoring view; use
        `lesson(slug, module_slug, lesson_slug)` for the learner view with
        resolved directives).
        """
        return self._course_call(
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
        return self._course_call(
            "PUT", f"/teacher/lessons/{lesson_id}", json_body=body,
            error_label="update_lesson",
        )

    def delete_lesson(self, lesson_id: int) -> dict:
        """Delete a lesson and its descendants — owner only.

        Mirrors `DELETE /api/teacher/lessons/{id}`.
        """
        return self._course_call(
            "DELETE", f"/teacher/lessons/{lesson_id}", error_label="delete_lesson",
        )

    def reorder_lessons(self, module_id: int,
                        ordered_lesson_ids: list[int]) -> list:
        """Reorder a module's lessons.

        Mirrors `PUT /api/teacher/modules/{id}/lessons/reorder`.
        `ordered_lesson_ids` must be exactly this module's lesson ids in the
        desired display order (`position` becomes the index).
        """
        return self._course_call(
            "PUT", f"/teacher/modules/{module_id}/lessons/reorder",
            json_body={"ordered_ids": ordered_lesson_ids},
            error_label="reorder_lessons",
        )

    # ---- Course authoring: module assets (teacher scope) --------------------
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
        return self._course_call(
            "GET", f"/teacher/modules/{module_id}/assets",
            error_label="list_module_assets",
        )

    def delete_module_asset(self, module_id: int, path: str) -> dict:
        """Delete one file of a module's asset store, by its `path`.

        Mirrors `DELETE /api/teacher/modules/{id}/assets/{path}`.
        """
        return self._course_call(
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
        caught before publishing; a challenge you may not see is refused as
        one that does not exist. Returns `{directives, warnings}`: the
        payloads as you would read them, and one warning per block and per
        course linking the lesson's module whose students would not see the
        block's challenge (it is private and that course does not assign it).
        """
        body = {"body_md": body_md} if body_md is not None else {}
        return self._course_call(
            "POST", f"/teacher/lessons/{lesson_id}/preview", json_body=body,
            error_label="preview_lesson",
        )

    # ---- Course authoring: course composition (teacher scope) ---------------

    def update_course(self, course_id: int, **fields) -> dict:
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
        pass are changed. Returns the course as `teacher_course(course_id)`
        does.
        """
        allowed = {"name", "code", "description", "visibility",
                   "instructor_name", "start_date", "end_date",
                   "ask_student_identity", "report_recipients",
                   "report_to_assistants"}
        body = _filtered_fields(fields, allowed, "update_course")
        return self._course_call(
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
        """List a course's linked modules in order, each with your rights on
        it (`can_edit` and `is_owned`, see `get_module`), its `lessons` (the
        lesson summaries of `get_module`, in order, each with
        `visible_to_students`) and its `challenges`.

        Mirrors `GET /api/teacher/course/{id}/modules`.
        """
        return self._course_call(
            "GET", f"/teacher/course/{course_id}/modules",
            error_label="list_course_modules",
        )

    def link_module(self, course_id: int, module_id: int,
                    position: int | None = None) -> dict:
        """Link a module into a course as a live reference.

        Mirrors `POST /api/teacher/course/{id}/modules`. Only owned or public
        modules are linkable. `position` defaults to the end.
        """
        body: dict = {"module_id": module_id}
        if position is not None:
            body["position"] = position
        return self._course_call(
            "POST", f"/teacher/course/{course_id}/modules", json_body=body,
            ok=(200, 201), error_label="link_module",
        )

    def create_course_module(self, course_id: int, title: str, *,
                             slug: str | None = None,
                             summary: str | None = None,
                             icon: str | None = None,
                             visibility: str | None = None,
                             is_published: bool | None = None) -> dict:
        """Create a module in a course: `create_module` and `link_module` in
        one call, with one difference that matters for a team.

        Mirrors `POST /api/teacher/course/{id}/modules/create` (the body of
        `create_module`). The module belongs to the course's owner, whoever
        of its staff creates it: created by an assistant, it is edited by the
        owner and by the assistant alike; `visibility` (library sharing) is
        then the owner's to set, so an assistant creates it `private` (any
        other value is a 403). (`create_module` followed by
        `link_module` makes a module of your own library: in a course you
        only assist, its owner could not edit it.) Linked at the end of the
        course, with a slug no module of the course uses. Returns the
        course's new module row (the `list_course_modules` shape). Owner,
        assistant or admin of the course.
        """
        body: dict = {"title": title}
        for key, value in (("slug", slug), ("summary", summary), ("icon", icon),
                           ("visibility", visibility), ("is_published", is_published)):
            if value is not None:
                body[key] = value
        return self._course_call(
            "POST", f"/teacher/course/{course_id}/modules/create", json_body=body,
            ok=(201,), error_label="create_course_module",
        )

    def copy_module_into_course(self, course_id: int, module_id: int) -> dict:
        """Copy a module from another course into this one.

        Mirrors `POST /api/teacher/course/{id}/modules/copy`. Deep-copies a
        module you may read as an author (yours, a public one, or one linked
        into a course you are the owner or an assistant of; 404 otherwise).
        Since the copy belongs to the target course's owner, a module that is
        not public is copied only by its owner (or an admin) or into a course
        of its owner (403 otherwise; `fork_module` asks the same question).
        What is copied:
        lessons, challenge attachments with their pass bars and assets, as
        `fork_module` does (and with its `dropped_challenges`: an attachment
        of a challenge you may not attach yourself is left out and named).
        The copy belongs to the course's owner (yours in your own course; in
        a course you assist, the owner's, and you edit it as its assistant),
        `private`, editor-written, linked at the end of the course, with a
        slug no module of the course uses; the source is untouched. Returns
        the course's new module row (the `list_course_modules` shape) plus
        `dropped_challenges`. Owner, assistant or admin of the target course.
        """
        return self._course_call(
            "POST", f"/teacher/course/{course_id}/modules/copy",
            json_body={"module_id": module_id},
            ok=(201,), error_label="copy_module_into_course",
        )

    def list_copyable_modules(self, course_id: int) -> list:
        """The modules `copy_module_into_course(course_id, …)` is offered:
        those of the other courses you are the owner or an assistant of
        (archived ones included) that the copy accepts into this course
        (public, yours, or owned by this course's owner), by course name then
        module order.

        Mirrors `GET /api/teacher/course/{id}/copyable-modules`. Each row is
        `{module_id, title, summary, lesson_count, challenge_count,
        course: {id, name}}`; a module two of your courses link is listed
        under each. A module outside the list (a public one) is still copied
        by id.
        """
        return self._course_call(
            "GET", f"/teacher/course/{course_id}/copyable-modules",
            error_label="list_copyable_modules",
        )

    def add_course_challenge(self, course_id: int, challenge_id: int, *,
                             title: str | None = None,
                             summary: str | None = None,
                             pass_threshold: float | None = None,
                             is_published: bool = True) -> dict:
        """Add a challenge to a course as one entry, in one transaction.

        Mirrors `POST /api/teacher/course/{id}/challenges` — the console's
        Challenges-tab "Add challenge". The server creates a private module
        (`title`, default the challenge's name; `summary` = the markdown page
        students see), links it at the end of the course and attaches the
        challenge with `pass_threshold` (see `attach_challenge`); either all
        three happen or none does. `is_published=False` stages it as a draft.
        Returns the course's new module row (the shape `list_course_modules`
        returns). Requires teacher/TA/admin on the course.
        """
        body: dict = {"challenge_id": challenge_id, "is_published": is_published}
        if title is not None:
            body["title"] = title
        if summary is not None:
            body["summary"] = summary
        if pass_threshold is not None:
            body["pass_threshold"] = pass_threshold
        return self._course_call(
            "POST", f"/teacher/course/{course_id}/challenges", json_body=body,
            ok=(200, 201), error_label="add_course_challenge",
        )

    def use_challenge_with_class(self, challenge: int | str, course: int | None = None, *,
                                 name: str | None = None,
                                 end_date: str | None = None) -> dict:
        """Run a challenge with your class: put it in a course, get the join code.

        The console's "Use with my class" (the Studio's New › "Run a challenge
        with my class"), composed client-side from the public routes; no
        route of its own. `challenge` is the challenge's id or its slug (the
        `/challenges/<slug>` of its page; a slug is resolved with
        `challenge_by_slug`).

        - `course` given: the challenge is added to that course of yours
          (`add_course_challenge`), then the course is read back
          (`teacher_course`). `name` and `end_date` describe a new course and
          are refused with it (`TypeError`).
        - `course` None: a new course is created (`create_course`), named
          `name` or, without one, after the challenge, with the optional
          `end_date` (`YYYY-MM-DD`; a course without one never closes), and
          the challenge is added to it. Creating a course is self-serve: any
          token scope, and it makes the account a teacher (up to
          `max_courses_limit` active courses).

        The entry gets the route's defaults: titled after the challenge,
        visible to students, the kind's default pass bar (70 for an exercise
        set, none otherwise); `add_course_challenge` sets them otherwise. The
        console offers only a started or draft challenge; the server accepts
        any challenge you may reuse (public, yours, or one you assist on).
        If the course is created but the challenge is refused, the course
        stays and the refusal is raised.

        Returns the course as `create_course` / `teacher_course` serve it:
        its `join_code` is what students enroll with (`enroll_in_course`, or
        the console's `/enroll/<join_code>` page).
        """
        if course is not None and (name is not None or end_date is not None):
            raise TypeError(
                "use_challenge_with_class: name and end_date describe a new course; "
                "pass course= or them, not both"
            )
        if isinstance(challenge, str):
            detail = self.challenge_by_slug(challenge)
            challenge_id = detail["id"]
        else:
            detail = None
            challenge_id = challenge
        if course is not None:
            self.add_course_challenge(course, challenge_id)
            return self.teacher_course(course)
        if name is None:
            name = (detail if detail is not None else self.challenge(challenge_id))["name"]
        created = self.create_course(name, end_date=end_date)
        self.add_course_challenge(created["id"], challenge_id)
        return created

    def add_course_exercise_set(self, course_id: int, module_id: int, name: str, *,
                                label: str | None = None,
                                pass_threshold: float | None | _Unset = _UNSET) -> dict:
        """Create an exercise set in a course's module, in one transaction.

        Mirrors `POST /api/teacher/course/{id}/modules/{module_id}/exercise-sets`
        — the console's "Add exercises". The server creates an exercise set
        (a private draft named `name`, a challenge name: unique on the
        platform) holding no exercise yet. It belongs to the course's owner;
        the course's assistants edit, self-test, start and stop it through
        the course. The server links it into the module
        with `pass_threshold`, the percentage of points to solve: left out,
        the kind's default (70); `None`, not graded. Returns the new link
        (`ModuleChallengeLinkOut`): `challenge_id` is the set. Then write its
        exercises with `update_exercise_definition`, check them with
        `self_test_exercises`, and start it.

        Requires teacher/TA/admin on the course and a module of that course
        (`NotFoundError` otherwise); a taken name raises `MLArenaError` (409).
        """
        body: dict = {"name": name}
        if label is not None:
            body["label"] = label
        if pass_threshold is not _UNSET:
            body["pass_threshold"] = pass_threshold
        return self._course_call(
            "POST", f"/teacher/course/{course_id}/modules/{module_id}/exercise-sets",
            json_body=body, ok=(201,), error_label="add_course_exercise_set",
        )

    def remove_course_challenge(self, course_id: int, module_id: int) -> dict:
        """Remove a challenge entry from a course, in one transaction.

        Mirrors `DELETE /api/teacher/course/{id}/challenges/{module_id}` — the
        console's Challenges-tab "Remove". The entry's module is always
        unlinked from the course; it is also deleted when it is a simple
        module (no lessons, at most one challenge) you may edit and no other
        course links it — the wrapper `add_course_challenge` created. An
        advanced or borrowed module is only unlinked. Returns
        `{"message", "module_deleted"}`. Requires teacher/TA/admin on the
        course; a module not linked to it raises `NotFoundError` (404).
        """
        return self._course_call(
            "DELETE", f"/teacher/course/{course_id}/challenges/{module_id}",
            error_label="remove_course_challenge",
        )

    def unlink_module(self, course_id: int, module_id: int) -> dict:
        """Unlink a module from a course (the module itself is untouched).

        Mirrors `DELETE /api/teacher/course/{id}/modules/{module_id}`.
        """
        return self._course_call(
            "DELETE", f"/teacher/course/{course_id}/modules/{module_id}",
            error_label="unlink_module",
        )

    def reorder_modules(self, course_id: int,
                        ordered_module_ids: list[int]) -> list:
        """Reorder a course's linked modules.

        Mirrors `PUT /api/teacher/course/{id}/modules/reorder`.
        `ordered_module_ids` must be exactly this course's linked module ids.
        """
        return self._course_call(
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
        return self._course_call(
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
        `teacher` key it always needed (a `creator` key reads it too).
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
        `archived_at_ts`, `enrollment_state`, `is_enrolled`) plus `role`
        (what you are to it: `owner`, `assistant`, or None for an admin who
        is neither), `can_edit` (true: you may open its workspace),
        `has_owner_rights` (whether you may do what only its owner may: the
        assistant roster, archive, duplicate, delete — the owner and an
        admin), `owner_username`, `created_at_ts`, the settings
        (`ask_student_identity`, `report_recipients`, `report_to_assistants`,
        `report_last_sent_at_ts`) and the counts `enrollment_count`,
        `assistant_count`, `module_count`, `challenge_count` (entries hidden
        from students included), `show_module_headings` (False for a course
        with a single module: its entries are then the course's, with no
        module heading) and `exercise_default_pass_threshold` (the pass bar
        `add_course_exercise_set` applies when you name none). It also carries what `workspace()` follows
        about the course, under the same names and from the same computation:
        `active_enrollment_count_7d`, `last_active_at_ts` and
        `enrolled_without_submission` (served while the course is not
        archived, ends within 7 days and assigns a challenge, else None).
        Owner, assistant or admin; 403 otherwise.
        """
        return self._course_call(
            "GET", f"/teacher/course/{course_id}", error_label="teacher_course",
        )

    def archive_course(self, course_id: int) -> dict:
        """Archive a course — its owner or an admin.

        Mirrors `POST /api/teacher/course/{id}/archive`. An archived course
        takes no enrollment (its join code answers 410), leaves the catalog and
        the listing of non-members, is skipped by the end-of-course report and
        no longer counts in your limit of active courses. Nothing is deleted:
        its students keep reading it and you keep following it. Returns the
        course (`archived_at_ts` set); 409 when it is already archived.
        """
        return self._course_call(
            "POST", f"/teacher/course/{course_id}/archive",
            error_label="archive_course",
        )

    def unarchive_course(self, course_id: int) -> dict:
        """Take a course out of the archive — its owner or an admin.

        Mirrors `POST /api/teacher/course/{id}/unarchive`. The course counts
        in its owner's limit again: 403 when the owner has no place left, 409
        when the course is not archived. Returns the course.
        """
        return self._course_call(
            "POST", f"/teacher/course/{course_id}/unarchive",
            error_label="unarchive_course",
        )

    def duplicate_course(self, course_id: int, name: str | None = None) -> dict:
        """Duplicate a course for a new term — its owner or an admin.

        Mirrors `POST /api/teacher/course/{id}/duplicate`. The copy is yours
        and counts in your limit (403 when it is reached). It gets the
        settings (code, description, instructor, cover, enrollment and report
        settings) and a copy of every module it links that you may copy (a
        public module, or one of yours: what `fork_module` accepts): lessons,
        assets, challenge links with their pass bars and labels. Modules are
        copied, not linked, so editing the copy never changes the source; the
        challenges stay the same challenges. Another teacher's private module
        linked in the course (an assistant's own library module) is neither
        copied nor linked: it is named in `left_out_modules`
        (`[{id, title}]`), and its owner copies it into the new course
        (`copy_module_into_course`). Not copied: enrollments, progress, dates, assistants; the copy is
        `private` and has a fresh `join_code`. `name` defaults to the
        source's name followed by " (copy)". Returns the new course (the
        `teacher_course` shape) plus `left_out_modules`. 409 when
        another course took the copy's address at the same moment (call it
        again) or when the source's cover file is missing from storage.
        """
        body: dict = {}
        if name is not None:
            body["name"] = name
        return self._course_call(
            "POST", f"/teacher/course/{course_id}/duplicate", json_body=body,
            ok=(201,), error_label="duplicate_course",
        )

    def delete_course(self, course_id: int) -> dict:
        """Delete a course — its owner or an admin.

        Mirrors `DELETE /api/teacher/course/{id}`. 409 while a user is
        enrolled, a student or yourself (`remove_student` them first, or
        `archive_course` instead). Deletes the course, its assistant rows, its
        cover and the entries the course made for itself: a private module of
        the course's owner with no lesson and at most one challenge (what
        `add_course_challenge` creates) that no other course links. Every
        other module is unlinked and stays in its owner's library
        (`list_modules`); challenges are never deleted. Returns what
        `course_deletion` announced: `{can_delete, enrollment_count,
        deleted_modules, kept_modules}`.
        """
        return self._course_call(
            "DELETE", f"/teacher/course/{course_id}", error_label="delete_course",
        )

    def course_deletion(self, course_id: int) -> dict:
        """What `delete_course` would do, without doing it — owner or admin.

        Mirrors `GET /api/teacher/course/{id}/deletion` (the console's
        confirmation dialog). Returns `can_delete` (False while a user is
        enrolled), `enrollment_count`, `deleted_modules` and `kept_modules`
        (each `[{id, title}]`): the modules that would go with the course and
        the ones that would stay in their owner's library.
        """
        return self._course_call(
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
        return self._course_call(
            "POST", f"/teacher/course/{course_id}/report/send",
            error_label="send_course_report",
        )

    def challenges_for_course(self):
        """The challenges you may attach to a module.

        Mirrors `GET /api/teacher/challenges-for-course`: public challenges plus
        your own, by name. Each row is the challenge's reference (`id`,
        `name`, `slug`, `status`, `end_date_ts`, `is_public`,
        `kernel_version`, `kind_label`, `user_id`, `owner_username`, `role` —
        yours on it: `"owner"`, `"assistant"` or None — and `statistics`
        with `unique_participants`), plus `metric` — the ranking MetricSpec:
        the scale a threshold is on (`label`, `format`, `precision`) and
        the direction it is compared in (`order`) — and
        `default_pass_threshold`, the bar `attach_challenge` stores when you
        name none (70 for an exercise set, None otherwise).
        A challenge with no evaluation is not listed, because `attach_challenge`
        refuses it.
        """
        return _to_dataframe(self._course_call(
            "GET", "/teacher/challenges-for-course",
            error_label="challenges_for_course",
        ))

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
        return self._course_call(
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
        return _to_dataframe(self._course_call(
            "GET", f"/teacher/course/{course_id}/assistants",
            error_label="course_assistants",
        ))

    def add_course_assistant(self, course_id: int, username: str) -> dict:
        """Add an assistant by username (course owner or admin).

        Mirrors `POST /api/teacher/course/{id}/assistants`. A TA gets the same
        teacher view of the course, but cannot manage the assistant list.
        """
        return self._course_call(
            "POST", f"/teacher/course/{course_id}/assistants",
            json_body={"username": username}, ok=(200, 201),
            error_label="add_course_assistant",
        )

    def remove_course_assistant(self, course_id: int, user_id: int) -> dict:
        """Remove an assistant (course owner or admin).

        Mirrors `DELETE /api/teacher/course/{id}/assistants/{user_id}`.
        """
        return self._course_call(
            "DELETE", f"/teacher/course/{course_id}/assistants/{user_id}",
            error_label="remove_course_assistant",
        )

    def export_course_csv(self, course_id: int, challenge_id: int, *,
                          by_participant: bool = False,
                          dest_dir: str = ".") -> str:
        """Download one attached challenge's course leaderboard as CSV;
        returns the written path.

        Mirrors `GET /api/teacher/export-csv/{course_id}` — the console's
        Students-tab export. `challenge_id` (required) is one of the course's
        attached challenges; the server answers 400 for any other.
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
        (teacher scope).

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
              media/loop.png         # linked from a body as ![](media/loop.png)

        Every path of the manifest and every relative link of a lesson body
        is relative to the directory's root.

        Manifest (`course.yaml`)::

            course:
              name: "Intro to RL"
              slug: intro-to-rl
              code: "RL101"          # optional
              description: "..."     # optional markdown
              visibility: public     # private | unlisted | public
              instructor_name: "Prof X"
              start_date: 2026-01-01
              end_date: 2026-06-01
              cover: cover.png       # optional, relative to the dir
            modules:
              - title: "Foundations"
                slug: foundations
                summary: "..."       # optional
                icon: book           # optional
                visibility: public   # optional: teacher reuse. Left out: private for a
                                     # new module, unchanged for an existing one
                is_published: true   # optional (default true): false = hidden from students
                challenges:          # optional
                  - slug: cartpole-v1       # the challenge's URL slug (or challenge_id:)
                    label: "CartPole"
                    pass_threshold: 195.0   # optional: the score that validates it; omitted =
                                            # the kind's default on a new link (70 for an
                                            # exercise set, else none), unchanged on an existing one
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
        its default, except in the `course` block, and a module without
        `challenges` keeps its attached challenges. A lesson that left the
        manifest is unpublished, not deleted (the students' progress on it is
        kept) and reported as orphaned; `prune=True` deletes it. Unknown
        keys, a missing file or an invalid lesson directive refuse the whole
        bundle: `MLArenaError` with the server's message, nothing written.
        `dry_run=True` returns the same plan and writes nothing.

        A module written by a bundle is managed by it: the console's editor
        and the module / lesson write methods answer 409 until
        `detach_module`. Relative links in a lesson are kept as written; the
        files they name become the module's assets.

        Returns the server's plan: `dry_run`, `course` (`id`, `slug`,
        `action`), `modules` (each `slug`, `id`, `action` — `created` |
        `updated` | `unchanged` | `adopted` —, `renamed_from`, `lessons`,
        `orphaned_lessons`, `pruned_lessons`, `assets`), `unmanaged_modules`,
        `source_revision`, the course's `url`, and `warnings`: one sentence
        per `mlarena:` block whose challenge the students of this course will
        not see (it is private and the course, as the push leaves it, does
        not assign it), naming its module and lesson; `[]` when every block
        is visible. A push with warnings succeeds. Manifests written before
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
        (teacher scope).

        Mirrors `POST /api/teacher/courses/{slug}/import-github`
        `{"repository", "ref", "path", "dry_run", "prune"}`. `repository` is
        `owner/name` or the repository's URL; `ref` a branch, tag or commit
        (default: the default branch); `path` the folder of the repository
        that holds `course.yaml`. The server downloads that commit and
        applies it as `push_course` does, with the same plan. A private
        repository is pushed from its own GitHub Actions workflow instead
        (`mlarena course push`).
        """
        return self._course_call(
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

    def export_course_to_dir(self, slug: str, path: str) -> dict:
        """Export a course to a local directory: a `course.yaml`, the lessons'
        markdown, the files they link and the cover.

        The inverse of `push_course`: pushing the exported directory changes
        nothing. Reads the public consumption surface (landing + lesson
        bodies + module assets), so it works for a course you can view
        (public, or one you own / are enrolled in; owners also get draft
        lessons *and draft modules* — for anyone else an unpublished module
        is simply absent, so exporting someone else's course gives you the
        published cut of it).

        Layout: `<module-slug>/<lesson-slug>.md` for each lesson body, and
        every relative link of a body (`![](media/loop.png)`,
        `[data](data.csv)`) downloaded from its module's assets to that same
        path from the directory's root, which is what a bundle's links are
        relative to. A linked file the module does not hold is reported with
        a warning and skipped; two modules that hold different files under
        one path cannot share a directory, and raise. An attached challenge
        is written by its slug. A module's `visibility` is written for a
        course you manage (it is a teacher-side field). Returns the manifest
        dict.
        """
        landing = self.course(slug)
        base = os.path.abspath(path)
        os.makedirs(base, exist_ok=True)

        modules_out = []
        asset_owner: dict[str, str] = {}
        for module in landing["modules"]:
            module_slug = module["slug"]
            os.makedirs(os.path.join(base, module_slug), exist_ok=True)

            entries: dict[int, dict] = {}
            lessons_out = []
            for toc in module["lessons"]:
                full = self.lesson(slug, module_slug, toc["slug"])
                body_md = full["body_md"]
                rel = f"{module_slug}/{toc['slug']}.md"
                with open(os.path.join(base, module_slug, f"{toc['slug']}.md"),
                          "w", encoding="utf-8") as fh:
                    fh.write(body_md)
                for asset_path in _relative_asset_links(body_md):
                    self._export_asset(full["asset_base_url"], asset_path, base,
                                       module_slug, asset_owner)
                entry = {
                    "title": toc["title"],
                    "slug": toc["slug"],
                    "kind": toc["kind"],
                    "file": rel,
                    "gated": toc["gated"],
                    "is_published": toc["is_published"],
                    "estimated_minutes": toc["estimated_minutes"],
                }
                entries[toc["id"]] = entry
                if toc["parent_lesson_id"] is None:
                    lessons_out.append(entry)
                else:
                    # A lesson's children follow it in the landing's order.
                    entries[toc["parent_lesson_id"]].setdefault(
                        "children", []).append(entry)

            module_out = {
                "title": module["title"],
                "slug": module_slug,
                "summary": module["summary"],
                "icon": module["icon"],
            }
            if landing["can_manage"]:
                module_out["visibility"] = self.get_module(module["id"])["visibility"]
            module_out.update({
                "is_published": module["is_published"],
                "challenges": [_manifest_challenge_entry(c)
                               for c in module["challenges"]],
                "lessons": lessons_out,
            })
            modules_out.append(module_out)

        course_out = {
            "name": landing["name"],
            "slug": landing["slug"],
            "code": landing["code"],
            "description": landing["description"],
            "visibility": landing["visibility"],
            "instructor_name": landing["instructor_name"],
            "start_date": landing["start_date"],
            "end_date": landing["end_date"],
        }
        if landing["has_cover"]:
            course_out["cover"] = self._export_course_cover(landing["id"], base)
        manifest = {"course": course_out, "modules": modules_out}
        _dump_course_manifest(base, manifest)
        return manifest

    def _export_asset(self, asset_base_url: str, asset_path: str, base: str,
                      module_slug: str, asset_owner: dict) -> None:
        """Download one linked module asset to `base/<asset_path>`.

        `asset_base_url` is the lesson's (a path of this server, `/api/…/`).
        `asset_owner` remembers which module wrote each path, across the
        export.
        """
        resp = self._request(
            "GET",
            f"{self._base_url}{asset_base_url}{quote(asset_path)}",
            headers=self._headers(), timeout=300,
        )
        if resp.status_code == 404:
            warnings.warn(
                f"module {module_slug!r}: a lesson links {asset_path!r}, which "
                f"is not among the module's assets; skipped",
                stacklevel=3,
            )
            return
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "export_course_to_dir (asset)", resp)
        content = resp.content
        out_path = os.path.join(base, *asset_path.split("/"))
        owner = asset_owner.get(asset_path)
        if owner is not None and owner != module_slug:
            with open(out_path, "rb") as fh:
                if fh.read() != content:
                    raise MLArenaError(
                        f"modules {owner!r} and {module_slug!r} both hold an "
                        f"asset {asset_path!r}, with different content: one "
                        f"course directory cannot hold both. Rename one of "
                        f"them, then export again."
                    )
            return
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "wb") as fh:
            fh.write(content)
        asset_owner[asset_path] = module_slug

    def _export_course_cover(self, course_id: int, base: str) -> str:
        """Write the course's cover as `cover.<ext>` in `base`; returns the
        file name. The extension is the served image type's."""
        resp = self._request(
            "GET", self._url(f"/academic_courses/assets/courses/{course_id}/cover"),
            headers=self._headers(), timeout=300, stream=True,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "course_cover", resp)
        content_type = resp.headers["Content-Type"].split(";")[0].strip().lower()
        if content_type not in _COVER_EXTENSIONS:
            raise MLArenaError(
                f"course cover is served as {content_type!r}, not an image type "
                f"this export knows ({', '.join(sorted(_COVER_EXTENSIONS))})"
            )
        name = f"cover{_COVER_EXTENSIONS[content_type]}"
        with open(os.path.join(base, name), "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    fh.write(chunk)
        return name

    def __repr__(self) -> str:
        # mlk_<scope>_<lookup>_<secret> / mlo_<lookup>_<secret>: never the secret.
        head = self._token.split("_")
        masked = "_".join(head[:-1] + ["…"]) if len(head) in (3, 4) else "…"
        return f"MLArenaClient(base_url='{self._base_url}', token='{masked}')"


def _as_student(as_student: bool) -> dict | None:
    """The query string of a learner read: ``?as_student=true`` when asked,
    nothing otherwise."""
    return {"as_student": "true"} if as_student else None


def _filtered_fields(fields: dict, allowed: set, method_name: str) -> dict:
    """Validate a ``**fields`` partial-update kwargs dict against the route's
    editable set, failing loud (Fail-Fast) on an unknown field rather than
    silently dropping it, and requiring at least one field."""
    unknown = set(fields) - allowed
    if unknown:
        raise MLArenaError(
            f"{method_name}: unknown field(s) {sorted(unknown)}. "
            f"Allowed: {sorted(allowed)}"
        )
    body = {k: v for k, v in fields.items() if v is not None}
    if not body:
        raise MLArenaError(
            f"{method_name} requires at least one field from {sorted(allowed)}"
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


def _manifest_challenge_entry(link: dict) -> dict:
    """A module's attached challenge as the manifest names it: by slug, which
    is the same on every deployment.

    A link without its challenge (an authoring link serves `challenge: None`
    for a challenge you may not see) cannot be written: the manifest would
    name a challenge by a slug you were not given.
    """
    challenge = link.get("challenge")
    if challenge is None:
        raise MLArenaError(
            f"Cannot export the link to challenge {link.get('challenge_id')}: you may "
            "not see that challenge, so the manifest cannot name it by its slug. Ask "
            "its owner for access, or remove the link from the module."
        )
    return {"slug": challenge["slug"], "label": link["label"],
            "pass_threshold": link["pass_threshold"]}


_COVER_EXTENSIONS = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
}

# What a lesson body links to: the server's rule, `relative_link_targets` and
# `asset_relpath` in `backend/app/services/course_manifest.py`. A course
# bundle holds exactly these files, so the export downloads exactly these.

# Text that renders as written, where a `[a](b)` is not a link.
_LINK_FREE_RES = (
    re.compile(r"(?ms)^[ \t]*(?P<fence>`{3,}|~{3,}).*?^[ \t]*(?P=fence)[ \t]*$"),
    re.compile(r"(?s)<!--.*?-->"),
    re.compile(r"(`+)[^`\n](?:[^\n]*?[^`\n])??\1(?!`)"),
    re.compile(r"(?s)\$\$.+?\$\$"),
    re.compile(r"(?<![\\$])\$(?!\s)[^$\n]+?(?<![\s\\])\$(?!\d)"),
)
# `[text](target "title")`, `![alt](target)` and `[label]: target`.
_MD_LINK_RES = (
    re.compile(
        r"\[(?:[^\[\]]|\[[^\[\]]*\])*\]\(\s*(?P<target><[^<>\n]*>|[^()\s]+)"
        r"(?:\s+(?:\"[^\"]*\"|'[^']*'|\([^()]*\)))?\s*\)"
    ),
    re.compile(r"(?m)^ {0,3}\[(?!\^)[^\]\n]+\]:[ \t]*(?P<target><[^<>\n]*>|\S+)"),
)
_HTML_LINK_RES = (
    re.compile(r"(?is)<(?:img|video|audio|source)\b[^>]*?\ssrc\s*=\s*"
               r"(?:\"(?P<dq>[^\"]*)\"|'(?P<sq>[^']*)')"),
    re.compile(r"(?is)<a\b[^>]*?\shref\s*=\s*(?:\"(?P<dq>[^\"]*)\"|'(?P<sq>[^']*)')"),
)
_URL_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")


def _relative_asset_links(body_md: str) -> list[str]:
    """The module asset paths a lesson body links to, sorted.

    A relative link (`![](media/fig.png)`, `[data](data.csv)`, an HTML `src`
    / `href`) resolves against the module's asset base, so it names a module
    asset. Not assets: anything with a scheme, `/`-absolute targets, `#`
    anchors, whatever sits in code, a comment or math, and a target that
    climbs out of the store (`../x`).
    """
    text = body_md
    for pattern in _LINK_FREE_RES:
        text = pattern.sub(" ", text)
    targets = []
    for pattern in _MD_LINK_RES:
        for match in pattern.finditer(text):
            target = match.group("target")
            if target.startswith("<") and target.endswith(">"):
                target = target[1:-1]
            targets.append(target)
    for pattern in _HTML_LINK_RES:
        for match in pattern.finditer(text):
            target = match.group("dq") if match.group("dq") is not None else match.group("sq")
            targets.append(target.strip())

    found: set[str] = set()
    for target in targets:
        if not target or target.startswith(("#", "/")) or _URL_SCHEME_RE.match(target):
            continue
        asset_path = unquote(re.split(r"[?#]", target, maxsplit=1)[0])
        while asset_path.startswith("./"):
            asset_path = asset_path[2:]
        parts = asset_path.split("/")
        if not asset_path or "\\" in asset_path or any(
                part in ("", ".", "..") for part in parts):
            continue
        found.add(asset_path)
    return sorted(found)


def _dump_course_manifest(base: str, manifest: dict) -> None:
    """Write a manifest as ``course.yaml``."""
    with open(os.path.join(base, "course.yaml"), "w", encoding="utf-8") as fh:
        yaml.safe_dump(manifest, fh, sort_keys=False, allow_unicode=True)


def _default_submission_name(agent, files) -> str:
    if agent is not None:
        return getattr(agent, "__name__", "submission")
    if files:
        return os.path.splitext(os.path.basename(files[0]))[0]
    return "submission"


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
    "attach_competition": "attach_challenge",
    "update_competition_link": "update_challenge_link",
    "detach_competition": "detach_challenge",
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


def _deprecated_alias(old_name, new_name, *, drops_challenge_id=False):
    def alias(self, *args, **kwargs):
        warnings.warn(
            f"MLArenaClient.{old_name}() is deprecated, "
            f"use .{new_name}() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        if drops_challenge_id:
            if args:
                args = args[1:]
            else:
                del kwargs["challenge_id"]
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
    setattr(MLArenaClient, _old, _deprecated_alias(_old, _new))

for _old, _new in _KEYED_BY_SUBMISSION_METHODS.items():
    if not hasattr(MLArenaClient, _new):
        raise AttributeError(
            f"compat map is stale: MLArenaClient has no .{_new}()"
        )
    # Wrapped so an old keyword (`competition_id=`) is renamed before the
    # challenge id is dropped.
    setattr(MLArenaClient, _old, _accepts_legacy_kwargs(
        _deprecated_alias(_old, _new, drops_challenge_id=True)))
