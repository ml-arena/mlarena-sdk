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
from typing import Iterator

import requests

from mlarena.chat import ChatConversation
from mlarena.exceptions import (
    AuthenticationError,
    ChallengeNotFoundError,
    ChatSessionNotFoundError,
    MaintenanceError,
    MLArenaError,
    NotFoundError,
    PermissionDeniedError,
    SubmissionError,
    SubmissionNotFoundError,
)


class _Unset:
    """Default of a keyword the caller did not pass — distinct from `None`,
    which `update_chat_settings` sends to clear a nullable limit."""

    def __repr__(self) -> str:
        return "UNSET"


_UNSET = _Unset()


class MLArenaClient:
    """Client for the ML Arena REST API.

    Pass the full bearer token returned by your Profile page (`mlk_user_...`,
    `mlk_creator_...`, or `mlk_teacher_...`). The token's scope segment dictates
    which endpoints are reachable — a `user`-scope token cannot call a
    `creator`-required route, and vice versa.
    """

    def __init__(self, token: str, base_url: str):
        self._token = token
        self._base_url = base_url.rstrip("/")
        self._last_submission_id = None
        self._last_challenge = None

    def _headers(self, *, json_body: bool = False) -> dict:
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
        "project_url", "enrolled_at_ts"}``. Works with any valid token scope
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
                       github_page: str | None = None,
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
        if github_page is not None:
            body["github_page"] = github_page
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
        ``email``, ``avatar_key``, ``is_admin``, ``is_teacher``, ``is_creator``,
        ``is_human_verified``, ``has_teacher_access``, ``has_creator_access``
        (the real creator rule — the flag *or* admin *or* a creator-assistant
        row) and ``can_create_course``. Every key is always present.

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

    # ---- Teams (per-challenge; any token scope) ----
    #
    # A team is a participant action, so every route below is `user`-scoped:
    # any valid token reaches it. One team per challenge per user; the team a
    # submission is attributed to is the one these methods manage.
    #
    # Shapes mirror the response models in backend/app/views/_schemas.py. A
    # team carries `id`, `challenge_id`, `name`, `looking_for_members`,
    # `created_at_ts`, `members` (each: `id` — the membership row id —,
    # `user_id`, `username`, `avatar_key`, `role`) and `my_role`, your own
    # role on it ("leader", "member" or None). An invitation carries `id`,
    # `team_id`, `team_name`, `user_id`, `username`, `sender_id`,
    # `sender_username`, `status` and `created_at_ts`.

    def challenge_team(self, challenge_id: int) -> dict:
        """Your team for a challenge.

        Mirrors ``GET /api/teams/challenge/{id}/team``. Raises
        ``ChallengeNotFoundError`` when you are not on a team for it (the
        route's 404), so ``my_role`` is never None on a returned team.
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

    def update_team(self, team_id: int, name: str,
                    looking_for_members: bool | None) -> dict:
        """Rename a team / flip its "looking for members" flag (leader only).

        Mirrors ``PUT /api/teams/{team_id}``: the route takes both keys, so
        both are required here — pass the team's current value for the one you
        are not changing.
        """
        resp = self._request("PUT", self._url(f"/teams/{team_id}"),
                             headers=self._headers(json_body=True),
                             json={"name": name,
                                   "looking_for_members": looking_for_members})
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

    # ---- Challenges (public read; create/update via creator scope) ----

    def challenges(
        self,
        *,
        q: str | None = None,
        tags: list[str] | str | None = None,
        status: str | None = None,
        page: int | None = None,
        per_page: int | None = None,
    ):
        """List challenges. Public; no auth required.

        With no args, returns all matching challenges as a DataFrame
        (auto-paginates internally). Pass ``page``/``per_page`` to fetch a
        single page; pass ``q``/``tags``/``status`` to filter server-side.
        ``status`` is ``"active"`` (only started challenges, the default) or
        ``"all"`` — the route knows no other value and answers 400.

        Each row carries ``id``, ``name``, ``description``, ``miniature``,
        ``is_started``, ``is_award_points``, ``number_of_agents``, ``tags``,
        ``statistics`` and ``course`` — ``{"id", "name", "code"}`` on the
        enrolled-course rows, null elsewhere. Call ``challenge(id)`` for the
        engine, the kernel and the limits.
        """
        base_params: dict = {}
        if q:
            base_params["q"] = q
        if status:
            base_params["status"] = status
        if tags:
            base_params["tags"] = ",".join(tags) if isinstance(tags, list) else tags

        # Public route, but it answers differently to the caller it can
        # identify (enrolled courses' challenges on page 1, admins' unstarted
        # challenges): send the token.
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

        Includes an `engine` sub-object with the engine's
        `k8s_workload_value` plus `vm_health_ok` / `vm_health_checked_at_ts`
        (only populated for `local_vm` engines, polled every minute by
        simulationmanager). `vm_health_ok=False` means the GPU VM is
        currently unreachable and submissions will queue rather than run.
        """
        # Public route, but a non-public course challenge is only visible to a
        # caller the backend can identify: send the token.
        resp = self._request("GET", self._url(f"/challenges/{challenge_id}"),
                             headers=self._headers(), timeout=30)
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "challenge", resp)
        return resp.json()

    # The infrastructure fields the console's admin panel edits
    # (`frontend/src/hooks/creatorChallenge/useAdmin.ts`, `emptyConfigFields`).
    _ADMIN_CONFIGURATION_FIELDS = frozenset({
        "engine_id", "docker_image_env_runtime_id", "render_delay_second",
    })

    def update_challenge_configuration(self, challenge_id: int, **fields) -> dict:
        """Patch a challenge's infrastructure configuration (admin only).

        Mirrors `PUT /api/challenges/{id}/configuration` — the call the
        console's admin panel makes to repoint a challenge at another engine.
        Accepts `engine_id`, `docker_image_env_runtime_id` and
        `render_delay_second`; an unknown field raises before any request.
        Only the fields you pass are changed.

        The step deadlines and the step budget (`agent_max_time_per_step_second`,
        `env_max_time_per_step_second`, `simulation_max_steps`, admin only),
        the simulation timeout and the upload / submission limits are
        `update_settings()`, which owns their bounds and their
        frozen-after-start rules.

        A new challenge gets the default engine of its kind; pinning one with
        more memory (e.g. `engine_id=...` for a scorer that needs 3Gi) is this
        call. The route checks the account's admin flag, not the key scope, so
        any key of an admin account works; a non-admin account is refused.
        Nothing here checks that the engine runs the challenge's kernel — the
        backend's image-consistency check rejects a mismatch at `run_benchmark`
        and `start_challenge`.

        Returns the updated configuration (`engine_id`, ...).
        """
        body = _filtered_fields(fields, self._ADMIN_CONFIGURATION_FIELDS,
                                "update_challenge_configuration")
        resp = self._request("PUT",
            self._url(f"/challenges/{challenge_id}/configuration"),
            headers=self._headers(json_body=True),
            json=body,
            timeout=30,
        )
        self._handle_response(resp)
        if resp.status_code in (301, 302, 303, 307, 308):
            # `admin_required` redirects a non-admin to the login page.
            raise AuthenticationError(
                "update_challenge_configuration requires an admin account")
        if resp.status_code != 200:
            raise _failed(MLArenaError, "update_challenge_configuration", resp)
        return resp.json()

    def creator_challenges(self) -> list:
        """List challenges the caller owns or assists on (creator scope).

        Mirrors `GET /api/creator_challenge/challenges`
        (`lifecycle.py`, `get_creator_challenges`). Unlike `challenges()`
        this includes the caller's **hidden** (`is_public=False`) challenges
        — useful for finding a challenge you created but did not make
        public. Each item carries
        `id`, `name`, `is_started`, `is_public`, `kind`, `role`.

        Requires a `creator`-scope token.
        """
        resp = self._request("GET",
            self._url("/creator_challenge/challenges"),
            headers=self._headers(),
            timeout=30,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "creator_challenges", resp)
        return resp.json()

    def creator_challenge(self, challenge_id: int) -> dict:
        """The challenge as its editor sees it (creator scope).

        Mirrors `GET /api/creator_challenge/challenge/{id}`
        (`lifecycle.py`, `get_creator_challenge`). Returns the challenge plus
        its three sibling rows as three objects, each under its own column
        names: `configuration` (engine, runtime image, step deadlines, upload
        limits, `submission_filename`, the GitHub sync), `evaluation`
        (`metric`, `metric_order`, `is_elo_score`, the ELO parameters, the
        episode budget brackets, `metrics_schema`) and `environment` (the
        latest benchmark run's `benchmark_simulation_result_id` and
        `attach_path_files`). The configuration's `kernel_version` is the
        env runtime's kernel. Also `engine_name`, the
        engine health subset, and `role` — `"owner"` or `"assistant"`.

        Unlike `challenge()` this works on your own hidden challenges and
        shows the authoring state; `challenge()` is the participant view.

        Requires a `creator`-scope token and ownership (or admin) of the
        challenge.
        """
        return self._creator_get(challenge_id, "", "creator_challenge")

    def available_kinds(self) -> list:
        """The challenge kinds `create_challenge` accepts (creator scope).

        Mirrors `GET /api/creator_challenge/available_kinds` (`kinds.py`).
        Each entry carries `kernel_version` (what `create_challenge` takes),
        `label`, `description`, `protocol`, `isolation`, `has_engine`,
        `capabilities` (`agent_template`, `benchmark`, `dataset`,
        `env_structural_check`, `runs`, `chat`) and
        `minimal_loop_snippet`. `has_engine` is False when
        the deployment has no engine for that kernel, and creating one then
        answers 503.
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
        (`lifecycle.py`). Returns `{"challenges": [{"id", "name",
        "is_started", "engine_name"}]}` — the ones you own (a challenge you
        only assist on is not a copy source).
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

        `kernel_version` carries a catalog KIND: one of `"flex_v1"`,
        `"gymnasium"`, `"pettingzoo"`, `"file_v1"` (list them live via
        `GET /creator_challenge/available_kinds`). Since the worker
        consolidation only two kernels exist — `gymnasium`/`pettingzoo` are
        presets over the flex_v1 kernel that pair the flexkit loop-library
        env template with the matching env-image family.

        The backend resolves the engine + default evaluation + env-image
        family from the kind. To pin a specific engine, use the creator UI
        or post-creation patch endpoints — the API does not let you pick an
        engine directly at creation time.

        If `tag_names` is given, each name is resolved against the public
        tag catalog (`GET /api/challenge_tags/tags`) before the create
        call. Unknown names raise `MLArenaError` (fail fast — the catalog
        is admin-curated; new tags are not auto-created).

        Pass `is_public=False` to hide the challenge from public listings,
        search, and direct URLs at creation time. Only the owner, creator
        assistants, and admins can view or interact with a hidden
        challenge; everyone else gets 404. Visibility can be toggled later
        via `update_challenge(is_public=...)`. Defaults to public.

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

        A public route; the bearer token travels anyway, as on every read.
        """
        resp = self._request("GET", self._url("/challenge_tags/tags"),
                             headers=self._headers(), timeout=30)
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "list_tags", resp)
        return _to_dataframe(resp.json())

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
                           is_public: bool | None = None) -> dict:
        """Patch top-level challenge fields (creator scope).

        Mirrors `PUT /api/creator_challenge/challenge/{id}`. Only fields
        explicitly passed are sent — everything else is left untouched.

        ``name`` is locked once the challenge has started — a rename strands
        the links and course material already pointing at it. ``description``
        and ``is_public`` stay editable while it runs, so a creator can fix the
        blurb or hide an active challenge without stopping it (stopping
        rewrites ``start_date_ts``). When ``is_public=False``, only the owner,
        creator assistants, and admins can view or interact with the challenge.
        """
        body: dict = {}
        if name is not None:
            body["name"] = name
        if description is not None:
            body["description"] = description
        if is_public is not None:
            body["is_public"] = is_public
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

    def update_settings(self, challenge_id: int, *,
                        simulation_timeout_sec: int | None = None,
                        max_upload_size_bytes: int | None = None,
                        max_upload_files: int | None = None,
                        max_active_submissions_per_participant: int | None = None,
                        submission_filename: str | None = None,
                        agent_max_time_per_step_second: float | None = None,
                        env_max_time_per_step_second: float | None = None,
                        simulation_max_steps: int | None = None,
                        metric: str | None = None,
                        metric2: str | None = None,
                        is_elo_score: bool | None = None,
                        metric_order: str | None = None,
                        is_stop_after_deployment: bool | None = None,
                        deployment_nb_constraint_run: int | None = None,
                        deployment_nb_initial_score_run: int | None = None,
                        episode_budget_brackets: list | None = None,
                        frontend_precision: int | None = None,
                        metrics_schema: list | None = None,
                        data_source_enabled: bool | _Unset = _UNSET,
                        data_source_url: str | None | _Unset = _UNSET,
                        data_source_asset: str | None | _Unset = _UNSET,
                        data_source_filter: dict[str, str] | None | _Unset = _UNSET,
                        data_source_history_hours: int | _Unset = _UNSET,
                        batch_cron: str | _Unset = _UNSET,
                        ) -> dict:
        """Update challenge settings + evaluation parameters.

        Mirrors `PUT /api/creator_challenge/challenge/{id}/settings` —
        the same call the console's Settings tab makes via `saveSettings()`.
        Only fields explicitly passed are sent; everything else is left
        untouched. Every keyword is the column's own name: the first eight
        and the data-feed ones are ChallengeConfiguration columns, the rest
        are Evaluation columns. Once
        the challenge has started only the fields that cannot rescore an
        existing run still apply — the leaderboard labels (`metric`,
        `metric2`, `frontend_precision`) and `is_stop_after_deployment`;
        everything else is rejected with 400 until the challenge is stopped.

        Returns `{"configuration": {...}, "evaluation": {...}}` — the two rows
        as the update left them, each under its own column names.

        Notable parameters:
            submission_filename: file_v1 only — the one file a participant
                uploads, e.g. "submission.csv" (the default), "pitch.txt" or
                "submission.csv.gz". A bare file name: letters, digits, '.',
                '_' and '-', starting with a letter or digit, at most 128
                characters. A name not ending in ".csv" skips the y_test.csv
                column check at upload and the y_test.csv start requirement,
                so env.py must validate the file itself. Frozen once the
                challenge has started.
            metric: free-text label for the primary metric (e.g. "reward",
                "accuracy", "bleu"). The DB column is String(20).
            is_elo_score: when True, the leaderboard ranks by ELO rather than
                mean metric (multi-agent kernels).
            metric_order: "desc" (the default: a higher score is better) or
                "asc" (a lower score is better, e.g. RMSE). Decides who wins
                each run, the episode budget tiers and, unless the challenge
                is ELO-ranked (ELO is always higher-is-better), the
                leaderboard order and course pass verdicts. Frozen once the
                challenge has started; refused on an ELO challenge whose
                submissions already have ratings. Checked against the stored
                brackets and metrics schema, so a flip may need new brackets
                in the same call.
            is_stop_after_deployment: True (the creation default) means an
                agent runs only its deployment runs and is never matched
                again, which pins an ELO leaderboard to its bootstrap ratings
                forever. Set False to let the matchmaker keep pairing agents
                (~10 matches per challenge every 2h). Editable on a running
                challenge.
            episode_budget_brackets: episode budget tiers, a list of
                `[threshold, n_episodes]` pairs from the worst threshold to
                the best. A submission whose running mean is worse than a
                tier's threshold gets that tier's episodes; one that beats
                every threshold gets the last tier's. Thresholds strictly
                ascending under metric_order "desc", strictly descending
                under "asc"; n_episodes non-decreasing; max 100 episodes per
                bracket. A single pair is a fixed budget.
            frontend_precision: decimal places for numeric leaderboard
                metrics (default 2). Per-metric `precision` in the schema
                overrides it.
            metrics_schema: the canonical leaderboard metric declaration —
                an ordered list of descriptors, e.g.
                `[{"key": "accuracy", "label": "Accuracy", "source": "env",
                   "agg": "mean", "format": "percent"},
                  {"key": "ram_max", "label": "RAM Max", "source": "platform",
                   "agg": "max", "format": "bytes"}]`.
                `source:"env"` keys are exactly what `env.evaluate` must
                return in each `agent_results[i]["metrics_detail"]`
                (equal-mapping, enforced at run time). See the leaderboard
                row's `metrics_schema` column. A `higher_is_better`
                on the `reward` descriptor must match metric_order.

        Run limits (admin only — anyone else gets `PermissionDeniedError`,
        403; strictly positive; frozen while the challenge runs):
            agent_max_time_per_step_second: deadline of one agent call
                (`AgentProxy.call`), in seconds.
            env_max_time_per_step_second: deadline of one env step, in
                seconds.
            simulation_max_steps: the step budget of one simulation.

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

        Requires a `creator`-scope token and ownership (or admin) of the
        target challenge.
        """
        sent = {
            "simulation_timeout_sec": simulation_timeout_sec,
            "max_upload_size_bytes": max_upload_size_bytes,
            "max_upload_files": max_upload_files,
            "max_active_submissions_per_participant":
                max_active_submissions_per_participant,
            "submission_filename": submission_filename,
            "agent_max_time_per_step_second": agent_max_time_per_step_second,
            "env_max_time_per_step_second": env_max_time_per_step_second,
            "simulation_max_steps": simulation_max_steps,
            "metric": metric,
            "metric2": metric2,
            "is_elo_score": is_elo_score,
            "metric_order": metric_order,
            "is_stop_after_deployment": is_stop_after_deployment,
            "deployment_nb_constraint_run": deployment_nb_constraint_run,
            "deployment_nb_initial_score_run": deployment_nb_initial_score_run,
            "episode_budget_brackets": episode_budget_brackets,
            "frontend_precision": frontend_precision,
            "metrics_schema": metrics_schema,
        }
        body = {key: value for key, value in sent.items() if value is not None}
        feed = {
            "data_source_enabled": data_source_enabled,
            "data_source_url": data_source_url,
            "data_source_asset": data_source_asset,
            "data_source_filter": data_source_filter,
            "data_source_history_hours": data_source_history_hours,
            "batch_cron": batch_cron,
        }
        body.update({key: value for key, value in feed.items() if value is not _UNSET})
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

    def sync_env_from_github(
        self,
        challenge_id: int,
        github_repo_url: str,
        github_token: str | None = None,
    ) -> dict:
        """Mirror a GitHub repo's default branch into the challenge env folder.

        Wipes the current env folder and replaces its contents with the repo
        (`.git` excluded). Only allowed before the challenge is started, same
        gate as `upload_env_file`. `github_token` is optional: when provided it
        is persisted Fernet-encrypted on the challenge configuration and
        reused on future syncs; leave it `None` to fall back to the saved
        token (if any).
        """
        body = {"github_repo_url": github_repo_url}
        if github_token is not None:
            body["github_token"] = github_token
        resp = self._request("POST",
            self._url(
                f"/creator_challenge/challenge/{challenge_id}/env/sync-github"
            ),
            headers=self._headers(json_body=True),
            json=body,
            timeout=180,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "sync_env_from_github", resp)
        return resp.json()

    def update_env_file_content(self, challenge_id: int, filename: str,
                                content: str) -> dict:
        """Write a text env file by content (e.g. a rendered env.py template).

        JSON PUT to the same endpoint as `upload_env_file`. The backend runs
        the kernel-specific AST validator on env.py and returns a
        `validation` block in the response.
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
        shape `benchmark_status` serves it (`job_status` `pending`).

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

    def benchmark_status(self, challenge_id: int) -> dict | None:
        """The latest benchmark run, or None before the first one.

        Mirrors `GET /api/creator_challenge/challenge/{id}/benchmark/status`
        (`benchmark.py`). The run is the same dict `creator_runs()` lists
        (`RunResult`): `job_status` (`pending`, `running`, then `completed`,
        `failed`, `reaped` or `cancelled`), `env_error_type` /
        `env_error_message` / `env_stdout_logs`, the timestamps, and
        `submission_results` — the benchmark submission's row, its score
        under `submission_reward`. When the run completes cleanly the backend
        scores the benchmark submission (`active`, `mean_reward`), which is
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

    def start_challenge(self, challenge_id: int) -> dict:
        """Flip a creator challenge into the started state.

        Backend gates this behind: env.py uploaded, benchmark submission
        ACTIVE with a non-null `mean_reward`. Failures bubble up as
        MLArenaError.
        """
        resp = self._request("PUT",
            self._url(
                f"/creator_challenge/challenge/{challenge_id}/start"
            ),
            headers=self._headers(),
            timeout=30,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "start_challenge", resp)
        return resp.json()

    def stop_challenge(self, challenge_id: int) -> dict:
        """Flip a creator challenge back into the not-started state.

        Mirrors `PUT /api/creator_challenge/challenge/{id}/stop`. Stopping
        only clears `is_started`; it leaves submissions, results and the
        benchmark record untouched, so the challenge can be re-started
        afterwards.
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
        passed are sent. Rejected once the challenge is started. Returns `{"id", "label", "description"}`.

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

    def creator_runs(self, challenge_id: int) -> dict:
        """The challenge's last 30 runs, with the env side's diagnostics.

        Mirrors `GET /api/creator_challenge/challenge/{id}/runs` (`runs.py`).
        Returns `{"runs": [...]}` — the same `RunResult` rows the submission
        status and the admin runs page serve, but served to the creator with
        the environment's error, its stdout and every agent's row inline.
        Includes benchmark tests, deployment games and participant runs.
        """
        return self._creator_get(challenge_id, "/runs", "creator_runs")

    def creator_submissions(self, challenge_id: int) -> dict:
        """Every non-deleted submission on a challenge you own.

        Mirrors `GET /api/creator_challenge/challenge/{id}/submissions`
        (`submissions.py`). Returns `{"submissions": [...], "is_elo_score":
        bool, "ranked_order": "asc"|"desc"}`; each row carries
        `submission_id`, `submission_name`, `user_id`, `username`,
        `created_at_ts`, `last_end_run_ts`, `elo_score`, `mean_reward`,
        `number_of_runs`, `rank`, `restart_allowed` and the status block.
        `rank` is over the active submissions only — the public leaderboard's
        scope — and null otherwise.
        """
        return self._creator_get(
            challenge_id, "/submissions", "creator_submissions"
        )

    def clean_redeploy_submission(self, challenge_id: int,
                                  submission_id: int) -> dict:
        """Wipe one submission's results and queue a fresh deployment.

        Mirrors `POST
        /api/creator_challenge/challenge/{cid}/submissions/{sid}/clean_redeploy`
        (`submissions.py`) — the Restart button of the creator's submissions
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

    def clean_redeploy_all(self, challenge_id: int) -> dict:
        """Clean and redeploy every eligible submission of a challenge.

        Mirrors `POST
        /api/creator_challenge/challenge/{id}/submissions/clean_redeploy_all`
        (`submissions.py`). Returns `{"total", "redeployed": [...],
        "failed": [{"submission_id", "error"}]}` — a per-submission failure
        does not stop the others, so retry only what `failed` names.
        """
        resp = self._request("POST",
            self._url(
                f"/creator_challenge/challenge/{challenge_id}"
                "/submissions/clean_redeploy_all"
            ),
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
        """The creator-assistants of a challenge.

        Mirrors `GET /api/creator_challenge/challenge/{id}/assistants`
        (`assistants.py`). Each entry carries `user_id`, `username` and
        `created_at_ts`. An assistant has the creator's editing rights on this
        challenge; only the owner may change the list.
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
        """Revoke a creator-assistant — owner only.

        Mirrors `DELETE
        /api/creator_challenge/challenge/{id}/assistants/{user_id}`
        (`assistants.py`).
        """
        return self._creator_delete(
            challenge_id, f"/assistants/{user_id}", "remove_challenge_assistant"
        )

    # ---- Submissions (user scope) ----

    def create_submission(self, challenge_id: int, submission_name: str,
                          copy_from_submission_id: int | None = None) -> dict:
        """Create a new submission for `challenge_id`.

        Mirrors `POST /api/submissions/challenge/{cid}` (`create_delete.py`).
        With `copy_from_submission_id`, the files (and a compatible runtime) of
        one of your existing submissions are copied into the new one. Returns
        `{"message", "submission_id", "status", "validation_message",
        "template_source", ...}`.

        A 404 is `ChallengeNotFoundError`, or `SubmissionNotFoundError` when
        `copy_from_submission_id` is given: the source ("Source submission not
        found", also for a submission that is not yours) is the id most
        likely to be wrong on that call. Requires a `user`-scope token.
        """
        body: dict = {"submission_name": submission_name}
        if copy_from_submission_id is not None:
            body["copy_from_submission_id"] = copy_from_submission_id
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
        name from ``challenge(challenge_id)["submission_filename"]`` rather
        than assuming ``submission.csv`` — a challenge may ask for
        ``pitch.txt`` or ``submission.csv.gz``. (On a ``flex_v1`` challenge the
        entry point is ``agent.py``, plus any helper files.)

        Returns the server response: ``message``, ``validation_message`` and the
        status block (see ``submission_status()``).
        Note: the file route returns HTTP 200 even when validation *rejects* the
        submission — the file is saved but the whole submission is re-validated and
        may come back ``status == "upload_failed"`` with an actionable
        ``validation_message``. A 2xx here is not proof the submission is deployable;
        read ``is_deployable`` (``submit()`` checks it once, on
        ``submission_status()``, after every pre-deploy step and before deploying)."""
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
        """Read the deploy quotas and last deploy of a submission.

        Mirrors `GET /api/submissions/challenge/{cid}/{sid}/deploy`
        (`deploy.py`). Snake_case, like every other payload — the envelope keys
        were `deploymentLimits` / `latestDeploy` until 2.1.0.

        - `deployment_limits` — `daily_deploy_limit`, `daily_deploys_used`,
          `daily_deploys_remaining`, `next_deploy_available_at`, `can_deploy`.
          The same block `my_submissions()` returns.
        - `active_submission_limits` — `max_active_submissions`,
          `active_submissions_count`, `active_submissions_remaining`.
          `active_submissions_count` counts the participant's (or their
          team's) submissions on this challenge that are active *or*
          deploying (`deploy_queue` / `deploy_run`): a deploy in flight
          already holds one of the slots.
        - `latest_deploy` — the attempt's own outcome: `id`, `created_at_ts`,
          `status` (`queued` | `running` | `succeeded` | `failed` |
          `cancelled`), `finished_at_ts` and `failure_message`. All-null when
          the submission has never deployed.
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

    def runtime_options(self, challenge_id: int):
        """List runtimes (language × framework × version) compatible with this challenge.

        Mirrors `GET /api/submissions/runtime_options/{cid}`
        (`runtime.py`, `get_runtime_options`). Each entry: `{id, language,
        language_version, framework, framework_version, requirement}`.
        """
        resp = self._request("GET",
            self._url(f"/submissions/runtime_options/{challenge_id}"),
            headers=self._headers(),
            timeout=30,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(SubmissionError, "runtime_options", resp)
        return _to_dataframe(resp.json())

    def agent_runtime(self, submission_id: int) -> dict | None:
        """Agent runtime currently pinned to a submission, or None when it pins
        none: a file or chat challenge's submission runs no agent container.

        Mirrors `GET /api/submissions/agent_runtime/{sid}` (`runtime.py`,
        `get_agent_runtime`).
        """
        resp = self._request("GET",
            self._url(f"/submissions/agent_runtime/{submission_id}"),
            headers=self._headers(),
            timeout=30,
        )
        self._handle_response(resp, not_found=SubmissionNotFoundError)
        if resp.status_code != 200:
            raise _failed(SubmissionError, "agent_runtime", resp)
        return resp.json()

    def set_agent_runtime(self, submission_id: int, runtime_id: int) -> dict:
        """Pin a `DockerImageAgentRuntime` onto a submission.

        Mirrors `PUT /api/submissions/agent_runtime/{sid}` (`runtime.py`,
        `update_agent_runtime`). The backend rejects runtimes whose
        `docker_image_worker_envagent_id` does not match the challenge's engine.
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

    def resolve_runtime(self, challenge_id: int, *,
                        language: str | None = None,
                        framework: str | None = None,
                        framework_version: str | None = None) -> dict:
        """Find one runtime matching the given language/framework on this challenge.

        Convenience wrapper over `runtime_options`. Returns the first match,
        or raises `MLArenaError` if none. Used internally by `submit()` when
        the caller passes `runtime={"language": ...}` instead of a numeric id.
        """
        opts = self.runtime_options(challenge_id)
        # Normalize back to list[dict] in case pandas converted it.
        rows = opts.to_dict("records") if hasattr(opts, "to_dict") else list(opts)
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
        """List files in a submission with their content (or binary marker).

        Mirrors `GET /api/submissions/challenge/{cid}/{sid}/file` (`file.py`,
        `get_submission_files`). Returns the raw
        `{"files": {<name>: {content, language, is_binary, size}}}` payload —
        keep the wrapper shallow so callers can drill into either the file map
        or the metadata.
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
        (`file.py`, `get_file_content`).
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

        Readable for the owner, for a public submission, and for the teacher
        of the submitter.
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

    # ---- Submission documentation (markdown shown on the submission page) ----

    def upload_submission_docs(self, challenge_id: int, submission_id: int,
                               file_path: str) -> dict:
        """Attach a markdown write-up to an active submission.

        Mirrors `PUT /api/submissions/challenge/{cid}/{sid}/docs` (`file.py`,
        `manage_documentation`) — what the console's Documentation panel does.
        Only `.md` files are accepted, and only while the submission is
        `active`: the route does not touch its status, so documentation can be
        added after a deploy without re-validating anything.
        """
        if not os.path.isfile(file_path):
            raise SubmissionError(f"File not found: {file_path}")
        with open(file_path, "rb") as fh:
            resp = self._request("PUT",
                self._url(
                    f"/submissions/challenge/{challenge_id}/{submission_id}/docs"
                ),
                headers=self._headers(),
                files={"file": (os.path.basename(file_path), fh)},
                timeout=120,
            )
        self._handle_response(resp, not_found=SubmissionNotFoundError)
        if resp.status_code != 200:
            raise _failed(SubmissionError, "upload_submission_docs", resp)
        return resp.json()

    def delete_submission_docs(self, challenge_id: int, submission_id: int,
                               filename: str) -> dict:
        """Remove one markdown documentation file from an active submission.

        Mirrors `DELETE /api/submissions/challenge/{cid}/{sid}/docs?filename=`
        (`file.py`, `manage_documentation`).
        """
        resp = self._request("DELETE",
            self._url(
                f"/submissions/challenge/{challenge_id}/{submission_id}/docs"
            ),
            headers=self._headers(),
            params={"filename": filename},
            timeout=30,
        )
        self._handle_response(resp, not_found=SubmissionNotFoundError)
        if resp.status_code != 200:
            raise _failed(SubmissionError, "delete_submission_docs", resp)
        return resp.json()

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

    # ---- Status / logs ----

    def submission_status(self, challenge_id: int, submission_id: int) -> dict:
        """Rich submission status: the status block, `queue_info`, `run_info`.

        Mirrors `GET /api/submissions/challenge/{cid}/{sid}/status`
        (`status.py`). Use this over `submission_deploy_status` when you want
        to see queue position or per-run errors.

        Returns `id`, `challenge_id`, `submission_name`, `user_id`,
        `created_at_ts`, `is_public`, `is_owner`, the three blocks below, and
        the status block every submission payload carries:

        - `status` — one of `created`, `upload_failed`, `upload_validated`,
          `deploy_queue`, `deploy_run`, `deploy_failed`, `active`, `deleted`.
          (A file upload is one request that ends validated or failed, so
          there is no `uploading` state on the wire.)
        - `phase` — `upload` | `deployment` | `active` | `terminal`.
        - `last_status_message` — the row's latest message, or None.
        - `status_update_ts` — ISO-8601 UTC, or None.
        - `is_uploadable` / `is_deployable` / `is_settled` — the server's own
          answer to "may I upload / deploy / stop polling". Read these instead
          of comparing `status` against a set of your own.

        The three blocks have the same keys in *every* state — they used to
        appear only inside their own phase, so the reason a deploy failed
        vanished the moment it failed:

        - `latest_deploy` — the most recent attempt: `id`, `created_at_ts`,
          `status`, `finished_at_ts`, `failure_message`. All-null before the
          first deploy.
        - `run_info` — `submission_deploy_id`, `number_of_agents`, `has_gpu`,
          `metric`, `is_elo_score`, `frontend_precision` and `results`, that
          attempt's runs.
          Each run is a `RunResult`, the one run model every run list on the
          API serves (`submission_games` too): the job's own `job_status`
          (`pending` | `running` | `completed` | `failed` | `cancelled` |
          `reaped`), `job_started_at_ts`, `job_completed_at_ts`,
          `job_error_message`, `job_retry_count`, `created_at_ts`,
          `simulate_end_time_ts`, `is_test`, `is_deployment`, `env_nb_steps`,
          the env's `step_time_*_sec` / `env_metric_*` columns and, when the
          challenge's own code is why the run ended, `env_error_type`
          (`code_error` | `simulation_error` | `pod_crash` | `unknown`).
          `submission_results` lists one `RunSubmissionResult` per agent in
          the run — **yours is the row whose `submission_id` is this
          submission's**, the others are the opponents. A row carries
          `submission_id`, `submission_name`, `user_name`,
          `agent_attached_player_id`, `env_player_name`, `submission_reward`,
          `submission_reward2`, `submission_reward_variance`,
          `submission_reward_n_episodes`, `reward_ci95`, `agent_nb_steps`,
          `game_outcome`, `final_rank`, `score_elo_before`, `score_elo_delta`,
          `metrics_detail`, `info_message`, the agent's `action_time_*_sec` /
          `agent_metric_*` columns and, when that agent is why the run ended,
          `agent_error_type` (the same four values) with
          `agent_error_message`. Timestamps are ISO-8601 UTC with a `Z`.
        - `queue_info` — `queue_position`, `queue_total` (numbers, not the
          `"116/127"` string this used to send), `created_at_ts` and
          `in_queue_for_second`. All-null when nothing is queued.

        `agent_error_message`, `agent_stdout_logs` and `failure_message` are
        None on a row that is not yours: they quote the owner's own traceback
        and stdout. `env_error_message` and `env_stdout_logs` belong to the
        challenge's creator and are None here.
        """
        resp = self._request("GET",
            self._url(
                f"/submissions/challenge/{challenge_id}/{submission_id}/status"
            ),
            headers=self._headers(),
            timeout=30,
        )
        self._handle_response(resp, not_found=SubmissionNotFoundError)
        if resp.status_code != 200:
            raise _failed(SubmissionError, "submission_status", resp)
        return resp.json()

    def submission_games(self, submission_id: int) -> dict:
        """Recent games for a submission with signed log URLs.

        Mirrors `GET /api/submissions/submission/{sid}/games`
        (`monitor.py`, `get_submission_games`). Returns `{"games",
        "challenge_context"}`. Each game is a `SubmissionGame`: the same
        `RunResult` that `submission_status` serves under `run_info.results`
        (`job_status`, `env_error_type`, `env_nb_steps`, and the
        `submission_results` rows — yours is the one whose `submission_id`
        is this submission's, with `agent_error_type` /
        `agent_error_message`) plus `signed_url`, the replay file (60-day
        retention; None when the run wrote none), and `render_delay_second`.
        A crashed run reads as a crash here too; it used to be listed as an
        ordinary lost game, `outcome: "loser"` with a reward of 0.0 and no
        error field at all.
        """
        resp = self._request("GET",
            self._url(f"/submissions/submission/{submission_id}/games"),
            headers=self._headers(),
            timeout=30,
        )
        self._handle_response(resp, not_found=SubmissionNotFoundError)
        if resp.status_code != 200:
            raise _failed(SubmissionError, "submission_games", resp)
        return resp.json()

    def submission_overview(self, challenge_id: int, submission_id: int) -> dict:
        """The submission's aggregate numbers and its place on the board.

        Mirrors `GET /api/submission_result/{cid}/{sid}/overview`
        (`submission_result.py`) — what the console's Dashboard tab shows.

        Returns `created_at_ts`, `mean_reward`, `elo_score`, `number_of_runs`,
        `last_end_run_ts`, the last-24h resource aggregates (`max_ram_usage`,
        `avg_cpu_usage`, `avg_steps`, `runs_last_24h`, `max_vram_bytes`),
        `submissions_in_queue`, and the challenge context for reading them
        (`rank`, `is_elo_score`, `ranked_order`, `metric`,
        `frontend_precision`, `has_gpu`).

        The warm-up counterparts (`warmup_mean_reward`, `warmup_number_of_runs`,
        `warmup_elo_score`) that older backends served are gone: nothing had
        written them a real value since the Celery deploy path was removed, so
        they were always null. A backend still serving them is simply older —
        this method returns the body verbatim either way.

        `agent_error_type` / `agent_error_message` are the newest run's
        failure, under the run's own column names (the keys every run payload
        uses). The message is None unless the submission is yours: it is your
        agent's traceback.
        """
        resp = self._request("GET",
            self._url(f"/submission_result/{challenge_id}/{submission_id}/overview"),
            headers=self._headers(),
            timeout=30,
        )
        self._handle_response(resp, not_found=SubmissionNotFoundError)
        if resp.status_code != 200:
            raise _failed(SubmissionError, "submission_overview", resp)
        return resp.json()

    def my_submissions(self) -> dict:
        """Every submission you have made, across every challenge.

        Mirrors `GET /api/submissions/mine` (`my_submissions.py`) — what the
        console's My Submissions page lists.

        Returns `submissions`, `started_submissions_count` and
        `deployment_limits` (the same quota block `submission_deploy_status`
        serves). Each row carries `id`, `submission_name`, `challenge_id`,
        `challenge_name`, `rank`, `created_at_ts`, `challenge_start_date_ts`
        (the challenge's own `start_date_ts` column), `last_end_run_ts` and the
        status block. Deleted submissions are not
        listed: a row you deleted is gone from here, as from the console.
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
        """Show or hide a submission in public listings.

        Mirrors `PUT /api/submissions/submission/{sid}/visibility`
        (`monitor.py`). A public submission's results and runs are readable by
        anyone; its error *messages* stay yours. A deleted submission cannot be
        made public again (409).
        """
        resp = self._request("PUT",
            self._url(f"/submissions/submission/{submission_id}/visibility"),
            headers=self._headers(json_body=True),
            json={"is_public": is_public},
            timeout=30,
        )
        self._handle_response(resp, not_found=SubmissionNotFoundError)
        if resp.status_code != 200:
            raise _failed(SubmissionError, "set_submission_visibility", resp)
        return resp.json()

    def recent_replays(self, challenge_id: int, limit: int = 10) -> dict:
        """List recent completed replays for a challenge.

        Mirrors `GET /api/challenges/{id}/recent-replays`. Each replay
        carries `simulation_id`, `created_at_ts`, `render_delay_second`,
        `signed_url` (replay file) and `participants`; each participant
        carries `submission_name`, `user_name`, `submission_reward`,
        `game_outcome` and `final_rank` — the SubmissionResult columns under
        their own names. Only runs from the last 59 days are listed: the
        render bucket deletes blobs after 60 days, so an older `signed_url`
        would 404.
        """
        resp = self._request("GET",
            self._url(f"/challenges/{challenge_id}/recent-replays"),
            headers=self._headers(),
            params={"limit": limit},
            timeout=30,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "recent_replays", resp)
        return resp.json()

    def tail_logs(self, challenge_id: int, submission_id: int, *,
                  follow: bool = False, poll_sec: float = 5.0,
                  timeout_sec: float | None = None) -> Iterator[str]:
        """Yield human-readable status / log lines for a submission.

        Polls `submission_status` and emits one line per status transition,
        including queue position for `deploy_queue` and, for each run of the
        attempt, the job's `job_status` with the steps / reward / outcome of
        **your own row** of that run (the `submission_results` entry whose
        `submission_id` is this submission's), then an
        `agent error[<agent_error_type>]: <agent_error_message>` line when
        your agent is why the run ended and an `env error[<env_error_type>]`
        line when the challenge's code is (its message is the creator's, so
        it is not printed). A line is emitted only when what it says changed:
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

        Note: this does NOT stream pod stdout — that's only available via
        the per-game signed URLs from `submission_games()`. For raw logs of a
        completed run, do:

            for game in client.submission_games(sid)["games"]:
                if game["signed_url"]:
                    print(requests.get(game["signed_url"]).text)

        Raises:
            SubmissionError: on `timeout_sec` elapsing.
        """
        last_signature = None
        last_queue_line = None
        # Per run index: the last line emitted for it. A run's line repeats
        # across polls until its steps / reward / outcome move.
        last_run_lines: dict[int, tuple] = {}
        deadline = (time.monotonic() + timeout_sec) if timeout_sec else None

        while True:
            status = self.submission_status(challenge_id, submission_id)
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

            # `run_info` is served in every post-deploy state now, so the runs
            # of a failed attempt are still here to print.
            ri = status.get("run_info") or {}
            for index, run in enumerate(ri.get("results") or []):
                # A run lists every agent's row under `submission_results`;
                # ours is the row whose `submission_id` is this submission's.
                # Direct key access on the shape: a backend serving another
                # run model fails here instead of printing `None` forever.
                mine = next((row for row in run["submission_results"]
                             if row["submission_id"] == submission_id), {})
                run_line = (f"  run: job_status={run.get('job_status')} "
                            f"steps={mine.get('agent_nb_steps')} "
                            f"reward={mine.get('submission_reward')} "
                            f"outcome={mine.get('game_outcome')}")
                error_lines = tuple(line for line in (
                    _error_line("agent", mine.get("agent_error_type"),
                                mine.get("agent_error_message")),
                    _error_line("env", run.get("env_error_type"),
                                run.get("env_error_message")),
                ) if line is not None)
                if last_run_lines.get(index) == (run_line, error_lines):
                    continue
                last_run_lines[index] = (run_line, error_lines)
                yield run_line
                yield from error_lines

            # The server decides what "done" means; direct key access so a
            # backend that does not send the block fails loudly here.
            if status["is_settled"] and not follow:
                return

            if deadline and time.monotonic() > deadline:
                raise SubmissionError(
                    f"tail_logs timed out after {timeout_sec}s: submission "
                    f"{submission_id} is still {status.get('status')!r}. Poll "
                    f"again with submission_status({challenge_id}, "
                    f"{submission_id}) or raise timeout_sec."
                )

            time.sleep(poll_sec)

    # ---- One-shot submit helper (still used by the notebook flow) ----

    def submit(self, challenge_id: int, agent=None, files=None,
               submission_name: str | None = None,
               runtime_id: int | None = None,
               runtime: dict | None = None,
               wait: bool = False,
               timeout_sec: float | None = None,
               poll_sec: float = 5.0) -> dict:
        """Convenience: create submission → (pick runner) → upload files → deploy.

        Either pass `agent=<class>` (its source is uploaded as `agent.py`) or
        `files=[...]` (each path is uploaded under its basename).
        `submission_name` defaults to the class name, or to the first file's
        base name without extension.

        Optional runner selection (between create and deploy):
            runtime_id: numeric `DockerImageAgentRuntime.id`.
            runtime:    dict spec resolved via `resolve_runtime()`, e.g.
                        `{"language": "python", "framework": "torch"}`. The
                        first runtime matching every supplied key is used.
        Pass at most one of `runtime_id` / `runtime`. With neither, the
        backend's challenge-default runtime is kept.

        After the uploads, the submission's `is_deployable` is checked (one
        `submission_status` call) and a `SubmissionError` carrying the
        server's `last_status_message` (and, on `.body`, that status payload)
        is raised rather than deploying files the backend already rejected.

        With `wait=True`, the call returns only once the deploy has settled:
        it drains `tail_logs()` (a client-side composition of the same public
        routes — there is no wait endpoint) and adds the final status block
        under `"status"`. An attempt that ends `deploy_failed` raises
        `SubmissionError` whose message is `latest_deploy.failure_message`
        (or `last_status_message`) and whose `.body` is that final status
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
        `"status"` when `wait=True`.
        """
        if (agent is None) == (files is None):
            raise SubmissionError("Provide exactly one of agent= or files=")
        if runtime_id is not None and runtime is not None:
            raise SubmissionError("Provide at most one of runtime_id= or runtime=")

        created = self.create_submission(
            challenge_id, submission_name or _default_submission_name(agent, files)
        )
        submission_id = created["submission_id"]
        # Remembered now, not after the deploy: `status()` must find the
        # submission that a rejected upload or a refused deploy leaves behind.
        self._last_submission_id = submission_id
        self._last_challenge = challenge_id

        # Pin runner before deploy so the JobPod uses the right image.
        if runtime is not None:
            runtime_id = self.resolve_runtime(challenge_id, **runtime)["id"]
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
            status = self.submission_status(challenge_id, submission_id)
            if not status["is_deployable"]:
                raise SubmissionError(
                    f"Submission {submission_id} did not pass upload validation: "
                    f"{status['last_status_message'] or 'files rejected'} "
                    f"(fix the files and re-upload, or delete the submission with "
                    f"delete_submission({challenge_id}, {submission_id})).",
                    body=status,
                )

            deploy = self.deploy_submission(challenge_id, submission_id)
            result = {
                "submission_id": submission_id,
                "deploy": deploy,
            }
            if wait:
                # Drain the generator: tail_logs() stops on `is_settled` and
                # raises on timeout, so this is the whole wait.
                for _ in self.tail_logs(challenge_id, submission_id,
                                        poll_sec=poll_sec,
                                        timeout_sec=timeout_sec):
                    pass
                final = self.submission_status(challenge_id, submission_id)
                if final["status"] == "deploy_failed":
                    # Same contract as the rejected upload above: the server's
                    # reason is the message, the payload rides on `.body`.
                    latest = final.get("latest_deploy") or {}
                    reason = (latest.get("failure_message")
                              or final.get("last_status_message"))
                    raise SubmissionError(
                        reason or f"Submission {submission_id} ended in "
                                  f"deploy_failed without a message",
                        body=final,
                    )
                result["status"] = final
            return result
        finally:
            if tmp_dir:
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)

    def status(self, submission_id: int | None = None,
               challenge_id: int | None = None) -> dict:
        """Return the rich status of a previously-made submission.

        Defaults to the last submission made through this client. Both
        submission_id and challenge_id are required by the backend route, so
        callers using a non-default submission_id must also pass challenge_id.
        Returns the same payload as `submission_status` (the status block,
        `queue_info`, `run_info`, `latest_deploy`), which is more informative
        than `submission_deploy_status`.
        """
        submission_id = submission_id or self._last_submission_id
        challenge_id = challenge_id or self._last_challenge
        if submission_id is None or challenge_id is None:
            raise SubmissionError(
                "submission_id and challenge_id are required (no previous submission found)"
            )
        return self.submission_status(challenge_id, submission_id)

    # ---- Leaderboard (public read) ----

    def leaderboard(self, challenge_id: int | None = None, top: int | None = None, *,
                    aggregate: str | None = None, course_id: int | None = None,
                    me: bool = False, q: str | None = None,
                    window: int | None = None):
        """Get the leaderboard for a challenge.

        Mirrors `GET /api/leaderboard/challenge/{id}` (`leaderboard.py`) with
        the query keys the console sends — `limit` (here `top`), `aggregate`,
        `course_id`, `me`, `q`, `window` — each sent only when passed.

        - ``aggregate="user"``: one row per participant (their best
          submission; a team's best for a team), **the console's default**.
          The backend also accepts ``"submission"``, its own default: every
          ranked submission is a row, which is what omitting it gives here.
        - ``course_id``: read the board through a course. Only that course's
          students are listed, rows gain ``passed`` when the course sets a bar
          (tri-state: None while the row has no ranked value), and the
          envelope carries a ``course_context`` block (`course_id`,
          `pass_threshold`).
        - ``top=N``: the first N rows (every ranked row without it).
          ``me=True`` adds your own row with `rank`, `percentile` and its
          ``window`` neighbours (default 3, at most 25) under `me`; ``q``
          adds the rows whose username contains it under `matches` (at most
          50).

        **Shape.** The backend serves one envelope: `challenge`, `total`,
        `leaders`, `me`, `matches` and, with ``course_id``,
        `course_context`. With pandas, the call returns `leaders` as a
        DataFrame and every other envelope key on ``df.attrs``
        (``df.attrs["challenge"]["metric"]``, ``df.attrs["me"]``, …);
        without pandas it returns the envelope dict as served.

        **The `challenge` block** — what every row shares, served once:
        `challenge_id`, `is_elo_score`, `metric_order`, `ranked_order`,
        `metric`, `metric2`, `frontend_precision`, `metrics_schema`,
        `has_gpu`, `is_continuous`. ``ranked_order`` says which way the board
        ranks (``"desc"``: higher is better, ``"asc"``: lower is better;
        always ``"desc"`` when ``is_elo_score``), and ``metric_order`` is the
        direction of the metric itself.

        **Columns** — the backend's own names, one set with the console:
        `rank`, `username`, `avatar_key`, `submission_id`,
        `submission_name`, `mean_reward`, `mean_reward2`, `reward_ci95`,
        `n_episodes_total`, `elo_score`, `elo_variance`, `number_of_runs`,
        `created_at_ts` and `last_end_run_ts` (ISO-8601 UTC with a `Z`),
        `is_my_submission`, `team_id`, `team_name`, `team_members`,
        `action_time_max_sec`, `agent_metric_total_ram_max_bytes`,
        `agent_metric_vram_max_bytes`, `mean_metrics_detail`,
        `mean_reward_30d`, `mean_metrics_detail_30d`, `is_public` (None when
        the row is not yours to know) and, through a course with a bar,
        `passed`. Rows come in server rank order.
        """
        challenge_id = challenge_id or self._last_challenge
        if challenge_id is None:
            raise SubmissionError("No challenge specified and no previous submission found")
        params: dict = {}
        if top is not None:
            params["limit"] = top
        if aggregate is not None:
            params["aggregate"] = aggregate
        if course_id is not None:
            params["course_id"] = course_id
        if window is not None:
            params["window"] = window
        if me:
            params["me"] = "true"
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
        rows = pd.DataFrame(envelope["leaders"])
        rows.attrs.update({k: v for k, v in envelope.items() if k != "leaders"})
        return rows

    def global_ranking(self, search: str | None = None, page: int = 1, per_page: int = 100):
        """Global user ranking across all challenges (points + medals).

        Mirrors `GET /api/ranking/` (`ranking.py`). Returns a DataFrame of the
        requested page (default: top 100) with the server's own column names:
        `user_id`, `username`, `avatar_key`, `rank`, `current_points`,
        `medals_gold`, `medals_silver`, `medals_bronze`. Pass ``search`` to
        filter by username; matched rows carry their true global rank.

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
        `medals_silver`, `medals_bronze`. The route used to nest all but the
        first two under a `stats` key that this method silently unwrapped;
        both the wrapper and the unwrap are gone.
        """
        resp = self._request("GET", self._url(f"/ranking/user/{user_id}"),
                             headers=self._headers(), timeout=30)
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "user_global_rank", resp)
        return resp.json()

    # ---- Live data sources (public read proxy over the collection sidecar) --
    #
    # The `data_source_enabled` challenges are scored on data the platform
    # collects (hourly weather for a city registry, a news feed). These
    # methods mirror `GET /api/data_sources/*` (`views/data_sources.py`): no
    # token scope is needed, the routes are public (the bearer token is sent
    # anyway, as on every read). A challenge names its
    # slice in `challenge(id)["data_source_asset"]`. Timestamps are naive UTC
    # ISO-8601 strings (no offset suffix). When the sidecar is down the server
    # answers 503 `{"error", "upstream": "data_collection"}`, raised here as
    # `MLArenaError` with that reason — and so does a payload that no longer
    # matches the contract, a key the sidecar *added* included: the keys below
    # are the whole payload, not a subset of it.

    def _data_source_get(self, path: str, params: dict | None = None):
        resp = self._request("GET", self._url(f"/data_sources{path}"),
                             params=params, headers=self._headers(), timeout=30)
        self._handle_response(resp, not_found=NotFoundError)
        if resp.status_code != 200:
            raise _failed(MLArenaError, f"data source {path}", resp)
        return resp.json()

    def data_source_weather_coverage(self, days: int = 60) -> dict:
        """Day-by-day completeness of the weather table over the last ``days``
        (1..400).

        Mirrors `GET /api/data_sources/weather/coverage`. Returns
        ``{"expected_cities", "first_hour", "last_hour", "days": [{"date",
        "hours_present", "hours_complete", "rows", "rows_by_source"}]}`` —
        one entry per calendar day that has any row, oldest first;
        ``hours_complete`` counts the hours with a row for every city.
        ``rows_by_source`` always carries both source keys
        (``openweathermap``, ``open_meteo_archive``), zero included.
        """
        return self._data_source_get("/weather/coverage", {"days": days})

    def data_source_weather_series(self, city_name: str, country_code: str,
                                   hours: int = 168) -> dict:
        """One city's hourly weather rows over the last ``hours`` (1..9600).

        Mirrors `GET /api/data_sources/weather/series`. ``city_name`` /
        ``country_code`` are a pair from :meth:`data_source_weather_cities`
        (the same keys every weather row carries); any other raises
        ``NotFoundError``. Returns ``{"city_name", "country_code", "latitude",
        "longitude", "points": [{"timestamp", "temperature", "rain",
        "wind_speed", "wind_direction", "humidity", "clouds", "visibility",
        "snow", "pressure", "apparent_temperature", "wind_gust", "source"}]}``,
        oldest first; a missing hour is simply absent, and ``pressure`` /
        ``apparent_temperature`` / ``wind_gust`` are null on older rows.
        """
        return self._data_source_get(
            "/weather/series",
            {"city_name": city_name, "country_code": country_code, "hours": hours})

    def data_source_weather_snapshot(self, hour: str | None = None) -> dict:
        """Every city's weather at one hour (default: the latest hour present).

        Mirrors `GET /api/data_sources/weather/snapshot`. ``hour`` is an
        ISO-8601 timestamp on the hour (``"2026-09-18T06:00:00"``); an offset
        is accepted and converted to UTC, a value off the hour is a 400
        (the weather table is keyed by hour slot). Returns
        ``{"timestamp", "cities": [{"city_name", "country_code", "latitude",
        "longitude", "temperature", "rain", "wind_speed", "wind_direction",
        "humidity", "clouds", "pressure", "source"}]}``.
        """
        params = {"hour": hour} if hour is not None else None
        return self._data_source_get("/weather/snapshot", params)

    def data_source_weather_cities(self) -> list:
        """The weather city registry, sorted by ``city_name``.

        Mirrors `GET /api/data_sources/weather/cities`. Returns
        ``[{"city_name", "country_code", "latitude", "longitude"}]``.
        """
        return self._data_source_get("/weather/cities")

    def data_source_news_volume(self, days: int = 30) -> dict:
        """Articles per day and per source over the last ``days`` (1..400).

        Mirrors `GET /api/data_sources/news/volume`. Returns ``{"days":
        [{"date", "total", "by_family"}], "sources": [{"source_key",
        "source_label", "source_family", "total", "avg_chars", "latest"}],
        "total"}`` — days oldest first, sources by total descending, and
        ``total`` the articles collected over the whole window (served, so it
        is not summed from ``days`` by every caller). A source is listed only
        because it has articles in the window, so its ``avg_chars`` and
        ``latest`` are never null; ``/news/sources`` is the route where they
        can be.
        """
        return self._data_source_get("/news/volume", {"days": days})

    def data_source_news_sources(self, hours: int = 24) -> list:
        """Per-source article counts over the last ``hours`` (1..9600).

        Mirrors `GET /api/data_sources/news/sources`. Returns
        ``[{"source_key", "source_label", "source_family", "n_items",
        "min_chars", "avg_chars", "latest", "window_hours"}]`` by count
        descending — the panel-sizing view: a source with 0 rows here cannot
        be a daily class.
        """
        return self._data_source_get("/news/sources", {"hours": hours})

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

        Mirrors `GET /api/chat/challenge/{cid}` (`backend/app/views/chat/`). Returns `challenge_id`, `challenge_name`, `is_started`,
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
        `session_count`, `rank`), `sessions` (yours and your teammates',
        newest first) and `open_session_id` (your open session, or None).

        Raises `ChallengeNotFoundError` when the challenge is not a chat
        challenge or is hidden from you.
        """
        return self._chat_call(
            "GET", f"/chat/challenge/{challenge_id}",
            error_label="chat_challenge",
        ).json()

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
        challenge is not started, the agent is offline, or
        `max_sessions_per_participant` is reached).
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
        `progress` while running, `error_message` when failed; None before the
        first message), `participant` (your group's `total_amount_eur` and
        `scoreboard`, the same keys `chat_challenge()` serves) and `can_send`
        with `can_send_reason` (`session_voided` | `session_closed` |
        `turn_in_flight` | `challenge_not_started` | `agent_offline` |
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
        poll will serve. `content` is 1..4000 characters.

        With `wait=True` (default) the call then polls `chat_session()` every
        `poll_interval` seconds — a client-side composition of the two public
        routes, the console's own cadence; there is no wait endpoint — until
        **that** turn's `status` is `completed` or `failed`, and returns the
        final `ChatSessionView`: the assistant's reply is the message whose
        `id` is `turn["assistant_message_id"]`, and the events this turn
        earned are the `scoring_events` with `turn_id == turn["id"]`. A
        `failed` turn raises `MLArenaError` carrying the turn's
        `error_message`; the session stays open, so you may send again. Not
        settled after `timeout` seconds raises `MLArenaError` too, naming the
        state the turn was left in.

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
        settling, so it is skipped rather than misread."""
        deadline = time.monotonic() + timeout
        while True:
            view = self.chat_session(session_id)
            turn = view["turn"]
            if (turn is not None and turn["id"] == turn_id
                    and turn["status"] in ("completed", "failed")):
                if turn["status"] == "failed":
                    raise MLArenaError(
                        f"chat turn {turn_id} failed: {turn['error_message']}"
                    )
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

        Raises `ChatSessionNotFoundError` when the session is not yours.
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
        charter), waits for the answer, prints the scoring events it earned
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
        `unvoid_chat_session()`.

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
        """Multipart helper for the media/cover upload routes (sibling of
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

        Mirrors `POST /api/academic_courses/` (`academic_courses/legacy.py`).
        Creating a course is the self-serve teacher-onboarding action: it works
        with any valid token scope and flips the owning account to a teacher.
        Each account may own at most `max_courses_limit` courses (default 1);
        admins are exempt. After creating, mint a `teacher`-scope key from your
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
        has_student_info}`; the course sits under `course` with its own field
        names (`id`, `name`, `code`, `slug`, ...), and the two flags stay false
        for an anonymous caller.
        """
        return self._course_call(
            "GET", f"/academic_courses/enroll/{join_code}",
            error_label="enrollment_info",
        )

    def enroll_in_course(self, join_code: str | None = None,
                         student_email: str | None = None,
                         student_number: str | None = None,
                         project_url: str | None = None) -> dict:
        """Enroll the authenticated user in a course.

        Mirrors `POST /api/academic_courses/enroll/<join_code>`. A course has a
        single enrollment token, its `join_code` — pass it positionally or as
        `join_code=`. A full `/enroll/<join_code>` URL is not accepted; pass the
        code itself.

        `student_email` / `student_number` fill the caller's global student
        identity when it isn't set yet (they never overwrite values already on
        the profile — set those via the Profile page / profile update). The
        backend requires both to be present before enrolling. `project_url` is
        stored on the enrollment. Returns `{message, challenge_ids,
        course_slug}`.
        """
        if join_code is None:
            raise MLArenaError("enroll_in_course requires a join code")
        body: dict = {}
        if student_email is not None:
            body["student_email"] = student_email
        if student_number is not None:
            body["student_number"] = student_number
        if project_url is not None:
            body["project_url"] = project_url
        return self._course_call(
            "POST", f"/academic_courses/enroll/{join_code}", json_body=body,
            ok=(200, 201), error_label="enroll_in_course",
        )

    def list_courses(self, *, show_all: bool = False,
                     challenge_id: int | None = None):
        """List courses (enrolled + currently-active by default).

        Mirrors `GET /api/academic_courses/`. With `show_all=True` returns every
        course; with `challenge_id` returns courses attached to that
        challenge. Each row carries `is_enrolled` for the caller. Returns a
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

    def course(self, slug: str) -> dict:
        """Course landing: meta + ordered modules + lesson TOC + own progress.

        Mirrors `GET /api/academic_courses/{slug}`. Lesson bodies are omitted
        here (use `lesson(...)`). When authenticated, the response includes
        `is_enrolled`, `can_manage`, and a `progress` summary; managers
        (teacher/TA/admin) additionally see unpublished draft lessons.
        `has_cover` says whether there is a cover image — fetch it with
        `course_cover(landing["id"])`.
        """
        return self._course_call(
            "GET", f"/academic_courses/{slug}", error_label="course",
        )

    def module_overview(self, slug: str, module_slug: str) -> dict:
        """A course module's overview: lesson list + attached challenge cards.

        Mirrors `GET /api/academic_courses/{slug}/modules/{module_slug}`.
        """
        return self._course_call(
            "GET", f"/academic_courses/{slug}/modules/{module_slug}",
            error_label="module_overview",
        )

    def lesson(self, slug: str, module_slug: str, lesson_slug: str) -> dict:
        """Full lesson body + server-resolved `mlarena:` directive payloads.

        Mirrors
        `GET /api/academic_courses/{slug}/modules/{module_slug}/lessons/{lesson_slug}`.
        A gated lesson requires enrollment (or manage rights) — otherwise the
        backend returns 403, surfaced here as `AuthenticationError`. The
        returned dict carries `body_md`, `directives`, and `directive_warnings`.
        """
        return self._course_call(
            "GET",
            f"/academic_courses/{slug}/modules/{module_slug}/lessons/{lesson_slug}",
            error_label="lesson",
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
        """Own progress across a course: content % + next lesson + challenge
        results.

        Mirrors `GET /api/academic_courses/{course_id}/progress/me`. Requires
        enrollment (or manage rights). Each challenge cell carries `value`,
        `pass_threshold`, `passed` (null when there is nothing to judge) and
        `ranked_order` ("desc": `passed` means value >= bar, "asc": <=).
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
        """Module detail + lesson TOC + challenge links (owner or public).

        Mirrors `GET /api/teacher/modules/{id}`.
        """
        return self._course_call(
            "GET", f"/teacher/modules/{module_id}", error_label="get_module",
        )

    def update_module(self, module_id: int, **fields) -> dict:
        """Update module meta — owner only. Mirrors `PUT /api/teacher/modules/{id}`.

        Accepts `title`, `summary`, `icon`, `visibility`, `is_published` (the
        slug is immutable for URL stability). Only the fields you pass are
        changed, so `update_module(id, is_published=False)` sends a module back
        to draft without touching anything else.
        """
        allowed = {"title", "summary", "icon", "visibility", "is_published"}
        body = _filtered_fields(fields, allowed, "update_module")
        return self._course_call(
            "PUT", f"/teacher/modules/{module_id}", json_body=body,
            error_label="update_module",
        )

    def delete_module(self, module_id: int, force: bool = False) -> dict:
        """Delete a module — owner only.

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
        """Deep-copy a readable (owned or public) module into a new owned module.

        Mirrors `POST /api/teacher/modules/{id}/fork`. Copies lessons (nesting
        preserved) + challenge attachments; the fork starts `private` and
        records provenance via `forked_from_module_id`.
        """
        return self._course_call(
            "POST", f"/teacher/modules/{module_id}/fork",
            ok=(200, 201), error_label="fork_module",
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
        the challenge ranks (`>=` when its `ranked_order` is "desc", `<=` when
        it is "asc"; the returned link carries `ranked_order`). Omit it for no
        pass/fail — do not pass 0, which would validate every entrant.
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

        Mirrors `PUT /api/teacher/lessons/{id}`. Accepts `title`, `body_md`,
        `is_published`, `gated`, `estimated_minutes`. Only the fields you pass
        are changed.
        """
        allowed = {"title", "body_md", "is_published", "gated", "estimated_minutes"}
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

    def upload_lesson_media(self, lesson_id: int, file_path: str) -> dict:
        """Upload a file for a lesson; returns `{filename, url}`.

        Mirrors `POST /api/teacher/lessons/{id}/media`. Any file type: put the
        returned `url` in the lesson body as `![alt](url)` for an image, or as
        `[label](url)` for anything else (notebook, dataset, handout) — the
        asset route serves non-images as a download.
        """
        return self._upload_course_file(
            "POST", f"/teacher/lessons/{lesson_id}/media", file_path,
            error_label="upload_lesson_media",
        )

    def list_lesson_media(self, lesson_id: int) -> list:
        """The files attached to a lesson: `[{filename, url, size_bytes}]`.

        Mirrors `GET /api/teacher/lessons/{id}/media` (owner scope). Use it to
        find a file whose link is no longer in the body — the upload response is
        otherwise the only place its URL ever appeared.
        """
        return self._course_call(
            "GET", f"/teacher/lessons/{lesson_id}/media",
            error_label="list_lesson_media",
        )

    def download_lesson_media(self, lesson_id: int, filename: str,
                              dest_dir: str = ".") -> str:
        """Download one lesson file to `dest_dir`; returns the written path.

        Mirrors `GET /api/academic_courses/assets/lessons/{id}/{filename}` — the
        *consumption* route, so this is the call a student (not just the author)
        makes to fetch a lab notebook or dataset a lesson links. Gated lessons
        require enrollment; the bearer token carries that.
        """
        os.makedirs(dest_dir, exist_ok=True)
        resp = self._request(
            "GET", self._url(f"/academic_courses/assets/lessons/{lesson_id}/{filename}"),
            headers=self._headers(), timeout=300, stream=True,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise _failed(MLArenaError, "download_lesson_media", resp)
        out_path = os.path.join(dest_dir, os.path.basename(filename))
        with open(out_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    fh.write(chunk)
        return out_path

    def delete_lesson_media(self, lesson_id: int, filename: str) -> dict:
        """Delete a lesson media file.

        Mirrors `DELETE /api/teacher/lessons/{id}/media/{filename}`.
        """
        return self._course_call(
            "DELETE", f"/teacher/lessons/{lesson_id}/media/{filename}",
            error_label="delete_lesson_media",
        )

    def preview_lesson(self, lesson_id: int, body_md: str | None = None) -> dict:
        """Server-validate a lesson body's `mlarena:` directives (strict mode).

        Mirrors `POST /api/teacher/lessons/{id}/preview`. Pass `body_md` to
        validate an unsaved buffer; omit it to validate the saved body. An
        unknown/unresolvable directive fails loud (400 → `MLArenaError`) so it's
        caught before publishing. Returns `{directives, warnings}`.
        """
        body = {"body_md": body_md} if body_md is not None else {}
        return self._course_call(
            "POST", f"/teacher/lessons/{lesson_id}/preview", json_body=body,
            error_label="preview_lesson",
        )

    # ---- Course authoring: course composition (teacher scope) ---------------

    def update_course(self, course_id: int, **fields) -> dict:
        """Update course meta — teacher/TA/admin.

        Mirrors `PUT /api/teacher/course/{id}`. Accepts `name`, `code`,
        `description`, `visibility`, `instructor_name`, `start_date`, `end_date`
        (ISO date strings). Only the fields you pass are changed.
        """
        allowed = {"name", "code", "description", "visibility",
                   "instructor_name", "start_date", "end_date"}
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
        """List a course's linked modules in order, each with an owned/borrowed flag.

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
        Challenge metas and cells carry `ranked_order` next to `pass_threshold`
        (see `my_progress`).
        """
        return self._course_call(
            "GET", f"/teacher/course/{course_id}/progress",
            error_label="course_progress",
        )

    # ---- Course administration: roster, assistants, exports ----------------

    def teacher_courses(self):
        """The courses you teach or assist on (admins see them all).

        Mirrors `GET /api/teacher/courses` — the console's course-authoring
        list. Each row is a course plus `role` (`teacher` | `assistant`) and
        `has_cover` (the image itself comes from `course_cover(id)`).
        Returns a DataFrame if pandas is installed, else a list of dicts.
        """
        return _to_dataframe(self._course_call(
            "GET", "/teacher/courses", error_label="teacher_courses",
        ))

    def challenges_for_course(self):
        """The challenges you may attach to a module.

        Mirrors `GET /api/teacher/challenges-for-course`: public challenges plus
        your own, each with `ranked_by` (the metric a threshold is on),
        `ranked_order` (the direction it is compared in) and `precision`.
        A challenge with no evaluation is not listed, because `attach_challenge`
        refuses it.
        """
        return _to_dataframe(self._course_call(
            "GET", "/teacher/challenges-for-course",
            error_label="challenges_for_course",
        ))

    def course_students(self, course_id: int,
                        challenge_id: int | None = None) -> dict:
        """The course roster, with team membership for one attached challenge.

        Mirrors `GET /api/teacher/students/{course_id}`. Enrollment is
        course-level, so the student list is the same whichever challenge you
        pass; only `team_id`/`team_name` change. Without `challenge_id` no
        challenge is picked: `selected_challenge_id` and every `team_*` are
        None (the route no longer defaults to the course's first challenge).
        A challenge not attached to the course is a 400. Returns
        `{challenges, selected_challenge_id, students}`, where each student
        carries `user_id`, `student_number`, `student_email`, `project_url` and
        `enrolled_at_ts`.
        """
        params = {"challenge_id": challenge_id} if challenge_id is not None else None
        return self._course_call(
            "GET", f"/teacher/students/{course_id}", params=params,
            error_label="course_students",
        )

    def remove_student(self, course_id: int, user_id: int) -> dict:
        """Remove a student's enrollment from a course.

        Mirrors `DELETE /api/teacher/student/{course_id}/{user_id}`. Their
        submissions are untouched — only the enrollment row goes.
        """
        return self._course_call(
            "DELETE", f"/teacher/student/{course_id}/{user_id}",
            error_label="remove_student",
        )

    def course_assistants(self, course_id: int):
        """The teaching assistants on a course (course owner only).

        Mirrors `GET /api/teacher/course/{id}/assistants`. Each row carries
        `user_id`, `username` and `created_at_ts`. Returns a DataFrame if
        pandas is installed, else a list of dicts.
        """
        return _to_dataframe(self._course_call(
            "GET", f"/teacher/course/{course_id}/assistants",
            error_label="course_assistants",
        ))

    def add_course_assistant(self, course_id: int, username: str) -> dict:
        """Add a teaching assistant by username (course owner only).

        Mirrors `POST /api/teacher/course/{id}/assistants`. A TA gets the same
        teacher view of the course, but cannot manage the assistant list.
        """
        return self._course_call(
            "POST", f"/teacher/course/{course_id}/assistants",
            json_body={"username": username}, ok=(200, 201),
            error_label="add_course_assistant",
        )

    def remove_course_assistant(self, course_id: int, user_id: int) -> dict:
        """Remove a teaching assistant (course owner only).

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
        best submission) instead of one row per submission.
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
        """Stream a course route's file body to ``out_path`` (sibling of
        ``download_lesson_media``); returns the written path."""
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

    def author_course_from_dir(self, path: str) -> dict:
        """Create/update a whole course from a local directory + `course.yaml`.

        The "write your course via the SDK" ergonomic — a pure composition of
        the authoring methods above (no new endpoint), mirroring how `submit()`
        bundles create + upload + deploy. Requires a `teacher`-scope token.

        Directory layout::

            my-course/
              course.yaml            # or course.json — the manifest
              cover.png              # optional, referenced from the manifest
              foundations/
                what-is-rl.md        # lesson bodies referenced from the manifest

        Manifest (`course.yaml`)::

            course:
              name: "Intro to RL"
              code: "RL101"          # optional
              slug: intro-to-rl      # optional (auto-derived)
              description: "..."     # optional markdown
              visibility: public     # private | unlisted | public
              instructor_name: "Prof X"
              start_date: 2026-01-01
              end_date: 2026-06-01
              cover: cover.png       # optional, relative to the dir
              course_id: 12          # optional — update this course instead of creating
            modules:
              - title: "Foundations"
                slug: foundations    # optional
                summary: "..."       # optional
                icon: book           # optional
                visibility: public   # optional (default private) — teacher reuse
                is_published: true   # optional (default true) — false = hidden from students
                module_id: 7         # optional — link this existing module instead of creating
                challenges:          # optional (`competitions:` also accepted)
                  - challenge_id: 42        # (`competition_id:` also accepted)
                    label: "CartPole"
                    pass_threshold: 195.0   # optional — score that validates it
                lessons:
                  - title: "What is RL?"
                    slug: what-is-rl       # optional
                    kind: lesson           # lesson | exercise
                    file: foundations/what-is-rl.md   # body markdown, relative to the dir
                    body_md: "inline..."   # alternative to `file`
                    gated: false           # optional
                    is_published: true     # optional (default false / draft)
                    estimated_minutes: 15  # optional

        Idempotency is explicit, not magical: pass `course.course_id` to update
        an existing course's meta, and a module's `module_id` to link an existing
        module rather than create a new one. Returns a summary
        `{course_id, slug, join_code, modules: [...]}`.

        Manifests written before SDK 2.0 name a module's challenge list
        `competitions:` and each entry's id `competition_id:`. Both spellings
        are still read, permanently; `export_course_to_dir` writes
        `challenges:` / `challenge_id:`. A module or entry that carries both
        spellings is rejected.
        """
        base = os.path.abspath(path)
        manifest = _load_course_manifest(base)
        course_spec = dict(manifest.get("course") or {})
        modules_spec = manifest.get("modules") or []

        existing_course_id = course_spec.pop("course_id", None)
        cover = course_spec.pop("cover", None)
        # Resolve every module's challenge links before the first request, so
        # a malformed list fails before anything is created.
        challenge_links = [
            [(_manifest_challenge_id(link), link) for link in _manifest_challenge_links(m)]
            for m in modules_spec
        ]

        if existing_course_id is not None:
            self.update_course(existing_course_id, **_course_meta_fields(course_spec))
            course_id = existing_course_id
            course_out = {"id": course_id, "slug": course_spec.get("slug")}
        else:
            course_out = self.create_course(
                name=course_spec["name"],
                code=course_spec.get("code"),
                start_date=course_spec["start_date"],
                end_date=course_spec["end_date"],
                instructor_name=course_spec.get("instructor_name"),
                slug=course_spec.get("slug"),
                description=course_spec.get("description"),
                visibility=course_spec.get("visibility"),
                teacher_user_id=course_spec.get("teacher_user_id"),
            )
            course_id = course_out["id"]

        if cover:
            self.set_course_cover(course_id, os.path.join(base, cover))

        module_summaries = []
        ordered_module_ids = []
        for position, m in enumerate(modules_spec):
            module_id = m.get("module_id")
            if module_id is None:
                created = self.create_module(
                    title=m["title"], slug=m.get("slug"), summary=m.get("summary"),
                    icon=m.get("icon"), visibility=m.get("visibility", "private"),
                    is_published=bool(m.get("is_published", True)),
                )
                module_id = created["id"]
                lesson_ids = []
                for lesson_spec in (m.get("lessons") or []):
                    lesson_ids.append(
                        self._author_lesson(base, module_id, lesson_spec)["id"]
                    )
                for challenge_id, link in challenge_links[position]:
                    self.attach_challenge(
                        module_id, challenge_id,
                        label=link.get("label"), position=link.get("position"),
                        pass_threshold=link.get("pass_threshold"),
                    )
                module_summaries.append(
                    {"module_id": module_id, "title": m["title"],
                     "lesson_ids": lesson_ids, "created": True}
                )
            else:
                module_summaries.append(
                    {"module_id": module_id, "title": m.get("title"),
                     "lesson_ids": [], "created": False}
                )

            self.link_module(course_id, module_id, position=position)
            ordered_module_ids.append(module_id)

        # Pin the manifest's module order (link positions can drift if some were
        # already linked from a prior run).
        if len(ordered_module_ids) > 1:
            self.reorder_modules(course_id, ordered_module_ids)

        return {
            "course_id": course_id,
            "slug": course_out.get("slug"),
            "join_code": course_out.get("join_code"),
            "modules": module_summaries,
        }

    def _author_lesson(self, base: str, module_id: int, spec: dict) -> dict:
        """Create one lesson from a manifest entry (body from `file` or inline
        `body_md`), then publish/set the fields `create_lesson` doesn't take."""
        body_md = spec.get("body_md", "")
        if spec.get("file"):
            with open(os.path.join(base, spec["file"]), "r", encoding="utf-8") as fh:
                body_md = fh.read()

        lesson = self.create_lesson(
            module_id, title=spec["title"], kind=spec.get("kind", "lesson"),
            slug=spec.get("slug"), parent_lesson_id=spec.get("parent_lesson_id"),
            body_md=body_md, gated=bool(spec.get("gated", False)),
        )

        # is_published / estimated_minutes are only settable via update.
        updates = {}
        if spec.get("is_published"):
            updates["is_published"] = True
        if spec.get("estimated_minutes") is not None:
            updates["estimated_minutes"] = spec["estimated_minutes"]
        if updates:
            lesson = self.update_lesson(lesson["id"], **updates)
        return lesson

    def export_course_to_dir(self, slug: str, path: str) -> dict:
        """Export a course to a local directory of markdown + a `course.yaml`.

        The inverse of `author_course_from_dir` — for backup / versioning a
        course as files. Reads the public consumption surface (landing + lesson
        bodies), so it works for a course you can view (public, or one you own /
        are enrolled in; owners also get draft lessons *and draft modules* — for
        anyone else an unpublished module is simply absent, so exporting someone
        else's course gives you the published cut of it). Returns the manifest dict.
        """
        landing = self.course(slug)
        base = os.path.abspath(path)
        os.makedirs(base, exist_ok=True)

        modules_out = []
        for module in landing.get("modules", []):
            module_slug = module["slug"]
            module_dir = os.path.join(base, module_slug)
            os.makedirs(module_dir, exist_ok=True)

            lessons_out = []
            for toc in module.get("lessons", []):
                full = self.lesson(slug, module_slug, toc["slug"])
                rel = os.path.join(module_slug, f"{toc['slug']}.md")
                with open(os.path.join(base, rel), "w", encoding="utf-8") as fh:
                    fh.write(full.get("body_md", ""))
                lessons_out.append({
                    "title": toc["title"],
                    "slug": toc["slug"],
                    "kind": toc.get("kind", "lesson"),
                    "file": rel,
                    "gated": toc.get("gated", False),
                    "is_published": toc.get("is_published", True),
                    "estimated_minutes": toc.get("estimated_minutes"),
                })

            modules_out.append({
                "title": module["title"],
                "slug": module_slug,
                "summary": module.get("summary"),
                "icon": module.get("icon"),
                "is_published": module.get("is_published", True),
                "challenges": [
                    {
                        # The manifest key stays `challenge_id`; the landing
                        # payload names the challenge itself.
                        "challenge_id": c["challenge"]["id"],
                        "label": c.get("label"),
                        "pass_threshold": c.get("pass_threshold"),
                    }
                    for c in module.get("challenges", [])
                ],
                "lessons": lessons_out,
            })

        manifest = {
            "course": {
                "name": landing["name"],
                "code": landing.get("code"),
                "slug": landing.get("slug"),
                "description": landing.get("description"),
                "visibility": landing.get("visibility"),
                "instructor_name": landing.get("instructor_name"),
                "start_date": landing.get("start_date"),
                "end_date": landing.get("end_date"),
            },
            "modules": modules_out,
        }
        _dump_course_manifest(base, manifest)
        return manifest

    def __repr__(self) -> str:
        head = self._token.split("_", 3)
        masked = "_".join(head[:3] + ["…"]) if len(head) == 4 else "…"
        return f"MLArenaClient(base_url='{self._base_url}', token='{masked}')"


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


def _course_meta_fields(course_spec: dict) -> dict:
    """The subset of a manifest ``course:`` block that ``update_course`` accepts."""
    allowed = {"name", "code", "description", "visibility",
               "instructor_name", "start_date", "end_date"}
    return {k: v for k, v in course_spec.items() if k in allowed and v is not None}


def _load_course_manifest(base: str) -> dict:
    """Read ``course.yaml`` / ``course.yml`` (PyYAML, lazy) or ``course.json``
    from a course directory. Fails loud if none is found or YAML isn't installed
    when a YAML manifest is present (no silent fallback)."""
    import json

    yaml_path = next(
        (os.path.join(base, name)
         for name in ("course.yaml", "course.yml")
         if os.path.isfile(os.path.join(base, name))),
        None,
    )
    json_path = os.path.join(base, "course.json")

    if yaml_path is not None:
        try:
            import yaml
        except ImportError as exc:
            raise MLArenaError(
                "Reading a course.yaml manifest needs PyYAML "
                "(`pip install pyyaml`), or provide a course.json instead."
            ) from exc
        with open(yaml_path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    if os.path.isfile(json_path):
        with open(json_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    raise MLArenaError(
        f"No course manifest found in {base!r} (expected course.yaml or course.json)"
    )


# course.yaml input alias: manifests on teachers' disks predate the rename and
# say `competitions:` / `competition_id:`. Those keys are read permanently;
# export writes only the new ones. Both spellings in one block is ambiguous, so
# it fails loud instead of picking one.

def _manifest_challenge_links(module_spec: dict) -> list:
    """A manifest module's challenge list, under `challenges:` or the legacy
    `competitions:` key."""
    if "challenges" in module_spec and "competitions" in module_spec:
        raise MLArenaError(
            f"Module {module_spec.get('title')!r} has both 'challenges' and "
            f"'competitions'; keep only 'challenges'"
        )
    if "competitions" in module_spec:
        return module_spec["competitions"] or []
    return module_spec.get("challenges") or []


def _manifest_challenge_id(link_spec: dict) -> int:
    """A manifest challenge entry's id, under `challenge_id:` or the legacy
    `competition_id:` key. An entry with neither raises KeyError."""
    if "challenge_id" in link_spec and "competition_id" in link_spec:
        raise MLArenaError(
            f"Challenge entry {link_spec!r} has both 'challenge_id' and "
            f"'competition_id'; keep only 'challenge_id'"
        )
    if "competition_id" in link_spec:
        return link_spec["competition_id"]
    return link_spec["challenge_id"]


def _dump_course_manifest(base: str, manifest: dict) -> None:
    """Write a manifest as ``course.yaml`` (PyYAML if available) else ``course.json``."""
    try:
        import yaml
        with open(os.path.join(base, "course.yaml"), "w", encoding="utf-8") as fh:
            yaml.safe_dump(manifest, fh, sort_keys=False, allow_unicode=True)
    except ImportError:
        import json
        with open(os.path.join(base, "course.json"), "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, ensure_ascii=False)


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

    `side` is `agent` (the caller's own `RunSubmissionResult`) or `env` (the
    run). The message is appended only when served — it is None on a row
    that is not the caller's and on the env side of a participant read.
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
    "agent_status": "submission_status",
    "agent_games": "submission_games",
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


def _deprecated_alias(old_name, new_name):
    def alias(self, *args, **kwargs):
        warnings.warn(
            f"MLArenaClient.{old_name}() is deprecated, "
            f"use .{new_name}() instead",
            DeprecationWarning,
            stacklevel=2,
        )
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
