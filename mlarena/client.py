"""ML Arena REST client.

Authentication is via Bearer token of the form `mlk_<scope>_<lookup>_<secret>`,
matching the backend's `auth_required` decorator (see backend/app/auth/decorators.py). The
older `key_id:key_pass` format that earlier README versions documented has been
retired — the canonical form is the full underscore-separated token.

Routes hit the canonical API blueprints under `/api/...`. There is no longer a
separate `/api/sdk/` namespace.
"""

import inspect
import io
import os
import tempfile
import time
from typing import Iterator

import requests

from mlarena.exceptions import (
    AuthenticationError,
    CompetitionNotFoundError,
    MLArenaError,
    SubmissionError,
)


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
        self._last_agent_id = None
        self._last_competition = None

    def _headers(self, *, json_body: bool = False) -> dict:
        h = {"Authorization": f"Bearer {self._token}"}
        if json_body:
            h["Content-Type"] = "application/json"
        return h

    def _url(self, path: str) -> str:
        return f"{self._base_url}/api{path}"

    def _handle_response(self, resp: requests.Response) -> requests.Response:
        if resp.status_code == 401:
            raise AuthenticationError("Invalid or missing API credentials.")
        if resp.status_code == 403:
            raise AuthenticationError(_safe_error(resp, "Access denied"))
        if resp.status_code == 404:
            raise CompetitionNotFoundError(_safe_error(resp, "Not found"))
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
        enrollments). Works with any valid token scope since the token itself
        identifies the user.
        """
        resp = self._request("GET", self._url("/profile/"), headers=self._headers())
        self._handle_response(resp)
        if resp.status_code != 200:
            raise MLArenaError(f"profile failed: {_safe_error(resp)}")
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
            raise MLArenaError(f"update_profile failed: {_safe_error(resp)}")
        return resp.json()

    # ---- Competitions (public read; create/update via creator scope) ----

    def competitions(
        self,
        *,
        q: str | None = None,
        tags: list[str] | str | None = None,
        status: str | None = None,
        page: int | None = None,
        per_page: int | None = None,
    ):
        """List competitions. Public; no auth required.

        With no args, returns all matching competitions as a DataFrame
        (auto-paginates internally). Pass ``page``/``per_page`` to fetch a
        single page; pass ``q``/``tags``/``status`` to filter server-side.
        """
        base_params: dict = {}
        if q:
            base_params["q"] = q
        if status:
            base_params["status"] = status
        if tags:
            base_params["tags"] = ",".join(tags) if isinstance(tags, list) else tags

        if page is not None or per_page is not None:
            params = {**base_params, "page": page or 1, "per_page": per_page or 24}
            resp = self._request("GET",self._url("/competitions/"), params=params, timeout=30)
            resp.raise_for_status()
            return _to_dataframe(resp.json().get("items", []))

        items: list = []
        page_n = 1
        per_page_n = 100
        while True:
            params = {**base_params, "page": page_n, "per_page": per_page_n}
            resp = self._request("GET",self._url("/competitions/"), params=params, timeout=30)
            resp.raise_for_status()
            body = resp.json()
            items.extend(body.get("items", []))
            if not body.get("metadata", {}).get("has_next"):
                break
            page_n += 1
        return _to_dataframe(items)

    def competition(self, competition_id: int) -> dict:
        """Fetch a single competition's detail (settings, limits, status).

        Useful for client-side preflight against `max_upload_size_bytes` and
        `max_upload_files` before calling `upload_agent_file` / `submit`.
        Mirrors `GET /api/competitions/{id}` (`competitions.py:164`).

        Includes an `engine` sub-object with the engine's
        `k8s_workload_value` plus `vm_health_ok` / `vm_health_checked_at`
        (only populated for `local_vm` engines, polled every minute by
        simulationmanager). `vm_health_ok=False` means the GPU VM is
        currently unreachable and submissions will queue rather than run.
        """
        resp = self._request("GET",self._url(f"/competitions/{competition_id}"), timeout=30)
        self._handle_response(resp)
        resp.raise_for_status()
        return resp.json()

    def creator_competitions(self) -> list:
        """List competitions the caller owns or assists on (creator scope).

        Mirrors `GET /api/creator_competition/competitions`
        (`lifecycle.py:35`). Unlike `competitions()` this includes the caller's
        **hidden** (`is_public=False`) competitions — useful for finding a
        competition you created but did not make public. Each item carries
        `id`, `name`, `is_started`, `is_public`, `simulation_version`, `role`.

        Requires a `creator`-scope token.
        """
        resp = self._request("GET",
            self._url("/creator_competition/competitions"),
            headers=self._headers(),
            timeout=30,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise MLArenaError(f"creator_competitions failed: {_safe_error(resp)}")
        return resp.json()

    def datasets(self, competition_id: int) -> dict:
        """List the public datasets attached to a competition.

        Mirrors `GET /api/competitions/{id}/datasets` (`competitions.py:854`,
        `@auth_required('user')`). Returns
        ``{"datasets": [{"id", "label", "description",
        "files": [{"id", "label", "download_url", "file_size_bytes", ...}]}]}``
        where each ``download_url`` is a short-lived (60 min) signed URL the
        caller can GET directly. This is how a participant pulls the training /
        test data a creator published for a file competition. Requires any
        valid token scope.
        """
        resp = self._request("GET",
            self._url(f"/competitions/{competition_id}/datasets"),
            headers=self._headers(),
            timeout=30,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise MLArenaError(f"datasets failed: {_safe_error(resp)}")
        return resp.json()

    def download_dataset(self, competition_id: int, dest_dir: str = ".") -> list[str]:
        """Download every published dataset file for a competition into `dest_dir`.

        Convenience composition over `datasets()`: resolves the signed
        `download_url`s and streams each file to `<dest_dir>/<label>`. The
        signed URLs are pre-authenticated GCS links, so they are fetched
        WITHOUT the bearer token (and redirects are followed, unlike the API
        calls). Returns the list of written file paths. This is the call a
        starter notebook makes to get the competition data.
        """
        os.makedirs(dest_dir, exist_ok=True)
        written: list[str] = []
        payload = self.datasets(competition_id)
        for ds in payload.get("datasets", []):
            for f in ds.get("files", []):
                url = f.get("download_url")
                if not url:
                    continue
                # Signed GCS URL: no auth header, follow redirects.
                resp = self._request("GET", url, allow_redirects=True, timeout=300, stream=True)
                resp.raise_for_status()
                out_path = os.path.join(dest_dir, f["label"])
                with open(out_path, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        if chunk:
                            fh.write(chunk)
                written.append(out_path)
        return written

    def create_competition(self, name: str, kernel_version: str,
                           description: str | None = None,
                           copy_from_competition_id: int | None = None,
                           tag_names: list[str] | None = None,
                           is_public: bool | None = None) -> dict:
        """Create a new competition via the creator-scope authoring flow.

        `kernel_version` carries a catalog KIND: one of `"flex_v1"`,
        `"gymnasium"`, `"pettingzoo"`, `"file_v1"` (list them live via
        `GET /creator_competition/available_kinds`). Since the worker
        consolidation only two kernels exist — `gymnasium`/`pettingzoo` are
        presets over the flex_v1 kernel that pair the flexkit loop-library
        env template with the matching env-image family; the retired kernel
        names (`grpc_gymnasiumv4`, `grpc_pettingzoov2`) are accepted as
        aliases during the transition.

        The backend resolves the engine + default evaluation + env-image
        family from the kind. To pin a specific engine, use the creator UI
        or post-creation patch endpoints — the API does not let you pick an
        engine directly at creation time.

        If `tag_names` is given, each name is resolved against the public
        tag catalog (`GET /api/competition_tags/tags`) before the create
        call. Unknown names raise `MLArenaError` (fail fast — the catalog
        is admin-curated; new tags are not auto-created).

        Pass `is_public=False` to hide the competition from public listings,
        search, and direct URLs at creation time. Only the owner, creator
        assistants, and admins can view or interact with a hidden
        competition; everyone else gets 404. Visibility can be toggled later
        via `update_competition(is_public=...)`. Defaults to public.

        Requires a `creator`-scope token.
        """
        body = {"name": name, "kernel_version": kernel_version}
        if description is not None:
            body["description"] = description
        if copy_from_competition_id is not None:
            body["copy_from_competition_id"] = copy_from_competition_id
        if tag_names:
            body["tag_ids"] = self._resolve_tag_names(tag_names)
        if is_public is not None:
            body["is_public"] = is_public
        resp = self._request("POST",
            self._url("/creator_competition/competition"),
            headers=self._headers(json_body=True),
            json=body,
            timeout=60,
        )
        self._handle_response(resp)
        if resp.status_code not in (200, 201):
            raise MLArenaError(f"create_competition failed: {_safe_error(resp)}")
        return resp.json()

    def list_tags(self):
        """Return the public tag catalog. No auth required."""
        resp = self._request("GET",self._url("/competition_tags/tags"), timeout=30)
        resp.raise_for_status()
        return _to_dataframe(resp.json())

    def set_competition_tags(self, competition_id: int,
                             tag_names: list[str] | None = None,
                             tag_ids: list[int] | None = None) -> dict:
        """Replace the tag set on a competition.

        Pass exactly one of `tag_names` or `tag_ids`. Names are resolved
        against the catalog; unknown names raise `MLArenaError`. Pass an
        empty list to clear all tags.

        Requires a `creator`-scope token and ownership (or admin) of the
        target competition.
        """
        if (tag_names is None) == (tag_ids is None):
            raise MLArenaError("Provide exactly one of tag_names= or tag_ids=")
        ids = tag_ids if tag_ids is not None else self._resolve_tag_names(tag_names)
        resp = self._request("PUT",
            self._url(f"/creator_competition/competition/{competition_id}/tags"),
            headers=self._headers(json_body=True),
            json={"tag_ids": ids},
            timeout=30,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise MLArenaError(f"set_competition_tags failed: {_safe_error(resp)}")
        return resp.json()

    def _resolve_tag_names(self, tag_names: list[str]) -> list[int]:
        """Resolve a list of tag names to ids via the public catalog.

        Matching is case-insensitive: prod seeds tags as "GYMNASIUM" while
        local seeds them as "gymnasium", and callers shouldn't have to know
        which environment they hit.
        """
        catalog_resp = self._request("GET",self._url("/competition_tags/tags"), timeout=30)
        catalog_resp.raise_for_status()
        catalog = catalog_resp.json()
        by_name_ci = {t["name"].casefold(): t["id"] for t in catalog}
        unknown = [n for n in tag_names if n.casefold() not in by_name_ci]
        if unknown:
            raise MLArenaError(
                f"Unknown tag name(s): {unknown}. "
                f"Known tags: {sorted(t['name'] for t in catalog)}"
            )
        return [by_name_ci[n.casefold()] for n in tag_names]

    def update_competition(self, competition_id: int, *,
                           name: str | None = None,
                           description: str | None = None,
                           is_public: bool | None = None) -> dict:
        """Patch top-level competition fields (creator scope).

        Mirrors `PUT /api/creator_competition/competition/{id}`. Only fields
        explicitly passed are sent — everything else is left untouched.

        ``name`` and ``description`` are locked once the competition has
        started; the backend rejects updates to those. ``is_public`` is
        editable at any time, including while the competition is running, so
        creators can hide an active competition without stopping it. When
        ``is_public=False``, only the owner, creator assistants, and admins
        can view or interact with the competition.
        """
        body: dict = {}
        if name is not None:
            body["name"] = name
        if description is not None:
            body["description"] = description
        if is_public is not None:
            body["is_public"] = is_public
        if not body:
            raise MLArenaError("update_competition requires at least one field")
        resp = self._request("PUT",
            self._url(f"/creator_competition/competition/{competition_id}"),
            headers=self._headers(json_body=True),
            json=body,
            timeout=30,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise MLArenaError(f"update_competition failed: {_safe_error(resp)}")
        return resp.json()

    def update_settings(self, competition_id: int, *,
                        simulation_timeout_sec: int | None = None,
                        max_upload_size_bytes: int | None = None,
                        max_upload_files: int | None = None,
                        max_active_agents_per_participant: int | None = None,
                        evaluation_metric: str | None = None,
                        evaluation_metric2: str | None = None,
                        evaluation_is_elo_score: bool | None = None,
                        evaluation_deployment_nb_constraint_run: int | None = None,
                        evaluation_deployment_nb_initial_score_run: int | None = None,
                        evaluation_episode_budget_brackets: list | None = None,
                        ) -> dict:
        """Update competition settings + evaluation parameters.

        Mirrors `PUT /api/creator_competition/competition/{id}/settings` —
        the same call the console's Settings tab makes via `saveSettings()`.
        Only fields explicitly passed are sent; everything else is left
        untouched. The backend rejects updates after the competition has
        been started.

        Notable parameters:
            evaluation_metric: free-text label for the primary metric (e.g.
                "reward", "accuracy", "bleu"). The DB column is String(20).
            evaluation_is_elo_score: when True, the leaderboard ranks by
                ELO rather than mean metric (multi-agent kernels).
            evaluation_episode_budget_brackets: list of `[threshold,
                n_episodes]` pairs implementing tiered early-stop.
                Thresholds must be strictly ascending; n_episodes
                non-decreasing; max 100 episodes per bracket.

        Requires a `creator`-scope token and ownership (or admin) of the
        target competition.
        """
        body: dict = {}
        if simulation_timeout_sec is not None:
            body["simulation_timeout_sec"] = simulation_timeout_sec
        if max_upload_size_bytes is not None:
            body["max_upload_size_bytes"] = max_upload_size_bytes
        if max_upload_files is not None:
            body["max_upload_files"] = max_upload_files
        if max_active_agents_per_participant is not None:
            body["max_active_agents_per_participant"] = max_active_agents_per_participant
        if evaluation_metric is not None:
            body["evaluation_metric"] = evaluation_metric
        if evaluation_metric2 is not None:
            body["evaluation_metric2"] = evaluation_metric2
        if evaluation_is_elo_score is not None:
            body["evaluation_is_elo_score"] = evaluation_is_elo_score
        if evaluation_deployment_nb_constraint_run is not None:
            body["evaluation_deployment_nb_constraint_run"] = evaluation_deployment_nb_constraint_run
        if evaluation_deployment_nb_initial_score_run is not None:
            body["evaluation_deployment_nb_initial_score_run"] = evaluation_deployment_nb_initial_score_run
        if evaluation_episode_budget_brackets is not None:
            body["evaluation_episode_budget_brackets"] = evaluation_episode_budget_brackets
        if not body:
            raise MLArenaError("update_settings requires at least one field")
        resp = self._request("PUT",
            self._url(f"/creator_competition/competition/{competition_id}/settings"),
            headers=self._headers(json_body=True),
            json=body,
            timeout=30,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise MLArenaError(f"update_settings failed: {_safe_error(resp)}")
        return resp.json()

    def upload_env_file(self, competition_id: int, file_path: str) -> dict:
        """Upload a single environment file (env.py, requirements.txt, …).

        Multipart PUT to `/api/creator_competition/competition/{id}/env/files`.
        Use this for binary files; for text content driven by a template,
        prefer `update_env_file_content`.
        """
        if not os.path.isfile(file_path):
            raise MLArenaError(f"File not found: {file_path}")
        with open(file_path, "rb") as fh:
            resp = self._request("PUT",
                self._url(
                    f"/creator_competition/competition/{competition_id}/env/files"
                ),
                headers=self._headers(),
                files={"file": (os.path.basename(file_path), fh)},
                timeout=120,
            )
        self._handle_response(resp)
        if resp.status_code not in (200, 201):
            raise MLArenaError(f"upload_env_file failed: {_safe_error(resp)}")
        return resp.json()

    def sync_env_from_github(
        self,
        competition_id: int,
        github_repo_url: str,
        github_token: str | None = None,
    ) -> dict:
        """Mirror a GitHub repo's default branch into the competition env folder.

        Wipes the current env folder and replaces its contents with the repo
        (`.git` excluded). Only allowed before the competition is started, same
        gate as `upload_env_file`. `github_token` is optional: when provided it
        is persisted Fernet-encrypted on the competition configuration and
        reused on future syncs; leave it `None` to fall back to the saved
        token (if any).
        """
        body = {"github_repo_url": github_repo_url}
        if github_token is not None:
            body["github_token"] = github_token
        resp = self._request("POST",
            self._url(
                f"/creator_competition/competition/{competition_id}/env/sync-github"
            ),
            headers=self._headers(json_body=True),
            json=body,
            timeout=180,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise MLArenaError(f"sync_env_from_github failed: {_safe_error(resp)}")
        return resp.json()

    def update_env_file_content(self, competition_id: int, filename: str,
                                content: str) -> dict:
        """Write a text env file by content (e.g. a rendered env.py template).

        JSON PUT to the same endpoint as `upload_env_file`. The backend runs
        the kernel-specific AST validator on env.py and returns a
        `validation` block in the response.
        """
        resp = self._request("PUT",
            self._url(
                f"/creator_competition/competition/{competition_id}/env/files"
            ),
            headers=self._headers(json_body=True),
            json={"filename": filename, "content": content},
            timeout=60,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise MLArenaError(f"update_env_file_content failed: {_safe_error(resp)}")
        return resp.json()

    def set_competition_image(self, competition_id: int, image_path: str) -> dict:
        """Upload (or replace) the competition thumbnail (`miniature.png`).

        Multipart POST to `/api/creator_competition/competition/{id}/image`.
        """
        if not os.path.isfile(image_path):
            raise MLArenaError(f"Image file not found: {image_path}")
        with open(image_path, "rb") as fh:
            resp = self._request("POST",
                self._url(
                    f"/creator_competition/competition/{competition_id}/image"
                ),
                headers=self._headers(),
                files={"file": (os.path.basename(image_path), fh)},
                timeout=120,
            )
        self._handle_response(resp)
        if resp.status_code not in (200, 201):
            raise MLArenaError(f"set_competition_image failed: {_safe_error(resp)}")
        return resp.json()

    def set_competition_markdown(self, competition_id: int, content: str) -> dict:
        """Replace the competition overview markdown (`overview.md`)."""
        resp = self._request("PUT",
            self._url(
                f"/creator_competition/competition/{competition_id}/markdown"
            ),
            headers=self._headers(json_body=True),
            json={"content": content},
            timeout=30,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise MLArenaError(f"set_competition_markdown failed: {_safe_error(resp)}")
        return resp.json()

    def upload_benchmark_file(self, competition_id: int, file_path: str) -> dict:
        """Upload a benchmark agent file (typically `agent.py`).

        Backend reads the benchmark folder when `run_benchmark` fires; you
        must upload the agent before kicking the run.
        """
        if not os.path.isfile(file_path):
            raise MLArenaError(f"File not found: {file_path}")
        with open(file_path, "rb") as fh:
            resp = self._request("PUT",
                self._url(
                    f"/creator_competition/competition/{competition_id}/benchmark/files"
                ),
                headers=self._headers(),
                files={"file": (os.path.basename(file_path), fh)},
                timeout=120,
            )
        self._handle_response(resp)
        if resp.status_code not in (200, 201):
            raise MLArenaError(f"upload_benchmark_file failed: {_safe_error(resp)}")
        return resp.json()

    def update_benchmark_file_content(self, competition_id: int, filename: str,
                                      content: str) -> dict:
        """Write a benchmark file by content (templated baseline agent)."""
        resp = self._request("PUT",
            self._url(
                f"/creator_competition/competition/{competition_id}/benchmark/files"
            ),
            headers=self._headers(json_body=True),
            json={"filename": filename, "content": content},
            timeout=60,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise MLArenaError(
                f"update_benchmark_file_content failed: {_safe_error(resp)}"
            )
        return resp.json()

    def run_benchmark(self, competition_id: int) -> dict:
        """Kick off the benchmark simulation. Returns `{simulation_id, status}`.

        Requires that env.py has been uploaded and a benchmark `agent.py`
        (or the competition's submission file for file_v1, e.g. submission.csv /
        pitch.txt) is already present. The call itself is fire-and-forget; poll
        `benchmark_status` for completion.
        """
        resp = self._request("POST",
            self._url(
                f"/creator_competition/competition/{competition_id}/benchmark/run"
            ),
            headers=self._headers(),
            timeout=60,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise MLArenaError(f"run_benchmark failed: {_safe_error(resp)}")
        return resp.json()

    def benchmark_status(self, competition_id: int) -> dict:
        """Read the latest benchmark run status. Status is one of
        `none | running | completed | failed`.
        """
        resp = self._request("GET",
            self._url(
                f"/creator_competition/competition/{competition_id}/benchmark/status"
            ),
            headers=self._headers(),
            timeout=30,
        )
        self._handle_response(resp)
        resp.raise_for_status()
        return resp.json()

    def start_competition(self, competition_id: int) -> dict:
        """Flip a creator competition into the started state.

        Backend gates this behind: env.py uploaded, benchmark agent ACTIVE
        with a non-null `mean_reward`. Failures bubble up as MLArenaError.
        """
        resp = self._request("PUT",
            self._url(
                f"/creator_competition/competition/{competition_id}/start"
            ),
            headers=self._headers(),
            timeout=30,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise MLArenaError(f"start_competition failed: {_safe_error(resp)}")
        return resp.json()

    def stop_competition(self, competition_id: int) -> dict:
        """Flip a creator competition back into the not-started state.

        Mirrors `PUT /api/creator_competition/competition/{id}/stop`. Stopping
        only clears `is_started`; it leaves agents, results and the benchmark
        record untouched, so the competition can be re-started afterwards.
        """
        resp = self._request("PUT",
            self._url(
                f"/creator_competition/competition/{competition_id}/stop"
            ),
            headers=self._headers(),
            timeout=30,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise MLArenaError(f"stop_competition failed: {_safe_error(resp)}")
        return resp.json()

    def update_agent_template(self, competition_id: int,
                              agent_template: str) -> dict:
        """Set the default agent.py template handed to participants.

        Mirrors `PUT /api/creator_competition/competition/{id}/agent-template`.
        The backend rejects this while the competition is started, so stop it
        first (`stop_competition`) and re-start afterwards if needed.
        """
        resp = self._request("PUT",
            self._url(
                f"/creator_competition/competition/{competition_id}/agent-template"
            ),
            headers=self._headers(json_body=True),
            json={"agent_template": agent_template},
            timeout=30,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise MLArenaError(f"update_agent_template failed: {_safe_error(resp)}")
        return resp.json()

    def create_dataset(self, competition_id: int, label: str,
                       description: str | None = None) -> dict:
        """Create a public dataset bucket on a competition (creator scope).

        Mirrors `POST /api/creator_competition/competition/{id}/datasets`
        (`datasets.py:68`). Returns `{"id", "label", "description"}`; use the
        returned `id` with `upload_dataset_file`. The backend rejects this once
        the competition is started, so call it before `start_competition`.
        Files uploaded here are served to participants via `datasets()`.

        Requires a `creator`-scope token.
        """
        body: dict = {"label": label}
        if description is not None:
            body["description"] = description
        resp = self._request("POST",
            self._url(f"/creator_competition/competition/{competition_id}/datasets"),
            headers=self._headers(json_body=True),
            json=body,
            timeout=30,
        )
        self._handle_response(resp)
        if resp.status_code not in (200, 201):
            raise MLArenaError(f"create_dataset failed: {_safe_error(resp)}")
        return resp.json()

    def creator_datasets(self, competition_id: int) -> dict:
        """List a competition's datasets + files from the creator side.

        Mirrors `GET /api/creator_competition/competition/{id}/datasets`
        (`datasets.py:24`). Like `datasets()` but creator-scoped (works on your
        own hidden competitions). Returns `{"datasets": [{"id","label","files":[…]}]}`.

        Requires a `creator`-scope token.
        """
        resp = self._request("GET",
            self._url(f"/creator_competition/competition/{competition_id}/datasets"),
            headers=self._headers(),
            timeout=30,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise MLArenaError(f"creator_datasets failed: {_safe_error(resp)}")
        return resp.json()

    def upload_dataset_file(self, competition_id: int, dataset_id: int,
                            file_path: str) -> dict:
        """Upload one file into a competition dataset (creator scope).

        Multipart POST to
        `/api/creator_competition/competition/{id}/datasets/{dataset_id}/files`
        (`datasets.py:193`); the file is stored in GCS and exposed to
        participants through `datasets()` as a signed `download_url`. Rejected
        once the competition is started. Returns
        `{"id", "label", "file_size_bytes"}`.

        Requires a `creator`-scope token.
        """
        if not os.path.isfile(file_path):
            raise MLArenaError(f"File not found: {file_path}")
        with open(file_path, "rb") as fh:
            resp = self._request("POST",
                self._url(
                    f"/creator_competition/competition/{competition_id}/datasets/{dataset_id}/files"
                ),
                headers=self._headers(),
                files={"file": (os.path.basename(file_path), fh)},
                timeout=300,
            )
        self._handle_response(resp)
        if resp.status_code not in (200, 201):
            raise MLArenaError(f"upload_dataset_file failed: {_safe_error(resp)}")
        return resp.json()

    def update_dataset(self, competition_id: int, dataset_id: int, *,
                       label: str | None = None,
                       description: str | None = None) -> dict:
        """Update a dataset's label / description (creator scope).

        PUT to `/api/creator_competition/competition/{id}/datasets/{dataset_id}`
        (`datasets.py:118`). Only fields explicitly passed are sent. Rejected
        once the competition is started. Returns `{"id", "label", "description"}`.

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
                f"/creator_competition/competition/{competition_id}/datasets/{dataset_id}"
            ),
            headers=self._headers(json_body=True),
            json=body,
            timeout=30,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise MLArenaError(f"update_dataset failed: {_safe_error(resp)}")
        return resp.json()

    def delete_dataset_file(self, competition_id: int, dataset_id: int,
                            file_id: int) -> dict:
        """Delete one file from a competition dataset (creator scope).

        DELETE to
        `/api/creator_competition/competition/{id}/datasets/{dataset_id}/files/{file_id}`
        (`datasets.py:251`); removes the DB row and its GCS blob. Rejected once
        the competition is started. Use this before re-uploading a file with the
        same name — `upload_dataset_file` always adds a new row, so replacing a
        file means delete-then-upload. Get `file_id` from `creator_datasets()`.

        Requires a `creator`-scope token.
        """
        resp = self._request("DELETE",
            self._url(
                f"/creator_competition/competition/{competition_id}/datasets/{dataset_id}/files/{file_id}"
            ),
            headers=self._headers(),
            timeout=60,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise MLArenaError(f"delete_dataset_file failed: {_safe_error(resp)}")
        return resp.json()

    # ---- Direct attached agents (user scope) ----

    def create_attached_agent(self, competition_id: int, agent_name: str,
                              copy_from_agent_id: int | None = None) -> dict:
        """Create a new agent attachment for `competition_id`.

        Requires a `user`-scope token.
        """
        body: dict = {"agent_name": agent_name}
        if copy_from_agent_id is not None:
            body["copy_from_agent_id"] = copy_from_agent_id
        resp = self._request("POST",
            self._url(f"/direct_attache_agents/competition/{competition_id}"),
            headers=self._headers(json_body=True),
            json=body,
            timeout=60,
        )
        self._handle_response(resp)
        if resp.status_code not in (200, 201):
            raise SubmissionError(f"create_attached_agent failed: {_safe_error(resp)}")
        return resp.json()

    def upload_agent_file(self, competition_id: int, attache_agent_id: int,
                          file_path: str) -> dict:
        """Upload (or overwrite) a single file in an existing agent attachment."""
        if not os.path.isfile(file_path):
            raise SubmissionError(f"File not found: {file_path}")
        with open(file_path, "rb") as fh:
            resp = self._request("PUT",
                self._url(
                    f"/direct_attache_agents/competition/{competition_id}/{attache_agent_id}/file"
                ),
                headers=self._headers(),
                files={"file": (os.path.basename(file_path), fh)},
                timeout=120,
            )
        self._handle_response(resp)
        if resp.status_code not in (200, 201):
            raise SubmissionError(f"upload_agent_file failed: {_safe_error(resp)}")
        return resp.json()

    def deploy_agent(self, competition_id: int, attache_agent_id: int) -> dict:
        """Trigger deployment of an uploaded agent."""
        resp = self._request("PUT",
            self._url(
                f"/direct_attache_agents/competition/{competition_id}/{attache_agent_id}/deploy"
            ),
            headers=self._headers(),
            timeout=60,
        )
        self._handle_response(resp)
        if resp.status_code not in (200, 201, 202):
            raise SubmissionError(f"deploy_agent failed: {_safe_error(resp)}")
        return resp.json()

    def agent_deploy_status(self, competition_id: int, attache_agent_id: int) -> dict:
        """Read the deploy / run status of an attached agent."""
        resp = self._request("GET",
            self._url(
                f"/direct_attache_agents/competition/{competition_id}/{attache_agent_id}/deploy"
            ),
            headers=self._headers(),
            timeout=30,
        )
        self._handle_response(resp)
        resp.raise_for_status()
        return resp.json()

    def delete_agent(self, competition_id: int, attache_agent_id: int) -> dict:
        """Soft-delete an attached agent (`DELETE /direct_attache_agents/competition/{cid}/{aid}`)."""
        resp = self._request("DELETE",
            self._url(
                f"/direct_attache_agents/competition/{competition_id}/{attache_agent_id}"
            ),
            headers=self._headers(),
            timeout=30,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise SubmissionError(f"delete_agent failed: {_safe_error(resp)}")
        return resp.json()

    # ---- Runner / runtime selection (DockerImageAgentRuntime) ----

    def runtime_options(self, competition_id: int):
        """List runtimes (language × framework × version) compatible with this competition.

        Mirrors `GET /api/direct_attache_agents/runtime_options/{cid}`
        (`runtime.py:25`). Each entry: `{id, language, language_version,
        framework, framework_version, requirement}`.
        """
        resp = self._request("GET",
            self._url(f"/direct_attache_agents/runtime_options/{competition_id}"),
            headers=self._headers(),
            timeout=30,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise MLArenaError(f"runtime_options failed: {_safe_error(resp)}")
        return _to_dataframe(resp.json())

    def agent_runtime(self, attache_agent_id: int) -> dict:
        """Currently selected runtime for an attached agent (`runtime.py:88`)."""
        resp = self._request("GET",
            self._url(f"/direct_attache_agents/agent_runtime/{attache_agent_id}"),
            headers=self._headers(),
            timeout=30,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise MLArenaError(f"agent_runtime failed: {_safe_error(resp)}")
        return resp.json()

    def set_agent_runtime(self, attache_agent_id: int, runtime_id: int) -> dict:
        """Pin a `DockerImageAgentRuntime` onto an attached agent (`runtime.py:151`).

        The backend rejects runtimes whose `docker_image_worker_envagent_id`
        does not match the competition's engine.
        """
        resp = self._request("PUT",
            self._url(f"/direct_attache_agents/agent_runtime/{attache_agent_id}"),
            headers=self._headers(json_body=True),
            json={"docker_image_agent_runtime_id": runtime_id},
            timeout=30,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise MLArenaError(f"set_agent_runtime failed: {_safe_error(resp)}")
        return resp.json()

    def resolve_runtime(self, competition_id: int, *,
                        language: str | None = None,
                        framework: str | None = None,
                        framework_version: str | None = None) -> dict:
        """Find one runtime matching the given language/framework on this competition.

        Convenience wrapper over `runtime_options`. Returns the first match,
        or raises `MLArenaError` if none. Used internally by `submit()` when
        the caller passes `runtime={"language": ...}` instead of a numeric id.
        """
        opts = self.runtime_options(competition_id)
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

    def list_agent_files(self, competition_id: int, attache_agent_id: int) -> dict:
        """List files in an agent attachment with their content (or binary marker).

        Mirrors `GET …/competition/{cid}/{aid}/file` (`file.py:244`). Returns
        the raw `{"files": {<name>: {content, language, is_binary, size}}}`
        payload — keep the wrapper shallow so callers can drill into either
        the file map or the metadata.
        """
        resp = self._request("GET",
            self._url(
                f"/direct_attache_agents/competition/{competition_id}/{attache_agent_id}/file"
            ),
            headers=self._headers(),
            timeout=30,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise SubmissionError(f"list_agent_files failed: {_safe_error(resp)}")
        return resp.json()

    def get_agent_file_content(self, competition_id: int, attache_agent_id: int,
                               filename: str) -> str:
        """Fetch the raw text content of one file in an agent attachment (`file.py:463`)."""
        resp = self._request("GET",
            self._url(
                f"/direct_attache_agents/competition/{competition_id}/{attache_agent_id}/file/{filename}/content"
            ),
            headers=self._headers(),
            timeout=30,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise SubmissionError(f"get_agent_file_content failed: {_safe_error(resp)}")
        return resp.json()["content"]

    def update_agent_file_content(self, competition_id: int, attache_agent_id: int,
                                  filename: str, content: str) -> dict:
        """Write a text file by content (template render → upload).

        Multipart PUT to the same upload endpoint as `upload_agent_file`,
        carrying the content as the `file` part. Sibling of
        `update_env_file_content` on the creator side. The backend validates
        the resulting attachment folder and returns `{status, validation_message}`.
        """
        resp = self._request("PUT",
            self._url(
                f"/direct_attache_agents/competition/{competition_id}/{attache_agent_id}/file"
            ),
            headers=self._headers(),
            files={"file": (filename, io.BytesIO(content.encode("utf-8")))},
            timeout=120,
        )
        self._handle_response(resp)
        if resp.status_code not in (200, 201):
            raise SubmissionError(f"update_agent_file_content failed: {_safe_error(resp)}")
        return resp.json()

    def delete_agent_file(self, competition_id: int, attache_agent_id: int,
                          filename: str) -> dict:
        """Delete one file from an agent attachment.

        The backend overloads the upload PUT with a `delete_file=<name>` form
        field — same endpoint, different verb-encoding (`file.py:128`).
        """
        resp = self._request("PUT",
            self._url(
                f"/direct_attache_agents/competition/{competition_id}/{attache_agent_id}/file"
            ),
            headers=self._headers(),
            data={"delete_file": filename},
            timeout=60,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise SubmissionError(f"delete_agent_file failed: {_safe_error(resp)}")
        return resp.json()

    # ---- Status / logs ----

    def agent_status(self, competition_id: int, attache_agent_id: int) -> dict:
        """Rich agent status: `queue_info`, `run_info`, `last_status_message`.

        Mirrors `GET …/competition/{cid}/{aid}/status` (`status.py:210`).
        Use this over `agent_deploy_status` when you want to see queue
        position or per-run errors.
        """
        resp = self._request("GET",
            self._url(
                f"/direct_attache_agents/competition/{competition_id}/{attache_agent_id}/status"
            ),
            headers=self._headers(),
            timeout=30,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise SubmissionError(f"agent_status failed: {_safe_error(resp)}")
        return resp.json()

    def agent_games(self, attache_agent_id: int) -> dict:
        """Recent games for an attached agent with signed log URLs.

        Mirrors `GET /api/direct_attache_agents/agent/{aid}/games`
        (`monitor.py:25`). Each game carries a `signed_url` to the GCS render
        log (60-day retention) plus reward/outcome and per-game metrics.
        """
        resp = self._request("GET",
            self._url(f"/direct_attache_agents/agent/{attache_agent_id}/games"),
            headers=self._headers(),
            timeout=30,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise SubmissionError(f"agent_games failed: {_safe_error(resp)}")
        return resp.json()

    def recent_replays(self, competition_id: int, limit: int = 10) -> dict:
        """List recent completed replays for a competition.

        Mirrors `GET /api/competitions/{id}/recent-replays`. Each replay
        carries `simulation_id`, `created_at`, `render_delay_second`,
        `signed_url` (GCS render file) and `participants`.
        """
        resp = self._request("GET",
            self._url(f"/competitions/{competition_id}/recent-replays"),
            headers=self._headers(),
            params={"limit": limit},
            timeout=30,
        )
        self._handle_response(resp)
        if resp.status_code != 200:
            raise SubmissionError(f"recent_replays failed: {_safe_error(resp)}")
        return resp.json()

    def tail_logs(self, competition_id: int, attache_agent_id: int, *,
                  follow: bool = False, poll_sec: float = 5.0,
                  timeout_sec: float | None = None) -> Iterator[str]:
        """Yield human-readable status / log lines for an attached agent.

        Polls `agent_status` and emits one line per status transition,
        including queue position for `deploy_queue` and per-run rewards /
        errors for `deploy_run`. With `follow=False` (default), returns once
        the agent reaches a terminal state (`active`, `deploy_failed`,
        `upload_failed`, `deleted`); with `follow=True`, keeps polling until
        the caller breaks out. `timeout_sec` caps total wait time.

        Note: this does NOT stream pod stdout — that's only available via
        the per-game signed URLs from `agent_games()`. For raw logs of a
        completed run, do:

            for game in client.agent_games(aid)["games"]:
                if game["signed_url"]:
                    print(self._request("GET",game["signed_url"]).text)
        """
        terminal = {"active", "deploy_failed", "upload_failed", "deleted"}
        last_signature = None
        deadline = (time.monotonic() + timeout_sec) if timeout_sec else None

        while True:
            status = self.agent_status(competition_id, attache_agent_id)
            sig = (status.get("status"), status.get("last_status_message"))
            if sig != last_signature:
                yield f"[{status.get('status')}] {status.get('last_status_message') or ''}".rstrip()
                last_signature = sig

            if status.get("status") == "deploy_queue":
                qi = status.get("queue_info") or {}
                if qi:
                    yield (f"  queued: position={qi.get('position_in_queue')} "
                           f"waiting={qi.get('in_queue_for_second')}s")

            if status.get("status") == "deploy_run":
                ri = status.get("run_info") or {}
                for r in ri.get("results") or []:
                    yield (f"  run: status={r.get('status')} "
                           f"steps={r.get('steps_completed')} "
                           f"reward={r.get('current_reward')} "
                           f"outcome={r.get('game_outcome')}")
                    if r.get("status") == "error":
                        yield f"    error[{r.get('error_type')}]: {r.get('error_message')}"

            if status.get("status") in terminal:
                if not follow:
                    return

            if deadline and time.monotonic() > deadline:
                return

            time.sleep(poll_sec)

    # ---- One-shot submit helper (still used by the notebook flow) ----

    def submit(self, competition_id: int, agent=None, files=None,
               agent_name: str | None = None,
               runtime_id: int | None = None,
               runtime: dict | None = None) -> dict:
        """Convenience: create attachment → (pick runner) → upload files → deploy.

        Either pass `agent=<class>` (its source is uploaded as `agent.py`) or
        `files=[...]` (each path is uploaded under its basename).

        Optional runner selection (between create and deploy):
            runtime_id: numeric `DockerImageAgentRuntime.id`.
            runtime:    dict spec resolved via `resolve_runtime()`, e.g.
                        `{"language": "python", "framework": "torch"}`. The
                        first runtime matching every supplied key is used.
        Pass at most one of `runtime_id` / `runtime`. With neither, the
        backend's competition-default runtime is kept.
        """
        if (agent is None) == (files is None):
            raise SubmissionError("Provide exactly one of agent= or files=")
        if runtime_id is not None and runtime is not None:
            raise SubmissionError("Provide at most one of runtime_id= or runtime=")

        attach = self.create_attached_agent(
            competition_id, agent_name or _default_agent_name(agent, files)
        )
        attache_agent_id = attach["attache_agent_id"]

        # Pin runner before deploy so the JobPod uses the right image.
        if runtime is not None:
            runtime_id = self.resolve_runtime(competition_id, **runtime)["id"]
        if runtime_id is not None:
            self.set_agent_runtime(attache_agent_id, runtime_id)

        tmp_dir = None
        try:
            if agent is not None:
                tmp_dir = tempfile.mkdtemp()
                agent_py = os.path.join(tmp_dir, "agent.py")
                with open(agent_py, "w") as f:
                    f.write(inspect.getsource(agent))
                self.upload_agent_file(competition_id, attache_agent_id, agent_py)
            else:
                for path in files:
                    self.upload_agent_file(competition_id, attache_agent_id, path)

            deploy = self.deploy_agent(competition_id, attache_agent_id)
            self._last_agent_id = attache_agent_id
            self._last_competition = competition_id
            return {
                "attache_agent_id": attache_agent_id,
                "agent_id": attache_agent_id,
                "deploy": deploy,
            }
        finally:
            if tmp_dir:
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)

    def status(self, agent_id: int | None = None,
               competition_id: int | None = None) -> dict:
        """Return the rich status of a previously-submitted agent.

        Defaults to the last submission made through this client. Both
        agent_id and competition_id are required by the backend route, so
        callers using a non-default agent_id must also pass competition_id.
        Returns the same payload as `agent_status` (queue_info, run_info,
        last_status_message), which is more informative than
        `agent_deploy_status`.
        """
        agent_id = agent_id or self._last_agent_id
        competition_id = competition_id or self._last_competition
        if agent_id is None or competition_id is None:
            raise SubmissionError(
                "agent_id and competition_id are required (no previous submission found)"
            )
        return self.agent_status(competition_id, agent_id)

    # ---- Leaderboard (public read) ----

    def leaderboard(self, competition_id: int | None = None):
        """Get the leaderboard for a competition.

        Mirrors `GET /api/leaderboard/competition/{id}` (`leaderboard.py:28`).
        """
        competition_id = competition_id or self._last_competition
        if competition_id is None:
            raise SubmissionError("No competition specified and no previous submission found")
        resp = self._request("GET",
            self._url(f"/leaderboard/competition/{competition_id}"),
            timeout=30,
        )
        self._handle_response(resp)
        resp.raise_for_status()
        return _to_dataframe(resp.json())

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
            raise MLArenaError(f"{error_label} failed: {_safe_error(resp)}")
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
            raise MLArenaError(f"{error_label} failed: {_safe_error(resp)}")
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
        Profile page to author modules/lessons and attach competitions.

        `start_date` / `end_date` are optional ISO date strings (`YYYY-MM-DD`) —
        a name is the only required field, so a course can be created without
        term dates (a course with no end date never closes). They are kept
        positional for backwards compatibility with the original signature. The
        backend mints both the 32-hex `enrollment_link` and the short shareable
        `join_code` and returns them in the course dict — read them from there to
        share with students.

        Course-content fields (added with the course-content model):
            slug:        kebab URL slug (`^[a-z0-9-]+$`); auto-derived from the
                         name when omitted.
            description: markdown landing-page intro.
            visibility:  `private` (default) | `unlisted` | `public`. Only
                         `public` courses appear in `course_catalog()`.
            teacher_user_id: an admin may name the owning teacher; otherwise the
                         caller owns the course.
        Competitions are attached afterwards via the modules you link (see
        `create_module` / `attach_competition` / `link_module`).
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

    def enrollment_info(self, link_or_code: str) -> dict:
        """Preview a course from an enrollment link or join code (no enroll).

        Mirrors `GET /api/academic_courses/enroll/<link_or_code>` — the same
        public lookup the console's EnrollPage makes. Accepts either the 32-hex
        `enrollment_link` or the short `join_code`. Returns course meta plus
        `already_enrolled` / `has_student_info` when authenticated.
        """
        return self._course_call(
            "GET", f"/academic_courses/enroll/{link_or_code}",
            error_label="enrollment_info",
        )

    def enroll_in_course(self, link_or_code: str | None = None,
                         student_email: str | None = None,
                         student_number: str | None = None,
                         project_url: str | None = None,
                         *,
                         enrollment_link: str | None = None,
                         join_code: str | None = None) -> dict:
        """Enroll the authenticated user in a course.

        Mirrors `POST /api/academic_courses/enroll/<link_or_code>`. The token to
        join with can be passed positionally (`link_or_code`) or, equivalently,
        as `enrollment_link=` (the 32-hex link) or `join_code=` (the short
        shareable code) — the backend resolves either form against the course.

        `student_email` / `student_number` fill the caller's global student
        identity when it isn't set yet (they never overwrite values already on
        the profile — set those via the Profile page / profile update). The
        backend requires both to be present before enrolling. `project_url` is
        stored on the enrollment. Returns `{message, competition_id,
        competition_ids}`.
        """
        token = link_or_code or enrollment_link or join_code
        if token is None:
            raise MLArenaError(
                "enroll_in_course requires an enrollment link or join code"
            )
        body: dict = {}
        if student_email is not None:
            body["student_email"] = student_email
        if student_number is not None:
            body["student_number"] = student_number
        if project_url is not None:
            body["project_url"] = project_url
        return self._course_call(
            "POST", f"/academic_courses/enroll/{token}", json_body=body,
            ok=(200, 201), error_label="enroll_in_course",
        )

    def list_courses(self, *, show_all: bool = False,
                     competition_id: int | None = None):
        """List courses (enrolled + currently-active by default).

        Mirrors `GET /api/academic_courses/`. With `show_all=True` returns every
        course; with `competition_id` returns courses attached to that
        competition. Each row carries `is_enrolled` for the caller. Returns a
        DataFrame if pandas is installed, else a list of dicts.
        """
        params: dict = {}
        if show_all:
            params["show_all"] = "true"
        if competition_id is not None:
            params["competition_id"] = competition_id
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
        appear. `search` matches name + description.
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
        """
        return self._course_call(
            "GET", f"/academic_courses/{slug}", error_label="course",
        )

    def module_overview(self, slug: str, module_slug: str) -> dict:
        """A course module's overview: lesson list + attached competition cards.

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

    def my_progress(self, course_id: int) -> dict:
        """Own progress across a course: content % + next lesson + competition
        results.

        Mirrors `GET /api/academic_courses/{course_id}/progress/me`. Requires
        enrollment (or manage rights).
        """
        return self._course_call(
            "GET", f"/academic_courses/{course_id}/progress/me",
            error_label="my_progress",
        )

    # ---- Course authoring: modules (teacher scope) --------------------------

    def create_module(self, title: str, slug: str | None = None,
                      summary: str | None = None, icon: str | None = None,
                      visibility: str = "private") -> dict:
        """Create an owned, reusable content module. Requires a `teacher` token.

        Mirrors `POST /api/teacher/modules`. `slug` (kebab, `^[a-z0-9-]+$`) is
        auto-derived from the title when omitted and is unique per owner.
        `visibility` is `private` (default) | `unlisted` | `public`; only
        `public` modules are forkable/linkable by other teachers.
        """
        body: dict = {"title": title, "visibility": visibility}
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
        """Module detail + lesson TOC + competition links (owner or public).

        Mirrors `GET /api/teacher/modules/{id}`.
        """
        return self._course_call(
            "GET", f"/teacher/modules/{module_id}", error_label="get_module",
        )

    def update_module(self, module_id: int, **fields) -> dict:
        """Update module meta — owner only. Mirrors `PUT /api/teacher/modules/{id}`.

        Accepts `title`, `summary`, `icon`, `visibility` (the slug is immutable
        for URL stability). Only the fields you pass are changed.
        """
        allowed = {"title", "summary", "icon", "visibility"}
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
        preserved) + competition attachments; the fork starts `private` and
        records provenance via `forked_from_module_id`.
        """
        return self._course_call(
            "POST", f"/teacher/modules/{module_id}/fork",
            ok=(200, 201), error_label="fork_module",
        )

    def attach_competition(self, module_id: int, competition_id: int,
                           label: str | None = None,
                           position: int | None = None) -> dict:
        """Attach a competition to a module.

        Mirrors `POST /api/teacher/modules/{id}/competitions`. `position`
        defaults to the end of the module's competition list.
        """
        body: dict = {"competition_id": competition_id}
        if label is not None:
            body["label"] = label
        if position is not None:
            body["position"] = position
        return self._course_call(
            "POST", f"/teacher/modules/{module_id}/competitions", json_body=body,
            ok=(200, 201), error_label="attach_competition",
        )

    def detach_competition(self, module_id: int, competition_id: int) -> dict:
        """Detach a competition from a module.

        Mirrors `DELETE /api/teacher/modules/{id}/competitions/{competition_id}`.
        """
        return self._course_call(
            "DELETE",
            f"/teacher/modules/{module_id}/competitions/{competition_id}",
            error_label="detach_competition",
        )

    def reorder_module_competitions(self, module_id: int,
                                    ordered_competition_ids: list[int]) -> list:
        """Reorder a module's attached competitions.

        Mirrors `PUT /api/teacher/modules/{id}/competitions/reorder`.
        `ordered_competition_ids` must be exactly the module's attached
        competition ids in the desired order.
        """
        return self._course_call(
            "PUT", f"/teacher/modules/{module_id}/competitions/reorder",
            json_body={"ordered_ids": ordered_competition_ids},
            error_label="reorder_module_competitions",
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
        """Upload an image/attachment for a lesson; returns `{filename, url}`.

        Mirrors `POST /api/teacher/lessons/{id}/media`. Embed the returned `url`
        in the lesson's markdown body to display the asset.
        """
        return self._upload_course_file(
            "POST", f"/teacher/lessons/{lesson_id}/media", file_path,
            error_label="upload_lesson_media",
        )

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

        Mirrors `POST /api/teacher/course/{id}/cover`. Returns
        `{cover_image_path, url}`.
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
        """Link a module into a course as a live reference (D2).

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
        """Teacher follow dashboard: per enrolled student × (content %, per-competition
        best result).

        Mirrors `GET /api/teacher/course/{id}/progress`. Requires teacher/TA/admin.
        """
        return self._course_call(
            "GET", f"/teacher/course/{course_id}/progress",
            error_label="course_progress",
        )

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
                visibility: public   # optional (default private)
                module_id: 7         # optional — link this existing module instead of creating
                competitions:        # optional
                  - competition_id: 42
                    label: "CartPole"
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
        `{course_id, slug, enrollment_link, join_code, modules: [...]}`.
        """
        base = os.path.abspath(path)
        manifest = _load_course_manifest(base)
        course_spec = dict(manifest.get("course") or {})
        modules_spec = manifest.get("modules") or []

        existing_course_id = course_spec.pop("course_id", None)
        cover = course_spec.pop("cover", None)

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
                )
                module_id = created["id"]
                lesson_ids = []
                for lesson_spec in (m.get("lessons") or []):
                    lesson_ids.append(
                        self._author_lesson(base, module_id, lesson_spec)["id"]
                    )
                for comp in (m.get("competitions") or []):
                    self.attach_competition(
                        module_id, comp["competition_id"],
                        label=comp.get("label"), position=comp.get("position"),
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
            "enrollment_link": course_out.get("enrollment_link"),
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
        are enrolled in; owners also get draft lessons). Returns the manifest dict.
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
                "competitions": [
                    {"competition_id": c["competition_id"], "label": c.get("label")}
                    for c in module.get("competitions", [])
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


def _default_agent_name(agent, files) -> str:
    if agent is not None:
        return getattr(agent, "__name__", "agent")
    if files:
        return os.path.splitext(os.path.basename(files[0]))[0]
    return "agent"


def _safe_error(resp: requests.Response, fallback: str = "request failed") -> str:
    try:
        return resp.json().get("error", fallback)
    except Exception:
        return resp.text or fallback


def _to_dataframe(data):
    """Return a pandas DataFrame if pandas is installed, otherwise a list/dict."""
    if isinstance(data, dict) and "columns" in data and "data" in data:
        rows = [dict(zip(data["columns"], row)) for row in data["data"]]
    elif isinstance(data, list):
        rows = data
    else:
        return data
    try:
        import pandas as pd
        return pd.DataFrame(rows)
    except ImportError:
        return rows
