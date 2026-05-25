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

    def create_competition(self, name: str, kernel_version: str,
                           description: str | None = None,
                           copy_from_competition_id: int | None = None,
                           tag_names: list[str] | None = None,
                           is_public: bool | None = None) -> dict:
        """Create a new competition via the creator-scope authoring flow.

        The backend resolves the engine + default evaluation + default env
        runtime from `kernel_version`. To pin a specific engine, use the
        creator UI or post-creation patch endpoints — the API does not let
        you pick an engine directly at creation time.

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
        (or `submission.csv` for csv_v2) is already present. The call itself
        is fire-and-forget; poll `benchmark_status` for completion.
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

    # ---- Academic courses (teacher scope to create, user scope to enroll) ----

    def create_course(self, name: str, code: str, start_date: str, end_date: str,
                      instructor_name: str | None = None,
                      competition_id: int | None = None) -> dict:
        """Create an academic course. Requires a `teacher`-scope token.

        `start_date` / `end_date` are ISO date strings (`YYYY-MM-DD`). The
        backend mints the `enrollment_link` itself and returns it in the
        response — read it from there to share with students.
        """
        body = {
            "name": name,
            "code": code,
            "start_date": start_date,
            "end_date": end_date,
        }
        if instructor_name is not None:
            body["instructor_name"] = instructor_name
        if competition_id is not None:
            body["competition_id"] = competition_id
        resp = self._request("POST",
            self._url("/academic_courses/"),
            headers=self._headers(json_body=True),
            json=body,
            timeout=30,
        )
        self._handle_response(resp)
        if resp.status_code not in (200, 201):
            raise MLArenaError(f"create_course failed: {_safe_error(resp)}")
        return resp.json()

    def enroll_in_course(self, enrollment_link: str,
                         student_email: str | None = None,
                         student_number: str | None = None,
                         project_url: str | None = None) -> dict:
        """Enroll the authenticated user in a course via its enrollment link."""
        body: dict = {}
        if student_email is not None:
            body["student_email"] = student_email
        if student_number is not None:
            body["student_number"] = student_number
        if project_url is not None:
            body["project_url"] = project_url
        resp = self._request("POST",
            self._url(f"/academic_courses/enroll/{enrollment_link}"),
            headers=self._headers(json_body=True),
            json=body,
            timeout=30,
        )
        self._handle_response(resp)
        if resp.status_code not in (200, 201):
            raise MLArenaError(f"enroll_in_course failed: {_safe_error(resp)}")
        return resp.json()

    def __repr__(self) -> str:
        head = self._token.split("_", 3)
        masked = "_".join(head[:3] + ["…"]) if len(head) == 4 else "…"
        return f"MLArenaClient(base_url='{self._base_url}', token='{masked}')"


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
