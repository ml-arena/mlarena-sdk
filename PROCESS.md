# mlarena-sdk — Python SDK

**Purpose**: Python client (`mlarena-sdk` 3.0.0 on PyPI, import `mlarena`) over the public backend REST routes (`/api/*`).

**Key features**:
- Bearer-token auth with scope-segmented keys `mlk_<scope>_<lookup>_<secret>` (`user`, `creator`, `teacher`).
- One method per backend route, plus client-side compositions (`submit`, `status`, `tail_logs`, `resolve_runtime`, `download_dataset`, `chat`, `author_course_from_dir`, `export_course_to_dir`) that add no endpoint.
- Reads the backend's names as served (snake_case `LeaderboardRow`, `RunResult`, the submission status block); keeps no status set or key map of its own.
- Deprecated Python names from before the challenge/submission rename still resolve (published client).

**Links**: [`../ARCHITECTURE.md`](../ARCHITECTURE.md) · [`../backend/PROCESS.md`](../backend/PROCESS.md) (route groups, schemas) · [`../frontend/PROCESS.md`](../frontend/PROCESS.md) (peer consumer) · [`../CLAUDE.md`](../CLAUDE.md) (parity rule) · [`../docs/drift_alignment.md`](../docs/drift_alignment.md) §5 (parity gaps)

## Parity rule

The SDK and the React console are peers over one REST surface; every user-facing workflow is reachable from both.
- No `/api/sdk/*`, no SDK-only or frontend-only route. A route change updates `mlarena/client.py` and `frontend/src/services/*` (+ hook) in the same change.
- Auth is the only divergence: session cookie (console) vs bearer token (SDK). Route, payload and response never branch on the caller.
- SDK ergonomics are compositions of public routes, never new endpoints.

## Module map

| File | Role |
|---|---|
| `mlarena/__init__.py` | `connect(api_key, base_url="https://ml-arena.com", allow_insecure=False)` → `MLArenaClient`; refuses `http://` to a non-loopback host unless `allow_insecure`. `__version__`. |
| `mlarena/client.py` | `MLArenaClient`: every REST method; `_handle_response` / `_failed` error mapping; `_RENAMED_METHODS`, `_LEGACY_KWARGS`, `_deprecated_alias` at the bottom. |
| `mlarena/chat.py` | `ChatConversation` (`client.chat(cid)`): `say`, `reset`, `transcript`, `total_amount_eur` — a loop over the `/api/chat/*` methods. |
| `mlarena/exceptions.py` | `MLArenaError` (`status_code`, `body`) → `AuthenticationError` (401) → `PermissionDeniedError` (403); `NotFoundError` → `ChallengeNotFoundError`, `SubmissionNotFoundError`, `ChatSessionNotFoundError`; `MaintenanceError` (503 with `maintenance_mode: true`; any other 503 stays `MLArenaError`); `SubmissionError`. |
| `tests/` | Offline unit tests (`test_submissions_and_compat.py`, `test_chat.py`, `test_course_content.py`, `test_creator.py`, `test_teacher_admin.py`, `test_teams.py`). |

## Methods ↔ routes

| Methods | Backend routes | Frontend service |
|---|---|---|
| `me` | `/api/auth/current_user` | `context/AuthContext.tsx` |
| `profile`, `update_profile` | `/api/profile/`, `/api/profile/update` | `apiService.ts` |
| `challenge_team`, `create_team`, `update_team`, `delete_team`, `leave_team`, `remove_team_member`, `search_teams`, `invite_to_team`, `pending_invitations`, `received_invitations`, `respond_to_invitation`, `cancel_invitation` | `/api/teams/*` | `teamsApi.ts` |
| `challenges`, `challenge`, `datasets`, `recent_replays`, `update_challenge_configuration` (admin) | `/api/challenges/*` | `apiService.ts` |
| `list_tags` | `/api/challenge_tags/tags` | `apiService.ts` |
| `creator_challenges`, `creator_challenge`, `available_kinds`, `copyable_challenges`, `create_challenge`, `update_challenge`, `update_settings`, `challenge_tags`, `set_challenge_tags`, `list_env_files`, `upload_env_file`, `update_env_file_content`, `delete_env_file`, `check_env`, `sync_env_from_github`, `challenge_image`, `set_challenge_image`, `delete_challenge_image`, `challenge_markdown`, `set_challenge_markdown`, `list_benchmark_files`, `upload_benchmark_file`, `update_benchmark_file_content`, `delete_benchmark_file`, `run_benchmark`, `benchmark_status`, `start_challenge`, `stop_challenge`, `update_agent_template`, `csv_ground_truth`, `create_dataset`, `creator_datasets`, `upload_dataset_file`, `update_dataset`, `delete_dataset`, `delete_dataset_file`, `creator_runs`, `creator_submissions`, `clean_redeploy_submission`, `clean_redeploy_all`, `soft_delete_submission`, `challenge_assistants`, `add_challenge_assistant`, `remove_challenge_assistant` | `/api/creator_challenge/*` | `creatorChallengeApi.ts` |
| `create_submission`, `copyable_submissions`, `upload_submission_file`, `update_submission_file_content`, `list_submission_files`, `get_submission_file_content`, `download_submission_file`, `delete_submission_file`, `upload_submission_docs`, `delete_submission_docs`, `deploy_submission`, `submission_deploy_status`, `submission_status`, `submission_games`, `set_submission_visibility`, `runtime_options`, `agent_runtime`, `set_agent_runtime`, `delete_submission`, `my_submissions` | `/api/submissions/*` | `submissionsApi.ts` |
| `submission_overview` | `/api/submission_result/{cid}/{sid}/overview` | `submissionsApi.ts` |
| `leaderboard` | `/api/leaderboard/challenge/{id}` | `leaderboardApi.ts` |
| `global_ranking`, `user_global_rank` | `/api/ranking/`, `/api/ranking/user/{id}` | `leaderboardApi.ts` |
| `data_source_weather_{coverage,series,snapshot,cities}`, `data_source_news_{volume,sources}` | `/api/data_sources/*` | `dataSourcesApi.ts` |
| `chat_challenge`, `open_chat_session`, `chat_session`, `send_chat_message`, `close_chat_session`, `export_chat_session` (user); `chat_admin`, `update_chat_settings`, `chat_sessions`, `void_chat_session`, `unvoid_chat_session`, `export_chat_evidence` (creator) | `/api/chat/*` | `chatApi.ts` |
| `create_course`, `list_courses`, `enrollment_info`, `enroll_in_course` | `/api/academic_courses/`, `/api/academic_courses/enroll/{join_code}` | `coursesApi.ts` |
| `course_catalog`, `course`, `module_overview`, `lesson`, `mark_lesson_viewed`, `mark_lesson_complete`, `mark_lesson_incomplete`, `my_progress`, `download_lesson_media` | `/api/academic_courses/*` (consumption, assets) | `coursesApi.ts` |
| `create_module`, `list_modules`, `get_module`, `update_module`, `delete_module`, `fork_module`, `attach_challenge`, `update_challenge_link`, `detach_challenge`, `reorder_module_challenges` | `/api/teacher/modules/*` | `coursesApi.ts` |
| `create_lesson`, `get_lesson`, `update_lesson`, `delete_lesson`, `reorder_lessons`, `list_lesson_media`, `upload_lesson_media`, `delete_lesson_media`, `preview_lesson` | `/api/teacher/lessons/*`, `/api/teacher/modules/{id}/lessons*` | `coursesApi.ts` |
| `update_course`, `set_course_cover`, `list_course_modules`, `link_module`, `unlink_module`, `reorder_modules`, `add_course_challenge`, `remove_course_challenge`, `course_progress` | `/api/teacher/course/{id}/*` | `coursesApi.ts` |
| `teacher_courses`, `challenges_for_course`, `course_students`, `remove_student`, `export_course_csv` | `/api/teacher/courses`, `/api/teacher/challenges-for-course`, `/api/teacher/students/{cid}`, `/api/teacher/student/{cid}/{uid}`, `/api/teacher/export-csv/{cid}` | `coursesApi.ts` |
| `course_assistants`, `add_course_assistant`, `remove_course_assistant`, `course_cover` | `/api/teacher/course/{id}/assistants*`, `/api/academic_courses/assets/courses/{id}/cover` | `coursesApi.ts` |
| `submit`, `status`, `tail_logs`, `resolve_runtime`, `download_dataset`, `chat`, `author_course_from_dir`, `export_course_to_dir` | compositions — no endpoint | — |

When this table drifts from `client.py`, fix the code, not the table.

### Parity gaps (frontend-only routes, no SDK method)

- The public overview markdown / miniature reads (`GET /api/challenge_asset/*`) have no participant method; creators read them through `challenge_markdown` / `challenge_image`. Every other participant, creator and teacher route has its method (table above).
- By design console-only: admin cluster-ops (engines, maintenance, data-quality, admin runs, tag CRUD), and **API-key management**. `GET /api/auth/api_keys` and `POST /api/auth/api_keys/{scope}/rotate` are `session_auth_required`: they answer 403 to a bearer caller, so that a leaked `mlk_user_…` key cannot mint a `mlk_creator_…` one for the same account. An SDK method would only ever raise 403, so there is none — `me()` covers "who am I", and keys are rotated from the console's Profile page. This is the auth axis of the parity rule, not a gap.

## Contracts

- **Data feed** — `update_settings` carries the admin-only `data_source_*` + `batch_cron` keywords (sentinel `_UNSET` = not sent, `None` = null; `data_source_filter` is `dict[str, str]`); the backend answers 403 to a non-admin. `data_source_weather_series(city_name, country_code)` sends the response's own keys.
- **Payload keys are the backend's exactly** (request schemas are `extra="forbid"`): e.g. `submission_name`, `copy_from_submission_id`, `copy_from_challenge_id`, `max_active_submissions_per_participant`, `challenge_id`; chat `charter_accepted`, `content`, `reason`, `llm_base_url`, `llm_model`, `llm_api_key`, `turn_timeout_sec`, `max_turns_per_session`, `max_sessions_per_participant`.
- **Submission status block** — every reply carrying a submission's `status` carries `SubmissionStatusFields` (`backend/app/views/submissions/_schemas.py`): `status`, `phase`, `last_status_message`, `status_update_ts`, `is_uploadable`, `is_deployable`, `is_settled`. `status` ∈ `SubmissionStatus` (`modelmanager/modelmanager/submissions.py`): `created`, `upload_failed`, `upload_validated`, `deploy_queue`, `deploy_run`, `deploy_failed`, `active`, `deleted`. The SDK gates on the served flags: `tail_logs` stops on `is_settled`; `submit` deploys only when `is_deployable` and returns `{"submission_id", "deploy"}`; `submit(wait=True)` returns an `active` submission or raises `SubmissionError`. `/submissions/mine` serves no `deleted` rows.
- **Run model** — `submission_status()["run_info"]["results"]` and `submission_games()["games"]` rows are the backend's `RunResult` (`backend/app/views/run_serializers.py`): run columns (`job_status`, `env_nb_steps`, `env_error_type`, `env_error_message`, env metrics) + `submission_results` (one `RunSubmissionResult` per agent: `submission_id`, `submission_reward`, `agent_nb_steps`, `game_outcome`, `agent_error_type`, `agent_error_message`, `agent_stdout_logs`). The caller finds its own row by `submission_id`; `env_*` diagnostics are null on participant routes. Games rows add `signed_url`, `render_delay_second`. `run_benchmark` / `benchmark_status` return the creator's latest benchmark run in the same shape (with the env diagnostics), `benchmark_status` `None` before the first run; the backend scores the benchmark submission when the run completes.
- **Kinds** — `available_kinds()` rows carry `kernel_version` (the name every payload uses, `creator_challenge()["configuration"]` and `creator_challenges()` rows included) and closed `capabilities`: `agent_template`, `benchmark`, `dataset`, `env_structural_check`, `runs`, `chat`.
- **Errors** — every refusal goes through `_handle_response` (401/403/404) or `_failed` (other codes) and carries `status_code` + `body`; no API call ends on `requests.raise_for_status()` (the presigned-R2 fetch in `download_dataset` is the one exception, pinned by `test_no_api_call_ends_on_raise_for_status`). A deploy 409 body carries `deployment_limits` and `active_submission_limits`.
- **Leaderboard** — `leaderboard(challenge_id, top=None, *, aggregate=None, course_id=None, me=False, q=None, window=None)` sends only the passed keys (`limit`, `aggregate`, `course_id`, `me`, `q`, `window`). The backend serves one `LeaderboardEnvelope` (`backend/app/views/_schemas.py`); the method returns its `leaders` as a DataFrame of `LeaderboardRow` with every other key (`challenge`, `total`, `me`, `matches`, `course_context`) on `df.attrs`, or the envelope dict as served without pandas.
- **Chat challenges** (`chat_v1`) — no upload/deploy; the team's submission is created by the first `open_chat_session` (born `active`; `create_submission` answers 409). `send_chat_message(wait=True)` sends (202 `{"turn", "message"}`) and polls `chat_session` every 0.7 s until that turn is `completed` or `failed`. Money is served as two-decimal strings; only `ChatConversation.total_amount_eur` parses a `Decimal`. The group's cumulative euros and per-rule scoreboard are `participant.total_amount_eur` / `participant.scoreboard` on both the challenge view and the session view. `update_chat_settings` sends only the keywords passed (`None` clears a limit, `llm_api_key=""` clears the key; the key is never read back, only `llm_api_key_set`).
- **Course manifest** — `author_course_from_dir` / `export_course_to_dir` compose the authoring/consumption methods. `course.yaml` is written with `challenges:` / `challenge_id:`; the old `competitions:` / `competition_id:` keys are read forever (permanent input aliases — CLAUDE.md, "Vocabulary"). A module block's `is_published` defaults to true. The consumption payload names the attached challenge `challenge: {id, name}`, so the export reads `c["challenge"]["id"]` to write the manifest's `challenge_id`.
- **Course payload names** — a course object carries `has_cover` (never the stored `cover_image_path`: the catalog and the landing are public reads and an NFS path would publish the share's layout) and the client builds the image URL from the course id (`course_cover` / `courseCoverUrl`); a module object is keyed `id` and a link object carries `module_id`/`challenge`; an enrollment row is keyed `user_id` and dated `enrolled_at_ts`; the enroll-info payload nests the course under `course`. A course's module row carries `layout` (`simple` | `advanced`) — the platform's own rule, not a client guess.

## Deprecated aliases (frozen — CLAUDE.md, "Vocabulary")

- `_RENAMED_METHODS`: old method names (`competitions`, `create_competition`, `deploy_agent`, `agent_status`, `agent_games`, …) warn (`DeprecationWarning`) and call the new method. The module raises at import if a target method is missing.
- `_LEGACY_KWARGS`: old keyword names (`competition_id`, `agent_id`, `agent_name`, …) are accepted on every public method; passing both spellings raises `TypeError`.
- `CompetitionNotFoundError` is `ChallengeNotFoundError`. `PermissionDeniedError` subclasses `AuthenticationError`, so `except AuthenticationError` still catches a 403.

## Auth scopes

Enforced by `auth_required` / `user_satisfies_scope` in `backend/app/auth/decorators.py`; the scope set is `API_KEY_SCOPES` in `modelmanager/modelmanager/api_keys.py`.
- `user` — own submissions, chat as participant, enroll, course reading and own lesson progress.
- `creator` — `/api/creator_challenge/*` and the chat creator routes, on challenges the user owns or assists.
- `teacher` — `/api/teacher/*` and `create_course`.
- Admins may mint every scope. `@login_required` routes accept any bearer token (Flask-Login `request_loader` → `load_user_from_request`). Public reads still send the token, because `challenges`, `challenge` and `leaderboard` answer differently to an identified caller.

## Commands

```bash
pip install -e mlarena-sdk                                  # Python >= 3.10; pandas optional ([pandas] extra)
backend/.venv/bin/python -m pytest mlarena-sdk/tests -q     # offline unit tests (pytest, requests, pandas)
python -c "import mlarena; c = mlarena.connect('mlk_…', base_url='http://localhost:4999'); print(c.challenges())"
```

The local backend is `http://localhost:4999` (`kubectl port-forward svc/backend 4999:4999`); seeded deterministic keys are in `tests-dummy/.dummy-keys.json` ([`../tests-dummy/README.md`](../tests-dummy/README.md)). `mlarena-sdk/` is its own repository (gitignored here); its `README.md` is the user-facing reference.
