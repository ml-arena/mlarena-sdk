# mlarena-sdk

**Purpose:** Python SDK that wraps the public ML Arena REST API (`/api/...`). Lets users make submissions, author challenges, manage academic courses, and read leaderboards from a notebook or script.

**Key Features:**
- Bearer-token auth with scope-segmented keys (`mlk_user_…`, `mlk_creator_…`, `mlk_teacher_…`).
- One-shot helpers (`submit`, `status`, `leaderboard`) layered on top of the granular REST methods.
- 2.0.0 speaks only the renamed wire (`/api/challenges`, `/api/submissions`, `challenge_*` / `submission_*` keys). Old **Python** names still resolve through `_RENAMED_METHODS` / `_LEGACY_KWARGS` / `_deprecated_alias` at the bottom of `client.py`.
- No `/api/sdk/*` namespace — the SDK calls the same canonical blueprints the React frontend calls.

**Links:**
- API contract: [`../backend/PROCESS.md`](../backend/PROCESS.md) (route groups)
- Frontend consumer of the same routes: [`../frontend/PROCESS.md`](../frontend/PROCESS.md)
- [`../ARCHITECTURE.md`](../ARCHITECTURE.md) — end-to-end request trace

## Frontend ↔ SDK parity rule (load-bearing)

**The SDK and the React console (frontend) are peers over the same REST surface. Every user-facing workflow must be reachable from both.**

- No SDK-only endpoint. No frontend-only endpoint. No `/api/sdk/*` blueprint.
- When you add a backend route for the frontend, the SDK gets a method with matching semantics in the same change. When you add a method here, the frontend hook is expected to use the same route.
- When you change a route's contract (path, payload, status codes, auth scope), update both consumers in lockstep — the SDK's `client.py` and the corresponding `frontend/src/services/*` + `frontend/src/hooks/**`.
- Auth is the *only* legitimate axis of divergence: the frontend uses session cookies, the SDK uses bearer tokens. The route, payload, and response shape must not branch on the caller.
- If a workflow genuinely needs SDK-only ergonomics (e.g. `submit()` bundling create + upload + deploy), build it as a *client-side composition of the public routes* — not a new backend endpoint.

Reviewers should reject PRs that introduce SDK-exclusive or frontend-exclusive routes.

## Module map

| File | Purpose |
|---|---|
| `mlarena/__init__.py` | `connect(api_key, base_url=…)` factory returning `MLArenaClient` |
| `mlarena/client.py` | All REST methods; mirrors backend route groups |
| `mlarena/chat.py` | `ChatConversation` — `client.chat(cid)`, the notebook face of a chat challenge (a composition of the `/api/chat/*` methods, no endpoint) |
| `mlarena/exceptions.py` | `MLArenaError`, `AuthenticationError` (+ `PermissionDeniedError`), `NotFoundError` with `ChallengeNotFoundError` (old name `CompetitionNotFoundError` is the same class), `SubmissionNotFoundError`, `ChatSessionNotFoundError`, `SubmissionError` |

## Method ↔ route groups

Methods on `MLArenaClient` map 1:1 to backend blueprints — same shape the frontend uses.

| SDK method group | Backend blueprint | Frontend equivalent |
|---|---|---|
| `challenges()`, `challenge()`, `datasets()`, `download_dataset()`, `recent_replays()`, `list_tags()` | `/api/challenges`, `/api/challenge_tags` | `services/apiService.ts`, `pages/Challenge/View.js` (datasets), `components/Challenge/RecentReplays.js`, `hooks/creatorChallenge/useTags.ts` |
| `update_challenge_configuration` (admin account: engine pin, runtime, step limits) | `PUT /api/challenges/{id}/configuration` | `hooks/creatorChallenge/useAdmin.ts` |
| `create_challenge`, `update_challenge`, `update_settings`, `set_challenge_tags`, `upload_env_file`, `update_env_file_content`, `set_challenge_image`, `set_challenge_markdown`, `upload_benchmark_file`, `update_benchmark_file_content`, `run_benchmark`, `benchmark_status`, `create_dataset`, `upload_dataset_file`, `start_challenge`, `update_agent_template` | `/api/creator_challenge/*` (`…/challenge/{id}/agent-template` keeps its segment) | `services/creatorChallengeApi.ts` + `hooks/creatorChallenge/*` |
| `create_submission`, `copyable_submissions`, `upload_submission_file`, `update_submission_file_content`, `list_submission_files`, `get_submission_file_content`, `download_submission_file`, `delete_submission_file`, `upload_submission_docs`, `delete_submission_docs`, `deploy_submission`, `submission_deploy_status`, `submission_status`, `submission_games`, `tail_logs`, `runtime_options`, `agent_runtime`, `set_agent_runtime`, `resolve_runtime`, `delete_submission`, `my_submissions`, `set_submission_visibility`, `submission_overview`, `submit`, `status` | `/api/submissions/*` (`mine`, `challenge/{cid}/{sid}/…`, `challenge/{cid}/{sid}/file/{name}` download, `challenge/{cid}/{sid}/docs`, `copyable_submissions`, `submission/{sid}/games`, `submission/{sid}/visibility`, `runtime_options/{cid}`, `agent_runtime/{sid}`), `/api/submission_result/{cid}/{sid}/overview` | `hooks/Submission/*` (`useSubmission`, `useSubmissionDeploy`, `useSubmissionFileManagement`, `useRuntimeOptions`), `components/Submission/Monitor/DocumentationManager.tsx`, `components/Submission/NameYourSubmissionModal.tsx` |
| `leaderboard` | `/api/leaderboard/challenge/{id}` | `services/leaderboardApi.ts`, `hooks/challenge/useLeaderboardData` |
| `data_source_weather_coverage`, `data_source_weather_series`, `data_source_weather_snapshot`, `data_source_weather_cities`, `data_source_news_volume`, `data_source_news_sources` (public, no scope) | `/api/data_sources/weather/{coverage,series,snapshot,cities}`, `/api/data_sources/news/{volume,sources}` — read proxy over the data-collection sidecar; a challenge's slice is `challenge(id)["data_source_asset"]`; sidecar down ⇒ 503 `{"error","upstream":"data_collection"}` raised as `MLArenaError`, unknown city ⇒ `NotFoundError` | `services/dataSourcesApi.ts` |
| `chat_challenge`, `open_chat_session`, `chat_session`, `send_chat_message`, `close_chat_session`, `export_chat_session` (user scope); `chat_admin`, `update_chat_settings`, `chat_sessions`, `void_chat_session`, `unvoid_chat_session`, `export_chat_evidence` (creator scope) | `/api/chat/*` — participant: `challenge/{cid}`, `challenge/{cid}/sessions`, `sessions/{sid}`, `sessions/{sid}/messages`, `sessions/{sid}/close`, `sessions/{sid}/export?format=`; creator: `challenge/{cid}/admin`, `challenge/{cid}/settings`, `challenge/{cid}/sessions?status=`, `sessions/{sid}/void`, `sessions/{sid}/unvoid`, `challenge/{cid}/export` (plan `docs/plan_chat_kernel.md` §5) | `services/chatApi.ts`, `pages/Chat/ChatPage.tsx` (`useChatSession`), creator editor `sections/ChatSection.tsx` + `sections/ChatSessionsSection.tsx` |
| `chat(cid)` → `ChatConversation.say` / `reset` / `transcript` / `total_eur` | *(client-side composition — no endpoint)* | *(SDK-only ergonomic, like `submit()`; the console's `ChatPage` is the same loop over the same routes)* |

`leaderboard` rows carry `MetricsSchema` (the challenge's declared metric columns) and `MeanMetricsDetail` (the precomputed per-metric aggregate) — passthrough columns in the returned DataFrame, no client code needed. `update_settings` accepts `evaluation_metrics_schema` (ordered descriptor list) + `evaluation_frontend_precision` to author them; `source:"env"` descriptor keys are exactly what `env.evaluate` must return in `metrics_detail`. `update_settings(submission_filename=…)` names a file_v1 challenge's upload (e.g. `submission.csv.gz`, which skips the `y_test.csv` column check); pair it with `upload_benchmark_file(id, path, filename=…)` for a binary benchmark — `update_benchmark_file_content` is text-only.
| `create_course`, `enroll_in_course`, `enrollment_info`, `list_courses` | `/api/academic_courses/` (list/enroll/create) | `services/coursesApi.ts`, `pages/EnrollPage.tsx` |
| `course_catalog`, `course`, `module_overview`, `lesson`, `mark_lesson_viewed`, `mark_lesson_complete`, `my_progress` | `/api/academic_courses/*` (consumption — catalog/landing/module/lesson/progress) | `services/coursesApi.ts` (learner, `04`) |
| `create_module`, `list_modules`, `get_module`, `update_module`, `delete_module`, `fork_module`, `attach_challenge`, `update_challenge_link`, `detach_challenge`, `reorder_module_challenges` | `/api/teacher/modules/*` | `services/coursesApi.ts` + `hooks/courseAuthoring/*` (`03`) |
| `create_lesson`, `get_lesson`, `update_lesson`, `delete_lesson`, `reorder_lessons`, `list_lesson_media`, `upload_lesson_media`, `download_lesson_media`, `delete_lesson_media`, `preview_lesson` | `/api/teacher/lessons/*`, `/api/teacher/modules/{id}/lessons/*` | `services/coursesApi.ts` + `hooks/courseAuthoring/*` (`03`) |
| `update_course`, `set_course_cover`, `list_course_modules`, `link_module`, `unlink_module`, `reorder_modules`, `course_progress` | `/api/teacher/course/{id}/*` | `services/coursesApi.ts` + `hooks/courseAuthoring/*` (`03`) |
| `author_course_from_dir`, `export_course_to_dir` | *(client-side compositions — no endpoint)* | *(SDK-only ergonomic, like `submit()`)* |

When the table above drifts from `client.py`, fix `client.py` — the table is a parity contract.

Payload keys are the backend's exactly (its request schemas are `extra="forbid"`): `submission_name`, `copy_from_submission_id`, `copy_from_challenge_id`, `max_active_submissions_per_participant`, `challenge_id` (module attach body and `list_courses` query). `submit()` returns `{"submission_id", "deploy"}`.

**Submission status is one block on the wire.** Every reply carrying a submission's `status` carries the same flat fields (`status`, `phase`, `last_status_message`, `status_update_ts`, `is_uploadable`, `is_deployable`, `is_settled`) — the backend's `SubmissionStatusFields`, the same block the console reads. The SDK **gates on those served fields and keeps no status set of its own**: `tail_logs()` stops on `is_settled`, `submit()` deploys only when `is_deployable`. `submission_status()` returns `submission_name` (not `name`); `/submissions/mine` rows send `phase` / `last_status_message` (not `lifecycle` / `error_info`) and no `is_started` — "is it live" is `status == "active"`. `is_started` on a *challenge* is a different field and is unchanged. The `course.yaml` manifest read by `author_course_from_dir` accepts the pre-2.0 keys `competitions:` / `competition_id:` permanently (files on teachers' disks); `export_course_to_dir` writes `challenges:` / `challenge_id:`.

**Chat challenges** (kernel `chat_v1`, plan `docs/plan_chat_kernel.md` §11) take conversations, not files: there is no upload and no deploy, the group's submission is created by the server on the first `open_chat_session` (born `active`), and `POST /api/submissions/challenge/<cid>` refuses such a challenge with 409. `open_chat_session` sends `{"charter_accepted": true}` — calling it is accepting the charter (`chat_challenge(cid)["manifest"]["charter_md"]`). `send_chat_message(wait=True)` is a composition of the message route and `chat_session` polling (700 ms, the console's cadence) until *that* `turn_id` is `completed` or `failed`; a failed turn raises `MLArenaError` with the turn's `error_message`. Money rides as two-decimal strings (`"300.00"`); the SDK returns them as served, and only `ChatConversation.total_eur` parses one into a `Decimal`. A 404 is `ChallengeNotFoundError` on a challenge-scoped route and `ChatSessionNotFoundError` on a session-scoped one. `update_chat_settings` sends only the keywords passed (sentinel default), so an explicit `None` clears `max_turns_per_session` / `max_sessions_per_participant`; `llm_api_key=""` clears the key, which is never read back (`llm_api_key_set`). Payload keys are the backend's exactly: `charter_accepted`, `content`, `reason`, `llm_base_url`, `llm_model`, `llm_api_key`, `turn_timeout_sec`, `max_turns_per_session`, `max_sessions_per_participant`.

`create_module(..., is_published=False)` / `update_module(id, is_published=…)` toggle the student-facing draft gate (distinct from `visibility`, which is teacher reuse); a manifest module block takes the same `is_published` key, defaulting to true so existing course dirs publish unchanged.

The course-content methods mirror the routes in `02-BACKEND-API.md` (`backend/app/views/teacher/{modules,lessons,course_content}.py` and `backend/app/views/academic_courses/{consumption,legacy,course_assets}.py`). `author_course_from_dir` / `export_course_to_dir` are pure compositions of the public authoring/consumption methods — they add no endpoint (the `submit()` idiom).

## Auth scopes

Token scope is encoded in the second segment (`mlk_<scope>_<lookup>_<secret>`). The backend's `auth_required` decorator enforces it:

- `user` — make and manage own submissions, chat with a chat challenge's agent (`open_chat_session`, `send_chat_message`, …), enroll in courses, read course content + write own lesson progress (`mark_lesson_*`, `my_progress`).
- `creator` — create / update / start challenges you own (also requires ownership or admin), including a chat challenge's LLM settings and session review (`chat_admin`, `update_chat_settings`, `void_chat_session`, …).
- `teacher` — author course content (`create_module`, `create_lesson`, `link_module`, …) and create academic courses. The `/api/teacher/*` routes require a `teacher`-scope token specifically.

Every read sends the bearer token, public routes included: `challenges`, `challenge` and `leaderboard` answer differently to a caller they can identify (enrolled-course and hidden challenges, `IsMySubmission`), so an anonymous read is not the same read. Course **consumption reads** (`course_catalog`, `course`, `module_overview`, `lesson`) are public for public/unlisted courses; gated lessons and progress writes require any authenticated token (they use `@login_required`, so a `user` token is enough — the bearer token authenticates via the backend's `request_loader`). A scope mismatch returns 403; the SDK surfaces this as `AuthenticationError`.

## Local dev

```bash
cd mlarena-sdk
pip install -e .
python -c "import mlarena; c = mlarena.connect(api_key='mlk_…', base_url='http://localhost:5000'); print(c.challenges())"
```

Point `base_url` at the local backend (`http://localhost:5000`) for end-to-end tests against minikube; the SDK has no local-mode shortcuts.
