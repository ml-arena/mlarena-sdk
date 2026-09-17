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
| `mlarena/exceptions.py` | `MLArenaError`, `AuthenticationError`, `ChallengeNotFoundError` (old name `CompetitionNotFoundError` is the same class), `SubmissionError` |

## Method ↔ route groups

Methods on `MLArenaClient` map 1:1 to backend blueprints — same shape the frontend uses.

| SDK method group | Backend blueprint | Frontend equivalent |
|---|---|---|
| `challenges()`, `challenge()`, `datasets()`, `download_dataset()`, `recent_replays()`, `list_tags()` | `/api/challenges`, `/api/challenge_tags` | `services/apiService.ts`, `pages/Challenge/View.js` (datasets), `components/Challenge/RecentReplays.js`, `hooks/creatorChallenge/useTags.ts` |
| `update_challenge_configuration` (admin account: engine pin, runtime, step limits) | `PUT /api/challenges/{id}/configuration` | `hooks/creatorChallenge/useAdmin.ts` |
| `create_challenge`, `update_challenge`, `update_settings`, `set_challenge_tags`, `upload_env_file`, `update_env_file_content`, `set_challenge_image`, `set_challenge_markdown`, `upload_benchmark_file`, `update_benchmark_file_content`, `run_benchmark`, `benchmark_status`, `create_dataset`, `upload_dataset_file`, `start_challenge`, `update_agent_template` | `/api/creator_challenge/*` (`…/challenge/{id}/agent-template` keeps its segment) | `services/creatorChallengeApi.ts` + `hooks/creatorChallenge/*` |
| `create_submission`, `upload_submission_file`, `update_submission_file_content`, `list_submission_files`, `get_submission_file_content`, `delete_submission_file`, `deploy_submission`, `submission_deploy_status`, `submission_status`, `submission_games`, `tail_logs`, `runtime_options`, `agent_runtime`, `set_agent_runtime`, `resolve_runtime`, `delete_submission`, `submit`, `status` | `/api/submissions/*` (`challenge/{cid}/{sid}/…`, `submission/{sid}/games`, `runtime_options/{cid}`, `agent_runtime/{sid}`) | `hooks/Submission/*` (`useSubmission`, `useSubmissionDeploy`, `useSubmissionFileManagement`, `useRuntimeOptions`) |
| `leaderboard` | `/api/leaderboard/challenge/{id}` | `services/leaderboardApi.ts`, `hooks/challenge/useLeaderboardData` |

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

`create_module(..., is_published=False)` / `update_module(id, is_published=…)` toggle the student-facing draft gate (distinct from `visibility`, which is teacher reuse); a manifest module block takes the same `is_published` key, defaulting to true so existing course dirs publish unchanged.

The course-content methods mirror the routes in `02-BACKEND-API.md` (`backend/app/views/teacher/{modules,lessons,course_content}.py` and `backend/app/views/academic_courses/{consumption,legacy,course_assets}.py`). `author_course_from_dir` / `export_course_to_dir` are pure compositions of the public authoring/consumption methods — they add no endpoint (the `submit()` idiom).

## Auth scopes

Token scope is encoded in the second segment (`mlk_<scope>_<lookup>_<secret>`). The backend's `auth_required` decorator enforces it:

- `user` — make and manage own submissions, enroll in courses, read course content + write own lesson progress (`mark_lesson_*`, `my_progress`).
- `creator` — create / update / start challenges you own (also requires ownership or admin).
- `teacher` — author course content (`create_module`, `create_lesson`, `link_module`, …) and create academic courses. The `/api/teacher/*` routes require a `teacher`-scope token specifically.

Course **consumption reads** (`course_catalog`, `course`, `module_overview`, `lesson`) are public for public/unlisted courses; gated lessons and progress writes require any authenticated token (they use `@login_required`, so a `user` token is enough — the bearer token authenticates via the backend's `request_loader`). A scope mismatch returns 403; the SDK surfaces this as `AuthenticationError`.

## Local dev

```bash
cd mlarena-sdk
pip install -e .
python -c "import mlarena; c = mlarena.connect(api_key='mlk_…', base_url='http://localhost:5000'); print(c.challenges())"
```

Point `base_url` at the local backend (`http://localhost:5000`) for end-to-end tests against minikube; the SDK has no local-mode shortcuts.
