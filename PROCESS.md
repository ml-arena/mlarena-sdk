# mlarena-sdk

**Purpose:** Python SDK that wraps the public ML Arena REST API (`/api/...`). Lets users submit agents, author competitions, manage academic courses, and read leaderboards from a notebook or script.

**Key Features:**
- Bearer-token auth with scope-segmented keys (`mlk_user_…`, `mlk_creator_…`, `mlk_teacher_…`).
- One-shot helpers (`submit`, `status`, `leaderboard`) layered on top of the granular REST methods.
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
| `mlarena/exceptions.py` | `MLArenaError`, `AuthenticationError`, `CompetitionNotFoundError`, `SubmissionError` |

## Method ↔ route groups

Methods on `MLArenaClient` map 1:1 to backend blueprints — same shape the frontend uses.

| SDK method group | Backend blueprint | Frontend equivalent |
|---|---|---|
| `competitions()`, `competition()`, `datasets()`, `download_dataset()`, `list_tags()` | `/api/competitions`, `/api/competition_tags` | `services/competitionApi.js`, tag hooks |
| `create_competition`, `update_competition`, `update_settings`, `set_competition_tags`, `upload_env_file`, `update_env_file_content`, `set_competition_image`, `set_competition_markdown`, `upload_benchmark_file`, `update_benchmark_file_content`, `run_benchmark`, `benchmark_status`, `create_dataset`, `upload_dataset_file`, `start_competition` | `/api/creator_competition/*` | `services/creatorCompetitionApi.js` + `hooks/creatorCompetition/*` |
| `create_attached_agent`, `upload_agent_file`, `update_agent_file_content`, `list_agent_files`, `get_agent_file_content`, `delete_agent_file`, `deploy_agent`, `agent_deploy_status`, `agent_status`, `agent_games`, `tail_logs`, `runtime_options`, `agent_runtime`, `set_agent_runtime`, `resolve_runtime`, `delete_agent`, `submit`, `status` | `/api/direct_attache_agents/*` | `services/directAgentAttachApi.js` + `hooks/DirectAgentAttach/*` |
| `leaderboard` | `/api/leaderboard/competition/{id}` | `hooks/competition/useLeaderboardData` |
| `create_course`, `enroll_in_course`, `enrollment_info`, `list_courses` | `/api/academic_courses/` (list/enroll/create) | `services/teacherApi.js`, `pages/EnrollPage.js` |
| `course_catalog`, `course`, `module_overview`, `lesson`, `mark_lesson_viewed`, `mark_lesson_complete`, `my_progress` | `/api/academic_courses/*` (consumption — catalog/landing/module/lesson/progress) | `services/coursesApi.ts` (learner, `04`) |
| `create_module`, `list_modules`, `get_module`, `update_module`, `delete_module`, `fork_module`, `attach_competition`, `detach_competition`, `reorder_module_competitions` | `/api/teacher/modules/*` | `services/coursesApi.ts` + `hooks/courseAuthoring/*` (`03`) |
| `create_lesson`, `get_lesson`, `update_lesson`, `delete_lesson`, `reorder_lessons`, `upload_lesson_media`, `delete_lesson_media`, `preview_lesson` | `/api/teacher/lessons/*`, `/api/teacher/modules/{id}/lessons/*` | `services/coursesApi.ts` + `hooks/courseAuthoring/*` (`03`) |
| `update_course`, `set_course_cover`, `list_course_modules`, `link_module`, `unlink_module`, `reorder_modules`, `course_progress` | `/api/teacher/course/{id}/*` | `services/coursesApi.ts` + `hooks/courseAuthoring/*` (`03`) |
| `author_course_from_dir`, `export_course_to_dir` | *(client-side compositions — no endpoint)* | *(SDK-only ergonomic, like `submit()`)* |

When the table above drifts from `client.py`, fix `client.py` — the table is a parity contract.

The course-content methods mirror the routes in `02-BACKEND-API.md` (`backend/app/views/teacher/{modules,lessons,course_content}.py` and `backend/app/views/academic_courses/{consumption,legacy,course_assets}.py`). `author_course_from_dir` / `export_course_to_dir` are pure compositions of the public authoring/consumption methods — they add no endpoint (the `submit()` idiom).

## Auth scopes

Token scope is encoded in the second segment (`mlk_<scope>_<lookup>_<secret>`). The backend's `auth_required` decorator enforces it:

- `user` — submit agents, manage own attachments, enroll in courses, read course content + write own lesson progress (`mark_lesson_*`, `my_progress`).
- `creator` — create / update / start competitions you own (also requires ownership or admin).
- `teacher` — author course content (`create_module`, `create_lesson`, `link_module`, …) and create academic courses. The `/api/teacher/*` routes require a `teacher`-scope token specifically.

Course **consumption reads** (`course_catalog`, `course`, `module_overview`, `lesson`) are public for public/unlisted courses; gated lessons and progress writes require any authenticated token (they use `@login_required`, so a `user` token is enough — the bearer token authenticates via the backend's `request_loader`). A scope mismatch returns 403; the SDK surfaces this as `AuthenticationError`.

## Local dev

```bash
cd mlarena-sdk
pip install -e .
python -c "import mlarena; c = mlarena.connect(api_key='mlk_…', base_url='http://localhost:5000'); print(c.competitions())"
```

Point `base_url` at the local backend (`http://localhost:5000`) for end-to-end tests against minikube; the SDK has no local-mode shortcuts.
