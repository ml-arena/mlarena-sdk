# mlarena

Python SDK for [ML Arena](https://ml-arena.com) — submit agents, manage competitions, manage courses, and read leaderboards from any notebook or IDE.

## Install

```bash
pip install mlarena-sdk
```

## Quick Start

```python
import mlarena

# Connect with your API key (Profile page → "API Keys"). The token is the full
# string starting with `mlk_…`, not a `key_id:key_pass` pair.
client = mlarena.connect(api_key="mlk_user_a1b2c3d4_<32-hex-secret>")

# List competitions (public, no auth)
client.competitions()

# Submit an agent class — creates an attachment, uploads, and deploys.
class MyAgent:
    def predict(self, observation):
        return 0

result = client.submit(competition_id=42, agent=MyAgent)

# Or submit files from disk
client.submit(competition_id=42, files=["agent.py", "model.pkl"])

# Check status of the last submission
client.status()

# View leaderboard (returns DataFrame if pandas is installed)
client.leaderboard(42)
```

## Auth & scopes

The token's *scope* segment dictates which routes you can call:

- `mlk_user_…` — submit agents, check status, manage your own attachments, enroll in and read courses, track your own lesson progress.
- `mlk_creator_…` — create / update competitions you own.
- `mlk_teacher_…` — create academic courses and author course content (modules, lessons, course composition).

A `user`-scope token cannot call a `creator`-required route (and vice versa). Mint scope-specific tokens from your Profile page.

## API reference

### `mlarena.connect(api_key, base_url="https://ml-arena.com")`

Create a client. `api_key` must be the full `mlk_<scope>_<lookup>_<secret>` token.

### Agents (user scope)

- `client.submit(competition_id, agent=None, files=None, agent_name=None, runtime_id=None, runtime=None)` — one-shot create + (pick runner) + upload + deploy.
- `client.create_attached_agent(competition_id, agent_name, copy_from_agent_id=None)`
- `client.upload_agent_file(competition_id, attache_agent_id, file_path)` — multipart upload from disk.
- `client.update_agent_file_content(competition_id, attache_agent_id, filename, content)` — upload from a string (template render → upload).
- `client.list_agent_files(competition_id, attache_agent_id)` — list files with their content / binary marker.
- `client.get_agent_file_content(competition_id, attache_agent_id, filename)` — fetch one file's text.
- `client.delete_agent_file(competition_id, attache_agent_id, filename)`
- `client.deploy_agent(competition_id, attache_agent_id)`
- `client.delete_agent(competition_id, attache_agent_id)`
- `client.agent_status(competition_id, attache_agent_id)` — rich status (queue, runs, errors).
- `client.agent_deploy_status(competition_id, attache_agent_id)` — deploy quotas + last deploy.
- `client.agent_games(attache_agent_id)` — recent games with signed log URLs (60-day GCS retention).
- `client.tail_logs(competition_id, attache_agent_id, follow=False, poll_sec=5.0)` — generator of status / run lines.
- `client.status(agent_id=None, competition_id=None)` — defaults to the last submission.

### Runners (DockerImageAgentRuntime, user scope)

- `client.runtime_options(competition_id)` — list runtimes compatible with the competition.
- `client.agent_runtime(attache_agent_id)` — read the runtime currently pinned to an agent.
- `client.set_agent_runtime(attache_agent_id, runtime_id)` — pin a runtime by id.
- `client.resolve_runtime(competition_id, language=None, framework=None, framework_version=None)` — resolve a (lang, framework, version) spec to one runtime row.

## Full participant workflow

```python
import mlarena, requests

c = mlarena.connect("mlk_user_…", base_url="http://localhost:5000")

cid = 42  # competition id

# 1. Pick a runner (language × framework)
runtimes = c.runtime_options(cid)
py_gym = c.resolve_runtime(cid, language="python", framework="gymnasium")

# 2. Create the agent + pin runner + upload files + deploy in one call
sub = c.submit(cid, files=["agent.py", "model.pkl"], runtime_id=py_gym["id"])
aid = sub["attache_agent_id"]

# 3. Inspect / edit a file in place after the initial upload
src = c.get_agent_file_content(cid, aid, "agent.py")
c.update_agent_file_content(cid, aid, "agent.py", src.replace("epsilon=0.1", "epsilon=0.05"))
c.deploy_agent(cid, aid)  # redeploy after edit

# 4. Watch status / run progress until terminal
for line in c.tail_logs(cid, aid):
    print(line)

# 5. Pull stdout from completed games via signed URLs (60d retention)
for game in c.agent_games(aid)["games"]:
    if game["signed_url"]:
        print(requests.get(game["signed_url"]).text)

# 6. Read the leaderboard
print(c.leaderboard(cid).head())
```

### Competitions

- `client.competitions()` — public list.
- `client.create_competition(name, kernel_version, description=None, copy_from_competition_id=None, tag_names=None)` — creator scope. The backend resolves the engine + default evaluation + default env runtime from `kernel_version`. Pass `tag_names=["rl", "research"]` to attach tags at creation time; unknown names raise `MLArenaError`.
- `client.list_tags()` — public read of the tag catalog.
- `client.set_competition_tags(competition_id, tag_names=None, tag_ids=None)` — creator scope. Replaces the tag set on a competition you own; pass `[]` to clear all tags.

### Datasets (file competitions)

For file competitions the creator publishes the participant-facing data as a
**dataset** (stored in GCS, served as short-lived signed URLs):

- `client.create_dataset(competition_id, label, description=None)` — creator scope. Make a dataset bucket (before `start_competition`).
- `client.upload_dataset_file(competition_id, dataset_id, file_path)` — creator scope. Add a file to the bucket.
- `client.datasets(competition_id)` — any scope. List datasets + files with signed `download_url`s.
- `client.download_dataset(competition_id, dest_dir=".")` — any scope. Stream every published file into `dest_dir`. This is the call a starter notebook makes to fetch the train/test data.

### Academic courses

A course is composed of reusable **modules**; each module holds **lessons**
(markdown) and may attach **competitions**. Authoring (`create_module`,
`create_lesson`, `link_module`, …) needs a `teacher`-scope token; reading and
enrolling need only a `user` token. See the SDK [`PROCESS.md`](PROCESS.md)
method↔route table for the full surface.

**Create + enroll**

- `client.create_course(name, code=None, start_date, end_date, slug=None, description=None, visibility=None, instructor_name=None)` — any token scope; creating a course makes the account a teacher (capped by `max_courses_limit`, default 1; admins exempt). Attach competitions afterwards via modules (`create_module` / `attach_competition` / `link_module`). The response carries both `enrollment_link` (32-hex) and a short shareable `join_code`.
- `client.enroll_in_course(link_or_code, student_email=None, student_number=None, project_url=None)` — accepts either the enrollment link **or** the short join code (also `join_code=` / `enrollment_link=`).
- `client.enrollment_info(link_or_code)` — preview a course before enrolling (public).
- `client.list_courses(show_all=False, competition_id=None)` — your enrolled + active courses.

**Author a whole course from a directory**

```python
import mlarena

teacher = mlarena.connect(api_key="mlk_teacher_…")

# my-course/course.yaml describes the course; lesson bodies are markdown files
# referenced from the manifest (see author_course_from_dir's docstring for the
# full schema). This is a pure composition of the authoring methods — no
# special endpoint, the same idiom as submit().
result = teacher.author_course_from_dir("my-course/")
print(result["join_code"])        # share this code with students

# Round-trip the other way for backup / versioning:
teacher.export_course_to_dir("intro-to-rl", "backup/")
```

**Enroll by join code, then read + complete lessons**

```python
import mlarena

student = mlarena.connect(api_key="mlk_user_…")
student.enroll_in_course("JOINME", student_email="s@uni.edu", student_number="42")

landing = student.course("intro-to-rl")                  # modules + lesson TOC
for module in landing["modules"]:
    for toc in module["lessons"]:
        page = student.lesson("intro-to-rl", module["slug"], toc["slug"])
        print(page["body_md"])                           # full markdown body
        student.mark_lesson_complete(toc["id"])

print(student.my_progress(landing["id"]))                # content % + next lesson
```

### Course authoring (teacher scope)

- `client.create_module(title, slug=None, summary=None, icon=None, visibility="private")`, `list_modules(library=None)`, `get_module`, `update_module(id, **fields)`, `delete_module(id, force=False)`, `fork_module(id)`.
- `client.attach_competition(module_id, competition_id, label=None, position=None)`, `detach_competition`, `reorder_module_competitions`.
- `client.create_lesson(module_id, title, kind="lesson", slug=None, parent_lesson_id=None, body_md="", gated=False)`, `get_lesson`, `update_lesson(id, **fields)`, `delete_lesson`, `reorder_lessons`.
- `client.upload_lesson_media(lesson_id, file_path)`, `delete_lesson_media`, `preview_lesson(lesson_id, body_md=None)` — validates `mlarena:` directives (fails loud on unknown).
- `client.update_course(id, **fields)`, `set_course_cover`, `list_course_modules`, `link_module(course_id, module_id, position=None)`, `unlink_module`, `reorder_modules`, `course_progress(course_id)` — teacher follow dashboard.

### Course consumption (public / user scope)

- `client.course_catalog(search=None, limit=None, offset=None)` — public courses.
- `client.course(slug)`, `client.module_overview(slug, module_slug)`, `client.lesson(slug, module_slug, lesson_slug)`.
- `client.mark_lesson_viewed(lesson_id, course_id=None)`, `client.mark_lesson_complete(lesson_id, course_id=None)`, `client.my_progress(course_id)`.

### Leaderboard

- `client.leaderboard(competition_id=None)` — defaults to last competition; returns DataFrame if pandas is installed.

## Get your API key

1. Go to [ml-arena.com](https://ml-arena.com).
2. Open your Profile page.
3. Mint a key for the scope you need (`user`, `creator`, or `teacher`).
4. Copy the full token (shown once) — it starts with `mlk_`.
