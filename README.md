# mlarena

Python SDK for [ML Arena](https://ml-arena.com) — submit agents, manage challenges, manage courses, and read leaderboards from any notebook or IDE.

## 1.0.0 — "competition" is now "challenge"

The platform vocabulary changed: what used to be a *competition* is a
**challenge**. The SDK follows, so `client.competitions()` is now
`client.challenges()`, and `competition_id=` is now `challenge_id=`.

**Existing code keeps working.** Every old method name still resolves and every
`competition_id=` keyword is still accepted; both emit a `DeprecationWarning`
and forward to the new spelling. So this:

```python
client.competitions()
client.leaderboard(competition_id=43)
```

still runs, and tells you to move to:

```python
client.challenges()
client.leaderboard(challenge_id=43)
```

To find every call site in a notebook, run Python with warnings visible:
`python -W once::DeprecationWarning your_script.py`.

Renamed: `competitions`→`challenges`, `competition`→`challenge`,
`creator_competitions`→`creator_challenges`, `create_competition`→`create_challenge`,
`set_competition_tags`→`set_challenge_tags`, `update_competition`→`update_challenge`,
`set_competition_image`→`set_challenge_image`, `set_competition_markdown`→`set_challenge_markdown`,
`start_competition`→`start_challenge`, `stop_competition`→`stop_challenge`,
`attach_competition`→`attach_challenge`, `update_competition_link`→`update_challenge_link`,
`detach_competition`→`detach_challenge`, `reorder_module_competitions`→`reorder_module_challenges`.
`CompetitionNotFoundError` is now `ChallengeNotFoundError` (the old name is an
alias for the same class, so `except CompetitionNotFoundError` still catches it).

The REST routes are unchanged — the server still speaks `/api/competitions`.
Only the client vocabulary moved.

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

# List challenges (public, no auth)
client.challenges()

# Submit an agent class — creates an attachment, uploads, and deploys.
class MyAgent:
    def predict(self, observation):
        return 0

result = client.submit(challenge_id=42, agent=MyAgent)

# Or submit files from disk
client.submit(challenge_id=42, files=["agent.py", "model.pkl"])

# Check status of the last submission
client.status()

# View leaderboard (returns DataFrame if pandas is installed)
client.leaderboard(42)
```

## Auth & scopes

The token's *scope* segment dictates which routes you can call:

- `mlk_user_…` — submit agents, check status, manage your own attachments, enroll in and read courses, track your own lesson progress.
- `mlk_creator_…` — create / update challenges you own.
- `mlk_teacher_…` — create academic courses and author course content (modules, lessons, course composition).

A `user`-scope token cannot call a `creator`-required route (and vice versa). Mint scope-specific tokens from your Profile page.

## API reference

### `mlarena.connect(api_key, base_url="https://ml-arena.com")`

Create a client. `api_key` must be the full `mlk_<scope>_<lookup>_<secret>` token.

### Agents (user scope)

- `client.submit(challenge_id, agent=None, files=None, agent_name=None, runtime_id=None, runtime=None)` — one-shot create + (pick runner) + upload + deploy.
- `client.create_attached_agent(challenge_id, agent_name, copy_from_agent_id=None)`
- `client.upload_agent_file(challenge_id, attache_agent_id, file_path)` — multipart upload from disk.
- `client.update_agent_file_content(challenge_id, attache_agent_id, filename, content)` — upload from a string (template render → upload).
- `client.list_agent_files(challenge_id, attache_agent_id)` — list files with their content / binary marker.
- `client.get_agent_file_content(challenge_id, attache_agent_id, filename)` — fetch one file's text.
- `client.delete_agent_file(challenge_id, attache_agent_id, filename)`
- `client.deploy_agent(challenge_id, attache_agent_id)`
- `client.delete_agent(challenge_id, attache_agent_id)`
- `client.agent_status(challenge_id, attache_agent_id)` — rich status (queue, runs, errors).
- `client.agent_deploy_status(challenge_id, attache_agent_id)` — deploy quotas + last deploy.
- `client.agent_games(attache_agent_id)` — recent games with signed log URLs (60-day GCS retention).
- `client.tail_logs(challenge_id, attache_agent_id, follow=False, poll_sec=5.0)` — generator of status / run lines.
- `client.status(agent_id=None, challenge_id=None)` — defaults to the last submission.

### Runners (DockerImageAgentRuntime, user scope)

- `client.runtime_options(challenge_id)` — list runtimes compatible with the challenge.
- `client.agent_runtime(attache_agent_id)` — read the runtime currently pinned to an agent.
- `client.set_agent_runtime(attache_agent_id, runtime_id)` — pin a runtime by id.
- `client.resolve_runtime(challenge_id, language=None, framework=None, framework_version=None)` — resolve a (lang, framework, version) spec to one runtime row.

## Full participant workflow

```python
import mlarena, requests

c = mlarena.connect("mlk_user_…", base_url="http://localhost:5000")

cid = 42  # challenge id

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

### Challenges

- `client.challenges()` — public list.
- `client.create_challenge(name, kernel_version, description=None, copy_from_challenge_id=None, tag_names=None)` — creator scope. The backend resolves the engine + default evaluation + default env runtime from `kernel_version`. Pass `tag_names=["rl", "research"]` to attach tags at creation time; unknown names raise `MLArenaError`.
- `client.list_tags()` — public read of the tag catalog.
- `client.set_challenge_tags(challenge_id, tag_names=None, tag_ids=None)` — creator scope. Replaces the tag set on a challenge you own; pass `[]` to clear all tags.

### Datasets (file challenges)

For file challenges the creator publishes the participant-facing data as a
**dataset** (stored in GCS, served as short-lived signed URLs):

- `client.create_dataset(challenge_id, label, description=None)` — creator scope. Make a dataset bucket (before `start_challenge`).
- `client.upload_dataset_file(challenge_id, dataset_id, file_path)` — creator scope. Add a file to the bucket.
- `client.datasets(challenge_id)` — any scope. List datasets + files with signed `download_url`s.
- `client.download_dataset(challenge_id, dest_dir=".")` — any scope. Stream every published file into `dest_dir`. This is the call a starter notebook makes to fetch the train/test data.

### Academic courses

A course is composed of reusable **modules**; each module holds **lessons**
(markdown) and may attach **challenges**. Authoring (`create_module`,
`create_lesson`, `link_module`, …) needs a `teacher`-scope token; reading and
enrolling need only a `user` token. See the SDK [`PROCESS.md`](PROCESS.md)
method↔route table for the full surface.

**Create + enroll**

- `client.create_course(name, code=None, start_date, end_date, slug=None, description=None, visibility=None, instructor_name=None)` — any token scope; creating a course makes the account a teacher (capped by `max_courses_limit`, default 1; admins exempt). Attach challenges afterwards via modules (`create_module` / `attach_challenge` / `link_module`). The response carries the course's `join_code` — the single enrollment token to share with students.
- `client.enroll_in_course(join_code, student_email=None, student_number=None, project_url=None)` — join with the course's short join code (the token in its `/enroll/<join_code>` link).
- `client.enrollment_info(join_code)` — preview a course before enrolling (public).
- `client.list_courses(show_all=False, challenge_id=None)` — your enrolled + active courses.

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
- `client.attach_challenge(module_id, challenge_id, label=None, position=None)`, `detach_challenge`, `reorder_module_challenges`.
- `client.create_lesson(module_id, title, kind="lesson", slug=None, parent_lesson_id=None, body_md="", gated=False)`, `get_lesson`, `update_lesson(id, **fields)`, `delete_lesson`, `reorder_lessons`.
- `client.upload_lesson_media(lesson_id, file_path)`, `delete_lesson_media`, `preview_lesson(lesson_id, body_md=None)` — validates `mlarena:` directives (fails loud on unknown).
- `client.update_course(id, **fields)`, `set_course_cover`, `list_course_modules`, `link_module(course_id, module_id, position=None)`, `unlink_module`, `reorder_modules`, `course_progress(course_id)` — teacher follow dashboard.

### Course consumption (public / user scope)

- `client.course_catalog(search=None, limit=None, offset=None)` — public courses.
- `client.course(slug)`, `client.module_overview(slug, module_slug)`, `client.lesson(slug, module_slug, lesson_slug)`.
- `client.mark_lesson_viewed(lesson_id, course_id=None)`, `client.mark_lesson_complete(lesson_id, course_id=None)`, `client.my_progress(course_id)`.

### Leaderboard

- `client.leaderboard(challenge_id=None)` — defaults to last challenge; returns DataFrame if pandas is installed.

## Get your API key

1. Go to [ml-arena.com](https://ml-arena.com).
2. Open your Profile page.
3. Mint a key for the scope you need (`user`, `creator`, or `teacher`).
4. Copy the full token (shown once) — it starts with `mlk_`.
