# mlarena

Python SDK for [ML Arena](https://ml-arena.com) — submit agents, manage competitions, manage courses, and read leaderboards from any notebook or IDE.

## Install

```bash
pip install mlarena
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

- `mlk_user_…` — submit agents, check status, manage your own attachments.
- `mlk_creator_…` — create / update competitions you own.
- `mlk_teacher_…` — create academic courses.

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

### Academic courses

- `client.create_course(name, code, start_date, end_date, instructor_name=None, competition_id=None, enrollment_link=None)` — teacher scope.
- `client.enroll_in_course(enrollment_link, student_email=None, student_number=None, project_url=None)` — user scope.

### Leaderboard

- `client.leaderboard(competition_id=None)` — defaults to last competition; returns DataFrame if pandas is installed.

## Get your API key

1. Go to [ml-arena.com](https://ml-arena.com).
2. Open your Profile page.
3. Mint a key for the scope you need (`user`, `creator`, or `teacher`).
4. Copy the full token (shown once) — it starts with `mlk_`.
