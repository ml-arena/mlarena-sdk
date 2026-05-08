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

- `client.submit(competition_id, agent=None, files=None, agent_name=None)` — one-shot create + upload + deploy.
- `client.create_attached_agent(competition_id, agent_name, copy_from_agent_id=None)`
- `client.upload_agent_file(competition_id, attache_agent_id, file_path)`
- `client.deploy_agent(competition_id, attache_agent_id)`
- `client.agent_deploy_status(competition_id, attache_agent_id)`
- `client.status(agent_id=None, competition_id=None)` — defaults to the last submission.

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
