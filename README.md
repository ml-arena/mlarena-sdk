# mlarena

Python SDK for [ML Arena](https://ml-arena.com) - submit agents, check status, and view leaderboards from any notebook or IDE.

## Install

```bash
pip install mlarena
```

## Quick Start

```python
import mlarena

# Connect with your API key (from Profile page on ml-arena.com)
client = mlarena.connect(api_key="your_key_id:your_key_pass")

# List competitions
client.competitions()

# Submit an agent class
class MyAgent:
    def predict(self, observation):
        return 0

client.submit("chess-challenge", agent=MyAgent)

# Or submit files from disk
client.submit("chess-challenge", files=["agent.py", "model.pkl"])

# Check submission status
client.status()

# View leaderboard (returns DataFrame if pandas is installed)
client.leaderboard("chess-challenge")
```

## API Reference

### `mlarena.connect(api_key, base_url="https://ml-arena.com")`

Create a client connection. `api_key` format: `"key_id:key_pass"`.

### `client.submit(competition, agent=None, files=None, agent_name=None)`

Submit an agent to a competition.

- `competition` - Competition name or ID
- `agent` - A Python class (source code is extracted automatically)
- `files` - List of file paths to upload (must include `agent.py`)
- `agent_name` - Optional display name

### `client.status(agent_id=None)`

Get status of a submitted agent. Defaults to last submission.

### `client.leaderboard(competition)`

Get competition leaderboard. Returns `pandas.DataFrame` if pandas is installed.

### `client.competitions()`

List active competitions.

## Get Your API Key

1. Go to [ml-arena.com](https://ml-arena.com)
2. Navigate to your Profile page
3. Click "Generate New API Key"
4. Copy your `key_id` and `key_pass`
