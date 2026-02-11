import inspect
import os
import tempfile

import requests

from mlarena.exceptions import AuthenticationError, SubmissionError, CompetitionNotFoundError


class MLArenaClient:
    """Client for interacting with the ML Arena API."""

    def __init__(self, key_id, key_pass, base_url):
        self._key_id = key_id
        self._key_pass = key_pass
        self._base_url = base_url
        self._last_agent_id = None
        self._last_competition = None

    def _headers(self):
        return {"Authorization": f"Bearer {self._key_id}:{self._key_pass}"}

    def _url(self, path):
        return f"{self._base_url}/api/sdk{path}"

    def _handle_response(self, resp):
        if resp.status_code == 401:
            raise AuthenticationError("Invalid API credentials. Check your key_id and key_pass.")
        if resp.status_code == 403:
            raise AuthenticationError(resp.json().get("error", "Access denied"))
        if resp.status_code == 404:
            raise CompetitionNotFoundError(resp.json().get("error", "Not found"))
        return resp

    def submit(self, competition, agent=None, files=None, agent_name=None):
        """
        Submit an agent to a competition.

        Args:
            competition: Competition name or ID
            agent: A class whose source code will be extracted and submitted as agent.py
            files: List of file paths to upload (must include agent.py)
            agent_name: Optional name for the agent

        Returns:
            dict with agent_id, status, and message

        Raises:
            SubmissionError: If submission fails
            AuthenticationError: If credentials are invalid
        """
        if agent is None and files is None:
            raise SubmissionError("Provide either agent= (a class) or files= (list of paths)")
        if agent is not None and files is not None:
            raise SubmissionError("Provide either agent= or files=, not both")

        multipart_files = []
        tmp_dir = None

        try:
            if agent is not None:
                source = inspect.getsource(agent)
                tmp_dir = tempfile.mkdtemp()
                tmp_path = os.path.join(tmp_dir, "agent.py")
                with open(tmp_path, "w") as f:
                    f.write(source)
                multipart_files.append(("files", ("agent.py", open(tmp_path, "rb"))))
            else:
                for path in files:
                    if not os.path.isfile(path):
                        raise SubmissionError(f"File not found: {path}")
                    filename = os.path.basename(path)
                    multipart_files.append(("files", (filename, open(path, "rb"))))

            data = {}
            if agent_name:
                data["agent_name"] = agent_name

            resp = requests.post(
                self._url(f"/submit/{competition}"),
                headers=self._headers(),
                files=multipart_files,
                data=data,
                timeout=120,
            )

            self._handle_response(resp)

            if resp.status_code not in (200, 201, 202):
                error_msg = resp.json().get("error", resp.text)
                raise SubmissionError(f"Submission failed: {error_msg}")

            result = resp.json()
            self._last_agent_id = result.get("agent_id")
            self._last_competition = competition
            return result

        finally:
            for _, file_tuple in multipart_files:
                file_tuple[1].close()
            if tmp_dir:
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)

    def status(self, agent_id=None):
        """
        Get status of a submitted agent.

        Args:
            agent_id: Agent ID (defaults to last submitted agent)

        Returns:
            dict with agent status info
        """
        agent_id = agent_id or self._last_agent_id
        if agent_id is None:
            raise SubmissionError("No agent_id provided and no previous submission found")

        resp = requests.get(
            self._url(f"/status/{agent_id}"),
            headers=self._headers(),
            timeout=30,
        )
        self._handle_response(resp)
        resp.raise_for_status()
        return resp.json()

    def leaderboard(self, competition=None):
        """
        Get the leaderboard for a competition.

        Args:
            competition: Competition name or ID (defaults to last submitted competition)

        Returns:
            pandas.DataFrame if pandas is installed, otherwise list of dicts
        """
        competition = competition or self._last_competition
        if competition is None:
            raise SubmissionError("No competition specified and no previous submission found")

        resp = requests.get(
            self._url(f"/leaderboard/{competition}"),
            timeout=30,
        )
        self._handle_response(resp)
        resp.raise_for_status()
        data = resp.json()
        return _to_dataframe(data)

    def competitions(self):
        """
        List active competitions.

        Returns:
            pandas.DataFrame if pandas is installed, otherwise list of dicts
        """
        resp = requests.get(
            self._url("/competitions"),
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return _to_dataframe(data)

    def __repr__(self):
        return f"MLArenaClient(base_url='{self._base_url}', key_id='{self._key_id[:8]}...')"


def _to_dataframe(data):
    """Convert tabular response to DataFrame if pandas available, else list of dicts."""
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
