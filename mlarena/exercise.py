"""Exercise challenges (kernel `exercise_v1`): the helpers behind `client.check()`.

An exercise challenge holds the short Python exercises of one course module.
A participant sends code (or, for a `text` / `choice` exercise, an answer);
the challenge's resident checker runs it against the creator's hidden asserts
and answers within about a second. The public routes are
`/api/exercise/*` (`MLArenaClient.exercise_challenge`, `submit_exercise`,
`exercise_attempt`, `exercise_attempts`); this module holds what the
notebook-friendly composition needs on top of them:

    def calculer_imc(poids_kg, taille_m):
        return round(poids_kg / taille_m ** 2, 1)

    client.check(42, "imc", calculer_imc)   # prints "imc: passed (8 ms)"

`source_of(obj)` is the code `check()` sends; `format_attempt(attempt)` is the
line it prints.
"""
from __future__ import annotations

import inspect
import textwrap

from mlarena.exceptions import MLArenaError


def source_of(obj) -> str:
    """The code `check()` sends for `obj`.

    A function or a class is sent as its own source (`inspect.getsource`,
    dedented so a definition nested in a block still parses); a `str` is sent
    as it is, which is how to send several definitions (a class and the helper
    it uses) or imports. Only the object's own definition travels: a helper it
    calls, or a module-level import it relies on, must be in a `str`.

    Raises `TypeError` for anything else, and `MLArenaError` when Python has
    no source for the object (defined in a bare REPL or by `exec`) — pass the
    code as a `str` then.
    """
    if isinstance(obj, str):
        return obj
    if not (inspect.isfunction(obj) or inspect.isclass(obj)):
        raise TypeError(
            f"check() takes a function, a class or a str of code, got "
            f"{type(obj).__name__}"
        )
    try:
        source = inspect.getsource(obj)
    except (OSError, TypeError) as e:
        raise MLArenaError(
            f"check(): no source code for {obj.__qualname__} ({e}); pass the "
            "code as a str instead"
        ) from e
    return textwrap.dedent(source)


def format_attempt(attempt: dict) -> str:
    """One settled attempt as the line `check()` prints.

    `<key>: passed (<duration> ms)` for a pass; otherwise
    `<key>: <status> (<error_type>)`, then the assertion or message, the
    student-code line for a syntax / runtime error and the captured stdout,
    each on its own indented line, as the server serves them
    (`ExerciseAttemptOut` / `ExerciseFeedback`).
    """
    key = attempt["exercise_key"]
    if attempt["status"] == "passed":
        duration = attempt["duration_ms"]
        return f"{key}: passed" + ("" if duration is None else f" ({duration} ms)")
    lines = [f"{key}: {attempt['status']} ({attempt['error_type']})"]
    feedback = attempt["feedback"]
    reason = attempt["message"]
    if feedback is not None and feedback["assertion"]:
        reason = feedback["assertion"]
    if reason:
        lines.append(f"  {reason}")
    if feedback is not None:
        if feedback["line"] is not None:
            lines.append(f"  at line {feedback['line']} of your code")
        if feedback["expected_hint"]:
            lines.append(f"  hint: {feedback['expected_hint']}")
        if feedback["stdout"]:
            lines.append("  output:")
            lines.extend(f"    {line}" for line in feedback["stdout"].rstrip("\n").splitlines())
    return "\n".join(lines)
