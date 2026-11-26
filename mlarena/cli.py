"""The `mlarena` command: submit, push a challenge, push or export a course.

    mlarena whoami
    mlarena submit --challenge <slug-or-id> [--files a b … | --dir DIR] …
    mlarena challenge push [DIR] --challenge <slug-or-id> [--path SUBPATH] [--dry-run]
    mlarena course push [DIR] [--slug SLUG] [--path SUBPATH] [--dry-run] [--prune]
    mlarena course export SLUG [DIR] [--overwrite]

Each command is a composition of `MLArenaClient` methods: the CLI has no
route of its own. The credential is resolved by `mlarena.connect()`
(`MLARENA_API_KEY`, else the GitHub Actions OIDC exchange), so the same line
runs on a laptop and in a workflow. Inside GitHub Actions the outcome is also
appended to the job summary, and a failure is an `::error::` annotation.

Exit codes: 0 done, 1 refused by the server or failed, 2 usage.
"""

import argparse
import fnmatch
import os
import shutil
import sys
import tempfile

import mlarena
from mlarena.bundle import DEFAULT_EXCLUDES
from mlarena.ci_auth import (
    ENV_BASE_URL,
    KIND_GITHUB_ACTIONS,
    PROD_BASE_URL,
    oidc_audience,
)
from mlarena.exceptions import MLArenaError, WaitTimeoutError

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2

# The backend's submission name limit.
SUBMISSION_NAME_MAX = 40
# `submit --wait` gives up after this long unless --timeout says otherwise.
DEFAULT_WAIT_TIMEOUT_SEC = 30 * 60
FILE_KERNEL = "file_v1"


class UsageError(Exception):
    """A command line the CLI refuses before any request (exit 2)."""


# --------------------------------------------------------------------------- #
# Output: stdout / stderr, plus the job summary and annotations in Actions
# --------------------------------------------------------------------------- #


def _in_github_actions() -> bool:
    return os.environ.get("GITHUB_ACTIONS") == "true"


def _out(line: str = "") -> None:
    print(line)


def _err(line: str) -> None:
    print(line, file=sys.stderr)


def _step_summary(markdown: str) -> None:
    """Append to the job summary (GitHub Actions only)."""
    if not _in_github_actions():
        return
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(markdown.rstrip("\n") + "\n\n")


def _set_outputs(outputs: dict) -> None:
    """Append `key=value` lines to `$GITHUB_OUTPUT` (the step's outputs), when
    it is set. Booleans are `true` / `false`; a value is one line."""
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as fh:
        for key, value in outputs.items():
            if isinstance(value, bool):
                value = "true" if value else "false"
            text = " ".join(str(value).splitlines())
            fh.write(f"{key}={text}\n")


def _annotate_error(message: str) -> None:
    """An `::error::` workflow command: the message shows on the run page."""
    if not _in_github_actions():
        return
    escaped = message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    print(f"::error title=mlarena::{escaped}")


def _fail(message: str, *, heading: str) -> int:
    _err(f"error: {message}")
    _annotate_error(message)
    _step_summary(f"### {heading}\n\n**Failed.** {message}")
    return EXIT_FAILED


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #


def _connect(base_url: str, scope: str, audience: str | None):
    """The client of a command (tests replace this function)."""
    return mlarena.connect(base_url=base_url, scope=scope, audience=audience)


def _site_url(args) -> str:
    """The console's URL: the OIDC audience, which is the platform's public
    URL (the base URL unless overridden, as on a local stack)."""
    return oidc_audience(args.base_url, args.audience)


def _resolve_challenge(client, value: str) -> dict:
    """`--challenge` is a numeric id or the slug of `/challenges/<slug>`."""
    if value.isdigit():
        return client.challenge(int(value))
    return client.challenge_by_slug(value)


def _revision_label(revision: dict) -> str:
    """A commit of a repository, or a change made on the platform (a push
    from outside a GitHub job: no repository, no commit)."""
    if revision["provider"] == "github":
        return f"{revision['repository']}@{revision['commit_sha'][:7]}"
    return f"platform ({revision['transport']})"


def _is_excluded(name: str) -> bool:
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in DEFAULT_EXCLUDES)


# --------------------------------------------------------------------------- #
# whoami
# --------------------------------------------------------------------------- #


def _auth_label(credentials) -> str:
    if credentials.kind == KIND_GITHUB_ACTIONS:
        return (f"GitHub Actions run of {_revision_label(credentials.source_revision)} "
                f"(scope {credentials.scope})")
    return f"API key (scope {credentials.scope})"


def cmd_whoami(args) -> int:
    client = _connect(args.base_url, "user", args.audience)
    me = client.me()
    # The route answers 200 to any bearer (the console's anonymous probe).
    if not me["is_authenticated"]:
        return _fail(
            f"the credential was not accepted by {args.base_url} (unknown, "
            f"revoked or expired); mint a key on your Profile page",
            heading="mlarena whoami")
    _out(f"username: {me['username']}")
    _out(f"auth:     {_auth_label(client.credentials)}")
    _out(f"base URL: {args.base_url}")
    return EXIT_OK


# --------------------------------------------------------------------------- #
# submit
# --------------------------------------------------------------------------- #


def default_submission_name(environ=None) -> str | None:
    """The tag when the run was triggered by one (`submission-3`), else
    `gh-<sha7>`; None outside GitHub Actions (the SDK's own default then)."""
    environ = os.environ if environ is None else environ
    if environ.get("GITHUB_REF_TYPE") == "tag" and environ.get("GITHUB_REF_NAME"):
        return environ["GITHUB_REF_NAME"][:SUBMISSION_NAME_MAX]
    if environ.get("GITHUB_SHA"):
        return f"gh-{environ['GITHUB_SHA'][:7]}"
    return None


def _submission_files(args) -> list[str]:
    """The files to upload. Uploads are flat (each file is stored under its
    base name), so a folder is refused rather than flattened."""
    if args.files and args.dir:
        raise UsageError("pass --files or --dir, not both")
    if args.dir:
        if not os.path.isdir(args.dir):
            raise UsageError(f"--dir {args.dir}: not a directory")
        files = []
        for name in sorted(os.listdir(args.dir)):
            if _is_excluded(name):
                continue
            full = os.path.join(args.dir, name)
            if os.path.isdir(full):
                raise UsageError(
                    f"{full} is a folder: a submission is a flat list of files "
                    f"(no sub-folders). Move the files it holds next to the "
                    f"others, or name the files to send with --files."
                )
            files.append(full)
        if not files:
            raise UsageError(f"--dir {args.dir}: no file to submit")
        return files
    if not args.files:
        raise UsageError("pass the files to submit: --files a b …, or --dir DIR")
    for path in args.files:
        if os.path.isdir(path):
            raise UsageError(
                f"{path} is a folder: a submission is a flat list of files "
                f"(no sub-folders). Name each file, or use --dir for a folder "
                f"that holds only files."
            )
        if not os.path.isfile(path):
            raise UsageError(f"{path}: no such file")
    names = [os.path.basename(path) for path in args.files]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise UsageError(
            f"two files are named {', '.join(duplicates)}: each file is stored "
            f"under its base name, so the names must differ"
        )
    return list(args.files)


def _warn_requirements(files: list[str], runtime: str | None) -> None:
    folders = {os.path.dirname(os.path.abspath(path)) for path in files}
    present = any(os.path.basename(path) == "requirements.txt" for path in files) or any(
        os.path.isfile(os.path.join(folder, "requirements.txt")) for folder in folders
    )
    if not present:
        return
    message = ("requirements.txt is not installed at run time: an agent runs "
               "on one of the challenge's runtime images, without network. "
               "Pick the image that has your packages with --runtime NAME "
               "(the challenge page lists them).")
    if runtime is not None:
        message = ("requirements.txt is not installed at run time: the "
                   f"submission runs on the '{runtime}' runtime image as it is.")
    _err(f"warning: {message}")
    if _in_github_actions():
        print(f"::warning title=mlarena::{message}")


def _as_submission_file(files: list[str], expected: str, workdir: str) -> list[str]:
    """A file challenge takes one file, stored under the name the challenge
    configured: the single file given is that file, whatever it is called
    on disk."""
    if len(files) != 1:
        raise UsageError(
            f"this is a file challenge: submit exactly one file (it is stored "
            f"as '{expected}'), got {len(files)}"
        )
    if os.path.basename(files[0]) == expected:
        return files
    target = os.path.join(workdir, expected)
    shutil.copyfile(files[0], target)
    _out(f"Uploading {files[0]} as {expected} (the challenge's submission file)")
    return [target]


def cmd_submit(args) -> int:
    files = _submission_files(args)
    if args.wait and args.no_deploy:
        raise UsageError("--wait needs a deploy: drop --no-deploy")

    name = args.name or default_submission_name()
    if name is not None and len(name) > SUBMISSION_NAME_MAX:
        raise UsageError(
            f"--name is {len(name)} characters; the limit is {SUBMISSION_NAME_MAX}")

    client = _connect(args.base_url, "user", args.audience)
    challenge = _resolve_challenge(client, args.challenge)
    challenge_id = challenge["id"]
    heading = f"ML Arena submission to {challenge['name']}"

    workdir = tempfile.mkdtemp(prefix="mlarena-submit-")
    try:
        if challenge["kernel_version"] == FILE_KERNEL:
            files = _as_submission_file(files, challenge["submission_filename"], workdir)
        else:
            _warn_requirements(files, args.runtime)
        timeout = None
        if args.wait:
            timeout = args.timeout if args.timeout is not None else DEFAULT_WAIT_TIMEOUT_SEC
        try:
            result = client.submit(
                challenge_id, files=files, submission_name=name,
                runtime=args.runtime, deploy=not args.no_deploy,
                wait=args.wait, timeout_sec=timeout,
                allow_same_commit=args.allow_same_commit,
            )
        except WaitTimeoutError as err:
            pending = err.body
            pending_url = (f"{_site_url(args)}/submission/view/"
                           f"{challenge_id}/{pending['id']}")
            return _fail(
                f"submission #{pending['id']} did not settle within "
                f"{timeout:g} s (status: {pending['status']}). It is still "
                f"running: {pending_url} . Raise --timeout to wait longer.",
                heading=heading)
        except MLArenaError as err:
            return _fail(str(err), heading=heading)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    submission_id = result["submission_id"]
    url = f"{_site_url(args)}/submission/view/{challenge_id}/{submission_id}"
    if result.get("already_submitted"):
        status = result["status"]["status"]
        _out(f"This commit was already submitted as #{submission_id} "
             f"(status: {status}); nothing new was created. "
             f"Pass --allow-same-commit to submit it again.")
        _out(url)
        _step_summary(
            f"### {heading}\n\nThis commit was already submitted as "
            f"[#{submission_id}]({url}) (status `{status}`).")
        _set_outputs({"submission-id": submission_id, "url": url,
                      "status": status, "already-submitted": True})
        return EXIT_OK

    outputs = {"submission-id": submission_id, "url": url,
               "already-submitted": False}
    if args.wait:
        final = result["status"]
        outcome = f"deployed, status {final['status']}"
        if final["score"] is not None:
            outcome += f", score {final['score']}"
            outputs["score"] = final["score"]
    else:
        final = client.submission(submission_id)
        outcome = "uploaded, not deployed" if args.no_deploy else "deploy queued"
    outputs["status"] = final["status"]
    _set_outputs(outputs)
    _out(f"Submission #{submission_id} ({outcome})")
    _out(url)
    _step_summary(f"### {heading}\n\nSubmission [#{submission_id}]({url}): {outcome}.")
    return EXIT_OK


# --------------------------------------------------------------------------- #
# challenge push
# --------------------------------------------------------------------------- #


def _challenge_report_lines(report: dict) -> list[str]:
    """`ChallengeBundleResponse` (`backend/app/views/creator_challenge/_schemas.py`)
    as text. The one place its keys are read."""
    lines = [report["message"]]
    files = report["env_files"]
    changed = "changed" if report["env_changed"] else "unchanged"
    lines.append(f"env files ({len(files)}, {changed}): {', '.join(files)}")
    updated = [label for key, label in (
        ("overview_updated", "overview"),
        ("agent_template_updated", "agent template"),
        ("image_updated", "image"),
    ) if report[key]]
    if updated:
        lines.append(f"updated: {', '.join(updated)}")
    config = report["config"]
    if config is not None:
        file_keys = sorted(key for key, origin in config["origin"].items()
                           if origin == "file")
        lines.append(
            f"config: {len(file_keys)} key(s) set by challenge.toml"
            + (f" ({', '.join(file_keys)})" if file_keys else ""))
    validation = report["validation"]
    if validation is not None:
        if validation["ok"]:
            lines.append("env.py: valid")
        else:
            lines.append(f"env.py: {len(validation['errors'])} error(s)")
            lines += [f"  env.py:{issue['line']}:{issue['col']}: {issue['msg']}"
                      for issue in validation["errors"]]
        lines += [f"  warning env.py:{issue['line']}:{issue['col']}: {issue['msg']}"
                  for issue in validation["warnings"]]
    if report["ignored"]:
        lines.append(f"ignored: {', '.join(report['ignored'])}")
    if report["source_revision"] is not None:
        lines.append(f"source: {_revision_label(report['source_revision'])}")
    return lines


def cmd_challenge_push(args) -> int:
    client = _connect(args.base_url, "author", args.audience)
    challenge = _resolve_challenge(client, args.challenge)
    heading = f"ML Arena challenge push to {challenge['name']}"
    try:
        report = client.push_challenge(
            challenge["id"], args.directory, subpath=args.path,
            dry_run=args.dry_run)
    except MLArenaError as err:
        return _fail(str(err), heading=heading)
    lines = _challenge_report_lines(report)
    for line in lines:
        _out(line)
    url = f"{_site_url(args)}/challenges/{challenge['slug']}"
    _out(url)
    validation = report["validation"]
    if validation is not None and not validation["ok"]:
        applied = "would be applied" if report["dry_run"] else "was applied"
        _set_outputs({"url": url, "env-changed": report["env_changed"]})
        return _fail(
            f"env.py has {len(validation['errors'])} error(s); the bundle "
            f"{applied}, and the challenge cannot start until they are fixed: "
            + "; ".join(f"line {issue['line']}: {issue['msg']}"
                        for issue in validation["errors"]),
            heading=heading)
    _step_summary(f"### {heading}\n\n" + "\n".join(f"- {line}" for line in lines)
                  + f"\n\n[{challenge['name']}]({url})")
    _set_outputs({"url": url, "env-changed": report["env_changed"]})
    return EXIT_OK


# --------------------------------------------------------------------------- #
# course push
# --------------------------------------------------------------------------- #


def _renamed(entry: dict) -> str:
    return f" (renamed from {entry['renamed_from']})" if entry["renamed_from"] else ""


# A plan names at most this many asset paths per list; the rest is a count.
PLAN_PATHS_SHOWN = 3


def _action(entry: dict) -> str:
    """`updated (body_md, title)`: the action, and the fields that changed
    when there are any (an `updated` or `adopted` item)."""
    if entry["changes"]:
        return f"{entry['action']} ({', '.join(entry['changes'])})"
    return entry["action"]


def _some_paths(paths: list[str]) -> str:
    shown = ", ".join(paths[:PLAN_PATHS_SHOWN])
    if len(paths) > PLAN_PATHS_SHOWN:
        shown += f" and {len(paths) - PLAN_PATHS_SHOWN} more"
    return shown


def _assets_line(assets: dict) -> str | None:
    """`assets: 2 written (a.png, b.png), 37 removed (…) — files no lesson
    links`. Read from the path lists only (the counts beside them are an
    expand step of the server)."""
    written, removed = assets["written_paths"], assets["removed_paths"]
    parts = []
    if written:
        parts.append(f"{len(written)} written ({_some_paths(written)})")
    if removed:
        parts.append(f"{len(removed)} removed ({_some_paths(removed)}) — files no lesson links")
    return f"assets: {', '.join(parts)}" if parts else None


def _course_plan_lines(report: dict) -> list[str]:
    """`CourseBundleResponse` (`backend/app/views/teacher/_schemas.py`) as
    text. The one place its keys are read."""
    course = report["course"]
    lines = []
    if report["dry_run"]:
        lines.append("Dry run: nothing was changed")
    lines.append(f"course {course['slug']}: {_action(course)}")
    for module in report["modules"]:
        lines.append(f"  module {module['slug']}: {_action(module)}{_renamed(module)}")
        lines += [f"    lesson {lesson['slug']}: {_action(lesson)}{_renamed(lesson)}"
                  for lesson in module["lessons"]]
        lines += [f"    lesson {slug}: orphaned (not in the manifest; unpublished, kept)"
                  for slug in module["orphaned_lessons"]]
        lines += [f"    lesson {slug}: pruned (deleted)"
                  for slug in module["pruned_lessons"]]
        lines += [f"    challenge {challenge['name']}: {_action(challenge)}"
                  for challenge in module["challenges"]]
        assets = _assets_line(module["assets"])
        if assets is not None:
            lines.append(f"    {assets}")
    for slug in report["unmanaged_modules"]:
        lines.append(f"  module {slug}: not in the manifest, untouched")
    lines += [f"  challenge {name}: in no challenges entry of the manifest, left where it is"
              for name in report["unmanaged_challenges"]]
    if report["source_revision"] is not None:
        lines.append(f"source: {_revision_label(report['source_revision'])}")
    lines += [f"warning: {warning}" for warning in report["warnings"]]
    return lines


def _annotate_warning(message: str) -> None:
    """A `::warning::` workflow command: the message shows on the run page."""
    if not _in_github_actions():
        return
    escaped = message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    print(f"::warning title=mlarena::{escaped}")


def cmd_course_push(args) -> int:
    client = _connect(args.base_url, "author", args.audience)
    heading = "ML Arena course push"
    try:
        report = client.push_course(
            args.directory, slug=args.slug, subpath=args.path,
            dry_run=args.dry_run, prune=args.prune)
    except MLArenaError as err:
        return _fail(str(err), heading=heading)
    lines = _course_plan_lines(report)
    for line in lines:
        _out(line)
    for warning in report["warnings"]:
        _annotate_warning(warning)
    url = f"{_site_url(args)}{report['url']}"
    _out(url)
    _step_summary(f"### {heading}\n\n```\n" + "\n".join(lines)
                  + f"\n```\n\n[Open the course]({url})")
    _set_outputs({"url": url, "course-slug": report["course"]["slug"]})
    return EXIT_OK


# --------------------------------------------------------------------------- #
# course export
# --------------------------------------------------------------------------- #


def _lesson_count(lessons: list) -> int:
    return sum(1 + len(lesson.get("children", [])) for lesson in lessons)


def cmd_course_export(args) -> int:
    directory = args.directory if args.directory is not None else os.path.join(".", args.slug)
    client = _connect(args.base_url, "author", args.audience)
    try:
        manifest = client.export_course_to_dir(args.slug, directory,
                                               overwrite=args.overwrite)
    except MLArenaError as err:
        return _fail(str(err), heading="ML Arena course export")
    _out(f"Exported course {manifest['course']['slug']} to {directory}")
    _out("  course.yaml")
    for module in manifest["modules"]:
        count = _lesson_count(module["lessons"])
        _out(f"  {module['slug']}/: {count} lesson{'s' if count != 1 else ''}")
    cover = manifest["course"].get("cover")
    if cover:
        _out(f"  {cover} (cover)")
    for extra in (os.path.join(".github", "workflows", "push.yml"), "README.md"):
        if os.path.isfile(os.path.join(directory, extra)):
            _out(f"  {extra.replace(os.sep, '/')}")
    _out("")
    _out(f"Next: mlarena course push --dry-run {directory}")
    _out("  every lesson should read 'unchanged', and a module written in the Studio "
         "'adopted'. Then commit the folder to a GitHub repository: each push to main "
         "updates the course. The course's owner makes the first push.")
    if os.path.isfile(os.path.join(directory, "README.md")):
        _out("  README.md lists what to fix before the first push, if anything.")
    return EXIT_OK


# --------------------------------------------------------------------------- #
# Parser + entry point
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlarena",
        description="ML Arena from the command line and from GitHub Actions. "
                    "Credential: MLARENA_API_KEY, else the GitHub Actions OIDC "
                    "token of a job with `permissions: id-token: write`.",
    )
    parser.add_argument("--version", action="version",
                        version=f"mlarena {mlarena.__version__}")
    parser.add_argument(
        "--base-url", default=os.environ.get(ENV_BASE_URL) or PROD_BASE_URL,
        help=f"the platform's URL (env {ENV_BASE_URL}; default {PROD_BASE_URL})")
    parser.add_argument(
        "--audience", default=None,
        help="GitHub Actions only: the OIDC audience, the platform's public "
             "URL (env MLARENA_OIDC_AUDIENCE; default: the base URL)")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    whoami = commands.add_parser("whoami", help="the account and credential in use")
    whoami.set_defaults(run=cmd_whoami)

    submit = commands.add_parser(
        "submit", help="create a submission, upload its files and deploy it")
    submit.add_argument(
        "--challenge", required=True, metavar="SLUG_OR_ID",
        help="the challenge: its id, or the slug of /challenges/<slug>")
    submit.add_argument("--files", nargs="+", metavar="FILE",
                        help="the files to upload, each stored under its base name")
    submit.add_argument("--dir", metavar="DIR",
                        help="upload every file of DIR (it must hold no sub-folder)")
    submit.add_argument("--name", help="the submission's name (default: the tag, "
                                       "else gh-<sha7>, in GitHub Actions)")
    submit.add_argument("--runtime", metavar="NAME",
                        help="the agent runtime image, by name (python, sklearn, torch…)")
    submit.add_argument("--no-deploy", action="store_true",
                        help="upload only; deploy later from the console")
    submit.add_argument("--wait", action="store_true",
                        help="wait until the deploy has settled; fail if it failed")
    submit.add_argument("--timeout", type=float, default=None, metavar="SEC",
                        help="with --wait: give up (exit 1) after SEC seconds "
                             f"(default {DEFAULT_WAIT_TIMEOUT_SEC})")
    submit.add_argument("--allow-same-commit", action="store_true",
                        help="submit again a commit that already made a submission")
    submit.set_defaults(run=cmd_submit)

    challenge = commands.add_parser("challenge", help="author a challenge")
    challenge_commands = challenge.add_subparsers(
        dest="challenge_command", required=True, metavar="COMMAND")
    challenge_push = challenge_commands.add_parser(
        "push", help="replace the challenge's files with a directory")
    challenge_push.add_argument(
        "directory", nargs="?", default=".", metavar="DIR",
        help="the challenge folder (env.py, challenge.toml, …) or a .zip / "
             ".tar.gz of it (default: the current directory)")
    challenge_push.add_argument(
        "--challenge", required=True, metavar="SLUG_OR_ID",
        help="the challenge to update: its id, or the slug of /challenges/<slug>")
    challenge_push.add_argument("--path", metavar="SUBPATH",
                                help="the folder inside DIR that holds the challenge")
    challenge_push.add_argument("--dry-run", action="store_true",
                                help="validate and report; change nothing")
    challenge_push.set_defaults(run=cmd_challenge_push)

    course = commands.add_parser("course", help="author a course")
    course_commands = course.add_subparsers(
        dest="course_command", required=True, metavar="COMMAND")
    course_push = course_commands.add_parser(
        "push", help="create or update a course from a directory (course.yaml)")
    course_push.add_argument(
        "directory", nargs="?", default=".", metavar="DIR",
        help="the course folder (course.yaml at its root) or a .zip / .tar.gz "
             "of it (default: the current directory)")
    course_push.add_argument(
        "--slug", help="the course's slug (default: course.slug of the "
                       "manifest; required for an archive)")
    course_push.add_argument("--path", metavar="SUBPATH",
                             help="the folder inside DIR that holds the course")
    course_push.add_argument("--dry-run", action="store_true",
                             help="print the plan; change nothing")
    course_push.add_argument("--prune", action="store_true",
                             help="delete the lessons that left the manifest, "
                                  "with their students' progress (default: "
                                  "unpublish them)")
    course_push.set_defaults(run=cmd_course_push)

    course_export = course_commands.add_parser(
        "export", help="write a course as a folder that `course push` and a "
                       "GitHub repository take as they are")
    course_export.add_argument(
        "slug", metavar="SLUG", help="the course's slug (ml-arena.com/courses/<slug>)")
    course_export.add_argument(
        "directory", nargs="?", default=None, metavar="DIR",
        help="the folder to write, new or empty (default: ./SLUG)")
    course_export.add_argument(
        "--overwrite", action="store_true",
        help="write into a folder that already holds files: the course's files "
             "replace theirs, every other file is kept (a git checkout keeps .git)")
    course_export.set_defaults(run=cmd_course_export)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.run(args)
    except UsageError as err:
        _err(f"mlarena: {err}")
        _annotate_error(str(err))
        return EXIT_USAGE
    except MLArenaError as err:
        return _fail(str(err), heading=f"mlarena {args.command}")


if __name__ == "__main__":
    sys.exit(main())
