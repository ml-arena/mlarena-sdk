"""Bundles: one archive of a challenge or course directory.

A bundle is what `push_challenge` / `push_course` upload: a .zip whose root
is the directory's content (no wrapping folder). The platform extracts and
applies it in one step, so what is on disk is what gets applied.
"""

import fnmatch
import io
import os
import zipfile

from mlarena.exceptions import MLArenaError

# Matched against every path component (a folder or a file name).
DEFAULT_EXCLUDES = (
    ".git",
    "__pycache__",
    "*.pyc",
    ".DS_Store",
    ".venv",
    "node_modules",
    ".ipynb_checkpoints",
)

ARCHIVE_SUFFIXES = (".zip", ".tar.gz", ".tgz")

# Every entry carries the same timestamp and mode: the same tree gives the
# same bytes, whatever the checkout's mtimes are.
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)
_FILE_MODE = 0o644 << 16


def _matches(name: str, patterns) -> bool:
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)


def bundle_files(directory: str, *, include=None,
                 exclude=DEFAULT_EXCLUDES) -> list[str]:
    """The files a bundle of `directory` holds, as sorted `/`-separated
    relative paths.

    `exclude` patterns (`fnmatch`) are matched against each path component:
    an excluded folder is not entered. `include`, when given, keeps only the
    files whose relative path matches one of its patterns (`"*.md"`,
    `"img/*"`). A symbolic link, to a file or to a folder, is refused: the
    platform would not follow it, and a link out of the directory would ship
    files the author did not mean to publish.
    """
    root = os.path.abspath(directory)
    if not os.path.isdir(root):
        raise MLArenaError(f"Not a directory: {directory}")
    files: list[str] = []
    for current, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not _matches(d, exclude))
        relative_dir = os.path.relpath(current, root)
        for name in dirnames + sorted(filenames):
            full = os.path.join(current, name)
            if _matches(name, exclude):
                continue
            relative = name if relative_dir == "." else os.path.join(relative_dir, name)
            relative = relative.replace(os.sep, "/")
            if os.path.islink(full):
                raise MLArenaError(
                    f"{relative} is a symbolic link; a bundle holds only regular "
                    f"files. Replace it with the file itself, or exclude it."
                )
            if name in dirnames:
                continue
            if not os.path.isfile(full):
                raise MLArenaError(f"{relative} is not a regular file")
            if include is not None and not _matches(relative, include):
                continue
            files.append(relative)
    return sorted(files)


def make_bundle(directory: str, *, include=None,
                exclude=DEFAULT_EXCLUDES) -> bytes:
    """Zip `directory` into a bundle and return its bytes.

    The archive is rooted at the directory (its files sit at the archive's
    root), holds the files `bundle_files()` lists, in that order, and is
    deterministic. An empty selection raises.
    """
    root = os.path.abspath(directory)
    files = bundle_files(root, include=include, exclude=exclude)
    if not files:
        raise MLArenaError(f"Nothing to bundle in {directory}")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for relative in files:
            info = zipfile.ZipInfo(relative, date_time=_ZIP_EPOCH)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = _FILE_MODE
            with open(os.path.join(root, *relative.split("/")), "rb") as fh:
                archive.writestr(info, fh.read())
    return buffer.getvalue()


def bundle_root(path: str, subpath: str | None) -> str:
    """The directory to bundle for `path` and its `subpath`: the sub-folder
    of `path` that holds the challenge or the course (one repository, several
    of them). `subpath` must stay inside `path`."""
    if subpath is None:
        return path
    parts = subpath.replace(os.sep, "/").split("/")
    if os.path.isabs(subpath) or ".." in parts:
        raise MLArenaError(f"subpath {subpath!r} is not a folder inside {path}")
    root = os.path.join(path, subpath)
    if not os.path.isdir(root):
        raise MLArenaError(f"{root} is not a directory (subpath {subpath!r} of {path})")
    return root


def read_bundle(path: str) -> tuple[str, bytes]:
    """`(filename, bytes)` of the archive to upload for `path`: a directory
    is bundled with `make_bundle`, an existing `.zip` / `.tar.gz` / `.tgz`
    file is sent as it is."""
    if os.path.isdir(path):
        return "bundle.zip", make_bundle(path)
    if not os.path.isfile(path):
        raise MLArenaError(f"No such directory or archive: {path}")
    name = os.path.basename(path)
    if not name.lower().endswith(ARCHIVE_SUFFIXES):
        raise MLArenaError(
            f"{path} is neither a directory nor a "
            f"{' / '.join(ARCHIVE_SUFFIXES)} archive"
        )
    with open(path, "rb") as fh:
        return name, fh.read()
