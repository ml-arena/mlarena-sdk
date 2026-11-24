"""Bundles: one archive of a challenge or course directory.

A bundle is what `push_challenge` / `push_course` upload: a .zip whose root
is the directory's content (no wrapping folder). The platform extracts and
applies it in one step, so what is on disk is what gets applied. A course
export (`export_course_to_dir`) is the same shape the other way:
`extract_bundle` writes it into a folder.
"""

import fnmatch
import io
import os
import shutil
import tempfile
import zipfile
import zlib

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


_SYMLINK_TYPE = 0o120000
_FILE_TYPE_MASK = 0o170000

# What a course push accepts (`BUNDLE_LIMITS` of
# backend/app/views/teacher/course_bundle.py): an export larger than this
# could not be pushed back, so it is refused before anything is written.
MAX_EXTRACT_FILES = 5000
MAX_EXTRACT_BYTES = 300 * 1024 * 1024
_CHUNK = 1 << 20


def check_extract_target(directory: str, overwrite: bool) -> None:
    """Refuse a target that is a file, or a directory that already holds
    something, unless `overwrite`."""
    if os.path.islink(directory):
        raise MLArenaError(f"{directory} is a symbolic link; name a real directory")
    if not os.path.exists(directory):
        return
    if not os.path.isdir(directory):
        raise MLArenaError(f"{directory} exists and is not a directory")
    if os.listdir(directory) and not overwrite:
        raise MLArenaError(
            f"{directory} is not empty. Export into a new or empty folder, or "
            f"pass overwrite=True (CLI: --overwrite) to write the course's files "
            f"over the ones there (other files are kept)."
        )


def _entry_parts(name: str) -> list[str]:
    """The path components of an archive entry; refuses a name that could
    land outside the target directory or that a file system reads otherwise
    (a `:` is a drive on Windows and a stream on NTFS)."""
    stripped = name[:-1] if name.endswith("/") else name
    parts = stripped.split("/")
    if (not stripped or name.startswith("/") or "\\" in name
            or any(part in ("", ".", "..") or ":" in part for part in parts)):
        raise MLArenaError(
            f"The archive holds an entry named {name!r}, which is not a path "
            f"inside the folder; nothing was written."
        )
    return parts


def _check_existing(root: str, files: dict, folders: set) -> None:
    """With `overwrite`: no path of the archive goes through a symbolic link
    of `root`, puts a folder where a file is, or a file where a folder is."""
    def walk(relative: str, is_file: bool) -> None:
        current = root
        parts = relative.split("/")
        for depth, part in enumerate(parts):
            current = os.path.join(current, part)
            if os.path.islink(current):
                raise MLArenaError(
                    f"{current} is a symbolic link; the export does not write "
                    f"through one. Remove it, then export again; nothing was written."
                )
            last = depth == len(parts) - 1
            if (not last or not is_file) and os.path.exists(current) \
                    and not os.path.isdir(current):
                raise MLArenaError(
                    f"{current} is a file, and the course has a folder there; "
                    f"nothing was written."
                )
            if last and is_file and os.path.isdir(current):
                raise MLArenaError(
                    f"{current} is a folder, and the course has a file there; "
                    f"nothing was written."
                )

    for relative in sorted(folders):
        walk(relative, is_file=False)
    for relative in sorted(files):
        walk(relative, is_file=True)


def _copy_entry(zf: zipfile.ZipFile, info: zipfile.ZipInfo, out_path: str,
                budget: list[int]) -> None:
    """Stream one entry to `out_path`, refusing more bytes than its header
    declares or than the archive's remaining budget (`budget[0]`)."""
    written = 0
    try:
        with zf.open(info) as source, open(out_path, "wb") as fh:
            while True:
                chunk = source.read(_CHUNK)
                if not chunk:
                    break
                written += len(chunk)
                budget[0] -= len(chunk)
                if written > info.file_size or budget[0] < 0:
                    raise MLArenaError(
                        f"The archive's entry {info.filename} holds more than it "
                        f"declares; nothing was written."
                    )
                fh.write(chunk)
    except (zipfile.BadZipFile, zlib.error, EOFError) as exc:
        raise MLArenaError(
            f"The archive's entry {info.filename} is damaged ({exc}); nothing was written."
        ) from exc


def extract_bundle(archive: bytes, directory: str, *,
                   overwrite: bool = False) -> list[str]:
    """Write a bundle (a .zip whose root is the directory's content) into
    `directory`; returns the written files as sorted `/`-separated paths.

    `directory` is created when missing. One that exists must be empty,
    unless `overwrite=True`: then each file of the archive replaces the file
    at its path and every other file is kept (a git checkout keeps its
    `.git`).

    Every entry is checked before anything is written, and the whole archive
    is refused when one is: an absolute path, a drive, a backslash, a `:`,
    a `.` / `..` / empty component, a symbolic link, two entries naming one
    path (also when they differ only by case, which collide on macOS and
    Windows), a file that is also a folder of another entry, more than
    `MAX_EXTRACT_FILES` files or `MAX_EXTRACT_BYTES` in all (the course
    push's limits), or (with `overwrite`) a path that goes through a
    symbolic link, a file where the archive has a folder or a folder where
    it has a file. The entries are streamed into a staging folder inside
    `directory`, each held to its declared size, and moved into place only
    once all of them are written: a refusal writes nothing.
    """
    check_extract_target(directory, overwrite)
    try:
        zf = zipfile.ZipFile(io.BytesIO(archive))
    except zipfile.BadZipFile as exc:
        raise MLArenaError(f"The server's answer is not a .zip archive: {exc}") from exc
    with zf:
        files: dict[str, zipfile.ZipInfo] = {}
        folders: set[str] = set()
        for info in zf.infolist():
            parts = _entry_parts(info.filename)
            if (info.external_attr >> 16) & _FILE_TYPE_MASK == _SYMLINK_TYPE:
                raise MLArenaError(
                    f"The archive holds a symbolic link ({info.filename}); "
                    f"nothing was written."
                )
            for depth in range(1, len(parts)):
                folders.add("/".join(parts[:depth]))
            relative = "/".join(parts)
            if info.is_dir():
                folders.add(relative)
                continue
            if relative in files:
                raise MLArenaError(
                    f"The archive names {relative} twice; nothing was written.")
            files[relative] = info
        clash = sorted(set(files) & folders)
        if clash:
            raise MLArenaError(
                f"The archive holds {clash[0]} both as a file and as a folder; "
                f"nothing was written."
            )
        seen: dict[str, str] = {}
        for relative in sorted(set(files) | folders):
            other = seen.setdefault(relative.casefold(), relative)
            if other != relative:
                raise MLArenaError(
                    f"The archive holds {other} and {relative}, which differ only "
                    f"by case and are one file on macOS and Windows; nothing was written."
                )
        if len(files) > MAX_EXTRACT_FILES:
            raise MLArenaError(
                f"The archive holds {len(files)} files, more than the "
                f"{MAX_EXTRACT_FILES} a course push takes; nothing was written."
            )
        declared = sum(info.file_size for info in files.values())
        if declared > MAX_EXTRACT_BYTES:
            raise MLArenaError(
                f"The archive holds {declared} bytes, more than the "
                f"{MAX_EXTRACT_BYTES} a course push takes; nothing was written."
            )

        root = os.path.abspath(directory)
        created = not os.path.exists(root)
        if not created:
            _check_existing(root, files, folders)
        os.makedirs(root, exist_ok=True)
        staging = tempfile.mkdtemp(prefix=".mlarena-export-", dir=root)
        try:
            budget = [MAX_EXTRACT_BYTES]
            for relative in sorted(files):
                out_path = os.path.join(staging, *relative.split("/"))
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                _copy_entry(zf, files[relative], out_path, budget)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            if created:
                shutil.rmtree(root, ignore_errors=True)
            raise
        # Everything is on disk: move it into place.
        for relative in sorted(folders):
            os.makedirs(os.path.join(root, *relative.split("/")), exist_ok=True)
        for relative in sorted(files):
            os.replace(os.path.join(staging, *relative.split("/")),
                       os.path.join(root, *relative.split("/")))
        shutil.rmtree(staging)
    return sorted(files)
