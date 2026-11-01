"""Hash-verified immutable commits and explicit task locks; no experiment on import."""
from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import tempfile
import time
import uuid
import zipfile

import numpy as np
from .simulation_config import canonical, digest


class StorageError(RuntimeError):
    pass


class CompatibilityError(StorageError):
    pass


class TaskLocked(StorageError):
    pass


def _sync_directory(directory):
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def read_json(path):
    try:
        envelope = json.loads(Path(path).read_text())
        if (not isinstance(envelope, dict) or set(envelope) != {'payload', 'sha256'}
                or digest(envelope['payload']) != envelope['sha256']):
            raise ValueError('content hash/schema mismatch')
    except (ValueError, TypeError) as exc:
        raise CompatibilityError(f'Content hash/schema mismatch: {path}') from exc
    return envelope['payload'], envelope['sha256']


def commit_bytes(path, content):
    """Atomic no-replace hard-link commit, with fsync; requires local hard links.

    Existing files are compared exactly; they are never truncated or replaced.
    No silent fallback to a non-atomic file write is used.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise CompatibilityError(f'Refusing to overwrite immutable file: {path}')
        return hashlib.sha256(content).hexdigest()
    fd, temporary = tempfile.mkstemp(prefix='.pending-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(content); handle.flush(); os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise CompatibilityError(f'Concurrent incompatible commit: {path}')
        _sync_directory(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return hashlib.sha256(content).hexdigest()


def commit_json(path, payload):
    sha = digest(payload)
    commit_bytes(path, canonical({'payload': payload, 'sha256': sha})+b'\n')
    return sha


def publish_pointer(path, target):
    """Replace only a convenience symlink, preserving every versioned export."""
    path, target = Path(path), Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not path.is_symlink():
        raise CompatibilityError(f'Refusing to replace a regular file with a latest pointer: {path}')
    temporary = path.parent/('.pointer-'+uuid.uuid4().hex)
    try:
        temporary.symlink_to(os.path.relpath(target, path.parent))
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def task_lock(path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    record = {'host': socket.gethostname(), 'pid': os.getpid(), 'token': token,
              'started_unix': time.time()}
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise TaskLocked(f'Task already locked: {path}; recover only after its worker has stopped') from exc
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(canonical(record)); handle.flush(); os.fsync(handle.fileno())
        yield
    finally:
        # Never release a lock that has been replaced by another worker.
        if path.exists() and json.loads(path.read_text()).get('token') == token:
            path.unlink()


def recover_locks(directory):
    """Explicit operator action only. Refuse known-live local owners.

    For remote owners, the operator must establish that all workers stopped.
    Committed datasets/results are never modified by recovery.
    """
    for path in sorted(Path(directory).rglob('*.lock')):
        record = json.loads(path.read_text())
        if record['host'] == socket.gethostname():
            try:
                os.kill(record['pid'], 0)
            except ProcessLookupError:
                pass
            except PermissionError as exc:
                raise TaskLocked(f'Cannot establish owner termination for {path}') from exc
            else:
                raise TaskLocked(f'Lock owner is still alive: {path}')
        path.unlink()


def save_draws(path, names, draws):
    """Deterministic compressed NPZ without pickle; no draws are saved by default."""
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, array in [('names', np.asarray(names, dtype=str)), ('draws', np.asarray(draws))]:
            item = io.BytesIO()
            np.lib.format.write_array(item, array, allow_pickle=False)
            info = zipfile.ZipInfo(name+'.npy', date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, item.getvalue())
    content = stream.getvalue()
    sha = hashlib.sha256(content).hexdigest()
    target = Path(path)/f'{sha}.npz'
    commit_bytes(target, content)
    return target, sha
