"""Fixed subprocess worker for :mod:`heliostune.offline_replay`."""

from __future__ import annotations

import ctypes
import errno
import os
import resource
import stat
import sys
from typing import NoReturn

from heliostune.offline_replay import (
    _MAX_FRAME_BYTES,
    _MAX_RESULT_BYTES,
    _decode_worker_request,
    _encode_worker_result,
    _registered_analyzer,
)
from heliostune.verification import _capture_verifier_identity_v1

_MAX_CPU_SECONDS = 10
_MAX_ADDRESS_SPACE = 512 * 1024 * 1024
_MAX_OPEN_FILES = 16
_MAX_PROCESSES = 1
_MS_RDONLY = 1
_MS_NOSUID = 2
_MS_NODEV = 4
_MS_NOEXEC = 8
_MS_REMOUNT = 32
_MS_BIND = 4096
_TMPFS_MOUNT_FLAGS = _MS_NOSUID | _MS_NODEV | _MS_NOEXEC
_TMPFS_REMOUNT_FLAGS = _TMPFS_MOUNT_FLAGS | _MS_RDONLY | _MS_REMOUNT | _MS_BIND
_TMPFS_OPTIONS = b"size=4096,nr_inodes=16,mode=0555"
_LIBC = ctypes.CDLL(None, use_errno=True)
_MOUNT = _LIBC.mount
_MOUNT.argtypes = [
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_ulong,
    ctypes.c_char_p,
]
_MOUNT.restype = ctypes.c_int
_DENIED_EXACT = frozenset(
    {
        "open",
        "import",
        "os.chdir",
        "os.chmod",
        "os.chown",
        "os.fchdir",
        "subprocess.Popen",
        "os.system",
        "os.fork",
        "os.forkpty",
        "os.posix_spawn",
        "os.link",
        "os.listdir",
        "os.mkdir",
        "os.remove",
        "os.rename",
        "os.rmdir",
        "os.scandir",
        "os.symlink",
        "os.truncate",
        "os.utime",
        "pty.spawn",
        "sys.addaudithook",
    }
)
_DENIED_PREFIXES = ("socket.", "ctypes.", "os.exec", "os.spawn")
_audit_denied = False


def _fail() -> NoReturn:
    raise SystemExit(1)


def _require_namespace_context() -> None:
    if os.getpid() != 1 or os.geteuid() != 0 or os.getegid() != 0:
        _fail()
    try:
        with open("/proc/self/status", encoding="ascii") as status_file:
            status = status_file.read().splitlines()
        with open("/proc/self/uid_map", encoding="ascii") as uid_map_file:
            uid_map = uid_map_file.read().split()
        with open("/proc/self/gid_map", encoding="ascii") as gid_map_file:
            gid_map = gid_map_file.read().split()
    except OSError:
        _fail()
    if "NoNewPrivs:\t1" not in status:
        _fail()
    for identity_map in (uid_map, gid_map):
        if len(identity_map) != 3 or identity_map[0] != "0" or identity_map[2] != "1":
            _fail()


def _mount_call(
    source: bytes | None,
    target: bytes,
    filesystem_type: bytes | None,
    flags: int,
    data: bytes | None,
) -> None:
    ctypes.set_errno(0)
    if _MOUNT(source, target, filesystem_type, flags, data) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))


def _enter_read_only_chroot() -> None:
    workspace_path = os.getcwd()
    if not os.path.isabs(workspace_path):
        _fail()
    workspace = os.stat(workspace_path, follow_symlinks=False)
    if not os.path.isdir(workspace_path) or workspace.st_mode & 0o222 or os.listdir(workspace_path):
        _fail()

    encoded_path = os.fsencode(workspace_path)
    _mount_call(
        b"tmpfs",
        encoded_path,
        b"tmpfs",
        _TMPFS_MOUNT_FLAGS,
        _TMPFS_OPTIONS,
    )
    _mount_call(
        None,
        encoded_path,
        None,
        _TMPFS_REMOUNT_FLAGS,
        None,
    )
    os.chdir(workspace_path)
    mounted = os.stat(".", follow_symlinks=False)
    if (
        not os.path.ismount(".")
        or stat.S_IMODE(mounted.st_mode) != 0o555
        or os.listdir(".")
        or not os.statvfs(".").f_flag & os.ST_RDONLY
    ):
        _fail()

    os.chroot(".")
    os.chdir("/")
    try:
        descriptor = os.open(
            "/.heliostune-write-probe",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except OSError as exc:
        if exc.errno != errno.EROFS:
            _fail()
    else:
        os.close(descriptor)
        _fail()


def _read_request() -> bytes:
    payload = sys.stdin.buffer.read(_MAX_FRAME_BYTES + 1)
    if len(payload) > _MAX_FRAME_BYTES:
        _fail()
    return payload


def _set_limit(kind: int, maximum: int) -> None:
    _, hard = resource.getrlimit(kind)
    value = maximum if hard == resource.RLIM_INFINITY else min(maximum, hard)
    resource.setrlimit(kind, (value, value))


def _install_limits() -> None:
    _set_limit(resource.RLIMIT_CPU, _MAX_CPU_SECONDS)
    _set_limit(resource.RLIMIT_AS, _MAX_ADDRESS_SPACE)
    _set_limit(resource.RLIMIT_FSIZE, _MAX_RESULT_BYTES)
    _set_limit(resource.RLIMIT_NOFILE, _MAX_OPEN_FILES)
    if hasattr(resource, "RLIMIT_NPROC"):
        _set_limit(resource.RLIMIT_NPROC, _MAX_PROCESSES)
    if hasattr(resource, "RLIMIT_CORE"):
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def _close_nonstdio_fds() -> None:
    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    maximum = 1 << 20 if hard == resource.RLIM_INFINITY else hard
    os.closerange(3, maximum)


def _audit_hook(event: str, args: tuple[object, ...]) -> None:
    del args
    global _audit_denied
    if event in _DENIED_EXACT or event.startswith(_DENIED_PREFIXES):
        _audit_denied = True
        raise PermissionError(f"offline replay denied audited action {event!r}")


def _validate_inputs(
    inputs: tuple[tuple[str, bytes], ...], expected: tuple[tuple[str, str], ...]
) -> None:
    if len(inputs) != len(expected):
        _fail()
    if tuple(role for role, _ in inputs) != tuple(role for role, _ in expected):
        _fail()
    if sum(len(payload) for _, payload in inputs) > _MAX_FRAME_BYTES:
        _fail()


def _validate_outputs(
    outputs: object, expected: tuple[tuple[str, str], ...]
) -> tuple[tuple[str, bytes], ...]:
    if type(outputs) is not tuple or len(outputs) != len(expected):
        _fail()
    checked: list[tuple[str, bytes]] = []
    size = 0
    for item, (expected_role, _) in zip(outputs, expected, strict=True):
        if type(item) is not tuple or len(item) != 2:
            _fail()
        role, payload = item
        if type(role) is not str or role != expected_role or type(payload) is not bytes:
            _fail()
        size += len(payload)
        if size > _MAX_RESULT_BYTES:
            _fail()
        checked.append((role, payload))
    return tuple(checked)


def _main() -> int:
    try:
        request = _read_request()
        analyzer_id, expected_implementation, expected_verifier, inputs = _decode_worker_request(
            request
        )
        _require_namespace_context()
        entry = _registered_analyzer(analyzer_id)
        if entry.implementation != expected_implementation:
            _fail()
        if _capture_verifier_identity_v1() != expected_verifier:
            _fail()
        _validate_inputs(inputs, entry.input_spec)
        _enter_read_only_chroot()
        _close_nonstdio_fds()
        _install_limits()
        sys.addaudithook(_audit_hook)
        outputs = entry.callable(inputs)
        if _audit_denied:
            _fail()
        checked = _validate_outputs(outputs, entry.output_spec)
        if _audit_denied:
            _fail()
        result = _encode_worker_result(checked)
        if _audit_denied:
            _fail()
        sys.stdout.buffer.write(result)
        sys.stdout.buffer.flush()
        return 0
    except BaseException:
        return 1


if __name__ == "__main__":
    raise SystemExit(_main())
