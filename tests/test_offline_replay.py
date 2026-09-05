from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from heliostune import _offline_worker
from heliostune.artifacts import strict_json_dumps, strict_json_loads
from heliostune.errors import ArtifactError, SchemaError
from heliostune.methodology import verify_bundle_v1
from heliostune.offline_replay import (
    _MAX_FRAME_BYTES,
    _MAX_RESULT_BYTES,
    _SANITIZED_ENV,
    ANALYZER_SOURCE_ROLE_V1,
    REFERENCE_ANALYZER_ID_V1,
    AnalyzerArtifactBindingV1,
    AnalyzerImplementationV1,
    AnalyzerManifestV1,
    OfflineReplayResult,
    _decode_worker_request,
    _decode_worker_result,
    _encode_worker_request,
    _encode_worker_result,
    _run_worker,
    _source_aggregate_sha256,
    _worker_argv,
    build_replay_verification_record_v1,
    load_analyzer_manifest_v1,
    replay_bundle_v1,
    write_offline_replay_record_v1,
)
from heliostune.verification import (
    VERIFICATION_CONTROL_NAMES_V1,
    build_verification_record_v1,
    load_verification_record_v1,
)

FIXTURE = Path(__file__).parent / "fixtures/offline-replay-v1"


def _copy_fixture(tmp_path: Path) -> Path:
    destination = tmp_path / "bundle"
    shutil.copytree(FIXTURE, destination)
    return destination / "bundle.json"


def _manifest_dict() -> dict[str, Any]:
    value = strict_json_loads((FIXTURE / "artifacts/analyzer.json").read_text())
    assert isinstance(value, dict)
    return value


def test_manifest_is_exact_canonical_and_registry_bound() -> None:
    path = FIXTURE / "artifacts/analyzer.json"
    manifest = load_analyzer_manifest_v1(path)
    assert path.read_bytes() == strict_json_dumps(manifest.to_dict()).encode()
    assert manifest.analyzer_id == REFERENCE_ANALYZER_ID_V1
    assert tuple(x.role for x in manifest.implementation.sources) == (ANALYZER_SOURCE_ROLE_V1,)
    assert tuple((x.role, x.media_type) for x in manifest.inputs) == (
        ("analysis_input", "application/json"),
    )
    assert tuple((x.role, x.media_type) for x in manifest.outputs) == (
        ("analysis_summary", "application/json"),
    )


@pytest.mark.parametrize("container", ["root", "implementation", "input", "output", "source"])
def test_manifest_rejects_unknown_fields(container: str) -> None:
    raw = _manifest_dict()
    target = (
        raw
        if container == "root"
        else raw["implementation"]
        if container == "implementation"
        else raw["inputs"][0]
        if container == "input"
        else raw["outputs"][0]
        if container == "output"
        else raw["implementation"]["sources"][0]
    )
    target["unknown"] = True
    with pytest.raises(SchemaError, match="unknown fields"):
        AnalyzerManifestV1.from_dict(raw)


@pytest.mark.parametrize(
    "field,value",
    [("schema", "bad"), ("runner_api", "bad"), ("representation", "semantic"), ("analyzer_id", 1)],
)
def test_manifest_rejects_wrong_literals_and_types(field: str, value: object) -> None:
    raw = _manifest_dict()
    raw[field] = value
    with pytest.raises(SchemaError):
        AnalyzerManifestV1.from_dict(raw)


def test_manifest_rejects_non_utf8_scalar_values() -> None:
    raw = _manifest_dict()
    raw["implementation"]["sources"][0]["role"] = "\ud800"

    with pytest.raises(SchemaError, match="valid Unicode"):
        AnalyzerManifestV1.from_dict(raw)


def test_manifest_loader_rejects_oversized_sparse_file_before_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "analyzer.json"
    with path.open("wb") as sparse_file:
        sparse_file.truncate(_MAX_FRAME_BYTES + 1)
    identity = path.stat().st_ino
    original_read = os.read

    def guarded_read(descriptor: int, size: int) -> bytes:
        if os.fstat(descriptor).st_ino == identity:
            raise AssertionError("oversized analyzer manifest was read")
        return original_read(descriptor, size)

    monkeypatch.setattr(os, "read", guarded_read)
    with pytest.raises(ArtifactError, match="exceeds the byte limit"):
        load_analyzer_manifest_v1(path)


def test_manifest_requires_nonempty_unique_disjoint_roles() -> None:
    raw = _manifest_dict()
    raw["outputs"] = []
    with pytest.raises(SchemaError, match="nonempty"):
        AnalyzerManifestV1.from_dict(raw)
    raw = _manifest_dict()
    raw["outputs"][0]["role"] = raw["inputs"][0]["role"]
    with pytest.raises(SchemaError, match="disjoint"):
        AnalyzerManifestV1.from_dict(raw)
    raw = _manifest_dict()
    raw["implementation"]["sources"].append(dict(raw["implementation"]["sources"][0]))
    with pytest.raises(SchemaError, match="unique"):
        AnalyzerManifestV1.from_dict(raw)


def test_implementation_digest_is_framed_and_checked() -> None:
    source = AnalyzerArtifactBindingV1(
        "analyzer_source", "text/x-python", 1, hashlib.sha256(b"x").hexdigest()
    )
    digest = _source_aggregate_sha256((source,))
    assert AnalyzerImplementationV1(digest, (source,)).source_sha256 == digest
    with pytest.raises(SchemaError, match="does not match"):
        AnalyzerImplementationV1("0" * 64, (source,))


def test_worker_frames_are_canonical_bounded_and_source_bound() -> None:
    manifest = load_analyzer_manifest_v1(FIXTURE / "artifacts/analyzer.json")
    verified = verify_bundle_v1(FIXTURE / "bundle.json")
    verifier = build_verification_record_v1(verified).verifier
    inputs = (("analysis_input", b"abc"),)
    request = _encode_worker_request(
        REFERENCE_ANALYZER_ID_V1,
        manifest.implementation,
        verifier,
        inputs,
    )
    assert request.endswith(b"\n")
    assert _decode_worker_request(request) == (
        REFERENCE_ANALYZER_ID_V1,
        manifest.implementation,
        verifier,
        inputs,
    )
    result = _encode_worker_result((("analysis_summary", b"xyz"),))
    assert _decode_worker_result(result) == (("analysis_summary", b"xyz"),)
    with pytest.raises((SchemaError, ArtifactError)):
        _decode_worker_result(result + b"\n")
    with pytest.raises(ArtifactError, match="byte limit"):
        _decode_worker_result(b"x" * (_MAX_RESULT_BYTES + 1))


def test_worker_argv_and_environment_are_exact() -> None:
    argv = _worker_argv()
    assert argv[:3] == ("/usr/bin/setpriv", "--no-new-privs", "/usr/bin/unshare")
    assert argv[3:11] == (
        "--user",
        "--map-root-user",
        "--net",
        "--mount",
        "--pid",
        "--fork",
        "--kill-child=SIGKILL",
        "--mount-proc",
    )
    assert argv[-5:] == ("-B", "-P", "-s", "-m", "heliostune._offline_worker")
    assert _SANITIZED_ENV == {
        "HOME": "/",
        "LC_ALL": "C",
        "LANG": "C",
        "TZ": "UTC",
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
    }


@pytest.mark.parametrize(
    "event",
    [
        "open",
        "import",
        "socket.connect",
        "socket.getaddrinfo",
        "subprocess.Popen",
        "os.fork",
        "os.exec",
        "os.system",
        "ctypes.dlopen",
        "sys.addaudithook",
    ],
)
def test_worker_audit_hook_denies_and_latches(event: str) -> None:
    _offline_worker._audit_denied = False
    with pytest.raises(PermissionError):
        _offline_worker._audit_hook(event, ())
    assert _offline_worker._audit_denied is True


def test_worker_establishes_empty_kernel_read_only_chroot(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace.chmod(0o555)
    code = (
        "import os, socket\n"
        "from heliostune import _offline_worker as worker\n"
        "worker._require_namespace_context()\n"
        "worker._enter_read_only_chroot()\n"
        "assert os.statvfs('/').f_flag & os.ST_RDONLY\n"
        "assert os.listdir('/') == []\n"
        "connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
        "connection.settimeout(0.5)\n"
        "try:\n"
        "    connection.connect(('192.0.2.1', 9))\n"
        "except OSError:\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError('network namespace reached TEST-NET')\n"
        "finally:\n"
        "    connection.close()\n"
        "print('read-only-offline')\n"
    )
    completed = subprocess.run(
        (*_worker_argv()[:-2], "-c", code),
        cwd=workspace,
        env=dict(_SANITIZED_ENV),
        capture_output=True,
        check=False,
        timeout=10,
    )

    assert completed.returncode == 0
    assert completed.stdout == b"read-only-offline\n"
    assert completed.stderr == b""
    assert list(workspace.iterdir()) == []


@pytest.mark.parametrize("mismatch", ["implementation", "verifier"])
def test_worker_rejects_parent_child_source_identity_mismatch(
    tmp_path: Path,
    mismatch: str,
) -> None:
    manifest = load_analyzer_manifest_v1(FIXTURE / "artifacts/analyzer.json")
    verified = verify_bundle_v1(FIXTURE / "bundle.json")
    verifier = build_verification_record_v1(verified).verifier
    implementation = manifest.implementation
    if mismatch == "implementation":
        source = AnalyzerArtifactBindingV1(
            ANALYZER_SOURCE_ROLE_V1,
            "text/x-python",
            1,
            hashlib.sha256(b"x").hexdigest(),
        )
        implementation = AnalyzerImplementationV1(
            _source_aggregate_sha256((source,)),
            (source,),
        )
    else:
        verifier = replace(verifier, version=verifier.version + ".mismatch")
    request = _encode_worker_request(
        REFERENCE_ANALYZER_ID_V1,
        implementation,
        verifier,
        (
            (
                "analysis_input",
                (FIXTURE / "artifacts/analysis_input.json").read_bytes(),
            ),
        ),
    )
    workspace = tmp_path / "mismatch-workspace"
    workspace.mkdir()
    workspace.chmod(0o555)

    with pytest.raises(ArtifactError, match="exited with status 1"):
        _run_worker(request, workspace, timeout_s=10)


def test_real_replay_is_exact_success_without_promoting_other_claims() -> None:
    result = replay_bundle_v1(FIXTURE / "bundle.json", timeout_s=10)
    assert result.first_run_outputs == result.second_run_outputs == result.manifest.outputs
    assert (
        result.record.controls.analyzer_replay
        == result.record.controls.offline_reproduction
        == "checked"
    )
    assert (
        result.verified.limitations.analyzer_replay
        == result.verified.limitations.offline_reproduction
        == "not_checked"
    )
    assert not result.record.claim_eligible and not result.record.publication_eligible


def test_replay_uses_two_distinct_empty_read_only_workspaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _copy_fixture(tmp_path)
    seen: list[Path] = []
    output = (root.parent / "artifacts/analysis_summary.json").read_bytes()

    def fake(request: bytes, workspace: Path, *, timeout_s: float) -> tuple[tuple[str, bytes], ...]:
        assert (
            request
            and timeout_s == 2
            and list(workspace.iterdir()) == []
            and os.stat(workspace).st_mode & 0o777 == 0o555
        )
        seen.append(workspace)
        return (("analysis_summary", output),)

    monkeypatch.setattr("heliostune.offline_replay._run_worker", fake)
    replay_bundle_v1(root, timeout_s=2)
    assert len(seen) == 2 and seen[0] != seen[1]


def test_nondeterminism_fails_without_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _copy_fixture(tmp_path)
    committed = (root.parent / "artifacts/analysis_summary.json").read_bytes()
    calls = 0

    def fake(*args: object, **kwargs: object) -> tuple[tuple[str, bytes], ...]:
        nonlocal calls
        calls += 1
        return (("analysis_summary", committed if calls == 1 else b"different"),)

    monkeypatch.setattr("heliostune.offline_replay._run_worker", fake)
    with pytest.raises(ArtifactError, match="nondeterministic"):
        replay_bundle_v1(root)


def test_source_mismatch_fails_before_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _copy_fixture(tmp_path)
    bundle = strict_json_loads(root.read_text())
    assert isinstance(bundle, dict)
    source_path = root.parent / "artifacts/analyzer_source.py"
    payload = source_path.read_bytes() + b"\n"
    source_path.write_bytes(payload)
    artifact = next(x for x in bundle["artifacts"] if x["role"] == "analyzer_source")
    artifact.update(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    root.write_text(strict_json_dumps(bundle))
    monkeypatch.setattr(
        "heliostune.offline_replay._run_worker", lambda *a, **k: pytest.fail("worker spawned")
    )
    with pytest.raises(ArtifactError, match="binding"):
        replay_bundle_v1(root)


def test_worker_timeout_kills_process_group_and_reaps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import heliostune.offline_replay as replay

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace.chmod(0o555)
    killed: list[tuple[int, int]] = []

    class FakeProcess:
        pid = 321
        returncode = -9

        def __init__(self, *args: object, **kwargs: object) -> None:
            self.calls = 0

        def communicate(self, **kwargs: object) -> None:
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired("worker", 0.1)

        def poll(self) -> int:
            return -9

        def wait(self) -> int:
            return -9

    monkeypatch.setattr(replay.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(replay.os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    with pytest.raises(ArtifactError, match="timed out"):
        replay._run_worker(b"request", workspace, timeout_s=0.1)
    assert killed == [(321, 9)]


def test_worker_communication_failure_kills_process_group_and_reaps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import heliostune.offline_replay as replay

    workspace = tmp_path / "communication-workspace"
    workspace.mkdir()
    workspace.chmod(0o555)
    killed: list[tuple[int, int]] = []
    communications: list[dict[str, object]] = []

    class FakeProcess:
        pid = 654
        returncode = -9

        def __init__(self, *args: object, **kwargs: object) -> None:
            assert kwargs["close_fds"] is True
            assert kwargs["start_new_session"] is True

        def communicate(self, **kwargs: object) -> None:
            communications.append(kwargs)
            if len(communications) == 1:
                raise OSError("pipe failure")

        def poll(self) -> int:
            return -9

    monkeypatch.setattr(replay.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(replay.os, "killpg", lambda pid, sig: killed.append((pid, sig)))

    with pytest.raises(ArtifactError, match="cannot communicate"):
        replay._run_worker(b"request", workspace, timeout_s=1)
    assert killed == [(654, 9)]
    assert len(communications) == 2


def test_only_runner_can_mint_result() -> None:
    with pytest.raises(SchemaError, match="completed replay"):
        OfflineReplayResult(None, None, (), (), None, object())  # type: ignore[arg-type]


def test_verified_bundle_alone_cannot_build_upgraded_record() -> None:
    verified = replay_bundle_v1(FIXTURE / "bundle.json", timeout_s=10).verified
    with pytest.raises(ArtifactError, match="completed replay"):
        build_replay_verification_record_v1(verified)  # type: ignore[arg-type]


def test_replay_writer_publishes_only_exact_upgrade(tmp_path: Path) -> None:
    root = _copy_fixture(tmp_path)
    result = replay_bundle_v1(root, timeout_s=10)
    destination = tmp_path / "record.json"
    write_offline_replay_record_v1(destination, result)
    assert load_verification_record_v1(destination) == result.record
    with pytest.raises(ArtifactError):
        write_offline_replay_record_v1(tmp_path / "second.json", object())  # type: ignore[arg-type]
    for name in VERIFICATION_CONTROL_NAMES_V1:
        expected = (
            "checked"
            if name in {"analyzer_replay", "offline_reproduction"}
            else getattr(result.verified.limitations, name)
        )
        assert getattr(result.record.controls, name) == expected
