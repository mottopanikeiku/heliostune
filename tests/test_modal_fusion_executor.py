from __future__ import annotations

import ast
import base64
import csv
import hashlib
import importlib.util
import io
import json
import os
import stat
import sys
import types
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from heliostune.errors import ArtifactError, SchemaError
from heliostune.fusion_execution_registry import fusion_execution_spec
from heliostune.remote_execution import (
    CLIENT_TIMEOUT_SECONDS,
    JournalState,
    RemoteJournal,
    RemoteJournalRecord,
    canonical_json_bytes,
    encode_remote_request,
    remote_artifact_paths,
    sha256_bytes,
)
from heliostune.scope import verify_plugin, verify_suite
from heliostune.wheel_verifier import source_digest, source_entries

ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / "modal_fusion_executor.py"
MLP_SUITE = ROOT / "benchmarks/suites/gated-mlp-epilogue-v1.json"
RMSNORM_SUITE = ROOT / "benchmarks/suites/residual-rmsnorm-v1.json"
NATIVE_SUITE = ROOT / "benchmarks/suites/residual-rmsnorm-triton-v1.json"
REFERENCE_PLUGIN = ROOT / "benchmarks/plugins/fusion-reference-plugin-v1.json"
NATIVE_PLUGIN = ROOT / "benchmarks/plugins/fusion-triton-rmsnorm-plugin-v1.json"
SUITE = MLP_SUITE
PLUGIN = REFERENCE_PLUGIN
HEAD = "a" * 40


class _Image:
    @classmethod
    def debian_slim(cls, **_: object) -> _Image:
        return cls()

    def pip_install(self, *_: object) -> _Image:
        return self

    def add_local_file(self, *_: object, **__: object) -> _Image:
        return self

    def run_commands(self, *_: object) -> _Image:
        return self

    def env(self, *_: object) -> _Image:
        return self


class _App:
    function_options: list[dict[str, object]] = []

    def __init__(self, _: str) -> None:
        pass

    def function(self, **options: object) -> Any:
        self.function_options.append(options)
        return lambda function: function

    def local_entrypoint(self, **_: object) -> Any:
        return lambda function: function


def _record_hash(payload: bytes) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=").decode()
    return f"sha256={encoded}"


def _wheel(
    wheel: Path,
    *,
    tamper: str | None = None,
    injected: tuple[str, bytes] | None = None,
    executable_member: str | None = None,
) -> str:
    entries = source_entries(ROOT / "src/heliostune")
    if tamper == "changed":
        first = sorted(entries)[0]
        entries[first] += b"\n# malicious\n"
    elif tamper == "extra":
        entries["heliostune/backdoor.py"] = b"print('owned')\n"
    if injected is not None:
        entries[injected[0]] = injected[1]
    dist_info = "heliostune-1.0.0.dist-info"
    entries[f"{dist_info}/METADATA"] = b"Metadata-Version: 2.1\nName: heliostune\nVersion: 1.0.0\n"
    entries[f"{dist_info}/WHEEL"] = b"Wheel-Version: 1.0\nTag: py3-none-any\n"
    entries[f"{dist_info}/entry_points.txt"] = (
        b"[console_scripts]\nheliostune = heliostune.cli:main\n"
    )
    entries[f"{dist_info}/licenses/LICENSE"] = (ROOT / "LICENSE").read_bytes()
    record_name = f"{dist_info}/RECORD"
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    for name, payload in sorted(entries.items()):
        writer.writerow((name, _record_hash(payload), str(len(payload))))
    writer.writerow((record_name, "", ""))
    entries[record_name] = stream.getvalue().encode()
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in sorted(entries.items()):
            if name == executable_member:
                info = zipfile.ZipInfo(name)
                info.external_attr = (stat.S_IFREG | 0o755) << 16
                archive.writestr(info, payload)
            else:
                archive.writestr(name, payload)
    packaged = {
        name: payload for name, payload in entries.items() if name.startswith("heliostune/")
    }
    return source_digest(packaged)


def _manifest(wheel: Path, *, source_sha256: str | None = None) -> Path:
    payload = {
        "schema_version": 1,
        "head_commit": HEAD,
        "source_sha256": source_digest(source_entries(ROOT / "src/heliostune"))
        if source_sha256 is None
        else source_sha256,
        "wheel_filename": wheel.name,
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "python_version": "3.11",
        "pip_dependencies": [
            "numpy==2.4.6",
            "rich==14.3.4",
            "zstandard==0.25.0",
            "torch==2.8.0",
            "triton==3.4.0",
        ],
        "build_dependencies": ["hatchling==1.32.0"],
        "build_tools": {"uv": "0.12.5", "hatchling": "1.32.0"},
        "wheel_install_args": ["--no-deps"],
    }
    path = wheel.with_name(f"{wheel.name}.manifest.json")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


@pytest.fixture
def entrypoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    wheel = tmp_path / "heliostune-1.0.0-py3-none-any.whl"
    _wheel(wheel)
    manifest = _manifest(wheel)
    modal_stub = types.ModuleType("modal")
    modal_stub.App = _App  # type: ignore[attr-defined]
    modal_stub.Image = _Image  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "modal", modal_stub)
    monkeypatch.setenv("HELIOSTUNE_MODAL_WHEEL", str(wheel))
    monkeypatch.setenv("HELIOSTUNE_MODAL_WHEEL_MANIFEST", str(manifest))

    completed = SimpleNamespace(stdout="")

    def clean_git(command: list[str], **_: object) -> object:
        if command[1:3] == ["status", "--porcelain"]:
            return completed
        return SimpleNamespace(stdout=HEAD + "\n")

    monkeypatch.setattr("subprocess.run", clean_git)
    name = f"_test_modal_fusion_executor_{tmp_path.name}"
    spec = importlib.util.spec_from_file_location(name, ENTRYPOINT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_modal_decorator_keeps_every_hardening_argument() -> None:
    tree = ast.parse(ENTRYPOINT.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "execute_fusion_suite"
    )
    decorator = next(
        item
        for item in function.decorator_list
        if isinstance(item, ast.Call)
        and isinstance(item.func, ast.Attribute)
        and item.func.attr == "function"
    )
    assert {keyword.arg: ast.unparse(keyword.value) for keyword in decorator.keywords} == {
        "image": "image",
        "gpu": "_MODAL_SELECTOR",
        "retries": "0",
        "timeout": "_REMOTE_TIMEOUT_SECONDS",
        "max_containers": "1",
        "single_use_containers": "True",
        "block_network": "True",
        "restrict_modal_access": "True",
        "_experimental_restrict_output": "True",
    }
    source = ENTRYPOINT.read_text()
    assert "gpu=_MODAL_SELECTOR" in source
    assert "provider attempts are unobservable" in source
    assert "methodology bundle" in source
    assert ").to_transport_json()" in source


def test_malicious_wheel_with_matching_forged_manifest_is_rejected(
    entrypoint: Any, tmp_path: Path
) -> None:
    malicious = tmp_path / "heliostune-9.9.9-py3-none-any.whl"
    forged_source = _wheel(malicious, tamper="extra")
    _manifest(malicious, source_sha256=forged_source)
    with pytest.raises(RuntimeError, match="outside the allowlist"):
        entrypoint.validate_wheel_manifest(malicious, repository=ROOT)


def test_record_hash_tampering_is_rejected(entrypoint: Any, tmp_path: Path) -> None:
    wheel = tmp_path / "heliostune-2.0.0-py3-none-any.whl"
    _wheel(wheel)
    _manifest(wheel)
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr("heliostune/backdoor.py", b"tampered")
    manifest = json.loads(entrypoint.wheel_manifest_path(wheel).read_text())
    manifest["wheel_sha256"] = sha256_bytes(wheel.read_bytes())
    entrypoint.wheel_manifest_path(wheel).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    with pytest.raises(RuntimeError, match="RECORD inventory"):
        entrypoint.validate_wheel_manifest(wheel, repository=ROOT)


@pytest.mark.parametrize(
    ("name", "payload"),
    [
        ("heliostune_loader.pth", b"import sitecustomize\n"),
        (
            "heliostune-1.0.0.data/purelib/sitecustomize.py",
            b"raise RuntimeError('owned')\n",
        ),
        ("sitecustomize.py", b"raise RuntimeError('owned')\n"),
        ("backdoor/__init__.py", b"raise RuntimeError('owned')\n"),
    ],
    ids=("root-pth", "wheel-data", "top-level-module", "top-level-package"),
)
def test_valid_record_wheel_rejects_payloads_outside_exact_allowlist(
    entrypoint: Any,
    tmp_path: Path,
    name: str,
    payload: bytes,
) -> None:
    wheel = tmp_path / "heliostune-1.0.0-py3-none-any.whl"
    _wheel(wheel, injected=(name, payload))
    _manifest(wheel)
    with pytest.raises(RuntimeError, match="outside the allowlist"):
        entrypoint.validate_wheel_manifest(wheel, repository=ROOT)


def test_valid_record_wheel_rejects_extra_dist_info(
    entrypoint: Any,
    tmp_path: Path,
) -> None:
    wheel = tmp_path / "heliostune-1.0.0-py3-none-any.whl"
    _wheel(
        wheel,
        injected=(
            "backdoor-1.0.0.dist-info/METADATA",
            b"Metadata-Version: 2.1\nName: backdoor\nVersion: 1.0.0\n",
        ),
    )
    _manifest(wheel)
    with pytest.raises(RuntimeError, match="exactly one .dist-info directory"):
        entrypoint.validate_wheel_manifest(wheel, repository=ROOT)


def test_valid_record_wheel_rejects_executable_source_member(
    entrypoint: Any,
    tmp_path: Path,
) -> None:
    wheel = tmp_path / "heliostune-1.0.0-py3-none-any.whl"
    member = sorted(source_entries(ROOT / "src/heliostune"))[0]
    _wheel(wheel, executable_member=member)
    _manifest(wheel)
    with pytest.raises(RuntimeError, match="executable member"):
        entrypoint.validate_wheel_manifest(wheel, repository=ROOT)


def test_preflight_retains_exact_plugin_suite_and_manifest_bytes(
    entrypoint: Any, tmp_path: Path
) -> None:
    benchmark_copy = tmp_path / "benchmarks"
    plugin = benchmark_copy / "plugins/fusion-reference-plugin-v1.json"
    suite = benchmark_copy / "suites/gated-mlp-epilogue-v1.json"
    rmsnorm = benchmark_copy / "suites/residual-rmsnorm-v1.json"
    plugin.parent.mkdir(parents=True)
    suite.parent.mkdir(parents=True)
    plugin.write_bytes(PLUGIN.read_bytes())
    suite.write_bytes(SUITE.read_bytes())
    rmsnorm.write_bytes((ROOT / "benchmarks/suites/residual-rmsnorm-v1.json").read_bytes())
    plan = entrypoint._preflight(suite, plugin, tmp_path / "receipt")
    assert plan.suite_bytes == verify_suite(suite).bytes
    assert plan.plugin_bytes == verify_plugin(plugin).bytes
    assert (
        plan.manifest_bytes == entrypoint.wheel_manifest_path(entrypoint._MODAL_WHEEL).read_bytes()
    )
    plugin.write_bytes(b'{"forged":true}\n')
    assert sha256_bytes(plan.plugin_bytes) == plan.intent.plugin_sha256


@pytest.mark.parametrize(
    ("suite", "plugin"),
    [
        (MLP_SUITE, REFERENCE_PLUGIN),
        (RMSNORM_SUITE, REFERENCE_PLUGIN),
        (NATIVE_SUITE, NATIVE_PLUGIN),
    ],
)
def test_preflight_accepts_exact_registry_suite_plugin_pairs(
    entrypoint: Any,
    tmp_path: Path,
    suite: Path,
    plugin: Path,
) -> None:
    plan = entrypoint._preflight(suite, plugin, tmp_path / suite.stem)
    execution = fusion_execution_spec(verify_suite(suite).sha256)
    assert plan.intent.suite_sha256 == execution.suite_sha256
    assert plan.intent.plugin_sha256 == execution.plugin_sha256
    assert sha256_bytes(plan.plugin_bytes) == execution.plugin_sha256


@pytest.mark.parametrize(
    ("suite", "plugin"),
    [
        (MLP_SUITE, NATIVE_PLUGIN),
        (NATIVE_SUITE, REFERENCE_PLUGIN),
    ],
)
def test_preflight_rejects_crossed_registry_pair_before_wheel_or_spawn(
    entrypoint: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    suite: Path,
    plugin: Path,
) -> None:
    monkeypatch.setattr(
        entrypoint,
        "validate_wheel_manifest",
        lambda *_args, **_kwargs: pytest.fail("registry mismatch must precede wheel validation"),
    )
    monkeypatch.setattr(
        entrypoint,
        "_execute_plan",
        lambda *_args, **_kwargs: pytest.fail("registry mismatch must prevent the Modal spawn"),
    )
    with pytest.raises(ValueError, match="plugin does not match"):
        entrypoint.main(str(suite), str(plugin), str(tmp_path / "rejected"))


def test_native_remote_request_dispatches_through_common_local_executor(
    entrypoint: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = entrypoint._preflight(NATIVE_SUITE, NATIVE_PLUGIN, tmp_path / "native")
    events: list[str] = []
    hardware = object()
    captured: dict[str, object] = {}
    original_validate_wheel = entrypoint.validate_wheel_manifest

    def validate_wheel(*args: object, **kwargs: object) -> object:
        events.append("wheel")
        return original_validate_wheel(*args, **kwargs)

    def hardware_profile(gpu: str) -> object:
        assert gpu == "H100"
        events.append("hardware")
        return hardware

    def validate_observed_hardware(observed: object, expectation: object) -> None:
        assert observed is hardware
        assert expectation is not None
        events.append("validate-hardware")

    @dataclass(frozen=True)
    class FakeResult:
        verified_suite_path: str
        verified_suite_sha256: str
        verified_suite_bytes: bytes
        environment: dict[str, object]

        def to_dict(self) -> dict[str, object]:
            return {"verified_suite_path": self.verified_suite_path}

    def execute(suite_path: Path) -> FakeResult:
        events.append("execute")
        assert suite_path.read_bytes() == plan.suite_bytes
        return FakeResult(
            str(suite_path),
            plan.intent.suite_sha256,
            plan.suite_bytes,
            {"executor": "native-fake"},
        )

    def envelope(**kwargs: object) -> object:
        captured.update(kwargs)
        return SimpleNamespace(to_transport_json=lambda: "native-transport")

    monkeypatch.setattr(entrypoint, "validate_wheel_manifest", validate_wheel)
    monkeypatch.setattr("heliostune.kernel.get_hardware_profile", hardware_profile)
    monkeypatch.setattr("heliostune.hardware.validate_hardware", validate_observed_hardware)
    monkeypatch.setattr("heliostune.local_executor.execute_local_suite", execute)
    monkeypatch.setattr("heliostune.remote_execution.RemoteResultEnvelope", envelope)

    assert entrypoint.execute_fusion_suite(plan.request_json) == "native-transport"
    assert events == ["wheel", "wheel", "hardware", "validate-hardware", "execute"]
    assert captured["environment"] == {"executor": "native-fake"}
    assert captured["result"] == {"verified_suite_path": str(NATIVE_SUITE)}


@pytest.mark.parametrize("mismatch", ["suite", "plugin"])
def test_remote_registry_mismatch_precedes_wheel_hardware_and_execution(
    entrypoint: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mismatch: str,
) -> None:
    plan = entrypoint._preflight(NATIVE_SUITE, NATIVE_PLUGIN, tmp_path / mismatch)
    suite_bytes = plan.suite_bytes
    intent = plan.intent
    expected_error: type[Exception]
    if mismatch == "suite":
        suite_bytes += b" "
        intent = replace(intent, suite_sha256=sha256_bytes(suite_bytes))
        expected_error = SchemaError
        message = "unsupported suite SHA-256"
    else:
        intent = replace(intent, plugin_sha256="0" * 64)
        expected_error = RuntimeError
        message = "plugin is outside"
    request_json = encode_remote_request(intent, suite_bytes)

    monkeypatch.setattr(
        entrypoint,
        "validate_wheel_manifest",
        lambda *_args, **_kwargs: pytest.fail("registry mismatch must precede wheel validation"),
    )
    monkeypatch.setattr(
        "heliostune.kernel.get_hardware_profile",
        lambda *_args, **_kwargs: pytest.fail("registry mismatch must precede hardware probing"),
    )
    monkeypatch.setattr(
        "heliostune.local_executor.execute_local_suite",
        lambda *_args, **_kwargs: pytest.fail("registry mismatch must precede execution"),
    )

    with pytest.raises(expected_error, match=message):
        entrypoint.execute_fusion_suite(request_json)


@pytest.mark.parametrize(
    ("outcome", "expected_code"),
    [("completed", 0), ("failed", 1), ("aborted", 1)],
)
def test_returned_terminal_results_publish_matching_receipts(
    entrypoint: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
    expected_code: int,
) -> None:
    plan = entrypoint._preflight(SUITE, PLUGIN, tmp_path / f"receipt-{outcome}")
    events: list[object] = []

    class Call:
        object_id = "fc-one"

        def get(self, *, timeout: int) -> str:
            events.append(("get", timeout))
            return "transport-json"

    class Remote:
        def spawn(self, request: str) -> Call:
            events.append("spawn")
            assert request == plan.request_json
            return Call()

    result = SimpleNamespace(outcome=outcome)
    envelope = SimpleNamespace(to_json=lambda: "canonical-envelope-json")
    monkeypatch.setattr(
        "heliostune.remote_execution.validate_remote_result",
        lambda *args, **kwargs: (envelope, result),
    )
    published: dict[str, object] = {}

    def capture(records: object, **kwargs: object) -> None:
        published.update(kwargs)

    monkeypatch.setattr("heliostune.remote_execution.write_remote_receipt", capture)
    code = entrypoint._execute_plan(plan, plan.intent.output_path, Remote())
    assert code == expected_code
    assert events == ["spawn", ("get", CLIENT_TIMEOUT_SECONDS)]
    assert published["status"] == outcome
    assert published["result_payload"] == "canonical-envelope-json"
    assert published["plugin_bytes"] == plan.plugin_bytes
    _, journal_path = remote_artifact_paths(plan.intent.output_path)
    states = [json.loads(line)["state"] for line in journal_path.read_text().splitlines()]
    assert states == ["intent", "spawned", "retrieval_started", outcome]


def test_spawn_exception_is_acknowledgement_lost_not_cancellation(
    entrypoint: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = entrypoint._preflight(SUITE, PLUGIN, tmp_path / "spawn-lost")
    published: dict[str, object] = {}
    monkeypatch.setattr(
        "heliostune.remote_execution.write_remote_receipt",
        lambda records, **kwargs: published.update(kwargs),
    )

    class Remote:
        def spawn(self, _: str) -> object:
            raise RuntimeError("transport acknowledgement lost")

    with pytest.raises(RuntimeError, match="acknowledgement"):
        entrypoint._execute_plan(plan, plan.intent.output_path, Remote())
    _, journal_path = remote_artifact_paths(plan.intent.output_path)
    states = [json.loads(line)["state"] for line in journal_path.read_text().splitlines()]
    assert states == ["intent", "spawn_acknowledgement_lost", "unresolved"]
    assert "cancellation_requested" not in states
    assert published["status"] == "unresolved"
    assert published["client_spawn_count"] == 1


@pytest.mark.parametrize("failure", [TimeoutError("deadline"), KeyboardInterrupt()])
def test_known_handle_baseexception_cancels_before_journaling_unresolved(
    entrypoint: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
) -> None:
    plan = entrypoint._preflight(SUITE, PLUGIN, tmp_path / f"cancel-{type(failure).__name__}")
    events: list[object] = []

    class Call:
        object_id = "fc-cancel"

        def get(self, *, timeout: int) -> str:
            events.append(("get", timeout))
            raise failure

        def cancel(self, terminate_containers: bool = False) -> None:
            events.append(("cancel", terminate_containers))

    class Remote:
        def spawn(self, _: str) -> Call:
            return Call()

    original_append = RemoteJournal.append

    def recording_append(
        self: RemoteJournal,
        state: JournalState,
        *,
        call_id: str | None = None,
        detail: str | None = None,
    ) -> RemoteJournalRecord:
        events.append(("journal", state))
        return original_append(self, state, call_id=call_id, detail=detail)

    monkeypatch.setattr(RemoteJournal, "append", recording_append)
    monkeypatch.setattr(
        "heliostune.remote_execution.write_remote_receipt", lambda *args, **kwargs: None
    )
    with pytest.raises(type(failure)):
        entrypoint._execute_plan(plan, plan.intent.output_path, Remote())
    cancel_index = events.index(("cancel", True))
    assert cancel_index < events.index(("journal", "cancellation_requested"))
    assert events[-2:] == [("journal", "cancellation_requested"), ("journal", "unresolved")]


def test_journal_failure_cannot_prevent_known_handle_cancellation(
    entrypoint: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = entrypoint._preflight(SUITE, PLUGIN, tmp_path / "journal-failure")
    cancelled: list[bool] = []

    class Call:
        object_id = "fc-cancel"

        def get(self, *, timeout: int) -> str:
            raise TimeoutError("deadline")

        def cancel(self, terminate_containers: bool = False) -> None:
            cancelled.append(terminate_containers)

    class Remote:
        def spawn(self, _: str) -> Call:
            return Call()

    original_append = RemoteJournal.append

    def failing_append(
        self: RemoteJournal,
        state: JournalState,
        *,
        call_id: str | None = None,
        detail: str | None = None,
    ) -> RemoteJournalRecord:
        if state == "cancellation_requested":
            raise OSError("disk full")
        return original_append(self, state, call_id=call_id, detail=detail)

    monkeypatch.setattr(RemoteJournal, "append", failing_append)
    with pytest.raises(TimeoutError):
        entrypoint._execute_plan(plan, plan.intent.output_path, Remote())
    assert cancelled == [True]


def test_parent_swap_after_spawn_cancels_and_never_publishes_to_substitute(
    entrypoint: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    plan = entrypoint._preflight(SUITE, PLUGIN, parent / "receipt")
    moved = tmp_path / "moved"
    cancelled: list[bool] = []

    class Call:
        object_id = "fc-parent"

        def cancel(self, terminate_containers: bool = False) -> None:
            cancelled.append(terminate_containers)

    class Remote:
        def spawn(self, _: str) -> Call:
            os.rename(parent, moved)
            parent.mkdir()
            return Call()

    monkeypatch.setattr(
        "heliostune.remote_execution.write_remote_receipt",
        lambda *args, **kwargs: pytest.fail("substituted parent must not receive a receipt"),
    )
    with pytest.raises(ArtifactError, match="parent identity changed"):
        entrypoint._execute_plan(plan, plan.intent.output_path, Remote())
    assert cancelled == [True]
    assert not (parent / "receipt").exists()
    assert (moved / "receipt.remote-intent.json").is_file()


def test_duplicate_intent_blocks_before_second_spawn(entrypoint: Any, tmp_path: Path) -> None:
    plan = entrypoint._preflight(SUITE, PLUGIN, tmp_path / "duplicate")
    intent_path, _ = remote_artifact_paths(plan.intent.output_path)
    intent_path.write_bytes(canonical_json_bytes({"already": "used"}))

    class Remote:
        def spawn(self, _: str) -> object:
            pytest.fail("duplicate intent must block before spawn")

    with pytest.raises(ArtifactError, match="remote intent"):
        entrypoint._execute_plan(plan, plan.intent.output_path, Remote())
