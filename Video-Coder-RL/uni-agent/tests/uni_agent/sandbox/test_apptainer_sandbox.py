from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from uni_agent.sandbox.apptainer import ApptainerSandbox, _image_uri, _sif_name
from uni_agent.sandbox.base import ExecResult, ImageMount, Sandbox, SandboxConfig
from uni_agent.sandbox.registry import SANDBOX_MODULES, build_sandbox


def _ok(stdout: str = "") -> ExecResult:
    return ExecResult(exit_code=0, stdout=stdout, stderr="")


def _sandbox(tmp_path: Path, **kwargs) -> ApptainerSandbox:
    kwargs.setdefault("image", "example/agent:1.0")
    return ApptainerSandbox(
        sif_dir=str(tmp_path / "sif"), work_root=str(tmp_path / "inst"), cache_dir=str(tmp_path / "cache"), **kwargs
    )


@pytest.mark.cpu
@pytest.mark.level0
def test_registry_builds_apptainer_sandbox_from_config(tmp_path):
    config = SandboxConfig(
        provider="apptainer",
        image="example/agent:1.0",
        image_mounts=[{"image": "example/tool:2", "mount_path": "/opt/tool2"}],
        executable_paths={"tool2": "/opt/tool2/bin/tool2"},
        sandbox_kwargs={"binds": ["/opt/tool:/opt/tool:ro"], "add_to_path": ["/opt/tool/bin"]},
    )

    sandbox = build_sandbox(config)

    assert SANDBOX_MODULES["apptainer"] == "uni_agent.sandbox.apptainer"
    assert isinstance(sandbox, ApptainerSandbox)
    assert sandbox.image == "example/agent:1.0"
    assert sandbox.image_mounts == [ImageMount(image="example/tool:2", mount_path="/opt/tool2")]
    assert sandbox.executable_paths == {"tool2": "/opt/tool2/bin/tool2"}
    assert sandbox.binds == ["/opt/tool:/opt/tool:ro"]
    assert sandbox.add_to_path == ("/opt/tool/bin",)
    assert sandbox.pull_policy == "missing"
    assert "exec" not in ApptainerSandbox.__dict__
    assert ApptainerSandbox.exec is Sandbox.exec


@pytest.mark.cpu
@pytest.mark.level0
def test_image_uri_and_sif_name():
    assert _image_uri("swebench/sweb.eval.x86_64.a:latest") == "docker://swebench/sweb.eval.x86_64.a:latest"
    assert _image_uri("swebench/sweb.eval.x86_64.a") == "docker://swebench/sweb.eval.x86_64.a:latest"
    assert _image_uri("localhost:5000/img") == "docker://localhost:5000/img:latest"
    assert _image_uri("img@sha256:abc") == "docker://img@sha256:abc"
    assert _image_uri("oras://reg/img:1") == "oras://reg/img:1"
    assert _sif_name("docker://swebench/x:latest") == "docker_swebench_x_latest.sif"


@pytest.mark.cpu
@pytest.mark.level0
def test_rejects_bad_arguments(tmp_path):
    with pytest.raises(ValueError, match="requires an image"):
        ApptainerSandbox()
    with pytest.raises(ValueError, match="pull_policy"):
        _sandbox(tmp_path, pull_policy="sometimes")
    with pytest.raises(ValueError, match="add_to_path"):
        _sandbox(tmp_path, add_to_path="/opt/bin")
    with pytest.raises(ValueError, match="max_concurrent_pulls"):
        _sandbox(tmp_path, max_concurrent_pulls=0)
    with pytest.raises(ValueError, match="start_timeout"):
        _sandbox(tmp_path, start_timeout=0)
    with pytest.raises(ValueError, match="mount_path must not contain"):
        _sandbox(tmp_path, image_mounts=[ImageMount(image="example/tool:2", mount_path="/opt/a,b")])


@pytest.mark.cpu
@pytest.mark.level0
def test_start_pulls_missing_image_then_starts_instance(tmp_path, monkeypatch):
    sandbox = _sandbox(tmp_path, binds=["/a:/b"], instance_args=["--nv"], pull_timeout=900)
    calls: list[tuple[tuple[str, ...], float | None]] = []

    async def fake_run(*args: str, timeout=None, env=None, **_):
        calls.append((args, timeout))
        if args[0] == "pull":
            cache = Path(env["APPTAINER_CACHEDIR"])
            assert cache.parent == tmp_path / "cache" and cache.is_dir()  # private per-pull cache
            Path(args[2]).write_bytes(b"sif")
        return _ok()

    monkeypatch.setattr(sandbox, "_run", fake_run)
    asyncio.run(sandbox.start())

    sif = tmp_path / "sif" / "docker_example_agent_1.0.sif"
    assert list((tmp_path / "cache").iterdir()) == []  # the private pull cache is removed
    name = sandbox._instance_name
    instance_dir = tmp_path / "inst" / name
    assert sif.read_bytes() == b"sif"
    (pull_args, pull_timeout), (start_args, _) = calls
    assert pull_args[:2] == ("pull", "--force") and pull_args[3] == "docker://example/agent:1.0"
    assert pull_timeout == 900.0
    assert start_args == (
        "instance",
        "start",
        "--cleanenv",
        "--overlay",
        str(instance_dir / "overlay"),
        "--fakeroot",
        "--contain",
        "--workdir",
        str(instance_dir / "workdir"),
        "--bind",
        "/a:/b",
        "--nv",
        str(sif),
        name,
    )


@pytest.mark.cpu
@pytest.mark.level0
def test_start_reuses_cached_sif_and_local_sif(tmp_path, monkeypatch):
    cached = _sandbox(tmp_path)
    (tmp_path / "sif").mkdir()
    (tmp_path / "sif" / "docker_example_agent_1.0.sif").write_bytes(b"sif")
    local_sif = tmp_path / "local.sif"
    local_sif.write_bytes(b"sif")
    local = _sandbox(tmp_path, image=str(local_sif), fakeroot=False, contain=False)
    calls: list[tuple[str, ...]] = []

    async def fake_run(*args: str, timeout=None, **_):
        calls.append(args)
        return _ok()

    for sandbox in (cached, local):
        monkeypatch.setattr(sandbox, "_run", fake_run)
        asyncio.run(sandbox.start())

    assert [args[:2] for args in calls] == [("instance", "start")] * 2
    assert "--fakeroot" not in calls[1] and "--contain" not in calls[1]
    assert calls[1][-2] == str(local_sif)


_SIF_LIST = """\
------------------------------------------------------------------------------
ID   |GROUP   |LINK    |SIF POSITION (start-end)  |TYPE
------------------------------------------------------------------------------
1    |1       |NONE    |32176-32214               |Def.FILE
2    |1       |NONE    |32214-36328               |JSON.Generic
4    |1       |NONE    |36864-2256896             |FS (Squashfs/*System/amd64)
"""


@pytest.mark.cpu
@pytest.mark.level0
def test_start_binds_image_mounts_and_links_executables(tmp_path, monkeypatch):
    sandbox = _sandbox(
        tmp_path,
        binds=["/a:/b"],
        image_mounts=[ImageMount(image="example/tool:2", mount_path="/opt/tool")],
        executable_paths={"tool": "/opt/tool/bin/tool"},
    )
    calls: list[tuple[str, ...]] = []

    async def fake_run(*args: str, timeout=None, env=None, **_):
        calls.append(args)
        if args[0] == "pull":
            Path(args[2]).write_bytes(b"sif")
        if args[:2] == ("sif", "list"):
            return _ok(_SIF_LIST)
        if "-c" in args and args[-1].startswith("command -v"):
            return _ok("/usr/bin/tool\n")
        return _ok()

    monkeypatch.setattr(sandbox, "_run", fake_run)
    asyncio.run(sandbox.start())

    tool_sif = tmp_path / "sif" / "docker_example_tool_2.sif"
    pulls = [args[3] for args in calls if args[0] == "pull"]
    assert pulls == ["docker://example/agent:1.0", "docker://example/tool:2"]
    assert ("sif", "list", str(tool_sif)) in calls
    start = next(args for args in calls if args[:2] == ("instance", "start"))
    binds = [start[i + 1] for i, arg in enumerate(start) if arg == "--bind"]
    assert binds == [f"{tool_sif}:/opt/tool:image-src=/,id=4", "/a:/b"]
    # executables are linked inside the started instance, after it is up
    execs = [args[args.index(f"instance://{sandbox._instance_name}") + 1 :] for args in calls if args[1:2] == ("exec",)]
    assert ("ln", "-sfn", "/opt/tool/bin/tool", "/usr/bin/tool") in execs
    assert calls.index(start) < next(i for i, args in enumerate(calls) if args[1:2] == ("exec",))


@pytest.mark.cpu
@pytest.mark.level0
def test_image_mount_without_root_partition_fails_before_instance_start(tmp_path, monkeypatch):
    tool_sif = tmp_path / "tool.sif"
    tool_sif.write_bytes(b"sif")
    mount = ImageMount(image=str(tool_sif), mount_path="/opt/t")
    sandbox = _sandbox(tmp_path, image=str(tool_sif), image_mounts=[mount])
    calls: list[tuple[str, ...]] = []

    async def fake_run(*args: str, timeout=None, **_):
        calls.append(args)
        return _ok(_SIF_LIST.replace("*System", "Data"))

    monkeypatch.setattr(sandbox, "_run", fake_run)
    with pytest.raises(RuntimeError, match="no root filesystem partition"):
        asyncio.run(sandbox.start())
    assert calls == [("sif", "list", str(tool_sif))]
    assert sandbox._instance_name is None


@pytest.mark.cpu
@pytest.mark.level0
def test_pull_policy_never_requires_cached_sif(tmp_path):
    sandbox = _sandbox(tmp_path, pull_policy="never")

    with pytest.raises(RuntimeError, match="not cached"):
        asyncio.run(sandbox.start())
    assert sandbox._instance_name is None


@pytest.mark.cpu
@pytest.mark.level0
def test_failed_start_is_cleaned_up_by_stop(tmp_path, monkeypatch):
    local_sif = tmp_path / "local.sif"
    local_sif.write_bytes(b"sif")
    sandbox = _sandbox(tmp_path, image=str(local_sif))
    calls: list[tuple[str, ...]] = []

    async def fake_run(*args: str, timeout=None, **_):
        calls.append(args)
        if args[:2] == ("instance", "start"):
            return ExecResult(exit_code=255, stdout="", stderr="FATAL: boom")
        return _ok()

    monkeypatch.setattr(sandbox, "_run", fake_run)
    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(sandbox.start())
    name = sandbox._instance_name
    asyncio.run(sandbox.stop())

    assert calls[-1] == ("instance", "stop", name)
    assert list((tmp_path / "inst").iterdir()) == []
    assert sandbox._instance_name is None


@pytest.mark.cpu
@pytest.mark.level0
def test_exec_exports_env_inside_container(tmp_path, monkeypatch):
    sandbox = _sandbox(tmp_path, add_to_path=["/opt/claude-code/bin"])
    sandbox._instance_name = "uni-agent-test"
    calls: list[tuple[str, ...]] = []

    async def fake_run(*args: str, timeout=None, **_):
        calls.append(args)
        return _ok("out")

    monkeypatch.setattr(sandbox, "_run", fake_run)
    asyncio.run(sandbox.exec(["claude", "-p", "hi"], workdir="/testbed", env={"NO_PROXY": "a,b"}))
    asyncio.run(sandbox.exec(["true"]))

    assert calls[0] == (
        "-q",
        "exec",
        "--cleanenv",
        "--cwd",
        "/testbed",
        "instance://uni-agent-test",
        "/bin/sh",
        "-c",
        'export PATH=/opt/claude-code/bin:"${PATH:-}"\nexport NO_PROXY=a,b\nexec "$@"',
        "sh",
        "claude",
        "-p",
        "hi",
    )
    assert calls[1][3:5] == ("--cwd", "/")
    assert calls[1][-4:] == ("-c", 'export PATH=/opt/claude-code/bin:"${PATH:-}"\nexec "$@"', "sh", "true")


@pytest.mark.cpu
@pytest.mark.level0
def test_exec_rejects_invalid_env_name(tmp_path):
    sandbox = _sandbox(tmp_path)
    sandbox._instance_name = "uni-agent-test"

    with pytest.raises(ValueError, match="invalid environment variable name"):
        asyncio.run(sandbox.exec(["true"], env={"BAD-NAME": "x"}))


@pytest.mark.cpu
@pytest.mark.level0
def test_exec_requires_started_instance(tmp_path):
    with pytest.raises(RuntimeError, match="not started"):
        asyncio.run(_sandbox(tmp_path).exec(["true"]))


@pytest.mark.cpu
@pytest.mark.level0
def test_cancelled_pull_lock_waiter_does_not_leak_lock(tmp_path):
    lock = tmp_path / "img.lock"

    async def run() -> None:
        held = await ApptainerSandbox._lock(lock)
        waiter = asyncio.create_task(ApptainerSandbox._lock(lock))
        await asyncio.sleep(0.1)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        os.close(held)  # closing the fd releases the flock
        os.close(await asyncio.wait_for(ApptainerSandbox._lock(lock), timeout=5))

    asyncio.run(run())


@pytest.mark.cpu
@pytest.mark.level0
def test_pulls_of_different_images_are_bounded_by_slots(tmp_path, monkeypatch):
    active = peak = 0

    async def fake_pull(self, uri, sif):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.3)
        sif.write_bytes(b"sif")
        active -= 1

    monkeypatch.setattr(ApptainerSandbox, "_pull", fake_pull)
    sandboxes = [_sandbox(tmp_path, image=f"example/img{i}:1", max_concurrent_pulls=2) for i in range(5)]

    async def run() -> list[str]:
        return await asyncio.gather(*(sb._resolve_sif(sb.image) for sb in sandboxes))

    sifs = asyncio.run(run())
    assert peak == 2
    assert len(set(sifs)) == 5 and all(Path(s).is_file() for s in sifs)


@pytest.mark.cpu
@pytest.mark.level0
def test_cancellation_kills_the_apptainer_process(tmp_path):
    pid_file = tmp_path / "pid"
    sandbox = _sandbox(tmp_path, apptainer_binary="sh")

    async def run() -> None:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(sandbox._run_apptainer("-c", f"echo $$ > {pid_file}; exec sleep 30"), timeout=1)

    asyncio.run(run())
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)


@pytest.mark.cpu
@pytest.mark.level0
def test_output_file_does_not_wait_for_daemons_holding_stdio(tmp_path):
    """``instance start`` leaves FUSE daemons holding the child's stdio; under uvloop a
    pipe would then never reach EOF. Output routed to a file must return promptly."""
    uvloop = pytest.importorskip("uvloop")
    sandbox = _sandbox(tmp_path, apptainer_binary="sh")
    pid_file = tmp_path / "daemon.pid"

    async def run() -> tuple[int, bytes, bytes]:
        script = f"sleep 30 & echo $! > {pid_file}; echo started"
        return await asyncio.wait_for(sandbox._run_apptainer("-c", script, output=tmp_path / "out.log"), timeout=10)

    try:
        assert uvloop.run(run()) == (0, b"started\n", b"")
    finally:
        os.kill(int(pid_file.read_text()), 9)


def _apptainer_usable() -> bool:
    if shutil.which("apptainer") is None:
        return False
    return subprocess.run(["apptainer", "--version"], capture_output=True).returncode == 0


@pytest.mark.cpu
@pytest.mark.level1
@pytest.mark.skipif(not _apptainer_usable(), reason="apptainer is not installed")
def test_real_apptainer_round_trip(tmp_path):
    """Opt-in end-to-end check against a real apptainer (needs network to pull a small image).

    Uses a glibc task image: rootless ``--fakeroot`` without ``/etc/subuid`` injects the
    host ``fakeroot`` binary, which cannot run in musl/static images like busybox. A static
    busybox is fine as a *mounted* image, which also exercises ``executable_paths``.
    """
    sandbox = _sandbox(
        tmp_path,
        image="docker://debian:bookworm-slim",
        image_mounts=[ImageMount(image="docker://busybox:1.36-musl", mount_path="/opt/bb")],
        # busybox picks its applet from argv[0], so the link keeps the binary's name
        executable_paths={"busybox": "/opt/bb/bin/busybox"},
    )

    async def run() -> None:
        async with sandbox:
            result = await sandbox.exec(["sh", "-c", "id -u; pwd; echo $V"], workdir="/tmp", env={"V": "a,b"})
            assert result.stdout.split() == ["0", "/tmp", "a,b"]
            linked = await sandbox.exec(["sh", "-c", "command -v busybox; busybox echo mounted"])
            assert linked.stdout.split() == ["/usr/bin/busybox", "mounted"]
            await sandbox.write_file("/data/x.bin", b"\x00\x01binary")
            assert await sandbox.read_file("/data/x.bin") == b"\x00\x01binary"
            assert (await sandbox.exec(["sh", "-c", "exit 3"])).exit_code == 3

    asyncio.run(run())
    assert list((tmp_path / "inst").iterdir()) == []
