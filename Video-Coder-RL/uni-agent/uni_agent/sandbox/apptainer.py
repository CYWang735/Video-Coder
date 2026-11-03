"""Apptainer sandbox: a daemonless, rootless container provider for HPC nodes.

Each sandbox is one ``apptainer instance`` started from a SIF image with:

- ``--fakeroot`` so the task runs as uid 0 (SWE images expect to own ``/testbed``);
- a private *directory* overlay, so the read-only SIF becomes writable without the
  tiny ``--writable-tmpfs`` session limit;
- ``--contain --workdir`` so ``/tmp`` / ``/var/tmp`` are private per sandbox;
- the host network namespace, so in-sandbox agents reach the gateway directly
  (the reason to prefer this over a remote provider on a cluster).

Without an ``/etc/subuid`` entry, ``--fakeroot`` falls back to a root-mapped
namespace that injects the host ``fakeroot`` binary, so the image must be
glibc-based (Debian/Ubuntu SWE images are; busybox/alpine are not). Set
``fakeroot: false`` for such images; the task then runs as the calling user.

Non-SIF images (``docker://...``, or a bare Docker reference) are pulled once into
``sif_dir`` under a cross-process file lock and reused by every later sandbox. At most
``max_concurrent_pulls`` pulls run at once per ``sif_dir`` (across processes), and each
uses a private layer cache under ``cache_dir`` (``$APPTAINER_CACHEDIR``, else
node-local scratch) that is deleted afterwards: apptainer's shared OCI cache is
corrupted by concurrent pulls of different images ("not found in layout",
"unexpected end of JSON input"), and its default ``~/.apptainer/cache`` would fill
a small HPC home quota. For large datasets, pre-pull the images before a run so
first pulls do not count against ``SANDBOX_STARTUP_TIMEOUT``.

``image_mounts`` images are resolved to SIFs the same way and bound read-only at their
``mount_path`` from the SIF's root-filesystem partition (``image-src=/,id=N``);
``executable_paths`` are linked into ``/usr/bin`` after start, which needs the
(fakeroot) root user to write the overlay's ``/usr/bin``.
"""

from __future__ import annotations

import asyncio
import fcntl
import getpass
import json
import os
import re
import shlex
import shutil
import signal
import stat
import sys
import tempfile
import uuid
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from .base import ExecResult, Sandbox, _to_str
from .registry import register_sandbox

if TYPE_CHECKING:
    from .base import ImageMount, SandboxConfig


def _positive_timeout(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if value <= 0:
        raise ValueError(f"{name} must be a positive number of seconds, got {value!r}")
    return value


def _default_root() -> Path:
    """Node-local scratch root shared by this user's sandboxes (``$TMPDIR`` or ``/tmp``)."""
    return Path(tempfile.gettempdir()) / f"uni-agent-apptainer-{getpass.getuser()}"


def _image_uri(image: str) -> str:
    """``swebench/x`` -> ``docker://swebench/x:latest``; explicit schemes pass through.

    Untagged Docker references get ``:latest`` so ``x`` and ``x:latest`` share one SIF.
    """
    if "://" in image:
        return image
    name = image.split("@", 1)[0]
    if "@" not in image and ":" not in name.rsplit("/", 1)[-1]:
        image = f"{image}:latest"
    return f"docker://{image}"


def _sif_name(uri: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", uri).strip("_") + ".sif"


def _rmtree(path: Path) -> None:
    """``rm -rf`` that also clears overlay dirs left without write/search permission."""

    def _retry(func, target, _exc) -> None:
        for p in (Path(target).parent, Path(target)):
            try:
                p.chmod(p.stat().st_mode | stat.S_IRWXU)
            except OSError:
                pass
        if os.path.isdir(target) and not os.path.islink(target):
            shutil.rmtree(target, ignore_errors=True)
        else:
            func(target)

    if not path.exists():
        return
    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=_retry)
    else:
        shutil.rmtree(path, onerror=_retry)


def _unlock(fd: int) -> None:
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)


def _env_prelude(env: dict[str, str] | None, add_to_path: tuple[str, ...]) -> str:
    """Shell ``export`` lines for ``env`` and the PATH prefix (empty when neither is set).

    ``apptainer exec --env`` splits its value on commas, so values like
    ``NO_PROXY=a,b`` cannot go through it; exporting inside the container is exact.
    """
    lines = []
    if add_to_path:
        lines.append(f'export PATH={shlex.quote(":".join(add_to_path))}:"${{PATH:-}}"')
    for key, value in (env or {}).items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"invalid environment variable name: {key!r}")
        lines.append(f"export {key}={shlex.quote(value)}")
    return "\n".join(lines)


@register_sandbox("apptainer")
class ApptainerSandbox(Sandbox):
    """Run an isolated sandbox as a local ``apptainer instance`` (no daemon, no root)."""

    def __init__(
        self,
        *,
        image: str | None = None,
        apptainer_binary: str = "apptainer",
        sif_dir: str | None = None,
        cache_dir: str | None = None,
        work_root: str | None = None,
        pull_policy: str = "missing",
        pull_timeout: float | None = None,
        max_concurrent_pulls: int = 4,
        start_timeout: float | None = None,
        fakeroot: bool = True,
        contain: bool = True,
        binds: list[str] | None = None,
        add_to_path: list[str] | None = None,
        instance_args: list[str] | None = None,
        default_workdir: str = "/",
        image_mounts: list[ImageMount] | None = None,
        executable_paths: dict[str, str] | None = None,
    ) -> None:
        if not image:
            raise ValueError("apptainer sandbox requires an image (a .sif path or a docker:// / registry reference)")
        self.image = image
        self.image_mounts = list(image_mounts or [])
        for mount in self.image_mounts:
            if any(c in mount.mount_path for c in ":,"):  # separators in an apptainer --bind spec
                raise ValueError(f"apptainer image mount_path must not contain ':' or ',': {mount.mount_path!r}")
        self.executable_paths = dict(executable_paths or {})
        self.apptainer_binary = apptainer_binary
        root = _default_root()
        self.sif_dir = Path(sif_dir) if sif_dir else root / "sif"
        self.cache_dir = Path(cache_dir or os.environ.get("APPTAINER_CACHEDIR") or root / "cache")
        self.work_root = Path(work_root) if work_root else root / "instances"
        if pull_policy not in {"always", "missing", "never"}:
            raise ValueError("pull_policy must be one of: 'always', 'missing', 'never'")
        self.pull_policy = pull_policy
        self.pull_timeout = _positive_timeout("pull_timeout", pull_timeout)
        if max_concurrent_pulls < 1:
            raise ValueError(f"max_concurrent_pulls must be >= 1, got {max_concurrent_pulls!r}")
        self.max_concurrent_pulls = int(max_concurrent_pulls)
        self.start_timeout = _positive_timeout("start_timeout", start_timeout)
        self.fakeroot = fakeroot
        self.contain = contain
        self.binds = list(binds or [])
        if add_to_path is not None and not isinstance(add_to_path, list):
            raise ValueError("add_to_path must be a list of non-empty strings")
        self.add_to_path = tuple(add_to_path or [])
        if any(not isinstance(path, str) or not path for path in self.add_to_path):
            raise ValueError("add_to_path must be a list of non-empty strings")
        self.instance_args = list(instance_args or [])
        self.default_workdir = default_workdir
        self._instance_name: str | None = None
        self._instance_dir: Path | None = None

    @classmethod
    def from_config(cls, config: SandboxConfig) -> ApptainerSandbox:
        return cls(
            image=config.image,
            image_mounts=config.image_mounts,
            executable_paths=config.executable_paths,
            **config.sandbox_kwargs,
        )

    async def _run_apptainer(
        self,
        *args: str,
        timeout: float | None = None,
        stdin: bytes | None = None,
        env: dict[str, str] | None = None,
        output: Path | None = None,
    ) -> tuple[int, bytes, bytes]:
        """Run apptainer and return ``(exit_code, stdout, stderr)``.

        With ``output``, stdout+stderr go to that file instead of pipes and come back
        as ``stdout``. ``instance start`` needs this: the FUSE daemons it leaves
        running inherit the child's stdio, so pipes never reach EOF (and under uvloop
        ``communicate()`` then hangs until the instance stops).
        """
        out = open(output, "wb") if output is not None else None  # noqa: SIM115 - closed below
        try:
            proc = await asyncio.create_subprocess_exec(
                self.apptainer_binary,
                *args,
                stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
                stdout=out or asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT if out else asyncio.subprocess.PIPE,
                # Own process group so a timeout kills the command's children too.
                start_new_session=True,
                env=env,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"Apptainer executable {self.apptainer_binary!r} was not found") from exc
        finally:
            if out is not None:
                out.close()  # the child holds its own copy

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(stdin), timeout=timeout)
            if output is not None:
                stdout, stderr = output.read_bytes(), b""
        except BaseException:
            # Timeout *or* cancellation (e.g. SANDBOX_STARTUP_TIMEOUT around a long pull):
            # never leave the apptainer process running behind us.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await asyncio.shield(proc.wait())
            raise
        return int(proc.returncode or 0), stdout, stderr

    async def _run(
        self,
        *args: str,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
        output: Path | None = None,
    ) -> ExecResult:
        code, stdout, stderr = await self._run_apptainer(*args, timeout=timeout, env=env, output=output)
        return ExecResult(exit_code=code, stdout=_to_str(stdout), stderr=_to_str(stderr))

    # ----- image -----
    async def _resolve_sif(self, image: str) -> str:
        """Return a local SIF path for ``image``, pulling it into ``sif_dir`` per ``pull_policy``."""
        if image.endswith(".sif") and "://" not in image:
            if not Path(image).is_file():
                raise RuntimeError(f"Apptainer image {image!r} does not exist")
            return image

        uri = _image_uri(image)
        sif = self.sif_dir / _sif_name(uri)
        if self.pull_policy == "never":
            if not sif.is_file():
                raise RuntimeError(f"Apptainer image {uri!r} is not cached at {sif} and pull_policy='never'")
            return str(sif)
        if self.pull_policy == "missing" and sif.is_file():
            return str(sif)

        # Serialize pulls of the same image across coroutines, actors and processes.
        self.sif_dir.mkdir(parents=True, exist_ok=True)
        lock_fd = await self._lock(sif.with_suffix(".lock"))
        try:
            if self.pull_policy == "missing" and sif.is_file():
                return str(sif)
            # Bound concurrent pulls node-wide: dozens at once only thrash the network/CPU.
            slots = [self.sif_dir / f".pull-slot-{i}.lock" for i in range(self.max_concurrent_pulls)]
            slot_fd = await self._lock(*slots)
            try:
                await self._pull(uri, sif)
            finally:
                _unlock(slot_fd)
        finally:
            _unlock(lock_fd)
        return str(sif)

    @staticmethod
    async def _lock(*paths: Path) -> int:
        """Take an exclusive ``flock`` on any one of ``paths``, polling until one is free.

        Polling (rather than a blocking ``flock`` in a worker thread) keeps cancellation
        clean: a blocked thread would keep waiting after the caller is cancelled, then
        acquire a lock nobody releases and wedge every later pull.
        """
        while True:
            for path in paths:
                fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return fd
                except BlockingIOError:
                    os.close(fd)
                except BaseException:
                    os.close(fd)
                    raise
            await asyncio.sleep(1.0)

    async def _pull(self, uri: str, sif: Path) -> None:
        tmp = sif.with_name(f".{sif.name}.{uuid.uuid4().hex[:8]}.tmp")
        cache = self.cache_dir / f"pull-{uuid.uuid4().hex[:12]}"
        try:
            try:
                cache.mkdir(parents=True)
                env = {**os.environ, "APPTAINER_CACHEDIR": str(cache)}
                pulled = await self._run("pull", "--force", str(tmp), uri, timeout=self.pull_timeout, env=env)
            except asyncio.TimeoutError as exc:
                raise TimeoutError(
                    f"Pulling Apptainer image {uri!r} exceeded pull_timeout={self.pull_timeout:g}s"
                ) from exc
            if pulled.exit_code != 0:
                detail = pulled.stderr.strip() or pulled.stdout.strip()
                raise RuntimeError(f"Failed to pull Apptainer image {uri!r}: {detail}")
            os.replace(tmp, sif)
        finally:
            tmp.unlink(missing_ok=True)
            await asyncio.shield(asyncio.to_thread(_rmtree, cache))

    async def _image_mount_bind(self, mount: ImageMount) -> str:
        """``--bind`` spec mounting ``mount.image``'s root filesystem at ``mount.mount_path``.

        A SIF bind defaults to the image's *data* partition, which a pulled image does
        not have; ``id=`` selects its primary system partition (the root filesystem).
        """
        sif = await self._resolve_sif(mount.image)
        if any(c in sif for c in ":,"):
            raise ValueError(f"apptainer cannot bind a SIF whose path contains ':' or ',': {sif!r}")
        listed = await self._run("sif", "list", sif, timeout=60.0)
        if listed.exit_code == 0:
            for line in listed.stdout.splitlines():
                cols = [col.strip() for col in line.split("|")]
                # ID |GROUP |LINK |SIF POSITION |TYPE, e.g. "4 |1 |NONE |36864-2256896 |FS (Squashfs/*System/amd64)"
                if len(cols) == 5 and cols[0].isdigit() and "*System" in cols[4]:
                    return f"{sif}:{mount.mount_path}:image-src=/,id={cols[0]}"
        detail = listed.stderr.strip() or listed.stdout.strip()
        raise RuntimeError(f"no root filesystem partition found in {sif} (image {mount.image!r}): {detail}")

    # ----- lifecycle -----
    async def start(self) -> None:
        if self._instance_name is not None:
            return
        sif = await self._resolve_sif(self.image)
        image_binds = [await self._image_mount_bind(mount) for mount in self.image_mounts]

        name = f"uni-agent-{uuid.uuid4().hex[:12]}"
        instance_dir = self.work_root / name
        overlay, workdir = instance_dir / "overlay", instance_dir / "workdir"
        overlay.mkdir(parents=True)
        workdir.mkdir()
        # Set before launching so a failed/timed-out start is still cleaned up by stop().
        self._instance_name, self._instance_dir = name, instance_dir

        args = ["instance", "start", "--cleanenv", "--overlay", str(overlay)]
        if self.fakeroot:
            args.append("--fakeroot")
        if self.contain:
            args.extend(["--contain", "--workdir", str(workdir)])
        for bind in [*image_binds, *self.binds]:
            args.extend(["--bind", bind])
        args.extend(self.instance_args)
        args.extend([sif, name])

        try:
            started = await self._run(*args, timeout=self.start_timeout, output=instance_dir / "start.log")
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                f"Starting Apptainer sandbox from {self.image!r} exceeded start_timeout={self.start_timeout:g}s"
            ) from exc
        if started.exit_code != 0:
            detail = started.stderr.strip() or started.stdout.strip()
            raise RuntimeError(f"Failed to start Apptainer sandbox from {self.image!r}: {detail}")
        await self._setup_executable_paths(self.executable_paths)

    async def stop(self) -> None:
        name, self._instance_name = self._instance_name, None
        instance_dir, self._instance_dir = self._instance_dir, None
        if name is not None:
            try:
                await self._run("instance", "stop", name, timeout=60.0)
            except asyncio.TimeoutError:
                await self._run("instance", "stop", "--force", name, timeout=30.0)
        if instance_dir is not None:
            await asyncio.to_thread(_rmtree, instance_dir)

    def _require_instance(self) -> str:
        if self._instance_name is None:
            raise RuntimeError("ApptainerSandbox not started; call start() first")
        return self._instance_name

    async def is_alive(self) -> bool:
        if self._instance_name is None:
            return False
        try:
            result = await self._run("instance", "list", "--json", self._instance_name, timeout=10.0)
            return result.exit_code == 0 and bool(json.loads(result.stdout).get("instances"))
        except Exception:
            return False

    # ----- data plane -----
    def _exec_args(self, argv: list[str], workdir: str | None, env: dict[str, str] | None) -> list[str]:
        args = ["-q", "exec", "--cleanenv", "--cwd", workdir or self.default_workdir]
        args.append(f"instance://{self._require_instance()}")
        prelude = _env_prelude(env, self.add_to_path)
        if prelude:
            args.extend(["/bin/sh", "-c", f'{prelude}\nexec "$@"', "sh"])
        args.extend(argv)
        return args

    async def _exec(
        self,
        argv: list[str],
        *,
        timeout: float | None = None,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        return await self._run(*self._exec_args(argv, workdir, env), timeout=timeout)

    async def read_file(self, path: str) -> bytes:
        """Stream the raw bytes over stdout (no base64 or argv size limits)."""
        code, stdout, stderr = await self._run_apptainer(*self._exec_args(["cat", "--", path], None, None))
        if code != 0:
            raise RuntimeError(f"read_file {path!r} failed: {_to_str(stderr).strip()}")
        return stdout

    async def write_file(self, path: str, content: bytes | str) -> None:
        """Stream the raw bytes over stdin (no base64 or argv size limits)."""
        data = content.encode("utf-8") if isinstance(content, str) else content
        q = shlex.quote(path)
        parent = str(PurePosixPath(path).parent)
        script = f"mkdir -p {shlex.quote(parent)} && cat > {q}"
        code, _, stderr = await self._run_apptainer(*self._exec_args(["sh", "-c", script], None, None), stdin=data)
        if code != 0:
            raise RuntimeError(f"write_file {path!r} failed: {_to_str(stderr).strip()}")
