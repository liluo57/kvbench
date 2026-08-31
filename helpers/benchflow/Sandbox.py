"""Sandbox isolation strategies for the BenchflowHelper.

The Helper owns the orchestration; the sandboxes own "where the agent /
verifier run." The :class:`BenchflowHelper` sets ``sandbox._helper`` after
construction so the sandbox can read paths / config without re-deriving
them, and from then on :meth:`_ResolveHelper` is a one-liner.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Mapping, Optional, Sequence, Tuple

from .Util import ResolveHostSitePackages

if TYPE_CHECKING:
    from .helper import BenchflowHelper


class _HelperBound:
    """Mixin: every sandbox can read paths / config from its owning helper.

    The Helper sets ``self._helper`` in :meth:`BenchflowHelper._StartSandboxAndAgent`
    (the import would otherwise be a cycle). Every sandbox picks up a
    back-reference and uses it to derive ``outputDir``, the SkillsBench task
    directory, and the per-case workspaces.
    """

    _helper: "BenchflowHelper"  # set by the owning helper

    def _ResolveHelper(self) -> "BenchflowHelper":
        return self._helper

    def _MakeWorkspace(self, kind: str) -> Path:
        helper = self._ResolveHelper()
        workspace = helper.outputDir / f"{kind}_workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        return workspace


class Sandbox(_HelperBound, ABC):
    """Where the agent and verifier run. SkillsBench semantics require the
    task's Dockerfile container; LocalSandbox is the non-isolated fallback.
    """

    @abstractmethod
    def prepare(self, task_id: str, task_dir: Path) -> None:
        """One-time setup (image build, dir create). May be a no-op."""

    @abstractmethod
    def run_agent(
        self,
        *,
        task_id: str,
        cmd: Sequence[str],
        env: Mapping[str, str],
        log_path: Path,
    ) -> int:
        """Run the agent. Returns the exit code when finished (blocking)."""

    @abstractmethod
    def run_verifier(
        self,
        *,
        task_id: str,
        verifier_test_sh: Path,
        log_path: Path,
    ) -> Tuple[int, Optional[Path]]:
        """Run ``bash verifier/test.sh`` and report the reward-file path.

        Returns ``(exit_code, reward_file_path)``; the reward file lives at
        ``/logs/verifier/reward.txt`` inside the sandbox.
        """

    @abstractmethod
    def cleanup(self) -> None:
        """Tear down any container / tempdirs. Idempotent."""


class LocalSandbox(Sandbox):
    """No isolation — runs everything on the host.

    The SkillsBench verifier expects ``/logs/verifier/reward.txt`` (root-only)
    and other in-container paths; LocalSandbox will fail those checks. Use
    this only for smoke-testing the HTTP plumbing.
    """

    def __init__(self, work_dir: Path):
        self.workDir = Path(work_dir)
        self.workDir.mkdir(parents=True, exist_ok=True)

    def prepare(self, task_id: str, task_dir: Path) -> None:
        return

    def run_agent(
        self,
        *,
        task_id: str,
        cmd: Sequence[str],
        env: Mapping[str, str],
        log_path: Path,
    ) -> int:
        logFile = open(log_path, "w", encoding="utf-8")
        try:
            proc = subprocess.Popen(
                list(cmd),
                stdout=logFile,
                stderr=subprocess.STDOUT,
                env=dict(env),
                cwd=str(self.workDir),
            )
            return proc.wait()
        finally:
            logFile.close()

    def run_verifier(
        self,
        *,
        task_id: str,
        verifier_test_sh: Path,
        log_path: Path,
    ) -> Tuple[int, Optional[Path]]:
        logFile = open(log_path, "w", encoding="utf-8")
        try:
            proc = subprocess.run(
                ["bash", str(verifier_test_sh)],
                stdout=logFile,
                stderr=subprocess.STDOUT,
                cwd=str(verifier_test_sh.parent),
                check=False,
            )
        finally:
            logFile.close()
        # The SkillsBench verifier writes its reward at /logs/verifier/reward.txt.
        # LocalSandbox usually can't create that path; treat absence as 0.0.
        rewardPath = Path("/logs/verifier/reward.txt")
        return proc.returncode, (rewardPath if rewardPath.is_file() else None)

    def cleanup(self) -> None:
        return


class DockerSandbox(Sandbox):
    """Build the task's Dockerfile and run agent + verifier inside containers.

    Wiring:

    - The image is built once per task id, tagged ``kvbench-skillsbench:<id>``.
      We copy the task's ``environment/Dockerfile`` into a temp build context
      and append a final ``RUN pip install mini-swe-agent`` layer so the
      upstream SkillsBench Dockerfile stays untouched.
    - **Agent run**: mount ``tasks/<id>/task.md -> /root/task.md`` and the
      output-dir ``agent_workspace`` -> ``/root`` so the agent's writes
      (notably ``/root/answer.json``) persist for the verifier. Run with
      ``--network host`` so the agent's LLM client can reach our kvbench
      endpoint on ``127.0.0.1``.
    - **Verifier run**: mount ``tasks/<id>/verifier -> /verifier`` and a
      fresh ``verifier_workspace`` -> ``/logs`` so the verifier can write
      ``/logs/verifier/reward.txt`` and we can read it back from the host.
    """

    _IMAGE_TAG_PREFIX = "kvbench-skillsbench"

    def __init__(
        self,
        *,
        network: str = "host",
        install_mini_swe: bool = True,
        extra_docker_run_args: Sequence[str] = (),
    ):
        if not shutil.which("docker"):
            raise RuntimeError(
                "DockerSandbox requires the `docker` CLI in PATH "
                "(install Docker Desktop / docker-ce first)"
            )
        self.network = network
        self.installMiniSwe = install_mini_swe
        self.extraDockerRunArgs = list(extra_docker_run_args)
        self._imageTag: Optional[str] = None
        self._agentWorkspace: Optional[Path] = None
        self._verifierWorkspace: Optional[Path] = None
        self._activeContainers: List[str] = []

    # ----------------------------------------------------------- public API
    def prepare(self, task_id: str, task_dir: Path) -> None:
        """Build the task image (idempotent: ``docker build`` is cached)."""
        dockerfileDir = task_dir / "environment"
        dockerfile = dockerfileDir / "Dockerfile"
        if not dockerfile.is_file():
            raise FileNotFoundError(
                f"SkillsBench task {task_id!r} has no environment/Dockerfile at "
                f"{dockerfile}"
            )
        self._imageTag = f"{self._IMAGE_TAG_PREFIX}:{task_id}"

        with tempfile.TemporaryDirectory(prefix=f"kvbench-sb-{task_id}-") as tmp:
            tmpPath = Path(tmp)
            # Build context: symlink every file in environment/ into the
            # tempdir so COPY directives like ``COPY test.bib /root/test.bib``
            # resolve correctly.
            for entry in dockerfileDir.iterdir():
                target = tmpPath / entry.name
                if entry.is_file():
                    target.symlink_to(entry.resolve())

            # Append our extra layer to the end of the Dockerfile (without
            # modifying the upstream file).
            with open(tmpPath / "Dockerfile", "a", encoding="utf-8") as f:
                f.write("\n# === appended by kvbench AgentBenchFlowTask ===\n")
                if self.installMiniSwe:
                    f.write(
                        "RUN pip install --break-system-packages --quiet "
                        "mini-swe-agent\n"
                    )

            subprocess.run(
                ["docker", "build", "-t", self._imageTag, str(tmpPath)],
                check=True,
            )

    def run_agent(
        self,
        *,
        task_id: str,
        cmd: Sequence[str],
        env: Mapping[str, str],
        log_path: Path,
    ) -> int:
        """Run the agent in a fresh container; return its exit code."""
        assert self._imageTag is not None

        taskDir = self._TaskDir(task_id)
        taskMdPath = taskDir / "task.md"
        if not taskMdPath.is_file():
            raise FileNotFoundError(f"missing {taskMdPath}")

        # The agent's ``/root`` lives in this dir so its writes (notably
        # ``/root/answer.json``) survive across the agent->verifier handoff.
        self._agentWorkspace = self._agentWorkspace or self._MakeWorkspace("agent")
        containerName = f"kvbench-sb-{task_id}-{int(time.time())}"
        envArgs: List[str] = []
        for key, value in env.items():
            envArgs.extend(["-e", f"{key}={value}"])

        with open(log_path, "w", encoding="utf-8") as logFile:
            dockerArgs = [
                "docker", "run", "--rm",
                "--name", containerName,
                "--network", self.network,
                "-v", f"{self._agentWorkspace}:/root",
                # Mount the task after /root: the later file mount must not
                # be hidden by the writable agent workspace mount.
                "-v", f"{taskMdPath.resolve()}:/root/task.md:ro",
                *self._SkillDockerMounts(taskDir),
                *self.extraDockerRunArgs,
                *envArgs,
                self._imageTag,
                *cmd,
            ]
            self._activeContainers.append(containerName)
            try:
                proc = subprocess.run(
                    dockerArgs,
                    stdout=logFile,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                return proc.returncode
            finally:
                self._CleanupContainer(containerName)

    def run_verifier(
        self,
        *,
        task_id: str,
        verifier_test_sh: Path,
        log_path: Path,
    ) -> Tuple[int, Optional[Path]]:
        """Run the SkillsBench verifier inside a sibling container."""
        assert self._imageTag is not None

        taskDir = self._TaskDir(task_id)
        # The agent ran first; agentWorkspace persists. verifierWorkspace
        # captures ``/logs/verifier/reward.txt`` for us to read on the host.
        self._agentWorkspace = self._agentWorkspace or self._MakeWorkspace("agent")
        self._verifierWorkspace = self._MakeWorkspace("verifier")
        containerName = f"kvbench-sb-verifier-{task_id}-{int(time.time())}"
        with open(log_path, "w", encoding="utf-8") as logFile:
            dockerArgs = [
                "docker", "run", "--rm",
                "--name", containerName,
                "--network", self.network,
                "-v", f"{self._verifierWorkspace}:/logs",
                "-v", f"{self._agentWorkspace}:/root:ro",
                "-v", f"{taskDir}/verifier:/verifier:ro",
                *self.extraDockerRunArgs,
                self._imageTag,
                "bash", str(verifier_test_sh),
            ]
            self._activeContainers.append(containerName)
            try:
                proc = subprocess.run(
                    dockerArgs,
                    stdout=logFile,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            finally:
                self._CleanupContainer(containerName)
            rewardPath = self._verifierWorkspace / "verifier" / "reward.txt"
            if not rewardPath.is_file():
                rewardPath = self._verifierWorkspace / "reward.txt"
            return proc.returncode, (rewardPath if rewardPath.is_file() else None)

    def cleanup(self) -> None:
        for name in list(self._activeContainers):
            self._CleanupContainer(name)

    # ---------------------------------------------------------- internals
    def _TaskDir(self, task_id: str) -> Path:
        """Resolve the SkillsBench task directory from outside the sandbox.

        Mirrors ``BenchflowHelper.skillsbenchDir / tasks / task_id``.
        """
        helper = self._ResolveHelper()
        return helper.skillsbenchDir / "tasks" / task_id

    def _SkillDockerMounts(self, task_dir: Path) -> List[str]:
        """Return mounts for the task's runtime skills, if it has any."""
        skillsDir = task_dir / "environment" / "skills"
        if not skillsDir.is_dir() or self._agentWorkspace is None:
            return []
        mounts: List[str] = []
        # These are the two conventional locations used by SkillsBench
        # agents.  Keeping both makes the portable skills usable by the
        # mini-SWE adapter even when a SKILL.md contains a home-relative path.
        for relative in (".agents/skills", ".claude/skills"):
            (self._agentWorkspace / relative).mkdir(parents=True, exist_ok=True)
            mounts.extend([
                "-v",
                f"{skillsDir.resolve()}:/root/{relative}:ro",
            ])
        return mounts

    def _CleanupContainer(self, name: str) -> None:
        if name in self._activeContainers:
            self._activeContainers.remove(name)
        subprocess.run(
            ["docker", "rm", "-f", name],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


class ApptainerSandbox(Sandbox):
    """Rootless container via Apptainer (no Docker / no root / no daemon).

    Why this design instead of "build the task's Dockerfile":

    - ``apptainer build`` from an arbitrary Dockerfile needs ``newuidmap`` /
      ``newgidmap`` / ``fuse-overlayfs``, none of which are installed in the
      conda env this helper targets. It is also unreliable for arbitrary
      RUN commands (apt-get, pip install, COPY) without root.
    - ``apptainer pull docker://<base>`` IS rootless; the OCI image is
      converted to a SIF transparently. We use that as the "image" for the
      agent and verifier.

    So ``prepare`` parses the first ``FROM`` line of
    ``environment/Dockerfile``, pulls that image, and stores the SIF path.
    The Dockerfile's RUN / COPY steps are intentionally not applied; any
    in-container dependency the task expects must already be in the base
    image. (For citation-check the FROM is ``ubuntu:24.04`` which has no
    Python, so the agent and verifier will both fail unless an
    ``image_override`` is supplied — this is the documented limitation.)

    Networking: ``--network host`` and ``--net`` both require root or a
    suid apptainer with /etc/subuid, neither of which is available here.
    With NO ``--network`` flag, the container shares the host network
    namespace by default (verified on the conda-forge 1.5.3 build) -- which
    is what we want, since the agent reaches the helper via
    ``127.0.0.1:<port>``.

    The helper also bind-mounts the host toolchain (``<output>/host_tools``)
    into the container at ``/host_tools`` and prepends it to PATH so
    ``mini-swe-agent`` resolves to the host install (the base image may
    not ship it).

    Cleanup deletes the pulled SIF on shutdown.
    """

    _SIF_PREFIX = "kvbench-skillsbench-sif"

    def __init__(
        self,
        *,
        apptainer_path: str = "apptainer",
        image_override: Optional[str] = None,
        cleanup_image: bool = True,
        # Rootless caveat: ``--network host`` and ``--net`` both require
        # root or a suid apptainer with /etc/subuid. The conda-forge build
        # supports neither. With NO --network flag, the container shares
        # the host network namespaces by default — which is what we want,
        # since the agent reaches the helper via 127.0.0.1:<port>. Pass an
        # empty string to skip the flag entirely; pass "none" to drop networking.
        network_mode: str = "",
        keep_image: bool = False,
        # ``--writable-tmpfs`` gives the in-container rootfs an in-memory
        # overlay (~64 MiB) without needing ``--fakeroot``. Without it, any
        # attempt by the agent or verifier to mutate the rootfs (apt install,
        # pip install, write to ``/root/.config``) fails silently with
        # ``Read-only file system``.
        writable_tmpfs: bool = True,
    ):
        apBin = shutil.which(apptainer_path)
        if not apBin:
            raise RuntimeError(
                "ApptainerSandbox requires the `apptainer` CLI in PATH "
                "(install apptainer / singularity first)"
            )
        self.apptainerPath = apBin
        self.imageOverride = image_override
        self.cleanupImage = bool(cleanup_image)
        self.networkMode = network_mode
        self.keepImage = bool(keep_image)
        self.writableTmpfs = bool(writable_tmpfs)

        self._sifPath: Optional[Path] = None
        self._resolvedImageURI: Optional[str] = None
        self._agentWorkspace: Optional[Path] = None
        self._verifierWorkspace: Optional[Path] = None
        self._hostTools: Optional[Path] = None
        self._hostSitePackages: Optional[Path] = None
        self._hostPythonRoot: Optional[Path] = None

    # ----------------------------------------------------------- public API
    def prepare(self, task_id: str, task_dir: Path) -> None:
        """Pull the base image as a SIF and stage host-side workspaces."""
        helper = self._ResolveHelper()
        # Shared asset dirs alongside the agent / verifier workspaces.
        self._agentWorkspace = self._MakeWorkspace("agent")
        self._verifierWorkspace = self._MakeWorkspace("verifier")
        self._hostTools = helper.outputDir / "host_tools"
        self._hostTools.mkdir(parents=True, exist_ok=True)
        # Resolve the host's site-packages so we can bind-mount it into
        # the SIF. Without this, mini-swe-agent (installed only on the
        # host) can't import its dependencies from inside the container.
        self._hostSitePackages = ResolveHostSitePackages()
        self._hostPythonRoot = Path(sys.executable).resolve().parent.parent
        self._StageHostTools(self._hostTools)
        self._StageEnvironmentInputs(task_dir)
        # Dockerfile COPY/RUN steps are not applied when we use a rootless
        # base-image SIF.  Keep the conventional paths writable/persistent so
        # generated task inputs and agent outputs are visible to both phases.
        (self._agentWorkspace / "data").mkdir(parents=True, exist_ok=True)
        (self._agentWorkspace / "output").mkdir(parents=True, exist_ok=True)
        envBin = self._agentWorkspace / ".local" / "bin"
        envBin.mkdir(parents=True, exist_ok=True)
        envFile = envBin / "env"
        if not envFile.exists():
            envFile.write_text(
                '#!/bin/sh\nexport PATH="/host_tools:$PATH"\n',
                encoding="utf-8",
            )

        # Resolve the image URI: override wins; else first FROM line.
        if self.imageOverride:
            imageURI = self.imageOverride
        else:
            imageURI = self._ResolveFromImage(task_dir)
        self._resolvedImageURI = imageURI

        # Pin SIF under the task workspace so cleanup can find it.
        taskWorkspace = helper.outputDir / "sif"
        taskWorkspace.mkdir(parents=True, exist_ok=True)
        self._sifPath = taskWorkspace / f"{self._SIF_PREFIX}-{task_id}.sif"

        # A completed SIF is immutable for the selected image and can be
        # reused by pair retries. Re-pulling with ``--disable-cache`` made a
        # transient registry failure turn a recoverable rollout into a second
        # failure.
        imageMarker = self._sifPath.with_suffix(".image")
        if self._sifPath.is_file():
            try:
                if imageMarker.read_text(encoding="utf-8").strip() == imageURI:
                    return
            except OSError:
                pass

        pullProc = subprocess.run(
            [self.apptainerPath, "pull", str(self._sifPath), imageURI],
            check=False,
        )
        if pullProc.returncode != 0 or not self._sifPath.is_file():
            raise RuntimeError(
                f"apptainer pull failed for {imageURI} (exit "
                f"{pullProc.returncode}); see logs above"
            )
        try:
            imageMarker.write_text(imageURI, encoding="utf-8")
        except OSError:
            pass

    def run_agent(
        self,
        *,
        task_id: str,
        cmd: Sequence[str],
        env: Mapping[str, str],
        log_path: Path,
    ) -> int:
        assert self._sifPath is not None
        helper = self._ResolveHelper()
        apArgs = self._ComposeApptainerArgs(
            task_id=task_id,
            extra_binds=self._AgentBinds(helper),
        )
        # Reproduce the environment image's data-generation RUN step.  The
        # generator uses the Dockerfile's absolute paths, which are provided
        # by the binds below; failures are logged but must not prevent the
        # agent from starting (some tasks ship static inputs only).
        generate = self._agentWorkspace / "generate_data.py"
        if generate.is_file():
            prepCmd = [self.apptainerPath, "exec", *apArgs, str(self._sifPath),
                       "python3", "/root/generate_data.py"]
            subprocess.run(prepCmd, env={**os.environ, **env}, check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        fullCmd = [self.apptainerPath, "exec", *apArgs, str(self._sifPath), *cmd]
        with open(log_path, "w", encoding="utf-8") as logFile:
            proc = subprocess.run(
                fullCmd,
                stdout=logFile,
                stderr=subprocess.STDOUT,
                env={**os.environ, **env},
                check=False,
            )
            return proc.returncode

    def run_verifier(
        self,
        *,
        task_id: str,
        verifier_test_sh: Path,
        log_path: Path,
    ) -> Tuple[int, Optional[Path]]:
        assert self._sifPath is not None
        helper = self._ResolveHelper()
        apArgs = self._ComposeApptainerArgs(
            task_id=task_id,
            extra_binds=self._VerifierBinds(helper, verifier_test_sh),
        )
        fullCmd = [
            self.apptainerPath, "exec", *apArgs, str(self._sifPath),
            "bash", str(verifier_test_sh),
        ]
        with open(log_path, "w", encoding="utf-8") as logFile:
            proc = subprocess.run(
                fullCmd,
                stdout=logFile,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
                check=False,
            )
        rewardPath = self._verifierWorkspace / "verifier" / "reward.txt"
        if not rewardPath.is_file():
            rewardPath = self._verifierWorkspace / "reward.txt"
        return proc.returncode, (rewardPath if rewardPath.is_file() else None)

    def cleanup(self) -> None:
        if self._sifPath and self._sifPath.is_file() and self.cleanupImage:
            try:
                self._sifPath.unlink()
            except OSError:
                pass
            marker = self._sifPath.with_suffix(".image") if self._sifPath else None
            if marker and marker.is_file():
                try:
                    marker.unlink()
                except OSError:
                    pass

    # ---------------------------------------------------------- internals
    def _StageEnvironmentInputs(self, task_dir: Path) -> None:
        """Copy the task's environment files into ``agent_workspace``.

        SkillsBench tasks may carry build / runtime inputs that are
        referenced via Dockerfile ``COPY`` directives. Rootless apptainer
        cannot apply those COPY steps at build time, so we mirror every
        file and directory under ``environment/`` into the agent workspace
        (the same trick the Docker sandbox gets for free). The agent
        then sees its inputs at the same in-container paths the upstream
        Dockerfile promised.
        """
        if self._agentWorkspace is None:
            return
        envDir = task_dir / "environment"
        if not envDir.is_dir():
            return
        for entry in envDir.iterdir():
            if entry.name == "Dockerfile":
                # Don't ship the Dockerfile into the agent's writable workspace.
                continue
            target = self._agentWorkspace / entry.name
            if entry.is_dir():
                if target.exists():
                    continue
                try:
                    shutil.copytree(entry, target, symlinks=True)
                except OSError:
                    pass
            else:
                if target.exists():
                    continue
                try:
                    if entry.is_symlink():
                        target.symlink_to(entry.resolve())
                    else:
                        shutil.copy2(entry, target)
                except OSError:
                    pass

    def _AgentBinds(self, helper: "BenchflowHelper") -> List[str]:
        """``--bind`` mounts needed by the agent invocation."""
        assert self._agentWorkspace is not None and self._hostTools is not None
        # /logs is what the agent + verifier write reward.txt under; we
        # point it at the agent workspace so a stray write survives.
        binds = [
            f"{self._agentWorkspace.resolve()}:/root",
            f"{self._agentWorkspace.resolve()}:/logs",
            f"{(self._agentWorkspace / 'data').resolve()}:/app/data",
            f"{(self._agentWorkspace / 'output').resolve()}:/app/output",
            f"{(helper.skillsbenchDir / 'tasks' / helper.taskId / 'environment' / 'data').resolve()}:/app/environment/data:ro",
            f"{(helper.skillsbenchDir / 'tasks' / helper.taskId / 'environment' / 'generate_data.py').resolve()}:/app/environment/generate_data.py:ro",
            f"{self._hostTools.resolve()}:/host_tools",
            f"{helper.skillsbenchDir.resolve()}/tasks/{helper.taskId}:/root/tasks_dir:ro",
        ]
        # ``/host_lib`` carries the host's site-packages (for the
        # ``mini-swe-agent`` import that lives only on the host).
        if self._hostSitePackages is not None:
            binds.append(f"{self._hostSitePackages.resolve()}:/host_lib")
        return binds

    def _VerifierBinds(
        self, helper: "BenchflowHelper", verifier_test_sh: Path
    ) -> List[str]:
        """``--bind`` mounts needed by the verifier invocation."""
        assert self._verifierWorkspace is not None and self._hostTools is not None
        # The verifier expects ``/logs/verifier/reward.txt``; capture it
        # to ``verifierWorkspace`` so we can read it back from the host.
        binds = [
            f"{self._verifierWorkspace.resolve()}:/logs",
            f"{self._agentWorkspace.resolve() if self._agentWorkspace else self._verifierWorkspace}:/root:ro",
            f"{(self._agentWorkspace / 'output').resolve() if self._agentWorkspace else self._verifierWorkspace}:/app/output:ro",
            f"{verifier_test_sh.parent.resolve()}:/verifier:ro",
            f"{self._hostTools.resolve()}:/host_tools",
        ]
        if self._hostSitePackages is not None:
            binds.append(f"{self._hostSitePackages.resolve()}:/host_lib")
        return binds

    def _ComposeApptainerArgs(
        self, *, task_id: str, extra_binds: List[str]
    ) -> List[str]:
        """Compose the ``apptainer exec`` flags this run needs."""
        apArgs: List[str] = []
        if self.networkMode:
            apArgs.extend(["--network", self.networkMode])
        for bind in extra_binds:
            apArgs.extend(["--bind", bind])
        if self.writableTmpfs:
            apArgs.append("--writable-tmpfs")
        # Prepend ``/host_tools`` to PATH so the staged ``mini-swe-agent``
        # launcher (no extension, see :meth:`_StageHostTools`) is found when
        # the agent subprocess invokes ``mini-swe-agent`` by name. The
        # container image (python:3.13-slim by default) doesn't ship
        # ``mini-swe-agent`` itself — only the launcher we stage into
        # ``/host_tools`` resolves it.
        apArgs.extend(["--env", "PATH=/host_tools:/usr/local/bin:/usr/bin:/bin"])
        # PYTHONPATH for the host's site-packages (mini-swe-agent imports).
        # Bind-mount the host's site-packages directly into the container at
        # ``/host_lib``. Earlier revisions symlinked ``_lib`` under
        # ``/host_tools``, but apptainer preserves the symlink target path
        # verbatim — pointing at a host filesystem path that doesn't exist
        # inside the sandbox — so Python in the container still couldn't
        # find ``minisweagent``. A direct bind-mount makes
        # ``/host_lib``/``minisweagent`` resolvable.
        if self._hostSitePackages is not None:
            apArgs.extend(["--env", "PYTHONPATH=/host_lib"])
            apArgs.extend(["--bind", f"{self._hostSitePackages.resolve()}:/host_lib"])
        # The upstream verifier installs uv/pytest through /root, which is a
        # read-only agent handoff mount.  Keep caches in the writable tmpfs and
        # make the pre-staged env script discoverable.
        apArgs.extend(["--env", "HOME=/root", "--env", "UV_CACHE_DIR=/tmp/uv-cache"])
        # Honor the user's environment if it asks for unshare.
        if os.environ.get("KVBENCH_APT_USE_UNSHARE", "0") == "1":
            return ["unshare", "-U", "-r", "--map-root-user", *apArgs]
        return apArgs

    def _ResolveFromImage(task_dir: Path) -> str:
        """Parse the first ``FROM <image>[:tag]`` from environment/Dockerfile.

        Returns a ``docker://<image>[:tag]`` URI for ``apptainer pull``.
        Raises if the Dockerfile is missing or has no FROM line.
        """
        dockerfile = task_dir / "environment" / "Dockerfile"
        if not dockerfile.is_file():
            raise FileNotFoundError(
                f"SkillsBench task has no environment/Dockerfile at {dockerfile}"
            )
        for rawLine in dockerfile.read_text(encoding="utf-8").splitlines():
            line = rawLine.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.split()
            if not tokens or tokens[0].upper() != "FROM":
                continue
            if len(tokens) < 2:
                raise RuntimeError(
                    f"Dockerfile FROM line has no image: {rawLine!r}"
                )
            image = tokens[1]
            # Strip stage-AS alias (``FROM ubuntu AS base``).
            for sep in (" AS ", " as "):
                if sep in image:
                    image = image.split(sep)[0]
                    break
            return f"docker://{image}"
        raise RuntimeError(
            f"No FROM line found in {dockerfile}; cannot resolve base image"
        )

    def _StageHostTools(self, hostToolsDir: Path) -> None:
        """Stage host binaries the agent needs.

        The container's base image is expected to ship a ``python3`` (use
        ``image_override="docker://continuumio/miniconda3:latest"`` or any
        other image that already has Python 3.x). The launcher invokes
        ``python3`` (resolved via the container's PATH) on the staged
        script.

        In addition to the script, the host's ``site-packages`` is exposed
        at ``/host_tools/_lib`` and prepended to ``PYTHONPATH`` so the
        agent can import ``minisweagent`` (installed on the host, not in
        the container's base image).
        """
        msaScript = shutil.which("mini-swe-agent")
        baseName = os.path.basename(msaScript) if msaScript else ""

        # Stage the script under a non-conflicting name so the launcher
        # (which gets the canonical ``baseName``) can call it via
        # ``python3``.
        if msaScript:
            stagedScript = hostToolsDir / f"{baseName}.py"
            try:
                stagedScript.write_text(
                    Path(msaScript).read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
            except OSError:
                msaScript = None
            if msaScript:
                try:
                    stagedScript.chmod(0o755)
                except OSError:
                    pass

        # Stage the host's site-packages dir at ``_lib`` so Python inside
        # the container can import minisweagent + its deps. Symlink, not
        # copy, so we don't bloat output dirs.
        hostSitePackages = ResolveHostSitePackages()
        if hostSitePackages is not None:
            libDir = hostToolsDir / "_lib"
            try:
                if libDir.is_symlink() or libDir.exists():
                    libDir.unlink()
                libDir.symlink_to(hostSitePackages.resolve())
            except OSError:
                pass

        # Launcher at the canonical name: prepended to PATH inside the
        # container so ``mini-swe-agent`` resolves to the launcher, which
        # invokes ``python3`` on the staged script. PYTHONPATH points at
        # /host_lib (bound directly from the host's site-packages dir).
        # Use ``python3`` (PATH-resolved) instead of ``/host_conda/bin/python``
        # — the container image (python:3.13-slim by default) doesn't ship
        # ``/host_conda``, so the hard-coded path fails.
        if msaScript:
            launcher = hostToolsDir / baseName
            inContainerScript = f"/host_tools/{baseName}.py"
            launcher.write_text(
                f"#!/bin/sh\nexec env PYTHONPATH=/host_lib python3 "
                f"{inContainerScript} \"$@\"\n",
                encoding="utf-8",
            )
            try:
                launcher.chmod(0o755)
            except OSError:
                pass

        # Rootless task verifiers commonly bootstrap with curl + uvx.  Those
        # tools are not present in python:*-slim, so stage host copies when
        # available; this avoids privileged apt/pip operations in the SIF.
        for toolName in ("curl", "uv", "uvx"):
            source = shutil.which(toolName)
            if not source:
                continue
            target = hostToolsDir / toolName
            try:
                shutil.copy2(Path(source).resolve(), target)
                target.chmod(0o755)
            except OSError:
                pass
