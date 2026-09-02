#!/usr/bin/env python3
"""One-shot local preparation for the SkillsBench Docker environments.

This is intentionally a small, disposable tool rather than a cache or build
framework.  It pulls missing base images with a host-side ``crane`` process,
builds task images with host networking, and records just enough backup state
to make ``clear`` safe.  Builds run in a small worker pool; rerunning the
command skips task images whose target tag already exists.

Typical use::

    python scripts/PrepareSkillsbench.py --proxy http://127.0.0.1:17891
    python scripts/PrepareSkillsbench.py --proxy http://127.0.0.1:17891 --jobs 8
    python scripts/PrepareSkillsbench.py --proxy http://127.0.0.1:17891 --skip-apt-update
    python scripts/PrepareSkillsbench.py clear
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import os
import platform
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import Request, ProxyHandler, build_opener


CRANE_VERSION = "v0.20.3"
PROXY_VARIABLES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "http_proxy",
    "https_proxy",
    "ALL_PROXY",
    "all_proxy",
)
COMPOSE_FILE_NAMES = (
    "docker-compose.yaml",
    "docker-compose.yml",
    "compose.yaml",
    "compose.yml",
)
BACKUP_SUFFIX = ".kvbench-backup"
PREBUILT_PREFIX = "kvbench-skillsbench/"
APT_BASE_PREFIX = "kvbench-skillsbench-base/"
DEFAULT_BUILD_JOBS = max(1, min(4, os.cpu_count() or 1))
DEFAULT_BUILD_RETRIES = 2


def PrintError(message: str) -> None:
    print(f"[error] {message}", file=sys.stderr)


def RunCommand(
    command: list[str],
    *,
    env: Mapping[str, str] | None = None,
    captureOutput: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run a command without going through a shell or sudo."""

    return subprocess.run(
        command,
        env=dict(env) if env is not None else None,
        text=True,
        capture_output=captureOutput,
        check=False,
    )


def ValidateProxy(proxy: str) -> None:
    parsed = urlsplit(proxy)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(
            "--proxy must be an HTTP(S) URL with a hostname, for example "
            "http://127.0.0.1:17891"
        )
    # Probe targets the build actually needs. We don't require HTTPS github.com
    # because some networks gate HTTPS CONNECT (github.com) but allow HTTP
    # directly through the proxy, which is all the mirror-wrapped build path
    # uses. crane pulls (Docker Hub) only run when a base image is missing,
    # which is rare after the first run.
    probes = (
        ("http://mirrors.tuna.tsinghua.edu.cn/", "tuna mirror (apt/pip fallback)"),
        ("http://archive.ubuntu.com/", "ubuntu archive (apt default)"),
    )
    opener = build_opener(
        ProxyHandler({"http": proxy, "https": proxy})
    )
    failures: list[str] = []
    for url, description in probes:
        request = Request(
            url,
            headers={"User-Agent": "kvbench-skillsbench-preflight"},
        )
        try:
            with opener.open(request, timeout=15) as response:
                response.read(1)
        except Exception as exc:  # noqa: BLE001 - retain the useful network error
            failures.append(f"{description} ({url}): {exc}")
            continue
        print(f"[proxy] {proxy} can reach {description}")
        return
    raise RuntimeError(
        f"proxy {proxy!r} could not reach required hosts: "
        + " | ".join(failures)
    )


def LoadYaml() -> Any:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to update task frontmatter and BenchFlow "
            "compose YAML"
        ) from exc
    return yaml


def ReadYaml(path: Path) -> Any:
    yaml = LoadYaml()
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise RuntimeError(f"could not parse YAML {path}: {exc}") from exc


def WriteTextAtomically(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o644
    temporaryPath: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporaryPath = Path(temporary.name)
            temporary.write(text)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.chmod(temporaryPath, mode)
        os.replace(temporaryPath, path)
        temporaryPath = None
    finally:
        if temporaryPath is not None:
            temporaryPath.unlink(missing_ok=True)


def BackupOnce(path: Path) -> Path:
    backupPath = Path(f"{path}{BACKUP_SUFFIX}")
    if not backupPath.exists():
        shutil.copy2(path, backupPath)
        print(f"[backup] {path} -> {backupPath}")
    return backupPath


def ReadTaskFrontmatter(path: Path) -> tuple[dict[str, Any], str]:
    yaml = LoadYaml()
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise RuntimeError(f"could not read {path}: {exc}") from exc

    normalized = text.replace("\r\n", "\n")
    lines = normalized.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        raise RuntimeError(f"{path} does not start with YAML frontmatter")
    closingIndex = next(
        (index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---"),
        None,
    )
    if closingIndex is None:
        raise RuntimeError(f"{path} has no closing YAML frontmatter delimiter")
    try:
        frontmatter = yaml.safe_load("".join(lines[1:closingIndex]))
    except yaml.YAMLError as exc:
        raise RuntimeError(f"could not parse frontmatter in {path}: {exc}") from exc
    if frontmatter is None:
        frontmatter = {}
    if not isinstance(frontmatter, dict):
        raise RuntimeError(f"frontmatter in {path} must be a mapping")
    return frontmatter, "".join(lines[closingIndex + 1 :])


def RenderTaskDocument(frontmatter: dict[str, Any], body: str) -> str:
    yaml = LoadYaml()
    rendered = yaml.safe_dump(
        frontmatter,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )
    if not rendered.endswith("\n"):
        rendered += "\n"
    return f"---\n{rendered}---\n{body}"


def ResolveSkillsbenchPath(explicitPath: str | None) -> Path:
    if explicitPath:
        path = Path(explicitPath).expanduser().resolve()
    else:
        configPath = Path(__file__).resolve().parent.parent / "config.yaml"
        path = None
        if configPath.is_file():
            try:
                config = ReadYaml(configPath)
                agentBenchFlow = (
                    config.get("AgentBenchFlow") if isinstance(config, dict) else None
                )
                configured = (
                    agentBenchFlow.get("SkillsBenchRepo")
                    if isinstance(agentBenchFlow, dict)
                    else None
                )
                if configured:
                    path = Path(str(configured)).expanduser().resolve()
            except RuntimeError as exc:
                print(f"[skillsbench] ignored invalid {configPath}: {exc}")
        if path is None:
            path = (Path(__file__).resolve().parent / "skillsbench").resolve()
    tasksPath = path / "tasks"
    if not tasksPath.is_dir():
        raise FileNotFoundError(
            f"SkillsBench tasks directory not found: {tasksPath}; use "
            "--skillsbench-path"
        )
    return path


def LocateBenchFlowCompose() -> Path:
    try:
        composeModule = importlib.import_module("benchflow.sandbox._compose")
        candidate = getattr(composeModule, "COMPOSE_BASE_PATH", None)
        if candidate:
            path = Path(candidate)
        else:
            benchflowModule = importlib.import_module("benchflow")
            packagePath = Path(benchflowModule.__file__).resolve().parent
            path = packagePath / "sandbox" / "_compose_files" / "docker-compose-base.yaml"
    except (ImportError, AttributeError, OSError) as exc:
        raise RuntimeError(
            "could not import the installed BenchFlow package to locate "
            f"docker-compose-base.yaml: {exc}"
        ) from exc
    if not path.is_file():
        raise FileNotFoundError(f"BenchFlow compose file not found: {path}")
    return path.resolve()


def DiscoverTasks(skillsbenchPath: Path) -> list[Path]:
    tasksPath = skillsbenchPath / "tasks"
    return sorted(
        (path for path in tasksPath.iterdir() if (path / "task.md").is_file()),
        key=lambda path: path.name,
    )


def LogicalDockerfileLines(text: str) -> Iterable[str]:
    pending = ""
    for line in text.replace("\r\n", "\n").splitlines():
        stripped = line.strip()
        # Docker strips blank lines and comment lines BEFORE joining
        # backslash continuations, so the in-flight ``RUN apt install a \\
        # # explanation \\
        # b`` still parses as one logical ``RUN`` whose body carries ``a`` and
        # ``b`` with the comment gone. Mirror that here so the wrap sees a
        # single RUN line and does not emit stray package names as their own
        # "Dockerfile instructions".
        if pending and not stripped:
            continue
        if pending and stripped.startswith("#"):
            continue
        piece = line.rstrip()
        if piece.endswith("\\"):
            pending += piece[:-1] + " "
        else:
            yield pending + piece
            pending = ""
    if pending:
        yield pending


def SubstituteDockerfileArgs(value: str, arguments: Mapping[str, str]) -> str | None:
    unresolved = False

    def replace(match: re.Match[str]) -> str:
        nonlocal unresolved
        name = match.group(1) or match.group(2)
        if name not in arguments:
            unresolved = True
            return match.group(0)
        return arguments[name]

    result = re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)", replace, value)
    return None if unresolved else result


def DockerfileBaseImages(path: Path) -> set[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise RuntimeError(f"could not read {path}: {exc}") from exc

    arguments: dict[str, str] = {}
    stageNames: set[str] = set()
    images: set[str] = set()
    heredocDelimiter: str | None = None
    for line in LogicalDockerfileLines(text):
        stripped = line.strip()
        if heredocDelimiter is not None:
            if stripped in {heredocDelimiter, f"-{heredocDelimiter}"}:
                heredocDelimiter = None
            continue
        if not stripped or stripped.startswith("#"):
            continue
        try:
            tokens = shlex.split(stripped, comments=True, posix=True)
        except ValueError as exc:
            raise RuntimeError(f"could not parse Dockerfile line in {path}: {exc}") from exc
        if not tokens:
            continue
        instruction = tokens[0].upper()
        if instruction == "ARG" and len(tokens) >= 2:
            name, separator, value = tokens[1].partition("=")
            if separator:
                arguments[name] = value
            continue
        if instruction in {"RUN", "COPY"}:
            heredoc = re.search(
                r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_.-]*)\1",
                stripped,
            )
            if heredoc is not None:
                heredocDelimiter = heredoc.group(2)
            continue
        if instruction != "FROM":
            continue
        imageIndex = 1
        while imageIndex < len(tokens) and tokens[imageIndex].startswith("--"):
            imageIndex += 1
        # Dockerfile instructions are case-insensitive, but shell/Python
        # snippets in a heredoc can still be encountered in malformed or
        # unusual files. ``from module import name`` is not a Docker FROM
        # because only ``AS <stage>`` may follow its image reference.
        if (
            tokens[0] != "FROM"
            and len(tokens) > imageIndex + 1
            and tokens[imageIndex + 1].upper() != "AS"
        ):
            continue
        if imageIndex >= len(tokens):
            raise RuntimeError(f"FROM has no image reference in {path}: {line}")
        image = SubstituteDockerfileArgs(tokens[imageIndex], arguments)
        if image is None:
            print(f"[base] skipping unresolved Dockerfile image in {path}: {tokens[imageIndex]}")
        elif image.lower() != "scratch" and image.lower() not in stageNames and not image.isdigit():
            images.add(image)
        aliasIndex = imageIndex + 1
        if (
            aliasIndex + 1 < len(tokens)
            and tokens[aliasIndex].upper() == "AS"
        ):
            stageNames.add(tokens[aliasIndex + 1].lower())
    return images


def ComposeFiles(environmentPath: Path) -> list[Path]:
    paths = {
        environmentPath / name
        for name in COMPOSE_FILE_NAMES
        if (environmentPath / name).is_file()
    }
    for pattern in ("docker-compose*.yaml", "docker-compose*.yml", "compose*.yaml", "compose*.yml"):
        paths.update(path for path in environmentPath.glob(pattern) if path.is_file())
    return sorted(paths)


def EnvironmentDockerfiles(environmentPath: Path) -> list[Path]:
    return sorted(
        path
        for path in environmentPath.rglob("Dockerfile*")
        if path.is_file() and (path.name == "Dockerfile" or path.name.startswith("Dockerfile."))
    )


def ImageValues(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "image" and isinstance(child, str):
                yield child.strip()
            yield from ImageValues(child)
    elif isinstance(value, list):
        for child in value:
            yield from ImageValues(child)


def ComposeImageReferences(path: Path) -> set[str]:
    yaml = LoadYaml()
    documents = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            documents = list(yaml.safe_load_all(stream))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise RuntimeError(f"could not parse compose YAML {path}: {exc}") from exc
    references: set[str] = set()
    for document in documents:
        if not isinstance(document, dict):
            continue
        services = document.get("services")
        if isinstance(services, dict):
            # An image paired with build: is the output name of a local
            # service, not an external image that needs a registry pull.
            for service in services.values():
                if not isinstance(service, dict) or "build" in service:
                    continue
                image = service.get("image")
                if isinstance(image, str):
                    image = image.strip()
                    if image and not image.startswith("$") and "${" not in image:
                        references.add(image)
            # Images in extension/top-level sections are still useful to
            # discover, but do not walk services a second time.
            document = {
                key: value for key, value in document.items() if key != "services"
            }
        for image in ImageValues(document):
            if image and not image.startswith("$") and "${" not in image:
                references.add(image)
    return references


def RequiredBaseImages(tasks: list[Path]) -> tuple[set[str], list[str]]:
    images: set[str] = set()
    errors: list[str] = []
    for taskPath in tasks:
        environmentPath = taskPath / "environment"
        for dockerfile in EnvironmentDockerfiles(environmentPath):
            try:
                images.update(DockerfileBaseImages(dockerfile))
            except RuntimeError as exc:
                errors.append(str(exc))
        for composePath in ComposeFiles(environmentPath):
            try:
                images.update(ComposeImageReferences(composePath))
            except RuntimeError as exc:
                errors.append(str(exc))
    return images, errors


def DockerfileHasAptUpdate(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise RuntimeError(f"could not read {path}: {exc}") from exc
    return re.search(r"(?i)\b(?:apt-get|apt)\s+update\b", text) is not None


def DockerfileAptUpdateCount(text: str) -> int:
    """Count shell updates while ignoring commented Dockerfile lines."""

    return sum(
        len(re.findall(r"(?i)\b(?:apt-get|apt)\s+update\b", line))
        for line in text.splitlines()
        if not line.lstrip().startswith("#")
    )


def DockerfileHasExplicitPlatform(text: str) -> bool:
    return re.search(r"(?im)^\s*FROM\s+--platform=", text) is not None


def RequiredAptBaseImages(tasks: list[Path]) -> tuple[set[str], list[str]]:
    """Find external bases whose apt indexes can be prepared once and reused."""

    images: set[str] = set()
    errors: list[str] = []
    for taskPath in tasks:
        environmentPath = taskPath / "environment"
        for dockerfile in EnvironmentDockerfiles(environmentPath):
            try:
                if DockerfileHasAptUpdate(dockerfile):
                    images.update(DockerfileBaseImages(dockerfile))
            except RuntimeError as exc:
                errors.append(str(exc))
    return images, errors


def ProxyEnvironment(proxy: str) -> dict[str, str]:
    environment = dict(os.environ)
    for variable in PROXY_VARIABLES:
        environment[variable] = proxy
    return environment


def DockerImageExists(image: str) -> bool:
    result = RunCommand(
        ["docker", "image", "inspect", image],
        captureOutput=True,
    )
    return result.returncode == 0


def ToolDirectory() -> Path:
    userId = str(getattr(os, "getuid", lambda: 0)())
    return Path(tempfile.gettempdir()) / f"kvbench-skillsbench-init-{userId}"


def DownloadUrl(proxy: str, url: str, destination: Path) -> None:
    opener = build_opener(ProxyHandler({"http": proxy, "https": proxy}))
    request = Request(url, headers={"User-Agent": "kvbench-skillsbench-preflight"})
    try:
        with opener.open(request, timeout=60) as response, destination.open("wb") as output:
            shutil.copyfileobj(response, output)
    except (OSError, URLError) as exc:
        raise RuntimeError(f"could not download {url}: {exc}") from exc


def EnsureCrane(proxy: str) -> Path:
    existing = shutil.which("crane")
    if existing:
        return Path(existing)

    architecture = platform.machine().lower()
    architectureName = {
        "x86_64": "x86_64",
        "amd64": "x86_64",
        "aarch64": "arm64",
        "arm64": "arm64",
    }.get(architecture)
    if architectureName is None:
        raise RuntimeError(f"unsupported host architecture for crane: {architecture}")

    toolDirectory = ToolDirectory()
    toolDirectory.mkdir(mode=0o700, parents=True, exist_ok=True)
    cranePath = toolDirectory / "crane"
    if cranePath.is_file() and os.access(cranePath, os.X_OK):
        return cranePath

    asset = f"go-containerregistry_Linux_{architectureName}.tar.gz"
    url = (
        f"https://github.com/google/go-containerregistry/releases/download/"
        f"{CRANE_VERSION}/{asset}"
    )
    archivePath = toolDirectory / f"{asset}.download"
    print(f"[tool] downloading crane {CRANE_VERSION}")
    try:
        DownloadUrl(proxy, url, archivePath)
        with tarfile.open(archivePath, "r:gz") as archive:
            member = next(
                (
                    candidate
                    for candidate in archive.getmembers()
                    if Path(candidate.name).name == "crane" and candidate.isfile()
                ),
                None,
            )
            if member is None:
                raise RuntimeError(f"crane binary not found in {url}")
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"could not extract crane from {url}")
            temporaryPath = cranePath.with_suffix(".tmp")
            with temporaryPath.open("wb") as output:
                shutil.copyfileobj(source, output)
            os.chmod(temporaryPath, 0o755)
            os.replace(temporaryPath, cranePath)
    except (OSError, tarfile.TarError, StopIteration) as exc:
        raise RuntimeError(f"could not install crane: {exc}") from exc
    finally:
        archivePath.unlink(missing_ok=True)
    return cranePath


def LoadedImageReferences(output: str) -> list[str]:
    references: list[str] = []
    for line in output.splitlines():
        if "Loaded image:" in line:
            references.append(line.split("Loaded image:", 1)[1].strip())
        elif "Loaded image ID:" in line:
            references.append(line.split("Loaded image ID:", 1)[1].strip())
    return references


def PullBaseImage(image: str, crane: Path, proxy: str) -> bool:
    print(f"[base] pulling {image}")
    toolDirectory = ToolDirectory()
    toolDirectory.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporaryFile = tempfile.NamedTemporaryFile(
        prefix="base-",
        suffix=".tar",
        dir=toolDirectory,
        delete=False,
    )
    tarPath = Path(temporaryFile.name)
    temporaryFile.close()
    # crane creates the destination itself; leave only a unique path, not an
    # empty pre-existing file that an implementation might refuse to replace.
    tarPath.unlink(missing_ok=True)
    environment = ProxyEnvironment(proxy)
    try:
        pull = RunCommand(
            [str(crane), "pull", image, str(tarPath)],
            env=environment,
            captureOutput=True,
        )
        if pull.returncode != 0:
            PrintError(
                f"crane pull failed for {image}: "
                f"{(pull.stderr or pull.stdout).strip()}"
            )
            return False
        loaded = RunCommand(
            ["docker", "load", "--input", str(tarPath)],
            captureOutput=True,
        )
        loadOutput = f"{loaded.stdout}\n{loaded.stderr}"
        if loaded.returncode != 0:
            PrintError(f"docker load failed for {image}: {loadOutput.strip()}")
            return False
        if not DockerImageExists(image):
            for loadedReference in LoadedImageReferences(loadOutput):
                tagged = RunCommand(
                    ["docker", "tag", loadedReference, image],
                    captureOutput=True,
                )
                if tagged.returncode == 0 and DockerImageExists(image):
                    break
        if not DockerImageExists(image):
            PrintError(
                f"docker load completed for {image}, but the original image "
                "reference was not available locally"
            )
            return False
        print(f"[base] {image} OK")
        return True
    finally:
        tarPath.unlink(missing_ok=True)


def PullMissingBaseImages(
    images: Iterable[str],
    proxy: str,
    jobs: int = 1,
) -> dict[str, int]:
    counts = {"existing": 0, "pulled": 0, "failed": 0}
    missing: list[str] = []
    for image in sorted(set(images)):
        if DockerImageExists(image):
            print(f"[base] {image} already exists")
            counts["existing"] += 1
        else:
            missing.append(image)
    if not missing:
        return counts
    try:
        crane = EnsureCrane(proxy)
    except RuntimeError as exc:
        PrintError(str(exc))
        counts["failed"] = len(missing)
        return counts
    print(
        f"[base] pulling {len(missing)} missing base image(s) with "
        f"{min(jobs, len(missing))} worker(s)"
    )

    def pullOne(image: str) -> tuple[str, bool]:
        try:
            return image, PullBaseImage(image, crane, proxy)
        except Exception as exc:  # noqa: BLE001 - continue with other images
            PrintError(f"base image {image} FAILED: {exc}")
            return image, False

    with ThreadPoolExecutor(
        max_workers=min(jobs, len(missing)),
        thread_name_prefix="skillsbench-base",
    ) as executor:
        futures = {executor.submit(pullOne, image): image for image in missing}
        for future in as_completed(futures):
            image, pulled = future.result()
            if pulled:
                counts["pulled"] += 1
            else:
                counts["failed"] += 1
    return counts


def SafeImageName(taskName: str) -> str:
    name = re.sub(r"[^a-z0-9_.-]+", "-", taskName.lower()).strip(".-")
    return name or "task"


def TaskImageName(taskPath: Path) -> str:
    return f"{PREBUILT_PREFIX}{SafeImageName(taskPath.name)}:latest"


def AptBaseImageName(image: str, content: str = "") -> str:
    # Include the (optional) Dockerfile content in the hash so any change to
    # what the apt-base installs (e.g. ca-certificates) yields a new image
    # tag instead of being silently masked by the old tag's "already exists"
    # short-circuit.
    digest = hashlib.sha256(
        f"{image}|{content}".encode("utf-8")
    ).hexdigest()[:16]
    return f"{APT_BASE_PREFIX}{digest}:latest"


def ProxyBuildArguments(proxy: str) -> list[str]:
    arguments: list[str] = []
    for variable in PROXY_VARIABLES:
        arguments.extend(["--build-arg", f"{variable}={proxy}"])
    return arguments


def CommandOutput(result: subprocess.CompletedProcess[str]) -> str:
    return "\n".join(
        output.strip()
        for output in (result.stdout or "", result.stderr or "")
        if output and output.strip()
    )


def WriteCommandLog(
    logDirectory: Path | None,
    name: str,
    result: subprocess.CompletedProcess[str],
) -> Path | None:
    if logDirectory is None:
        return None
    logDirectory.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = logDirectory / f"{SafeImageName(name)}.log"
    try:
        path.write_text(CommandOutput(result) + "\n", encoding="utf-8")
    except OSError as exc:
        PrintError(f"could not write build log {path}: {exc}")
        return None
    return path


def OutputTail(output: str, lineCount: int = 40) -> str:
    lines = [line.rstrip() for line in output.replace("\r", "").splitlines()]
    if len(lines) > lineCount:
        lines = lines[-lineCount:]
    return "\n".join(lines)


def RemoveAptUpdateCommands(text: str) -> str:
    """Remove update-and-chain prefixes from generated Dockerfiles.

    The SkillsBench Dockerfiles use ``apt-get update && apt-get install``
    (including the multiline form).  The update is removed only in a
    temporary Dockerfile used for this invocation; upstream task files are
    never modified.
    """

    update = re.compile(
        r"(?i)\b(?:apt-get|apt)\s+update"
        r"(?:[ \t]+[^\s&\\]+)*"
        r"(?:[ \t]*\\[ \t]*\r?\n[ \t]*)?"
        r"[ \t]*&&[ \t]*"
    )
    return update.sub("", text)


def ReplaceDockerfileBaseImages(
    text: str,
    replacements: Mapping[str, str],
) -> str:
    """Point FROM instructions at locally prepared apt-base images."""

    fromLine = re.compile(
        r"^(?P<prefix>\s*FROM\s+(?:(?:--[^\s]+)\s+)*)(?P<image>\S+)(?P<suffix>.*)$",
        re.IGNORECASE,
    )
    heredocDelimiter: str | None = None
    rendered: list[str] = []
    heredocPattern = re.compile(
        r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_.-]*)\1"
    )
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if heredocDelimiter is not None:
            rendered.append(line)
            if stripped in {heredocDelimiter, f"-{heredocDelimiter}"}:
                heredocDelimiter = None
            continue

        lineBody = line.rstrip("\r\n")
        lineEnding = line[len(lineBody) :]
        match = fromLine.match(lineBody)
        if match:
            replacement = replacements.get(match.group("image"))
            if replacement is not None:
                line = (
                    f"{match.group('prefix')}{replacement}{match.group('suffix')}"
                    f"{lineEnding}"
                )
        rendered.append(line)

        if stripped.upper().startswith(("RUN ", "COPY ")):
            heredoc = heredocPattern.search(stripped)
            if heredoc is not None:
                heredocDelimiter = heredoc.group(2)
    return "".join(rendered)


# Shell fragment run before any wrapped body: --network=host lets the container
# reach the mirror directly, but Docker daemon (and some base images) still
# inject HTTP(S)_PROXY into RUNs. Unsetting here forces the wrapped commands
# onto the direct path. Final images never see this fragment because it lives
# inside the temporary Dockerfile generated per build invocation.
PROXY_UNSET_SHELL = (
    "unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY"
)

# Stable, fast Chinese academic mirrors. Their content is byte-for-byte
# identical to the upstream archives, so swapping/reverting leaves the image
# identical to one built against archive.ubuntu.com / deb.debian.org.
APT_MIRROR_HOST = "mirrors.tuna.tsinghua.edu.cn"
APT_UBUNTU_PATH = "ubuntu"
APT_DEBIAN_PATH = "debian"
PIP_MIRROR_URL = "https://pypi.tuna.tsinghua.edu.cn/simple"
RPM_MIRROR_HOST = "mirrors.tuna.tsinghua.edu.cn"
# Tuna's RPM mirror is incomplete: it serves ``centos`` and ``fedora`` but
# returns 404 for ``almalinux`` and ``rockylinux`` (AlmaLinux 9 / Rocky 9
# images like ``jasonish/suricata:7.0.11`` need these). aliyun.com mirrors
# all four distros. We list both as fallbacks so the rewritten ``baseurl=``
# block carries both candidates and dnf fails over to the working one.
RPM_MIRROR_FALLBACK_HOST = "mirrors.aliyun.com"
RPM_DISTRO_PATHS = {
    "almalinux": "almalinux",
    "rocky": "rockylinux",
    "centos": "centos",
    "fedora": "fedora",
}
# GitHub proxy: front-end for github.com / raw.githubusercontent.com. Returns
# the original asset bytes, so wrapping preserves content integrity.
GITHUB_PROXY_HOST = "gh-proxy.com"
# Fallback GitHub proxy. ``gh-proxy.com`` works for direct curl/wget downloads
# but its availability is not guaranteed; if the primary host fails the wrap
# retries against this one. The ``gh-proxy.org`` host accepts the same
# ``https://<host>/https://github.com/...`` URL shape as the primary, so a
# mirror-prefix swap is the only change needed.
GITHUB_PROXY_FALLBACK = "gh-proxy.org"
# Coursier reads ``COURSIER_MIRRORS`` to redirect Maven Central / GitHub
# Maven artifacts the JVM resolves at run time (Scala compiler fetch,
# self-update, etc.). Without this, coursier's ``./cs setup`` self-update
# hits ``github.com`` directly through the host's proxy and dies on the
# upstream S3 redirect's TLS handshake.
COURSIER_MIRROR_URL = f"https://{GITHUB_PROXY_HOST}/https://github.com/coursier/maven"
# npmmirror.com serves the official npm registry as a read-through mirror,
# plus binaries/ (node, etc.). Used for both ``npm install`` and node tarballs.
NPM_MIRROR_REGISTRY = "https://registry.npmmirror.com"
NODE_BINARIES_MIRROR = "https://cdn.npmmirror.com/binaries/node"
# HuggingFace mirror: ``huggingface_hub`` honors ``HF_ENDPOINT`` and the
# ``huggingface-cli`` follows the same env var. Setting this to the
# Chinese mirror sidesteps the build-time SSL flakiness against
# huggingface.co from networks where the proxy is on, but the endpoint
# itself is blocked or handshake-broken.
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"


def WrapAptInstallRun(body: str) -> str:
    """Wrap an apt-get/apt install body so it uses a mirror and restores sources.

    The original body is preserved verbatim. Sources, sources.list.d, and
    apt cache are restored/cleaned so the resulting image is byte-identical
    to one built against the upstream Ubuntu/Debian archives.
    """

    match = re.search(r"(?i)\b(?:apt-get|apt)\s+install\b", body)
    if match is None:
        return body
    # Preserve any leading shell prefix that may exist before apt-get install
    # (e.g. an explicit ``apt-get update &&`` written by the task author).
    prefix = body[: match.start()]
    aptPortion = body[match.start() :]
    backupList = (
        "if [ -f /etc/apt/sources.list ]; then "
        "cp /etc/apt/sources.list /tmp/.kvbench-sources.bak; fi; "
        "if [ -d /etc/apt/sources.list.d ]; then "
        "cp -r /etc/apt/sources.list.d /tmp/.kvbench-sources.d.bak; fi"
    )
    restoreList = (
        "if [ -f /tmp/.kvbench-sources.bak ]; then "
        "mv /tmp/.kvbench-sources.bak /etc/apt/sources.list; fi; "
        "if [ -d /tmp/.kvbench-sources.d.bak ]; then "
        "rm -rf /etc/apt/sources.list.d; "
        "mv /tmp/.kvbench-sources.d.bak /etc/apt/sources.list.d; fi; "
        "rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*.deb"
    )
    # Pick distro+path at run time from /etc/os-release; defaults fall back to
    # the URL shape that tuna mirrors serve for that family.
    # NOTE: ``http://`` (not ``https://``) so apt can fetch indexes and
    # packages without a populated /etc/ssl/certs. Package integrity is
    # still guaranteed by the InRelease/Release.gpg signature that apt
    # verifies against ``/etc/apt/trusted.gpg.d/``. Slim ubuntu images do
    # not ship ca-certificates; installing them via apt requires apt itself
    # to work first -- a chicken-and-egg that http-only sources sidestep.
    rewriteList = (
        f'dist=$(. /etc/os-release && echo "$ID"); '
        f'codename=$(. /etc/os-release && echo "$VERSION_CODENAME"); '
        f'mirror={APT_MIRROR_HOST}; '
        f'case "$dist" in '
        f'  ubuntu) baseurl="http://$mirror/{APT_UBUNTU_PATH}";; '
        f'  debian) baseurl="http://$mirror/{APT_DEBIAN_PATH}";; '
        f'  *) baseurl="http://$mirror/$dist";; '
        f'esac; '
        f'{{ printf "deb %s/ %s main restricted universe multiverse\\n" '
        f'"$baseurl" "$codename"; '
        f'printf "deb %s/ %s-updates main restricted universe multiverse\\n" '
        f'"$baseurl" "$codename"; '
        f'printf "deb %s/ %s-security main restricted universe multiverse\\n" '
        f'"$baseurl" "$codename"; '
        f'}} > /etc/apt/sources.list; '
        f'rm -rf /etc/apt/sources.list.d'
    )
    return (
        f"{PROXY_UNSET_SHELL}; "
        f"{backupList}; "
        f"{rewriteList}; "
        # Always re-fetch the index against the mirror sources we just wrote.
        # The script's apt-base optimization normally strips the original
        # ``apt-get update &&`` so we rely on indexes baked into the base
        # image, but those indexes are against archive.ubuntu.com /
        # deb.debian.org and cannot satisfy an install against the mirror.
        # The semicolon (not ``&&``) keeps :func:`RemoveAptUpdateCommands`
        # from stripping this update.
        f"apt-get update; "
        f"{prefix}{aptPortion}; "
        f"{restoreList}"
    )


def WrapDnfInstallRun(body: str) -> str:
    """Wrap a dnf/yum install body to use a mirror and restore repo files."""

    match = re.search(r"(?i)\b(?:dnf|yum)\s+(?:-y\s+)?install\b", body)
    if match is None:
        return body
    # Preserve any leading shell prefix (e.g. ``set -eux;`` or chained ``dnf
    # clean all &&``) so we never drop commands the task author wrote.
    prefix = body[: match.start()]
    rpmPortion = body[match.start() :]
    backupRepos = (
        "mkdir -p /tmp/.kvbench-repo.bak; "
        "if [ -d /etc/yum.repos.d ]; then "
        "cp -r /etc/yum.repos.d /tmp/.kvbench-repo.bak/; fi"
    )
    rewriteRepos = (
        "for f in /etc/yum.repos.d/*.repo; do "
        "  [ -f \"$f\" ] || continue; "
        "  sed -i -E "
        "    -e 's|^mirrorlist=|#mirrorlist=|g' "
        # The baseurl match consumes arbitrary whitespace before ``#``
        # AND the trailing distro subpath (``/almalinux/`` /
        # ``/rocky/`` / etc.) — AlmaLinux/Rocky ship the fallback line
        # as ``# baseurl=https://repo.<distro>.org/<distro>/<releasever>/...``
        # and the path has to align with the mirror's layout
        # (``https://mirrors.aliyun.com/<distro>/<releasever>/...``),
        # otherwise dnf gets ``…/almalinux/almalinux/9/...`` and 404s.
        #
        # dnf uses only ONE ``baseurl=`` per repo section. Tuna serves
        # centos/fedora but returns 404 for almalinux/rockylinux; aliyun
        # mirrors all four. We pick the working mirror per distro:
        # aliyun for almalinux/rocky (tuna 404s), tuna for centos/fedora
        # (tuna is closer / faster from this host).
        "    -e 's|^[[:space:]]*#[[:space:]]*baseurl=https?://[^/]*almalinux\\.org/almalinux/?|"
        f"baseurl=https://{RPM_MIRROR_FALLBACK_HOST}/almalinux/|g' "
        "    -e 's|^[[:space:]]*#[[:space:]]*baseurl=https?://[^/]*rockylinux\\.org/rocky/?|"
        f"baseurl=https://{RPM_MIRROR_FALLBACK_HOST}/rockylinux/|g' "
        "    -e 's|^[[:space:]]*#[[:space:]]*baseurl=https?://[^/]*centos\\.org/centos/?|"
        f"baseurl=https://{RPM_MIRROR_HOST}/centos/|g' "
        "    -e 's|^[[:space:]]*#[[:space:]]*baseurl=https?://[^/]*fedora\\.org/fedora/?|"
        f"baseurl=https://{RPM_MIRROR_HOST}/fedora/|g' "
        "    \"$f\"; "
        "done"
    )
    restoreRepos = (
        "if [ -d /tmp/.kvbench-repo.bak/yum.repos.d ]; then "
        "rm -rf /etc/yum.repos.d; "
        "mv /tmp/.kvbench-repo.bak/yum.repos.d /etc/yum.repos.d; fi"
        # NOTE: deliberately NOT appending ``|| true`` here. The previous
        # ``dnf clean all >/dev/null 2>&1 || true`` masked install failures:
        # the upstream ``dnf install ... && dnf clean all && rm -rf ...``
        # chain returned non-zero but the trailing ``|| true`` made the
        # whole RUN exit 0, so Docker committed the layer with broken state
        # (e.g. xz missing) and a ``---> Using cache`` stamp tricked later
        # steps. Letting this exit propagate surfaces the real failure.
    )
    return (
        f"{PROXY_UNSET_SHELL}; "
        f"{backupRepos}; "
        f"{rewriteRepos}; "
        # dnf/microdnf cache metadata under ``/var/cache/dnf``; force a refresh
        # against the mirror repos we just wrote so the install can resolve
        # package names that the original repo files never indexed.
        f"dnf -y makecache --disablerepo='*' --enablerepo='*' 2>/dev/null || true; "
        f"{prefix}{rpmPortion}; "
        f"{restoreRepos}"
    )


def WrapPipInstallRun(body: str) -> str:
    """Add --index-url mirror to pip/pip3 install commands.

    ``--index-url`` is per-invocation and does not touch pip's user/global
    config files, so the final image's ``~/.pip/pip.conf`` etc. stay clean.
    """

    if not re.search(r"(?i)\bpip3?\s+install\b", body):
        return body
    rewritten = re.sub(
        r"(?i)(\bpip3?\s+install\b)",
        rf"\1 --index-url {PIP_MIRROR_URL}",
        body,
    )
    return f"{PROXY_UNSET_SHELL}; {rewritten}"


# Mirror rewrite tables used by :func:`WrapCurlWgetRun`. Order matters: the
# first matching prefix wins. We never touch hosts outside this list, so
# arbitrary user URLs (e.g. private artifact servers) keep flowing through
# unchanged.
_CURL_URL_REWRITES: tuple[tuple[re.Pattern[str], str], ...] = (
    # nodejs.org tarballs / index.json -> tuna binaries mirror (same files)
    (re.compile(r"https?://nodejs\.org/dist/"), f"{NODE_BINARIES_MIRROR}/dist/"),
    # GitHub releases / raw assets -> gh-proxy.com front-end
    (re.compile(r"https?://raw\.githubusercontent\.com/"),
     f"https://{GITHUB_PROXY_HOST}/https://raw.githubusercontent.com/"),
    (re.compile(r"https?://github\.com/"),
     f"https://{GITHUB_PROXY_HOST}/https://github.com/"),
)


def RewriteCurlUrls(body: str) -> str:
    """Rewrite full URLs whose host appears in the mirror table.

    Leaves every other URL (including http:// variants of the same hosts)
    untouched, so private mirrors and authenticated hosts keep working.
    Skips URLs that contain shell variable references (``${VAR}`` / ``$VAR``)
    because we cannot safely splice the mirror prefix into them.
    """

    def replace(match: re.Match[str]) -> str:
        url = match.group(0)
        if "$" in url:
            return url
        for pattern, replacement in _CURL_URL_REWRITES:
            if pattern.match(url):
                return pattern.sub(replacement, url, count=1)
        return url

    return re.sub(r"\bhttps?://[^\s'\"\\)]+", replace, body)


def WrapCurlWgetRun(body: str) -> str:
    """Add retry flags to / curl / / wget / invocations and mirror their URLs.

    The flaky proxy intermittently returns ``SSL_ERROR_SYSCALL`` on long
    github.com downloads. ``--retry 5 --retry-delay 3 --retry-connrefused``
    covers both transient connection resets and bare SSL resets, and only
    touches invocations that did not already specify ``--retry``. We rewrite
    only URLs whose host appears in :data:`_CURL_URL_REWRITES`, so unrelated
    curls (e.g. to a private artifact bucket) pass through verbatim.

    A curl/wget is treated as a command (not a package name) only when it
    appears at the start of the body or right after a shell operator
    (``&&`` / ``||`` / ``;`` / ``|``). This avoids rewriting the bare
    ``curl`` in ``apt-get install -y curl python3`` or ``RUN apt install
    curl wget``.

    Flag-with-argument handling: ``curl -o FILE`` and ``wget -O FILE`` are
    the canonical form that puts the new filename token *immediately* after
    a single-letter flag. The regex captures the command word in group 1
    and any pre-existing flag tokens in group 2; the new flags are inserted
    between them so that ``-O`` and its filename stay adjacent. The same
    shape protects ``curl --output FILE`` / ``wget --output-document FILE``
    (the regex stops before the bare filename because it lacks a leading
    dash, and before ``=FILE`` because ``=`` is not in ``[\\w-]``).
    """

    command_pos = r"(?:^|(?:&&|\|\||;|\|)\s+)"
    if not re.search(
        rf"(?i){command_pos}(?:curl|wget)\s+(?:-{{1,2}}[A-Za-z-]+|https?://|\$\w|\"|')",
        body,
    ):
        return body
    rewritten = RewriteCurlUrls(body)
    if re.search(rf"(?i){command_pos}curl\b", rewritten) and "--retry" not in rewritten:
        rewritten = re.sub(
            rf"(?i)({command_pos}curl\b)((?:\s+--?[A-Za-z][\w-]*)*)",
            r"\1 --retry 5 --retry-delay 3 --retry-connrefused --max-time 1800\2",
            rewritten,
        )
    if re.search(rf"(?i){command_pos}wget\b", rewritten) and "--tries" not in rewritten:
        rewritten = re.sub(
            rf"(?i)({command_pos}wget\b)((?:\s+--?[\w-]+)*)",
            r"\1 --tries=5 --waitretry=3 --timeout=60\2",
            rewritten,
        )
    return _WithProxyUnset(rewritten)


def WrapGitCloneRun(body: str) -> str:
    """Rewrite ``git clone https://github.com/...`` to use the GitHub mirror.

    Inline URL substitution is safer than ``git config --global
    url.<...>.insteadOf`` because the config would persist for any later
    git invocation in the same RUN. The mirror only proxies github.com; git
    clones against any other host are left untouched.
    """

    def replace(match: re.Match[str]) -> str:
        verb = match.group(1)
        url = match.group(2)
        if url.startswith("https://github.com/") or url.startswith(
            "http://github.com/"
        ):
            url = f"https://{GITHUB_PROXY_HOST}/https://github.com/" + url.split(
                "github.com/", 1
            )[1]
        return f"{verb} {url}"

    if not re.search(r"(?i)\bgit\s+clone\b", body):
        return body
    rewritten = re.sub(
        r"(?i)(git\s+clone)\s+(https?://[^\s'\"\\$|&;]+)",
        replace,
        body,
    )
    return _WithProxyUnset(rewritten)


# Nodesource's setup script (``deb.nodesource.com/setup_X.x``) is hosted on a
# Cloudflare CDN that is intermittently blocked from networks whose egress
# proxy's TLS path is flaky. The script itself also has no working Chinese
# mirror (npmmirror has the redirect registered but the file is never
# synced; tuna/aliyun/huawei all 404). The official Node.js binary tarball
# IS mirrored on ``cdn.npmmirror.com/binaries/node/vX.Y.Z/`` (verified
# reachable), so we replace the ``curl setup_X.x | bash - && apt-get install
# nodejs`` pipeline with a direct tarball install. ``NODESOURCE_NODE_VERSION``
# is the latest stable minor per major line; the wrap is only consulted when
# ``deb.nodesource.com/setup_<N>.x`` actually appears in the body.
NODESOURCE_NODE_VERSION = {
    "18": "v18.20.4",
    "20": "v20.18.0",
    "22": "v22.12.0",
}


def WrapNodeSourceSetupRun(body: str) -> str:
    """Replace ``curl ... setup_X.x | bash - && apt-get install -y nodejs``
    with a tarball download from the npmmirror CDN.

    ``setup_X.x`` writes the nodesource apt repo to ``/etc/apt/sources.list.d``
    and runs ``apt-get install -y nodejs`` to fetch the binary. Both legs
    depend on hosts we cannot reach; the official Node.js release tarball
    IS mirrored at ``cdn.npmmirror.com/binaries/node/``, so we substitute
    the whole pipeline with a self-contained tarball unpack that puts
    ``node``/``npm``/``npx`` on ``/usr/local/bin``. ``apt-get install -y
    nodejs`` is dropped because the unpacked tarball already provides
    them — keeping it would only consume apt indexes against a missing
    nodesource repo and fail.

    The wrap only fires when the body mentions ``deb.nodesource.com/setup_``,
    so unrelated tasks are untouched.
    """

    match = re.search(
        r"https?://deb\.nodesource\.com/setup_(\d+)\.x",
        body,
    )
    if match is None:
        return body
    version = match.group(1)
    fullVersion = NODESOURCE_NODE_VERSION.get(version)
    if fullVersion is None:
        return body
    archive = f"node-{fullVersion}-linux-x64"
    archiveUrl = f"{NODE_BINARIES_MIRROR}/{fullVersion}/{archive}.tar.gz"
    replacement = (
        f"curl -fsSL --retry 5 --retry-delay 3 --retry-connrefused --max-time 1800 "
        f"{archiveUrl} -o /tmp/.kvbench-node.tar.gz "
        f"&& tar -xzf /tmp/.kvbench-node.tar.gz -C /opt "
        f"&& ln -sf /opt/{archive}/bin/node /usr/local/bin/node "
        f"&& ln -sf /opt/{archive}/bin/npm /usr/local/bin/npm "
        f"&& ln -sf /opt/{archive}/bin/npx /usr/local/bin/npx "
        f"&& ln -sf /opt/{archive}/bin/corepack /usr/local/bin/corepack"
    )
    # Match the entire ``curl ... setup_X.x | bash -`` segment and the
    # immediately-trailing ``apt-get install -y nodejs`` (any quoting/flag
    # shape, including the long ``--retry 5`` flags the other wraps add),
    # stopping before the next &&-chained command. We keep the trailing
    # commands (e.g. ``&& apt-get clean && rm -rf ...``) intact. The
    # flag token regex accepts ``-x VALUE`` / ``--long VALUE`` / ``--flag=VALUE``
    # in any combination so the wrap stays correct after ``WrapCurlWgetRun``
    # has inserted ``--retry 5 --retry-delay 3 --retry-connrefused
    # --max-time 1800`` ahead of the URL.
    pattern = re.compile(
        r"curl\s+"
        r"(?:--?[A-Za-z][\w-]*(?:[ =][^\s|]+)?\s+)*"
        r"https?://deb\.nodesource\.com/setup_\d+\.x\s*"
        r"\|\s*(?:sudo\s+)?bash\s+-?\s*"
        r"(?:\s*&&\s*apt-get\s+install\s+(?:--?[A-Za-z][\w-]*\s+)*nodejs\s*)?",
        re.IGNORECASE,
    )
    rewritten, count = pattern.subn(replacement, body, count=1)
    if count == 0:
        return body
    return rewritten


def WrapCoursierRun(body: str) -> str:
    """Make coursier (``cs setup`` / ``cs install``) survive flaky networks.

    Two failure modes hit when the host proxy can't reach ``github.com``:

    1. ``./cs setup --yes`` self-updates to the latest release straight from
       ``github.com`` (not through the curl/wget rewrites), and the upstream
       redirect to ``release-assets.githubusercontent.com`` times out on TLS.
       M23's ``cs setup`` does NOT honor ``--no-self-update`` (the flag is
       only available on ``cs launch`` / ``cs java``), so we drop the
       ``cs setup`` segment entirely. The downstream ``cs install`` is
       self-sufficient — it adds the artifacts to PATH via ``cs install``
       itself, which is what the task actually needs.
    2. ``./cs install <coord>`` resolves artifacts from Maven Central. With
       no mirror env set, that hits ``repo1.maven.org`` directly. Setting
       ``COURSIER_MIRRORS`` to the GitHub-proxy-fronted coursier mirror
       lets coursier fetch through the proxy instead of the upstream.

    Both fixes are no-ops when the body does not mention ``cs `` / ``coursier``
    so existing tasks are unaffected.
    """

    if not re.search(r"(?i)\b(?:cs|coursier)\s+(?:setup|launch|install)\b", body):
        return body
    rewritten = body
    # Drop the ``cs setup --yes`` segment and the ``&&`` that precedes
    # it, since M23 doesn't expose a flag to skip its self-update, but
    # the downstream ``cs install`` is self-sufficient. We consume the
    # preceding operator but leave the trailing ``&&`` so the chain
    # stays intact (e.g. ``... && chmod cs && ./cs install scala``).
    rewritten = re.sub(
        r"\s*&&\s*\.?/?cs\s+setup\b[^\n&|;]*?(?=\s*&&|\s*\|\||\s*;|\s*\||$)",
        "",
        rewritten,
        count=1,
    )
    if "COURSIER_MIRRORS" not in rewritten:
        rewritten = f"COURSIER_MIRRORS={COURSIER_MIRROR_URL} {rewritten}"
    return rewritten


def WrapNpmInstallRun(body: str) -> str:
    """Add ``--registry`` to ``npm install`` so it pulls from the npmmirror.

    ``--registry`` is per-invocation and does not write to ``~/.npmrc`` or
    the project config, so the final image's npm settings are pristine.
    """

    if not re.search(r"(?i)\bnpm\s+(?:install|i|add|ci)\b", body):
        return body
    if "--registry" in body or "registry=" in body:
        return body
    rewritten = re.sub(
        r"(?i)(\bnpm\s+(?:install|i|add|ci))\b",
        rf"\1 --registry={NPM_MIRROR_REGISTRY}",
        body,
        count=1,
    )
    return _WithProxyUnset(rewritten)


def _WithProxyUnset(body: str) -> str:
    """Prepend the proxy-unset snippet unless it is already the first command.

    Multiple wraps (apt + curl + git + npm) may apply to the same RUN line.
    Each wrap would otherwise re-add the same ``unset ...;`` prefix; we
    detect that and skip it so the final RUN has a single ``unset`` once.
    """

    if body.lstrip().startswith(PROXY_UNSET_SHELL + ";"):
        return body
    return f"{PROXY_UNSET_SHELL}; {body}"


def InjectHuggingfaceEndpoint(text: str) -> str:
    """Insert an ``ENV HF_ENDPOINT=...`` line right after the first ``FROM``
    so ``huggingface_hub`` (and ``huggingface-cli``) fetch through the
    Chinese mirror at build time.

    Some SkillsBench tasks run ``python3 -c "from <pkg> import <Model>; ..."``
    to bake model weights into the image. The host's MITM proxy
    intermittently drops the SSL stream to ``huggingface.co``; the
    ``HF_ENDPOINT`` env var is honored by every official HuggingFace
    client, so a single ``ENV`` line redirects every subsequent ``RUN``
    (the env is inherited). Only used when ``--use-mirror`` is on so
    default builds stay byte-identical to the upstream Dockerfile.

    The ``ENV`` MUST come after a ``FROM`` (Docker requires ``FROM`` to be
    the first non-ARG instruction), so we splice it in immediately after
    the last ``FROM ... AS <stage>`` line. ARG-only Dockerfiles (a rare
    multi-stage preamble) get the ``ENV`` appended at the end instead.
    """

    lines = text.splitlines(keepends=True)
    lastFromIndex = -1
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.upper().startswith("FROM "):
            lastFromIndex = index
    if lastFromIndex == -1:
        return f"ENV HF_ENDPOINT={HF_MIRROR_ENDPOINT}\n" + text
    inserted = f"ENV HF_ENDPOINT={HF_MIRROR_ENDPOINT}\n"
    lines.insert(lastFromIndex + 1, inserted)
    return "".join(lines)


def RewritePackageManagerRuns(text: str) -> str:
    """Walk Dockerfile RUN lines and wrap apt/dnf/pip install commands.

    Heredoc bodies (``RUN <<EOF``) are passed through unchanged; the heredoc
    opener itself is not considered an apt/dnf/pip invocation and so is also
    not wrapped. Continuation lines (trailing ``\\``) are joined via
    :func:`LogicalDockerfileLines` so a multi-line RUN becomes one wrap site.
    """

    heredocPattern = re.compile(
        r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_.-]*)\1"
    )
    rendered: list[str] = []
    heredocDelimiter: str | None = None
    for logical in LogicalDockerfileLines(text):
        stripped = logical.strip()
        if heredocDelimiter is not None:
            rendered.append(logical + "\n")
            if stripped in {heredocDelimiter, f"-{heredocDelimiter}"}:
                heredocDelimiter = None
            continue
        if not stripped or stripped.startswith("#"):
            rendered.append(logical + "\n")
            continue
        try:
            tokens = shlex.split(stripped, comments=True, posix=True)
        except ValueError:
            rendered.append(logical + "\n")
            continue
        if not tokens:
            rendered.append(logical + "\n")
            continue
        instruction = tokens[0].upper()
        if instruction in {"RUN", "COPY"}:
            heredoc = heredocPattern.search(stripped)
            if heredoc is not None:
                heredocDelimiter = heredoc.group(2)
        if instruction != "RUN":
            rendered.append(logical + "\n")
            continue
        match = re.match(r"^(\s*)RUN\s+(.*)$", stripped, re.DOTALL)
        if match is None:
            rendered.append(logical + "\n")
            continue
        leadingWs = match.group(1)
        body = match.group(2)
        newBody = body
        # Apply every applicable wrap so a single RUN can mix managers
        # (e.g. apt install + pip install) and still get clean sources.
        if re.search(r"(?i)\b(?:apt-get|apt)\s+install\b", newBody):
            newBody = WrapAptInstallRun(newBody)
        if re.search(r"(?i)\b(?:dnf|yum)\s+(?:-y\s+)?install\b", newBody):
            newBody = WrapDnfInstallRun(newBody)
        if re.search(r"(?i)\bpip3?\s+install\b", newBody):
            newBody = WrapPipInstallRun(newBody)
        if re.search(
            r"(?i)\b(?:curl|wget)\s+(?:-{1,2}[A-Za-z-]+|https?://|\$\w|\"|')",
            newBody,
        ):
            newBody = WrapCurlWgetRun(newBody)
        if re.search(r"(?i)\bgit\s+clone\b", newBody):
            newBody = WrapGitCloneRun(newBody)
        if re.search(r"(?i)\bnpm\s+(?:install|i|add|ci)\b", newBody):
            newBody = WrapNpmInstallRun(newBody)
        if re.search(r"(?i)\b(?:cs|coursier)\s+(?:setup|launch|install)\b", newBody):
            newBody = WrapCoursierRun(newBody)
        if "deb.nodesource.com/setup_" in newBody:
            newBody = WrapNodeSourceSetupRun(newBody)
        if newBody == body:
            rendered.append(logical + "\n")
            continue
        rendered.append(f"{leadingWs}RUN {newBody}\n")
    return "".join(rendered)


def PrepareAptBaseImage(
    image: str,
    proxy: str,
    logDirectory: Path | None,
    retries: int = DEFAULT_BUILD_RETRIES,
    useMirror: bool = False,
) -> tuple[str, bool]:
    """Create one apt-indexed local base image for an external base."""

    # The base must include ca-certificates: ``ubuntu:24.04`` (and other
    # slim bases) ship without it, which means ``apt-get update`` cannot
    # verify the mirror's HTTPS certificate in downstream task builds.
    # Installing it here lets every task image that ``FROM`` this base
    # inherit the trust store.
    dockerfileContent = (
        f"FROM {image}\nUSER root\n"
        + "".join(f"ARG {variable}\n" for variable in PROXY_VARIABLES)
        + "RUN apt-get update && apt-get install -y ca-certificates "
        "&& rm -rf /var/lib/apt/lists/*\n"
    )
    if useMirror:
        # Re-route through the same tuna wrap as task images so the base
        # build does not depend on the flaky host proxy.
        dockerfileContent = RewritePackageManagerRuns(dockerfileContent)
    preparedImage = AptBaseImageName(image, dockerfileContent)
    if DockerImageExists(preparedImage):
        print(f"[apt] {image} already prepared as {preparedImage}")
        return preparedImage, True

    toolDirectory = ToolDirectory()
    toolDirectory.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporaryDirectory = tempfile.mkdtemp(prefix="apt-base-", dir=toolDirectory)
    temporaryPath = Path(temporaryDirectory)
    dockerfile = temporaryPath / "Dockerfile"
    dockerfile.write_text(dockerfileContent, encoding="utf-8")
    command = [
        "docker",
        "build",
        "--network=host",
        "--force-rm",
        "--file",
        str(dockerfile),
        "--tag",
        preparedImage,
        *ProxyBuildArguments(proxy),
        str(temporaryPath),
    ]
    print(f"[apt] preparing {image} ...")
    try:
        for attempt in range(1, retries + 1):
            try:
                result = RunCommand(
                    command,
                    env=ProxyEnvironment(proxy),
                    captureOutput=True,
                )
            except Exception as exc:  # noqa: BLE001 - retry transient runner errors
                PrintError(
                    f"[apt] preparing {image} attempt {attempt}/{retries} FAILED: {exc}"
                )
                if attempt < retries:
                    print(f"[apt] retrying {image}")
                    continue
                return preparedImage, False

            if result.returncode == 0 and DockerImageExists(preparedImage):
                print(f"[apt] {image} OK")
                return preparedImage, True

            logPath = WriteCommandLog(
                logDirectory,
                f"apt-{image}-attempt-{attempt}",
                result,
            )
            detail = OutputTail(CommandOutput(result))
            suffix = f"; log: {logPath}" if logPath is not None else ""
            PrintError(
                f"[apt] preparing {image} attempt {attempt}/{retries} "
                f"FAILED (exit {result.returncode}){suffix}"
            )
            if detail:
                PrintError(f"[apt] {image} last output:\n{detail}")
            if attempt < retries:
                print(f"[apt] retrying {image}")
        return preparedImage, False
    finally:
        shutil.rmtree(temporaryPath, ignore_errors=True)


def PrepareAptBaseImages(
    images: Iterable[str],
    proxy: str,
    jobs: int,
    logDirectory: Path | None,
    retries: int = DEFAULT_BUILD_RETRIES,
    useMirror: bool = False,
) -> dict[str, str]:
    prepared: dict[str, str] = {}
    imageList = sorted(set(images))
    if not imageList:
        return prepared
    print(
        f"[apt] preparing indexes once for {len(imageList)} base image(s) "
        f"with {min(jobs, len(imageList))} worker(s)"
    )
    with ThreadPoolExecutor(
        max_workers=min(jobs, len(imageList)),
        thread_name_prefix="skillsbench-apt",
    ) as executor:
        futures = {
            executor.submit(
                PrepareAptBaseImage,
                image,
                proxy,
                logDirectory,
                retries,
                useMirror,
            ): image
            for image in imageList
        }
        for future in as_completed(futures):
            image = futures[future]
            try:
                preparedImage, success = future.result()
            except Exception as exc:  # noqa: BLE001 - continue with other bases
                PrintError(f"[apt] preparing {image} FAILED: {exc}")
                continue
            if success:
                prepared[image] = preparedImage
    return prepared


def BuildTaskImage(
    taskPath: Path,
    proxy: str,
    *,
    aptUpdateMode: str = "once",
    aptBaseImages: Mapping[str, str] | None = None,
    logDirectory: Path | None = None,
    rebuild: bool = False,
    retries: int = DEFAULT_BUILD_RETRIES,
    useMirror: bool = False,
) -> bool:
    taskName = taskPath.name
    image = TaskImageName(taskPath)
    dockerfile = taskPath / "environment" / "Dockerfile"
    if not dockerfile.is_file():
        PrintError(f"[build] {taskName} has no environment/Dockerfile")
        return False
    if not rebuild and DockerImageExists(image):
        print(f"[build] {taskName} already has {image}")
        return True

    temporaryDockerfile: Path | None = None
    buildDockerfile = dockerfile
    try:
        dockerfileText = dockerfile.read_text(encoding="utf-8")
        rewritten = dockerfileText
        if useMirror:
            # Mirror wrapping happens regardless of the apt-base optimization:
            # some skills use apt without ``apt-get update`` (already satisfied
            # by the base) but the actual install step still goes through the
            # proxy and benefits from the rewrite.
            rewritten = RewritePackageManagerRuns(rewritten)
            # Redirect huggingface_hub to the Chinese mirror so build-time
            # ``python3 -c "from <pkg> import <Model>; ..."`` model
            # downloads survive the host's flaky SSL path to huggingface.co.
            rewritten = InjectHuggingfaceEndpoint(rewritten)
        if DockerfileHasAptUpdate(dockerfile):
            canReuseAptBase = aptUpdateMode == "never"
            if (
                aptUpdateMode == "once"
                and DockerfileAptUpdateCount(dockerfileText) == 1
                and not DockerfileHasExplicitPlatform(dockerfileText)
            ):
                requiredBases = DockerfileBaseImages(dockerfile)
                canReuseAptBase = bool(requiredBases) and requiredBases.issubset(
                    aptBaseImages or {}
                )
            if canReuseAptBase:
                rewritten = RemoveAptUpdateCommands(rewritten)
                if aptUpdateMode == "once":
                    rewritten = ReplaceDockerfileBaseImages(
                        rewritten, aptBaseImages or {}
                    )
        if rewritten != dockerfileText:
            toolDirectory = ToolDirectory()
            toolDirectory.mkdir(mode=0o700, parents=True, exist_ok=True)
            fileDescriptor, temporaryName = tempfile.mkstemp(
                prefix=f".{SafeImageName(taskName)}-",
                suffix=".Dockerfile",
                dir=toolDirectory,
            )
            os.close(fileDescriptor)
            temporaryDockerfile = Path(temporaryName)
            temporaryDockerfile.write_text(rewritten, encoding="utf-8")
            buildDockerfile = temporaryDockerfile

        print(f"[build] {taskName} ...")
    except Exception:
        if temporaryDockerfile is not None:
            temporaryDockerfile.unlink(missing_ok=True)
        raise

    command = [
        "docker",
        "build",
        "--network=host",
        "--force-rm",
        "--file",
        str(buildDockerfile),
        "--tag",
        image,
        *ProxyBuildArguments(proxy),
        str(taskPath / "environment"),
    ]
    try:
        for attempt in range(1, retries + 1):
            try:
                result = RunCommand(
                    command,
                    env=ProxyEnvironment(proxy),
                    captureOutput=True,
                )
            except Exception as exc:  # noqa: BLE001 - retry transient runner errors
                PrintError(
                    f"[build] {taskName} attempt {attempt}/{retries} FAILED: {exc}"
                )
                if attempt < retries:
                    print(f"[build] retrying {taskName}")
                    continue
                return False

            if result.returncode == 0 and DockerImageExists(image):
                print(f"[build] {taskName} OK")
                return True

            logPath = WriteCommandLog(
                logDirectory,
                f"{taskName}-attempt-{attempt}",
                result,
            )
            detail = OutputTail(CommandOutput(result))
            suffix = f" (exit {result.returncode})"
            if logPath is not None:
                suffix += f"; log: {logPath}"
            PrintError(
                f"[build] {taskName} attempt {attempt}/{retries} FAILED{suffix}"
            )
            if detail:
                PrintError(f"[build] {taskName} last output:\n{detail}")
            if attempt < retries:
                print(f"[build] retrying {taskName}")
        return False
    finally:
        if temporaryDockerfile is not None:
            temporaryDockerfile.unlink(missing_ok=True)


def BuildTaskImages(
    tasks: list[Path],
    proxy: str,
    jobs: int,
    *,
    aptUpdateMode: str,
    aptBaseImages: Mapping[str, str],
    logDirectory: Path | None,
    rebuild: bool,
    retries: int,
    useMirror: bool = False,
) -> tuple[list[Path], list[str], list[Path]]:
    """Build task images concurrently and return success, failure, skipped."""

    successful: list[Path] = []
    failed: list[str] = []
    skipped: list[Path] = []
    pending: list[Path] = []
    for taskPath in tasks:
        image = TaskImageName(taskPath)
        if not rebuild and DockerImageExists(image):
            print(f"[build] {taskPath.name} already has {image} (skipping)")
            successful.append(taskPath)
            skipped.append(taskPath)
        else:
            pending.append(taskPath)

    if not pending:
        return successful, failed, skipped

    workerCount = min(jobs, len(pending))
    print(f"[build] building {len(pending)} task image(s) with {workerCount} worker(s)")
    with ThreadPoolExecutor(
        max_workers=workerCount,
        thread_name_prefix="skillsbench-build",
    ) as executor:
        futures = {
            executor.submit(
                BuildTaskImage,
                taskPath,
                proxy,
                aptUpdateMode=aptUpdateMode,
                aptBaseImages=aptBaseImages,
                logDirectory=logDirectory,
                rebuild=rebuild,
                retries=retries,
                useMirror=useMirror,
            ): taskPath
            for taskPath in pending
        }
        for future in as_completed(futures):
            taskPath = futures[future]
            try:
                built = future.result()
            except Exception as exc:  # noqa: BLE001 - continue with other tasks
                PrintError(f"[build] {taskPath.name} FAILED: {exc}")
                built = False
            if built:
                successful.append(taskPath)
            else:
                failed.append(taskPath.name)
    return successful, failed, skipped


def SetPrebuiltImage(frontmatter: dict[str, Any], image: str) -> None:
    sandbox = frontmatter.get("sandbox")
    if isinstance(sandbox, dict):
        if "docker_image" in sandbox:
            sandbox["docker_image"] = image
        if "image" in sandbox:
            sandbox["image"] = image
    if "image" in frontmatter:
        frontmatter["image"] = image
    elif not (
        isinstance(sandbox, dict)
        and ("docker_image" in sandbox or "image" in sandbox)
    ):
        # BenchFlow's supported shorthand. Its parser normalizes this to
        # sandbox.docker_image, while keeping the upstream task readable.
        frontmatter["image"] = image


def TaskComposeUsesMainImageVariable(taskPath: Path) -> bool:
    for composePath in ComposeFiles(taskPath / "environment"):
        document = ReadYaml(composePath)
        if not isinstance(document, dict):
            continue
        services = document.get("services")
        main = services.get("main") if isinstance(services, dict) else None
        image = main.get("image") if isinstance(main, dict) else None
        if isinstance(image, str) and "${MAIN_IMAGE_NAME" in image:
            return True
    return False


def PatchTaskImage(taskPath: Path, image: str) -> None:
    taskFile = taskPath / "task.md"
    frontmatter, body = ReadTaskFrontmatter(taskFile)
    SetPrebuiltImage(frontmatter, image)
    if TaskComposeUsesMainImageVariable(taskPath):
        sandbox = frontmatter.get("sandbox")
        if sandbox is None:
            sandbox = {}
            frontmatter["sandbox"] = sandbox
        if not isinstance(sandbox, dict):
            raise RuntimeError(f"sandbox in {taskFile} must be a mapping")
        environment = sandbox.setdefault("env", {})
        if not isinstance(environment, dict):
            raise RuntimeError(f"sandbox.env in {taskFile} must be a mapping")
        # BenchFlow supplies MAIN_IMAGE_NAME for the normal build compose
        # overlay. A task compose file loaded later can otherwise override the
        # prebuilt image with that generated name even though build was skipped.
        environment["MAIN_IMAGE_NAME"] = image
    rendered = RenderTaskDocument(frontmatter, body)
    if rendered == taskFile.read_text(encoding="utf-8"):
        print(f"[task] {taskPath.name} already points to {image}")
        return
    BackupOnce(taskFile)
    WriteTextAtomically(taskFile, rendered)
    print(f"[task] {taskPath.name} -> {image}")


def PatchSuccessfulTasks(successfulTasks: list[Path]) -> list[str]:
    failures: list[str] = []
    for taskPath in successfulTasks:
        try:
            PatchTaskImage(taskPath, TaskImageName(taskPath))
        except Exception as exc:  # noqa: BLE001 - continue to patch other tasks
            PrintError(f"[task] {taskPath.name} FAILED: {exc}")
            failures.append(taskPath.name)
    return failures


def PatchBenchFlowCompose(composePath: Path) -> None:
    document = ReadYaml(composePath)
    if not isinstance(document, dict):
        raise RuntimeError(f"BenchFlow compose root must be a mapping: {composePath}")
    services = document.get("services")
    if not isinstance(services, dict) or not isinstance(services.get("main"), dict):
        raise RuntimeError(f"BenchFlow compose has no services.main: {composePath}")
    main = services["main"]
    capAdd = main.get("cap_add")
    if capAdd is None:
        capAdd = []
        main["cap_add"] = capAdd
    elif isinstance(capAdd, str):
        capAdd = [capAdd]
        main["cap_add"] = capAdd
    elif not isinstance(capAdd, list):
        raise RuntimeError("services.main.cap_add must be a list or string")
    if "NET_ADMIN" not in capAdd:
        capAdd.append("NET_ADMIN")

    environment = main.get("environment")
    if environment is None:
        environment = {}
        main["environment"] = environment
    if isinstance(environment, dict):
        for variable in PROXY_VARIABLES:
            environment[variable] = ""
    elif isinstance(environment, list):
        present: set[str] = set()
        for index, entry in enumerate(environment):
            if not isinstance(entry, str):
                continue
            variable = entry.split("=", 1)[0]
            if variable in PROXY_VARIABLES:
                environment[index] = f"{variable}="
                present.add(variable)
        for variable in PROXY_VARIABLES:
            if variable not in present:
                environment.append(f"{variable}=")
    else:
        raise RuntimeError("services.main.environment must be a mapping or list")

    yaml = LoadYaml()
    rendered = yaml.safe_dump(
        document,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )
    if not rendered.endswith("\n"):
        rendered += "\n"
    original = composePath.read_text(encoding="utf-8")
    if rendered == original:
        print(f"[compose] {composePath} already patched")
        return
    BackupOnce(composePath)
    WriteTextAtomically(composePath, rendered)
    print(f"[compose] patched {composePath}")


def RemoveBuiltTaskImages() -> bool:
    if not shutil.which("docker"):
        PrintError("docker is not on PATH; could not remove task images")
        return False
    result = RunCommand(
        ["docker", "image", "ls", "--format", "{{.Repository}}:{{.Tag}}"],
        captureOutput=True,
    )
    if result.returncode != 0:
        PrintError(f"could not list Docker images: {result.stderr.strip()}")
        return False
    images = sorted(
        {
            line.strip()
            for line in result.stdout.splitlines()
            if line.strip().startswith((PREBUILT_PREFIX, APT_BASE_PREFIX))
        }
    )
    success = True
    for image in images:
        removed = RunCommand(["docker", "image", "rm", image], captureOutput=True)
        if removed.returncode != 0:
            PrintError(f"could not remove {image}: {removed.stderr.strip()}")
            success = False
        else:
            print(f"[clear] removed {image}")
    if not images:
        print("[clear] no kvbench-skillsbench images found")
    return success


def RestoreTaskBackups(skillsbenchPath: Path) -> bool:
    success = True
    backupPaths = sorted((skillsbenchPath / "tasks").rglob(f"task.md{BACKUP_SUFFIX}"))
    for backupPath in backupPaths:
        taskPath = backupPath.with_name("task.md")
        try:
            os.replace(backupPath, taskPath)
            print(f"[clear] restored {taskPath}")
        except OSError as exc:
            PrintError(f"could not restore {taskPath}: {exc}")
            success = False
    if not backupPaths:
        print("[clear] no task.md backups found")
    return success


def RestoreBenchFlowCompose(composePath: Path | None) -> bool:
    if composePath is None:
        PrintError("could not locate BenchFlow compose; compose backup was not restored")
        return False
    backupPath = Path(f"{composePath}{BACKUP_SUFFIX}")
    if not backupPath.exists():
        print("[clear] no BenchFlow compose backup found")
        return True
    try:
        os.replace(backupPath, composePath)
    except OSError as exc:
        PrintError(f"could not restore {composePath}: {exc}")
        return False
    print(f"[clear] restored {composePath}")
    return True


def RemoveTemporaryTools() -> bool:
    path = ToolDirectory()
    if not path.exists():
        print("[clear] no downloaded crane/temp files found")
        return True
    try:
        shutil.rmtree(path)
    except OSError as exc:
        PrintError(f"could not remove temporary tool directory {path}: {exc}")
        return False
    print(f"[clear] removed {path}")
    return True


def CreateBuildLogDirectory(logDirectory: Path | None) -> Path:
    if logDirectory is not None:
        logDirectory.mkdir(mode=0o700, parents=True, exist_ok=True)
        return logDirectory
    root = ToolDirectory() / "logs"
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="run-", dir=root))


def Initialize(
    proxy: str,
    skillsbenchPath: Path,
    composePath: Path,
    *,
    jobs: int = DEFAULT_BUILD_JOBS,
    retries: int = DEFAULT_BUILD_RETRIES,
    skipAptUpdate: bool = False,
    rebuild: bool = False,
    logDirectory: Path | None = None,
    useMirror: bool = False,
) -> int:
    if not shutil.which("docker"):
        raise FileNotFoundError("docker is not on PATH")
    if jobs < 1:
        raise ValueError("--jobs must be at least 1")
    if retries < 1:
        raise ValueError("--retries must be at least 1")
    dockerInfo = RunCommand(["docker", "info"], captureOutput=True)
    if dockerInfo.returncode != 0:
        raise RuntimeError(
            "Docker daemon is not available: "
            f"{(dockerInfo.stderr or dockerInfo.stdout).strip()}"
        )

    tasks = DiscoverTasks(skillsbenchPath)
    if not tasks:
        raise RuntimeError(f"no SkillsBench tasks found under {skillsbenchPath / 'tasks'}")
    tasksToBuild = [
        taskPath
        for taskPath in tasks
        if rebuild or not DockerImageExists(TaskImageName(taskPath))
    ]
    skippedCount = len(tasks) - len(tasksToBuild)
    if skippedCount:
        print(f"[build] {skippedCount} existing task image(s) will be skipped")

    buildLogDirectory = CreateBuildLogDirectory(logDirectory)
    print(f"[log] failed build output: {buildLogDirectory}")

    images, discoveryErrors = RequiredBaseImages(tasksToBuild)
    for error in discoveryErrors:
        PrintError(error)
    baseCounts = PullMissingBaseImages(images, proxy, jobs)

    aptBaseImages: dict[str, str] = {}
    aptDiscoveryErrors: list[str] = []
    aptImageCount = 0
    if skipAptUpdate:
        print(
            "[apt] skipping apt update in task Dockerfiles; this may fail "
            "when the base image has no package indexes"
        )
    else:
        aptImages, aptDiscoveryErrors = RequiredAptBaseImages(tasksToBuild)
        aptImageCount = len(aptImages)
        for error in aptDiscoveryErrors:
            PrintError(error)
        availableAptImages = {
            image for image in aptImages if DockerImageExists(image)
        }
        for image in sorted(aptImages - availableAptImages):
            PrintError(
                f"[apt] base {image} is unavailable; tasks using it will "
                "fall back to their original apt-get update"
            )
        aptBaseImages = PrepareAptBaseImages(
            availableAptImages,
            proxy,
            jobs,
            buildLogDirectory,
            retries,
            useMirror=useMirror,
        )

    successfulTasks, failedTasks, skippedTasks = BuildTaskImages(
        tasks,
        proxy,
        jobs,
        aptUpdateMode="never" if skipAptUpdate else "once",
        aptBaseImages=aptBaseImages,
        logDirectory=buildLogDirectory,
        rebuild=rebuild,
        retries=retries,
        useMirror=useMirror,
    )
    taskPatchFailures = PatchSuccessfulTasks(successfulTasks)
    failedTasks.extend(taskPatchFailures)

    composeFailure = False
    try:
        PatchBenchFlowCompose(composePath)
    except Exception as exc:  # noqa: BLE001 - report all task failures first
        PrintError(f"[compose] FAILED: {exc}")
        composeFailure = True

    print("\nSummary:")
    print("Base images:")
    print(f"  existing: {baseCounts['existing']}")
    print(f"  pulled: {baseCounts['pulled']}")
    print(
        f"  failed: {baseCounts['failed'] + len(discoveryErrors) + len(aptDiscoveryErrors)}"
    )
    print("APT base indexes:")
    print(f"  requested: {aptImageCount}")
    print(f"  prepared/reused: {len(aptBaseImages)}")
    print(f"  fallback: {aptImageCount - len(aptBaseImages)}")
    print("Task images:")
    print(f"  succeeded: {len(successfulTasks) - len(skippedTasks)}")
    print(f"  skipped: {len(skippedTasks)}")
    print(f"  failed: {len(failedTasks)}")
    if failedTasks:
        print("Failed tasks:")
        for taskName in sorted(set(failedTasks)):
            print(f"  {taskName}")
    if composeFailure:
        print("BenchFlow compose: FAILED")

    return int(bool(baseCounts["failed"] or discoveryErrors or failedTasks or composeFailure))


def Clear(skillsbenchPath: Path) -> int:
    success = RestoreTaskBackups(skillsbenchPath)
    try:
        composePath = LocateBenchFlowCompose()
    except Exception as exc:  # noqa: BLE001 - still clean task/tools/images
        PrintError(str(exc))
        composePath = None
        success = False
    success = RestoreBenchFlowCompose(composePath) and success
    success = RemoveBuiltTaskImages() and success
    success = RemoveTemporaryTools() and success
    return 0 if success else 1


def PositiveInteger(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def BuildArgumentParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        nargs="?",
        choices=("clear",),
        help="use 'clear' to restore backups and remove only kvbench images",
    )
    parser.add_argument(
        "--proxy",
        help="host HTTP(S) proxy used for preflight, crane, and Docker builds",
    )
    parser.add_argument(
        "--skillsbench-path",
        help="SkillsBench repository (default: AgentBenchFlow.SkillsBenchRepo in config.yaml)",
    )
    parser.add_argument(
        "--jobs",
        type=PositiveInteger,
        default=DEFAULT_BUILD_JOBS,
        help=f"maximum concurrent base pulls/builds (default: {DEFAULT_BUILD_JOBS})",
    )
    parser.add_argument(
        "--retries",
        type=PositiveInteger,
        default=DEFAULT_BUILD_RETRIES,
        help=(
            "attempt count for each apt-base/task build (default: "
            f"{DEFAULT_BUILD_RETRIES})"
        ),
    )
    parser.add_argument(
        "--skip-apt-update",
        "--no-apt-update",
        dest="skip_apt_update",
        action="store_true",
        help=(
            "remove apt-get update from temporary Dockerfiles; unsafe for "
            "fresh base images without apt indexes"
        ),
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="rebuild images even when the target kvbench tag already exists",
    )
    parser.add_argument(
        "--use-mirror",
        dest="use_mirror",
        action="store_true",
        help=(
            "rewrite apt/dnf/pip install commands in temporary Dockerfiles "
            "to use TUNA mirrors; original sources.list, yum.repos.d, and "
            "pip config are restored before each RUN ends so the final "
            "image is byte-identical to one built with the upstream archives"
        ),
    )
    parser.add_argument(
        "--log-dir",
        help="directory for complete failed docker build logs (default: temporary directory)",
    )
    return parser


def Main(argv: list[str] | None = None) -> int:
    args = BuildArgumentParser().parse_args(argv)
    try:
        if args.command == "clear":
            skillsbenchPath = ResolveSkillsbenchPath(args.skillsbench_path)
            return Clear(skillsbenchPath)
        if not args.proxy:
            raise ValueError("--proxy is required for initialization")
        ValidateProxy(args.proxy)
        skillsbenchPath = ResolveSkillsbenchPath(args.skillsbench_path)
        composePath = LocateBenchFlowCompose()
        print(f"[skillsbench] {skillsbenchPath}")
        print(f"[benchflow] {composePath}")
        logDirectory = (
            Path(args.log_dir).expanduser().resolve() if args.log_dir else None
        )
        return Initialize(
            args.proxy,
            skillsbenchPath,
            composePath,
            jobs=args.jobs,
            retries=args.retries,
            skipAptUpdate=args.skip_apt_update,
            rebuild=args.rebuild,
            logDirectory=logDirectory,
            useMirror=args.use_mirror,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        PrintError(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(Main())
