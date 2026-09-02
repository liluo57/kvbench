"""apply_benchflow_patches.py — re-apply KVBench smoke-run patches.

This single script re-applies every non-kvbench change accumulated during
the smoke-run integration work, so a clean `pip install --upgrade benchflow`
(or a fresh container from a freshly built image) can be brought back to a
known-good state with one command.

Patches applied:

  1. benchflow/agents/registry.py: pi-acp pinned to 0.0.32 (the 0.0.33 release
     is a regression: its `case "agent_settled"` handler is dead code because
     no published @mariozechner/pi-coding-agent version emits that event).
  2. benchflow/agents/registry.py: new module-level constants
     `_PI_AUTONOMOUS_DIRECTIVE` and `_PI_AUTONOMY_FILE` plus a reworked
     `pi-acp` install_cmd that writes the directive file and sed-patches
     pi-acp's dist/index.js to inject `--append-system-prompt` into pi's
     spawn args. This makes pi work autonomously (no user confirmation)
     in non-interactive KVBench smoke runs.
  3. benchflow/sandbox/docker.py: drop `--rmi all` from the
     `compose down` in `stop()`. The prebuilt task image is expensive to
     rebuild (~25 min on slow apt) and we want to reuse it across runs.

All three patches are idempotent: each looks for a marker string (a comment
or a distinctive post-patch substring) first and aborts that step with a
"[skip]" message if the marker is found.  Run again after a benchflow
upgrade and the markers will be absent — the script re-applies everything
from scratch.

Usage:
    python3 scripts/apply_benchflow_patches.py           # apply all
    python3 scripts/apply_benchflow_patches.py --check    # exit 1 if any missing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BENCHFLOW_ROOT = Path(
    "/data/lyh/.local/cpython-3.13.9/lib/python3.13/site-packages/benchflow"
)
REGISTRY_PATH = BENCHFLOW_ROOT / "agents" / "registry.py"
DOCKER_PATH = BENCHFLOW_ROOT / "sandbox" / "docker.py"

# Marker comments — also serve as the patch "fingerprint" for idempotency.
# Re-applying the patch on a file that already has the marker is a no-op.
MARKER_PIACP_VERSION = "# Patched for KVBench smoke runs: pi-acp 0.0.32 pin"
MARKER_AUTONOMY = "# Patched for KVBench smoke runs: pi autonomous directive"
MARKER_NO_RMI = "# Patched for kvbench smoke runs: do NOT pass `--rmi all`"

# Belt-and-braces idempotency markers: post-patch substrings that are unique
# to the patched state. The session's hand-edits left the actual code
# changes in place but did not always preserve the marker comments, so
# checking both gives us a robust "already applied" check.
POST_PATCH_PIACP = "pi-acp@0.0.32"
POST_PATCH_AUTONOMY = "_PI_AUTONOMOUS_DIRECTIVE = ("
POST_PATCH_NO_RMI = "--volumes"  # this string only appears in the patched block

# Source-of-truth for the autonomous-directive text.  Kept verbatim with
# the user-supplied wording.
_AUTONOMY_LINES = [
    '    "You are running in a **non-interactive benchmark environment**. "',
    '    "No user will reply to follow-up questions or approve your plan.\\n\\n"',
    '    "Work autonomously until the task is actually completed.\\n\\n"',
    '    "* Do not stop after analysis, diagnosis, summaries, or proposed next steps.\\n"',
    '    "* Do not ask for confirmation or wait for user input.\\n"',
    '    "* If multiple approaches are possible, choose one and proceed.\\n"',
    '    "* If information is missing, inspect the available tools/files and make reasonable assumptions.\\n"',
    '    "* If there is still a concrete action or tool call that can advance the task, take it.\\n\\n"',
    '    "Only stop when the task is completed, or when completion is genuinely impossible "',
    '    "with the available information and tools.\\n\\n"',
    '    "**Planning what to do is not completion.**"',
]


# ---------------------------------------------------------------------------
# Patch 1+2: registry.py — pi-acp version pin + autonomous-directive injection
# ---------------------------------------------------------------------------

# Anchor: the line _just before_ the existing _NODE_INSTALL block comment.
# We insert the directive constants above this anchor.
_REGISTRY_ANCHOR = (
    "# Node 22.20.0 supports OpenClaw 2026.6.9. Keep their pin pair in sync."
)


def _build_autonomy_block() -> str:
    """Return the multi-line text inserted above _REGISTRY_ANCHOR."""
    return (
        MARKER_AUTONOMY
        + "\n"
        + "# Appended to pi's default system prompt on every turn so pi works"
        + "\n"
        + '# autonomously during KVBench smoke runs (no human in the loop).\n'
        + "_PI_AUTONOMOUS_DIRECTIVE = (\n"
        + "\n".join(_AUTONOMY_LINES)
        + "\n)\n"
        + '_PI_AUTONOMY_FILE = "/opt/benchflow/share/pi-autonomous-directive.md"\n'
        + "\n"
    )


# Anchor: the upstream line that installs pi-acp via _js_agent_install.
# We replace it with a version-pinned install plus a directive-file write
# and a sed-patch of pi-acp's spawn() args.
_OLD_PIACP_LINE = (
    "            f\"{_js_agent_install('pi-acp', 'pi-acp')} && \""
)


def _build_piacp_block() -> str:
    """Return the multi-line install_cmd replacement for pi-acp."""
    return (
        MARKER_PIACP_VERSION
        + "\n"
        + "            f\"{_js_agent_install('pi-acp', 'pi-acp@0.0.32')} && \"\n"
        + "            f\"mkdir -p {shlex.quote(str(Path(_PI_AUTONOMY_FILE).parent))} && \"\n"
        + "            f\"printf '%s\\\\n' {shlex.quote(_PI_AUTONOMOUS_DIRECTIVE)}"
        + " > {shlex.quote(_PI_AUTONOMY_FILE)} && \"\n"
        + "            f\"sed -i.bak "
        + "\"'s|if (params.sessionPath) args.push(\\\"--session\\\", params.sessionPath);|\"\n"
        + "            f\"if (true) args.push(\\\"--append-system-prompt\\\", \\\"{_PI_AUTONOMY_FILE}\\\");\"\n"
        + "            f\" if (params.sessionPath) args.push(\\\"--session\\\", params.sessionPath);|' \"\n"
        + "            f\"{_BENCHFLOW_JS_AGENT_PREFIX}/lib/node_modules/pi-acp/dist/index.js && \"\n"
    )


def _has_autonomy_patch(text: str) -> bool:
    """True iff registry.py has the autonomy-directive block (either via
    marker comment or via the post-patch constant declaration)."""
    return MARKER_AUTONOMY in text or POST_PATCH_AUTONOMY in text


def _has_piacp_pin(text: str) -> bool:
    """True iff registry.py has the pi-acp 0.0.32 pin (either via marker
    comment or via the post-patch version string)."""
    return MARKER_PIACP_VERSION in text or POST_PATCH_PIACP in text


def patch_registry() -> str:
    """Apply patches 1+2 to registry.py. Idempotent."""
    if not REGISTRY_PATH.exists():
        return f"[registry.py] not found at {REGISTRY_PATH}"
    text = REGISTRY_PATH.read_text()
    msgs: list[str] = []

    # Sub-patch: autonomy constants
    if _has_autonomy_patch(text):
        msgs.append("[registry.py] autonomous-directive constants already present, skip")
    else:
        if _REGISTRY_ANCHOR not in text:
            return (
                f"[registry.py] anchor not found ({_REGISTRY_ANCHOR!r}); "
                "registry layout drifted — manual patch needed"
            )
        text = text.replace(
            _REGISTRY_ANCHOR,
            _build_autonomy_block() + _REGISTRY_ANCHOR,
            1,
        )
        msgs.append("[registry.py] injected autonomous-directive constants above _NODE_INSTALL")

    # Sub-patch: pi-acp 0.0.32 + install_cmd rewrite
    if _has_piacp_pin(text):
        msgs.append("[registry.py] pi-acp 0.0.32 pin already applied, skip")
    else:
        if _OLD_PIACP_LINE not in text:
            return (
                "[registry.py] pi-acp install_cmd anchor not found — "
                "upstream may have rewritten the install block; manual patch needed"
            )
        text = text.replace(_OLD_PIACP_LINE, _build_piacp_block(), 1)
        msgs.append(
            "[registry.py] rewrote pi-acp install_cmd: pinned 0.0.32, "
            "writes directive file, sed-patches dist/index.js"
        )

    REGISTRY_PATH.write_text(text)
    return "\n".join(msgs)


def check_registry() -> bool:
    """Return True iff all registry patches are present."""
    if not REGISTRY_PATH.exists():
        return False
    text = REGISTRY_PATH.read_text()
    return _has_autonomy_patch(text) and _has_piacp_pin(text)


# ---------------------------------------------------------------------------
# Patch 3: docker.py — drop --rmi all from compose down
# ---------------------------------------------------------------------------

# The upstream `elif delete:` branch has a unique substring: "down", "--rmi", "all"
# across multiple lines. After the patch, the same branch has "down", "--volumes",
# "--remove-orphans" with NO "--rmi all".
_DOCKER_OLD_BLOCK = (
    '                        "down",\n'
    '                        "--rmi",\n'
    '                        "all",'
)
_DOCKER_NEW_BLOCK = (
    '                        "down",\n'
    '                        "--volumes",\n'
    '                        "--remove-orphans",\n'
    '                        "-t",\n'
    '                        "5",'
)
# Anchor: line just after `elif delete:`. We prepend the marker + a multi-line
# explanation comment that documents why the patch exists.
_DOCKER_OLD_IF = "            elif delete:\n"
_DOCKER_NEW_IF = (
    "            elif delete:\n"
    "                " + MARKER_NO_RMI + "\n"
    '                # `compose down --rmi all` deletes every image used\n'
    '                # by the project, including the prebuilt task image\n'
    '                # (kvbench-skillsbench/<task>:latest) we want to reuse\n'
    '                # across runs. Containers + volumes + orphans are\n'
    '                # still cleaned; only the image removal is skipped.\n'
)


def _has_no_rmi_patch(text: str) -> bool:
    """True iff docker.py has the no-rmi patch (either via marker comment
    or via the post-patch `--volumes` block)."""
    return MARKER_NO_RMI in text or POST_PATCH_NO_RMI in text


def patch_docker() -> str:
    """Apply patch 3 to docker.py. Idempotent."""
    if not DOCKER_PATH.exists():
        return f"[docker.py] not found at {DOCKER_PATH}"
    text = DOCKER_PATH.read_text()
    if _has_no_rmi_patch(text):
        return "[docker.py] --rmi all already removed, skip"
    if _DOCKER_OLD_BLOCK not in text:
        return (
            "[docker.py] upstream --rmi marker not found — "
            "either already patched or layout changed"
        )
    text = text.replace(_DOCKER_OLD_BLOCK, _DOCKER_NEW_BLOCK, 1)
    text = text.replace(_DOCKER_OLD_IF, _DOCKER_NEW_IF, 1)
    DOCKER_PATH.write_text(text)
    return "[docker.py] dropped --rmi all from compose down; inserted marker comment"


def check_docker() -> bool:
    if not DOCKER_PATH.exists():
        return False
    return _has_no_rmi_patch(DOCKER_PATH.read_text())


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else "apply_benchflow_patches"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if any expected patch marker is missing; do not modify files",
    )
    args = parser.parse_args(argv)

    if args.check:
        missing: list[str] = []
        if not check_registry():
            missing.append("registry.py: pi-acp pin + autonomous directive")
        if not check_docker():
            missing.append("docker.py: --rmi all removal")
        if missing:
            print("Missing patches:", file=sys.stderr)
            for m in missing:
                print(f"  - {m}", file=sys.stderr)
            return 1
        print("All KVBench smoke-run patches present.")
        return 0

    # Apply path
    print(patch_registry())
    print(patch_docker())
    return 0


if __name__ == "__main__":
    sys.exit(main())
