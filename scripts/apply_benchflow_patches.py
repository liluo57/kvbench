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
  4. benchflow/agents/registry.py: configure Pi's provider request timeout
     to 3 hours and disable SDK-level timeout retries. The OpenAI-compatible
     client defaults to 10 minutes, while a single vLLM generation in KVBench
     can legitimately take longer; retries create duplicate in-flight
     requests against the endpoint.
  5. benchflow/agents/registry.py: prepend a `sed` step to the pi-acp
     install_cmd that rewrites `/etc/apt/sources.list.d/*.sources` and
     `/etc/apt/sources.list` from `deb.debian.org` to
     `mirrors.tuna.tsinghua.edu.cn` before the bootstrap's
     `apt-get update` runs. SkillsBench task images ship the official
     Debian mirror; from China the 9 MB trixie Packages.xz downloads at
     ~8 KB/s and the 900 s install timeout can be hit before
     apt-get update finishes. Tuna serves the same files from a
     Tsinghua CDN mirror (~50 MB/s from China).
  6. benchflow/providers/litellm_config.py + litellm_runtime.py: add a
     `BENCHFLOW_LITELLM_NO_AUTH=1` opt-in that forces the LiteLLM
     proxy master_key to empty, disabling the per-request API-key
     check. With no master_key, the auth path's `master_key is None`
     branch returns a UserAPIKeyAuth(INTERNAL_USER) immediately and
     never queries the prisma DB. KVBench's smoke runs use a vLLM
     endpoint that does not ship the `prisma` Python wheel, so any
     request whose auth cache expires (or was never warmed) hits
     `import prisma` → ModuleNotFoundError → 500 Internal Server
     Error, freezing the agent mid-turn until taskTimeout fires.

All patches are idempotent: each looks for a marker string (a comment
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
LITELLM_CONFIG_PATH = BENCHFLOW_ROOT / "providers" / "litellm_config.py"
LITELLM_RUNTIME_PATH = BENCHFLOW_ROOT / "providers" / "litellm_runtime.py"

# Marker comments — also serve as the patch "fingerprint" for idempotency.
# Re-applying the patch on a file that already has the marker is a no-op.
MARKER_PIACP_VERSION = "# Patched for KVBench smoke runs: pi-acp 0.0.32 pin"
MARKER_AUTONOMY = "# Patched for KVBench smoke runs: pi autonomous directive"
MARKER_NO_RMI = "# Patched for kvbench smoke runs: do NOT pass `--rmi all`"
MARKER_APT_MIRROR = "# Patched for KVBench smoke runs: rewrite apt mirror to tuna"
MARKER_PI_TIMEOUT = "# Patched for KVBench smoke runs: extend Pi provider timeout"

# Belt-and-braces idempotency markers: post-patch substrings that are unique
# to the patched state. The session's hand-edits left the actual code
# changes in place but did not always preserve the marker comments, so
# checking both gives us a robust "already applied" check.
POST_PATCH_PIACP = "pi-acp@0.0.32"
POST_PATCH_AUTONOMY = "_PI_AUTONOMOUS_DIRECTIVE = ("
POST_PATCH_NO_RMI = "--volumes"  # this string only appears in the patched block
POST_PATCH_PI_TIMEOUT = '"timeoutMs": 10800000'
POST_PATCH_LITELLM_NO_AUTH_CONFIG = (
    '"general_settings": ({"master_key": master_key} if master_key else {})'
)
POST_PATCH_LITELLM_NO_AUTH_RUNTIME = (
    'os.environ.get("BENCHFLOW_LITELLM_NO_AUTH")'
)

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


# ---------------------------------------------------------------------------
# Patch 4: registry.py — rewrite deb.debian.org → tuna before apt-get update
# ---------------------------------------------------------------------------
#
# SkillsBench task images ship /etc/apt/sources.list.d/debian.sources pointing
# at deb.debian.org (the official EU/US mirror). From China the 9MB
# trixie/main Packages.xz downloads at ~8 KB/s — slow enough that the 900s
# install timeout can be hit before apt-get update finishes.
#
# We prepend a sed step to the start of the pi-acp install_cmd so it rewrites
# both /etc/apt/sources.list.d/*.sources and /etc/apt/sources.list to the
# Tsinghua mirror BEFORE _js_agent_install's apt-get update runs.
#
# The `; true` guard and `2>/dev/null` make this safe to run on images where
# either file is missing.

# Anchor: the FIRST line of the pi-acp install_cmd (the `_js_agent_install('pi', ...)`
# call). Prepending the sed here means it runs before any apt-get update
# inside _js_agent_install.
_PIACP_FIRST_LINE = (
    "            f\"{_js_agent_install('pi', '@mariozechner/pi-coding-agent')} && \""
)

# Anchor: the pinned pi-acp install command. The settings merge is placed
# immediately after it so the generated launcher sees the setting at runtime.
_PIACP_VERSIONED_LINE = (
    "            f\"{_js_agent_install('pi-acp', 'pi-acp@0.0.32')} && \"\n"
)


def _has_pi_timeout_patch(text: str) -> bool:
    """True iff the Pi provider timeout/retry setting is installed."""
    return MARKER_PI_TIMEOUT in text or POST_PATCH_PI_TIMEOUT in text


def _build_pi_timeout_block() -> str:
    """Return the settings merge inserted after the pi-acp installation."""
    mutator = (
        'd.setdefault("retry", {}).setdefault("provider", {}).update('
        '{"timeoutMs": 10800000, "maxRetries": 0})'
    )
    return (
        "            # Pi/OpenAI defaults to a 10-minute request timeout. A "
        "single KVBench generation can exceed that, and retrying it creates "
        "a second live request against the endpoint.\n"
        + "            "
        + MARKER_PI_TIMEOUT
        + "\n"
        + "            f\"{_json_settings_merge('/home/agent/.pi/agent/settings.json', "
        + repr(mutator)
        + ")} && \"\n"
    )

# Post-patch fingerprint: the sed command we inject.
POST_PATCH_APT_MIRROR = "mirrors.tuna.tsinghua.edu.cn"


def _has_apt_mirror_patch(text: str) -> bool:
    """True iff registry.py has the apt-mirror rewrite (marker comment or
    the post-patch tuna URL)."""
    return MARKER_APT_MIRROR in text or POST_PATCH_APT_MIRROR in text


def _build_apt_mirror_block() -> str:
    """Return the sed-rewrite line prepended to the pi-acp install_cmd.

    The image ships two apt source files pointing at deb.debian.org:

    * ``/etc/apt/sources.list.d/debian.sources`` — the .sources-style file
      with two stanzas (debian + debian-security), both on deb.debian.org.
    * ``/etc/apt/sources.list`` — already rewritten to tuna by the image
      author, BUT with a *broken* ``trixie-security`` line that points at
      ``/debian/`` (tuna keeps ``trixie-security`` under
      ``/debian-security/``).

    A naive ``s|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g`` over BOTH
    files causes apt-get update to fail because (a) ``trixie`` /
    ``trixie-updates`` get duplicated across both files (warning only),
    and (b) the broken ``trixie-security`` line still tries
    ``trixie-security`` under ``/debian/`` which tuna does not serve.

    The fix below does three things:
      1. rewrite the .sources file to use tuna (debian + debian-security
         URLs both work on tuna).
      2. delete the broken ``trixie-security`` line from sources.list —
         debian.sources now provides it correctly.
      3. leave ``trixie`` / ``trixie-updates`` lines in sources.list alone
         (their duplicate-config warnings are harmless and don't fail
         apt-get update).
    """
    return (
        MARKER_APT_MIRROR
        + "\n"
        + "            f\"( sed -i.bak "
        + "'s|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g' "
        + "/etc/apt/sources.list.d/*.sources; "
        + "sed -i.bak '/trixie-security/d' /etc/apt/sources.list; "
        + "true ) && \"\n"
    )


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

    # Sub-patch: make Pi's provider request longer than the default 10 minutes
    if _has_pi_timeout_patch(text):
        msgs.append("[registry.py] Pi provider timeout/retry setting already applied, skip")
    else:
        if _PIACP_VERSIONED_LINE not in text:
            return (
                "[registry.py] pinned pi-acp install line not found — "
                "cannot place the Pi provider timeout setting"
            )
        text = text.replace(
            _PIACP_VERSIONED_LINE,
            _PIACP_VERSIONED_LINE + _build_pi_timeout_block(),
            1,
        )
        msgs.append(
            "[registry.py] extended Pi provider timeout to 3h and disabled "
            "SDK timeout retries"
        )

    # Sub-patch: prepend apt-mirror rewrite so apt-get update hits tuna
    if _has_apt_mirror_patch(text):
        msgs.append("[registry.py] apt-mirror rewrite already applied, skip")
    else:
        if _PIACP_FIRST_LINE not in text:
            return (
                "[registry.py] pi-acp first install_cmd line not found — "
                "upstream may have rewritten the install block; manual patch needed"
            )
        text = text.replace(
            _PIACP_FIRST_LINE,
            _build_apt_mirror_block() + _PIACP_FIRST_LINE,
            1,
        )
        msgs.append(
            "[registry.py] prepended apt-mirror rewrite to pi-acp install_cmd "
            "(deb.debian.org → mirrors.tuna.tsinghua.edu.cn)"
        )

    REGISTRY_PATH.write_text(text)
    return "\n".join(msgs)


def check_registry() -> bool:
    """Return True iff all registry patches are present."""
    if not REGISTRY_PATH.exists():
        return False
    text = REGISTRY_PATH.read_text()
    return (
        _has_autonomy_patch(text)
        and _has_piacp_pin(text)
        and _has_pi_timeout_patch(text)
        and _has_apt_mirror_patch(text)
    )


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
# Patch 6: litellm_config.py + litellm_runtime.py — disable LiteLLM proxy auth
# ---------------------------------------------------------------------------
#
# Symptom: ~30 min into a smoke run the agent's chat-completion requests
# start returning 500 Internal Server Error. LiteLLM stderr shows:
#
#     ModuleNotFoundError: No module named 'prisma'
#
# Root cause: the LiteLLM env used by benchflow's smoke runs does not ship
# the `prisma` Python wheel. When the per-request API-key auth cache
# expires (cache TTL = ~30 min in newer LiteLLM), `_user_api_key_auth`
# re-runs the auth check. master_key check fails (the agent sends the
# BENCHFLOW_PROVIDER_API_KEY the caller set, which is usually NOT the
# auto-generated `sk-benchflow-...` value), so auth falls through to the
# DB-lookup path, which calls `PrismaDBExceptionHandler.is_database_*`
# which does `import prisma` → ModuleNotFoundError → 500.
#
# Fix: opt-in via `BENCHFLOW_LITELLM_NO_AUTH=1` (set in Main.py's engine
# config or the shell). When set, benchflow passes master_key="" all the
# way down: (a) litellm_proxy_config writes an empty general_settings
# (no master_key at all → LiteLLM hits the `master_key is None` branch
# and returns UserAPIKeyAuth(INTERNAL_USER) immediately); (b) the
# `_host_litellm_executable` spawn step skips `LITELLM_MASTER_KEY` in the
# child env (so the env var can't re-enable auth).
#
# Idempotency: each anchor matches a single line in the source. The
# post-patch fingerprints `POST_PATCH_LITELLM_NO_AUTH_CONFIG` /
# `_RUNTIME` are unique substrings of the rewritten blocks. Re-running
# the script after a benchflow upgrade that rewrote the anchor lines
# will print "anchor not found" and exit with the failed step noted.

# Anchor 1 (litellm_config.py:465) — the line that writes master_key
# into general_settings unconditionally.
_LITELLM_CONFIG_OLD = (
    '        "general_settings": {"master_key": master_key},\n'
)
_LITELLM_CONFIG_NEW = (
    "        # Patched for KVBench smoke runs: empty master_key disables auth.\n"
    '        "general_settings": ({"master_key": master_key} if master_key else {}),\n'
)


def _has_litellm_no_auth_config_patch(text: str) -> bool:
    return (
        POST_PATCH_LITELLM_NO_AUTH_CONFIG in text
        or "Patched for KVBench smoke runs: empty master_key disables auth"
        in text
    )


def _has_litellm_no_auth_runtime_patch(text: str) -> bool:
    return (
        POST_PATCH_LITELLM_NO_AUTH_RUNTIME in text
        or "Patched for KVBench smoke runs: BENCHFLOW_LITELLM_NO_AUTH"
        in text
    )


# Anchor 2 (litellm_runtime.py:1645-1648) — the master_key generator.
# We rewrite the whole `master_key = (...)` expression so it short-circuits
# to "" when BENCHFLOW_LITELLM_NO_AUTH is set.
_LITELLM_RUNTIME_MASTER_KEY_OLD = (
    "    master_key = (\n"
    "        agent_env.get(LITELLM_MASTER_KEY_ENV)\n"
    '        or f"sk-benchflow-{secrets.token_urlsafe(24)}"\n'
    "    )\n"
)
_LITELLM_RUNTIME_MASTER_KEY_NEW = (
    "    # Patched for KVBench smoke runs: BENCHFLOW_LITELLM_NO_AUTH=1 forces\n"
    "    # master_key to '' so the proxy's auth bypass branch fires and\n"
    "    # never tries to import prisma (which is missing from this venv).\n"
    '    master_key = (\n'
    "        \"\"\n"
    '        if os.environ.get("BENCHFLOW_LITELLM_NO_AUTH")\n'
    "        else (\n"
    "            agent_env.get(LITELLM_MASTER_KEY_ENV)\n"
    '            or f"sk-benchflow-{secrets.token_urlsafe(24)}"\n'
    "        )\n"
    "    )\n"
)


# Anchor 3 (litellm_runtime.py — three sites): the proxy's `LITELLM_MASTER_KEY`
# env-var line. There are three call sites (host proxy, sandbox proxy, agent
# env wire-up); the line is identical in all three, so replace_all=True.
_LITELLM_RUNTIME_ENV_OLD = '            "LITELLM_MASTER_KEY": master_key,\n'
_LITELLM_RUNTIME_ENV_NEW = (
    "            # Patched for KVBench smoke runs: skip env var when no-auth.\n"
    '            **({"LITELLM_MASTER_KEY": master_key} if master_key else {}),\n'
)


def patch_litellm() -> str:
    """Apply patch 6 (LiteLLM no-auth) to litellm_config.py + litellm_runtime.py.
    Idempotent."""
    msgs: list[str] = []

    # --- litellm_config.py: rewrite the master_key writer ---
    if not LITELLM_CONFIG_PATH.exists():
        msgs.append(f"[litellm_config.py] not found at {LITELLM_CONFIG_PATH}")
    else:
        text = LITELLM_CONFIG_PATH.read_text()
        if _has_litellm_no_auth_config_patch(text):
            msgs.append(
                "[litellm_config.py] no-auth general_settings already applied, skip"
            )
        else:
            if _LITELLM_CONFIG_OLD not in text:
                msgs.append(
                    "[litellm_config.py] anchor not found — upstream may have "
                    "rewritten the general_settings line; manual patch needed"
                )
            else:
                text = text.replace(_LITELLM_CONFIG_OLD, _LITELLM_CONFIG_NEW, 1)
                LITELLM_CONFIG_PATH.write_text(text)
                msgs.append(
                    "[litellm_config.py] rewrote general_settings to skip master_key "
                    "when empty (no-auth mode)"
                )

    # --- litellm_runtime.py: master_key generator + env-var lines ---
    if not LITELLM_RUNTIME_PATH.exists():
        msgs.append(f"[litellm_runtime.py] not found at {LITELLM_RUNTIME_PATH}")
    else:
        text = LITELLM_RUNTIME_PATH.read_text()
        if _has_litellm_no_auth_runtime_patch(text):
            msgs.append(
                "[litellm_runtime.py] BENCHFLOW_LITELLM_NO_AUTH guard already applied, skip"
            )
        else:
            if _LITELLM_RUNTIME_MASTER_KEY_OLD not in text:
                msgs.append(
                    "[litellm_runtime.py] master_key generator anchor not found — "
                    "upstream may have rewritten the block; manual patch needed"
                )
            else:
                text = text.replace(
                    _LITELLM_RUNTIME_MASTER_KEY_OLD,
                    _LITELLM_RUNTIME_MASTER_KEY_NEW,
                    1,
                )
                LITELLM_RUNTIME_PATH.write_text(text)
                msgs.append(
                    "[litellm_runtime.py] rewrote master_key generator to honor "
                    "BENCHFLOW_LITELLM_NO_AUTH=1"
                )

        # Env-var line patch — replace_all (3 sites in same file).
        text = LITELLM_RUNTIME_PATH.read_text()
        if '"LITELLM_MASTER_KEY": master_key' in text:
            new_text = text.replace(_LITELLM_RUNTIME_ENV_OLD, _LITELLM_RUNTIME_ENV_NEW)
            if new_text != text:
                LITELLM_RUNTIME_PATH.write_text(new_text)
                msgs.append(
                    "[litellm_runtime.py] gated LITELLM_MASTER_KEY env-var assignment "
                    "behind master_key truthiness (3 sites)"
                )
        else:
            msgs.append(
                "[litellm_runtime.py] LITELLM_MASTER_KEY env-var line already gated, skip"
            )

    return "\n".join(msgs)


def check_litellm() -> bool:
    if not (LITELLM_CONFIG_PATH.exists() and LITELLM_RUNTIME_PATH.exists()):
        return False
    return (
        _has_litellm_no_auth_config_patch(LITELLM_CONFIG_PATH.read_text())
        and _has_litellm_no_auth_runtime_patch(LITELLM_RUNTIME_PATH.read_text())
    )


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
            missing.append("registry.py: pi-acp pin + autonomous directive + apt mirror")
        if not check_docker():
            missing.append("docker.py: --rmi all removal")
        if not check_litellm():
            missing.append(
                "litellm_config.py + litellm_runtime.py: BENCHFLOW_LITELLM_NO_AUTH gate"
            )
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
    print(patch_litellm())
    return 0


if __name__ == "__main__":
    sys.exit(main())
