"""Render SkillsBench skills into a system-prompt block.

SkillsBench tasks ship a curated set of skills under
``tasks/<id>/environment/skills/<name>/SKILL.md`` (plus ``scripts/``,
``assets/``, ``references/``). The official BenchFlow harness lets the agent
discover them at runtime; this adapter skips discovery entirely and surfaces
the SKILL.md bodies directly to the LLM so the first turn already knows the
task's reusable guidance.

Why a dedicated module
----------------------
The BenchflowHelper is already large; keeping the skill-loading logic in its
own file makes it independently testable and reusable if other task types
ever need the same injection (e.g. a hypothetical SkillsBench-shaped
non-agent task).

Frontmatter parsing
-------------------
SKILL.md frontmatter is intentionally **not** parsed with PyYAML. The
``citation-management`` SKILL.md contains markdown tables with colons and
nested ``metadata:`` blocks; feeding those through a real YAML parser can
either succeed with surprising structure or throw on the table syntax. The
frontmatter we care about is two scalar fields (``name``, ``description``),
so a hand-rolled splitter is both faster and more predictable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple


def ParseSkillFrontmatter(text: str) -> Tuple[Dict[str, str], str]:
    """Split a SKILL.md into ``(fields, body)``.

    Args:
        text: Raw SKILL.md contents.

    Returns:
        A pair ``(fields, body)`` where ``fields`` is the parsed frontmatter
        mapping (only top-level scalar keys are kept; nested blocks like
        ``metadata:`` are skipped). ``body`` is everything from the first
        non-frontmatter line onward, with leading ``\\n`` stripped. If the
        file has no frontmatter, ``fields`` is empty and ``body`` is the
        whole input.
    """
    # Frontmatter must start on the very first line with a ``---`` fence.
    if not text.startswith("---"):
        return {}, text.lstrip("\n")

    lines = text.splitlines()
    # The opening ``---`` is line 0; the closing fence is the next ``---``
    # on its own line. Anything before the closing fence is frontmatter
    # (lines[1:close_idx]), anything after is body (lines[close_idx+1:]).
    closeIdx = -1
    for idx in range(1, len(lines)):
        if lines[idx].rstrip() == "---":
            closeIdx = idx
            break
    if closeIdx == -1:
        # Unterminated frontmatter — treat the whole file as body.
        return {}, text.lstrip("\n")

    fields: Dict[str, str] = {}
    for raw in lines[1:closeIdx]:
        # Only accept unindented ``key: value`` lines. Nested entries
        # (``    skill-author: ...``) are skipped so the top-level mapping
        # stays flat.
        if not raw or raw[:1] in (" ", "\t"):
            continue
        colon = raw.find(":")
        if colon <= 0:
            # ``---`` separators inside the body would already have been
            # consumed; anything here without ``:`` ends the frontmatter
            # implicitly.
            continue
        key = raw[:colon].strip()
        if not key or any(ch.isspace() for ch in key):
            continue
        value = raw[colon + 1:].strip()
        # ``key:`` with no value marks the start of a nested block (e.g.
        # ``metadata:`` followed by indented children). Stop accumulating
        # top-level scalars and ignore the empty-value key entirely.
        if not value:
            break
        # Strip a single layer of surrounding matching quotes — SkillsBench
        # wraps multi-word descriptions in double quotes when they would
        # otherwise be ambiguous to a downstream YAML reader.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        # Only keep the first occurrence so duplicate keys (rare but seen
        # in nested-metadata snippets) don't shadow the top-level value.
        fields.setdefault(key, value)

    body = "\n".join(lines[closeIdx + 1:]).lstrip("\n")
    return fields, body


def _DiscoverSkills(skillsRoot: Path) -> List[Tuple[str, Path]]:
    """Return ``[(skill_name, skill_md_path), ...]`` sorted by name.

    A skill folder is anything containing a ``SKILL.md`` directly under it.
    Folders without ``SKILL.md`` are ignored — they are usually auxiliary
    reference data or partial downloads.
    """
    if not skillsRoot.is_dir():
        return []
    found: List[Tuple[str, Path]] = []
    for entry in sorted(skillsRoot.iterdir(), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        skillMd = entry / "SKILL.md"
        if skillMd.is_file():
            found.append((entry.name, skillMd))
    return found


def _SkillBody(name: str, frontmatter: Dict[str, str], body: str) -> str:
    """Render a single skill section.

    The section starts with ``## <name>`` then a one-line ``description``
    (the trigger statement), then the body. Description is omitted when
    the frontmatter has none.
    """
    description = frontmatter.get("description", "").strip()
    parts: List[str] = [f"## {name}"]
    if description:
        parts.append("")
        parts.append(description)
    if body:
        parts.append("")
        parts.append(body.rstrip())
    return "\n".join(parts)


def _UsageBlock() -> str:
    return (
        "## How to use these skills\n"
        "\n"
        "- The full skill file system is mounted at "
        "`/root/.agents/skills/<name>/` (and `/root/.claude/skills/<name>/`) "
        "inside the sandbox.\n"
        "- Helper scripts are at "
        "`/root/.agents/skills/<name>/scripts/<script>.py`; invoke them "
        "with `python /root/.agents/skills/<name>/scripts/<script>.py`.\n"
        "- Reference docs are at "
        "`/root/.agents/skills/<name>/references/<file>.md`.\n"
        "- Assets (templates, fixtures, lookup tables) are at "
        "`/root/.agents/skills/<name>/assets/`.\n"
    )


def BuildSkillsBlock(
    skillsbench_dir: Path,
    task_id: str,
    *,
    skills_root: Optional[Path] = None,
) -> str:
    """Render every skill under the task's ``environment/skills/`` into one block.

    Args:
        skillsbench_dir: Root of the cloned SkillsBench repo.
        task_id: The SkillsBench task id (folder under ``tasks/``).
        skills_root: Override for the skills directory; defaults to
            ``<skillsbench_dir>/tasks/<task_id>/environment/skills``.

    Returns:
        ``""`` if the task has no skills directory. Otherwise a string
        intended to be prepended to the first ``system`` message in the
        Qwen3 chat template (it does NOT include the ``<|im_start|>system``
        / ``<|im_end|>`` markers themselves — the existing renderer adds
        those).
    """
    root = Path(skills_root) if skills_root is not None else (
        Path(skillsbench_dir) / "tasks" / task_id / "environment" / "skills"
    )
    skills = _DiscoverSkills(root)
    if not skills:
        return ""

    sections: List[str] = []
    for name, skillMd in skills:
        try:
            text = skillMd.read_text(encoding="utf-8")
        except OSError:
            # Skip unreadable skills rather than failing the whole rollout —
            # the agent still has the others.
            continue
        frontmatter, body = ParseSkillFrontmatter(text)
        # Trust the folder name over the frontmatter ``name`` field — the
        # two always match in practice, but the folder is what's actually
        # mounted at ``/root/.agents/skills/<folder>/``.
        sections.append(_SkillBody(name, frontmatter, body))

    header = (
        "# Skills Available for This Task\n"
        "\n"
        "The following skills have been curated for this task. Use them "
        "directly — do NOT search for, evaluate, or download additional "
        "skills. The skill filesystem is mounted at "
        "`/root/.agents/skills/<name>/` (see the \"How to use these "
        "skills\" section at the end for details)."
    )
    parts: List[str] = [header, ""]
    parts.extend(sections)
    parts.append("")
    parts.append(_UsageBlock())
    return "\n".join(parts).rstrip() + "\n"


__all__ = ["BuildSkillsBlock", "ParseSkillFrontmatter"]