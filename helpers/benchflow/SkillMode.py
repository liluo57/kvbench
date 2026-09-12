"""Shared SkillsBench skill-mode normalization.

BenchFlow calls the baseline condition ``no-skill``.  KVBench also accepts
``WithoutSkill`` (and its kebab/snake-case spellings) as a readable config
alias, but always passes the canonical value to BenchFlow.
"""

from __future__ import annotations


SKILL_MODE_WITH_SKILL = "with-skill"
SKILL_MODE_NO_SKILL = "no-skill"
SKILL_MODES = frozenset({SKILL_MODE_WITH_SKILL, SKILL_MODE_NO_SKILL})

_SKILL_MODE_ALIASES = {
    "with-skill": SKILL_MODE_WITH_SKILL,
    "with_skill": SKILL_MODE_WITH_SKILL,
    "withskill": SKILL_MODE_WITH_SKILL,
    "no-skill": SKILL_MODE_NO_SKILL,
    "no_skill": SKILL_MODE_NO_SKILL,
    "noskill": SKILL_MODE_NO_SKILL,
    "without-skill": SKILL_MODE_NO_SKILL,
    "without_skill": SKILL_MODE_NO_SKILL,
    "withoutskill": SKILL_MODE_NO_SKILL,
}


def NormalizeSkillMode(value: str, *, field: str = "skill_mode") -> str:
    """Return the BenchFlow spelling for a configured skill mode."""

    if not isinstance(value, str):
        raise ValueError(
            f"{field} must be 'with-skill' or 'no-skill' "
            "(WithoutSkill is also accepted)"
        )
    normalized = _SKILL_MODE_ALIASES.get(value.strip().casefold())
    if normalized is None:
        raise ValueError(
            f"{field} must be 'with-skill' or 'no-skill' "
            "(WithoutSkill is also accepted)"
        )
    return normalized


__all__ = [
    "NormalizeSkillMode",
    "SKILL_MODE_NO_SKILL",
    "SKILL_MODE_WITH_SKILL",
    "SKILL_MODES",
]
