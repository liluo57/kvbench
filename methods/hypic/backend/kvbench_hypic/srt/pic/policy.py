"""PICPolicy: attribute-driven description of a PIC method.

Replaces scattered ``pic_mode`` string comparisons with orthogonal
attributes. The public method names remain as aliases (see
:data:`POLICIES`); code branches on attributes, not names, so adding a method
means adding one table row rather than editing every conditional.

Note: the *seam width* is NOT a policy attribute — it is an independent
runtime knob (``PIC_SEAM_SINK`` env / ``resolve_seam_sink_tokens``). The
policy only records *whether* a mode recomputes the seam.
"""
from __future__ import annotations

from enum import Enum, auto

import msgspec


class PICCompose(Enum):
    ADDITION = auto()    # cache and add S_{C|0} only, no transition operator
    TRANSITION = auto()  # transition-operator compose (T_C, S_{C|0})


class PICStateInit(Enum):
    """How a hybrid recurrent layer is initialized for a PIC request."""

    ADDITION = auto()
    EXACT_TRANSITION = auto()
    LAST_BLOCK = auto()


class PICPolicy(msgspec.Struct, frozen=True):
    compose: PICCompose  # compose operator
    rope: bool           # rope re-rotation correction (also selects cache schema / mamba_idx)
    recompute: bool      # recompute the hit-segment seam window
    # None preserves the three-argument construction used by existing HYPIC
    # tests/config helpers; legacy modes derive this strategy from `compose`.
    state_init: PICStateInit | None = None
    selector: str = "none"

    @property
    def uses_transition(self) -> bool:
        return self.state_init is PICStateInit.EXACT_TRANSITION or (
            self.state_init is None and self.compose is PICCompose.TRANSITION
        )

    @property
    def is_linearkv(self) -> bool:
        return self.state_init is PICStateInit.LAST_BLOCK


# Names stay as aliases (config / experiments / logs are unchanged); dispatch
# reads attributes, not names.
POLICIES: dict[str, PICPolicy] = {
    "addition": PICPolicy(
        PICCompose.ADDITION,
        rope=False,
        recompute=False,
    ),
    "transition": PICPolicy(
        PICCompose.TRANSITION,
        rope=False,
        recompute=False,
    ),
    "transition_rope": PICPolicy(
        PICCompose.TRANSITION,
        rope=True,
        recompute=False,
    ),
    "transition_rope_recompute": PICPolicy(
        PICCompose.TRANSITION,
        rope=True,
        recompute=True,
    ),
    # LinearKV deliberately has no transition composition.  It uses the
    # position-independent FA path plus a single cached recurrent state.
    "linearkv": PICPolicy(
        PICCompose.ADDITION,
        rope=True,
        recompute=True,
        state_init=PICStateInit.LAST_BLOCK,
        selector="epic",
    ),
}


def resolve_policy(pic_mode: str) -> PICPolicy:
    """Look up the :class:`PICPolicy` for a mode name. Raises KeyError on unknown."""
    return POLICIES[pic_mode]
