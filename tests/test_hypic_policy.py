# Upstream PIC/LinearKV policy tests; see methods/hypic/UPSTREAM.
from methods.hypic.backend.kvbench_hypic.srt.pic.policy import PICCompose, PICPolicy, POLICIES, resolve_policy


def test_four_aliases_map_to_expected_attrs():
    assert POLICIES["addition"] == PICPolicy(PICCompose.ADDITION, rope=False, recompute=False)
    assert POLICIES["transition"] == PICPolicy(PICCompose.TRANSITION, rope=False, recompute=False)
    assert POLICIES["transition_rope"] == PICPolicy(PICCompose.TRANSITION, rope=True, recompute=False)
    assert POLICIES["transition_rope_recompute"] == PICPolicy(
        PICCompose.TRANSITION, rope=True, recompute=True
    )


def test_resolve_policy_known_and_unknown():
    assert resolve_policy("transition_rope") is POLICIES["transition_rope"]
    try:
        resolve_policy("bogus")
    except KeyError:
        pass
    else:
        raise AssertionError("unknown mode should raise KeyError")


def test_derived_predicates_replace_string_checks():
    # addition check
    assert (POLICIES["addition"].compose is PICCompose.ADDITION)
    assert not (POLICIES["transition"].compose is PICCompose.ADDITION)
    # transition (non-addition) check
    for m in ("transition", "transition_rope", "transition_rope_recompute"):
        assert POLICIES[m].compose is PICCompose.TRANSITION
    # rope check (== old _ROPE_PIC_MODES membership)
    assert POLICIES["transition_rope"].rope and POLICIES["transition_rope_recompute"].rope
    assert not POLICIES["addition"].rope and not POLICIES["transition"].rope
    # mamba_idx = 2 if rope else 1
    assert (2 if POLICIES["transition_rope"].rope else 1) == 2
    assert (2 if POLICIES["transition"].rope else 1) == 1
    # is_recompute
    assert POLICIES["transition_rope_recompute"].recompute
    assert not POLICIES["transition_rope"].recompute


def test_policy_is_frozen_and_hashable():
    p = POLICIES["transition"]
    assert hash(p) == hash(PICPolicy(PICCompose.TRANSITION, rope=False, recompute=False))
    try:
        p.rope = True
    except (AttributeError, TypeError):
        pass
    else:
        raise AssertionError("PICPolicy should be frozen")


from methods.hypic.backend.kvbench_hypic.srt.pic.policy import PICCompose, PICStateInit, POLICIES
from methods.hypic.backend.kvbench_hypic.srt.pic.selector import merge_online_positions, select_epic_positions


def test_linearkv_policy_does_not_use_transition():
    policy = POLICIES["linearkv"]
    assert policy.state_init is PICStateInit.LAST_BLOCK
    assert policy.compose is PICCompose.ADDITION  # legacy field only
    assert policy.is_linearkv
    assert not policy.uses_transition
    assert policy.rope
    assert policy.selector == "epic"


def test_epic_positions_are_chunk_local_and_prompt_ordered():
    hits = [
        (20, 27, b"c2"),
        (0, 8, b"c1"),
        (40, 41, b"c3"),
    ]
    selected = select_epic_positions(hits, tokens_per_chunk=2)
    assert selected[(0, 8)] == [0, 1]
    assert selected[(20, 27)] == [20, 21]
    assert selected[(40, 41)] == [40]
    assert merge_online_positions([(8, 20), (27, 40)], selected) == [
        0,
        1,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
    ]


def test_epic_ratio_selects_chunk_prefix_and_keeps_short_chunks_nonempty():
    hits = [(0, 10, b"c1"), (20, 24, b"c2"), (40, 41, b"c3")]
    selected = select_epic_positions(hits, ratio=0.2)
    assert selected[(0, 10)] == [0, 1]
    assert selected[(20, 24)] == [20]
    assert selected[(40, 41)] == [40]


def test_epic_selector_rejects_both_budget_forms():
    try:
        select_epic_positions([(0, 8, b"c1")], tokens_per_chunk=2, ratio=0.2)
    except ValueError as exc:
        assert "mutually exclusive" in str(exc)
    else:
        raise AssertionError("EPIC selector accepted both budget forms")
