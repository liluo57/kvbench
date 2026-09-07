from types import SimpleNamespace

import pytest

from methods.Hypic import HypicMethod, _SegmentedPrompt, _WARMUP_TAIL
from tasks.FreshGap import FreshGapTask


class _FakeEngine:
    def __init__(self):
        self.calls = []
        self.flushes = 0
        self.closed = False

    def generate(self, prompt, sampling_params, stream):
        self.calls.append((prompt, sampling_params, stream))
        is_run = sampling_params["max_new_tokens"] > 1
        cached = 7 if is_run else 0
        prompt_tokens = 10 if is_run else 4
        yield {
            "text": None,
            "output_ids": [31],
            "meta_info": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": 1,
                "cached_tokens": cached,
            },
        }
        yield {
            "text": "answer",
            "output_ids": [31, 32],
            "meta_info": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": 2,
                "cached_tokens": cached,
            },
        }

    def flush_cache(self):
        self.flushes += 1
        return SimpleNamespace(success=True, message="ok")

    def shutdown(self):
        self.closed = True


def _method():
    method = HypicMethod(maxNewTokens=8)
    method.engine = _FakeEngine()
    return method


def test_segmented_prompt_drops_empty_spans_and_rejects_separator_collision():
    assert _SegmentedPrompt(["a", "", "b"], "<sep>") == "a<sep>b"
    with pytest.raises(ValueError, match="contains"):
        _SegmentedPrompt(["a<sep>b"], "<sep>")


def test_prepare_makes_every_user_chunk_non_final_and_run_marks_reordered_hits():
    method = _method()
    method.Prepare([["A", "B"]])

    warmPrompt, warmParams, warmStream = method.engine.calls[0]
    assert warmPrompt == method.separator.join(["A", "B", _WARMUP_TAIL])
    assert warmParams == {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "max_new_tokens": 1,
    }
    assert warmStream is True

    result = method.Run(["head-B-middle-A-tail"])[0]
    runPrompt = method.engine.calls[1][0]
    assert runPrompt == method.separator.join(
        ["head-", "B", "-middle-", "A", "-tail"]
    )
    assert result.output == "answer"
    assert result.metadata == {
        "backend": "hypic-sglang",
        "pic_mode": "addition",
        "n_input": 10,
        "num_cached_tokens": 7,
        "reuse_ratio": 0.7,
        "n_pic_segments": 5,
        "matched_prepared_segments": 2,
    }
    assert result.performance["num_output_tokens"] == 2
    assert result.performance["ttft"] >= 0
    assert result.performance["total_time"] >= result.performance["ttft"]


def test_fresh_gap_keeps_a_fresh_query_tail_after_final_prepared_chunk():
    task = FreshGapTask(nCases=1, linesPerChunk=1)
    case = next(task.Cases())

    assert case.input.prepare_input == [task._a, task._c]
    assert case.input.run_input == (
        task._a + case.metadata["fresh"] + task._c + task._query_tail
    )
    assert case.input.run_input.endswith(task._c + task._query_tail)


def test_run_without_a_match_preserves_the_original_prompt():
    method = _method()
    method.Prepare([["cached"]])
    result = method.Run(["plain prompt"])[0]

    assert method.engine.calls[-1][0] == "plain prompt"
    assert result.metadata["n_pic_segments"] == 1
    assert result.metadata["matched_prepared_segments"] == 0


def test_run_retains_output_as_a_segment_for_a_later_run():
    method = _method()
    method.Prepare([["cached"]])

    method.Run(["cached prompt"], retainOutput=[True])

    retentionPrompt, retentionParams, retentionStream = method.engine.calls[2]
    assert retentionPrompt == method.separator.join(["answer", _WARMUP_TAIL])
    assert retentionParams["max_new_tokens"] == 1
    assert retentionStream is True
    assert method._states[0]["prepare"] == ["cached", "answer"]

    method.Run(["prefix answer suffix"])
    assert method.engine.calls[3][0] == method.separator.join(
        ["prefix ", "answer", " suffix"]
    )


def test_run_retains_output_even_without_a_prepare_action():
    method = _method()

    method.Run(["prompt"], retainOutput=[True])

    assert method._states[0]["prepare"] == ["answer"]


def test_run_does_not_retain_output_containing_the_pic_separator():
    method = _method()
    method.Prepare([["cached"]])

    def generate(prompt, sampling_params, stream):
        method.engine.calls.append((prompt, sampling_params, stream))
        yield {
            "text": "answer" + method.separator,
            "output_ids": [31],
            "meta_info": {
                "prompt_tokens": 10,
                "completion_tokens": 1,
                "cached_tokens": 7,
            },
        }

    method.engine.generate = generate
    method.Run(["cached prompt"], retainOutput=[True])

    assert method._states[0]["prepare"] == ["cached"]
    assert len(method.engine.calls) == 2


def test_full_prefill_baseline_skips_prepare_and_disables_reuse_metadata():
    method = HypicMethod(maxNewTokens=8, fullPrefill=True, tag="full_prefill")
    method.engine = _FakeEngine()

    method.Prepare([["cached"]])
    assert method.engine.calls == []

    result = method.Run(["plain prompt"])[0]
    assert method.engine.calls[0][0] == "plain prompt"
    assert result.metadata == {
        "backend": "hypic-sglang",
        "n_input": 10,
        "num_cached_tokens": 0,
        "reuse_ratio": 0.0,
        "full_prefill": True,
    }
    assert method.Label == "hypic(full_prefill)"


def test_reset_flushes_picache_and_close_shuts_down_engine():
    method = _method()
    engine = method.engine
    method._states = [{"prepare": ["A"]}]

    method.Reset()
    assert method._states == []
    assert engine.flushes == 1

    method.Close()
    assert method.engine is None
    assert engine.closed is True


@pytest.mark.parametrize("picMode", ["bogus", "transition-rope"])
def test_constructor_rejects_unknown_pic_mode(picMode):
    with pytest.raises(ValueError, match="unknown picMode"):
        HypicMethod(picMode=picMode)


def test_constructor_rejects_non_boolean_full_prefill():
    with pytest.raises(TypeError, match="fullPrefill"):
        HypicMethod(fullPrefill=1)
