"""CPU-only equivalence checks for the proposed direct KV capture packing.

This deliberately imports neither vLLM nor the training worker. It compares the
existing snapshot -> per-request layer stack -> packed cat path with writes into
the final owned BF16 buffer using a non-contiguous layer view.
"""
from collections import defaultdict, deque
import unittest

import torch


def _match_prompt_occurrences(expected_prompts, scheduled_requests):
    """Reproduce the candidate's token-tuple queue mapping, including duplicates."""
    prompt_indices = defaultdict(deque)
    for index, prompt in enumerate(expected_prompts):
        prompt_indices[tuple(prompt)].append(index)

    result = {}
    for request_id, prompt in scheduled_requests:
        indices = prompt_indices.get(tuple(prompt))
        if not indices:
            raise ValueError("unexpected or repeated prompt")
        result[request_id] = indices.popleft()
    return result


def _partition_offsets(total_tokens, lengths):
    if sum(lengths) != total_tokens or any(length < 1 for length in lengths):
        raise ValueError("invalid partition lengths")
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    return offsets


def _old_stack_and_cat(expected_prompts, request_to_index, kv_by_prompt, layer_ids):
    """Current implementation's materialized per-layer snapshots and export packing."""
    captured = {}
    # Deliberately invoke hooks in the supplied order; output order is sorted IDs,
    # matching the current export's `layers = sorted(...)` contract.
    for request_id, prompt_index in request_to_index.items():
        prompt = tuple(expected_prompts[prompt_index])
        captured[request_id] = {}
        for layer_id in layer_ids:
            key, value = kv_by_prompt[prompt][layer_id]
            captured[request_id][layer_id] = torch.stack((key, value), dim=-2).detach()

    sorted_layers = sorted(layer_ids)
    pieces = []
    for prompt_index in range(len(expected_prompts)):
        request_id = next(req for req, idx in request_to_index.items() if idx == prompt_index)
        pieces.append(torch.stack(
            [captured[request_id][layer] for layer in sorted_layers], dim=1))
    return torch.cat(pieces, dim=0).contiguous(), sorted_layers


def _direct_stack_out(expected_prompts, request_to_index, kv_by_prompt,
                      layer_slots, hook_order):
    """Candidate's one final-layout allocation, filled in arbitrary hook order."""
    offsets = [0]
    for prompt in expected_prompts:
        offsets.append(offsets[-1] + len(prompt))

    first = next(iter(next(iter(kv_by_prompt.values())).values()))[0]
    heads, dim = first.shape[1:]
    packed = torch.empty((offsets[-1], len(layer_slots), heads, 2, dim),
                         dtype=torch.bfloat16, device="cpu")

    # Scheduler order differs from expected order; token matching already maps
    # request IDs to expected offsets. Hook layer order is independent as well.
    for layer in hook_order:
        slot = layer_slots[layer]
        for request_id, prompt_index in request_to_index.items():
            prompt = tuple(expected_prompts[prompt_index])
            start, end = offsets[prompt_index:prompt_index + 2]
            key, value = kv_by_prompt[prompt][layer]
            out_view = packed[start:end, slot]
            out_storage_ptr = out_view.untyped_storage().data_ptr()
            torch.stack((key, value), dim=-2, out=out_view)
            if out_view.untyped_storage().data_ptr() != packed.untyped_storage().data_ptr():
                raise AssertionError("stack(out=...) replaced the destination storage")
            if out_view.untyped_storage().data_ptr() != out_storage_ptr:
                raise AssertionError("stack(out=...) resized/replaced its strided view")

    # The packed layer axis is in ascending canonical layer-ID order, even if
    # hook invocation or module registration order differs.
    sorted_layers = sorted(layer_slots)
    packed = packed[:, [layer_slots[layer] for layer in sorted_layers]]
    return packed.contiguous(), sorted_layers, offsets


class CapturePackingTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(271828)
        self.expected_prompts = [
            [11, 12, 13, 14, 15],
            [31, 32, 33],
            [11, 12, 13, 14, 15],  # duplicate occurrence exercises the queue
        ]
        self.scheduled = [
            ("request-c", self.expected_prompts[2]),
            ("request-b", self.expected_prompts[1]),
            ("request-a", self.expected_prompts[0]),
        ]
        self.request_to_index = _match_prompt_occurrences(
            self.expected_prompts, self.scheduled)
        self.layer_ids = list(range(16))
        # Arbitrary module registration/slot order. Candidate slot assignment
        # must be based on sorted layer IDs, not this order or hook order.
        self.registration_order = [7, 2, 15, 0, 11, 4, 13, 1,
                                   9, 5, 14, 3, 12, 6, 10, 8]
        self.layer_slots = {layer: slot for slot, layer in enumerate(
            sorted(self.layer_ids))}
        self.hook_order = list(reversed(self.registration_order))

        # Use prompt-content keyed values, so duplicate token prompts correctly
        # receive identical deterministic K/V, independent of request ID.
        self.kv_by_prompt = {}
        for prompt in {tuple(p) for p in self.expected_prompts}:
            length = len(prompt)
            self.kv_by_prompt[prompt] = {}
            for layer in self.layer_ids:
                key = torch.randn(length, 2, 8, dtype=torch.float32).to(torch.bfloat16)
                value = torch.randn(length, 2, 8, dtype=torch.float32).to(torch.bfloat16)
                self.kv_by_prompt[prompt][layer] = (key, value)

    def test_direct_strided_stack_matches_old_bytes_and_partitions(self):
        old_packed, old_layers = _old_stack_and_cat(
            self.expected_prompts, self.request_to_index,
            self.kv_by_prompt, self.hook_order)
        direct_packed, direct_layers, offsets = _direct_stack_out(
            self.expected_prompts, self.request_to_index,
            self.kv_by_prompt, self.layer_slots, self.hook_order)

        self.assertEqual(old_layers, list(range(16)))
        self.assertEqual(direct_layers, old_layers)
        self.assertTrue(torch.equal(direct_packed, old_packed))
        self.assertTrue(torch.equal(
            direct_packed.view(torch.uint8), old_packed.view(torch.uint8)))

        self.assertEqual(offsets, [0, 5, 8, 13])
        for start, end in zip(offsets, offsets[1:]):
            old_view = old_packed[start:end]
            new_view = direct_packed[start:end]
            self.assertTrue(torch.equal(new_view, old_view))
            self.assertEqual(new_view.untyped_storage().data_ptr(),
                             direct_packed.untyped_storage().data_ptr())
        # Duplicate occurrences map to distinct expected offsets while holding
        # identical prompt-derived KV bytes.
        # Equal-token duplicates are indistinguishable to token matching; the
        # candidate consumes their expected indices in first-seen order.
        self.assertEqual(self.request_to_index["request-c"], 0)
        self.assertEqual(self.request_to_index["request-a"], 2)
        self.assertTrue(torch.equal(direct_packed[:5], direct_packed[8:13]))

    def test_duplicate_prompts_are_consumed_once_each(self):
        mapping = _match_prompt_occurrences(
            self.expected_prompts,
            [("later", self.expected_prompts[2]),
             ("middle", self.expected_prompts[1]),
             ("earlier", self.expected_prompts[0])])
        self.assertEqual(mapping, {"later": 0, "middle": 1, "earlier": 2})
        with self.assertRaises(ValueError):
            _match_prompt_occurrences(
                self.expected_prompts,
                [("a", self.expected_prompts[0]),
                 ("b", self.expected_prompts[0]),
                 ("c", self.expected_prompts[0]),
                 ("extra", [999])])

    def test_partition_boundaries_reject_invalid_lengths(self):
        packed = torch.empty((13, 16, 2, 2, 8), dtype=torch.bfloat16)
        offsets = _partition_offsets(packed.shape[0], [5, 3, 5])
        self.assertEqual(offsets, [0, 5, 8, 13])
        self.assertEqual([offsets[i+1] - offsets[i] for i in range(3)], [5, 3, 5])
        for invalid in ([5, 3, 4], [5, 3, 5, 1], [5, 0, 8]):
            with self.assertRaises(ValueError):
                _partition_offsets(packed.shape[0], invalid)


if __name__ == "__main__":
    unittest.main()
