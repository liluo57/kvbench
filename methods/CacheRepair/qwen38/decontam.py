"""Exact text, verified 50-Qwen-token spans, sentence/provenance, and MinHash LSH.

Index the bounded candidate corpus, then stream evaluation content once. This
avoids storing every evaluation n-gram and avoids pairwise corpus comparisons.
"""

from __future__ import annotations

from collections import defaultdict
import functools
import hashlib
import re

import numpy as np

from .eval_inventory import inventory_contents, normalize, text_hash


def window_hashes(ids, width=50):
    """Vectorized rolling uint64 hashes; candidate matches are verified exactly."""
    if len(ids) < width:
        return np.empty(0, dtype=np.uint64)
    values = np.asarray(ids, dtype=np.uint64) + np.uint64(1)
    base = 1_000_003
    inverse = pow(base, -1, 1 << 64)
    powers = np.empty(len(values), dtype=np.uint64)
    powers[0] = 1
    powers[1:] = np.uint64(base)
    np.multiply.accumulate(powers, out=powers)
    prefix = np.concatenate((np.zeros(1, dtype=np.uint64), np.cumsum(values * powers, dtype=np.uint64)))
    inverse_powers = np.empty(len(values) - width + 1, dtype=np.uint64)
    inverse_powers[0] = 1
    inverse_powers[1:] = np.uint64(inverse)
    np.multiply.accumulate(inverse_powers, out=inverse_powers)
    return (prefix[width:] - prefix[:-width]) * inverse_powers


@functools.lru_cache(maxsize=200_000)
def _word_hash(word):
    return int.from_bytes(hashlib.blake2b(word.encode(), digest_size=8).digest(), "little")


def shingles(text):
    words = re.findall(r"\w+", normalize(text))
    return set(map(int, window_hashes([_word_hash(word) for word in words], width=5)))


class MinHashLSH:
    """64 deterministic permutations, 16 bands of four, exact Jaccard scores."""

    def __init__(self):
        rng = np.random.default_rng(7)
        self.a = rng.integers(1, 1 << 63, 64, dtype=np.uint64) | np.uint64(1)
        self.b = rng.integers(0, 1 << 63, 64, dtype=np.uint64)
        self.bands, self.sets = defaultdict(set), {}

    def signature(self, values):
        if not values:
            return None
        # Bound temporary memory for long evaluation documents.
        result = np.full(64, np.iinfo(np.uint64).max, dtype=np.uint64)
        values = np.fromiter(values, dtype=np.uint64)
        for start in range(0, len(values), 4096):
            block = values[start:start + 4096, None]
            result = np.minimum(result, (block * self.a + self.b).min(axis=0))
        return result

    def keys(self, signature):
        if signature is not None:
            for index in range(16):
                yield index, signature[index * 4:(index + 1) * 4].tobytes()

    def add(self, key, text):
        values = shingles(text)
        self.sets[key] = values
        for band in self.keys(self.signature(values)):
            self.bands[band].add(key)

    def query(self, text, threshold=0.65):
        values = shingles(text)
        candidates = set()
        for band in self.keys(self.signature(values)):
            candidates.update(self.bands.get(band, ()))
        for key in sorted(candidates):
            other = self.sets[key]
            score = len(values & other) / max(1, len(values | other))
            if score >= threshold:
                yield key, score


def sentences(text):
    return {normalize(part) for part in re.split(r"(?<=[.!?])\s+", text)
            if len(re.findall(r"\w+", part)) >= 12}


def screen(acquired, evaluation_inventory, tokenizer):
    passages, records = acquired["passages"], acquired["records"]
    exact, evidence, questions = (defaultdict(set) for _ in range(3))
    # Packed numeric postings keep full-corpus windows out of Python sets.
    window_parts, owner_parts, offset_parts = [], [], []
    identifiers = list(passages)
    tokens, near = {}, MinHashLSH()
    for owner, (identifier, passage) in enumerate(passages.items()):
        text = passage["contents"]
        exact[text_hash(text)].add(identifier)
        ids = tokenizer.encode(text, add_special_tokens=False)
        tokens[identifier] = ids
        hashes = window_hashes(ids)
        window_parts.append(hashes)
        owner_parts.append(np.full(len(hashes), owner, dtype=np.uint32))
        offset_parts.append(np.arange(len(hashes), dtype=np.uint32))
        for sentence in sentences(text):
            evidence[text_hash(sentence)].add(identifier)
        near.add(identifier, text)
        if (owner + 1) % 10000 == 0:
            print({"indexed_training_passages": owner + 1, "total": len(passages)}, flush=True)
    for row in records:
        questions[text_hash(row["query"])].add((row["config"], row["query_id"]))
    # Search sorted packed postings; verify the actual tokens for every hit.
    raw_keys = np.concatenate(window_parts) if window_parts else np.empty(0, dtype=np.uint64)
    ordering = np.argsort(raw_keys, kind="stable")
    window_keys = raw_keys[ordering]
    window_owners = np.concatenate(owner_parts)[ordering] if owner_parts else np.empty(0, dtype=np.uint32)
    window_offsets = np.concatenate(offset_parts)[ordering] if offset_parts else np.empty(0, dtype=np.uint32)
    del window_parts, owner_parts, offset_parts, raw_keys, ordering
    removals, question_matches, near_candidates = [], [], []
    units, token_count = 0, 0
    for digest, text, refs in inventory_contents(evaluation_inventory):
        units += 1
        reference = [{"workload": w, "record_id": r, "kind": k} for w, r, k in refs]
        for identifier in sorted(exact.get(digest, ())):
            removals.append({"passage_id": identifier, "reason": "exact_text", "evaluation": reference})
        for config, query_id in sorted(questions.get(digest, ())):
            question_matches.append({"config": config, "query_id": query_id,
                                     "reason": "question_provenance", "evaluation": reference})
        ids = tokenizer.encode(text, add_special_tokens=False)
        token_count += len(ids)
        hashes = window_hashes(ids)
        if len(hashes) and len(window_keys):
            locations = np.searchsorted(window_keys, hashes)
            safe = np.minimum(locations, len(window_keys) - 1)
            hits = np.flatnonzero((locations < len(window_keys)) & (window_keys[safe] == hashes))
            matched = set()
            for offset in hits:
                left = locations[offset]
                right = np.searchsorted(window_keys, hashes[offset], side="right")
                for posting in range(left, right):
                    identifier, start = identifiers[window_owners[posting]], int(window_offsets[posting])
                    if identifier not in matched and tokens[identifier][start:start + 50] == ids[offset:offset + 50]:
                        matched.add(identifier)
                        removals.append({"passage_id": identifier, "reason": "verified_50_token_span",
                                         "source_token_offset": start, "evaluation_token_offset": int(offset),
                                         "evaluation": reference})
        for sentence in sentences(text):
            for identifier in sorted(evidence.get(text_hash(sentence), ())):
                removals.append({"passage_id": identifier, "reason": "exact_evidence_sentence",
                                 "sentence_sha256": text_hash(sentence), "evaluation": reference})
        for identifier, score in near.query(text):
            near_candidates.append({"passage_id": identifier, "jaccard": score, "status": "needs_review",
                                    "evaluation": reference})
        if units % 5000 == 0:
            print({"screened_units": units, "evaluation_tokens": token_count,
                   "contaminated_passages": len({item['passage_id'] for item in removals})}, flush=True)
    removed = {item["passage_id"] for item in removals}
    pending = {item["passage_id"] for item in near_candidates} - removed
    return {"initial_records": len(records), "initial_passages": len(passages),
            "evaluation_content_units_scanned": units, "evaluation_tokens_scanned": token_count,
            "removals": removals, "question_matches": question_matches,
            "near_candidates": near_candidates, "removed_passage_ids": sorted(removed),
            "pending_review_passage_ids": sorted(pending),
            "near_policy": "retain review records; quarantine unreviewed candidates from this pilot, no semantic-only deletion"}
