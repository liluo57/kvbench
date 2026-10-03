"""Token boundary validation is independent of CUDA and text tokenization."""
import pytest

from methods.hypic.backend.kvbench_hypic.srt.pic.segmenter import validate_segments


def test_explicit_segments_cover_special_token_ids_exactly():
    validate_segments([151644, 3, 151645, 4], [(0, 1), (1, 3), (3, 4)])


@pytest.mark.parametrize('ids,segments', [
    ([], []), ([[1, 2]], [(0, 2)]), ([True], [(0, 1)]),
    ([1, 2], [(1, 2)]), ([1, 2], [(0, 1)]),
    ([1, 2], [(0, 1), (0, 2)]), ([1, 2], [(0, 3)]),
    ([1, 2], [(0, 0), (0, 2)]), ([1, 2], [(False, 2)]),
])
def test_rejects_missing_overlapping_empty_or_invalid_ranges(ids, segments):
    with pytest.raises(ValueError):
        validate_segments(ids, segments)
