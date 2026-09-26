from .KVPacketRecompute import EPIC


def cal_epic_indices(sample_lens, num_recompute, skip_first=False):
    """Match KVPacket's EPIC boundary-index helper."""
    if not isinstance(num_recompute, int) or isinstance(num_recompute, bool) or num_recompute < 0:
        raise ValueError("num_recompute must be a non-negative integer")
    indices, current = [], 0
    for i, length in enumerate(sample_lens):
        end = current + min(int(length), num_recompute)
        if not skip_first or i > 0:
            indices.extend(range(current, end))
        current += int(length)
    return indices


__all__ = ["EPIC"]
