#!/usr/bin/env python3
"""Full KVBench partial Table-1 reproduction for ProphetKV.

Default run: all locally available samples for the six overlapping KVBench
tasks, comparing FullPrefill, NaiveReuse, CacheBlend, and ProphetKV@20%.
"""

from prophetkv_kvbench_common import main


if __name__ == "__main__":
    main("full")

