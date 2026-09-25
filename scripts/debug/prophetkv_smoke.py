#!/usr/bin/env python3
"""Small ProphetKV KVBench correctness smoke.

Default run: two samples per task, FullPrefill, ProphetKV@1.0, and
ProphetKV@20%.  ``--methods`` and all common runner options are supported.
"""

if __package__:
    from .prophetkv_kvbench_common import main
else:
    from prophetkv_kvbench_common import main


if __name__ == "__main__":
    main("smoke")
