#!/usr/bin/env python3
"""Run the complete five-dataset A3 Table 1 reproduction subset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from table1_common import add_common_arguments, preflight, run_group
except ImportError:  # ``python -m scripts.run_table1``
    from scripts.table1_common import add_common_arguments, preflight, run_group


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser, default_samples=200)
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=("qa", "summary", "samsum"),
        default=("qa", "summary", "samsum"),
    )
    args = parser.parse_args()
    try:
        preflight(args)
        payloads = [
            run_group(args, group, max_samples=args.max_samples)
            for group in args.groups
        ]
        output = Path(args.output_root) / "run_summary.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payloads, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"[table1] PASS: {output}")
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI must report a useful failure
        print(f"[table1] ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
