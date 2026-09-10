#!/usr/bin/env python3
"""Run one sample per Table 1 subset task and method."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from table1_common import add_common_arguments, preflight, run_group
except ImportError:  # ``python -m scripts.table1_smoke``
    from scripts.table1_common import add_common_arguments, preflight, run_group


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser, default_samples=1)
    args = parser.parse_args()
    args.max_samples = 1
    try:
        preflight(args)
        payloads = [run_group(args, group, max_samples=1) for group in ("qa", "summary", "samsum")]
        output = Path(args.output_root) / "smoke_summary.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payloads, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"[table1-smoke] PASS: {output}")
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI must report a useful failure
        print(f"[table1-smoke] ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
