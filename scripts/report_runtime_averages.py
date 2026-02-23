#!/usr/bin/env python3
"""Report average runtime by model/dataset/task from layout progress logs.

This script parses RUN_ROOT/progress_*.log produced by scripts/remote_infer_layout.sh.

- For wan21/wan22: uses per-sample `done ... elapsed=<s>s` lines directly.
- For lvp: estimates per-sample runtime from slice walltime:
    (SLICE DONE timestamp - SLICE START timestamp) / checked
  where `checked` is read from the SLICE DONE line.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple


TS_RE = re.compile(r"^\[(?P<ts>[^\]]+)\]\s+(?P<msg>.*)$")
WAN_DONE_RE = re.compile(
    r"^(?P<key>wan(?:21|22)/[^/]+/(?:t2v|i2v)):\s+done\s+sample=.*\selapsed=(?P<elapsed>\d+)s"
)
LVP_START_RE = re.compile(r"^SLICE START (?P<key>lvp/[^/]+/(?:t2v|i2v)):")
LVP_DONE_RE = re.compile(
    r"^SLICE DONE (?P<key>lvp/[^/]+/(?:t2v|i2v)):\s+.*\schecked=(?P<checked>\d+)\s+"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize runtime averages from layout progress logs.")
    p.add_argument("--run-root", required=True, help="RUN_ROOT path.")
    p.add_argument(
        "--latest-only",
        action="store_true",
        help="Use only the newest progress_*.log instead of all progress logs.",
    )
    p.add_argument(
        "--top",
        type=int,
        default=0,
        help="Optional limit for number of rows to print (0 = all).",
    )
    return p.parse_args()


def parse_ts(ts: str) -> Optional[dt.datetime]:
    # Handles formats like "2026-02-20T21:12:32-05:00" from `date -Is`.
    try:
        return dt.datetime.fromisoformat(ts)
    except ValueError:
        return None


def add_sample(store: Dict[str, List[float]], key: str, sec: float) -> None:
    if sec <= 0:
        return
    store.setdefault(key, []).append(sec)


def summarize(store: Dict[str, List[float]], top: int = 0) -> List[Tuple[str, int, float, float, float]]:
    rows: List[Tuple[str, int, float, float, float]] = []
    for key, vals in store.items():
        if not vals:
            continue
        n = len(vals)
        avg = statistics.fmean(vals)
        med = statistics.median(vals)
        p90 = statistics.quantiles(vals, n=10)[8] if n >= 10 else max(vals)
        rows.append((key, n, avg, med, p90))
    rows.sort(key=lambda x: x[0])
    if top > 0:
        rows = rows[:top]
    return rows


def main() -> int:
    args = parse_args()
    run_root = Path(args.run_root).expanduser().resolve()
    if not run_root.is_dir():
        raise FileNotFoundError(f"RUN_ROOT not found: {run_root}")

    progress_logs = sorted(run_root.glob("progress_*.log"), key=lambda p: p.stat().st_mtime)
    if not progress_logs:
        raise FileNotFoundError(f"No progress_*.log files under {run_root}")
    if args.latest_only:
        progress_logs = [progress_logs[-1]]

    elapsed_by_key: Dict[str, List[float]] = {}
    lvp_open_slice_start: Dict[str, dt.datetime] = {}

    for log_path in progress_logs:
        with log_path.open("r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.rstrip("\n")
                m = TS_RE.match(raw)
                if not m:
                    continue
                ts = parse_ts(m.group("ts"))
                msg = m.group("msg")

                # Wan per-sample elapsed.
                wm = WAN_DONE_RE.match(msg)
                if wm:
                    key = wm.group("key")
                    sec = float(wm.group("elapsed"))
                    add_sample(elapsed_by_key, key, sec)
                    continue

                # LVP slice start/done (derive per-sample estimate).
                sm = LVP_START_RE.match(msg)
                if sm and ts is not None:
                    lvp_open_slice_start[sm.group("key")] = ts
                    continue

                dm = LVP_DONE_RE.match(msg)
                if dm and ts is not None:
                    key = dm.group("key")
                    checked = int(dm.group("checked"))
                    start = lvp_open_slice_start.get(key)
                    if start is None or checked <= 0:
                        continue
                    sec = (ts - start).total_seconds() / checked
                    add_sample(elapsed_by_key, key, sec)
                    lvp_open_slice_start.pop(key, None)

    rows = summarize(elapsed_by_key, top=args.top)
    if not rows:
        print("No runtime rows parsed.")
        return 0

    print(f"RUN_ROOT={run_root}")
    print(f"progress_logs={len(progress_logs)}")
    print()
    print(f"{'slice':42} {'n':>5} {'avg_s':>10} {'med_s':>10} {'p90_s':>10} {'avg_min':>10}")
    print("-" * 95)
    for key, n, avg, med, p90 in rows:
        print(f"{key:42} {n:5d} {avg:10.1f} {med:10.1f} {p90:10.1f} {avg/60:10.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

