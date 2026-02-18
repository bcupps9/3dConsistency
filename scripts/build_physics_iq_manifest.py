#!/usr/bin/env python3
"""
Build a prepare_inference_layout-compatible manifest from Physics-IQ descriptions.csv.

Outputs JSONL rows with:
  - sample_id
  - prompt
  - ground_truth_video

Resolution strategy:
  1) exact filename match under --video-search-roots
  2) normalized filename match (handles inserted tokens like video-masks/30FPS)
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


NOISE_TOKENS = {
    "video",
    "videos",
    "mask",
    "masks",
    "full",
    "split",
    "switch",
    "frames",
    "fullvideos",
    "splitvideos",
    "videomasks",
    "switchframes",
    "conditioning",
    "testing",
    "real",
    "fps",
    "8fps",
    "16fps",
    "24fps",
    "30fps",
    "take",
    "1",
    "2",
}


def _safe_id(value: str, idx: int) -> str:
    value = value.strip()
    if not value:
        value = f"physics_{idx:06d}"
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return value or f"physics_{idx:06d}"


def _parse_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _split_csv_arg(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _normalize_name(name: str) -> str:
    stem = Path(name).stem.lower()
    # Remove common container/source tokens and fps/take markers that may be inserted
    # by downloaded filenames but absent in descriptions.csv.
    stem = re.sub(r"(?:^|[_-])(?:full[-_]?videos|split[-_]?videos|video[-_]?masks|switch[-_]?frames)(?:[_-]|$)", "_", stem)
    stem = re.sub(r"(?:^|[_-])(?:8|16|24|30)fps(?:[_-]|$)", "_", stem)
    stem = re.sub(r"(?:^|[_-])take[-_]?[12](?:[_-]|$)", "_", stem)
    parts = re.split(r"[^a-z0-9]+", stem)
    filtered = [p for p in parts if p and p not in NOISE_TOKENS]
    return "".join(filtered)


def _build_indexes(search_roots: Iterable[Path]) -> tuple[Dict[str, List[Path]], Dict[str, List[Path]]]:
    by_basename: Dict[str, List[Path]] = {}
    by_normalized: Dict[str, List[Path]] = {}
    for root in search_roots:
        if not root.exists():
            continue
        for p in root.rglob("*.mp4"):
            rp = p.resolve()
            by_basename.setdefault(rp.name, []).append(rp)
            norm = _normalize_name(rp.name)
            by_normalized.setdefault(norm, []).append(rp)
    return by_basename, by_normalized


def _rank_match(path: Path, prefer_substrings: Sequence[str], take_filter: str) -> tuple[int, int, int, str]:
    s = str(path).lower()
    take_rank = 1
    if take_filter and take_filter.lower() in s:
        take_rank = 0
    pref_rank = len(prefer_substrings)
    for i, sub in enumerate(prefer_substrings):
        if sub and sub.lower() in s:
            pref_rank = i
            break
    depth_rank = len(path.parts)
    len_rank = len(s)
    return (take_rank, pref_rank, depth_rank, len_rank)


def _pick_best_match(matches: List[Path], prefer_substrings: Sequence[str], take_filter: str) -> Path | None:
    if not matches:
        return None
    ranked = sorted(matches, key=lambda p: _rank_match(p, prefer_substrings, take_filter))
    return ranked[0]


def _resolve_video_path(
    row: dict,
    filename_columns: list[str],
    by_basename: Dict[str, List[Path]],
    by_normalized: Dict[str, List[Path]],
    prefer_substrings: Sequence[str],
    take_filter: str,
) -> Path | None:
    candidates: list[str] = []
    for col in filename_columns:
        raw = row.get(col)
        if raw:
            raw_s = str(raw).strip()
            if raw_s:
                candidates.append(raw_s)

    for cand in candidates:
        names_to_try = [Path(cand).name]
        if not Path(cand).suffix:
            names_to_try.append(f"{cand}.mp4")

        for name in names_to_try:
            matches = by_basename.get(name, [])
            best = _pick_best_match(matches, prefer_substrings, take_filter)
            if best is not None:
                return best

        for name in names_to_try:
            norm = _normalize_name(name)
            nmatches = by_normalized.get(norm, [])
            best = _pick_best_match(nmatches, prefer_substrings, take_filter)
            if best is not None:
                return best

    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Physics-IQ JSONL manifest for prepare_inference_layout.py")
    parser.add_argument("--descriptions-csv", required=True, help="Path to descriptions.csv")
    parser.add_argument("--output-manifest", required=True, help="Output JSONL path")
    parser.add_argument(
        "--video-search-roots",
        required=True,
        help="Comma-separated directories containing Physics-IQ mp4 files",
    )
    parser.add_argument(
        "--filename-columns",
        default="scenario,generated_video_name",
        help="Comma-separated columns to resolve video filenames from",
    )
    parser.add_argument(
        "--prompt-column",
        default="description",
        help="CSV column used for prompt text",
    )
    parser.add_argument(
        "--id-column",
        default="generated_video_name",
        help="CSV column used for sample_id (falls back to filename columns/index)",
    )
    parser.add_argument(
        "--take-filter",
        default="",
        help="Optional substring filter (e.g., take-1 or take-2)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Max rows to emit (0 means all)")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle before limit")
    parser.add_argument("--seed", type=int, default=0, help="Shuffle seed")
    parser.add_argument(
        "--prefer-path-substrings",
        default="full-videos,split-videos,video-masks",
        help="Preferred path substrings in priority order when multiple videos match.",
    )
    parser.add_argument(
        "--debug-misses",
        type=int,
        default=5,
        help="Print up to this many unresolved rows for debugging (0 disables).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    csv_path = Path(args.descriptions_csv).expanduser().resolve()
    rows = _parse_csv(csv_path)
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")

    filename_columns = _split_csv_arg(args.filename_columns)
    search_roots = [Path(x).expanduser().resolve() for x in _split_csv_arg(args.video_search_roots)]
    prefer_substrings = _split_csv_arg(args.prefer_path_substrings)
    by_basename, by_normalized = _build_indexes(search_roots)

    indices = list(range(len(rows)))
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(indices)

    out_path = Path(args.output_manifest).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wrote = 0
    skipped = 0
    seen: set[str] = set()
    debug_miss_count = 0
    considered_after_filter = 0

    with out_path.open("w", encoding="utf-8") as f:
        for out_idx, idx in enumerate(indices, 1):
            row = rows[idx]

            prompt = str(row.get(args.prompt_column, "")).strip()
            if not prompt:
                skipped += 1
                continue

            if args.take_filter:
                probe = " ".join(str(row.get(c, "")) for c in filename_columns).lower()
                if args.take_filter.lower() not in probe:
                    skipped += 1
                    continue
            considered_after_filter += 1
            if args.limit > 0 and considered_after_filter > args.limit:
                break

            video_path = _resolve_video_path(
                row,
                filename_columns,
                by_basename,
                by_normalized,
                prefer_substrings,
                args.take_filter,
            )
            if video_path is None or not video_path.is_file():
                if args.debug_misses > 0 and debug_miss_count < args.debug_misses:
                    debug_miss_count += 1
                    raw_names = [str(row.get(c, "")).strip() for c in filename_columns if row.get(c)]
                    print(f"[debug miss {debug_miss_count}] row_index={idx} names={raw_names}")
                skipped += 1
                continue

            raw_id = str(row.get(args.id_column, "")).strip()
            if not raw_id:
                for c in filename_columns:
                    if row.get(c):
                        raw_id = str(row[c]).strip()
                        break
            sample_id = _safe_id(Path(raw_id).stem if raw_id else "", out_idx)
            if sample_id in seen:
                sample_id = f"{sample_id}_{out_idx:06d}"
            seen.add(sample_id)

            f.write(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "prompt": prompt,
                        "ground_truth_video": str(video_path),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            wrote += 1

    print(f"Wrote {wrote} rows to {out_path}")
    print(f"Skipped {skipped} rows (missing prompt/video/match)")
    print(f"Considered rows after take-filter: {considered_after_filter}")
    print(f"Indexed {sum(len(v) for v in by_basename.values())} mp4 files from search roots")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
