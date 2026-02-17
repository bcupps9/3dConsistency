#!/usr/bin/env python3
"""
Build a prepare_inference_layout-compatible manifest from WISA in Hugging Face datasets.

Outputs JSONL rows with:
  - sample_id
  - prompt
  - ground_truth_video

Supports either:
  1) loading from HF hub/cache via load_dataset(dataset_id, split, cache_dir)
  2) loading from a save_to_disk directory via load_from_disk(dataset_path)
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk


DEFAULT_PROMPT_KEYS = (
    "prompt",
    "caption",
    "captions",
    "description",
    "video_description",
    "video_caption",
    "text",
    "instruction",
    "query",
)
DEFAULT_VIDEO_KEYS = (
    "video",
    "video_path",
    "path",
    "video_name",
    "filename",
    "file_name",
    "video_filename",
    "video_file",
)
DEFAULT_ID_KEYS = ("sample_id", "id", "uid", "name", "video_id", "sha256")


def _parse_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _pick(record: Dict[str, Any], keys: Iterable[str]) -> Optional[Any]:
    for k in keys:
        if k in record and record[k] is not None:
            val = record[k]
            if isinstance(val, str) and not val.strip():
                continue
            return val
    return None


def _safe_id(value: str, index: int) -> str:
    value = value.strip()
    if not value:
        value = f"wisa_{index:06d}"
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return value or f"wisa_{index:06d}"


def _resolve_video_path(
    sample: Dict[str, Any],
    video_keys: list[str],
    sample_id: str,
    materialize_dir: Optional[Path],
    video_root: Optional[Path],
    basename_index: Optional[Dict[str, Path]],
) -> Optional[Path]:
    def _resolve_string_path(raw_path: str) -> Optional[Path]:
        candidate = Path(raw_path)
        if candidate.exists():
            return candidate.resolve()

        if video_root is not None:
            rooted = (video_root / raw_path).expanduser()
            if rooted.exists():
                return rooted.resolve()

            if not Path(raw_path).suffix:
                rooted_mp4 = (video_root / f"{raw_path}.mp4").expanduser()
                if rooted_mp4.exists():
                    return rooted_mp4.resolve()

        if basename_index is not None:
            base = Path(raw_path).name
            if base in basename_index:
                return basename_index[base]
            if not Path(base).suffix and f"{base}.mp4" in basename_index:
                return basename_index[f"{base}.mp4"]

        return None

    raw_video = _pick(sample, video_keys)
    if raw_video is None:
        return None

    # Common cases in HF datasets:
    # - string path
    # - dict {"path": "...", "bytes": ...}
    if isinstance(raw_video, str):
        return _resolve_string_path(raw_video)

    if isinstance(raw_video, dict):
        path_val = raw_video.get("path")
        if isinstance(path_val, str) and path_val:
            path = _resolve_string_path(path_val)
            if path is not None:
                return path

        bytes_val = raw_video.get("bytes")
        if bytes_val is not None:
            if materialize_dir is None:
                raise ValueError(
                    f"Sample {sample_id} has in-memory video bytes but no --materialize-video-dir was provided."
                )
            materialize_dir.mkdir(parents=True, exist_ok=True)
            out_path = materialize_dir / f"{sample_id}.mp4"
            out_path.write_bytes(bytes_val)
            return out_path

    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build WISA JSONL manifest for prepare_inference_layout.py")
    parser.add_argument("--output-manifest", required=True, help="Output JSONL path.")
    parser.add_argument(
        "--dataset-id",
        default="qihoo360/WISA-80K",
        help="HF dataset id (used when --dataset-path is not set).",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to load (for --dataset-id mode).",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional HF cache dir for load_dataset mode.",
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Optional local directory from datasets.save_to_disk/load_from_disk.",
    )
    parser.add_argument(
        "--arrow-dir",
        default=None,
        help="Optional directory containing local .arrow files. Uses no HF hub call.",
    )
    parser.add_argument(
        "--prompt-keys",
        default=",".join(DEFAULT_PROMPT_KEYS),
        help="Comma-separated preferred prompt columns.",
    )
    parser.add_argument(
        "--video-keys",
        default=",".join(DEFAULT_VIDEO_KEYS),
        help="Comma-separated preferred video columns.",
    )
    parser.add_argument(
        "--id-keys",
        default=",".join(DEFAULT_ID_KEYS),
        help="Comma-separated preferred id columns.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max rows to emit (0 means all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Shuffle seed. Ignored unless --shuffle is set.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle before applying --limit.",
    )
    parser.add_argument(
        "--materialize-video-dir",
        default=None,
        help="Where to write mp4 files if dataset rows only have in-memory video bytes.",
    )
    parser.add_argument(
        "--video-root",
        default=None,
        help="Optional root dir used when video column only stores filename/relative path.",
    )
    parser.add_argument(
        "--video-search-roots",
        default=None,
        help=(
            "Optional comma-separated roots to index *.mp4 by basename for filename-only rows. "
            "Example: /n/netscratch/.../datasets/raw/.hf,/n/netscratch/.../raw/wisa_videos"
        ),
    )
    return parser.parse_args()


def _load_arrow_dir(arrow_dir: Path) -> Dataset:
    arrow_files = sorted(arrow_dir.rglob("*.arrow"))
    if not arrow_files:
        raise FileNotFoundError(f"No .arrow files found under {arrow_dir}")
    datasets = [Dataset.from_file(str(p)) for p in arrow_files]
    if len(datasets) == 1:
        return datasets[0]
    return concatenate_datasets(datasets)


def _build_basename_index(search_roots: list[Path]) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for root in search_roots:
        if not root.exists():
            continue
        for p in root.rglob("*.mp4"):
            name = p.name
            if name not in index:
                index[name] = p.resolve()
    return index


def main() -> int:
    args = parse_args()

    prompt_keys = _parse_list(args.prompt_keys)
    video_keys = _parse_list(args.video_keys)
    id_keys = _parse_list(args.id_keys)
    materialize_dir = (
        Path(args.materialize_video_dir).expanduser().resolve()
        if args.materialize_video_dir
        else None
    )
    video_root = Path(args.video_root).expanduser().resolve() if args.video_root else None
    search_roots: list[Path] = []
    if args.video_search_roots:
        search_roots = [
            Path(x.strip()).expanduser().resolve()
            for x in args.video_search_roots.split(",")
            if x.strip()
        ]
    basename_index = _build_basename_index(search_roots) if search_roots else None

    mode_count = int(bool(args.dataset_path)) + int(bool(args.arrow_dir))
    if mode_count > 1:
        raise ValueError("Use only one of --dataset-path or --arrow-dir.")

    if args.arrow_dir:
        ds = _load_arrow_dir(Path(args.arrow_dir).expanduser().resolve())
    elif args.dataset_path:
        ds = load_from_disk(args.dataset_path)
        if hasattr(ds, "keys"):
            if args.split not in ds:
                raise ValueError(f"Split '{args.split}' not found in dataset path {args.dataset_path}")
            ds = ds[args.split]
    else:
        ds = load_dataset(args.dataset_id, split=args.split, cache_dir=args.cache_dir)

    n = len(ds)
    indices = list(range(n))
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(indices)
    if args.limit > 0:
        indices = indices[: args.limit]

    out_path = Path(args.output_manifest).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wrote = 0
    skipped = 0
    seen_ids: set[str] = set()
    with out_path.open("w", encoding="utf-8") as f:
        for out_idx, idx in enumerate(indices, 1):
            sample = ds[idx]

            raw_id = _pick(sample, id_keys)
            sample_id = _safe_id(str(raw_id) if raw_id is not None else "", out_idx)
            if sample_id in seen_ids:
                sample_id = f"{sample_id}_{out_idx:06d}"
            seen_ids.add(sample_id)

            prompt_val = _pick(sample, prompt_keys)
            if prompt_val is None:
                skipped += 1
                continue
            prompt = str(prompt_val).strip()
            if not prompt:
                skipped += 1
                continue

            video_path = _resolve_video_path(
                sample,
                video_keys,
                sample_id,
                materialize_dir,
                video_root,
                basename_index,
            )
            if video_path is None or not video_path.is_file():
                skipped += 1
                continue

            row = {
                "sample_id": sample_id,
                "prompt": prompt,
                "ground_truth_video": str(video_path.resolve()),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            wrote += 1

    print(f"Wrote {wrote} rows to {out_path}")
    print(f"Skipped {skipped} rows (missing prompt/video)")
    if basename_index is not None:
        print(f"Indexed {len(basename_index)} mp4 filenames from search roots")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
