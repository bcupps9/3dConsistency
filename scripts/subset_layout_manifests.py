#!/usr/bin/env python3
"""Subset prepared layout manifests consistently across models/tasks.

Use case:
- prepared layout contains left/center/right perspective triplets
- keep exactly one perspective per triplet (with preference order)
- cap selected samples per dataset
- apply same selected sample_ids to all model/task manifests
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


DEFAULT_MODELS = ("wan22", "wan21", "lvp")
DEFAULT_TASKS = ("t2v", "i2v")
PERSPECTIVES = ("left", "center", "right")

PERSPECTIVE_RE = re.compile(r"_perspective-(left|center|right)_")
LEADING_INDEX_RE = re.compile(r"^\d+_")


@dataclass
class Row:
    sample_id: str
    payload: Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Subset existing layout manifests.")
    parser.add_argument("--run-root", required=True, help="Existing prepared RUN_ROOT.")
    parser.add_argument(
        "--datasets",
        required=True,
        help="Comma-separated dataset names (e.g. physics_iq,wisa80k).",
    )
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated models to update. Default: wan22,wan21,lvp",
    )
    parser.add_argument(
        "--tasks",
        default=",".join(DEFAULT_TASKS),
        help="Comma-separated tasks to update. Default: t2v,i2v",
    )
    parser.add_argument(
        "--max-per-dataset",
        type=int,
        default=20,
        help="Maximum selected sample_ids per dataset.",
    )
    parser.add_argument(
        "--perspective-preference",
        default="center,left,right",
        help="Comma-separated preference order for perspective triplets.",
    )
    parser.add_argument(
        "--reference-model",
        default="wan22",
        help="Model used to discover canonical sample ordering.",
    )
    parser.add_argument(
        "--reference-task",
        default="t2v",
        help="Task used to discover canonical sample ordering.",
    )
    parser.add_argument(
        "--backup-suffix",
        default=".pre_subset.bak",
        help="Backup suffix for original manifests.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selection and counts without writing manifests.",
    )
    return parser.parse_args()


def _split_csv(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _read_jsonl(path: Path) -> List[Row]:
    rows: List[Row] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            sample_id = str(payload.get("sample_id", "")).strip()
            if not sample_id:
                continue
            rows.append(Row(sample_id=sample_id, payload=payload))
    return rows


def _write_jsonl(path: Path, rows: Iterable[Row]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.payload, ensure_ascii=False) + "\n")


def _sample_group(sample_id: str) -> Tuple[str, Optional[str]]:
    normalized = LEADING_INDEX_RE.sub("", sample_id)
    match = PERSPECTIVE_RE.search(normalized)
    if not match:
        return normalized, None
    perspective = match.group(1)
    group_key = PERSPECTIVE_RE.sub("_", normalized)
    return group_key, perspective


def _select_ids(
    ordered_ids: Sequence[str],
    max_per_dataset: int,
    preference: Sequence[str],
) -> List[str]:
    grouped: Dict[str, Dict[str, str]] = {}
    group_order: List[str] = []

    for sid in ordered_ids:
        gkey, perspective = _sample_group(sid)
        if gkey not in grouped:
            grouped[gkey] = {}
            group_order.append(gkey)
        if perspective is None:
            # singleton/non-triplet sample
            grouped[gkey]["__single__"] = sid
        else:
            grouped[gkey][perspective] = sid

    selected: List[str] = []
    for gkey in group_order:
        bucket = grouped[gkey]
        chosen: Optional[str] = None

        # singleton/non-triplet fallback
        if "__single__" in bucket:
            chosen = bucket["__single__"]
        else:
            for p in preference:
                if p in bucket:
                    chosen = bucket[p]
                    break
            if chosen is None and bucket:
                # deterministic fallback
                chosen = bucket[sorted(bucket.keys())[0]]

        if chosen is None:
            continue
        selected.append(chosen)
        if max_per_dataset > 0 and len(selected) >= max_per_dataset:
            break

    return selected


def _manifest_path(run_root: Path, model: str, dataset: str, task: str) -> Path:
    return run_root / model / dataset / task / "inputs" / "manifest.jsonl"


def main() -> int:
    args = parse_args()

    run_root = Path(args.run_root).expanduser().resolve()
    if not run_root.is_dir():
        raise FileNotFoundError(f"RUN_ROOT not found: {run_root}")

    datasets = _split_csv(args.datasets)
    models = _split_csv(args.models)
    tasks = _split_csv(args.tasks)
    preference = _split_csv(args.perspective_preference)
    bad = [p for p in preference if p not in PERSPECTIVES]
    if bad:
        raise ValueError(f"Invalid perspective(s): {bad}. Allowed: {PERSPECTIVES}")

    if not datasets or not models or not tasks:
        raise ValueError("datasets/models/tasks must be non-empty.")

    for dataset in datasets:
        ref_manifest = _manifest_path(run_root, args.reference_model, dataset, args.reference_task)
        if not ref_manifest.is_file():
            raise FileNotFoundError(
                f"Reference manifest not found for dataset '{dataset}': {ref_manifest}"
            )

        ref_rows = _read_jsonl(ref_manifest)
        ordered_ids = [r.sample_id for r in ref_rows]
        selected_ids = _select_ids(
            ordered_ids=ordered_ids,
            max_per_dataset=args.max_per_dataset,
            preference=preference,
        )
        selected_set: Set[str] = set(selected_ids)

        print(
            f"[{dataset}] selected {len(selected_ids)} sample_ids "
            f"(max_per_dataset={args.max_per_dataset}, preference={','.join(preference)})"
        )
        if selected_ids:
            print(f"[{dataset}] first 5 selected: {selected_ids[:5]}")

        for model in models:
            for task in tasks:
                manifest_path = _manifest_path(run_root, model, dataset, task)
                if not manifest_path.is_file():
                    print(f"skip missing manifest: {manifest_path}")
                    continue

                rows = _read_jsonl(manifest_path)
                out_rows = [r for r in rows if r.sample_id in selected_set]

                backup_path = manifest_path.with_name(manifest_path.name + args.backup_suffix)
                print(
                    f"  [{model}/{dataset}/{task}] {len(rows)} -> {len(out_rows)} rows "
                    f"(backup: {backup_path.name})"
                )
                if args.dry_run:
                    continue

                if not backup_path.exists():
                    backup_path.write_text(
                        "".join(json.dumps(r.payload, ensure_ascii=False) + "\n" for r in rows),
                        encoding="utf-8",
                    )
                _write_jsonl(manifest_path, out_rows)

    if args.dry_run:
        print("Dry-run complete; no files were written.")
    else:
        print("Manifest subsetting complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
