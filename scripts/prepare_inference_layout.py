#!/usr/bin/env python3
"""
Prepare a model/dataset/task inference layout for Wan2.2, Wan2.1, and LVP.

Given a manifest of tuples (prompt + ground-truth video, optional image), this
script creates:
  - shared sample assets
  - per-model/per-task directory trees
  - per-task JSONL manifests with resolved input/output paths

Target layout:
  <run_root>/
    datasets/<dataset>/samples/<sample_id>/
    wan22/<dataset>/{t2v,i2v}/...
    wan21/<dataset>/{t2v,i2v}/...
    lvp/<dataset>/{t2v,i2v}/...
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

DEFAULT_MODELS = ("wan22", "wan21", "lvp")
DEFAULT_TASKS = ("t2v", "i2v")

ID_KEYS = ("sample_id", "id", "uid", "name")
PROMPT_KEYS = ("prompt", "text_prompt", "caption")
GT_KEYS = ("ground_truth_video", "ground_truth", "gt_video", "video_path", "video")
IMAGE_KEYS = ("i2v_image", "image", "image_path", "first_frame", "first_frame_path")


@dataclass
class SampleRecord:
    sample_id: str
    prompt: str
    gt_video_src: Path
    image_src: Optional[Path]
    row_index: int


@dataclass
class SampleAssets:
    sample_id: str
    prompt_text: str
    prompt_path: Path
    gt_video_path: Path
    image_path: Optional[Path]
    metadata_path: Path


def _pick_value(record: Dict[str, str], keys: Iterable[str]) -> Optional[str]:
    for key in keys:
        value = record.get(key)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return None


def _safe_sample_id(raw_value: str, row_index: int) -> str:
    value = raw_value.strip()
    if not value:
        value = f"sample_{row_index:05d}"
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return value or f"sample_{row_index:05d}"


def _resolve_path(raw_path: str, manifest_dir: Path) -> Path:
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = manifest_dir / candidate
    return candidate.resolve()


def _read_manifest(path: Path) -> List[Dict[str, str]]:
    suffix = path.suffix.lower()
    rows: List[Dict[str, str]] = []
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for i, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}: invalid JSON on line {i}") from exc
                if not isinstance(data, dict):
                    raise ValueError(f"{path}: JSONL line {i} is not an object")
                rows.append({str(k): str(v) if v is not None else "" for k, v in data.items()})
    elif suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows.append({str(k): str(v) if v is not None else "" for k, v in row.items()})
    else:
        raise ValueError(f"Unsupported manifest extension: {suffix}. Use .jsonl or .csv")
    return rows


def _normalize_rows(rows: List[Dict[str, str]], manifest_dir: Path) -> List[SampleRecord]:
    seen: Dict[str, int] = {}
    out: List[SampleRecord] = []
    for index, row in enumerate(rows, 1):
        sample_id_raw = _pick_value(row, ID_KEYS) or ""
        sample_id = _safe_sample_id(sample_id_raw, index)

        if sample_id in seen:
            raise ValueError(
                f"Duplicate sample_id '{sample_id}' in rows {seen[sample_id]} and {index}"
            )
        seen[sample_id] = index

        prompt = _pick_value(row, PROMPT_KEYS)
        if not prompt:
            raise ValueError(f"Row {index}: missing prompt field ({', '.join(PROMPT_KEYS)})")

        gt_raw = _pick_value(row, GT_KEYS)
        if not gt_raw:
            raise ValueError(f"Row {index}: missing ground-truth video field ({', '.join(GT_KEYS)})")
        gt_path = _resolve_path(gt_raw, manifest_dir)
        if not gt_path.is_file():
            raise FileNotFoundError(f"Row {index}: ground-truth video not found: {gt_path}")

        image_raw = _pick_value(row, IMAGE_KEYS)
        image_path = None
        if image_raw:
            image_path = _resolve_path(image_raw, manifest_dir)
            if not image_path.is_file():
                raise FileNotFoundError(f"Row {index}: image not found: {image_path}")

        out.append(
            SampleRecord(
                sample_id=sample_id,
                prompt=prompt,
                gt_video_src=gt_path,
                image_src=image_path,
                row_index=index,
            )
        )
    return out


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _materialize_file(src: Path, dst: Path, mode: str) -> None:
    _ensure_parent(dst)
    if dst.exists() or dst.is_symlink():
        if dst.is_dir():
            raise IsADirectoryError(f"Refusing to overwrite directory: {dst}")
        dst.unlink()

    if mode == "symlink":
        dst.symlink_to(src)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unsupported materialization mode: {mode}")


def _extract_first_frame(video_path: Path, image_path: Path, ffmpeg_bin: str) -> None:
    _ensure_parent(image_path)
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        str(image_path),
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Failed to extract first frame with ffmpeg from {video_path}"
        ) from exc


def _write_text(path: Path, content: str) -> None:
    _ensure_parent(path)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, payload: Dict) -> None:
    _ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _prepare_shared_assets(
    run_root: Path,
    dataset_name: str,
    records: List[SampleRecord],
    mode: str,
    ffmpeg_bin: str,
    extract_first_frame: bool,
    strict_i2v_inputs: bool,
) -> List[SampleAssets]:
    shared_dataset_root = run_root / "datasets" / dataset_name / "samples"
    assets: List[SampleAssets] = []

    for record in records:
        sample_root = shared_dataset_root / record.sample_id
        sample_root.mkdir(parents=True, exist_ok=True)

        gt_ext = record.gt_video_src.suffix or ".mp4"
        gt_dst = sample_root / f"ground_truth{gt_ext}"
        _materialize_file(record.gt_video_src, gt_dst, mode)

        prompt_path = sample_root / "prompt.txt"
        _write_text(prompt_path, record.prompt + "\n")

        image_dst: Optional[Path]
        if record.image_src is not None:
            img_ext = record.image_src.suffix or ".png"
            image_dst = sample_root / f"input_image{img_ext}"
            _materialize_file(record.image_src, image_dst, mode)
        elif extract_first_frame:
            image_dst = sample_root / "input_image.png"
            _extract_first_frame(gt_dst, image_dst, ffmpeg_bin)
        else:
            image_dst = None
            if strict_i2v_inputs:
                raise ValueError(
                    f"Sample '{record.sample_id}' has no image and first-frame extraction is disabled."
                )

        metadata = {
            "sample_id": record.sample_id,
            "prompt": record.prompt,
            "source_ground_truth_video": str(record.gt_video_src),
            "source_image": str(record.image_src) if record.image_src else None,
            "ground_truth_video": str(gt_dst),
            "input_image": str(image_dst) if image_dst else None,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "row_index": record.row_index,
        }
        metadata_path = sample_root / "sample.json"
        _write_json(metadata_path, metadata)

        assets.append(
            SampleAssets(
                sample_id=record.sample_id,
                prompt_text=record.prompt,
                prompt_path=prompt_path,
                gt_video_path=gt_dst,
                image_path=image_dst,
                metadata_path=metadata_path,
            )
        )
    return assets


def _write_jsonl(path: Path, rows: List[Dict]) -> None:
    _ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_model_task_layout(
    run_root: Path,
    models: List[str],
    tasks: List[str],
    dataset_name: str,
    assets: List[SampleAssets],
    mode: str,
) -> None:
    for model in models:
        for task in tasks:
            task_root = run_root / model / dataset_name / task
            inputs_dir = task_root / "inputs"
            outputs_dir = task_root / "outputs"
            logs_dir = task_root / "logs"
            gt_dir = task_root / "ground_truth"
            prompt_dir = inputs_dir / "prompts"
            image_dir = inputs_dir / "images"

            for directory in (inputs_dir, outputs_dir, logs_dir, gt_dir, prompt_dir):
                directory.mkdir(parents=True, exist_ok=True)
            if task == "i2v":
                image_dir.mkdir(parents=True, exist_ok=True)

            manifest_rows: List[Dict] = []
            skipped_i2v: List[str] = []
            for sample in assets:
                prompt_dst = prompt_dir / f"{sample.sample_id}.txt"
                _materialize_file(sample.prompt_path, prompt_dst, mode)

                gt_dst = gt_dir / f"{sample.sample_id}{sample.gt_video_path.suffix}"
                _materialize_file(sample.gt_video_path, gt_dst, mode)

                output_video = outputs_dir / f"{sample.sample_id}.mp4"
                row = {
                    "sample_id": sample.sample_id,
                    "task": task,
                    "prompt": sample.prompt_text,
                    "prompt_path": str(prompt_dst),
                    "ground_truth_video": str(gt_dst),
                    "output_video": str(output_video),
                    "metadata_path": str(sample.metadata_path),
                }

                if task == "i2v":
                    if sample.image_path is None:
                        skipped_i2v.append(sample.sample_id)
                        continue
                    image_dst = image_dir / f"{sample.sample_id}{sample.image_path.suffix}"
                    _materialize_file(sample.image_path, image_dst, mode)
                    row["image_path"] = str(image_dst)

                manifest_rows.append(row)

            _write_jsonl(inputs_dir / "manifest.jsonl", manifest_rows)

            task_summary = {
                "model": model,
                "dataset": dataset_name,
                "task": task,
                "num_samples": len(manifest_rows),
                "skipped_i2v_samples": skipped_i2v,
                "manifest_path": str(inputs_dir / "manifest.jsonl"),
                "created_utc": datetime.now(timezone.utc).isoformat(),
            }
            _write_json(task_root / "layout_summary.json", task_summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare structured inference layout and manifests.")
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to tuple manifest (.jsonl or .csv).",
    )
    parser.add_argument(
        "--run-root",
        required=True,
        help="Root output directory (e.g., /n/netscratch/.../results/<run_id>).",
    )
    parser.add_argument(
        "--dataset-name",
        required=True,
        help="Dataset key used in output hierarchy (e.g., physics_iq).",
    )
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated model keys. Default: wan22,wan21,lvp",
    )
    parser.add_argument(
        "--tasks",
        default=",".join(DEFAULT_TASKS),
        help="Comma-separated task keys. Default: t2v,i2v",
    )
    parser.add_argument(
        "--materialize-mode",
        choices=("symlink", "hardlink", "copy"),
        default="symlink",
        help="How to place files in task directories.",
    )
    parser.add_argument(
        "--ffmpeg-bin",
        default="ffmpeg",
        help="ffmpeg binary used for first-frame extraction.",
    )
    parser.add_argument(
        "--no-extract-first-frame",
        action="store_true",
        help="Disable first-frame extraction when image is not present in manifest.",
    )
    parser.add_argument(
        "--allow-missing-i2v-image",
        action="store_true",
        help="Allow i2v entries to be skipped if no image is available.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    manifest_dir = manifest_path.parent

    run_root = Path(args.run_root).expanduser().resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if not models:
        raise ValueError("No models provided.")
    if not tasks:
        raise ValueError("No tasks provided.")

    unsupported_tasks = [task for task in tasks if task not in DEFAULT_TASKS]
    if unsupported_tasks:
        raise ValueError(f"Unsupported tasks: {unsupported_tasks}. Supported: {DEFAULT_TASKS}")

    rows = _read_manifest(manifest_path)
    records = _normalize_rows(rows, manifest_dir)
    if not records:
        raise ValueError("Manifest has no valid rows.")

    extract_first_frame = not args.no_extract_first_frame
    strict_i2v_inputs = not args.allow_missing_i2v_image

    assets = _prepare_shared_assets(
        run_root=run_root,
        dataset_name=args.dataset_name,
        records=records,
        mode=args.materialize_mode,
        ffmpeg_bin=args.ffmpeg_bin,
        extract_first_frame=extract_first_frame,
        strict_i2v_inputs=strict_i2v_inputs,
    )

    _build_model_task_layout(
        run_root=run_root,
        models=models,
        tasks=tasks,
        dataset_name=args.dataset_name,
        assets=assets,
        mode=args.materialize_mode,
    )

    summary = {
        "manifest": str(manifest_path),
        "run_root": str(run_root),
        "dataset_name": args.dataset_name,
        "models": models,
        "tasks": tasks,
        "num_samples": len(assets),
        "materialize_mode": args.materialize_mode,
        "extract_first_frame": extract_first_frame,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(run_root / "layout_summary.json", summary)

    print(f"Prepared layout for {len(assets)} samples.")
    print(f"run_root={run_root}")
    print(f"dataset={args.dataset_name}")
    print(f"models={','.join(models)} tasks={','.join(tasks)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
