#!/usr/bin/env python3
"""Build and optionally serve a private HTML gallery for inference outputs.

This script reads prepared per-task manifests from one or more RUN_ROOTs,
collects generated outputs for wan22/wan21/lvp, writes a static HTML page, and
creates symlinks to media files under the gallery output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
from dataclasses import dataclass, field
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


DEFAULT_MODELS = ("wan22", "wan21", "lvp")


@dataclass
class Entry:
    dataset: str
    task: str
    sample_id: str
    prompt: str = ""
    ground_truth_video: Optional[Path] = None
    image_path: Optional[Path] = None
    outputs: Dict[str, Optional[Path]] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and optionally serve a private gallery for run outputs."
    )
    parser.add_argument(
        "--run-root",
        action="append",
        default=[],
        help="RUN_ROOT to include (repeat flag for multiple roots).",
    )
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated model keys to render. Default: wan22,wan21,lvp",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory where gallery files are written. Default: <first-run-root>/gallery",
    )
    parser.add_argument(
        "--include-missing",
        action="store_true",
        help="Include rows with no generated model outputs yet.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Optional cap on rendered rows (0 means no cap).",
    )
    parser.add_argument(
        "--no-serve",
        action="store_true",
        help="Only build gallery files; do not launch HTTP server.",
    )
    parser.add_argument("--bind", default="127.0.0.1", help="Bind address for HTTP server.")
    parser.add_argument("--port", type=int, default=8000, help="HTTP server port.")
    return parser.parse_args()


def _normalize_run_roots(raw_run_roots: Iterable[str]) -> List[Path]:
    roots: List[Path] = []
    for raw in raw_run_roots:
        value = raw.strip()
        if not value:
            continue
        p = Path(value).expanduser().resolve()
        if not p.is_dir():
            raise FileNotFoundError(f"RUN_ROOT not found: {p}")
        roots.append(p)
    if not roots:
        env_single = os.environ.get("RUN_ROOT", "").strip()
        env_many = os.environ.get("RUN_ROOTS", "").strip()
        if env_single:
            roots.append(Path(env_single).expanduser().resolve())
        elif env_many:
            for token in env_many.split(","):
                token = token.strip()
                if token:
                    roots.append(Path(token).expanduser().resolve())
    if not roots:
        raise ValueError("Provide --run-root (or set RUN_ROOT / RUN_ROOTS).")
    return roots


def _discover_manifests(run_root: Path, models: List[str]) -> List[Tuple[str, str, str, Path]]:
    manifests: List[Tuple[str, str, str, Path]] = []
    for model in models:
        model_root = run_root / model
        if not model_root.is_dir():
            continue
        for dataset_dir in sorted(p for p in model_root.iterdir() if p.is_dir()):
            for task_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
                manifest_path = task_dir / "inputs" / "manifest.jsonl"
                if manifest_path.is_file():
                    manifests.append((model, dataset_dir.name, task_dir.name, manifest_path))
    return manifests


def _safe_media_name(path: Path) -> str:
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:16]
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem)[:80]
    suffix = path.suffix.lower()
    if not suffix:
        suffix = ".bin"
    return f"{digest}_{stem}{suffix}"


def _link_media(src: Optional[Path], media_dir: Path) -> Optional[str]:
    if src is None:
        return None
    if not src.is_file():
        return None

    media_dir.mkdir(parents=True, exist_ok=True)
    dst = media_dir / _safe_media_name(src)
    target = Path(os.path.relpath(src, dst.parent))
    if dst.exists() or dst.is_symlink():
        if dst.is_dir():
            return None
        try:
            current = Path(os.readlink(dst))
        except OSError:
            current = None
        if current != target:
            dst.unlink()
    if not dst.exists():
        dst.symlink_to(target)
    return f"media/{dst.name}"


def _maybe_pick_existing(current: Optional[Path], candidate: Optional[Path]) -> Optional[Path]:
    if candidate is None:
        return current
    if not candidate.is_file() or candidate.stat().st_size <= 0:
        return current
    if current is None:
        return candidate
    try:
        if candidate.stat().st_mtime > current.stat().st_mtime:
            return candidate
    except OSError:
        return current
    return current


def _collect_entries(run_roots: List[Path], models: List[str]) -> Dict[Tuple[str, str, str], Entry]:
    entries: Dict[Tuple[str, str, str], Entry] = {}

    for run_root in run_roots:
        for model, dataset, task, manifest_path in _discover_manifests(run_root, models):
            with manifest_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    sample_id = str(row.get("sample_id", "")).strip()
                    if not sample_id:
                        continue
                    key = (dataset, task, sample_id)
                    entry = entries.get(key)
                    if entry is None:
                        entry = Entry(dataset=dataset, task=task, sample_id=sample_id)
                        entry.outputs = {m: None for m in models}
                        entries[key] = entry

                    if not entry.prompt:
                        entry.prompt = str(row.get("prompt", "")).strip()

                    gt = str(row.get("ground_truth_video", "")).strip()
                    if gt and entry.ground_truth_video is None:
                        entry.ground_truth_video = Path(gt)

                    image = str(row.get("image_path", "")).strip()
                    if image and entry.image_path is None:
                        entry.image_path = Path(image)

                    output_video = str(row.get("output_video", "")).strip()
                    candidate = Path(output_video) if output_video else None
                    entry.outputs[model] = _maybe_pick_existing(entry.outputs.get(model), candidate)

    return entries


def _render_html(
    entries: List[Entry],
    models: List[str],
    output_dir: Path,
    run_roots: List[Path],
    include_missing: bool,
) -> Tuple[int, int]:
    media_dir = output_dir / "media"
    rows_written = 0
    rows_with_any = 0

    parts: List[str] = []
    parts.append("<!doctype html>")
    parts.append("<html><head><meta charset='utf-8'>")
    parts.append("<meta name='viewport' content='width=device-width, initial-scale=1'>")
    parts.append("<title>Private Inference Gallery</title>")
    parts.append(
        "<style>"
        "body{font-family:Arial,sans-serif;margin:16px;}"
        "table{border-collapse:collapse;width:100%;font-size:13px;}"
        "th,td{border:1px solid #ddd;padding:6px;vertical-align:top;}"
        "th{position:sticky;top:0;background:#fafafa;z-index:1;}"
        "video,img{max-width:280px;height:auto;display:block;}"
        "pre{white-space:pre-wrap;word-break:break-word;margin:0;max-width:420px;}"
        ".muted{color:#666;font-size:12px;}"
        "</style>"
    )
    parts.append("</head><body>")
    parts.append("<h2>Private Inference Gallery</h2>")
    parts.append("<div class='muted'>")
    parts.append("Run roots:<br>")
    for root in run_roots:
        parts.append(f"{html.escape(str(root))}<br>")
    parts.append("</div><br>")
    parts.append("<table>")
    parts.append(
        "<tr><th>dataset/task/sample</th><th>prompt + i2v image</th><th>ground truth</th>"
        + "".join(f"<th>{html.escape(m)}</th>" for m in models)
        + "</tr>"
    )

    for entry in entries:
        has_any = any(p is not None and p.is_file() for p in entry.outputs.values())
        if has_any:
            rows_with_any += 1
        if (not include_missing) and (not has_any):
            continue

        prompt = html.escape(entry.prompt)
        img_ref = _link_media(entry.image_path, media_dir)
        gt_ref = _link_media(entry.ground_truth_video, media_dir)

        parts.append("<tr>")
        parts.append(
            f"<td><b>{html.escape(entry.dataset)}</b>/<b>{html.escape(entry.task)}</b>"
            f"<br>{html.escape(entry.sample_id)}</td>"
        )
        prompt_cell = "<details><summary>prompt</summary><pre>{}</pre></details>".format(prompt)
        if img_ref:
            prompt_cell += f"<img src='{html.escape(img_ref)}' loading='lazy'>"
        parts.append(f"<td>{prompt_cell}</td>")

        if gt_ref:
            parts.append(
                "<td><video controls preload='metadata' src='{}'></video></td>".format(
                    html.escape(gt_ref)
                )
            )
        else:
            parts.append("<td>missing</td>")

        for model in models:
            ref = _link_media(entry.outputs.get(model), media_dir)
            if ref:
                cell = "<video controls preload='metadata' src='{}'></video>".format(html.escape(ref))
            else:
                cell = "missing"
            parts.append(f"<td>{cell}</td>")
        parts.append("</tr>")
        rows_written += 1

    parts.append("</table>")
    parts.append(
        f"<p class='muted'>Rendered rows: {rows_written} | Rows with any model output: {rows_with_any}</p>"
    )
    parts.append("</body></html>")

    index_path = output_dir / "index.html"
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path.write_text("\n".join(parts), encoding="utf-8")
    return rows_written, rows_with_any


def main() -> int:
    args = parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        raise ValueError("No models specified.")

    run_roots = _normalize_run_roots(args.run_root)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (run_roots[0] / "gallery").resolve()
    )

    entries_map = _collect_entries(run_roots, models)
    entries = sorted(entries_map.values(), key=lambda e: (e.dataset, e.task, e.sample_id))
    if args.max_rows > 0:
        entries = entries[: args.max_rows]

    rendered, with_any = _render_html(
        entries=entries,
        models=models,
        output_dir=output_dir,
        run_roots=run_roots,
        include_missing=args.include_missing,
    )

    print(f"Wrote gallery: {output_dir / 'index.html'}")
    print(f"Rendered rows: {rendered} (rows with any model output: {with_any})")

    if args.no_serve:
        return 0

    url = f"http://{args.bind}:{args.port}"
    print(f"Serving {output_dir} at {url}")
    print("For SSH tunnel from local machine:")
    print(
        f"  ssh -N -L {args.port}:127.0.0.1:{args.port} $USER@login.rc.fas.harvard.edu"
    )

    handler = partial(SimpleHTTPRequestHandler, directory=str(output_dir))
    server = ThreadingHTTPServer((args.bind, args.port), handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
