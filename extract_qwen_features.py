#!/usr/bin/env python3
"""Extract a Qwen2.5-VL final prompt-token feature for paired VideoQA items.

The item-level cache makes a long A6000 extraction resumable. These features
support an offline CL-transfer diagnostic; they are not corrected answers.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np

from same_input_probe import (
    DEFAULT_CHECKPOINT, DEFAULT_REVISION, append_jsonl, atomic_json,
    make_item, question_prompt, read_jsonl, sha256_file, tensor_fingerprint,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                   help="Official baseline or compatible Qwen2.5-VL method checkpoint")
    p.add_argument("--revision", help="Model revision; official baseline defaults to a pinned commit")
    p.add_argument("--prompt-template", type=Path,
                   help="Optional text file with {question} and {options} placeholders")
    p.add_argument("--fps", type=float, default=1.0)
    p.add_argument("--max-pixels", type=int, default=360 * 420)
    p.add_argument("--max-sampled-frames", type=int, default=32)
    return p


def load_manifest(path: Path):
    rows = read_jsonl(path)
    if not rows:
        raise ValueError("Manifest is empty")
    items = []
    for number, row in enumerate(rows, 1):
        item = make_item(row, str(row.get("id", number)), path.parent)
        if item.gold is None:
            raise ValueError(f"{item.item_id}: gold option is required")
        split = str(row.get("split", ""))
        benchmark = str(row.get("benchmark", ""))
        if split not in ("anchor", "eval") or not benchmark:
            raise ValueError(f"{item.item_id}: split=anchor/eval and benchmark are required")
        items.append((item, split, benchmark))
    ids = [item.item_id for item, _, _ in items]
    if len(ids) != len(set(ids)):
        raise ValueError("Manifest item IDs must be unique")
    return items


def format_prompt(item, template: str | None) -> str:
    if template is None:
        return question_prompt(item)
    options = "\n".join(f"{letter}. {option}" for letter, option in
                        zip("ABCDE", item.options))
    return template.format(question=item.question, options=options)


def save_item(path: Path, **arrays) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
    temporary.replace(path)


def main() -> int:
    args = parser().parse_args()
    manifest = args.manifest.resolve()
    items = load_manifest(manifest)
    if args.fps <= 0 or args.max_pixels <= 0 or args.max_sampled_frames <= 0:
        raise ValueError("fps, max-pixels and max-sampled-frames must be positive")
    template = args.prompt_template.read_text(encoding="utf-8") if args.prompt_template else None
    revision = args.revision or (DEFAULT_REVISION if args.checkpoint == DEFAULT_CHECKPOINT else None)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    item_dir = output / "items"
    item_dir.mkdir(exist_ok=True)
    config = {"manifest_sha256": sha256_file(manifest), "checkpoint": args.checkpoint,
              "revision": revision, "prompt_template": template,
              "fps": args.fps, "max_pixels": args.max_pixels,
              "max_sampled_frames": args.max_sampled_frames,
              "feature": "final prompt token, final hidden layer before LM readout"}
    config_path = output / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError("Output directory has a different extraction config")
    else:
        atomic_json(config_path, config)

    paths = [item_dir / (hashlib.sha256(item.item_id.encode()).hexdigest() + ".npz")
             for item, _, _ in items]
    hashes = [sha256_file(item.video) for item, _, _ in items]
    pending = []
    for (item, _, _), path, video_hash in zip(items, paths, hashes):
        if path.exists():
            with np.load(path, allow_pickle=False) as cached:
                if str(cached["id"]) != item.item_id or str(cached["video_sha256"]) != video_hash:
                    raise ValueError(f"Cached item changed: {item.item_id}")
        else:
            pending.append((item, path, video_hash))

    if pending:
        import torch
        import transformers
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from tqdm import tqdm

        if not torch.cuda.is_available():
            raise RuntimeError("Official Qwen2.5-VL-7B feature extraction requires a CUDA GPU")
        model_args = {"revision": revision} if revision else {}
        processor = AutoProcessor.from_pretrained(args.checkpoint, **model_args)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.checkpoint, torch_dtype=torch.bfloat16, device_map="auto", **model_args
        ).eval()
        devices = {str(parameter.device) for parameter in model.parameters()}
        if any(not device.startswith("cuda") for device in devices):
            raise RuntimeError(f"Model is partly offloaded: {sorted(devices)}")
        model_device = next(model.parameters()).device
        atomic_json(output / "runtime.json", {
            "python": sys.version.split()[0], "torch": torch.__version__,
            "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name(0),
            "started_at": datetime.now(timezone.utc).isoformat(),
        })
        for item, path, video_hash in tqdm(pending, desc="Qwen features", unit="item"):
            try:
                prompt = format_prompt(item, template)
                messages = [{"role": "user", "content": [
                    {"type": "video", "video": item.video.as_uri(),
                     "fps": args.fps, "max_pixels": args.max_pixels},
                    {"type": "text", "text": prompt},
                ]}]
                chat_text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                images, videos, video_kwargs = process_vision_info(
                    messages, return_video_kwargs=True
                )
                frames = int(videos[0].shape[0])
                if frames > args.max_sampled_frames:
                    raise ValueError(f"{item.item_id}: {frames} frames exceed cap")
                inputs = processor(text=[chat_text], images=images, videos=videos,
                                   padding=True, return_tensors="pt", **video_kwargs)
                input_sha, _ = tensor_fingerprint(inputs, torch)
                inputs = inputs.to(model_device)
                with torch.inference_mode():
                    outputs = model(**inputs, output_hidden_states=True,
                                    use_cache=False, return_dict=True)
                if not outputs.hidden_states:
                    raise RuntimeError("Model returned no hidden states")
                feature = outputs.hidden_states[-1][0, -1].float().cpu().numpy()
                save_item(path, id=item.item_id, feature=feature,
                          video_sha256=video_hash, processed_input_sha256=input_sha,
                          sampled_frames=frames)
                append_jsonl(output / "progress.jsonl", {
                    "id": item.item_id, "status": "done", "sampled_frames": frames,
                    "processed_input_sha256": input_sha,
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                })
                del outputs, inputs, images, videos
                torch.cuda.empty_cache()
            except Exception as error:
                append_jsonl(output / "failures.jsonl", {
                    "id": item.item_id, "status": "failed", "error": repr(error),
                    "traceback": traceback.format_exc(),
                    "failed_at": datetime.now(timezone.utc).isoformat(),
                })
                raise

    packed = {"id": [], "split": [], "benchmark": [], "gold": [],
              "video_sha256": [], "processed_input_sha256": [], "feature": []}
    for (item, split, benchmark), path in zip(items, paths):
        with np.load(path, allow_pickle=False) as row:
            packed["id"].append(item.item_id)
            packed["split"].append(split)
            packed["benchmark"].append(benchmark)
            packed["gold"].append(item.gold)
            packed["video_sha256"].append(str(row["video_sha256"]))
            packed["processed_input_sha256"].append(str(row["processed_input_sha256"]))
            packed["feature"].append(row["feature"])
    packed["feature"] = np.stack(packed["feature"])
    save_item(output / "features.npz", **packed)
    print(f"Saved {len(items)} features to {output / 'features.npz'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
