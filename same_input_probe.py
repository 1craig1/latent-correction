#!/usr/bin/env python3
"""Repeat one or more unchanged video questions with the official Qwen2.5-VL-7B.

Preprocessing runs once per item. Every repeat receives the same processed
video/question tensors; only the generation seed changes in sampling mode.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback
from typing import Any


DEFAULT_CHECKPOINT = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"


@dataclass(frozen=True)
class Item:
    item_id: str
    video: Path
    question: str
    options: tuple[str, ...]
    gold: str | None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(canonical_json(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid {path}:{line_number}: {error}") from error
    return rows


def make_item(raw: dict[str, Any], item_id: str, video_root: Path) -> Item:
    required = ("video", "question", "options")
    for field in required:
        if field not in raw:
            raise ValueError(f"Item {item_id}: missing {field}")
    video = Path(raw["video"]).expanduser()
    if not video.is_absolute():
        video = video_root / video
    video = video.resolve()
    if not video.is_file():
        raise FileNotFoundError(f"Item {item_id}: video not found: {video}")
    question = str(raw["question"]).strip()
    options = tuple(str(option).strip() for option in raw["options"])
    if not question or not 2 <= len(options) <= 5 or any(not option for option in options):
        raise ValueError(f"Item {item_id}: supply a question and 2-5 nonempty options")
    gold = raw.get("gold")
    if gold is not None:
        gold = str(gold).strip().upper()
        if gold not in "ABCDE"[: len(options)]:
            raise ValueError(f"Item {item_id}: gold must be an option letter")
    return Item(item_id, video, question, options, gold)


def load_items(args: argparse.Namespace) -> list[Item]:
    if args.manifest:
        if args.video or args.question or args.option or args.gold:
            raise ValueError("Use either --manifest or --video/--question/--option")
        path = args.manifest.resolve()
        rows = read_jsonl(path)
        if not rows:
            raise ValueError(f"Manifest is empty: {path}")
        items = [make_item(row, str(row.get("id", number)), path.parent)
                 for number, row in enumerate(rows, start=1)]
    else:
        if not args.video or not args.question or not args.option:
            raise ValueError("Supply --video, --question and at least two --option values")
        items = [make_item({"video": str(args.video), "question": args.question,
                            "options": args.option, "gold": args.gold},
                           "single", Path.cwd())]
    ids = [item.item_id for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("Every manifest item id must be unique")
    return items


def question_prompt(item: Item) -> str:
    options = "\n".join(f"{letter}. {option}" for letter, option in
                        zip("ABCDE", item.options))
    return ("Answer the question using the video. Choose one option. "
            "Output only its letter.\n"
            f"Question: {item.question}\n{options}\nAnswer:")


def parse_choice(raw_answer: str, option_count: int) -> str | None:
    letters = "ABCDE"[:option_count]
    bare = re.fullmatch(
        rf"\s*(?:<answer>\s*)?([{letters}])\s*[.)]?\s*(?:</answer>)?\s*",
        raw_answer, re.IGNORECASE,
    )
    if bare:
        return bare.group(1).upper()
    explicit = re.search(
        rf"\b(?:answer|option|choice)\s*(?:is|:)?\s*([{letters}])\b",
        raw_answer, re.IGNORECASE,
    )
    return explicit.group(1).upper() if explicit else None


def answer_label(record: dict[str, Any]) -> str:
    if record["parsed_answer"] is not None:
        return record["parsed_answer"]
    return "<INVALID> " + " ".join(record["raw_answer"].casefold().split())


def summarize(items: list[Item], records: list[dict[str, Any]],
              greedy_repeats: int, sample_repeats: int) -> dict[str, Any]:
    by_key = {(row["item_id"], row["mode"], row["repeat"]): row for row in records}
    if len(by_key) != len(records):
        raise ValueError("Duplicate run records found")
    result: dict[str, Any] = {"updated_at": utc_now(), "modes": {}}
    for mode, target in (("greedy", greedy_repeats), ("sample", sample_repeats)):
        if target == 0:
            continue
        complete = []
        for item in items:
            rows = [by_key.get((item.item_id, mode, repeat)) for repeat in range(target)]
            if any(row is None for row in rows):
                continue
            labels = [answer_label(row) for row in rows]
            counts = Counter(labels)
            correct = ([row["parsed_answer"] == item.gold for row in rows]
                       if item.gold else None)
            complete.append({
                "item_id": item.item_id,
                "different_answers": len(counts) > 1,
                "answer_counts": dict(counts),
                "modal_fraction": max(counts.values()) / target,
                "correct_fraction": sum(correct) / target if correct is not None else None,
                "consistently_correct": bool(correct and all(correct)),
                "consistently_wrong": bool(correct is not None and not any(correct)
                                           and len(counts) == 1),
            })
        known = [row["correct_fraction"] for row in complete
                 if row["correct_fraction"] is not None]
        result["modes"][mode] = {
            "completed_items": len(complete),
            "planned_items": len(items),
            "repeats_per_item": target,
            "disagreement_fraction": (
                sum(row["different_answers"] for row in complete) / len(complete)
                if complete else None),
            "mean_correct_fraction": sum(known) / len(known) if known else None,
            "items": complete,
        }
    return result


def tensor_fingerprint(inputs: Any, torch: Any) -> tuple[str, dict[str, Any]]:
    """Hash every processor output before it is moved to the GPU."""
    digest = hashlib.sha256()
    shapes = {}
    for name, value in sorted(inputs.items()):
        digest.update(name.encode())
        if isinstance(value, torch.Tensor):
            raw = value.detach().contiguous().view(torch.uint8).numpy().tobytes()
            digest.update(raw)
            shapes[name] = {"shape": list(value.shape), "dtype": str(value.dtype)}
        else:
            encoded = canonical_json(value).encode()
            digest.update(encoded)
            shapes[name] = {"type": type(value).__name__}
    return digest.hexdigest(), shapes


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_argument_group("input")
    source.add_argument("--manifest", type=Path, help="JSONL of video QA items")
    source.add_argument("--video", type=Path, help="One local video file")
    source.add_argument("--question", help="Question for --video")
    source.add_argument("--option", action="append", help="Option text, in A-E order; repeat")
    source.add_argument("--gold", help="Correct option letter, if known")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--revision", default=DEFAULT_REVISION)
    p.add_argument("--fps", type=float, default=1.0,
                   help="Sample uniformly across the full video duration")
    p.add_argument("--max-pixels", type=int, default=360 * 420,
                   help="Maximum pixels per sampled frame")
    p.add_argument("--max-sampled-frames", type=int, default=32,
                   help="Fail if preprocessing samples more frames; no silent truncation")
    p.add_argument("--greedy-repeats", type=int, default=3)
    p.add_argument("--sample-repeats", type=int, default=10)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--base-seed", type=int, default=42)
    return p


def main() -> int:
    args = parser().parse_args()
    items = load_items(args)
    if args.fps <= 0 or args.max_pixels <= 0 or args.max_sampled_frames <= 0:
        raise ValueError("fps, max-pixels and max-sampled-frames must be positive")
    if args.greedy_repeats < 0 or args.sample_repeats < 0:
        raise ValueError("Repeat counts cannot be negative")
    if args.greedy_repeats + args.sample_repeats == 0:
        raise ValueError("Request at least one repeat")
    if args.sample_repeats and (args.temperature <= 0 or not 0 < args.top_p <= 1):
        raise ValueError("Sampling needs temperature > 0 and 0 < top-p <= 1")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "checkpoint": args.checkpoint, "revision": args.revision,
        "fps": args.fps, "max_pixels": args.max_pixels,
        "max_sampled_frames": args.max_sampled_frames,
        "greedy_repeats": args.greedy_repeats,
        "sample_repeats": args.sample_repeats,
        "temperature": args.temperature, "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens, "base_seed": args.base_seed,
        "items": [{"id": item.item_id, "video": str(item.video),
                   "video_sha256": sha256_file(item.video),
                   "question": item.question, "options": list(item.options),
                   "gold": item.gold} for item in items],
    }
    config_path = output_dir / "config.json"
    if config_path.exists():
        prior = json.loads(config_path.read_text(encoding="utf-8"))
        if prior != config:
            raise ValueError("Output directory has a different config; choose another directory")
    else:
        atomic_json(config_path, config)

    runs_path = output_dir / "runs.jsonl"
    input_path = output_dir / "input_fingerprints.jsonl"
    records = read_jsonl(runs_path)
    known_runs = {(row["item_id"], row["mode"], row["repeat"]) for row in records}
    if len(known_runs) != len(records):
        raise ValueError("Duplicate run records found")
    known_inputs = {row["item_id"]: row for row in read_jsonl(input_path)}
    plan = [(item, mode, repeat)
            for item in items
            for mode, count in (("greedy", args.greedy_repeats),
                                ("sample", args.sample_repeats))
            for repeat in range(count)]
    pending = [entry for entry in plan
               if (entry[0].item_id, entry[1], entry[2]) not in known_runs]
    if not pending:
        atomic_json(output_dir / "summary.json",
                    summarize(items, records, args.greedy_repeats, args.sample_repeats))
        print(f"All {len(plan)} runs are already complete: {output_dir}")
        return 0

    import torch
    import transformers
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    from tqdm import tqdm

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for the official 7B checkpoint")
    runtime = {
        "started_at": utc_now(), "python": sys.version.split()[0],
        "torch": torch.__version__, "transformers": transformers.__version__,
        "qwen_vl_utils": importlib.metadata.version("qwen-vl-utils"),
        "gpu": torch.cuda.get_device_name(0),
        "cuda_total_gib": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
    }
    atomic_json(output_dir / "runtime.json", runtime)
    print(f"Loading {args.checkpoint} at {args.revision} on {runtime['gpu']}", flush=True)
    processor = AutoProcessor.from_pretrained(args.checkpoint, revision=args.revision)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.checkpoint, revision=args.revision, torch_dtype="auto", device_map="auto"
    )
    model.eval()
    devices = {str(parameter.device) for parameter in model.parameters()}
    if any(not device.startswith("cuda") for device in devices):
        raise RuntimeError(f"Model was partly offloaded from CUDA: {sorted(devices)}")
    model_device = next(model.parameters()).device

    try:
        with tqdm(total=len(pending), desc="Repeated answers", unit="run") as bar:
            for item in items:
                item_plan = [(mode, repeat) for target, mode, repeat in pending
                             if target.item_id == item.item_id]
                if not item_plan:
                    continue
                prompt = question_prompt(item)
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
                sampled_frames = int(videos[0].shape[0])
                if sampled_frames > args.max_sampled_frames:
                    raise ValueError(
                        f"Item {item.item_id}: sampled {sampled_frames} frames, "
                        f"above --max-sampled-frames={args.max_sampled_frames}. "
                        "Use a shorter clip or a new documented FPS setting."
                    )
                inputs = processor(
                    text=[chat_text], images=images, videos=videos,
                    padding=True, return_tensors="pt", **video_kwargs
                )
                input_sha, input_shapes = tensor_fingerprint(inputs, torch)
                input_record = {
                    "item_id": item.item_id,
                    "video_sha256": sha256_file(item.video),
                    "prompt_sha256": hashlib.sha256(chat_text.encode()).hexdigest(),
                    "processed_input_sha256": input_sha,
                    "sampled_frames": sampled_frames,
                    "sampled_fps": video_kwargs.get("fps"),
                    "input_shapes": input_shapes,
                }
                prior_input = known_inputs.get(item.item_id)
                if prior_input and prior_input != input_record:
                    raise RuntimeError(
                        f"Item {item.item_id}: processed input changed since prior run"
                    )
                if not prior_input:
                    append_jsonl(input_path, input_record)
                    known_inputs[item.item_id] = input_record
                inputs = inputs.to(model_device)
                input_length = inputs["input_ids"].shape[1]

                for mode, repeat in item_plan:
                    seed = args.base_seed + repeat
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    generation = {"max_new_tokens": args.max_new_tokens,
                                  "do_sample": mode == "sample"}
                    if mode == "sample":
                        generation.update(temperature=args.temperature, top_p=args.top_p)
                    try:
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        with torch.inference_mode():
                            generated = model.generate(**inputs, **generation)
                        torch.cuda.synchronize()
                        seconds = time.perf_counter() - start
                        raw = processor.batch_decode(
                            generated[:, input_length:], skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        )[0].strip()
                        parsed = parse_choice(raw, len(item.options))
                        row = {
                            "item_id": item.item_id, "mode": mode,
                            "repeat": repeat, "seed": seed,
                            "raw_answer": raw, "parsed_answer": parsed,
                            "correct": parsed == item.gold if item.gold else None,
                            "seconds": round(seconds, 3),
                            "processed_input_sha256": input_sha,
                            "finished_at": utc_now(),
                        }
                        append_jsonl(runs_path, row)
                        records.append(row)
                        bar.update(1)
                        bar.set_postfix_str(f"{item.item_id} {mode} {parsed or 'invalid'}")
                    except Exception as error:
                        append_jsonl(output_dir / "failures.jsonl", {
                            "item_id": item.item_id, "mode": mode,
                            "repeat": repeat, "seed": seed,
                            "error": repr(error), "traceback": traceback.format_exc(),
                            "failed_at": utc_now(),
                        })
                        raise
                del inputs, images, videos
                torch.cuda.empty_cache()
    finally:
        atomic_json(output_dir / "summary.json",
                    summarize(items, records, args.greedy_repeats, args.sample_repeats))
    print(f"Saved {len(records)}/{len(plan)} runs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
