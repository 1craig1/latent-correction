# Latent Correction: VideoQA stability diagnostics

This repository prepares a controlled test of **cross-benchmark negative transfer**: a reasoning enhancement can improve some benchmarks while hurting others under the same evaluation protocol. The proposed transfer from local continual-learning (CL) drift compensation is an **unverified hypothesis**. No VideoQA model result has been produced in this repository yet.

## What is included

- `cl_reference/`: unmodified copies of the three local CL scripts (`run_all_cl.py`, `geom_cl_sequence.py`, `geom_cl_maptype.py`). They are provenance snapshots, **not** standalone scripts in this repository; their original `config`, `geom_common`, cached data, and datasets remain in the original CL project.
- `latent_correction/alignment.py`: reusable source-to-target CL operators: identity, orthogonal rotation, rotation with scale, affine, ridge, and SDC. Source means the enhanced method; target means original Qwen for the **same video and question**.
- `score_benchmarks.py`: measure actual answer accuracy per benchmark, gain/loss relative to original Qwen, worst drop, and number of harmed benchmarks.
- `extract_qwen_features.py` → `pair_features.py` → `diagnose_alignment.py`: collect paired Qwen representations and test whether a CL map improves held-out alignment and a fixed nearest-class-mean (NCM) readout. This **does not** measure corrected generated answers.
- `same_input_probe.py`: secondary diagnostic that repeats the same processed video/question under greedy and sampled decoding. This measures answer repeatability, a separate definition of stability.

The original CL code and its adapted operator implementation are deliberately separate so that local historical results are not presented as VideoQA results. Source file SHA-256 values are in [MOTIVATION.md](MOTIVATION.md).

## Run on the lab A6000

Use Python 3.10+ with a CUDA build of PyTorch suitable for the lab machine, then install `requirements.txt`. The official Qwen2.5-VL-7B checkpoint is used at BF16 with a pinned revision. Model weights, videos, and benchmark data are **not** included.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Prepare a JSONL manifest with one line per question. Paths are relative to the manifest file. `split` is either `anchor` (fit maps and the fixed NCM readout) or `eval` (held-out test). Keep entire videos, not only questions, disjoint between splits to avoid leakage. Use the same manifest and visual sampling settings for every model.

```json
{"id":"clip01-q1","video":"videos/clip01.mp4","question":"What happens after the person stands up?","options":["They leave","They sit again","They fall","They wave"],"gold":"A","benchmark":"example-benchmark","split":"anchor"}
{"id":"clip02-q1","video":"videos/clip02.mp4","question":"Which object moves first?","options":["Ball","Box","Chair","Cup"],"gold":"B","benchmark":"example-benchmark","split":"eval"}
```

Extract original Qwen and one compatible method variant on the **same manifest**:

```bash
python extract_qwen_features.py --manifest data/items.jsonl --output-dir results/base
python extract_qwen_features.py --manifest data/items.jsonl --output-dir results/method --checkpoint /path/to/compatible-qwen-checkpoint
python pair_features.py --source results/method/features.npz --target results/base/features.npz --output results/paired.npz
python diagnose_alignment.py --paired results/paired.npz --output results/diagnostic.json
```

A method that changes only the prompt can be tested by supplying `--prompt-template /path/to/template.txt` in the second extraction. The template should contain `{question}` and `{options}`. This is a **prompt experiment**, not a reproduction of TextCoT fine-tuning. Method-specific architectures such as Mull-Tokens need their own inference adapter and actual checkpoint; a generic Qwen loader does not reproduce them. Features from different checkpoints/prompts must be checked for comparable semantics before interpreting a map.

The extraction saves each item separately, records completion/failure, and resumes completed items when config and video hash match. The probe and all diagnostics store results under `results/`, which Git ignores.

To score **actual generated answers** from original Qwen and any method, prepare one prediction per item and method:

```json
{"id":"clip02-q1","method":"baseline","answer":"B"}
{"id":"clip02-q1","method":"mull","answer":"A"}
```

Every method must cover every manifest item. An invalid or unparseable answer counts as wrong. Then run:

```bash
python score_benchmarks.py --manifest data/items.jsonl --predictions results/predictions.jsonl --output results/benchmark_scores.json
```

For the previously requested unchanged-input test on original Qwen:

```bash
python same_input_probe.py --video data/clip.mp4 --question 'What happens first?' \
  --option 'The man sits' --option 'The man leaves' --gold A \
  --output-dir results/same-input
```

Frame sampling uses the full clip duration at a fixed FPS, with a maximum-frame safeguard; it does not pass every decoded frame of a long video. See [MOTIVATION.md](MOTIVATION.md) for definitions, evidence, and limits.
