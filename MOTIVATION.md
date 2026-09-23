# Research motivation and falsifiable CL transfer

## 1. Define the instability precisely

For benchmark `b`, let `Acc(base,b)` be the accuracy of original Qwen2.5-VL-7B and `Acc(method,b)` the accuracy after adding one reasoning method **under the same data, visual budget, prompt/scoring rules, and checkpoint comparison contract**. Define `Δ(method,b) = Acc(method,b) − Acc(base,b)` in percentage points. A method exhibits **cross-benchmark negative transfer** when at least one tested benchmark has `Δ < 0`. Report the full delta vector, number of benchmarks with `Δ < 0`, worst delta, and macro mean. The target for a correction is fewer or smaller negative deltas **without erasing gains**; zero drops across predeclared benchmarks is a strong empirical goal, not an assumed result.

Raw accuracy of original Qwen varies across benchmarks because tasks differ in difficulty. That variation alone does not show model instability. Repeating an identical processed video/question with fixed decoding tests **answer repeatability**, a separate property; `same_input_probe.py` measures it.

## 2. Evidence motivating the question

In [Mull-Tokens Table 1](https://arxiv.org/html/2512.10941v2), all cited rows use a Qwen2.5-VL-7B backbone, and deltas are relative to the *direct-answer fine-tuned* row, **not raw Qwen**. TextCoT FT gains **+10.64 pp** on BLINK Jigsaw but loses **−12.90 pp** on BLINK Relative Depth. Mull-Tokens gains **+15.34 pp** on Jigsaw and **+3.05 pp** on the table's overall average, yet loses **−4.03 pp** on Relative Depth. These are task-level tradeoffs. The table spans image, multi-image, and video benchmarks; it does not by itself establish a VideoQA-only effect or a causal mechanism.

[VideoLatent's stated limitations](https://arxiv.org/html/2606.22870v1) say self-generated latent thoughts **may** be irrelevant to video/question context and that results may not generalize to all video-language benchmarks. This is an author-stated limitation, not direct evidence that particular tokens were irrelevant or that CL-style drift caused task regressions.

## 3. What the local CL method actually does

The original CL pipeline extracts **paired old/new features for the same anchor images**, estimates drift, moves old prototypes, and evaluates nearest-class-mean classification. Its `m_rot` centers source and target anchors, fits orthogonal Procrustes, then transforms source-space prototypes. `m_rotscale`, `m_affine`, and `m_sdc` are other local operators. `geom_cl_maptype.py` compares rotation, linear, and ridge on a separate cached-feature test. The method does not train the backbone during compensation.

Original local results are mixed: in a full CIFAR-100 CL run, final accuracy was 17.18% without compensation and 20.98% with rotation. In a *different* map-type diagnostic at 2,000 anchors, rotation scored 59.7%, linear 70.4%, and ridge 70.0%; ridge beat rotation at all tested anchor counts. These are **CL results under different protocols**, never VideoQA evidence. The reusable VQA port therefore includes ridge as a required comparator. The archival sources are byte-for-byte copies from the local project:

| Source snapshot | SHA-256 |
|---|---|
| `cl_reference/run_all_cl.py` | `a2e5e2b20abc96e4773488f120e8e0ebc7550344ff7e451a20cb246342f938d1` |
| `cl_reference/geom_cl_sequence.py` | `5a848e4339a0b5ee5e59dd192bc8f532435955add85b8f8e184201cbd4e90b7d` |
| `cl_reference/geom_cl_maptype.py` | `00e5481613e9e9139e59bf700d4462ec167455e2c9641b06340bcd7e9196b2e0` |

## 4. Testable transfer to VideoQA

**Hypothesis:** for some method/benchmark regressions, task-relevant information survives in the method-enhanced hidden state, but the answer readout is misaligned with the baseline's representation. Given the same anchor video/questions, fit a map from method features to original-Qwen features. Evaluate on held-out videos/questions; compare identity, original rotation, rotation+scale, affine, SDC, and ridge. Then test *actual generated answers* with and without an operational correction module. If aligned feature error improves but answer accuracy does not, the feature map is insufficient. If errors arise because the video evidence was never sampled or encoded, a linear map cannot recover it.

The current code reaches the **paired-feature and fixed-readout diagnostic**. It does not hook a fitted map into Qwen's autoregressive generation, does not run Mull-Tokens or VideoLatent, and has not yet measured an A6000 VideoQA result. Using a last-prompt-token vector and a fixed NCM readout is an intentional small test of the CL analogy, not a replacement for end-to-end VideoQA evaluation. A later generation adapter must use each method's actual checkpoint and token path, and be judged on per-benchmark answer deltas, per-item gain/loss, and whether existing gains survive.

An enhanced method may change vocabulary, prompts, latent-token positions, and weights. In that case its final prompt-token representation may not correspond to the same semantic position as baseline Qwen. A failure of one linear map could mean noncomparable features, poor anchors, nonlinear shift, or missing visual evidence. Keep these as competing explanations rather than calling every benchmark drop "representation drift."
