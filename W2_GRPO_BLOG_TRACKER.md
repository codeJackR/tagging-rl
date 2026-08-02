# From SFT to RL: technical blog tracker for difficulty scoring and GRPO

**Document type:** living technical tracker and future blog brief
**Started:** 2026-08-02
**Current scope:** locked SFT checkpoint → sampled difficulty measurement → GRPO prompt selection
**Current status:** full 3,600-product difficulty run completed and independently audited; 1,702 mixed-outcome products are eligible for the predeclared GRPO band
**Update rule:** record measured results only after their artifact and checksum exist; keep planned settings clearly labeled as planned

---

## 1. Proposed post in one sentence

Before applying GRPO to a structured catalog tagger, we sampled eight answers from a locked SFT model for every training prompt and kept the prompts where the model sometimes succeeded and sometimes failed, creating the within-group reward variation that policy-gradient learning needs.

This sentence is still provisional because the sampled difficulty run and GRPO experiment have not happened yet.

## 2. Candidate titles

- From SFT to GRPO: finding the training examples that still teach a small model
- Why RL needs disagreement: building a difficulty filter for a catalog tagger
- Eight attempts per product: measuring where an SFT model is ready for RL
- Before GRPO, measure the reward variance
- Turning a verifier into an RL curriculum for structured extraction

## 3. Core technical thesis

GRPO compares multiple completions generated for the same prompt. If all eight completions receive the same reward, their group-relative advantages collapse toward zero and that prompt contributes little or no useful policy-gradient signal.

For this project, a product with pass rate `0.0` is currently too hard under the strict pass definition, while a product with pass rate `1.0` is already solved. The useful starting band is:

```text
0 < sft_pass_rate < 1
```

Examples with mixed outcomes let the optimizer compare better and worse answers produced by the same policy for the same product.

Important nuance for the final article: GRPO can use continuous and multi-component rewards, so binary pass-rate filtering is not a universal requirement. It is our curriculum-selection rule for this first controlled run. The eventual GRPO reward can still contain partial signals such as format validity, vocabulary compliance, rule compliance, and gold agreement.

---

## 4. Where this work begins

The preceding SFT experiment is complete and documented separately in `W2_BLOG_BRIEF.md`. This tracker should summarize only enough of that experiment to make the RL transition understandable.

### Locked baseline

| item | locked value |
|---|---|
| base model | `unsloth/Qwen2.5-1.5B-Instruct` |
| selected LoRA arm | attention + MLP |
| selected checkpoint | `runs/sft-combined-2epoch/checkpoint-406` |
| training epochs | 2 |
| optimizer steps | 406 |
| LoRA rank / alpha | 16 / 16 |
| trainable parameters | 18,464,768 |
| adapter weight file | `adapter_model.safetensors` |
| adapter bytes | 73,911,112 |
| adapter SHA-256 | `00ae54af4e380cff66695b36b244e3f1ff9aca85076b59a8eb6649d8c3a051af` |
| selection manifest | `runs/sft-selection.json` |
| selection-manifest SHA-256 | `e425635d323b3ffe9e7350fb61a2d9e1848345a95abab6b92032bf64d2718299` |

### Baseline evidence to carry into the next post

| metric | locked SFT | zero-shot comparison |
|---|---:|---:|
| frozen-eval products attempted | 300 | 300 |
| macro-F1 | 0.6411 | 0.1969, conditional on 196 parsed survivors |
| selective macro-F1 | 0.7170 | 0.2047 |
| coverage | 94.3% | 88.1% |
| schema validity | 100% | 62.7% |
| fully vocabulary-valid outputs | 88.7% | 0% |
| verifier-rule violations | 12 | 1,204 |
| missing predictions | 0 | 104 |

Bootstrap evidence from the frozen evaluation:

- SFT macro-F1 95% interval: `0.6258–0.6714`.
- Zero-shot conditional macro-F1 95% interval: `0.1609–0.2133`.
- Paired SFT-minus-zero-shot interval: `+0.4249–+0.4958`.
- Method: paired nonparametric row bootstrap, 5,000 replicates, seed `20260801`.

The absolute label scores remain provisional because only 78 individual evaluation cells received explicit human review. The like-for-like improvement, format validity, and verifier compliance are stronger claims than production accuracy.

### Data entering difficulty scoring

| item | value |
|---|---:|
| source | `data/train_weak.jsonl` |
| products | 3,600 |
| source SHA-256 | `1cbcbfba5ad379e7c66895d720a997edf913030ee1e76e4917101dfccb09530b` |
| rollouts per product | 8 |
| planned full completions | 28,800 |
| frozen test set used for difficulty scoring | no |

Difficulty is measured on the weakly labeled training pool, not the frozen 300-product test set. The frozen set remains reserved for model comparison after training.

---

## 5. The problem discovered before implementation

The dataset model already had this field:

```json
"difficulty": {
  "sft_pass_rate": null
}
```

`training.dataset.load_grpo_prompts()` could filter to `0 < sft_pass_rate < 1`, and deliberately raised an error while every value was null. Its error message referenced `training/score_difficulty.py`, but that script did not exist.

The larger gap was conceptual: the plan said to sample eight completions and record the fraction that were “correct,” but it did not define “correct.” Several interpretations were possible:

1. **Verifier-clean only.** This measures whether the output is structurally acceptable, but ignores whether the labels match the product.
2. **Every scorable label exactly correct.** Clear and non-arbitrary, but strict for a 15-field record.
3. **At least some percentage of fields correct.** More forgiving, but requires choosing an arbitrary threshold.
4. **A continuous reward threshold.** Potentially aligned with GRPO rewards, but makes the curriculum depend on an as-yet-unlocked reward scale.

We locked option 2, with verifier cleanliness added as a requirement.

---

## 6. Locked definition of a passing rollout

A sampled completion passes only if all of the following are true:

1. The raw response parses as one JSON object.
2. Its field names, shapes, and arities satisfy the schema.
3. Every committed value belongs to the controlled vocabulary.
4. It triggers no declarative cross-field verifier rules.
5. Every scorable gold field exactly matches.

Additional scoring rules:

- A gold field marked `unknown` is excluded because the listing provides no ground truth for it.
- `null` remains a scorable `not_applicable` answer.
- The multi-valued `details` field is compared as an order-insensitive set.
- A record with zero scorable gold fields cannot pass.
- Verifier normalization and output repair are disabled; raw output is graded exactly as generated.

### Why verifier validity alone is insufficient

The verifier answers, “Could this be a legal catalog record?” It does not answer, “Is this the correct record for this product?” A perfectly formatted answer with the wrong material is valid but incorrect. Difficulty scoring therefore combines verifier cleanliness with exact gold agreement.

### Sanity check against existing SFT predictions

Before spending GPU time, we applied the strict whole-record criterion to the 300 already-saved deterministic SFT predictions. This was not the difficulty run; it was a feasibility check.

| row-level scorable-label accuracy | products at or above threshold |
|---|---:|
| 50% | 296 / 300 |
| 60% | 285 / 300 |
| 70% | 252 / 300 |
| 75% | 239 / 300 |
| 80% | 200 / 300 |
| 90% | 109 / 300 |
| 100%, strict pass | 61 / 300 |

Observed row-accuracy quantiles:

| quantile | row accuracy |
|---|---:|
| minimum | 0.286 |
| p10 | 0.667 |
| p25 | 0.769 |
| median | 0.857 |
| p75 | 0.923 |
| p90 | 1.000 |
| maximum | 1.000 |

The 61/300 exact records showed that the definition was strict but not impossible. This check used deterministic evaluation outputs, while difficulty scoring will sample eight stochastic outputs per training product; their distributions should not be treated as directly comparable.

---

## 7. What `sft_pass_rate` means

For product `i`, let `pass(i,j)` be 1 when rollout `j` satisfies the locked pass definition and 0 otherwise. With eight rollouts:

```text
sft_pass_rate(i) = sum(pass(i, 0..7)) / 8
```

The only possible values are:

```text
0.000, 0.125, 0.250, 0.375, 0.500,
0.625, 0.750, 0.875, 1.000
```

Example: five passing answers and three failing answers produce `5 / 8 = 0.625`.

### Interpretation

| pass rate | interpretation for this run | initial GRPO selection |
|---:|---|---|
| 0.000 | all eight failed; possibly too hard or systematically broken | exclude |
| 0.125–0.875 | model produced both success and failure | retain |
| 1.000 | all eight passed; already solved under this sample | exclude |

Eight samples give a coarse estimate, not a stable psychological property of the product. Temperature, seed, prompt, checkpoint, and verifier version are part of the measurement.

---

## 8. Artifact design

The original `data/train_weak.jsonl` must remain unchanged. The run produces derived evidence instead.

### 8.1 Scored dataset

Planned full-run path:

```text
data/train_weak_sft_scored.jsonl
```

Each source row is preserved, with only this measured field populated:

```json
"difficulty": {
  "sft_pass_rate": 0.625
}
```

Safety properties:

- The writer refuses to use the source path as the output path.
- Source and pass-rate SKU sets must match exactly.
- Duplicate source SKUs are rejected.
- Rates outside `[0,1]` are rejected.
- Manifest construction later requires every rate to be a multiple of `1/8`.

### 8.2 Auditable rollout records

Planned full-run path:

```text
runs/sft-difficulty-k8/rollouts.jsonl.gz
```

There will be 28,800 records after a successful full run. Each record contains:

```json
{
  "sku_id": "shopify:example:123",
  "rollout_index": 3,
  "raw_output": "{...literal model text...}",
  "passed": false,
  "schema_valid": true,
  "vocab_valid": true,
  "rule_violations": [],
  "errors": [],
  "correct_labels": 11,
  "scorable_labels": 13,
  "incorrect_labels": ["material", "occasion"],
  "excluded_gold_unknown_labels": ["fit", "waistline"],
  "generation_seed": 20260802,
  "completion_tokens": 108
}
```

The exact incorrect-field list was added after review because “11 of 13 correct” did not fully explain why a rollout failed.

Writer guarantees:

- Every SKU must have indices `0–7` exactly once.
- Duplicate or incomplete groups fail.
- Records are sorted by SKU and rollout index.
- JSON keys and separators are canonical.
- Gzip uses timestamp `0` and no embedded output filename.
- Identical logical records therefore produce identical compressed bytes and SHA-256 hashes, independent of batching order or destination filename.

### 8.3 Run manifest

Planned full-run path:

```text
runs/sft-difficulty-k8/manifest.json
```

The manifest records:

- Base model and checkpoint path.
- Adapter filename, byte size, and SHA-256.
- Locked selection-manifest path and SHA-256.
- Source and scored-dataset paths, row counts, and SHA-256 hashes.
- Vocabulary and rule-pack hashes.
- System-prompt hash.
- Sampling settings and batch size.
- Rollout artifact path, record count, and hash.
- Exact scoring policy.
- Git commit and whether tracked files were dirty.
- Pass-rate histogram.
- Counts always failing, always passing, and retained for GRPO.

Manifest construction does not trust caller-supplied summaries. It reopens the scored dataset and compressed rollout file, then verifies:

1. Source, scored, pass-rate, and rollout SKU sets agree.
2. Every product has exactly eight rollout records.
3. The number of passing records reproduces the saved pass rate.
4. Every saved rate matches the in-memory result.
5. All files named in the manifest exist before they are hashed.

---

## 9. End-to-end code flow

```mermaid
flowchart TD
    A[CLI: --smoke or --full] --> B[Parse and validate arguments]
    B --> C[Refuse existing output paths]
    C --> D[Read runs/sft-selection.json]
    D --> E[Verify base model, checkpoint path, adapter bytes and SHA]
    E --> F[Verify data/train_weak.jsonl SHA]
    F --> G[Load verifier pack]
    G --> H[Load Qwen base in vLLM]
    H --> I[Attach locked checkpoint-406 LoRA]
    I --> J[Render same SFT system and user prompts]
    J --> K[Generate 8 unconstrained completions per product]
    K --> L[Preserve raw text and token IDs]
    L --> M[verify raw completion]
    M --> N[Compare all known gold fields semantically]
    M --> N2[Score explicit unknown as a separate abstention decision]
    N --> O[Build one auditable rollout record]
    N2 --> O
    O --> P[Calculate pass rate from 8 booleans]
    P --> Q[Write deterministic rollouts.jsonl.gz]
    P --> R[Write derived scored JSONL]
    Q --> S[Reopen and cross-check artifacts]
    R --> S
    S --> T[Write manifest.json]
```

### Main code locations

| file | role |
|---|---|
| `training/score_difficulty.py` | CLI, lock verification, generation, grading, artifact writing and manifest checks |
| `tests/test_score_difficulty.py` | deterministic scoring, artifact, CLI, fake-vLLM and end-to-end workflow tests |
| `training/dataset.py` | shared SFT/GRPO prompt shape and `load_grpo_prompts()` filter |
| `training/predict.py` | shared `prompt_messages()` renderer used by SFT prediction and difficulty generation |
| `labeling/records.py` | row schema, three-state labels and canonical JSONL I/O |
| `verifier/` | strict JSON/schema/vocabulary/rule grading |
| `packs/vastraa_taste_v1/` | controlled vocabulary and declarative rules |
| `runs/sft-selection.json` | immutable checkpoint-selection evidence |

### Important functions

| function | responsibility |
|---|---|
| `score_completion()` | verifier-clean plus exact scorable-label grading |
| `rescore_rollout_records()` | deterministically re-grades saved raw outputs without model generation |
| `summarize_rollout_metrics()` | reports known-gold semantic quality and abstention quality separately |
| `calculate_pass_rate()` | requires exactly eight booleans |
| `build_rollout_record()` | combines raw generation evidence with its grade |
| `generate_rollouts()` | renders, samples, validates indices and immediately grades |
| `write_rollout_records()` | canonical deterministic gzip artifact |
| `write_scored_dataset()` | safe derived JSONL without source overwrite |
| `verify_locked_inputs()` | checks selection state, model, adapter and source hashes |
| `build_difficulty_manifest()` | reopens and cross-checks all completed artifacts |
| `run_difficulty()` | guarded smoke/full orchestration |

---

## 10. Generation configuration

These are the implemented defaults used by the completed real smoke run.

| parameter | value | reason |
|---|---:|---|
| backend | vLLM 0.23.0 | efficient batched generation with PEFT LoRA support |
| completions per product | 8 | plan requirement and pass-rate resolution |
| temperature | 0.7 | enough stochastic variation to reveal mixed outcomes |
| top-p | 0.95 | preserve diversity while trimming the weakest probability tail |
| max new tokens | 170 | above measured completion maximum; same SFT completion budget |
| maximum model length | 896 | covers measured prompt plus completion budget |
| batch size | 32 products | starting throughput setting; smoke has only two products |
| seed | 20260802 | reproducibility identity for this difficulty run |
| dtype | bfloat16 | stable Ampere-native inference precision |
| GPU-memory utilization | 0.65 | reserve headroom on the dual-use RTX 3090 |
| LoRA rank cap | 16 | matches the selected checkpoint |
| structured output constraint | disabled | format validity must remain observable rather than forced |

The backend is loaded lazily. CLI parsing, `--help`, collision checks, checkpoint verification, and source verification happen before CUDA initialization where possible.

### Why generation is unconstrained

If JSON-schema decoding forced every answer into a legal shape, schema-validity reward would become unobservable. This experiment deliberately grades literal model output. Malformed JSON is evidence, not something silently repaired away.

### vLLM API verification

The server’s installed API was inspected before implementation:

- vLLM version: `0.23.0`.
- `LLM.generate()` accepts a sequence of prompts, one `SamplingParams`, and one `LoRARequest`.
- `SamplingParams(n=8, ...)` returns indexed alternatives for each prompt.
- `LoRARequest` accepts the adapter name, positive process-local ID, adapter directory, and base-model name.

The environment emits a warning that vLLM’s Transformers-v4 path is deprecated for a future vLLM release. It is currently a warning, not a run failure, and no dependency upgrade was performed before this controlled experiment.

---

## 11. CLI safety contract

The command requires an explicit mode.

### Smoke mode

```bash
HF_HOME=/workspace/.hf_home PYTHONPATH=. /venv/rl/bin/python \
  -m training.score_difficulty --smoke
```

Expected scope:

- First 2 products.
- 8 completions each.
- 16 total completions.
- Output directory: `runs/sft-difficulty-k8-smoke/`.

Expected smoke artifacts:

```text
runs/sft-difficulty-k8-smoke/source-smoke.jsonl
runs/sft-difficulty-k8-smoke/scored-smoke.jsonl
runs/sft-difficulty-k8-smoke/rollouts.jsonl.gz
runs/sft-difficulty-k8-smoke/manifest.json
```

### Full mode

```bash
HF_HOME=/workspace/.hf_home PYTHONPATH=. /venv/rl/bin/python \
  -m training.score_difficulty --full
```

Expected scope:

- All 3,600 products.
- 28,800 total completions.
- Rollout/manifest directory: `runs/sft-difficulty-k8/`.
- Derived dataset: `data/train_weak_sft_scored.jsonl`.

### Guardrails

- Omitting both `--smoke` and `--full` is an error.
- Supplying both modes is an error.
- `--limit` is allowed only with smoke mode.
- Temperature must be greater than zero.
- `top_p` must be in `(0,1]`.
- Batch size and token limit must be positive.
- GPU-memory utilization must be in `(0,1)`.
- Existing output paths cause failure unless `--overwrite` is explicit.
- Source overwrite is refused even when constructing a derived dataset.

---

## 12. Implementation and test history

### Implementation sequence

1. Located the missing `training/score_difficulty.py` reference.
2. Identified the undefined meaning of “correct.”
3. Locked strict verifier-clean whole-record correctness.
4. Tested the definition against saved frozen-eval predictions.
5. Defined immutable source, rollout, scored-data and manifest artifacts.
6. Wrote red tests before the scorer existed.
7. Implemented deterministic grading and safe dataset writing.
8. Added canonical compressed rollout records and cross-checking manifest builder.
9. Added field-level incorrect/excluded evidence after audit review.
10. Inspected the exact installed vLLM API remotely.
11. Implemented an injectable generation loop and fake backend tests.
12. Implemented guarded CLI orchestration and checkpoint provenance checks.
13. Ran a complete fake smoke workflow locally.
14. Committed and synchronized the code to the GPU server.
15. Ran remote non-GPU tests and a read-only hardware/data preflight.
16. Added separate known-gold semantic and explicit-abstention metrics.
17. Re-scored the preserved smoke outputs without loading the model or sampling again.

### Test milestones

| milestone | focused tests | full suite |
|---|---:|---:|
| deterministic scorer and derived writer | 6 | 191 |
| artifact builders | 10 | 195 |
| safe CLI parser | 13 | 198 |
| injectable generation loop | 15 | 200 |
| guarded end-to-end fake smoke | 16 | 201 |
| native vLLM sampler fallback | 17 | 202 |
| semantic plus abstention metrics | 19 | 204 |

Current verification:

- Local full suite after the abstention-metric addition: **204 passed**.
- Remote focused difficulty suite under `/venv/rl`: **19 passed**.
- Python compilation check: passed.
- Diff whitespace check: passed.
- Remote CLI `--help` without GPU import: passed.

### Code provenance

| item | value |
|---|---|
| initial implementation commit | `0ad922dabd7d8b852eac5c2541c0420181017567` |
| native-sampler corrective commit | `6a88a50d0eec3c2fc38274d0c146f20fb26e5ec9` |
| dual-metric scoring commit | `69f1af14e3bbf0de6aa23c426e1190605f392802` |
| commit subjects | `Add auditable SFT difficulty scoring`; `Fall back to native vLLM sampler`; `Add abstention metrics to difficulty scoring` |
| remote branch | `master` |
| GPU worktree at successful smoke | `6a88a50d0eec3c2fc38274d0c146f20fb26e5ec9` |

The commit contained only:

```text
training/score_difficulty.py
tests/test_score_difficulty.py
```

Existing local `.gitignore`, HTML and attention-run artifacts were not included. Existing remote checkpoints, caches and run artifacts were preserved during a fast-forward pull.

---

## 13. Remote environment and preflight

Preflight date: 2026-08-02.

### Installed stack

| package | version |
|---|---|
| PyTorch | `2.11.0+cu130` |
| Transformers | `4.57.6` |
| TRL | `0.24.0` |
| PEFT | `0.19.1` |
| Unsloth | `2026.7.5` |
| vLLM | `0.23.0` |

Relevant imports previously verified successfully:

```text
GRPOConfig
GRPOTrainer
vllm.LLM
PeftModel
FastLanguageModel
```

### Hardware and storage immediately before the planned smoke

| item | measured value |
|---|---:|
| GPU | NVIDIA GeForce RTX 3090 |
| total VRAM | 24,576 MiB |
| used VRAM | 428 MiB |
| free VRAM | 23,699 MiB |
| GPU utilization | 0% |
| temperature | 38°C |
| idle application allocation | approximately 420 MiB |
| filesystem size | 27 GB |
| filesystem used | 23 GB / 83% |
| filesystem free | 4.8 GB |
| checkpoint-406 directory | 86 MB |
| cached base model | 2.9 GB |

The model is already cached at the server’s Hugging Face cache, so the smoke should not trigger a multi-gigabyte model download.

### Lock verification immediately before smoke

- Selection state: `locked_before_frozen_eval`.
- Adapter SHA matched: `00ae54af4e380cff66695b36b244e3f1ff9aca85076b59a8eb6649d8c3a051af`.
- Source SHA matched: `1cbcbfba5ad379e7c66895d720a997edf913030ee1e76e4917101dfccb09530b`.
- Adapter config present.
- Adapter config rank: 16.
- Adapter config alpha: 16.
- Adapter base: `unsloth/Qwen2.5-1.5B-Instruct`.
- All four planned smoke output paths were clear.

No vLLM engine or Qwen weights were loaded during preflight.

---

## 14. Real run tracker

Everything below must be updated from generated artifacts, not terminal memory.

### 14.1 Two-product smoke run

**Status:** COMPLETED AND ACCEPTED
**Command:** completed after one instrumentation-only failure and one vLLM dependency failure
**Expected completions:** 16; observed 16

| measurement | result |
|---|---|
| start UTC | exact shell timestamp was not persisted; first successful-run log at 20:03:41 UTC |
| end UTC | approximately 20:05:10 UTC |
| wall time | 96 seconds end to end |
| peak GPU memory | 16,793 MiB used, sampled every 500 ms |
| disk free before | 4.8 GB |
| disk free after | 4.7 GB; first-run vLLM compile cache is 93 MB |
| products | 2 |
| rollout records | 16; 16 unique `(sku_id, rollout_index)` pairs |
| schema-valid rollouts | 16/16, 100% |
| vocabulary-valid rollouts | 14/16, 87.5% |
| rule-clean rollouts | 16/16, 100% |
| passing rollouts | 13/16, 81.25% |
| pass-rate values | 0.625 and 1.000 |
| GRPO-eligible smoke products | 1/2 |
| completion-token range | 107–109 |
| output-at-token-cap count | 0/16 |
| manifest SHA-256 | `145861947267ea50676d0db6f56b3b006f8df4ae6a3146660dab8653f1b77dd7` |
| rollout artifact SHA-256 | `479b64e4b53c8cf549152bbaeab36fb15cfefadd24e25cf5722cd3ffbdd7e4e8` |
| scored dataset SHA-256 | `3fb3972263e5929a72af266e316d6bf5a6ef007e378cb0c0276442d34a949623` |
| smoke source SHA-256 | `7453cc6a674d0e091c88f2a24f768d3a017636c678aacc425d524b954d5957a9` |

Smoke acceptance checklist:

- [x] Process exits with code 0.
- [x] Exactly 16 unique `(sku_id, rollout_index)` records exist.
- [x] Each product has indices 0–7.
- [x] Raw output is preserved.
- [x] Scored source has exactly two rows.
- [x] Pass rates equal passing count divided by eight.
- [x] Manifest rebuild checks pass.
- [x] Adapter and source hashes in manifest match the lock.
- [x] No output unexpectedly hits 170 tokens.
- [x] No CUDA out-of-memory error.
- [x] GPU memory returns after process exit: 428 MiB idle.
- [x] Disk remains sufficient for the full run: 4.7 GB free.

Smoke product findings:

1. **Organic Cotton Pull On Pant** — 5/8 passed, pass rate `0.625`.
   All three failures differed only on `closure`. Two were vocabulary failures:
   the model emitted `"pull"` and `"pull_over"` instead of the allowed
   `"pullover"`. One emitted the valid abstention `"unknown"`, which remained an
   exact-match failure because closure was scorable gold. All eight outputs were
   schema-valid and rule-clean.
2. **Bode Puffer Jacket** — 8/8 passed, pass rate `1.0`. All eight outputs were
   identical. Only two gold fields were scorable; the model got both right and
   abstained on most remaining fields. This is a useful reminder that a strict
   whole-record pass can still be easy when weak gold contains many unknowns.

The first product demonstrates the intended mixed group: same prompt and policy,
five accepted answers and three rejected answers. The second demonstrates why
all-pass groups are filtered out.

#### Pre-full-run audit: how much gold is actually scorable?

The all-pass puffer prompted a corpus-wide audit using the exact scoring policy:
`labeled` and `not_applicable` fields count as scorable, while gold `unknown`
fields do not. Across all 3,600 training products:

| statistic | scorable fields out of 15 |
|---|---:|
| mean | 8.48 |
| minimum | 1 |
| p10 | 4 |
| p25 | 6 |
| median | 8 |
| p75 | 11 |
| p90 | 13 |
| maximum | 15 |

Low-scorable tail:

| threshold | products | share |
|---:|---:|---:|
| at most 2 | 73 | 2.0% |
| at most 3 | 188 | 5.2% |
| at most 4 | 361 | 10.0% |
| at most 5 | 599 | 16.6% |

No product has zero scorable fields. Eight products have only one, and 65 have
two. The Bode Puffer Jacket is in that 2.0% tail: one substantive labeled field,
one `not_applicable` field, and 13 gold unknowns.

Scorable count is not identical to information richness because
`not_applicable` is correctly scorable. Separating the three gold states gives:

| gold state per product | mean | median | p10–p90 | range |
|---|---:|---:|---:|---:|
| substantive `labeled` | 5.41 | 5 | 2–8 | 1–13 |
| `not_applicable` | 3.08 | 1 | 1–9 | 0–11 |
| excluded `unknown` | 6.52 | 7 | 2–11 | 0–14 |

There are 387 products (10.8%) with at most two substantive labeled fields. This
means pass rate can reflect both model competence and weak-label completeness.
The low-scorable tail is not large enough to invalidate the planned full
difficulty measurement, but the eventual GRPO pool must be stratified by both
total scorable and substantive-labeled counts. We will generate all 3,600 rows
first rather than introducing a post-smoke threshold, then inspect whether
all-pass rates are concentrated in sparse-gold products before locking the GRPO
selection rule.

#### Separate semantic and abstention metrics

The sparse-gold audit exposed an ambiguity in a single F1 score. Gold
`unknown` means the weak labels do not tell us the correct category; it is not a
normal category that should be mixed into semantic macro-F1. However, whether
the model explicitly abstains is still behavior worth measuring. We therefore
locked two complementary views:

1. **Known-gold semantic quality:** macro-F1 and exact cell accuracy only where
   gold is scorable.
2. **Abstention quality:** treat gold `unknown` as the positive class and report
   precision, recall and F1 for the model's explicit `unknown` decision.

An exploratory **unknown-aware pass** is also reported. It requires the normal
strict semantic pass *and* explicit `unknown` on every gold-unknown field. It
does not replace `sft_pass_rate` and is not used to select GRPO prompts.

Commit `69f1af1` added field-level `correct_abstention_labels`,
`missed_abstention_labels`, `false_abstention_labels`, and
`unknown_aware_passed` to each rollout record. Old v1 records remain readable,
which allowed the 16 preserved raw outputs to be re-graded without another GPU
generation run.

The v2 artifacts were written to a new directory so the accepted v1 smoke was
not overwritten:

```text
runs/sft-difficulty-k8-smoke-v2/source-smoke.jsonl
runs/sft-difficulty-k8-smoke-v2/scored-smoke.jsonl
runs/sft-difficulty-k8-smoke-v2/rollouts.jsonl.gz
runs/sft-difficulty-k8-smoke-v2/manifest.json
```

The v2 manifest records `mode: smoke-rescore-v2`, code commit `69f1af1`, the
original v1 manifest and rollout hashes, `generation_reused: true`, and
`raw_outputs_unchanged: true`. An independent comparison confirmed the same 16
record identities and exactly the same raw strings; their canonical raw-output
set SHA-256 is
`bca12f34dea57fa101aea4e60ec816038d229de0362ede432770537e68b8eea8`.

Measured smoke results:

| metric | result |
|---|---:|
| ordinary strict passes | 13/16, 81.25% |
| known-gold semantic macro-F1 | 0.9744 |
| known-gold exact cell accuracy | 85/88, 96.59% |
| abstention true positives | 146 |
| abstention false positives | 1 |
| abstention false negatives | 6 |
| abstention true negatives | 87 |
| abstention micro precision | 0.9932 |
| abstention micro recall | 0.9605 |
| abstention micro-F1 | 0.9766 |
| supported-attribute abstention macro-F1 | 0.9777 |
| exploratory unknown-aware passes | 10/16, 62.5% |

The ordinary pass count and both per-product pass rates remained unchanged:
`0.625` for Organic Cotton Pull On Pant and `1.000` for Bode Puffer Jacket.
This proves that the new reporting did not silently change the GRPO difficulty
label. The six missed abstentions were all in `details`; the single false
abstention was `closure`, corresponding to the pant rollout that predicted
`unknown` where the known gold was `pullover`.

V2 artifact hashes:

| artifact | SHA-256 |
|---|---|
| manifest | `6ab9da9013d97967447900ba8b29f08ee9b79e7e5155823bc8412a2b4d71207f` |
| enriched rollout records | `8feb7c93c889e2558a290b42d600d7cd3f09ff40041e20cfd4a7a69d574ac764` |
| scored smoke dataset | `3fb3972263e5929a72af266e316d6bf5a6ef007e378cb0c0276442d34a949623` |
| smoke source | `7453cc6a674d0e091c88f2a24f768d3a017636c678aacc425d524b954d5957a9` |

This is a two-product diagnostic, not a population estimate. The metric values
validate the definitions and artifact pipeline; the full 3,600-product run is
needed before drawing conclusions about model behavior.

### 14.2 Full 3,600-product difficulty run

**Status:** COMPLETED AND ACCEPTED
**Prerequisites:** smoke accepted; final non-generating preflight passed on 2026-08-02

Final preflight was deliberately read-only with respect to model execution: it
parsed the real `--full` configuration and called the same locked-input verifier
used by the runner, but did not initialize vLLM or CUDA.

| preflight check | measured result |
|---|---|
| remote code | `69f1af14e3bbf0de6aa23c426e1190605f392802` on `master` |
| tracked worktree | clean |
| source scope | 3,600 unique run rows loaded by the production reader |
| rollout scope | 8 per product; 28,800 expected total |
| selection state | `locked_before_frozen_eval` |
| selection-manifest SHA-256 | `e425635d323b3ffe9e7350fb61a2d9e1848345a95abab6b92032bf64d2718299` |
| adapter SHA-256 | verified `00ae54af4e380cff66695b36b244e3f1ff9aca85076b59a8eb6649d8c3a051af` |
| source SHA-256 | verified `1cbcbfba5ad379e7c66895d720a997edf913030ee1e76e4917101dfccb09530b` |
| full output directory | `runs/sft-difficulty-k8/`; absent |
| scored output | `data/train_weak_sft_scored.jsonl`; absent |
| model cache | 2.9 GB, 9 resolved snapshot files, no `.incomplete` files |
| disk | 4.7 GB free |
| GPU | RTX 3090; 23,699 MiB free; 0% utilization; 38°C |
| idle service allocation | 420 MiB, consistent with the known dual-use service |
| logging destination | `runs/` exists and is writable; no prior full-run log or monitor CSV |
| monitoring tools | `date`, `tee`, `nvidia-smi`, `df`, and `timeout` available |

The launch must preserve stdout/stderr and periodic GPU/disk samples in new log
files beside—not inside—the immutable artifact directory. This keeps a failed
launch log from creating an output-path collision on a clean retry. No full-run
generation occurred during this preflight.

#### Full-run GPU and disk sizing

The accepted two-product smoke peaked at **16,793 MiB of total GPU memory**,
including the approximately 420 MiB idle service allocation. The full run
should have a similar peak because vLLM processes products in bounded batches:
VRAM depends primarily on model size, batch size, context length and KV-cache
allocation, not on whether the dataset contains two or 3,600 products.

| GPU sizing level | VRAM |
|---|---:|
| measured smoke peak | 16,793 MiB |
| practical minimum from this measurement | approximately 17 GB |
| recommended capacity with operational headroom | at least 20 GB |
| available RTX 3090 capacity | 24,576 MiB |
| headroom above measured peak | 7,783 MiB |

A 16 GB card would not fit the measured configuration unchanged. The 24 GB RTX
3090 has comfortable headroom for the locked batch size, token ceiling and
`gpu_memory_utilization=0.65` configuration.

Disk requirements have two distinct parts. The large reusable inputs already
exist on the server:

| existing item | measured size |
|---|---:|
| cached Qwen2.5-1.5B-Instruct model | 2.9 GB |
| complete checkpoint-406 adapter directory | 86 MB |
| vLLM compilation cache created by the smoke | 94 MB |

The full run does not write another model checkpoint or optimizer state. It
only writes rollout evidence, the scored dataset, a manifest and operational
logs. The v2 smoke rollout records contain 17,707 uncompressed bytes for 16
rollouts. Linear scaling to 28,800 rollouts gives approximately 31.9 MB
uncompressed before gzip. The 3,600-row source dataset is 7.89 MB, so its scored
copy should remain around 8–10 MB. Compression, content diversity and log
duration make an exact prediction inappropriate, but the expected incremental
storage is below 75 MB.

| disk sizing level | free space required |
|---|---:|
| expected incremental output | less than approximately 75 MB |
| conservative minimum reserve | 200 MB |
| comfortable operational reserve | 500 MB |
| measured free space before launch | 4.7 GB |

The server therefore has ample disk capacity for this inference-only run. This
estimate must be checked against actual artifact and log sizes after completion.

#### Reviewed full-run launcher

The full run is wrapped by `scripts/run_difficulty_full.sh`. Constructing the
launcher separately from executing it makes the operational contract reviewable
and prevents a long GPU job from being started by an incidental command.

Safe remote verification, which exits before creating logs or initializing the
model:

```bash
bash scripts/run_difficulty_full.sh --preflight-only
```

Actual launch command, intentionally not executed during this step:

```bash
bash scripts/run_difficulty_full.sh
```

The launcher performs the following sequence:

1. Resolves the repository root independently of the caller's current directory.
2. Requires `/venv/rl/bin/python` and the existing Hugging Face cache.
3. Proves reviewed scorer commit `69f1af1` is an ancestor of `HEAD` and that the
   scorer and its tests are unchanged from that review point.
4. Refuses staged or unstaged tracked changes.
5. Refuses any existing full output, scored dataset, run log or monitor CSV.
6. Writes UTC start/end timestamps, actual Git commit, exact Python command,
   initial/final GPU state, disk state and scorer exit code to the run log.
7. Samples GPU memory, utilization, temperature and available disk once per
   second into a separate CSV.
8. Stops only its own monitor process through signal/exit traps.
9. Reports peak sampled VRAM, minimum sampled disk and final artifact hashes.
10. Preserves the scorer's exit code and never deletes partial evidence.

Operational files are kept outside the immutable artifact directory:

```text
runs/sft-difficulty-k8-full.log
runs/sft-difficulty-k8-monitor.csv
```

The scorer alone owns `runs/sft-difficulty-k8/` and
`data/train_weak_sft_scored.jsonl`. If generation fails before artifact writing,
the external logs remain available while the intended artifact paths stay
clear. If writing fails partway through, the launcher reports what exists and
stops; cleanup requires a separate audited decision rather than automatic
deletion.

Shell syntax and whitespace validation passed locally. ShellCheck was not
installed, so no ShellCheck result is claimed. A first review caught and fixed
an overly strict `HEAD == 69f1af1` check: committing the launcher necessarily
advances `HEAD`, so the final guard locks the scorer file contents to that commit
while allowing a later launcher/documentation commit.

The launcher was committed as `78e97d7` (`Add guarded full difficulty
launcher`), pushed to the Vast.ai bare remote, and fast-forwarded into
`/workspace/tagging-rl`. Its remote SHA-256 is
`abf9a655af08e6de93d759bf32e3d1558351d81dfffd780d43ef4e3c4b91fa53`.
The real remote invocation

```bash
bash scripts/run_difficulty_full.sh --preflight-only
```

exited successfully and printed the expected current and reviewed commits. GPU
state was exactly 428 MiB used, 23,699 MiB free and 0% utilization both before
and after. A separate post-check found no active `training.score_difficulty`
process and confirmed that the output directory, scored dataset, run log and
monitor CSV were all still absent. The tracked remote checkout remained clean
and disk remained at 4.7 GB free. This proves the launcher's preflight-only path
does not initialize the model, start generation or contaminate run evidence.

| measurement | result |
|---|---|
| status | completed successfully; scorer exit code 0 |
| start/end UTC | 2026-08-02 21:38:20–21:46:09 |
| wall time | 7 minutes 49 seconds |
| code commit | `78e97d7ca5df73069f12c06212fbb46525c4dd85`; clean tracked worktree |
| products | 3,600 |
| rollout records | 28,800; all unique `(sku_id, rollout_index)` pairs |
| passed rollouts | 12,609, 43.78% |
| failed rollouts | 16,191, 56.22% |
| always failed, rate 0 | 1,116 products, 31.00% |
| retained, rate 0.125–0.875 | 1,702 products, 47.28% |
| always passed, rate 1 | 782 products, 21.72% |
| schema-valid rate | 28,800/28,800, 100% |
| vocabulary-valid rate | 28,476/28,800, 98.88% |
| rule-clean rate | 28,385/28,800, 98.56% |
| completion-token range | 99–120 |
| token-cap count | 0/28,800 at 170 tokens |
| peak sampled GPU memory | 17,575 MiB used; 6,552 MiB remained free |
| peak GPU utilization / temperature | 100% / 74°C |
| monitor samples | 436 at one-second cadence |
| scored dataset bytes / SHA | 7,889,667 / `ec68b0ccf3ba84a82cdb0799956d36f53c9113d3cb7c7fd5232ecd50412a975f` |
| rollout gzip bytes / SHA | 509,163 / `f17360b157287caaea8d0f8e907f0a4bf4fd107977452442e2e447628e95bf8b` |
| manifest bytes / SHA | 11,042 / `5c6fcc41bab65b36904cef256c56e747a19302b30b6da3c7208382e4dfdd3e5b` |
| run log bytes / SHA | 25,642 / `273a8657db00edb3f6d0f34eefdc660a49d28a13665daf41c651a2a2c02ca110` |
| monitor CSV bytes / SHA | 21,683 / `dc88883e3bb52c47280f24c1d2464a68125730206f66184a8d13975fd91b1bbc` |
| disk free after | 4.7 GB; minimum sampled 5,005,783,040 bytes |

Pass-rate histogram:

| pass rate | number of products | percent |
|---:|---:|---:|
| 0.000 | 1,116 | 31.00% |
| 0.125 | 395 | 10.97% |
| 0.250 | 266 | 7.39% |
| 0.375 | 186 | 5.17% |
| 0.500 | 189 | 5.25% |
| 0.625 | 171 | 4.75% |
| 0.750 | 208 | 5.78% |
| 0.875 | 287 | 7.97% |
| 1.000 | 782 | 21.72% |

The mixed-outcome pool is large enough for GRPO: 1,702 prompts survive the
predeclared `0 < sft_pass_rate < 1` filter. This is 47.28% of the weak-training
set, so the strict whole-record reward did not collapse into mostly uniform
groups.

#### Full semantic and abstention measurements

| metric | full-run result |
|---|---:|
| known-gold semantic macro-F1 | 0.7768 |
| known-gold exact cell accuracy | 219,180/244,360, 89.70% |
| abstention true positives | 176,612 |
| abstention false positives | 16,349 |
| abstention false negatives | 11,028 |
| abstention true negatives | 228,011 |
| abstention micro precision | 0.9153 |
| abstention micro recall | 0.9412 |
| abstention micro-F1 | 0.9281 |
| supported-attribute abstention macro-F1 | 0.9187 |
| exploratory unknown-aware passes | 8,860/28,800, 30.76% |

These values describe sampled predictions on the weak training labels and must
not be compared directly with the locked 300-row frozen evaluation. Their role
here is to characterize reward difficulty and abstention behavior before GRPO.
The normal strict pass remains the selection signal; unknown-aware pass remains
exploratory.

The most frequent known-gold errors were `closure` (4,183), `details` (3,695),
`occasion` (3,028), `collar_type` (2,948), `colour_primary` (2,161), and
`pattern` (2,078). The weakest abstention attributes were `details` (F1 0.7856),
`occasion` (0.8303), `collar_type` (0.8358), and `closure` (0.8859). This shows
that the same semantically ambiguous fields create both incorrect commitments
and incorrect abstentions.

#### Does sparse gold make products look artificially easy?

Yes, especially at the extreme low-scorable tail, but that tail contributes few
GRPO prompts.

| scorable fields | products | mean pass rate | always failed | mixed/retained | always passed | retained share |
|---:|---:|---:|---:|---:|---:|---:|
| 1–2 | 73 | 0.971 | 1 | 6 | 66 | 8.2% |
| 3–5 | 526 | 0.533 | 128 | 215 | 183 | 40.9% |
| 6–8 | 1,248 | 0.394 | 419 | 599 | 230 | 48.0% |
| 9–11 | 998 | 0.369 | 367 | 466 | 165 | 46.7% |
| 12–15 | 755 | 0.484 | 201 | 416 | 138 | 55.1% |

Outcome groups also differ in average gold density:

| outcome group | products | mean scorable fields | mean substantive labeled fields |
|---|---:|---:|---:|
| always failed | 1,116 | 8.73 | 6.04 |
| mixed/retained | 1,702 | 8.81 | 5.61 |
| always passed | 782 | 7.42 | 4.05 |

The Pearson correlation between scorable-field count and pass rate is only
`-0.099`, so scorable count alone does not explain difficulty across the full
dataset. However, 66 of the 73 rows with only one or two scorable fields are
always-pass, confirming the smoke-run concern at that extreme. Only six enter
the retained pool. The retained dataset must still carry scorable and
substantive-label counts in its audit manifest so this bias remains visible.

#### Independent artifact validation

The saved gzip was reopened independently after the runner exited. Validation
confirmed 28,800 nonempty raw outputs, 28,800 unique record keys, indices 0–7
for every SKU, 3,600 unique scored rows, exact agreement between raw pass counts
and saved `sft_pass_rate`, and exact agreement with the manifest histogram. The
unknown-aware count also independently reproduced 8,860. No scoring process
remained active, and GPU memory returned to the 428 MiB idle baseline.

The five full-run evidence files were then copied from Vast.ai to the same
relative paths in the local repository before any downstream GRPO work:

```text
runs/sft-difficulty-k8/manifest.json
runs/sft-difficulty-k8/rollouts.jsonl.gz
data/train_weak_sft_scored.jsonl
runs/sft-difficulty-k8-full.log
runs/sft-difficulty-k8-monitor.csv
```

Fresh local and remote SHA-256 calculations matched for every file, as did all
byte sizes. The local project environment reopened the copied gzip and scored
JSONL and independently reproduced 28,800 unique rollout keys, 3,600 complete
eight-rollout groups, all saved pass rates, the complete histogram, 436 monitor
samples, and the run log's exit-code-zero marker. The copied evidence is
included in the same scoped Git commit as this tracker, so the technical claims
and their backing artifacts travel together.

#### Retained-pool composition and family audit

Before treating the 1,702 mixed-outcome rows as a GRPO dataset, we audited
whether difficulty filtering accidentally collapsed category/store coverage or
overweighted near-duplicate products. The audit reuses the exact SFT family key
from `training.split_sft.group_key()`: normalized brand plus normalized title
before a conservative spaced variant separator. Store is parsed from the domain
inside `sku_id`; the generic row-level source is `shopify` for all 3,600 rows and
therefore cannot measure store diversity.

The reproducible builder is `training/audit_grpo_pool.py`, covered by two
dataset-locked regression tests. The full project suite passed with **207
tests**. It produced:

```text
runs/sft-difficulty-k8/retained-pool-audit.json
```

| artifact property | value |
|---|---|
| report version | `grpo-retained-pool-audit-v1` |
| bytes | 42,290 |
| SHA-256 | `3cce5b94adcc140ed6ff08f58243a6e90b201413cd3b284cabadf920fe12ef7e` |
| scored-data SHA verified | `ec68b0ccf3ba84a82cdb0799956d36f53c9113d3cb7c7fd5232ecd50412a975f` |
| difficulty-manifest SHA verified | `5c6fcc41bab65b36904cef256c56e747a19302b30b6da3c7208382e4dfdd3e5b` |
| retained SKU-set SHA-256 | `e8e318a46b8c11e9898e9bfdd6f8df2c821129002ac12279eae6dda7e8aab3e7` |

Rebuilding the report with its saved timestamp reproduced the complete JSON
structure exactly.

Category coverage is broadly preserved. The total-variation distance between
the full and retained category distributions is 5.77%. The largest supported
increases are shoes, from 14.94% to 17.27% (`+2.33` percentage points), and tops,
from 16.58% to 18.51% (`+1.92`). The largest decreases are dresses, from 6.89%
to 5.41% (`-1.48`), and jackets, from 8.03% to 6.64% (`-1.39`). Jumpsuits are the
only absent category, but the full pool contains only two jumpsuit rows. This is
a rare-support limitation, not evidence of broad category collapse.

All 14 stores remain represented. Store-distribution total variation is 8.47%,
larger than category shift and worth tracking during training:

| store | full rows | retained rows | retention rate | retained-share shift |
|---|---:|---:|---:|---:|
| Thursday Boots | 490 | 300 | 61.2% | +4.02 pp |
| Everlane | 382 | 212 | 55.5% | +1.84 pp |
| American Giant | 384 | 196 | 51.0% | +0.85 pp |
| Faherty | 419 | 192 | 45.8% | -0.36 pp |
| Taylor Stitch | 332 | 166 | 50.0% | +0.53 pp |
| Naadam | 274 | 148 | 54.0% | +1.08 pp |
| UNTUCKit | 236 | 114 | 48.3% | +0.14 pp |
| tentree | 212 | 90 | 42.5% | -0.60 pp |
| Marine Layer | 306 | 79 | 25.8% | -3.86 pp |
| Allbirds | 215 | 73 | 34.0% | -1.68 pp |
| Rothy's | 172 | 71 | 41.3% | -0.61 pp |
| Outdoor Voices | 102 | 41 | 40.2% | -0.42 pp |
| Ministry of Supply | 51 | 11 | 21.6% | -0.77 pp |
| Girlfriend Collective | 25 | 9 | 36.0% | -0.17 pp |

Seven nominal brand strings have no retained row, but together they account for
only 19 of 3,600 source rows. Brand-distribution total variation is 9.47%; most
of that reflects the same store and catalog-family effects rather than loss of a
large brand.

Family concentration is the main actionable finding:

| family measurement | result |
|---|---:|
| full families | 2,255 |
| full multi-product families | 491 |
| retained families | 1,150 |
| retained rows | 1,702 |
| duplicate rows beyond one per retained family | 552, 32.43% |
| retained rows originating in multi-product families | 921, 54.11% |
| represented multi-product families | 369 |
| fully retained multi-product families | 114 |
| partially retained multi-product families | 255 |
| largest retained family | 40 rows, 2.35% of the pool |
| top five families | 82 rows, 4.82% |
| top ten families | 125 rows, 7.34% |

The largest family is Thursday Boots Women's Bags’ **Everyday Tote**: 57 source
variants, of which 40 have mixed outcomes. Its source-family pass-rate histogram
spans one always-fail row, 40 mixed rows and 16 always-pass rows. That variation
shows the filter is not merely selecting the entire family, but row-uniform GRPO
sampling would still give this one product concept forty times the weight of a
singleton family.

Gold completeness improves slightly after filtering rather than degrading:

| gold-density measure | full 3,600 | retained 1,702 |
|---|---:|---:|
| mean scorable fields | 8.48 | 8.81 |
| mean substantive labeled fields | 5.41 | 5.61 |
| mean unknown fields | 6.52 | 6.19 |
| rows with at most two scorable fields | 73 | 6 |
| rows with at most two substantive fields | 387 | 97 |

**Audit conclusion:** the strict difficulty filter leaves enough category and
store diversity for GRPO, and sparse-gold rows do not dominate. The pool is not
distribution-neutral: Thursday Boots is overrepresented, Marine Layer is
underrepresented, and repeated product families create meaningful row weights.
We will not change the locked `0 < pass_rate < 1` eligibility rule; training
weight is handled separately by the sampling policy below.

##### What these findings mean

The retained pool is **usable for GRPO, but not perfectly balanced**.

1. **Category diversity survived filtering.** Shoes, tops, sweaters, dresses,
   bags and the other major garment categories remain represented. The modest
   5.77% category-distribution shift means difficulty filtering did not collapse
   the training pool into one dominant garment type.
2. **Store coverage survived, but store weights changed.** All 14 stores remain,
   yet Thursday Boots grows from 13.6% of the source to 17.6% of retained rows,
   while Marine Layer falls from 8.5% to 4.6%. The filter therefore creates a
   curriculum concentrated on catalogs where the locked SFT model produced more
   mixed outcomes.
3. **Sparse weak gold is not driving the retained pool.** Retained rows have
   slightly more scorable and substantive fields than the source average, and
   only six retained rows have at most two scorable fields. Most GRPO rewards
   will therefore be based on meaningful multi-field comparisons rather than
   trivially easy one- or two-field records.
4. **Near-duplicate weighting is the principal sampling risk.** A row-uniform
   loader would treat the 40 retained Everyday Tote variants as 40 separate
   votes while a singleton family gets one. This could spend disproportionate
   optimization effort on one product concept even though the formal membership
   rule is correct.
5. **The mixed pool is an intentional curriculum.** These are prompts near the
   current policy's decision boundary: neither always solved nor always failed
   under eight sampled attempts. That reward variance is useful for GRPO, but it
   also means the retained distribution is no longer the original catalog
   distribution.

The supported operational decision is to preserve all 1,702 rows as the locked
**eligible** pool while separately choosing a family-aware sampling policy. Pool
membership and training weight are different decisions: retaining a row keeps
the evidence complete; sampling controls how often it influences optimization.

##### Sampling-policy comparison and decision

We compared five policies without training. Deterministic caps order retained
rows inside each canonical family by SHA-256 of `42\0<sku_id>` and keep the first
`N`. This makes every proposed active set reproducible rather than dependent on
input order.

“Effective families” is the inverse Herfindahl index of family sampling
probability. It answers: *how many equally weighted families would create the
same concentration?* Higher is more balanced; the real maximum is 1,150.

| policy | active rows | largest family weight | top-10 family weight | effective families | category TVD vs full | store TVD vs full | difficulty TVD vs eligible |
|---|---:|---:|---:|---:|---:|---:|---:|
| row-uniform | 1,702, 100% | 2.350% | 7.344% | 513 | 5.77% | 8.47% | 0.00% |
| family cap 2 | 1,390, 81.7% | 0.144% | 1.439% | 1,033 | 4.99% | 9.15% | 2.12% |
| **family cap 4** | **1,565, 92.0%** | **0.256%** | **2.556%** | **855** | **5.53%** | **7.85%** | **1.20%** |
| family cap 8 | 1,657, 97.4% | 0.483% | 4.828% | 710 | 6.27% | 7.56% | 0.90% |
| family-uniform | 1,702, 100% | 0.087% | 0.870% | 1,150 | 7.62% | 16.24% | 2.04% |

Row-uniform is simplest and preserves every eligible row, but it has the worst
practical family concentration: only 513 effective families and 2.35% of all
updates assigned to Everyday Tote. Family-uniform removes that concentration
entirely, but it gives every family equal weight regardless of catalog/store
structure. In this dataset that nearly doubles store shift to 16.24%, requires a
custom weighted sampler, and makes “one epoch” less intuitive because rows no
longer have equal inclusion probability.

Cap two balances families strongly but removes 312 eligible rows and slightly
worsens store balance. Cap eight keeps nearly everything but leaves materially
more duplicate weighting. **Cap four is the selected first-run policy** because
it keeps all 1,150 families and 1,565 of 1,702 eligible rows, raises effective
family diversity from 513 to 855, cuts the largest family's weight by about 9×,
slightly improves category and store balance, and shifts the difficulty
histogram by only 1.20%. Its expected pass rate is 0.4624 versus 0.4666 under
row-uniform sampling.

The deterministic cap-four active SKU-set SHA-256 is
`77d926c88447e3cda5852f015629a4eae8bb9c7e32a00a67694abd985fb75c76`.
The 137 uncapped variants remain in the locked eligible evidence; they are not
deleted or reclassified, only excluded from the first run's active sampling
set. A later ablation can compare row-uniform or cap eight without rerunning
difficulty scoring.

##### What these findings do not prove

- They do not prove that Thursday Boots products are inherently harder or that
  Marine Layer products are inherently easier. The observed rates combine the
  locked model, weak-label quality, catalog wording, product mix and one decoding
  configuration.
- They do not prove that the retained pool matches real production traffic.
  Every source row is from Shopify catalogs, and no traffic weighting exists.
- They do not measure generalization. These are weak-training prompts already
  seen during SFT; the locked 300-row frozen evaluation remains the comparison
  set for model quality.
- They do not prove cap four is universally optimal. It is the best measured
  engineering compromise for this first run; only a controlled training
  ablation could establish whether another weighting improves frozen evaluation.
- They do not make exact pass rate an intrinsic property of a product. With
  eight rollouts, rates move in 0.125 increments and may shift under another
  random seed or decoding configuration.

Remaining post-run breakdowns:

- Retained composition and retention rates by garment category are complete;
  the full nine-bin histogram within each category remains optional follow-up.
- Retained composition by product family/store is complete; the sampling
  comparison selected deterministic family cap four for the first run.
- ~~Incorrect-label frequency across all failed rollouts.~~ Completed.
- Schema and vocabulary failures by field.
- Rule-violation histogram.
- ~~Completion-token distribution and cap hits.~~ Range and cap audit completed.
- Relationship between prompt length and pass rate.
- A few products from rates 0, 0.5 and 1 with all eight answers compared.
- Sensitivity check on a small subset using a second seed before treating the ranking as stable.

---

## 15. GRPO handoff tracker

The dataset handoff is complete and independently checked. Reward design and
training remain future work.

### Dataset handoff

The handoff deliberately separates three sets:

1. The 3,600-row scored source remains immutable.
2. The 1,702-row eligible set remains fully recorded as difficulty evidence.
3. The first GRPO run receives the deterministic 1,565-row cap-four active set.

`training/build_grpo_pool.py` constructs the active set with the same selector
used by the sampling-policy audit. It writes two new artifacts without modifying
the scored source or audit report:

```text
data/train_weak_grpo_cap4.jsonl
runs/sft-difficulty-k8/grpo-pool-cap4-manifest.json
```

| handoff artifact | bytes | SHA-256 |
|---|---:|---|
| cap-four active JSONL | 3,467,347 | `3e378187a8147923bae1e0753a750d6e252336e911fa8c91cd57a4a8ddc3a102` |
| selection manifest | 88,568 | `d166325a0c4ef3d78023ba492881fb3971e290b1b3606ee4ac8cd6aa733175e0` |

Selection identities:

| set | rows | SKU-set SHA-256 |
|---|---:|---|
| eligible mixed-outcome rows | 1,702 | `e8e318a46b8c11e9898e9bfdd6f8df2c821129002ac12279eae6dda7e8aab3e7` |
| active cap-four rows | 1,565 | `77d926c88447e3cda5852f015629a4eae8bb9c7e32a00a67694abd985fb75c76` |
| capped but still eligible rows | 137 | `23b0e276e463cb052da754bf2199ce872d31fcd42b3737b37243a30bfcb6047a` |

The manifest embeds all 1,565 active SKU IDs and all 137 capped SKU IDs in
source order. These lists are disjoint and their union exactly reproduces the
eligible SKU set. The output JSONL also preserves source order and every active
row is byte-for-structure identical to its row in
`data/train_weak_sft_scored.jsonl`; only row membership changes.

Active composition:

| measurement | result |
|---|---:|
| represented families | all 1,150 eligible families |
| maximum rows per family | 4 |
| families with 1 / 2 / 3 / 4 active rows | 910 / 125 / 55 / 60 |
| mean scorable fields | 8.70 |
| mean substantive labeled fields | 5.58 |
| rows with at most two scorable fields | 6 |
| pass-rate 0.125 / 0.250 / 0.375 | 379 / 240 / 168 |
| pass-rate 0.500 / 0.625 / 0.750 / 0.875 | 176 / 150 / 192 / 260 |

All 14 stores remain in the active dataset. The selection manifest records the
full category, store, source, family-size, pass-rate and gold-density counts so
downstream training can prove it consumed the intended pool.

The builder records the exact implementation-file SHA-256
`8c373bfe2d58bbf2b2ed3f82cb3445cc1a1feca26361e38d0567c1d182821ba0`.
It also records parent commit `6120ad6` and `tracked_worktree_dirty: true`
because the handoff code and this tracker were intentionally generated and
reviewed before their own commit. The implementation hash, tests, output hashes
and eventual scoped commit together provide the exact provenance; the dirty flag
is retained rather than rewritten after the fact.

Independent validation proved:

- all active rows satisfy `0 < sft_pass_rate < 1`;
- active and capped lists are disjoint and cover all eligible rows;
- every eligible family remains represented;
- no active family exceeds four rows;
- output membership, order and row contents match the manifest;
- the active SKU-set hash matches the prior policy audit;
- `load_grpo_prompts(pack, path=active_path,
  require_pass_rate_band=True)` returns exactly 1,565 examples;
- each loaded example contains system/user prompts, hidden gold JSON and SKU,
  but no SFT completion.

The builder refuses existing output paths unless `--overwrite` is explicit.
Six focused handoff/audit tests and the full **210-test** project suite pass.
No GRPO model, reward function or GPU training process was started in this step.

`load_grpo_prompts()` now has a proven handoff contract. It:

- reads the derived scored dataset;
- retains only rows where `0 < sft_pass_rate < 1` when required;
- returns prompt-only chat messages;
- carries gold JSON as a hidden dataset column for reward functions;
- carries SKU for auditing;
- never includes the SFT completion as the model’s answer.

### Questions to answer before the first GRPO run

- [x] What fraction survives? 1,702/3,600 eligible; 1,565 active after cap four.
- [x] Is the pool diverse? All major categories, all 14 stores and all 1,150
  eligible families are represented; measured skews are documented.
- [x] Are there enough mixed groups? Yes: 1,565 active prompts, each selected
  from the predeclared mixed-outcome band.
- Should the first GRPO reward remain binary or use multiple reward components?
- How will gold-unknown fields be handled consistently in the reward function?
- What checkpoint frequency fits the remaining disk budget?
- What is the smallest GRPO smoke that proves reward variance, nonzero gradients and stable memory?
- Which frozen metrics determine whether GRPO beats or harms SFT?

### Required GRPO comparison

The eventual comparison must use the same locked 300-product frozen evaluation and report at least:

| metric | SFT locked baseline | GRPO | delta |
|---|---:|---:|---:|
| macro-F1 | 0.6411 | pending | pending |
| selective macro-F1 | 0.7170 | pending | pending |
| coverage | 94.3% | pending | pending |
| schema validity | 100% | pending | pending |
| vocabulary validity | 88.7% | pending | pending |
| rule violations | 12 | pending | pending |
| missing predictions | 0 | pending | pending |

GRPO succeeds only if it improves the predeclared metrics, or if the experiment produces a technically defensible explanation for why it did not.

---

## 16. Risks, limitations and claims to avoid

### Strict pass definition

Whole-record correctness may classify a nearly perfect 14-of-15 answer as a failure. That is intentional for initial curriculum selection but should not be confused with the eventual shaped reward.

### Weak-label dependence

Difficulty is agreement with weak gold under a verifier, not direct human truth. Products with bad weak labels may appear artificially hard or teach the model the wrong target.

### Training-data difficulty is not evaluation

The SFT model has already trained on most of this pool. Pass rate measures residual uncertainty on training prompts; it is not a generalization metric. The frozen 300 remains the only locked model-comparison set.

### Eight samples are noisy

With `k=8`, rates move in 0.125 increments. A product measured at 0.875 under one seed might measure at 1.0 under another. Avoid presenting the exact rate as an intrinsic property.

### Selection changes the training distribution

Removing all-easy and all-hard rows creates a curriculum concentrated near the current policy boundary. This may improve learning efficiency while reducing coverage of rare categories. The retained distribution must be audited.

### Binary filtering versus continuous reward

The filter uses binary pass/fail. GRPO can still learn from partial or continuous rewards even when no completion is fully correct. If too few products survive, reward variance—not the arbitrary pass band—should drive redesign.

### Reproducibility is conditional

The manifest captures seed, prompt, hashes, backend settings and code commit. Exact token sampling may still depend on GPU kernels, library versions and scheduling. Reproducibility should mean the full configuration and observed distribution are recoverable, not that every bit is guaranteed across hardware stacks.

### Disk pressure

Only 4.8 GB was free at preflight. The compressed rollout evidence should be small relative to model checkpoints, but GRPO checkpoint retention must remain bounded. Never save unlimited optimizer checkpoints on this machine.

### Claims to avoid

- Do not call pass rate “model confidence.” It is sampled success frequency under one decoding configuration.
- Do not claim that `0.0` examples are impossible.
- Do not claim that `1.0` examples are universally solved.
- Do not call the weak labels human ground truth.
- Do not claim a GRPO gain until the locked frozen evaluation is complete.
- Do not compare a conditional metric on parsed survivors with an unconditional metric without the validity caveat.

---

## 17. Debugging and failure log

Keep failures here, including failed experiments. Do not erase them when fixed.

### Local Python mismatch during an exploratory calculation

The system `python3` was Python 3.7 and could not import `typing.Literal` as expected by the project. Project tests and calculations use `uv run` with the declared Python 3.12 environment. No project code was changed to accommodate the obsolete system interpreter.

### First unit-test fixture mistake

The initial “valid but wrong” test changed `colour_primary` on a row where the gold color was `unknown`. The scorer correctly ignored that field, so the rollout still passed. The test was fixed to change a known `material` label instead. This was useful evidence that gold-unknown exclusion worked.

### vLLM warning

Remote import prints a deprecation warning for the Transformers-v4 code path. Required imports and signatures still work. Dependencies were not upgraded immediately before the controlled run.

### Smoke launch attempt 1: missing timing utility

The first wrapper used `/usr/bin/time -v`, but that binary is absent from the
container. The shell exited with code 127 before Python started. GPU use, disk and
output paths were unchanged. The retry used shell epoch timestamps instead.

### Smoke launch attempt 2: missing optional FlashInfer package

The second launch reached vLLM engine initialization but failed before loading
weights or generating outputs. vLLM 0.23 attempted to import `flashinfer`, which
is not installed in `/venv/rl`, and raised `ModuleNotFoundError`. The process
exited 1 after 31 seconds; GPU memory returned to 428 MiB and no artifacts were
written.

Inspection of the installed vLLM source identified its supported fallback:
`VLLM_USE_FLASHINFER_SAMPLER=0`. Commit `6a88a50` now sets that variable before
importing vLLM only when `flashinfer` is absent, uses vLLM's native top-p sampler,
and records `sampler_backend: native` in the manifest. No package was installed,
and the sampling parameters remained temperature 0.7 and top-p 0.95.

### Successful smoke operational notes

- Engine initialization took 56.86 seconds, including 23.67 seconds of
  `torch.compile` and 19 seconds of CUDA graph capture.
- Model loading used 3.01 GiB and took about 1.9 seconds.
- vLLM allocated 10.89 GiB of KV cache and reported capacity far beyond this
  smoke workload.
- First-run compile artifacts occupied 93 MB under `/root/.cache/vllm`.
- Generation finished without OOM, and the engine shut down cleanly.
- The manifest records code commit `6a88a50`, a clean tracked worktree, locked
  adapter/source hashes, native sampler, unconstrained decoding and all sampling
  settings.

### Future smoke/full failures

Add timestamp, exact command, last progress line, traceback, GPU state, disk
state, artifacts left behind, diagnosis, and corrective commit for every future
failure.

---

## 18. Suggested final article structure

1. Start with the apparent temptation: “SFT is done, so run GRPO.”
2. Explain why group-relative learning needs reward differences within a prompt.
3. Introduce the locked SFT catalog tagger and its structured-output verifier.
4. Show the missing definition of “correct” and the alternatives considered.
5. Define strict pass/fail, including unknown versus null.
6. Show the eight-rollout pass-rate calculation with one real product.
7. Explain the three auditable artifacts and why raw outputs are preserved.
8. Walk through the vLLM + LoRA generation flow.
9. Present smoke findings and operational corrections.
10. Present the full pass-rate distribution and category/error analysis.
11. Show how the mixed-outcome band becomes the GRPO prompt pool.
12. Present GRPO versus locked SFT on the same frozen evaluation.
13. Close with what failed, what was noisy, and what should be changed next.

The strongest narrative is not “we used GRPO.” It is “we made the reward signal measurable, auditable and comparable before spending the training budget.”

---

## 19. Evidence checklist for publication

- [x] Locked SFT checkpoint and adapter checksum.
- [x] Locked source-data checksum.
- [x] Exact pass definition.
- [x] Feasibility check of strict correctness.
- [x] Artifact schemas.
- [x] Deterministic rollout compression.
- [x] Manifest cross-check logic.
- [x] Safe smoke/full CLI.
- [x] Unit and fake-integration tests.
- [x] Remote environment and preflight.
- [x] Real smoke artifacts and measurements.
- [x] Smoke output examples with failure explanations.
- [x] Separate known-gold semantic and abstention definitions.
- [x] Deterministic v2 re-score with raw-output lineage proof.
- [x] Full 28,800-rollout artifact with independent integrity checks.
- [x] Full pass-rate histogram.
- [x] Full semantic, abstention and unknown-aware measurements.
- [x] Scorable-density bias audit of the retained pool.
- [x] Retained-pool category/store/family and gold-density audit.
- [x] Row-uniform, family-cap and family-uniform sampling comparison.
- [x] Deterministic cap-four active dataset and complete SKU selection manifest.
- [ ] Second-seed sensitivity sample.
- [ ] GRPO reward specification.
- [ ] GRPO smoke and gradient evidence.
- [ ] GRPO training curve and resource use.
- [ ] Locked frozen evaluation after GRPO.
- [ ] SFT-versus-GRPO uncertainty estimate.
- [ ] Final limitations and reproducibility package.

---

## 20. Tracker update protocol

After every real run:

1. Record the exact command and commit before interpretation.
2. Link the manifest and quote its SHA-256.
3. Copy counts from artifacts, not scrolling terminal output.
4. Record wall time, peak VRAM and disk before/after if measured.
5. Add at least one successful and one failed raw rollout example.
6. Explain unexpected distributions before changing thresholds or rewards.
7. If configuration changes, create a new output directory and manifest; do not overwrite evidence from a completed run.
8. Label exploratory findings separately from locked comparisons.
9. Keep failed runs and corrective commits in the debugging log.
10. Update the opening thesis only after the final frozen evaluation supports it.

This file is the working evidence ledger. The polished blog should be shorter, but every important claim in it should be traceable back to a row, manifest, checksum, command, test, or frozen metric recorded here.
