# From SFT to RL: technical blog tracker for difficulty scoring and GRPO

**Document type:** living technical tracker and future blog brief
**Started:** 2026-08-02
**Current scope:** locked SFT checkpoint → sampled difficulty measurement → GRPO prompt selection → first reward implementation → minimal smoke preflight → locked-model load gate → no-training trainer-construction gate → one-group rollout/reward gate → one-group gradient-only gate → optimizer-construction-only gate → one-real-update gate
**Current status:** the deterministic pool, reward contract, five-row smoke fixture and staged integration gates now reach one real trainer update; the first step used the intended nonzero learning rate, initialized complete 8-bit AdamW state and changed every LoRA tensor finitely while leaving the locked source checkpoint untouched, but multi-step stability and saved-checkpoint quality remain intentionally untested
**Update rule:** record measured results only after their artifact and checksum exist; keep planned settings clearly labeled as planned

---

## 1. Proposed post in one sentence

Before applying GRPO to a structured catalog tagger, we sampled eight answers from a locked SFT model for every training prompt and kept the prompts where the model sometimes succeeded and sometimes failed, creating the within-group reward variation that policy-gradient learning needs.

The difficulty sampling and pre-update GRPO gates have now happened. This
sentence remains provisional only because optimizer updates and the frozen
SFT-versus-GRPO evaluation have not happened yet.

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

The handoff package was committed as `009c9df` and fast-forwarded into
`/workspace/tagging-rl` on Vast.ai. Fresh remote SHA-256 checks reproduced the
active dataset, selection manifest and audit-report hashes above. The six
focused audit/handoff tests passed under `/venv/rl` in 5.71 seconds, the tracked
remote worktree remained clean, and GPU state stayed at 428 MiB used, 23,699 MiB
free and 0% utilization. This validates the same handoff on the eventual
training environment without loading the model.

`load_grpo_prompts()` now has a proven handoff contract. It:

- reads the derived scored dataset;
- retains only rows where `0 < sft_pass_rate < 1` when required;
- returns prompt-only chat messages;
- carries gold JSON as a hidden dataset column for reward functions;
- carries SKU for auditing;
- never includes the SFT completion as the model’s answer.

### First-run reward-contract audit

Before implementing a trainer, we mapped the existing verifier and difficulty
scorer onto the exact reward callback used by the installed training stack. The
plain reward functions were then implemented and tested without importing TRL,
loading a model or using the GPU.

The proposed first unconstrained run uses three binary reward components:

1. **Format validity:** `1` when `verify(raw_output, pack).schema_valid` is true,
   otherwise `0`. This requires literal JSON with the schema's allowed keys and
   value shapes; the verifier does not repair prose, markdown fences or malformed
   JSON.
2. **Vocabulary/rule compliance:** `1` when
   `verify(raw_output, pack).ok` is true, otherwise `0`. `ok` requires schema
   validity, controlled-vocabulary validity and zero rule violations.
3. **Golden agreement:** `1` when
   `score_completion(raw_output, gold, pack).passed` is true, otherwise `0`.
   This requires verifier-clean output and exact agreement on every scorable
   known-gold field. Gold fields labeled `unknown` remain excluded because they
   do not provide a trustworthy answer key.

TRL will combine the raw components with `reward_weights=[1.0, 1.0, 2.0]`.
Keeping each callback binary makes its individual mean and standard deviation
easy to interpret in TRL/W&B logs; the separate weights make exact agreement
worth twice either validity component. The resulting ladder, confirmed with
real rows from the cap-four dataset, is:

| example completion | format | compliance | agreement ×2 | total |
|---|---:|---:|---:|---:|
| invalid prose | 0 | 0 | 0 | 0 |
| schema-valid JSON containing an out-of-vocabulary value | 1 | 0 | 0 | 1 |
| empty JSON object | 1 | 1 | 0 | 2 |
| verifier-clean but gold-wrong record | 1 | 1 | 0 | 2 |
| verifier-clean exact answer on all scorable known-gold fields | 1 | 1 | 2 | 4 |

#### Why the initial weights are `1:1:2`

These weights are a transparent starting hypothesis, not values derived by an
optimizer or proven optimal in an ablation. Format and compliance each receive
one point because they are necessary intermediate behaviors: first emit the
required structure, then stay inside the vocabulary and rules. Golden agreement
receives two points because semantic correctness is the real task objective; an
exact answer should gain as much semantic credit as format and compliance
combined.

Equal `1:1:1` weights were rejected for this first run because a fully correct
answer would total `3`, only one point above empty or verifier-clean but wrong
JSON at `2`. That gives the final semantic step the same value as either
prerequisite even though correctness is what the model will ultimately be
evaluated on. Conversely, an overwhelmingly large agreement weight was also
rejected. Whole-record exact agreement is binary and sparse; when a group has no
exact completion, format and compliance should still distinguish some outputs
and provide an intermediate learning signal.

TRL computes the weighted total and then turns reward differences within each
prompt's completion group into advantages. Because GRPO normalizes within the
group, multiplying every weight by the same constant would have little effect;
the important choice is the relative spacing. `1:1:2` produces the intended
ladder `0 → 1 → 2 → 4` without allowing semantic correctness to erase
the earlier signals. We will keep this ratio fixed for the first auditable run
and change it only in a separately tracked experiment after rollout evidence
reveals a concrete failure mode.

This table exposes an intentional weakness in the first contract: under the
current optional-field schema, `{}` is schema-valid, vocabulary-valid and
rule-clean, so it earns the same two points as a complete but semantically wrong
record. We are not hiding or prematurely fixing that loophole. W2's first GRPO
run is deliberately unconstrained so that empty-but-valid JSON, safe-tag
conservatism and other verifier gaming can be observed and preserved as
evidence. The later shaped run can make validity a gate and add per-attribute or
class-balanced credit after the actual failure modes are measured.

The installed TRL 0.24.0 callback flow is:

```text
cap-four dataset row
  └─ prompt + hidden gold JSON + SKU
       └─ TRL generates G completions and repeats hidden columns G times
            ├─ format reward(completions, ...)
            ├─ compliance reward(completions, ...)
            └─ agreement reward(completions, gold, ...)
                 └─ weighted sum → within-prompt group advantage
```

For this conversational dataset, each completion normally arrives as an
assistant message list such as
`[{"role": "assistant", "content": "..."}]`. Each callback must return one
numeric reward per completion. TRL also passes every non-prompt dataset column
as a same-length keyword argument, which is how the agreement callback receives
the repeated hidden `gold` values. `sku_id` is available for audit logging but
must not influence reward.

The environment audit also found a dependency-order constraint. Importing
`GRPOTrainer` directly fails because TRL 0.24.0 imports a vLLM symbol that is no
longer present in installed vLLM 0.23.0. Importing Unsloth first applies its
compatibility patches, after which `GRPOConfig` and `GRPOTrainer` import
successfully. The existing W0 proof-of-rig already documents and follows this
order:

```python
from unsloth import FastLanguageModel  # must precede transformers/trl imports
from trl import GRPOConfig, GRPOTrainer
```

No dependency was changed during this audit. The eventual training entry point
must preserve this ordering, and a CPU-level reward test should remain separate
from a minimal trainer-import/integration smoke on the Vast environment.

Implementation evidence:

- `training/rewards.py` contains the three plain callbacks, the ordered function
  tuple and the aligned `(1.0, 1.0, 2.0)` weight tuple. It imports no trainer,
  model, tokenizer, Torch, Unsloth or vLLM code.
- `tests/test_rewards.py` exercises the installed TRL conversational completion
  shape, plain-text completions, literal schema validity, combined vocabulary
  and rule compliance, exact known-gold agreement, gold-unknown exclusion,
  extra callback keyword arguments and default-pack loading.
- Malformed model text receives ordinary zero rewards. Malformed callback
  containers, misaligned gold batches and corrupt hidden-gold JSON raise loudly
  because they indicate integration or data corruption, not model quality.
- The **8 focused reward tests** pass; the **42-test** combined
  reward/scorer/dataset/handoff suite passes; the complete local suite passes
  with **218 tests**. These are CPU-only tests and do not claim trainer or GPU
  integration yet.

#### Remote reward and import integration probe

On 2026-08-04, commit
`8bca11354dd579b0a6a23e31e5853d1aa51c1337` was pushed to the Vast bare
remote and `/workspace/tagging-rl` was fast-forwarded to that exact commit.
Existing untracked checkpoints and run artifacts were preserved. The eight
focused reward tests then passed under the remote `/venv/rl` environment in
**0.67 seconds**. GPU state was unchanged at **428 MiB used, 23,699 MiB free and
0% utilization** before and after those CPU tests.

A separate no-training integration probe then imported the stack in the
required order and called all three committed functions with the keyword shape
TRL uses: conversational `completions`, `prompts`, `completion_ids`, hidden
`gold`, `sku_id` and `trainer_state`. It used the first real cap-four row,
`shopify:naadam.co:7696137453664`, rather than a synthetic record.

Measured environment and import result:

| check | observed result |
|---|---|
| Unsloth | `2026.7.5` |
| TRL | `0.24.0` |
| imported config class | `UnslothGRPOConfig` |
| imported trainer class | `UnslothGRPOTrainer` |
| reward function order | format, vocabulary/rule compliance, golden agreement |
| reward weights | `[1.0, 1.0, 2.0]` |
| exact-gold component outputs | `[1.0, 1.0, 1.0]` |
| exact-gold weighted total | `4.0` |
| empty-object component outputs | `[1.0, 1.0, 0.0]` |
| empty-object weighted total | `2.0` |

This reproduces the planned reward ladder through the actual installed import
path. Unsloth successfully patched TRL before the GRPO classes were imported.
The import still emitted the known Transformers-v4/vLLM deprecation warning;
no dependency was changed in response because the patched classes and callback
execution succeeded.

The probe did **not** instantiate `GRPOTrainer`, load Qwen or the SFT adapter,
create an optimizer, generate a rollout or execute a training step. GPU memory
remained **428 MiB**. Utilization was sampled at **12%** immediately after the
imports, then settled to **0%** on the follow-up reading; the settled reading
also recorded **53°C**. The remote tracked worktree remained clean at the exact
commit above. This is callback/import integration evidence, not GRPO smoke or
gradient evidence.

#### Planned minimal GRPO training smoke

**Status:** configuration design only. No trainer was instantiated and no model
was loaded while choosing these values.

The smoke has one narrow purpose: prove that the locked SFT adapter can continue
training through the real GRPO path with reward variance, finite nonzero
gradients, bounded memory and an auditable saved adapter. It is not long enough
to establish a quality trend or compare against the frozen evaluation.

##### Deterministic smoke prompts

The smoke will use exactly five prompts. With one GPU,
`per_device_train_batch_size=8`, `num_generations=8` and one generation step per
optimizer step, each optimizer step represents **one product and eight sampled
answers**. Five optimizer steps therefore produce **40 rollouts across five
products**, not 40 different products.

To maximize the chance of observing within-group reward differences without
cherry-picking individual raw outputs, the fixtures are selected mechanically:

1. Start from the committed 1,565-row cap-four active dataset.
2. Keep the 176 rows whose measured SFT pass rate is exactly `0.5`.
3. Sort by SHA-256 of `"42\0<sku_id>"`.
4. Keep only the first row from each canonical product family.
5. Take the first five rows.

The `0.5` band was measured from only eight earlier samples and is not treated
as permanent model confidence. It is useful here only because four passes and
four failures under the locked difficulty sampler indicate a practical policy
boundary. The one-row-per-family rule prevents near-duplicate variants from
occupying multiple smoke steps.

Planned fixtures, all from distinct canonical families:

| step order | SKU | store/brand | title | prior pass rate |
|---:|---|---|---|---:|
| 1 | `shopify:www.tentree.com:8106124673210` | tentree | Seaforestation Print T-Shirt | 0.5 |
| 2 | `shopify:fahertybrand.com:8164246683717` | Faherty | NY Knicks Sunwashed Regenerative Tee - Antilles Blue | 0.5 |
| 3 | `shopify:fahertybrand.com:7552625803333` | Faherty | Surf Ghana Short-Sleeve Flag Graphic Tee - White | 0.5 |
| 4 | `shopify:www.rothys.com:7543272243294` | Rothy's | The Wrap Sandal | 0.5 |
| 5 | `shopify:www.outdoorvoices.com:7686276677710` | Outdoor Voices | SuperForm Crop Top | 0.5 |

Dataset shuffling will be disabled for this five-row fixture so the step-to-SKU
mapping is reproducible. The later full run will return to the complete active
pool and seeded shuffling; smoke prompt selection must never be reported as a
representative training or evaluation sample.

##### Implemented fixture handoff

The deterministic fixture layer is now implemented independently of the
trainer. `training/build_grpo_smoke.py` verifies the parent cap-four dataset
against its committed selection manifest before applying the five-step policy.
It refuses duplicate SKUs, wrong parent hashes or row counts, invalid target
rates, insufficient distinct families and existing output paths unless
`--overwrite` is explicit.

The builder writes committed inputs for the future trainer rather than creating
the eventual `runs/grpo-first-smoke/` output directory:

| artifact | bytes | SHA-256 |
|---|---:|---|
| `data/train_weak_grpo_smoke_v1.jsonl` | 11,090 | `268373ceb08c53125976493340d972a47c90e10911e919002716590f75ca4084` |
| `data/splits/grpo-smoke-v1.json` | 5,619 | `e898510534d967b9a35367e0aba5a564e6cb564e2c326d45804d0319d528dd05` |
| `training/build_grpo_smoke.py` | 12,980 | `55e08782c123e623546ae53dd04d665730046dde55a7cd303d021e162c68882d` |

The separation matters: fixture files can be versioned and reviewed before the
GPU run, while the trainer can still require that its run-output directory does
not exist. At launch, their hashes will be copied into the run manifest rather
than regenerated or silently replaced.

The selection manifest records:

- parent active-data hash
  `3e378187a8147923bae1e0753a750d6e252336e911fa8c91cd57a4a8ddc3a102`;
- parent cap-four manifest hash
  `d166325a0c4ef3d78023ba492881fb3971e290b1b3606ee4ac8cd6aa733175e0`;
- 176 target-rate candidates across 160 canonical families;
- all five selected SKUs in optimizer-step order;
- ordered-SKU hash
  `99820aef9777190af82b999d587a260198e65ef9c420c0ca5d6befde06fe7af0`;
- each row's source position, selection hash, family key, store, brand, title,
  category and prior pass rate;
- output byte count/hash, category/store composition and gold density;
- six explicit invariants, all true.

The five fixtures span four stores and three categories: two `shirt_blouse`, two
`top` and one `shoe`. Their mean scorable-field count is 8.2, ranging from 5 to
10, and mean substantive labeled fields is 5.6. This composition is descriptive
smoke evidence, not a claim of representativeness.

Five focused CPU tests prove deterministic selection, exact expected SKUs,
family uniqueness, parent-handoff verification, locked artifact hashes, prompt
loader compatibility, output-collision refusal and invalid-request rejection.
An independent temporary rebuild reproduced the JSONL byte for byte and matched
the manifest's selection, policy, composition, invariants, inputs and
implementation hash. The complete local suite now passes with **223 tests**.
No TRL, Unsloth, model, trainer or GPU was used in this fixture step.

##### Model and adapter loading contract

| setting | planned smoke value | reason |
|---|---|---|
| starting model path | `runs/sft-combined-2epoch/checkpoint-406` | load the cryptographically locked SFT policy, not a fresh adapter |
| base model resolved by adapter | `unsloth/Qwen2.5-1.5B-Instruct` | recorded in `adapter_config.json` and already cached |
| local files only | `True` | prevent an accidental network download or model revision |
| precision | bf16 base, `load_in_4bit=False` | match SFT and fit the RTX 3090 without changing quantization |
| sequence ceiling | `896` | preserve the measured SFT ceiling; prompt plus completion budgets fit below it |
| LoRA rank / alpha | `16 / 16` | continue the selected adapter unchanged |
| LoRA modules | q/k/v/o plus gate/up/down projections | continue all 18,464,768 selected trainable parameters |
| gradient checkpointing | Unsloth mode | bound activation memory |
| fast inference / colocated vLLM | disabled | use the already-proven W0 Transformers-generation path for the first trainer smoke |

Unsloth's installed loader recognizes the checkpoint as PEFT, resolves its base
model and calls `PeftModel.from_pretrained(..., is_trainable=True)`. The smoke
must not call `get_peft_model()` again, because that would attach a new randomly
initialized LoRA instead of continuing the locked SFT adapter. Before training,
the entry point must assert the starting adapter SHA-256, rank, alpha, target
modules and exact **18,464,768** trainable-parameter count. It must verify again
afterward that the source checkpoint bytes were not modified.

##### Locked smoke `GRPOConfig`

| parameter | smoke value | reasoning |
|---|---:|---|
| `max_prompt_length` | `600` | measured maximum is 585 tokens; left truncation should be zero |
| `max_completion_length` | `170` | measured rollout maximum was 120; keeps 50 tokens of headroom |
| `num_generations` | `8` | matches difficulty scoring and gives one comparison group per prompt |
| `per_device_train_batch_size` | `8` | one prompt repeated into its eight completions on one GPU |
| `gradient_accumulation_steps` | `1` | one group per optimizer update; simplest gradient diagnosis |
| `max_grad_norm` | `1.0` | explicit trainer clipping ceiling; first real update remained below it |
| `steps_per_generation` | `1` | make the generation/update relationship explicit |
| `max_steps` | `5` | one update for each deterministic smoke prompt |
| `shuffle_dataset` | `False` | preserve the declared SKU-to-step mapping |
| `remove_unused_columns` | `False` | retain hidden `gold` and `sku_id` callback columns |
| `temperature` | `0.7` | match the difficulty run that selected these prompts |
| `top_p` | `0.95` | match the difficulty run |
| `repetition_penalty` | `1.0` | avoid introducing an unmeasured decoding intervention |
| `use_vllm` | `False` | isolate trainer/reward correctness before testing colocated acceleration |
| `learning_rate` | `5e-6` | conservative LoRA-RL rate already proven by the W0 rig |
| `warmup_ratio` | `0.0` | avoid making the first of only five smoke steps a zero-LR no-op; revisit warmup for a longer run |
| `lr_scheduler_type` | `cosine` | preserve the W0 recipe |
| `optim` | `adamw_8bit` | reduce optimizer memory; already exercised by W0 |
| `weight_decay` | `0.001` | make the observed Transformers default explicit for this smoke |
| `adam_beta1` / `adam_beta2` | `0.9 / 0.999` | lock the observed Adam moment coefficients |
| `adam_epsilon` | `1e-8` | lock the observed numerical-stability term |
| `beta` | `0.0` | avoid a reference-model path in the smallest smoke and match TRL's memory-saving default |
| `num_iterations` | `1` | one policy update per generated batch |
| `epsilon` | `0.2` | installed TRL default clipping range |
| `epsilon_high` | `0.28` | make Unsloth's DAPO upper-bound patch explicit rather than relying on an implicit override |
| `scale_rewards` | `"group"` | normalize advantages within each eight-completion group |
| `loss_type` | `"dapo"` | installed default avoids the original GRPO length bias |
| `mask_truncated_completions` | `True` | exclude any unexpected cap-hit completion from the policy loss |
| `reward_weights` | `[1.0, 1.0, 2.0]` | locked format/compliance/agreement contract |
| `bf16` / `fp16` | `True / False` | use the 3090's bf16 path without conflicting precision modes |
| `seed` / `data_seed` | `42 / 42` | deterministic model-side and data ordering where supported |
| `logging_steps` | `1` | capture every smoke update |
| `logging_first_step` | `True` | preserve the first observed gradient/reward state |
| `log_completions` | `True` | retain prompt/completion/reward evidence every step |
| `num_completions_to_print` | `8` | expose the complete comparison group during smoke debugging |
| `report_to` | `"none"` | keep the integration smoke local; reserve W&B for the real run |
| `save_strategy` | `"no"` | write no intermediate trainer checkpoint or optimizer state |
| `save_only_model` | `True` | defensive if save behavior changes; never retain smoke optimizer state |

The exact reward callbacks remain, in order,
`format_validity_reward`, `vocab_rule_compliance_reward` and
`golden_agreement_reward`. Decoding remains unconstrained: no JSON schema,
guided regex, output repair or markdown stripping is permitted.

The exact table above was instantiated as `UnslothGRPOConfig` under the remote
TRL 0.24.0 / Unsloth 2026.7.5 environment without constructing a trainer. TRL
computed `generation_batch_size=8` and accepted eight generations, batch eight,
one accumulation step and one step per generation. It normalized
`report_to="none"` to an empty reporter list. Unsloth announced that DAPO sets
`epsilon_high=0.28`; that value is therefore declared explicitly above. The
probe created no output directory, loaded no model and left GPU memory at 428
MiB. This validates configuration syntax and arithmetic only, not model memory
or a training step.

`beta=0.0` is a smoke simplification, not a claim that KL regularization is
unnecessary for the full experiment. It avoids introducing a reference-policy
memory/logic branch before basic updates are proven. The full-run value must be
locked separately after the smoke, and any change from zero must receive its
own memory preflight.

##### Output and disk contract

The smoke will refuse to overwrite an existing output directory. It will write
to a new path such as `runs/grpo-first-smoke/` and retain:

- the five-row smoke-source JSONL and deterministic selection manifest;
- all 40 raw completions with SKU, step, rollout index, three component rewards,
  weighted total, completion length and truncation status;
- trainer log history and a run manifest containing code/input/model hashes and
  every configuration value;
- stdout/stderr plus periodic GPU, temperature and disk samples;
- one final model-only adapter, saved outside the immutable SFT checkpoint.

There will be **no intermediate checkpoint** and **no optimizer state**. The
existing selected adapter directory is 86 MB; one similarly sized smoke adapter
plus text/JSON logs is small against the currently measured **4.7 GB free**.
The launcher should abort before model loading if free disk is below **3 GB**.
Full-run checkpoint frequency and retention will be locked only after this smoke
measures the actual GRPO adapter/output footprint.

##### Implemented CPU-only fail-closed preflight

`training/train_grpo.py` now implements only the read-only preflight boundary.
Calling it without `--preflight-only` exits with “training is intentionally
unavailable.” The module imports no Torch, Transformers, TRL, PEFT, Unsloth or
vLLM code, so all checks complete before any CUDA-capable library can initialize.
A structural AST test enforces that forbidden-import boundary.

| implementation artifact | bytes | SHA-256 |
|---|---:|---|
| `training/train_grpo.py` | 15,071 | `4e84320143902a112bf2572d599efb45b4ff89c388e8ecffbfdd4b35e22db279` |
| `tests/test_grpo_preflight.py` | 6,176 | `fda4e7db8d0c740419876fe5170890fa1b9b873293744647fd483407f9b75453` |

The preflight resolves and reports, without writing a run directory:

1. exact Git commit, clean tracked worktree and clean index; untracked remote
   checkpoints are allowed because they are expected run assets;
2. optional caller-supplied expected commit, which must equal `HEAD`;
3. locked fixture JSONL hash, fixture-manifest hash and six manifest invariants;
4. exactly five training rows in the declared SKU/optimizer-step order, all at
   prior pass rate 0.5, plus the ordered-SKU checksum;
5. locked SFT-selection-manifest hash and `locked_before_frozen_eval` status;
6. exact checkpoint path, base model, adapter byte count and SHA-256;
7. adapter-config rank 16, alpha 16, zero dropout, no bias and all seven combined
   attention-plus-MLP target modules;
8. locked expectation of 18,464,768 trainable LoRA parameters;
9. absence of the proposed `runs/grpo-first-smoke` output path; and
10. at least 3 GiB free on the nearest existing output filesystem.

The parameter-count wording is deliberately precise. A CPU preflight can prove
that the SFT lock expects 18,464,768 trainable parameters and that the adapter
configuration agrees. It cannot prove the loaded model's actual `requires_grad`
count without loading Qwen. The returned report therefore sets
`runtime_trainable_parameter_assertion_required: true`; the later model-load
stage must measure that count before constructing the trainer.

On success, the function returns a JSON-serializable report with Git, fixture,
SFT lock, adapter, disk and output evidence, while explicitly recording
`cuda_imports_performed: false`, `model_loaded: false` and
`trainer_constructed: false`. Output collision, dirty Git, unexpected commit,
fixture drift, adapter drift, LoRA-config drift or low disk each raises before
the function can report success.

Six focused tests use a tiny synthetic adapter to cover the successful report
and every fail-closed branch above. The first negative-test run produced two
passes and three failures because the test helper did not create a nested
temporary parent directory; changing the helper to `mkdir(parents=True)` fixed
the fixture, with no production-preflight change. The final focused result is
**6 passed**, the reward/fixture/preflight group is **19 passed**, and the full
CPU suite is **229 passed**. The real 73.9 MB locked adapter is available only
on Vast, so synthetic tests alone were not treated as the final preflight gate.

##### Real-adapter remote preflight

Commit `6b288cdf2b0584fd11d6a64c9b631d40574e9a69` was pushed to the Vast bare
remote and `/workspace/tagging-rl` was fast-forwarded to that exact commit. The
tracked worktree and index were clean. The following read-only command then ran
under `/venv/rl`:

```bash
python -m training.train_grpo \
  --preflight-only \
  --expected-commit 6b288cdf2b0584fd11d6a64c9b631d40574e9a69
```

The returned `grpo-smoke-preflight-v1` report had `status: passed` and reproduced:

| real remote check | observed value |
|---|---|
| Git commit | `6b288cdf2b0584fd11d6a64c9b631d40574e9a69` |
| tracked worktree / index dirty | `false / false` |
| fixture rows | `5` |
| fixture data SHA-256 | `268373ceb08c53125976493340d972a47c90e10911e919002716590f75ca4084` |
| fixture manifest SHA-256 | `e898510534d967b9a35367e0aba5a564e6cb564e2c326d45804d0319d528dd05` |
| ordered-SKU SHA-256 | `99820aef9777190af82b999d587a260198e65ef9c420c0ca5d6befde06fe7af0` |
| SFT selection-manifest SHA-256 | `e425635d323b3ffe9e7350fb61a2d9e1848345a95abab6b92032bf64d2718299` |
| adapter bytes | `73,911,112` |
| adapter SHA-256 | `00ae54af4e380cff66695b36b244e3f1ff9aca85076b59a8eb6649d8c3a051af` |
| LoRA rank / alpha | `16 / 16` |
| expected trainable parameters | `18,464,768`; runtime assertion still required |
| free disk / required minimum | `4,992,131,072 / 3,221,225,472` bytes |
| proposed output collision | none |
| output directory created | `false` |
| CUDA imports / model loaded / trainer constructed | `false / false / false` |

GPU state was **428 MiB used, 23,699 MiB free and 0% utilization** immediately
before and after. `runs/grpo-first-smoke` remained absent, the tracked worktree
remained clean and all existing untracked checkpoints were preserved. This is
the first preflight using the real 73.9 MB adapter, but it is still not a model
load, runtime trainable-parameter measurement or GRPO training smoke.

##### Real locked-model load gate

Commit `01fc14913ed5a05533798ece9e658f671e169e61` added a mutually exclusive
`--model-load-only` mode. It always reruns the complete CPU-visible preflight
before importing Unsloth or Torch. The mode loads the selected PEFT checkpoint
directly with `FastLanguageModel.from_pretrained()`; it never calls
`get_peft_model()`, so it cannot silently replace the selected SFT LoRA with a
new randomly initialized adapter. It then checks every runtime parameter and
exits before importing or constructing `GRPOTrainer`.

The pure trainability inspector rejects the model unless all of these are true:

- exactly `18,464,768` parameters have `requires_grad=True`;
- every trainable parameter name belongs to a LoRA matrix;
- the observed targets are exactly q/k/v/o and gate/up/down projections;
- a larger frozen base model is present; and
- the source adapter's SHA-256 remains unchanged after loading.

Three new CPU test cases cover the passing lock and failures for a one-parameter
count drift, a trainable original-model weight and target-module drift. The
focused preflight/model-gate file passes **8 tests**, and the complete local
suite passes **231 tests**. These tests validate the guard logic with fake
parameters; they do not substitute for the GPU load below.

The Vast worktree was fast-forwarded to that exact commit while preserving all
untracked checkpoints. Remote focused tests and the preflight passed first.
Only then was this command allowed to run under `/venv/rl`:

```bash
python -m training.train_grpo \
  --model-load-only \
  --expected-commit 01fc14913ed5a05533798ece9e658f671e169e61
```

Measured result:

| real runtime check | observed value |
|---|---:|
| gate status | passed |
| loaded model class | `PeftModelForCausalLM` |
| tokenizer class | `Qwen2TokenizerFast` |
| model load time | 4.607 seconds |
| total parameters exposed by loaded PEFT model | 1,562,179,072 |
| trainable parameters | 18,464,768 |
| trainable share | 1.182% |
| trainable tensors | 392 |
| trainable device | `cuda:0` |
| trainable dtype | `torch.float32` |
| observed LoRA targets | q/k/v/o plus gate/up/down |
| source adapter SHA-256 unchanged | yes |
| trainer / optimizer constructed | no / no |
| generations / training steps | 0 / 0 |
| GRPO output directory created | no |

The 392 tensors have a useful architectural explanation: Qwen has 28 decoder
layers, each layer has seven selected projection modules, and every LoRA module
has an A and a B matrix. Therefore `28 × 7 × 2 = 392`. The exact
18,464,768 count proves that the selected SFT adapter was made trainable rather
than loaded only for inference. The `1.182%` denominator is the loaded PEFT
model's full parameter count, including the adapter.

The base model was requested in bf16, but the actual trainable LoRA tensors were
float32. This is not evidence that the frozen base became float32; the gate
reports the dtype of trainable tensors specifically. Keeping small trainable
adapters in float32 gives their updates more numerical precision while the much
larger frozen base remains memory-efficient.

| CUDA memory reading | bytes | approximate GiB |
|---|---:|---:|
| Torch allocated after load | 3,190,780,416 | 2.97 |
| Torch peak allocated | 3,264,639,488 | 3.04 |
| Torch reserved after load | 3,202,351,104 | 2.98 |
| Torch peak reserved | 3,275,751,424 | 3.05 |
| Torch allocated after release | 12,713,984 | 0.012 |
| Torch reserved after release | 20,971,520 | 0.020 |

The external GPU reading was **428 MiB used and 23,699 MiB free** before the
command and returned to the same values after process exit. A settled follow-up
also showed **0% utilization** and **47°C**. Free disk remained
`4,990,865,408` bytes, the tracked worktree/index remained clean,
`runs/grpo-first-smoke` remained absent, and the adapter checksum remained
`00ae54af4e380cff66695b36b244e3f1ff9aca85076b59a8eb6649d8c3a051af`.

This result proves that the locked SFT policy fits, is attached correctly and
can expose exactly the intended LoRA weights for optimization. It does **not**
yet prove that eight-completion generation, reward callbacks, optimizer state,
backpropagation or a full GRPO update fit together. Those belong to the next
gates.

##### Real no-training GRPO trainer-construction gate

Commit `c4a07a277c132c3fb9269ac5d023ce1be5e7aad4` added a third mutually
exclusive mode, `--trainer-construction-only`. The normal entry point still has
no training mode. This gate reruns the entire preflight, loads and reasserts the
locked LoRA, builds the five-row conversational Hugging Face dataset, creates
the exact `GRPOConfig`, constructs `GRPOTrainer`, inspects the result and then
releases everything without calling `generate()` or `train()`.

The configuration is produced by one pure function rather than being repeated
between probes and the future training path. CPU tests lock all planned values,
including rewards `(1, 1, 2)`, eight generations, batch eight, accumulation one,
five maximum steps, disabled vLLM, `beta=0`, DAPO loss, bf16, and no checkpoint
saves. A second pure inspector verifies TRL's post-construction values and
allows only the documented normalization of `report_to="none"` into `[]`.

The focused file now passes **10 tests** and the complete local suite passes
**233 tests**. Before the GPU command, those same ten focused tests passed under
the remote `/venv/rl`. An initial preflight invocation used the abbreviated
commit `c4a07a2`; it was correctly rejected because the gate compares the exact
40-character commit identity. No CUDA-capable import occurred on that rejected
attempt. The corrected full-hash preflight then passed.

The real command was:

```bash
python -m training.train_grpo \
  --trainer-construction-only \
  --expected-commit c4a07a277c132c3fb9269ac5d023ce1be5e7aad4
```

Measured integration result:

| real trainer check | observed value |
|---|---:|
| gate status | passed |
| trainer class | `UnslothGRPOTrainer` |
| config class | `UnslothGRPOConfig` |
| model class inside trainer | `PeftModelForCausalLM` |
| construction time, including model load | 4.985 seconds |
| trainer dataset rows | 5 |
| columns retained by trainer | `prompt`, `gold`, `sku_id` |
| deterministic SKU order retained | yes |
| reward order retained | format, compliance, golden agreement |
| runtime reward weights | `[1.0, 1.0, 2.0]` |
| computed generation batch size | 8 |
| prompt groups per generation batch | 1 |
| trainable parameters before / after trainer | 18,464,768 / 18,464,768 |
| optimizer constructed | no |
| LR scheduler constructed | no |
| reference model constructed | no |
| global step | 0 |
| generations / training steps | 0 / 0 |

The hidden-column check matters because the agreement reward cannot score a
completion without its trusted `gold`, and rollout audit records cannot be tied
back to a product without `sku_id`. `remove_unused_columns=False` worked in the
actual trainer: neither column was stripped during construction.

The batch arithmetic also survived the installed TRL/Unsloth normalization:
`generation_batch_size=8`, divided by `num_generations=8`, gives exactly **one
product comparison group per generation batch**. With five locked products and
five planned updates, this preserves the intended one-product-per-step smoke
interpretation.

`beta=0` produced no reference model, exactly as intended for the smallest
memory-safe integration smoke. The optimizer and LR scheduler also remained
`None`; TRL creates those lazily only when training starts. Therefore this gate
tests trainer wiring without yet spending optimizer memory or changing a
weight. The LoRA count and source adapter checksum were unchanged before and
after trainer construction.

CUDA memory after trainer construction was identical to the earlier model-only
gate: **3,190,780,416 bytes (2.97 GiB) allocated**, with a **3,264,639,488-byte
(3.04 GiB) peak**. In this environment, constructing the trainer itself added
no measurable persistent CUDA allocation beyond loading the policy. This does
not estimate optimizer, generated-token activation or backward-pass memory.

The gate used a temporary output directory
`/tmp/grpo-trainer-construction-up5k2o58` so trainer logging setup could not
claim the reserved real-run path. The temporary directory was removed,
`runs/grpo-first-smoke` remained absent, and free disk finished at
`4,995,141,632` bytes. The external GPU reading settled back to **428 MiB used,
23,699 MiB free, 0% utilization and 49°C**. The tracked worktree/index remained
clean and the source adapter still matched
`00ae54af4e380cff66695b36b244e3f1ff9aca85076b59a8eb6649d8c3a051af`.

This gate proves the complete static integration boundary: locked policy +
ordered prompts + hidden reward data + three callbacks + aligned weights +
planned TRL configuration. It still does **not** prove generation, reward
variance, optimizer creation, gradients or parameter updates. The next gate
should cross only one of those boundaries at a time.

##### Real one-group rollout and reward gate

Commit `2e8b6792b0e9351e121f2824fac73abd24540944` added
`--rollout-only`. It reuses the exact trainer-construction path, takes the first
batch from the installed trainer's real `get_train_dataloader()`, and calls
`_prepare_inputs()` once. In the installed Unsloth trainer, that is the method
which invokes `_generate_and_score_completions()`, decodes model text, calls all
three rewards, applies `[1, 1, 2]`, computes group-normalized advantages and
returns prepared tensors. The gate never calls `compute_loss()`, `backward()`,
`optimizer.step()` or `trainer.train()`.

The installed repeat sampler was inspected before implementation. With
`shuffle_dataset=False`, `generation_batch_size=8` and `num_generations=8`, its
first batch is the first locked SKU repeated exactly eight times. Runtime
assertions confirmed that exact batch and verified that hidden `gold` remained
present before generation.

The gate also hashes the names, dtype, shape and raw bytes of every trainable
LoRA tensor immediately before and after rollout. This is stronger than merely
checking `global_step=0`: it can detect an accidental in-memory weight mutation
even if trainer state was not advanced.

Two pure CPU tests validate rollout-evidence alignment, raw-output retention,
component-to-weight ordering, exact weighted totals and rejection of NaN or
misaligned arrays. A later collision-safe report-file addition brings the
focused total to **13 tests** and the complete local suite to **236 tests**.
The remote focused suite also passed before either GPU rollout.

The first rollout ran at commit `2e8b679...` as terminal evidence. Commit
`617306cb7549b40d477c9d8688c11fb5e4fc2776` then added an optional
`--report-file` which is valid only for rollout mode, checks for collisions
before CUDA import, refuses overwrite and requires an existing parent. The
deterministic rollout was rerun to produce the versioned evidence artifact:

| artifact | bytes | SHA-256 |
|---|---:|---|
| `runs/grpo-rollout-gate-v1.json` | 81,578 | `84a01ab1840a430f0b5b066d48499b8b6d544c2feb224d8b459ff4ea9be5f8bc` |

The report artifact and this technical write-up were committed together as
`8933af544277de414ce23081b4da92b337702695`, pushed to the Vast bare remote and
fast-forwarded into `/workspace/tagging-rl`. The tracked remote copy was then
byte-compared with the original GPU-generated file before the latter was kept
as the recoverable temporary backup
`/tmp/grpo-rollout-gate-v1-617306c.json`.

The ordered eight raw outputs inside that artifact have SHA-256
`029d63fd3391b1e35fa76668d1a6a2e4c8125dcf4ea61af044a8f44612408c4a`.
The persisted rerun reproduced the first run's reward totals, advantages and
completion-token sequence.

The product was the first declared smoke fixture,
`shopify:www.tentree.com:8106124673210`, the Seaforestation Print T-Shirt. Its
prior eight-sample SFT pass rate was `0.5`; this Transformers rollout produced
three exact known-gold passes out of eight (`0.375`). A one-completion difference
at this tiny sample size, and across the earlier vLLM versus current
Transformers generation paths, is not evidence of policy drift.

Measured reward result:

| component/result | observed count |
|---|---:|
| structurally valid JSON | 8 / 8 |
| vocabulary- and rule-compliant | 8 / 8 |
| exact agreement on every scorable gold field | 3 / 8 |
| weighted total `2.0` | 5 / 8 |
| weighted total `4.0` | 3 / 8 |
| nonzero normalized advantage | 8 / 8 |
| truncated and masked | 0 / 8 |

The exact weighted totals were:

```text
[2, 4, 2, 2, 4, 2, 4, 2]
```

The corresponding installed-TRL group-normalized advantages were:

```text
[-0.724499, 1.207498, -0.724499, -0.724499,
  1.207498, -0.724499, 1.207498, -0.724499]
```

This is the first direct evidence that the chosen prompt can produce a GRPO
learning signal. Format and compliance were constant at one for all eight
samples, so they contributed the same two-point baseline and did not determine
relative advantage in this group. Golden agreement was the discriminator: its
two-point weight raised the three exact outputs from total two to total four.
Those three received positive advantage; the other five received negative
advantage. GRPO therefore has a clear local preference signal without needing
an absolute hand-written score threshold.

This also illustrates why the components remain separately logged. Saying
“mean reward was 2.75” would hide the important fact that syntax and vocabulary
were already saturated and the useful variation came entirely from semantic
agreement. If future groups show all totals equal, their advantages will be
zero and they will contribute no gradient regardless of their absolute score.

The effective completion lengths were `[112, 113, 111, 111, 111, 110, 112,
111]` tokens. All ended normally below the 170-token cap, so
`mask_truncated_completions=True` discarded none. The prepared batch contained
prompt IDs/masks, completion IDs/masks, advantages, maximum left padding and
item count; aligned settings meant no old-policy or reference-policy log-prob
tensor was required.

The in-memory trainable-LoRA fingerprint was identical before and after:

```text
1c9f10100bfc250323ad43c0e8b1b170a909d842fb2579903200f95a786e711e
```

`global_step` stayed zero, no optimizer or scheduler existed, and the immutable
source adapter retained SHA-256
`00ae54af4e380cff66695b36b244e3f1ff9aca85076b59a8eb6649d8c3a051af`.
This proves generation and reward calculation did not change either the live
LoRA or its source checkpoint.

The persisted rollout itself took **5.318 seconds**; model load, trainer setup,
fingerprinting, rollout and reporting together took **10.532 seconds**. CUDA
memory moved as follows:

| CUDA reading | bytes | approximate GiB |
|---|---:|---:|
| allocated before rollout | 3,190,780,416 | 2.97 |
| allocated after rollout preparation | 3,402,417,664 | 3.17 |
| peak allocated | 4,250,522,112 | 3.96 |
| peak reserved | 4,318,035,968 | 4.02 |
| allocated after release | 13,762,560 | 0.013 |
| reserved after release | 23,068,672 | 0.021 |

The rollout increased peak allocation by about **0.918 GiB** over the earlier
model-only peak. That comfortably fits the RTX 3090, but it still excludes loss
activations, gradients, optimizer state and an update. The temporary trainer
directory was removed, `runs/grpo-first-smoke` remained absent, free disk before
the persisted run was `4,989,554,688` bytes, and the settled external GPU state
returned to **428 MiB used, 23,699 MiB free and 0% utilization**.

The next narrow boundary was a **gradient-only gate**: reuse one prepared
eight-completion group, compute the policy loss and call backward, inspect
finite/nonzero LoRA gradients and memory, but do not create or step an optimizer.

##### Real one-group loss and gradient-only gate

Commit `de16b440a4fb943b9e5de10daa3784387f22844f` added
`--gradient-only`. This mode deliberately reuses the same real trainer and
prepared rollout path as `--rollout-only`, then crosses exactly two new
boundaries:

1. `trainer.compute_loss(...)` computes the installed Unsloth/TRL GRPO loss for
   the prepared eight-completion group.
2. `trainer.accelerator.backward(loss)` differentiates that loss through the
   policy into the trainable LoRA tensors.

It does **not** call `trainer.train()`, construct an optimizer or scheduler, call
`optimizer.step()`, save a checkpoint or advance `global_step`. This separation
is useful because it answers “can the reward signal reach every trainable LoRA
tensor?” before an optimizer is allowed to change anything.

The implementation fails closed unless the runtime trainable footprint is
exactly **392 LoRA tensors and 18,464,768 elements**, every tensor has a
gradient, every gradient is finite, and at least one tensor and element are
nonzero. It records aggregate and per-target-module norms, hashes the live LoRA
weights immediately before and after backward, checks that optimization state
is still absent, then clears all gradients and verifies that no gradient remains
attached. CPU-only tests cover the passing contract plus missing-gradient,
nonfinite-gradient and all-zero-gradient failures. The focused suite passed
**14 tests** locally and remotely; the full local repository passed **237
tests**.

The exact GPU command was:

```bash
/venv/rl/bin/python -m training.train_grpo \
  --gradient-only \
  --report-file runs/grpo-gradient-gate-v1.json \
  --expected-commit de16b440a4fb943b9e5de10daa3784387f22844f
```

Preflight on the same commit found `4,994,162,688` free bytes, a clean tracked
worktree/index, an absent reserved output directory and the expected locked
adapter SHA. The run produced this immutable evidence file:

| artifact | bytes | SHA-256 |
|---|---:|---|
| `runs/grpo-gradient-gate-v1.json` | 86,805 | `f6a19ce530e350bb73db4553ef5e3424822a04133ebc34ee21c6ad49bae67252` |

The ordered raw-output SHA is again
`029d63fd3391b1e35fa76668d1a6a2e4c8125dcf4ea61af044a8f44612408c4a`.
Therefore this gate reused exactly the rollout observed by the earlier
rollout-only artifact: reward totals `[2, 4, 2, 2, 4, 2, 4, 2]`, three positive
advantages, five negative advantages, no truncation and useful within-group
variance. Holding the sampled texts fixed makes the new evidence specifically
about loss and backward behavior, rather than a lucky change in generation.

Measured backward result:

| measurement | observed value |
|---|---:|
| scalar GRPO loss | `-0.0062339865` |
| forward + backward time | 88.854 seconds |
| trainable LoRA tensors | 392 |
| tensors with gradients | 392 |
| tensors with a nonzero gradient | 392 |
| trainable/gradient elements | 18,464,768 |
| nonzero gradient elements | 18,464,227 |
| exactly zero gradient elements | 541 |
| NaN or infinite gradient elements | 0 |
| global LoRA gradient L2 norm | 0.489256 |
| largest absolute gradient element | 0.0230713 |
| mean absolute gradient element | 0.0000397039 |
| gradient dtype | float32 |
| optimizer / scheduler constructed | no / no |
| optimizer step performed | no |
| `global_step` | 0 |
| gradients remaining after cleanup | 0 |

All **392 of 392** tensors had some nonzero signal. Only 541 of 18,464,768
individual elements were exactly zero—about **0.0029%**—so approximately
**99.9971%** of LoRA gradient elements were nonzero. More importantly, there
were no missing, NaN or infinite gradients. The reward differences therefore
survived the complete path from decoded outputs, through component rewards and
group-relative advantages, through the policy loss, and into every configured
attention and MLP LoRA target.

The scalar loss being slightly negative is not a failure. A policy-gradient
loss is an optimization objective, not an error percentage like supervised
cross-entropy, and its absolute value can sit near zero after positive and
negative group advantages balance. At this gate the decisive evidence is the
finite nonzero gradient norm, not whether the loss is positive or large.

Per-target gradient norms were:

| LoRA target | tensors with nonzero gradient | elements | L2 norm | max absolute element |
|---|---:|---:|---:|---:|
| `gate_proj` | 56 / 56 | 4,702,208 | 0.352358 | 0.0209961 |
| `up_proj` | 56 / 56 | 4,702,208 | 0.257834 | 0.0230713 |
| `down_proj` | 56 / 56 | 4,702,208 | 0.149495 | 0.0016098 |
| `o_proj` | 56 / 56 | 1,376,256 | 0.099865 | 0.0013809 |
| `v_proj` | 56 / 56 | 802,816 | 0.093296 | 0.0043640 |
| `q_proj` | 56 / 56 | 1,376,256 | 0.072514 | 0.0025177 |
| `k_proj` | 56 / 56 | 802,816 | 0.049525 | 0.0043335 |

`gate_proj` and `up_proj` had the largest aggregate norms in this one group,
but that does not prove they are the most important modules: module groups have
different element counts, and one prompt is not a stable importance estimate.
The defensible inference is narrower—all seven intended target families and all
28 model layers participated in backward.

CUDA memory around backward was:

| CUDA reading | bytes | approximate GiB |
|---|---:|---:|
| allocated before backward | 3,402,417,664 | 3.17 |
| allocated after backward | 3,294,333,952 | 3.07 |
| peak allocated during loss/backward | 3,788,344,832 | 3.53 |
| peak reserved during loss/backward | 4,357,881,856 | 4.06 |
| allocated after gradient clear | 3,212,697,088 | 2.99 |
| allocated after full model/trainer release | 33,189,888 | 0.031 |
| reserved after full release | 69,206,016 | 0.064 |

The backward-specific peak allocation was lower than the earlier generation
peak because Unsloth reported smart gradient offloading and the peak counter was
reset after rollout preparation. These are phase-local measurements, so they
should not be added together. The highest peak allocated anywhere in this
combined gate remained the rollout's **4,250,522,112 bytes (3.96 GiB)**, while
the highest reserved value was the backward phase's **4,357,881,856 bytes (4.06
GiB)**. This still excludes optimizer state and an optimizer update, which are
the next material memory boundary.

The live LoRA fingerprint remained byte-identical across backward:

```text
1c9f10100bfc250323ad43c0e8b1b170a909d842fb2579903200f95a786e711e
```

The source adapter retained SHA-256
`00ae54af4e380cff66695b36b244e3f1ff9aca85076b59a8eb6649d8c3a051af`,
`runs/grpo-first-smoke` remained absent, and settled external GPU use returned
to the **428 MiB** idle ComfyUI process at 0% utilization. After writing the
artifact, disk had `4,818,882,560` bytes free. Thus backward generated
temporary gradient state, the audit measured it, cleanup erased it, and no
persistent model state changed.

This gate proves that the locked policy, real sampled completions, rewards and
installed GRPO loss form a differentiable learning path with bounded memory. It
does **not** prove that AdamW can be constructed within the remaining memory,
that an update changes the adapter in the intended direction, that five steps
are stable, or that GRPO improves frozen-evaluation quality. Those claims remain
behind later gates.

##### Real 8-bit AdamW construction-only gate

The next isolated boundary was optimizer construction: connect the component
that will eventually turn gradients into weight changes, but do not generate a
rollout, compute a loss, backpropagate, initialize per-parameter moment state or
call `optimizer.step()`.

Commit `9693dc023249dafd4703974649649ae02bb2a42d` added
`--optimizer-construction-only`. The gate calls the installed trainer's real
`create_optimizer()` method and fails unless the result:

- is bitsandbytes `AdamW` configured for 8-bit, non-paged state;
- references every one of the 392 trainable LoRA tensors exactly once;
- references all 18,464,768 trainable elements and no frozen parameter;
- has no attached gradients or initialized parameter state;
- leaves `global_step` at zero and creates no LR scheduler; and
- leaves the in-memory LoRA fingerprint byte-identical.

The first real construction probe passed those scope checks and revealed a
valuable audit gap. Learning rate and optimizer name were explicit, but weight
decay, Adam betas and epsilon still came from installed-library defaults. The
probe observed `0.001`, `(0.9, 0.999)` and `1e-8`. Commit
`93a560d69a680b0c6780f028faba6f63f42f327f` then made those same values explicit
in `grpo_smoke_config_kwargs()` and added fail-closed runtime assertions. This
did not deliberately change behavior; it changed hidden defaults into a
versioned contract. The original probe remains recoverable on the GPU box as
`/tmp/grpo-optimizer-construction-v1-9693dc0.json`.

The publishable rerun used:

```bash
/venv/rl/bin/python -m training.train_grpo \
  --optimizer-construction-only \
  --report-file runs/grpo-optimizer-construction-v1.json \
  --expected-commit 93a560d69a680b0c6780f028faba6f63f42f327f
```

The final artifact is:

| artifact | bytes | SHA-256 |
|---|---:|---|
| `runs/grpo-optimizer-construction-v1.json` | 76,373 | `bf602f55be70f93dc06b25fc90e22e712b7b172f6b698913ff21d710bccf9f05` |

Focused tests passed **15/15** locally and remotely, and the complete local
suite passed **238 tests**. Preflight on the final commit found
`4,813,078,528` free bytes, a clean tracked worktree/index, an absent reserved
training output and the expected source-adapter checksum.

Measured optimizer wiring:

| property | observed value |
|---|---:|
| implementation | `bitsandbytes.optim.adamw.AdamW` |
| optimizer precision | 8-bit |
| paged optimizer | no |
| learning rate | `5e-6` |
| weight decay | `0.001` |
| Adam betas | `(0.9, 0.999)` |
| Adam epsilon | `1e-8` |
| parameter groups | 2 |
| referenced trainable tensors | 392 |
| unique referenced tensors | 392 |
| referenced elements | 18,464,768 |
| missing / duplicate / frozen references | 0 / 0 / 0 |
| attached gradients | 0 |
| optimizer state entries / tensors / bytes | 0 / 0 / 0 |
| optimizer initialized flag | false |
| scheduler constructed | no |
| optimizer step / global step | no / 0 |

Transformers created its standard decay and no-decay groups. All 392 LoRA
matrix tensors were in the nonempty `weight_decay=0.001` group; the no-decay
group existed but contained zero tensors. This is internally consistent for
the current trainable scope because no bias or normalization parameter is
trainable. It is a smoke-run choice, not evidence that `0.001` is the best
weight decay for a longer experiment.

Optimizer construction took **0.0337 seconds**. CUDA allocation was exactly
`3,190,780,416` bytes immediately before and after construction, with the same
`3,190,780,416`-byte phase peak; reserved CUDA memory likewise stayed at
`3,202,351,104` bytes. The only newly measured optimizer-owned tensors were two
shared float32 quantization lookup maps on CPU, 1,024 bytes each, or **2,048
bytes total**.

The reason this looks almost free is important: bitsandbytes AdamW is lazy. At
construction it stores references to the LoRA parameters and basic settings.
Its `state1` and `state2` moment buffers are created only when the first update
is attempted. Therefore **zero CUDA growth here does not mean optimizer state
will cost zero memory during training**.

From the installed bitsandbytes state layout, a rough lower-bound estimate for
18,464,768 fully 8-bit LoRA elements is two one-byte moment buffers plus two
float32 block scales per 256 elements:

```text
2 × 18,464,768 bytes for moments
+ 2 × (18,464,768 / 256) × 4 bytes for scales
= 37,506,560 bytes, about 35.77 MiB
```

That estimate excludes allocator rounding, metadata, temporary update buffers
and any additional trainer allocations. The first real update—not this
construction gate—must provide the authoritative peak.

The live LoRA fingerprint stayed unchanged:

```text
1c9f10100bfc250323ad43c0e8b1b170a909d842fb2579903200f95a786e711e
```

The source adapter also retained SHA-256
`00ae54af4e380cff66695b36b244e3f1ff9aca85076b59a8eb6649d8c3a051af`.
The optimizer was detached and released, its one temporary global embedding
override was removed, the temporary trainer directory disappeared, and
`runs/grpo-first-smoke` remained absent. Final external GPU use returned to the
**428 MiB** idle ComfyUI process at 0% utilization; disk had `4,812,922,880`
bytes free after the evidence file was written.

This proves the configured optimizer points at exactly the intended LoRA and
that merely creating it is safe. It does **not** yet prove moment-state
allocation, scheduler behavior, gradient clipping, a weight update, update
magnitude or post-update model quality. Those begin at the one-real-update
boundary.

##### One real trainer update, then stop

Before implementing the first update, an installed-scheduler probe exposed a
five-step-smoke bug. With `warmup_ratio=0.1`, Transformers computes one warmup
step and initializes the first learning rate to zero:

```text
planned steps:                    5
ceil(5 × 0.1) warmup steps:       1
learning rate used at step 1:     0.0
```

That would create optimizer state and increment `global_step`, but would not
change a weight. In a five-step smoke it would waste 20% of the experiment and
contradict the acceptance requirement for five real updates. Commit
`ea87577bac7b5b12f9344ba5cf8d5ed21f2bd163` therefore locked smoke
`warmup_ratio=0.0`. This is not a claim that warmup is generally bad; warmup is
reasonable for a longer run. It is inappropriate when one warmup step is one
fifth of a tiny integration smoke.

Commit `feae967b2e3b351ef971f667317f3216dc1c1b9b` added
`--one-update-only`. It uses the real `trainer.train()` loop and unchanged
five-step schedule, but installs a callback that sets `should_training_stop`
immediately after `global_step` becomes one. This exercises the normal trainer
path—rollout, rewards, loss, backward, gradient clipping check, optimizer step,
scheduler step and logging—without manually imitating training-loop order.

The gate also locks `max_grad_norm=1.0`, copies the starting LoRA to CPU, and
fails unless:

- the callback is called exactly once at `global_step=1`;
- the first step uses learning rate `5e-6` and the cosine scheduler produces
  the expected next learning rate;
- all 392 LoRA tensors change, all deltas and resulting values are finite, and
  the live LoRA hash differs;
- bitsandbytes initializes step-one 8-bit state for exactly all 392 LoRA
  tensors and no frozen tensor;
- the rollout contains eight raw completions with nonzero reward variance; and
- the immutable source adapter retains its locked SHA-256.

The exact run was:

```bash
/venv/rl/bin/python -m training.train_grpo \
  --one-update-only \
  --report-file runs/grpo-one-update-gate-v1.json \
  --expected-commit feae967b2e3b351ef971f667317f3216dc1c1b9b
```

Its evidence artifact is:

| artifact | bytes | SHA-256 |
|---|---:|---|
| `runs/grpo-one-update-gate-v1.json` | 88,815 | `bed88d14c83771a634a26f5ff538d5f9a89d7caa5441823183439662e963d50d` |

The focused CPU suite passed **18 tests** locally and remotely; the complete
local suite passed **241 tests**. Preflight found `4,886,654,976` free bytes, a
clean tracked worktree/index, an absent reserved training output and the exact
source-adapter checksum.

The trainer reported five planned steps but the callback stopped it after the
first:

| trainer result | observed value |
|---|---:|
| completed global steps | 1 / 5 |
| epoch fraction | 0.2 |
| callback step-end calls | `[1]` |
| trainer runtime | 17.966 seconds |
| audited update section | 19.981 seconds |
| model load + trainer + update + audits | 27.903 seconds |
| train loss | `-0.006234` |
| gradient norm before clipping | `0.489256` |
| configured max gradient norm | `1.0` |
| learning rate used | `5e-6` |
| next cosine-scheduled learning rate | `4.5225425e-6` |

Because `0.489256 < 1.0`, gradient clipping did not need to shrink this step's
gradient. This gradient norm should not be confused with the logged GRPO
`clip_ratio/*` metrics, which measure policy-ratio clipping; those were also
zero, but describe a different mechanism.

The exact same deterministic first-prompt rollout reappeared. Its ordered raw
outputs again hash to
`029d63fd3391b1e35fa76668d1a6a2e4c8125dcf4ea61af044a8f44612408c4a`,
with totals:

```text
[2, 4, 2, 2, 4, 2, 4, 2]
```

Format and compliance remained 8/8, golden agreement remained 3/8, all eight
advantages were nonzero, mean reward was `2.75`, reward standard deviation was
`1.0351`, mean completion length was `111.375` tokens and no completion was
clipped. Reusing identical sampled text isolates this gate's new evidence to the
optimizer/scheduler/update path rather than generation luck.

The live LoRA fingerprint changed from:

```text
before: 1c9f10100bfc250323ad43c0e8b1b170a909d842fb2579903200f95a786e711e
after:  be665ea319128f176c5f546d4ceeff2d4bd0f187f13ff325233dcf1699a5342b
```

Measured parameter movement:

| update measurement | observed value |
|---|---:|
| LoRA tensors changed | 392 / 392 |
| elements changed | 18,464,225 / 18,464,768 |
| unchanged elements | 543 |
| nonfinite resulting values / deltas | 0 / 0 |
| global update L2 norm | `0.0213529` |
| starting LoRA L2 norm | `33.49877` |
| relative update L2 | `0.0006374` = 0.06374% |
| largest absolute element change | `5.00027e-6` |
| mean absolute element change | `4.96642e-6` |

All seven target families changed in all 56 of their tensors. Their delta L2
norms ranged from `0.004412` for `v_proj` to `0.010820` for `up_proj`. The
near-`5e-6` element changes are consistent with Adam's normalized first update
at a `5e-6` learning rate. The small 0.06374% relative L2 movement is reassuring
for an integration step, but it does not prove the direction improves quality.

The optimizer-state estimate from the construction gate was exact. After step
one, every LoRA tensor had state at optimizer step one:

| optimizer-state measurement | observed value |
|---|---:|
| parameter state entries | 392 |
| `state1` elements / dtype | 18,464,768 / uint8 |
| `state2` elements / dtype | 18,464,768 / uint8 |
| first / second block scales | 72,128 / 72,128 float32 values |
| unique state tensors | 1,570 |
| unique state storage | 37,508,608 bytes = 35.77 MiB |
| missing / foreign state entries | 0 / 0 |
| nonfinite state elements | 0 |

The 1,570 unique tensors are 392 first moments, 392 second moments, 392 first
scale arrays, 392 second scale arrays and two shared quantization maps. This is
why 8-bit AdamW state is much smaller than two full float32 moment copies.

CUDA memory remained bounded:

| CUDA reading | bytes | approximate GiB |
|---|---:|---:|
| allocated before update | 3,190,780,416 | 2.97 |
| allocated after update/state initialization | 3,250,374,144 | 3.03 |
| peak allocated during full step | 4,250,522,112 | 3.96 |
| peak reserved during full step | 4,318,035,968 | 4.02 |
| allocated after complete release | 33,189,888 | 0.031 |
| reserved after complete release | 69,206,016 | 0.064 |

Persistent allocation rose by about **56.83 MiB** after the update. The
measured optimizer state accounts for 35.77 MiB; allocator and other retained
training objects account for the remainder. Peak allocation did not exceed the
earlier rollout gate's 3.96 GiB because the new state fit within the already
observed generation/backward memory envelope.

No updated checkpoint was intentionally saved at this gate. The changed model
existed only in memory and was released after auditing, while the locked source
adapter retained SHA-256
`00ae54af4e380cff66695b36b244e3f1ff9aca85076b59a8eb6649d8c3a051af`.
The temporary trainer directory was removed, `runs/grpo-first-smoke` remained
absent, disk ended at `4,886,253,568` free bytes, and external GPU use settled
at **264 MiB and 0% utilization**.

This gate proves one real update is numerically active, correctly scoped and
memory-safe. It does **not** prove that later prompts remain stable, that a
five-step adapter can be saved/reloaded, or that model quality improves. Those
are the remaining purposes of the full five-step smoke and frozen evaluation.

##### Five-step output persistence contract (implemented, not yet run)

Commit `6da88ff` added a fail-closed persistence layer before connecting the
full five-step trainer. This was deliberately tested without launching CUDA or
performing another update. The purpose is to make a successful smoke
recoverable and auditable while preventing a partial run, optimizer checkpoint
or full-model save from being mistaken for the final LoRA result.

The completed output is allowed to contain exactly:

```text
runs/grpo-first-smoke/
├── adapter/           # final LoRA adapter plus tokenizer files
├── manifest.json      # locks code, inputs, config, runtime and hashes
├── rollouts.jsonl     # all 5 × 8 raw completions and reward details
└── trainer-log.json   # one complete scalar record for each update
```

Publication uses a same-parent temporary directory named
`.grpo-first-smoke.staging-*`. The code first refuses an existing final output,
then validates the entire temporary bundle, and only then renames the directory
to `runs/grpo-first-smoke`. Because staging and final output share a filesystem,
the rename is atomic: readers see either no final run or the completed run, not
a half-written final directory. A failed validation leaves the staging
directory visible for diagnosis and does not publish it as completed evidence.

The validator locks the following conditions:

- exactly five ordered SKU groups and 40 total rollout records;
- rollout indices `0..7` per step, nonempty raw output, all three reward
  components, a correctly recomputed `1:1:2` weighted total, finite advantage,
  `1..170` effective completion tokens and no truncation/masking;
- at least one reward-varying group and at least one nonzero advantage;
- exactly five complete trainer step logs, finite required GRPO metrics,
  positive gradient norms, zero completion clipping and the expected no-warmup
  cosine learning rates;
- the exact smoke configuration: batch/generation size eight, one accumulation
  step, learning rate `5e-6`, zero warmup, max gradient norm `1.0`, 8-bit AdamW,
  beta zero, no vLLM trainer backend and no intermediate saves;
- exactly 392 trainable LoRA tensors and 18,464,768 trainable parameters, five
  optimizer steps, 40 rollout records, a changed live LoRA fingerprint and an
  unchanged source adapter;
- a clean tracked Git/index state plus full code, source-adapter, fixture and
  selection hashes;
- positive CUDA peak allocation, reserved peak no smaller than allocated peak,
  and at least 3 GiB free disk after the run.

The saved adapter has an exact allowlist matching the locked PEFT/tokenizer
layout. Its `adapter_model.safetensors` must remain exactly 73,911,112 bytes,
retain rank 16 and all seven combined attention/MLP targets, and differ in
SHA-256 from the starting adapter. Optimizer/scheduler/RNG/trainer state,
intermediate `checkpoint-*` directories and full-model weight names are
forbidden. The complete bundle is capped at 256 MiB, which prevents an
accidental full checkpoint from consuming the machine's limited disk.

Seven focused CPU tests cover the valid bundle and deliberately broken cases:
misordered or truncated rollouts, wrong learning-rate schedules, missing log
metrics, optimizer-state leakage, unchanged or wrong-sized adapter weights,
configuration drift, less than the 3 GiB disk floor, an existing final output,
an unrelated staging directory and an incomplete staging root. They passed
locally and on Vast with `CUDA_VISIBLE_DEVICES=""`; the full local repository
suite passed **248 tests**. No training output was created and no GPU smoke was
run in this step.

Commit `a33fce3` connected the final save/publish boundary to
`training/train_grpo.py`, still without exposing a full-smoke CLI or running the
GPU. The handoff now:

1. refuses an existing final output or an unrelated/nonempty staging directory
   before `save_pretrained()` is called;
2. re-hashes the locked source adapter immediately before saving;
3. calls the live model with `safe_serialization=True` and saves the tokenizer
   beside the LoRA;
4. re-hashes the source adapter after the save to detect accidental mutation;
5. measures free disk after the adapter has consumed space, rather than relying
   only on the earlier preflight reading;
6. maps the preflight hashes, exact trainer configuration, trainable scope,
   global/optimizer-step counts, LoRA before/after fingerprints and measured
   CUDA peaks into the publisher context; and
7. invokes the fail-closed validator and atomic directory publication.

New fake-model/tokenizer integration tests prove the successful handoff and
also prove that source mutation, a post-save disk reading one byte below 3 GiB,
or an unbound staging path leaves the final output absent. The combined
preflight/persistence/handoff suite passed **29 CPU-only tests** locally and on
Vast with CUDA hidden. The complete local repository suite passed **252
tests**.

This remains implementation evidence, not evidence that the five-step smoke
passes. Installed-TRL inspection showed why another boundary is required:
`GRPOTrainer._logs` uses deques capped at the generation batch size and retains
only the latest eight completions. Reading it after five updates would silently
lose the first 32 rollouts. The next gate must therefore capture and freeze each
eight-completion group when it is generated, assign it to the correct ordered
step/SKU, and test all five groups before enabling training.

Commit `b96cd93` implemented that generation-time capture boundary. A narrow
`GRPOTrainer` subclass wrapper calls the installed trainer's real
`_generate_and_score_completions()` first, copies its just-produced evidence to
ordinary CPU-owned Python values, and then returns the original prepared tensor
dictionary unchanged so normal loss/backward behavior is not altered.

For every group, the collector fails unless:

- the run is single-process, matching the one-RTX-3090 smoke contract;
- `global_step + 1` is exactly the next step from one through five;
- all eight input rows repeat the SKU predeclared for that step;
- completion text, three component-reward arrays, reward weights and advantages
  are aligned and finite; and
- the prepared completion mask has eight binary rows from which effective token
  counts and truncation/masking status can be copied.

Each record receives its step number before the next optimizer step can replace
TRL's deque. Returned evidence is deep-copied so a downstream caller cannot
mutate the collector's internal audit trail. Finalization requires exactly five
groups and 40 ordered records, then runs the same strict rollout validator used
by atomic publication. A truncated/masked completion is retained as failure
evidence but prevents finalization from passing.

CPU fakes reproduced TRL's latest-eight behavior across five steps. After the
fake trainer retained only step-five completions, the collector still held
step-one record 0, step-four record 31 and step-five record 39 in exact order.
Negative tests covered a duplicate step, wrong SKU, multi-process execution,
missing collector, caller mutation and a truncated completion. The combined
collector/preflight/persistence/handoff suite passed **34 tests** locally and on
Vast with CUDA hidden; the complete local suite passed **257 tests**. No model
was loaded and no optimizer step was run.

Installed TRL source inspection also confirmed the timing assumption behind the
mapping. With the locked `steps_per_generation=1` and `num_iterations=1`,
generation occurs before every optimizer update. Therefore generation while
`global_step=0` belongs to update one, through generation while
`global_step=4` belonging to update five. Any duplicate or skipped mapping now
aborts immediately.

The next gate is orchestration: instantiate this capturing trainer for the
full-smoke-only path, require five completed updates and finalized rollout
evidence, then pass those records and trainer logs through the already tested
save/publish handoff. That path should receive CPU-level control-flow tests
before its CLI is allowed to spend GPU time.

Commit `5753784` implemented and CPU-tested that orchestration boundary. It
accepts an already constructed capturing trainer but remains unreachable from
the command line, so this commit cannot launch the GPU smoke.

The orchestrator now executes the following fail-closed sequence:

1. Refuse an existing final output, a missing collector, collector/SKU-order
   drift, a nonzero starting step, pre-existing optimizer/scheduler state or a
   missing model.
2. Recheck the exact trainable scope and fingerprint the starting live LoRA.
3. Record CUDA state, call the trainer's normal `train()` method once, and
   record CUDA state immediately afterward.
4. Require both trainer state and returned `TrainOutput` to report exactly five
   completed optimizer steps, with optimizer and scheduler now present.
5. Finalize exactly 40 captured rollout records and validate all five complete
   trainer logs before any output directory is created.
6. Recheck trainable tensor/parameter counts and names, prove all 18,464,768
   live LoRA values are finite, and require the final LoRA fingerprint to differ
   from the starting fingerprint.
7. Generalize the existing real bitsandbytes state audit from step one to an
   explicitly required step five, still requiring complete 8-bit state for all
   392 LoRA tensors and no frozen/foreign state.
8. Sample CUDA again after fingerprint/value/optimizer audits and use this
   later peak in the manifest, so audit-time temporary allocations are not
   omitted.
9. Only after every in-memory check passes, create staging and invoke the
   previously tested source-rehash, post-save-disk and atomic-publish handoff.

The end-to-end CPU fake followed the same call shape: five generation groups,
five optimizer steps, five scheduled trainer logs, finite changed LoRA state,
step-five optimizer evidence, adapter/tokenizer save and final atomic
publication. Negative tests proved that a four-step result, only four captured
groups, an unchanged LoRA fingerprint or nonfinite LoRA values all fail before
`save_pretrained()` and before a final or staging output is created. The
generalized optimizer validator also retains the existing one-update behavior
while accepting only `[5]` for the future full smoke.

The combined focused suite passed **40 CPU-only tests** locally and on Vast;
the complete local suite passed **263 tests**. No Torch/TRL model was imported,
no CUDA context was initialized and no optimizer update occurred in this gate.

This proves orchestration logic and failure ordering, not compatibility with a
live `GRPOTrainer`. The next small gate is to construct the real capturing
trainer and connect it to this orchestrator inside a dedicated full-smoke
function while keeping that function unavailable from CLI dispatch. Only after
that construction path passes static/CPU checks should a separately reviewed
CLI flag and remote preflight make GPU execution possible.

Commit `f7cc285` connected that live construction path without exposing it to
CLI dispatch. The module still performs no Torch, TRL, Unsloth, vLLM or PEFT
import when imported. Only a direct call to the new private execution boundary
loads Unsloth first, followed by Torch, bitsandbytes and TRL in the required
patching order.

The dedicated full-smoke gate now:

1. Requires the preflight's exact five-SKU order and an absent final output.
2. Loads the locked adapter through the existing local-only, bf16,
   non-quantized policy loader.
3. Rechecks the 392-tensor/18,464,768-parameter trainable LoRA scope before
   trainer construction.
4. Loads the deterministic five-row pass-rate-0.5 dataset and requires exactly
   the `prompt`, hidden `gold` and hidden `sku_id` columns in manifest order.
5. Requires the three reward functions to retain their exact names and `1:1:2`
   weights before constructing anything expensive.
6. Builds the locked `GRPOConfig` in a disposable trainer-output directory,
   wraps the installed `GRPOTrainer` with the generation-time collector, and
   verifies that the trainer retained dataset order, reward order and weights.
7. Requires global step zero and no optimizer, scheduler or beta-zero reference
   model at the handoff boundary.
8. Re-hashes the source adapter after construction, then passes this exact live
   trainer to the already tested five-step orchestrator.
9. Accepts success only if orchestration reports a completed manifest and the
   final output directory actually exists.
10. Removes the disposable trainer directory, clears only bitsandbytes global
    optimizer overrides added after the gate began, releases model/trainer
    references and records post-release CUDA state.

The timing report now separates model-plus-trainer construction from total gate
time; it does not label training time as construction time. CPU dependency
injection proved the exact real call shape without importing the GPU stack. The
tests also rejected reversed dataset order, source-adapter mutation and renamed
reward functions, and confirmed that `--five-step-smoke` remains an unknown CLI
argument.

The combined construction/orchestration/persistence suite passed **44 CPU-only
tests** locally and on Vast with CUDA hidden. The complete local suite passed
**267 tests**. No real model was loaded, no CUDA context was initialized and no
optimizer update occurred in this gate.

The next gate is the final launch-control boundary: add a mutually exclusive
five-step CLI mode that can only run after the existing preflight, can only
write the reserved atomic bundle, and cannot also write a standalone gate
report. Its parser/dispatch and collision failures should be CPU-tested first;
the actual remote command should still wait for a separate explicit step.

Commit `61c3a48` added that launch-control boundary without launching it. The
new mutually exclusive `--five-step-smoke` mode is the only CLI route to the
full-smoke function and refuses to continue unless all of these conditions are
met before preflight:

- `--expected-commit` is present as an exact 40-character lowercase Git hash;
- output resolves exactly to the reserved
  `runs/grpo-first-smoke` directory;
- the requested preflight disk floor is at least 3 GiB; and
- `--report-file` is absent, because the only accepted evidence output is the
  atomic bundle itself.

The mode remains mutually exclusive with preflight-only, model-load,
trainer-construction, rollout, gradient, optimizer-construction and one-update
gates. After launch-argument validation, it calls the existing CPU preflight,
which still requires a clean tracked worktree/index, exact fixture/selection/
adapter hashes, matching expected commit, collision-free final output and
sufficient free disk. A non-passing preflight now explicitly blocks dispatch to
the GPU gate.

Only after the guarded full-smoke function returns successful publication does
the top-level report mark five training steps, 40 rollout records, initialized
optimizer/scheduler state, final-adapter save and atomic-bundle publication as
true. The full mode cannot produce the older standalone gate-report file, so a
successful run has one canonical evidence location rather than two potentially
divergent records.

CPU tests rejected a missing or uppercase commit, alternate output directory,
2.99 GiB disk floor, standalone report request, mixed mode flags and a failed
preflight. Mocked dispatch proved the order `launch validation → preflight →
five-step gate`, while preserving the no-heavy-import module contract. The
combined focused suite passed **46 tests** locally and on Vast with CUDA hidden;
the complete local suite passed **269 tests**.

No invocation of `--five-step-smoke` occurred in this gate. No model was loaded,
CUDA context initialized, rollout generated or optimizer update performed. The
next small step is a remote **preflight-only** launch-readiness audit at the
fully synced commit: confirm clean tracked Git state, exact source hashes,
reserved-output absence, current disk/GPU state and no stale staging directory.
That audit must still not call the new training flag.

##### Smoke acceptance gates

The smoke passes only if all of the following hold:

1. The code commit, active-dataset hash, smoke-selection hash and locked SFT
   adapter hash match before CUDA initialization.
2. Exactly five optimizer steps and 40 rollout records complete with exit code
   zero.
3. The trainable parameter count is exactly 18,464,768 and only LoRA parameters
   require gradients.
4. At least one of the five groups has nonzero weighted reward standard
   deviation, so at least one relative advantage is nonzero.
5. At least one finite, nonzero gradient norm is logged; loss, rewards,
   advantages and gradient norms contain no NaN or infinity.
6. Each step records all three reward-component means/standard deviations,
   weighted reward, reward standard deviation, zero-variance fraction,
   completion lengths and clipping ratio.
7. Every raw completion is recoverable and re-scores to its stored component
   rewards using the committed CPU reward functions.
8. No OOM occurs; measured peak allocated/reserved VRAM, final idle VRAM and GPU
   temperature are written to the manifest.
9. The starting checkpoint hash remains unchanged, while the saved smoke adapter
   differs from the starting adapter if any nonzero-gradient update occurred.
10. Disk remains above 3 GB and only the declared output paths are created.

Uniform rewards across all five groups, all-zero gradients, malformed hidden
gold, incomplete rollout logging, a cap-hit ratio above zero, parameter-count
drift, an OOM or a source-checkpoint hash change is a failed smoke. The response
is diagnosis and a new tracked smoke attempt, not continuation into a full run.

Even a passing smoke does **not** show that GRPO improves the tagger. It proves
only that the selected SFT policy, prompt pool, rewards and installed trainer can
perform real policy updates safely enough to justify designing the longer run.

### Questions to answer before the first GRPO run

- [x] What fraction survives? 1,702/3,600 eligible; 1,565 active after cap four.
- [x] Is the pool diverse? All major categories, all 14 stores and all 1,150
  eligible families are represented; measured skews are documented.
- [x] Are there enough mixed groups? Yes: 1,565 active prompts, each selected
  from the predeclared mixed-outcome band.
- [x] Should the first GRPO reward remain binary or use multiple reward
  components? Three separately logged binary components, weighted `1:1:2`, are
  proposed for the first unconstrained run.
- [x] How will gold-unknown fields be handled consistently in the reward
  function? Exclude them from golden agreement, retain the existing abstention
  telemetry and report the stricter unknown-aware metric separately.
- What checkpoint frequency fits the remaining disk budget? Smoke policy is
  locked to no intermediate checkpoint and one final model-only adapter; the
  full-run interval remains pending measured smoke footprint and runtime.
- [x] What is the smallest GRPO smoke that proves reward variance, nonzero
  gradients and stable memory? Five deterministic prompts, eight completions per
  prompt and five optimizer steps, with the ten acceptance gates above.
- [x] Which frozen metrics determine whether GRPO beats or harms SFT? The locked
  comparison below uses macro-F1 as the primary quality result plus selective
  macro-F1, coverage, schema/vocabulary validity, rule violations and missing
  predictions as safety and behavior constraints.

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

### Gradient-gate launch attempt: package entrypoint

The first gradient-gate command used
`/venv/rl/bin/python training/train_grpo.py`. It exited before model loading
with `ModuleNotFoundError: No module named 'training'` because executing the file
path puts `training/`, rather than the repository root, first on Python's module
search path while the entrypoint imports `training.dataset`. The pre-existing
documented project convention is module execution. The corrected command used
`/venv/rl/bin/python -m training.train_grpo` from the repository root. Before
retrying, collision checks confirmed that neither the report artifact nor the
reserved training output existed. The failed launch created no model, optimizer,
checkpoint or evidence file.

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
- [x] GRPO reward specification and CPU-only callback tests.
- [x] Remote reward callback and patched TRL import integration probe.
- [x] Deterministic five-row smoke fixture, manifest and rebuild proof.
- [x] CPU-only fail-closed GRPO preflight and synthetic negative tests.
- [x] Real-adapter GRPO preflight on the Vast training environment.
- [x] Real locked-adapter model load and runtime trainability assertions.
- [x] Real GRPO trainer construction with no optimizer, generation or training.
- [x] One real eight-completion rollout with rewards, variance and unchanged LoRA.
- [x] One-group real GRPO loss/backward evidence with complete finite gradients,
  unchanged LoRA weights and no optimizer.
- [x] Real 8-bit AdamW construction with exact LoRA scope, explicit
  hyperparameters, lazy-state proof and unchanged weights.
- [x] Real trainer-loop update at nonzero LR with complete 8-bit state, finite
  all-LoRA parameter deltas and unchanged source checkpoint.
- [x] Atomic five-step smoke bundle contract with exact adapter allowlist,
  collision checks, disk bounds and CPU-only negative tests.
- [x] CPU-tested live-adapter save and atomic-publisher handoff with source
  re-hashing and post-save disk measurement.
- [x] Generation-time collector preserving all five ordered rollout groups
  before TRL's latest-eight deque overwrites earlier evidence.
- [x] CPU-only end-to-end five-step orchestration with fail-before-save negative
  tests, finite-LoRA audit and exact step-five optimizer-state requirement.
- [x] Live Unsloth/TRL construction path connected to guarded orchestration but
  deliberately unavailable from CLI dispatch.
- [x] Mutually exclusive, commit-locked five-step CLI launch control with
  reserved output, 3 GiB floor, preflight-before-GPU dispatch and CPU tests.
- [ ] Five-step GRPO smoke with optimizer construction and real parameter
  updates.
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
