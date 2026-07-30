# Draft brief — teaching a small model to tag a catalog before asking it to learn from reward

**Status:** pre-training draft. This document describes the design, constraints, and planned comparison for Week 2. It does not claim SFT or GRPO results yet.

## The post in one sentence

Before using GRPO to improve a catalog tagger, establish an honest LoRA-SFT baseline, make the model's output contract mechanically verifiable, and compare small adaptation choices without quietly using the final test set as a tuning knob.

## Why this project exists

The project is a controlled catalog-tagging task for apparel listings. Given text from a product page—title, brand, category, description, and merchant tags—the model must emit one JSON object with 15 controlled fields, such as garment category, material, fit, pattern, closure, and occasion.

The broader project is built around one idea: the same verifier should serve two consumers.

1. During RL, it provides a reward signal for a generated answer.
2. In the production path, it acts as a QA gate before a catalog record is accepted.

That means “correct” is not a fuzzy prompt instruction. It has a concrete definition in code: valid JSON, expected keys and shapes, controlled vocabulary compliance, and declarative cross-field rules.

## What was completed before Week 2

Week 1 created the measuring instrument that makes an RL comparison meaningful.

- A pack-agnostic schema/verifier exists under `verifier/`.
- The apparel pack has 15 fields, 156 controlled values, and 34 cross-field rules (25 explicit, 9 derived from applicability metadata).
- The data contains 3,600 weakly labeled training rows, a frozen 300-row evaluation set, and a 100-row production-style probe set.
- The evaluation harness reports per-attribute metrics, macro-F1, schema validity, vocabulary validity, and rule violations.
- The evaluation set is checksummed and tagged `eval-v1`, so accidental edits fail the harness rather than silently changing the scoreboard.

One design choice matters especially: `null` and `"unknown"` are different.

- `null` means a field does not apply. A pair of pants has no neckline.
- `"unknown"` means the field could apply, but the listing does not provide enough evidence. A listing may omit the color.

This difference will matter later because abstention is a first-class RL action. A model should not receive credit for pretending a missing fact is inapplicable.

## An actual SFT training example

For a listing titled “Organic Cotton Pull On Pant,” the model receives a compact system instruction plus text such as:

```text
Title: Organic Cotton Pull On Pant
Brand: NAADAM
Category: Woven Pants
Description: The Organic Cotton Pull On Pant is a warm-weather essential in
100% organic cotton ... the easy elastic waist makes every occasion feel effortless.
Tags: 100% cotton, Bottoms, elastic, full length, pant, summer, ...
```

Its target is one JSON record:

```json
{
  "closure": "pullover",
  "collar_type": null,
  "colour_primary": "unknown",
  "details": ["unknown"],
  "fit": "loose",
  "garment_category": "pants",
  "material": "cotton",
  "occasion": "casual",
  "sleeve_length": null
}
```

The real target contains all 15 fields. The verifier confirms that this exact output is structurally valid, uses only permitted vocabulary, and violates no rules.

This is supervised fine-tuning in its plainest form: show the model many examples of “catalog text in, controlled JSON out.” The first objective is not to make a grand claim about intelligence. It is to teach a small model the format, vocabulary, and task behavior that RL will later refine.

## A necessary warning about labels and evaluation

The training labels are weak labels from a frontier model, not ground truth. The frozen evaluation set was partially human-corrected, but only 78 of 4,500 attribute cells received human review. Most remaining evaluation cells still equal the frontier model’s original answer.

That creates two constraints on how results should be written:

- Absolute scores are not trustworthy claims of real-world accuracy. A “99%” agreement number would be partly tautological.
- Deltas between models scored against the same frozen set are still useful. If GRPO improves over SFT under the same evaluator, that comparison is meaningful even if the absolute number has caveats.

The reward-reliability file is deliberately marked unusable: the human review sample is too small to support per-attribute reward weights. The first reward implementation must detect that flag and use uniform weights rather than accidentally assigning zero reward everywhere.

## The SFT experiment: two LoRA arms

The first experiment will compare two ways of adapting the same Qwen2.5-1.5B-Instruct base model.

| Arm | LoRA targets | Rank | Purpose |
|---|---|---:|---|
| A | Attention projections: Q, K, V, output | 16 | Cheap baseline that adjusts how the model connects information across tokens. |
| B | Attention projections plus MLP projections: gate, up, down | 16 | More adaptable baseline that can change both information routing and internal transformations. |

Everything else stays constant: base model, training data, deterministic split, seed, batch configuration, learning rate, sequence budget, and maximum epochs. This is an ablation, not a scavenger hunt for the best-looking number.

The stronger SFT arm will become the starting policy for GRPO.

## LoRA in plain language

Full fine-tuning changes every number in the original model. LoRA freezes the base model and adds small trainable adjustment paths beside selected layers.

For a typical 1,536-by-1,536 model layer, full fine-tuning would change about 2.36 million numbers. A rank-16 LoRA update instead learns two thin tables:

- 1,536 numbers wide down to 16;
- 16 numbers back up to 1,536.

Together, that is about 49,000 trainable numbers for that layer—about 2% of the full layer.

The 1,536 width is part of Qwen’s original architecture. It is the size of the internal representation carried by each token through much of the model. The rank 16 is our choice: it determines the size of LoRA’s narrow adjustment path.

The model computes both paths and adds them:

```text
original frozen layer output + small learned LoRA adjustment = layer output
```

For this Qwen configuration, rank-16 LoRA on attention and MLP projections is estimated at about 18.5 million trainable parameters. In bf16, an adapter-only checkpoint should be roughly 37 MB. Attention-only LoRA is much smaller, around 9 MB. These are estimates until the exact target module list is fixed.

## Preventing product variants from leaking into validation

A naïve random split would make validation look better than it is. Retail feeds
often contain many versions of nearly the same product:

```text
Everyday Tote | Red (XL)
Everyday Tote | Black (M)
Everyday Tote | Navy (L)
```

If the red tote trains the model while the black tote tests it, the model has
already seen nearly the same title, description, category, and expected tags.
That measures recognition of a sibling product more than generalization.

The split therefore operates on product families, not individual rows. A family
key combines the normalized brand with a conservative base title:

- lowercase the title;
- remove punctuation and extra whitespace;
- remove variant text after spaced separators such as ` | `, ` / `, or ` - `;
- remove trailing parenthetical size text;
- compare titles only within the same normalized brand.

For example, all 57 color and size versions of Thursday Boots' `Everyday Tote`
become one family:

```text
Thursday Boots + everyday tote
```

Likewise, these American Giant products become one 22-row family:

```text
Women's Classic Cotton V-Neck Tee - Black
Women's Classic Cotton V-Neck Tee - Pearl Blue
Women's Classic Cotton V-Neck Tee - Pine Bark

→ American Giant + womens classic cotton v neck tee
```

A unique listing such as `NAADAM + Organic Cotton Pull On Pant` has no matching
variants and forms a one-row family.

The 3,600 rows break down as:

```text
1,764 one-product families       → 1,764 rows
  491 multi-product families     → 1,836 rows
---------------------------------------------
2,255 total product families     → 3,600 rows
```

Every family is assigned wholly to training or validation. The method is
deliberately conservative rather than broad fuzzy matching: two brands can both
sell a “Classic Tee” without those products being treated as the same family.

Grouping alone was not enough. The first deterministic family split put only
3.4% of bag rows and 7.1% of shoe rows into validation, so it was rejected before
training. The final split assigns whole families while targeting roughly 10% of
each garment category. It contains 3,240 training rows and 360 validation rows;
the major categories now land near the target—for example, shoes at 10.0%, bags
at 9.4%, dresses at 10.1%, and pants at 10.1%.

The frozen split manifest records the seed, grouping rule, source-data checksum,
category counts, and exact SKU assignments. Both LoRA arms must reuse it.

## How long should SFT train?

The answer is not “a magic number of steps.” We will use a small held-out validation slice from the weak training data to choose duration.

- 90% of the 3,600 weak rows train the model.
- 10% are held out as validation data through the frozen grouped-family split.
- The frozen 300-row evaluation set is not used to choose an epoch.
- Each arm trains for one epoch first, then has the option of a second epoch.

After each epoch, inspect weak-validation loss and generated-output behavior: complete JSON, vocabulary/rule compliance, and whether outputs are becoming more useful rather than merely repetitive. Continue to epoch two only if there is meaningful improvement.

The frozen evaluation set is then run once per pre-declared arm, and both results are reported. It is not a hidden tuning loop.

The goal of SFT is not necessarily the final maximum score. For GRPO, the useful starting policy is one that succeeds on some examples and fails on others. If every sampled completion is always right or always wrong, GRPO sees no within-group difference to learn from.

## Practical constraints that shape the experiment

The training machine is an RTX 3090 with 24 GB of VRAM and roughly 5.3 GB of free disk space.

- The selected Qwen2.5-1.5B-Instruct checkpoint is already cached on the machine.
- Measured training-data lengths: prompt p95 is 268 tokens, prompt maximum is 585, and target maximum is 118.
- The fully rendered prompt-plus-target maximum is 833 tokens, so SFT uses a 896-token ceiling; the earlier 768-token estimate would have clipped seven rows.
- GRPO will use separately reserved limits: 600 prompt tokens and 170 completion tokens.
- Only two checkpoints per experiment arm should be retained. Adapter-only saves preserve the learned LoRA changes for evaluation without storing expensive optimizer state.

An optimizer is the training component that decides how each learned number should change after an error. Its saved state is not part of Qwen’s attention architecture; it is extra per-parameter bookkeeping. Omitting it means a run cannot resume with exactly the same training momentum, but the saved adapter still loads and evaluates normally. For this short two-epoch baseline, adapter checkpoints plus experiment metadata are the more useful trade-off.

### What the five-step attention-only smoke test actually proved

This was an integration test, not a quality experiment. Its purpose was to prove
that the model, grouped data split, LoRA setup, optimizer, validation pass, and
adapter save all work together before paying for a full epoch.

The smoke run used 64 rows from the frozen training split and 32 from validation.
The GPU processed four examples at a time and accumulated four micro-batches
before each optimizer update, giving an effective batch size of 16:

```text
5 optimizer steps × 16 examples = 80 example-visits
```

Because the smoke subset contains only 64 training rows, the fifth update begins
a second pass. The run therefore ended at 1.25 smoke-subset epochs. These small
subsets are not guaranteed to represent the full data distribution; they exist
to test the machinery.

#### Where 4,358,144 trainable parameters came from

Qwen has 28 transformer blocks. Attention-only rank-16 LoRA attaches to four
projections in every block:

| projection | LoRA parameters per block |
|---|---:|
| Q | 49,152 |
| K | 28,672 |
| V | 28,672 |
| attention output | 49,152 |
| **total** | **155,648** |

Across all blocks:

```text
155,648 × 28 = 4,358,144 trainable LoRA parameters
```

The loaded model, including its adapters, contained 1,548,072,448 parameters:

```text
4,358,144 / 1,548,072,448 × 100 = 0.2815%
```

So the run changed only 0.28% of the model; the other 99.72% remained frozen.
Unsloth independently printed both counts when it constructed the model.

#### What “five optimizer steps completed” means

The trainer reached 100%, ran validation, exited successfully, and saved the
adapter. Its progress matched the expected effective batch size:

```text
step 1 → 0.25 smoke-subset epoch
step 2 → 0.50
step 3 → 0.75
step 4 → 1.00
step 5 → 1.25
```

This confirms that micro-batching and gradient accumulation behaved as intended.

#### Why the gradients looked healthy

The gradient norm measures the combined strength of the proposed updates to the
trainable parameters.

| step | gradient norm |
|---:|---:|
| 1 | 0.541 |
| 2 | 0.574 |
| 3 | 0.592 |
| 4 | 0.606 |
| 5 | 0.568 |

All five were nonzero, so a learning signal reached LoRA. They were finite—no
`NaN` or infinity—and stayed in a narrow 0.54–0.61 range below the default
clipping threshold of 1.0. In this context, “healthy” means no dead, exploding,
or numerically broken gradients appeared in the short run. Five steps cannot
prove that gradients will remain healthy for a full epoch.

#### How to read the training and validation losses

The five training losses were:

```text
0.4419, 0.5103, 0.4988, 0.4640, 0.4749
```

The trainer reported their average as `0.477982`. Loss here measures how
surprised Qwen was by each correct next JSON token, calculated only on assistant
answer tokens—not the system or product prompt.

The values should not descend smoothly because each update contains different
products, and five updates are too few to establish a trend.

After step five, the trainer evaluated 32 validation rows in eight batches of
four and reported `eval_loss = 0.460474`. That confirms the validation pipeline
works; it does not prove generalization. There was only one measurement, the
sample was tiny, and predictable JSON punctuation and field names can lower
token loss without making the tags correct. The full run will evaluate all 360
validation rows after each epoch.

#### GPU memory: what was and was not measured

The run completed without an out-of-memory error, CUDA crash, or numerical
failure, and GPU use returned to ComfyUI's 428 MiB idle footprint afterward.
That proves this configuration fits on the RTX 3090 for the smoke test.

Peak VRAM was not recorded, so it would be inaccurate to claim an exact amount
of remaining headroom. The full run should log peak allocated and reserved GPU
memory.

#### Why the adapter was 17.5 MB instead of the estimated 9 MB

The saved adapter weight file was exactly 17,462,432 bytes. The measured size
follows directly from the trainable parameter count:

```text
4,358,144 parameters × 4 bytes per fp32 number ≈ 17.43 MB
```

The small remainder is file metadata. The earlier ~9 MB estimate assumed the
adapter would be stored in bf16 at two bytes per number. PEFT saved these adapter
tensors in fp32, doubling that estimate.

The complete adapter directory occupied about 32 MiB because it also contained
the tokenizer:

```text
adapter weights        17.5 MB
tokenizer.json         11.4 MB
vocab.json              2.8 MB
merges.txt              1.7 MB
small configuration files
```

Disk tools mix decimal MB and binary MiB, so the displayed directory total does
not add up identically to the decimal file sizes. The machine still reported
5.3 GB free before and after the smoke run; a 32 MB artifact is too small to
change that rounded display.

#### W&B status

The smoke deliberately used local logging with `report_to=none`. Separately, the
W0 run log shows that this machine loaded W&B credentials, authenticated, created
a run, and synchronized it successfully. Both full attention-only runs later
confirmed the same path: W&B authenticated from `/root/.netrc` and synchronized
their metrics without exposing credentials in the repository.

### Attention-only SFT: epoch 1

The first full attention-only epoch ran against Git commit `8a781e6`. It used all
3,240 training rows, evaluated all 360 held-out weak-validation rows, and made
203 optimizer updates. With four products on the GPU at once and four
micro-batches accumulated per update, each update represented 16 products.

Training completed in 269 seconds—about four minutes and 29 seconds—without an
out-of-memory or CUDA error. Peak allocated GPU memory was 4.75 GiB and peak
reserved memory was 4.79 GiB. The disk had 5.17 GiB free afterward. W&B synced
the run as `vmmfmi7y`.

The average training loss across the epoch was `0.12625`. The logged loss fell
from `0.5245` at the first update to roughly `0.06` near the end. Gradient norms
remained finite and nonzero; sampled logs ranged from about `0.19` to `0.61`,
with no numerical failure. The all-row validation loss was `0.06236`.

Those token losses are useful health signals, but they do not by themselves
prove that the generated tags are correct. We therefore loaded the saved LoRA
adapter and generated answers for the exact 360 validation SKUs named by the
checksum-bound split manifest. Sampling was disabled.

The generated-output results were:

| measure | epoch-1 result |
|---|---:|
| attempted products | 360 |
| parseable, schema-valid records | 358 / 360 (99.4%) |
| completely vocabulary-valid records | 243 / 358 (67.9%) |
| rule violations | 25 |
| coverage | 96.5% |
| macro-F1 | 0.650 |
| selective macro-F1 | 0.679 |

This is weak-validation performance, not the final frozen 300-row evaluation.
It is for choosing training duration and the LoRA arm; reporting it as final
test performance would leak model-selection decisions into the test set.

The most important gap after one epoch is no longer JSON syntax. Only two
outputs failed to parse. The larger issue is controlled vocabulary: the model
occasionally emitted plausible human terms that the catalog contract does not
allow. Examples included:

```text
garment_length: "full"   (not in the permitted garment-length list)
material:       "woven"  (not in the permitted material list)
pattern:        "patchwork" (not in the permitted pattern list)
```

The most common invalid-value fields were material (23 records), details (20),
garment length (17), neckline (17), and pattern (15). A single record can fail
more than one field, so those counts should not be added to get a row count.

The two syntax failures were also informative. One answer began with stray
vocabulary-like prose before eventually producing JSON; another emitted the
bare token `unknown` without JSON quotes. Epoch 1 has therefore learned the
output shape extremely well, but has not fully learned that every value must
come from a closed list.

The per-attribute macro-F1 range exposed where correctness remains uneven.
`fit` was strongest at 0.898, followed by `neckline` at 0.838 and
`garment_category` at 0.777. `closure` was weakest at 0.470, followed by
`sleeve_style` at 0.486, `occasion` at 0.516, and `waistline` at 0.523. These
weak labels are not fully human-trusted, so the numbers are best used as
comparative diagnostics rather than absolute truth.

The evidence supports testing epoch 2: vocabulary compliance still has
substantial room to improve, the first epoch remained numerically stable, and
the run is cheap enough to continue. We should still keep the epoch-1 adapter
and generated predictions. Epoch 2 earns selection only if generated-output
metrics improve rather than merely producing a lower token loss.

### Why “continue to epoch 2” required a fresh two-epoch run

The one-epoch adapter was saved with `save_only_model=True`. That preserved the
LoRA weights for inference, but not AdamW's internal moving averages or enough
state to resume the optimizer exactly. Loading that adapter and training for one
more epoch would reset the optimizer and learning-rate schedule. It would be a
new training phase, not a faithful continuation.

There was a second subtlety. The cosine learning-rate schedule is calculated
from the total planned number of updates. In the one-epoch run it was designed
to reach zero at step 203. In a two-epoch run, step 203 is only the midpoint, so
the learning rate is still above zero. Comparing the original one-epoch adapter
directly with the end of a two-epoch run would mix together two effects:

1. seeing the data for a second time; and
2. following a different learning-rate schedule during the first pass.

We therefore launched a clean two-epoch experiment from the same Qwen base
model, frozen split, LoRA configuration, and seed. It ran continuously for 406
optimizer updates and saved checkpoints at both step 203 and step 406. Comparing
those two checkpoints isolates what the second pass added within one optimizer
and one schedule.

The new run used its own output directory,
`runs/sft-attention-2epoch`, so it did not overwrite the original epoch-1
artifacts.

### Attention-only SFT: continuous two-epoch run

The continuous run used the same 4,358,144 attention-only LoRA parameters,
representing 0.28% of Qwen's 1.548 billion parameters. The data and effective
batch size were unchanged:

```text
training products                 3,240
validation products                 360
products loaded on GPU at once        4
gradient accumulation                 4
effective batch size                  16
optimizer updates per epoch          203
total optimizer updates              406
```

Training took 530 seconds—about eight minutes and 50 seconds. The trainer itself
reported 528 seconds, 12.27 samples per second, and 0.768 optimizer steps per
second. Peak allocated GPU memory was 4.75 GiB and peak reserved memory was
4.79 GiB, essentially identical to the one-epoch run. No CUDA out-of-memory,
`NaN`, infinity, or training crash occurred. The box still had 5.08 GiB free
after saving two epoch checkpoints and the final adapter.

The aggregate training loss over both passes was `0.08482`. Near the start it
was `0.5245`; near the end of the second pass, logged ten-step losses were
roughly `0.036–0.041`. Gradient norms remained finite and nonzero. During the
second pass, representative logged values ranged from about `0.19` to `0.45`,
so a usable learning signal continued reaching LoRA even as the cosine learning
rate approached zero.

Validation token loss improved at the epoch boundary:

| checkpoint | validation loss |
|---|---:|
| step 203, after pass 1 | 0.05013 |
| step 406, after pass 2 | 0.04202 |

The run synchronized to W&B as
[`iwsrgsn2`](https://wandb.ai/rushabhsp95-vastraa/tagging-rl/runs/iwsrgsn2).
Its display name remained `sft-attention`; the unique run ID is the unambiguous
reference.

### The fair epoch-1 versus epoch-2 generation test

Loss was not the selection criterion by itself. We separately loaded
`checkpoint-203` and `checkpoint-406`, then generated unconstrained answers for
the same 360 held-out SKUs. Both generations used the same system prompt, input
length, 170-token output ceiling, batch size of eight, greedy decoding with
sampling disabled, and checksum-verified validation manifest.

| generated-output measure | step 203 | step 406 | change |
|---|---:|---:|---:|
| schema-valid records | 359/360 (99.7%) | 359/360 (99.7%) | unchanged |
| fully vocabulary-valid records | 280/359 (78.0%) | 323/359 (90.0%) | +43 records |
| rule violations | 19 | 15 | -4 |
| coverage | 95.2% | 97.1% | +1.9 points |
| macro-F1 | 0.708 | 0.751 | +0.043 |
| selective macro-F1 | 0.728 | 0.777 | +0.049 |

This is the load-bearing comparison. The second pass did not merely lower token
loss: it produced more useful answers, obeyed the closed vocabulary much more
often, broke fewer cross-field rules, and raised both headline and selective
macro-F1. Schema validity was already nearly saturated, so remaining flat at
99.7% is expected rather than a failure.

The original standalone one-epoch run scored 0.650 macro-F1 and 67.9%
vocabulary validity. The two-epoch final scored 0.751 and 90.0%, respectively.
That larger difference is useful operational evidence, but it is not the clean
measure of the second epoch because the runs used different cosine horizons.
The step-203 versus step-406 comparison above is the defensible duration
ablation.

### Where the second epoch helped

Twelve of the 15 attribute macro-F1 scores improved between the same-run
checkpoints. The largest changes were:

| attribute | step 203 | step 406 | change |
|---|---:|---:|---:|
| closure | 0.285 | 0.549 | +0.264 |
| pattern | 0.757 | 0.919 | +0.162 |
| sleeve length | 0.793 | 0.870 | +0.077 |
| silhouette | 0.562 | 0.610 | +0.048 |
| fit | 0.876 | 0.904 | +0.028 |
| material | 0.804 | 0.827 | +0.022 |

`waistline` was unchanged at 0.518. `garment_category` moved slightly from
0.874 to 0.868, and `occasion` from 0.614 to 0.609. Those small regressions are
why we inspect per-field results rather than hiding everything behind one
average. They do not outweigh the broad improvements, especially the 12-point
gain in whole-record vocabulary validity.

At the final checkpoint, `neckline` was strongest at 0.988 macro-F1, followed
by `pattern` at 0.919, `fit` at 0.904, `sleeve_length` at 0.870, and
`garment_category` at 0.868. The weakest fields remained `waistline` at 0.518,
`closure` at 0.549, `details` at 0.596, `occasion` at 0.609, and `silhouette`
at 0.610.

These scores are against weak validation labels, not fully reviewed human
truth. They are suitable for comparing checkpoints trained and measured under
the same protocol, but not for claiming absolute production accuracy.

### What epoch 2 still did not solve

One SKU failed schema validation at both same-run checkpoints:
`shopify:naadam.co:6619044577376`. Instead of JSON, it emitted a long,
comma-separated vocabulary-like sequence beginning with terms such as
`xlarge`, `xl`, `yeezy`, and `zipped`, then repeated `unknown` until the output
limit. This looks like a stubborn generation loop or memorized vocabulary
fragment, not ordinary JSON punctuation damage.

Among the 36 parsed records that were not completely vocabulary-valid, the
remaining invalid-value fields were:

```text
neckline           8
closure            7
details            7
material           5
pattern            3
silhouette         3
garment_category   2
occasion           2
sleeve_style       1
```

A record may appear in more than one field count. Representative invented
values included:

```text
material:          "woven"
pattern:           "patchwork"
closure:           "snap"
neckline:          "open" or "asymmetric"
garment_category:  "scarf"
silhouette:        "popovers"
details:           ["gather"]
```

These are often understandable English descriptions, but they violate the
operational contract because downstream systems expect the exact controlled
vocabulary. That distinction—semantically plausible versus contract-valid—is
one of the reasons this task is a useful RL testbed.

The final output also contained 15 rule violations:

```text
solid_is_not_multicolour             5
athletic_material_subset             2
dress_needs_length_and_neckline      2
auto applies-to violations           6
```

The model has therefore learned the JSON shell and most individual labels
before fully learning every relationship among fields. That remaining gap is a
natural target for verifier-based rewards later.

### Epoch-duration decision

Epoch 2 is selected as the attention-only SFT checkpoint. It earned selection
on held-out generated-output behavior, not because “two is more than one” or
because training loss kept falling.

This does **not** yet select attention-only LoRA as the final SFT arm. The
combined attention-plus-MLP arm still needs the same controlled training and
validation protocol. Only after comparing the two selected-duration arms should
one become the baseline handed to GRPO.

## The RL handoff: GRPO

GRPO comes only after a defensible SFT baseline exists.

The planned reward is built as ordinary functions over the same verifier used by tests. It begins with three signals:

1. Format validity: did the model emit parseable JSON with the expected structure?
2. Vocabulary and rule compliance: did it stay inside the allowed tag space and obey cross-field constraints?
3. Per-attribute agreement with the training row’s label: did it provide useful tags rather than just valid empty JSON?

The model must generate unconstrained during RL. Constrained JSON decoding would make format validity free and erase the very behavior the experiment is intended to observe. The point is to watch the model learn not to emit malformed JSON, empty-but-valid records, majority-class defaults, or blanket `unknown` answers.

The expected reward-hacking story is part of the deliverable, not an embarrassment to hide. The first GRPO runs should reveal shortcuts; later reward shaping should close them with per-attribute partial credit, class balancing, and validity acting as a gate rather than the whole objective.

## Current blocker before GRPO

The GPU machine can import the SFT components, but its installed GRPO stack is internally inconsistent: `trl==0.24.0` expects an older vLLM API while the machine has `vllm==0.23.0`. `GRPOTrainer` currently fails at import time because the expected `GuidedDecodingParams` symbol is absent.

This has to be resolved before any GRPO run. It does not prevent the conceptual work or the eventual SFT baseline, but it must be treated as a deliberate environment preflight rather than discovered after spending time on training.

## What the final post should be able to say

If the experiment lands cleanly, the finished version can make a modest, defensible claim:

> On a frozen catalog-tagging evaluation set, we compared attention-only and combined LoRA SFT baselines, then used the selected baseline to study how verifier-based GRPO changes structured-output behavior. We report the improvements and the failure modes, not a misleading absolute-accuracy claim.

The interesting story is not “RL makes a number go up.” It is how a small model learns an operational output contract, how the verifier exposes shortcuts, and why baseline discipline is what makes an RL result worth believing.

## Evidence to add once runs exist

- Exact Git commit hash for the selected final arm.
- W&B charts for the combined LoRA arm; the attention-only run is `iwsrgsn2`.
- Frozen-eval table: macro-F1, schema validity, vocabulary validity, rule violations, and cost per SKU.
- Example generations from both arms, including failures.
- GRPO reward curves and at least three documented reward hacks.
- The dependency-resolution decision for the GRPO/vLLM environment.
