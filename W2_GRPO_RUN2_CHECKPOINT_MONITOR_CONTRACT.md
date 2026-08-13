# GRPO Run 2 checkpoint-monitor contract

**Version:** `grpo-run2-checkpoint-monitor-contract-v1`

**Status:** locked before Run 2 checkpoint generation or evaluation

**Purpose:** detect checkpoint regression, retention loss, and sampled-versus-
greedy divergence using development data without opening final confirmation

## Population and roles

The monitor uses exactly the authoritative 360-row SFT validation population in
the order recorded by `grpo-run2-data-role-manifest-v1`. It reports these fixed
views without resampling:

- representative: all 360 rows;
- difficult: starting-SFT whole-record pass count 0–2 of 8, 204 rows;
- middle: pass count 3–5 of 8, 46 rows;
- easy retention: pass count 6–8 of 8, 110 rows.

The development population has zero SKU and normalized-family overlap with the
corrected 1,438-row Run 2 training pool. Its known limitations remain: prior SFT
checkpoint use and 21 rows historically seen by GRPO Run 1. It can guide Run 2
checkpoint decisions but cannot support final confirmation claims.

The monitor must not read any path under `data/confirmation_run2_v1`, any
confirmation role successor, or the legacy exposed frozen 300.

## Decoding

Every production checkpoint uses the same prompt order and generation limits:

| mode | repetitions | sampling | temperature | top-p | seeds |
|---|---:|---|---:|---:|---|
| greedy | 1 | disabled | n/a | n/a | deterministic |
| sampled | 8 | enabled | 0.7 | 0.95 | 20260813–20260820 |

The sampled settings match GRPO training. Eight repetitions match the training
group size and provide a distribution rather than a single lucky sample.
Maximum prompt length is 600 tokens and maximum completion length is 170. Batch
size is fixed at 8 so a seed maps to one reproducible generation schedule.

The bounded GPU smoke may use four predeclared development SKUs, two sampled
repetitions, and the same temperature, top-p, prompt and completion limits. This
is a machinery proof, not baseline variability or model-quality evidence.

## Measurements

For greedy output and for each sampled repetition, report on every fixed view:

- primary-evaluator macro-F1 and selective macro-F1;
- coverage;
- schema and vocabulary validity over all attempted rows;
- rule-violation count and rate;
- original `1:1:2` reward mean and distribution;
- selected dense Candidate UA reward mean and distribution.

For sampled decoding, report mean, population standard deviation, minimum,
maximum, and all eight repetition values for every scalar. Raw literal model
outputs are retained so format failures remain measurable.

An unparseable row is explicitly counted and makes primary quality metrics
conditional on parseable outputs. It must never disappear from the attempted
denominator. Guardrails must include validity, not read survivor-only F1 alone.

## Runtime and publication

Checkpoint evaluation is synchronous after a retained checkpoint save. The
supervisor has a per-checkpoint timeout, captures stdout/stderr, terminates and
then kills an unresponsive evaluator, and exclusively publishes a failure
artifact. Any evaluator failure aborts training; quality-based aborting remains
disabled until baseline variability is measured and a material/repeated rule is
predeclared in the Phase G run contract.

A successful monitor bundle is collision-protected and atomic. It contains raw
greedy and sampled JSONL, a scored report, resource/timing evidence, and a
manifest binding the checkpoint bytes, contract, data, pack, reward code, Git
commit, command, and exact file identities.

The evaluator loads one base model plus one PEFT adapter, uses the same model for
greedy and sampled generation, releases all model/tokenizer references, runs
garbage collection and empties the CUDA cache before publishing success.

## Gate

Phase F passes only when:

1. CPU tests cover population drift, output pairing, all metrics/rewards,
   sampled aggregation, collision safety, timeout/failure publication and
   checkpoint callback ordering;
2. one bounded RTX 3090 smoke creates a checkpoint, evaluates greedy and
   repeated sampled outputs, proves cleanup, and atomically publishes an
   auditable bundle;
3. the smoke is clearly marked non-quality evidence;
4. no confirmation data, full GRPO training, or quality abort threshold is
   opened by this gate.
