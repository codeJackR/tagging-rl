# W2 GRPO Run 2: diagnosis-to-experiment plan

**Status:** active execution plan
**Created:** 2026-08-11
**Working style:** execute one small, auditable step at a time; do not advance to
GPU training until every CPU-only gate below passes.
**Run 1 diagnosis:** [`W2_GRPO_RUN1_DIAGNOSIS.md`](W2_GRPO_RUN1_DIAGNOSIS.md)
**Run 2 blog tracker:** [`W2_GRPO_RUN2_BLOG_TRACKER.md`](W2_GRPO_RUN2_BLOG_TRACKER.md)
**Reward payoff contract:** [`W2_GRPO_RUN2_REWARD_PAYOFFS.md`](W2_GRPO_RUN2_REWARD_PAYOFFS.md)
**Reward scale contract:** [`W2_GRPO_RUN2_REWARD_SCALES.md`](W2_GRPO_RUN2_REWARD_SCALES.md)
**Earlier technical tracker:** [`W2_GRPO_BLOG_TRACKER.md`](W2_GRPO_BLOG_TRACKER.md)

## 1. Goal and stopping rule

The goal is not to tune GRPO indefinitely. It is to run one disciplined
corrective experiment cycle that tests the strongest Run 1 diagnosis:

1. the original reward provided sparse, mostly whole-record signal;
2. clean wrong commitments were usually not penalized relative to abstention;
3. `beta=0` left weakly rewarded behavior without a KL anchor;
4. sampled training behavior was not measured against fixed greedy and sampled
   validation prompts.

W2 is complete when either:

- a predeclared GRPO recipe beats the locked SFT baseline on a new untouched
  confirmation set without unacceptable coverage or rule regressions; or
- one clean corrective cycle fails and the result can be explained with
  hash-pinned evidence and clearly bounded claims.

Do not add more reward or optimizer sweeps merely to find a positive result.

## 2. Confirmed state entering this plan

### 2.1 Run 1 result

- Starting checkpoint: `runs/sft-combined-2epoch/checkpoint-406`.
- Training: 300 steps, one prompt and eight completions per step.
- Reward: binary format, compliance and whole-record agreement weighted
  `1:1:2`.
- KL: `beta=0`.
- Frozen macro-F1: SFT `0.6411`, GRPO `0.6223`, delta `-0.0188`.
- Frozen selective macro-F1 delta: `-0.0591`.
- Frozen coverage delta: `+2.47` percentage points.
- Rule violations: `12 -> 28`.
- Training groups with total reward variation: `202/300`.
- Training groups with zero total reward variation: `98/300` (`32.7%`).
- Component variation across the 300 groups:
  - format: `0` groups;
  - compliance: `33` groups;
  - golden agreement: `199` groups;
  - weighted total: `202` groups.

The frozen 300-row set has been inspected during diagnosis and is no longer an
untouched model-selection or confirmatory set. It remains a disclosed legacy
benchmark only.

### 2.2 Newly confirmed split-boundary defect

The authoritative SFT manifest at `data/splits/sft-v1.json` contains 3,240
training SKUs and 360 validation SKUs. The existing cap-four GRPO pool contains:

| collection | total | SFT-train SKUs | SFT-validation SKUs |
|---|---:|---:|---:|
| cap-four GRPO pool | 1,565 | 1,438 | **127** |
| Run 1 products actually trained | 300 | 279 | **21** |

Every row in `data/train_weak_grpo_cap4.jsonl` carries embedded
`split="train"`, but 127 of those SKUs belong to the authoritative validation
list. The embedded field and external manifest therefore disagree.

Consequences:

- the SFT 360-row validation split is not disjoint from Run 1;
- the pool builder and launch gates must use the authoritative manifest;
- the old Run 1 remains historically valid, but after repairing the pool it
  cannot serve as the clean causal control for a dense-reward run because the
  training data/order will also have changed;
- a clean reward ablation requires a corrected-pool original-reward control and
  a corrected-pool dense-reward arm.

### 2.3 Other available datasets

- `data/probe_100.jsonl` contains 100 exact SKUs disjoint from SFT train, SFT
  validation, the existing GRPO pool, Run 1 products and the frozen 300.
- Exact-SKU disjointness is not family disjointness: the probe shares 37
  normalized product families with the GRPO pool and 15 with the 300 products
  actually used by Run 1.
- The legacy frozen 300 also has zero exact-SKU overlap with the GRPO pool, but
  shares 33 normalized families with the pool and 10 with Run 1's actual
  products.
- Its prior use and label quality must be audited before deciding whether to
  consume it as development data.
- One hundred rows are probably insufficient as the sole final rare-class
  macro-F1 confirmation set.
- A new untouched confirmation set is preferred for the final publishable
  comparison.

## 3. Non-negotiable experiment rules

1. Never overwrite Run 1 data, manifests, predictions or checkpoints.
2. Never use the exposed frozen 300 to select a reward, checkpoint, beta or
   stopping threshold.
3. Treat `data/splits/sft-v1.json` as the authoritative split source unless a
   new versioned manifest is deliberately created.
4. Every model-selection dataset must have zero SKU and family overlap with the
   corresponding training pool.
5. Compare reward candidates on the same rollout groups. Do not compare an
   offline starting-policy statistic directly with Run 1's changing-policy
   `32.7%` statistic.
6. Change one causal factor per arm. Reward densification and KL are separate
   ablations.
7. Record failed experiments; do not erase or replace them.
8. Every durable result receives a version, input hashes, code commit, config,
   counts, invariants and output hash.
9. GPU dispatch remains blocked until the run contract and all required gates
   are committed.

## 4. Execution overview

| phase | purpose | compute | dispatch gate |
|---|---|---:|---|
| A | bank reviewed state | local CPU | coherent commits only |
| B | repair train/validation boundary | local CPU | zero authoritative validation overlap |
| C | test Run 1 overcommitment mechanism | local CPU | transition artifact published |
| D | design and replay candidate rewards | local CPU | dense reward beats original on predeclared signal tests |
| E | lock development and confirmation data | local CPU | no training overlap; roles documented |
| F | build checkpoint monitoring | CPU + GPU smoke | fixed-prompt greedy and sampled evaluation proven |
| G | predeclare corrected control and treatment | local CPU | signed contract and config diff |
| H | run corrected control and dense treatment | RTX 3090 | all launch checks pass |
| I | optional KL arm | RTX 3090 | dense reward justified; KL memory preflight passes |
| J | untouched confirmation and W2 closeout | inference only | final recipe selected without confirmation data |

## 5. Phase A: bank reviewed state

The diagnosis document is already committed as `7bce070` and pushed to the
remote default branch, `master`.

Pending local files must not be committed together automatically. Review and
group `.gitignore`, `blog/`, `runs/sft-attention-2epoch/README.md` and
`w2-brief.html` by purpose. Check generated figures, large files, credentials,
data provenance and publication safety before making a repository or W&B run
public.

**Done when:** each intended file is either committed in a coherent commit or
explicitly documented as local-only; unrelated work remains untouched.

## 6. Phase B: repair the train/validation boundary

### B1. Publish a split audit

Create a deterministic CPU audit that reports SKU and family overlap among:

- SFT train and validation manifests;
- the 3,600-row difficulty source;
- retained and capped GRPO pools;
- the 300 Run 1 products;
- probe 100;
- legacy frozen 300;
- any proposed development, easy-retention and confirmation slices.

Proposed artifact:

`runs/grpo-run2-data-boundary-audit.json`

It must record exact overlap counts, offending SKU lists, source hashes and the
split authority used.

**Completed B1 result.** `training/audit_data_boundaries.py` produced
`runs/grpo-run2-data-boundary-audit.json`, SHA-256
`95997f6b89d6c821b8e81b9a4f53acb05a29e6c90b422d8530275f7a82cac61f`.
The artifact rebuilds byte-for-byte from the same inputs and records status
`issues_found`:

| boundary | exact SKU overlap | family overlap |
|---|---:|---:|
| GRPO pool vs SFT validation | 127 | 99 |
| Run 1 trained products vs SFT validation | 21 | 20 |
| GRPO pool vs probe 100 | 0 | 37 |
| Run 1 trained products vs probe 100 | 0 | 15 |
| GRPO pool vs legacy frozen 300 | 0 | 33 |
| Run 1 trained products vs legacy frozen 300 | 0 | 10 |

The six focused CPU tests pass. They cover deterministic reporting,
SKU-versus-family overlap, duplicate and unmapped IDs, source and pool hash
drift, Run 1 rollout structure and collision-safe publication. No source data,
manifest or pool was changed.

### B2. Find and fix the source of disagreement

Trace where embedded `split="train"` was assigned and why the pool builder did
not consult `data/splits/sft-v1.json`. Fix the builder so a GRPO training pool:

- accepts an explicit authoritative split manifest;
- selects only manifest-training SKUs;
- rejects unknown SKUs;
- fails if any manifest-validation SKU or validation family appears;
- records the manifest path and hash in the output manifest.

**Completed B2 result.** The root cause was two meanings of `train`: the
embedded W1 field means “member of the 3,600-row weak-training corpus,” while
`data/splits/sft-v1.json` later subdivides that corpus into SFT train and
validation. Difficulty scoring correctly preserved the source rows, but the
GRPO pool builder and launch checks treated the older embedded value as the
training authority.

The pool writer now accepts `--sft-split-manifest`, verifies the manifest's
source hash and complete/disjoint SKU assignments, rejects train/validation
family overlap, filters to authoritative SFT-training rows, recomputes
difficulty eligibility and family capping, and records the split manifest and
hash in the new pool manifest. Historical calls without this explicit option
remain reproducible and do not rewrite Run 1 artifacts. The corrected expected
cap-four output is 1,438 rows across 1,051 families.

### B3. Rebuild without overwriting history

Create a new versioned train-only pool and manifest. Preserve the original
cap-four pool because it is part of Run 1 provenance. Recompute composition,
difficulty, family-cap and gold-density statistics for the corrected pool.

**Completed B3 result.** The corrected pool was written to new paths without
overwriting Run 1:

- dataset: `data/train_weak_grpo_cap4_sft_train_v1.jsonl`, 1,438 rows,
  SHA-256 `1ca64f668b0359c2e83850832d5db2ffaf5a5f621556ac4776cf9d5c3fb26a53`;
- manifest:
  `runs/sft-difficulty-k8/grpo-pool-cap4-sft-train-v1-manifest.json`,
  SHA-256 `42ca7b3ad0b1a1e61539493a693b33ea56238f30004e6a25bcdb9bd24e19282a`.

From 3,240 authoritative SFT-training rows, 1,565 were difficulty-eligible;
1,438 rows across 1,051 families survived cap four and 127 were capped. An
independent verification reproduced the output hash/order, confirmed every
manifest invariant, found zero authoritative-validation SKU overlap and zero
validation-family overlap, and proved the corrected set equals the historical
active pool minus its 127 validation SKUs.

### B4. Add tests and launch invariants

Tests must deliberately inject:

- a validation SKU marked as embedded train;
- a train/validation family overlap;
- a SKU absent from the authoritative manifest;
- a changed manifest hash.

All must fail closed. The production preflight must independently repeat the
zero-overlap assertion.

**Completed B4 result.** `training/grpo_run2_preflight.py` is a separate,
CPU-only Run 2 pool boundary; the historical Run 1 verifier is unchanged. It
locks the corrected data, pool-manifest and SFT split-manifest hashes, requires
manifest version `grpo-pool-cap4-sft-train-v1`, verifies the split source hash
and assignments, reconstructs normalized families independently, and confirms
1,438 active rows across 1,051 families with zero validation SKU/family overlap.
It reports `cuda_imports_performed=false` and does not expose GPU dispatch.

Five focused tests pass: the real pool, lock constants, wrong manifest version,
failed builder invariant and a synthetic family leak reconstructed independently
by preflight. Any later Run 2 launch contract must invoke this verifier before
model loading; no Run 2 launch path exists yet that can bypass it.

**Phase B gate:** corrected pool and manifest exist; SKU and family overlap with
authoritative validation is exactly zero; focused and full CPU suites pass.

## 7. Phase C: test H1 with prediction transitions

Use the hash-locked SFT and GRPO frozen predictions for diagnosis only. For
every gold-known field, classify the paired transition as:

- abstain -> correct;
- abstain -> wrong;
- correct -> abstain;
- wrong -> abstain;
- correct -> wrong;
- wrong -> correct;
- unchanged correct;
- unchanged wrong.

Also report:

- counts and rates overall and by attribute;
- the same counts for `collar_type`;
- predicted-class frequency shifts;
- common-answer concentration;
- multi-value-field handling separately;
- SKU-level examples for the largest transition classes.

Proposed artifacts:

- `runs/grpo-first-300-frozen-eval-300-transitions.json`
- focused CPU tests with synthetic abstain/correct/wrong cases.

Interpretation gate:

- excess abstain -> wrong strengthens H1;
- abstain -> correct without comparable wrong growth weakens H1;
- little abstention movement outside `collar_type` narrows H1 to an
  attribute-specific failure;
- common-class frequency growth must be measured before claiming majority-class
  flooding.

**Completed Phase C result.** `evalharness/transition_analysis.py` published
`runs/grpo-first-300-frozen-eval-300-transitions.json`, SHA-256
`a842a766b6fdcbcbf6539cad707845c69b03a93676315a22ab5f7761f29ad19d`,
from the hash-locked SFT and GRPO prediction files. Across 2,755 gold-known
cells, abstentions fell `157 -> 89`, correct cells rose `2,310 -> 2,329`, and
wrong cells rose `288 -> 337`. Of 70 baseline abstentions that became answers,
31 became correct and 39 became wrong. The eight-cell harmful excess modestly
supports H1, but four of those eight cells came from `collar_type`; outside
that attribute the split was nearly even (`29 -> correct`, `33 -> wrong`) and
direct correct/wrong churn was balanced (`29 correct -> wrong`, `30 wrong ->
correct`). The multi-value `details` field moved in the opposite direction:
nine of 12 abstain exits became correct.

Cell-level exact accuracy rose from 83.85% to 84.54% even while
attribute-macro-F1 fell. H1 therefore cannot be the complete regression
mechanism: the location of correctness across attributes and frequent versus
rare classes matters in addition to the decision to abstain or commit.

Common committed-class concentration was nearly flat overall and outside
`collar_type`, so universal majority-class flooding is not supported. Local
increases occurred for `pattern: solid`, `collar_type: not_applicable`, and
`details: none`. Four focused tests cover exhaustive transitions, wrong-answer
churn, unknown exclusion, multi-value semantics and fail-closed pairing. A
second build was byte-identical; the full CPU suite passes with 507 tests.
These are diagnostic, exploratory findings on an exposed set, not causal or
model-selection evidence.

**Phase C gate:** deterministic artifact, tests and tracker explanation are
complete. Findings may refine the dense reward but cannot rehabilitate the
legacy frozen set for selection.

## 8. Phase D: design rewards and forecast signal offline

### D1. Establish an apples-to-apples baseline

Filter the existing 28,800 difficulty rollouts through the authoritative SFT
training manifest. Replay the original `1:1:2` reward over those groups first.
This is the offline baseline for candidate comparison.

Do not use Run 1's observed `32.7%` zero-variance rate as the direct baseline:
that statistic came from a changing GRPO policy, while the difficulty artifact
came from the starting SFT policy.

**Completed D1 result.** `training/replay_original_reward.py` re-executed the
exact original functions and `1:1:2` weights on hash-locked difficulty
rollouts, excluded all 360 SFT-validation SKUs, and published
`runs/grpo-run2-original-reward-training-replay.json`, SHA-256
`48f0d8737f9fab231d594dcf178e335fcd5711f817c5158557a7aee839f789b8`.
All 25,920 authoritative-training reward triples matched their durable grades,
and a second run was byte-identical.

On the exact 1,438-group corrected active pool, total reward varies in every
group because mixed strict-pass selection guarantees golden-agreement
variation. However, format varies in zero groups, compliance in only 98, and
1,362 groups (94.7%) contain only two total-reward values; 569 groups have a
largest tie of seven completions. On all 3,240 authoritative training groups,
total reward has zero variance in 1,571 groups (48.5%): all 710 always-pass
groups and 861 of 965 always-fail groups. The original reward is coarse and
selection-dependent rather than devoid of starting-policy variance on the
active pool.

This corrects the candidate gate: active-pool zero variance cannot be reduced
below its 0% baseline. Candidates must preserve active-group variation while
improving ranking resolution and field-level alignment. Reduced zero variance
on the full training scope remains useful secondary evidence, not a reason by
itself to widen the pool. Neither scope is directly comparable with Run 1's
changing-policy 32.7%. Four focused tests and the full 511-test CPU suite pass.

### D2. Define a small candidate family

Do not lock arbitrary weights before measuring them. Candidate rewards should
explore a small predeclared family with these properties:

- invalid schema is gated or strongly penalized, not rewarded as a useful
  saturated component;
- field-level partial credit;
- `correct > abstain > wrong` for gold-known fields;
- explicit handling of gold-unknown and not-applicable fields;
- normalization by the number of scorable fields;
- per-rule-violation cost rather than one saturated compliance bit;
- bounded class balancing aligned with macro-F1;
- explicit set scoring for multi-value `details`;
- a documented total range and interpretable component scale.

Before replay, write payoff tables covering correct, abstain, clean wrong,
rule-violating and malformed outputs. Reject any candidate with an obvious
strategic shortcut.

**Completed D2 payoff-contract substep.**
`W2_GRPO_RUN2_REWARD_PAYOFFS.md`, version
`grpo-run2-reward-payoffs-v1`, SHA-256
`ad50b23923992d8f888657de9ffff5be93d87a39b31d9cbe39c6fc0a86456db7`,
locks ordinal behavior before numeric tuning. Known fields require
`correct > abstain > wrong > malformed floor`; gold unknown is an explicit
unknown-neutral versus unknown-aware ablation; `details` receives monotonic
set-quality partial credit; and every additional rule violation lowers an
otherwise identical valid output.

The three predeclared candidates are U (uniform, unknown-neutral), UA (uniform,
unknown-aware) and CB (capped class-balanced, unknown-aware). All share a strict
15-field semantic gate, per-rule cost and density normalization. The gate
rejects three forms currently accepted by the base verifier—`{}`, empty
`details`, and `details` mixing `unknown` with committed values—because they
create dense-reward shortcuts. No production code or winner was selected in
this ordinal substep.

**Completed D2 bounded-scale substep.**
`W2_GRPO_RUN2_REWARD_SCALES.md`, version
`grpo-run2-reward-scale-contract-v1`, SHA-256
`5a9ab0c7567f1f43e468cd66a6f5e12a018eb2cb86748294de4cfcafa2020d13`,
maps the ordinal contract onto one centered, bounded scale without calculating
candidate rewards on the rollout completions. Known-field utility is
`correct=+1`, `abstain=0`, `wrong=-1`; unknown-aware behavior uses `+1/-1`;
and each separately normalized semantic component remains in `[-1,+1]`.

The unknown-aware combination uses the active pool's exact training-only cell
mix: `12,533/21,570` known and `9,037/21,570` unknown (approximately
`0.581/0.419`). `details` uses set-F1 mapped by `P(q)=2q-1`, so set-F1 `0.5`
is the explicit abstention crossover. Candidate CB uses clipped inverse-square-
root support weights in `[0.5,2]`, computed relative to each attribute's
median positive support and re-normalized by summed weights.

Rule violations cost `0.05` each through a cap of three, for maximum cost
`0.15`; malformed output receives `-1.25`. The worst valid complete total is
therefore `-1.15`, safely above the malformed floor, and the maximum is `+1`.
The three-violation cap is grounded in the locked active starting-policy
distribution (`0: 11,418`, `1: 81`, `2: 1`, `3: 4`) while protecting against
the pack's 34-rule inventory.

The executable contract at `training/reward_scale_contract.py` produced
`runs/grpo-run2-reward-scale-contract.json`, 18,444 bytes, SHA-256
`a343d93b02022701060f59a20ad9a124bb25b6cbc3ec08602c47b8a9dad3c56d`.
Five focused CPU tests prove all ordinal inequalities, field-density and
class-weight normalization, details monotonicity, rule capping, malformed-floor
safety, and fail-closed input bounds. The artifact explicitly records that no
candidate completion reward was calculated. D2 is complete; production
candidate implementation begins next.

**Completed strict semantic-gate implementation substep.**
`training/run2_rewards.py` now contains the shared gate that every U/UA/CB
candidate must pass before semantic scoring. It parses literal JSON without
repair, detects duplicate JSON keys before Python can silently keep the last
value, requires exactly all 15 pack fields, delegates shape and controlled-
vocabulary checks to the base verifier, and adds nonempty, duplicate-free,
unambiguous multi-value rules. Cross-field rule violations remain eligible and
are returned for the later bounded rule cost.

Seventeen focused CPU tests cover clean and all-abstain records, null
multi-values, rule-violating-but-eligible records, malformed/fenced/non-object
JSON, missing/extra fields, OOV values, wrong shapes, duplicate object keys,
empty/duplicate/mixed-unknown `details`, and integration-type errors. They also
prove the gate closes `{}`, `{"details":[]}` and
`{"details":["unknown","none"]}` even though the historical base verifier
accepts them. The full CPU suite passes 533 tests. No U/UA/CB semantic payoff
has yet been calculated and no rollout replay has begun.

**Completed Candidate U known-field semantic substep.**
`score_uniform_known_fields` now emits an auditable field ledger and the
uniform mean over gold-known fields. Scalar and not-applicable targets follow
`correct=+1`, `abstain=0`, `wrong=-1`; gold `unknown` fields are explicitly
listed and excluded. Multi-value committed predictions use order-insensitive
set-F1 mapped through `P(q)=2q-1`, while explicit `unknown` remains abstention
and `null` remains a not-applicable claim.

The scorer rechecks both prediction and trusted gold against the complete
record contract, fails closed if no gold-known field exists, and normalizes by
the number of known fields. The active pool independently guarantees at least
two such fields per product. Its output keeps field name, outcome, utility and
set-F1 so later replay can explain a total rather than expose only a scalar.

The first focused run caught an audit/order bug: exact `details` returned the
right utility before traversing the set-F1 path, leaving `set_f1` empty and
making reordered exact sets vulnerable to list-order semantics. Routing every
committed multi-value prediction through set-F1 fixed both issues; a reversed
exact-set fixture now receives `set_f1=1` and utility `+1`.

The focused file now passes 34 tests and the full CPU suite passes 550 tests.
No malformed-floor handling, rule-cost composition, unknown-aware score,
class-balanced score, TRL batch wrapper or rollout replay was added in this
substep.

**Completed Candidate U single-record composition substep.**
`score_candidate_u` now validates trusted gold first, applies the strict gate to
one literal completion, returns `-1.25` with no semantic/rule components for an
ineligible output, and otherwise returns the uniform known-field semantic score
plus `-0.05` per observed rule violation through the locked cap. The result
retains gate errors, rule IDs, rule adjustment and the complete known-field
ledger alongside the final scalar.

Validating gold before model output is intentional: malformed model text may
earn the floor, but a missing gold field or all-unknown gold record is a data
contract failure and raises even when the completion is malformed. This keeps
training-data corruption from being silently counted as poor model behavior.

Synthetic fixtures prove clean exact output (`1.0`), malformed floor (`-1.25`),
nontrivial semantic averaging, and two independent rule violations reducing an
otherwise exact semantic score from `1.0` to `0.9`. Malformed records expose no
invented zero-valued components. The focused file passes 41 tests and the full
CPU suite passes 557 tests. UA, CB, the TRL batch interface and rollout replay
remain untouched.

**Completed Candidate U TRL-compatible batch-wrapper substep.**
`candidate_u_reward` accepts TRL's plain-text or one-assistant-message
completion shapes, dict or JSON-string gold, and ignored extra trainer columns.
It checks completion/gold length, parses every gold record with indexed error
messages, validates the full trusted-gold batch, and only then maps aligned
pairs through `score_candidate_u`. It returns the ordered float list expected by
TRL and introduces no reward arithmetic of its own.

The wrapper reuses the already tested first-run completion/gold adapters and
loads the default pack lazily when one is not injected. Focused tests prove
plain and conversational shapes, mixed gold encodings, reward order, default
pack use, extra kwargs, empty batches, length mismatch, scalar-sequence misuse,
invalid message containers, indexed malformed gold and validate-all-gold-first
behavior. The focused file passes 48 tests and the full CPU suite passes 564
tests with exit code zero. Neither TRL nor CUDA is imported, and no rollout is
replayed.

**Completed Candidate UA gold-unknown component substep.**
`score_uniform_unknown_fields` evaluates only fields whose gold target is the
shape-correct `unknown` token. Honest abstention receives `+1`; every canonical
value, `null`, or committed multi-value list receives `-1`. Gold-known fields
are excluded, and the component is normalized by its own unknown-field count
rather than being diluted by known labels.

The result retains one outcome per unknown field plus the excluded known-field
names. When a product has no gold-unknown fields it returns
`semantic_score=None`, not zero: absence means the later combiner must use the
known score directly, whereas a measured zero means abstentions and unsupported
commitments balanced. Focused tests cover scalar and multi-value abstention,
canonical/null commitments, separate normalization, no-unknown absence and
invalid prediction/gold contracts. The focused file passes 57 tests and the
full CPU suite passes 573 tests with exit code zero. UA combination, total
composition, batch wrapper, CB and rollout replay remain untouched.

**Completed Candidate UA semantic-combination substep.**
`score_uniform_unknown_aware_semantics` calculates the proven known and unknown
components independently, then combines their means with the locked active-
training cell shares: known `12,533/21,570` and unknown `9,037/21,570`. The
result retains both complete component ledgers beside the final semantic score.

Synthetic fixtures prove exact weighting, the `[-1,+1]` bounds, and that one
versus two unknown fields produces the same combined value when component means
are unchanged. When the unknown component is absent, the known score is returned
exactly rather than multiplied by `0.581`. A perfect known score plus a fully
committed unknown score yields approximately `0.1621`, making the tradeoff
explicit. The focused file passes 62 tests and the full CPU suite passes 578
tests with exit code zero. Gate/floor, rule, batch, CB and replay composition
remain untouched for UA.

**Completed Candidate UA single-record composition substep.**
`score_candidate_ua` validates gold first, applies the shared strict gate, gives
ineligible output only the fixed `-1.25` floor, and otherwise adds the bounded
rule adjustment once after combined UA semantics. Its result keeps gate errors,
rule IDs/cost and both known/unknown ledgers beside the final scalar.

A direct synthetic contrast fixes the intended causal difference: for a record
with perfect known fields and one unsupported gold-unknown commitment,
Candidate U scores `1.0` because that field is excluded, while UA scores
approximately `0.1621` under the locked cell weights. Exact but incoherent UA
semantics remain `1.0` before two rule violations reduce the total to `0.9`.
Malformed output exposes no invented components, and invalid/all-unknown gold
fails before model text. The focused file passes 69 tests and the full CPU suite
passes 585 tests with exit code zero. UA batch, CB and replay remain untouched.

**Completed Candidate UA TRL-compatible batch-wrapper substep.**
The U adapter was factored into `_candidate_reward_batch`, which owns completion
shape parsing, alignment, indexed gold parsing, full-batch gold validation and
strict ordered mapping. `candidate_u_reward` and `candidate_ua_reward` now differ
only in the proven single-record scorer they pass to that helper; neither adds
reward arithmetic.

UA fixtures prove ordered exact/floor/unsupported-commitment outputs across
plain and conversational forms, mixed gold encodings, default pack use, ignored
trainer kwargs, empty input, misalignment and gold-first failure. All existing U
adapter tests remain green after the refactor. The focused file passes 73 tests
and the full CPU suite passes 589 tests with exit code zero. U and UA are now
implementation-complete for offline replay; CB and replay remain untouched.

### D3. Replay every candidate on identical groups

For the original reward and each candidate, report:

- component mean, standard deviation and quantiles;
- within-group variance distribution;
- zero-variance group count and share;
- reward ties within each eight-completion group;
- contribution by field, class, product category and difficulty band;
- correlation/rank agreement with per-field correctness, selective correctness,
  coverage and rule violations;
- examples where candidate ranking differs from original ranking;
- sensitivity to class-weight caps and scorable-field count.

### D4. Candidate acceptance criteria

Lock exact thresholds before reading candidate identities where practical. At a
minimum, the selected reward must:

- preserve nonzero weighted-total variance across the corrected active groups;
- materially increase active-group ranking resolution by reducing large ties
  and/or increasing distinct reward levels on the same ordered rollouts;
- reduce zero-variance groups on the full authoritative-training scope where
  the original baseline is 48.5%, without treating that secondary result as a
  reason by itself to widen the active pool;
- reward partial correctness monotonically;
- rank correct above abstain above wrong in direct fixtures;
- not reward increased coverage by itself;
- preserve sensitivity to rule violations;
- avoid domination by one attribute, class or high-label-density product;
- pass all CPU tests and deterministic replay checks.

**Completed D3/D4 comparison-contract substep.** Before opening the raw
candidate replay records for aggregate analysis,
`W2_GRPO_RUN2_COMPARISON_CONTRACT.md` and
`training/run2_comparison_contract.py` locked the product-group unit, 12-decimal
tie rule, group-first pairwise resolution/alignment/coverage metrics, required
segments, 10,000-replicate paired group bootstrap and exact numeric gates. The
published `runs/grpo-run2-comparison-contract.json` reads candidate manifest
metadata but explicitly does not open the candidate gzip, calculate candidate
aggregates, apply thresholds or select a winner.

The universal gates require: 0% active zero variance; at least 50% of active
groups with three reward levels; at most 50% with a largest tie of six or more;
at least +0.10 mean pairwise discrimination with a positive interval lower
bound; known-utility and harmful-coverage noninferiority; 20% field and 15% CB
class contribution caps; and at most 40% zero variance on the full 3,240-row
training scope. A complexity hierarchy prefers U unless UA earns unknown-aware
improvement, then prefers the surviving uniform policy unless CB earns a
class-balanced improvement. Seven focused tests pass. No real candidate
aggregate was read.

This contract also makes one remaining evidence gap explicit: the published
candidate replay covers only the 1,438 active groups, while the full-scope gate
requires a separate raw replay of the already-locked k=8 outputs for all 3,240
authoritative training rows. No new model generation is needed or allowed.

**Completed D3 analyzer-core substep.**
`training/analyze_run2_candidates.py` now implements file-I/O-free, group-first
reward summaries, optional-target directional alignment, harmful-coverage
preference and deterministic paired whole-product bootstrap intervals. Seven
synthetic tests prove that products remain equally weighted even when their
numbers of usable completion pairs differ, reward ties are not counted as
wins, missing target cells are skipped pairwise, all eight completions stay
together during resampling, and malformed group inputs fail closed. The core
explicitly records that no real replay was opened, no gate was applied and no
winner was selected. Ledger extraction, contributions and segments remain a
separate adapter step.

**Completed D3 synthetic ledger-adapter substep.**
`training/run2_replay_adapter.py` converts one already-materialized nested
replay group into exactly one analyzer observation plus separate field- and
CB-class-contribution child records. It validates ordered k=8 identity,
known/unknown denominators, cross-candidate gate/rule/utility agreement, class
support and weight lineage, semantic contribution sums and final reward
reconstruction. Difficulty and class-support bands are fixed without candidate
outcomes. Six focused synthetic tests cover malformed null handling,
one-to-many class allocation without join inflation and direct analyzer-core
integration. The full suite passes 643 tests. The module performs no file I/O;
the real replay remains unopened for analysis and no gate or winner was applied.

**Completed D3 synthetic dominance/segment-summary substep.**
`training/run2_segment_summaries.py` separately summarizes product observations,
field children and CB class children. Product category, starting-policy
difficulty and gold-known count each cover every product exactly once; fields,
attribute/class pairs and support bands never become product denominators. The
20% field and 15% class thresholds remain unapplied references, and segments
below 30 products are explicitly non-interpretable. Five adversarial synthetic
tests cover 80/20 dominance recovery, opposite-direction group averaging,
membership subtotals, duplicate children and broken parent allocation. The full
suite passes 648 tests. No real replay or candidate outcome was read.

**Completed D3 synthetic streaming-orchestrator substep.**
`training/run2_analysis_orchestrator.py` requires explicit manifest, gzip,
comparison-contract, class-weight and output paths. It verifies all small
control files plus compressed byte/hash identity before opening gzip, streams
and adapts each canonical group once, reconciles counts and ordered hashes, and
publishes only through exclusive atomic JSON. Synthetic and production roles
are mutually exclusive; production also requires the exact 1,438/11,504 scope
and locked 10,000-replicate settings. Six temporary-fixture tests cover
byte-deterministic success, pre-decompression hash failure, internal order
failure with no partial output, mode isolation, collision protection and no
implicit CLI paths. The first run caught a missing group-size import; after the
one-line fix, all focused tests and the 654-test full suite pass. The real replay
was not decompressed or aggregated.

**Completed production preflight-only substep.**
The orchestrator now exposes `run_preflight` and `--preflight-only`. A synthetic
test forbids gzip opening and adapter calls while proving all identity checks
still run. The real production preflight then verified the 6,435-byte manifest,
1,921,202-byte compressed replay, 12,079-byte comparison contract and
27,446-byte CB map; exact hashes, 1,438/11,504 counts, ordered SKU/key hashes,
production role and locked 20260812/10,000/95% settings all match. It explicitly
parsed zero replay records, calculated zero candidate aggregates, applied no
gates, selected no winner and published no artifact. Seven focused tests and
the full 655-test suite pass.

**Completed D3 production-launcher integration substep.**
The orchestrator now separates `build_analysis_artifact`, which returns the
complete aggregate in memory, from exclusive publication. A dedicated
`run_active_preflight` verifies only the four D3 sources and stops before gzip
decompression; the existing dual-scope Gate G10 preflight remains unchanged.
`training/run2_d3_production.py` fixes those four paths and the one D3 output.
Its CLI exposes only `--preflight-only`. The future execution composition was
tested with fabricated production-shaped substitutes and enforced the order
build, complete-contract validation, then exclusive publication; an invalid
artifact never reached the publisher. The real locked preflight passed with
`gzip.open` patched to raise, reconfirming 1,438 groups, 11,504 completions and
both ordered hashes while leaving the D3 result absent. The focused integration
passes 29 tests, the related D3 stack passes 47 tests and the full CPU suite
passes 741 tests. No active replay record was parsed and no Gate G1-G9 result,
ranking or winner exists.

**Phase D gate:** one reward definition and all constants are selected from
training-only evidence, versioned and hash-pinned. If no candidate passes, do
not dispatch GPU training.

## 9. Phase E: lock development and confirmation data

Assign each dataset exactly one role before additional model inference:

- training;
- representative development;
- difficult/mixed development;
- easy-retention development;
- untouched final confirmation;
- legacy reporting only.

### Development-data decision

Audit these options before selecting one:

1. SFT validation rows not directly trained in Run 1, with contamination and
   prior-use limitations documented;
2. the disjoint 100-row probe, if it has not already been consumed for model
   selection;
3. a newly sampled development set from data excluded from all training pools.

The development set should include representative, difficult and easy-retention
slices. It must be large enough to interpret macro-F1 and must report class
support rather than hiding small cells.

### Final confirmation

Prefer a newly collected and frozen confirmation set, sampled before the final
recipe is selected but not opened or scored until selection is complete. Record
provenance and human corrections. The exposed legacy frozen 300 cannot fill
this role.

**Phase E gate:** role manifest is committed; every development/confirmation
slice has zero SKU and family overlap with every corresponding training arm;
the confirmation labels and model outputs remain unopened for selection.

## 10. Phase F: build checkpoint monitoring

At each retained checkpoint, evaluate the same fixed non-frozen development
prompts under:

- deterministic greedy decoding;
- repeated sampled decoding matching training temperature/top-p;
- the primary evaluator;
- original reward replay;
- dense reward replay;
- representative, difficult and easy-retention slices.

Track at least:

- macro-F1;
- selective macro-F1;
- coverage;
- schema and vocabulary validity;
- rule violations;
- original and dense reward;
- sampled mean and dispersion;
- slice-level results.

Greedy-only monitoring can detect regression but cannot isolate sampled/greedy
divergence. Both are required to test H4.

Live aborting is new runtime behavior. Do not assume the existing detached
training machinery already supports it. Implement and smoke checkpoint
evaluation, GPU/model lifecycle, timeout behavior and failure publication.

Set quality stop thresholds only after measuring baseline variability on the
development set. Predeclare them before GPU dispatch. Avoid an arbitrary
single-checkpoint threshold; prefer a material guardrail breach or repeated
directional degradation according to the locked rule.

**Phase F gate:** CPU tests pass; one bounded GPU smoke proves checkpoint save,
greedy evaluation, sampled evaluation, resource cleanup and auditable
publication.

## 11. Phase G: predeclare the causal experiment

Repairing the data boundary changes the training pool, so the historical Run 1
cannot isolate reward design by itself. The clean minimum comparison is:

| arm | corrected pool/order | reward | beta | purpose |
|---|---|---|---:|---|
| A | fixed and identical | original `1:1:2` | 0 | corrected control |
| B | fixed and identical | selected dense reward | 0 | isolate reward design |
| C, optional | fixed and identical | selected dense reward | >0 | isolate KL anchor |

Run 1 remains a historical comparator, not Arm A.

Lock for Arms A and B:

- starting adapter and hash;
- corrected product sequence;
- seed and all stochastic settings;
- optimizer, learning rate and schedule;
- step/checkpoint budget;
- generation parameters;
- LoRA configuration;
- reward implementation and constants;
- explicit `beta`;
- validation prompts and decoding seeds;
- stop/abort criteria;
- disk floor and retention policy;
- output paths and artifact schemas;
- expected code commit;
- smoke-to-full config diff;
- unresolved-deferral scan.

If budget permits only one new GPU arm, run dense reward on the corrected pool
and label it exploratory; do not claim that reward densification alone caused
its difference from historical Run 1.

**Phase G gate:** signed run contract is committed, construction/preflight is
read-only, no deferred parameter remains unresolved and no output path exists.

**Phase G completed on 2026-08-13.** The causal question is now encoded in
`W2_GRPO_RUN2_CAUSAL_EXPERIMENT_CONTRACT.md` and the machine-readable
`runs/grpo-run2-causal-experiment-contract.json`. Arm A is the corrected
original-reward control and Arm B changes only the reward definition to
Candidate UA; `beta=0` in both arms. Both begin from the same hash-locked SFT
adapter and consume the same 300 unique products in the same optimizer-step
order. Run 1 is retained only as historical context.

The contract pins 18 input artifacts and 24 execution files to code commit
`e3c4d6f9c31ba8c136107f7d123c9da1a107f91a`. Its fixed schedule has order hash
`7a73e21387b344ee67606008b41f3c57c04da7a7345dfb5fdc10a0cb07f344f6`.
The production Vast.ai preflight verified all hashes, the exact software/GPU
environment, 3 GiB suite-start disk floor and ten unused arm/monitor/failure
paths. It found zero deferred choices and created no arm path. A separate
construction proof instantiated the real TRL `GRPOConfig` for both arms and
verified that only reward bindings plus bookkeeping output names differ. It
constructed no model or trainer and dispatched no training.

A full 360-product SFT monitor baseline, with deterministic greedy decoding and
eight sampled repeats, sets the live quality policy before training. Each mode
is anchored to its own baseline. A single breach warns; the same metric, view
and decoding mode must breach at two consecutive checkpoints to abort. Monitor
failure or insufficient checkpoint-boundary GPU headroom aborts immediately.
The primary endpoint is the paired checkpoint-300 representative greedy
macro-F1 difference, B minus A, with a 10,000-replicate product bootstrap.
Earlier checkpoints are diagnostic only. No confirmation or legacy frozen-300
output was opened.

## 12. Phase H: corrected control and dense treatment

1. Run Arm A under the existing detached, fail-closed publication machinery.
2. Validate and hash all artifacts before starting Arm B.
3. Run Arm B with only the reward definition changed.
4. Apply the predeclared checkpoint and abort policy.
5. Compare paired checkpoint outputs on fixed development prompts.
6. Report effect sizes, paired uncertainty, coverage/rule guardrails and slice
   behavior; do not select on one anecdotal class.

The main causal question is whether Arm B improves over Arm A under the same
corrected data and training configuration.

**Phase H gate:** both arms close out successfully or fail with complete
evidence. Select no checkpoint using the legacy frozen 300 or untouched
confirmation set.

**Arm A read-only launch bridge completed on 2026-08-13.**
`training/run2_arm_a_launcher.py` composes the locked corrected-control surface
without modifying any of the 24 execution files pinned by Phase G. The launcher
independently pins its own source to commit
`b6bc2ce32c0efd5009064981a7dea5b8f0617b45`, reruns the causal preflight, checks
the accepted config-construction lineage, verifies all 300 unique scheduled
SKUs and their optimizer-step order, binds the original three reward callables
and constructs the Phase F plus Phase G monitor coordinator/callback graph.

The production proof fixed exact evaluator commands for checkpoints 100, 200
and 300 and verified both the current 3 GiB disk floor and 6 GiB monitor-memory
floor. Its CLI exposes only `validate`; it cannot dispatch work. No dataset was
materialized, no callback lifecycle was entered, no monitor process started,
and no Arm A output/control/monitor/failure path was created. This is a launch-
wiring proof, not a training or concurrent-memory proof.

## 13. Phase I: optional KL arm

Only proceed if dense reward is sufficiently promising to justify another arm.

1. Select and justify one `beta > 0` value before training.
2. Profile reference-policy GPU memory and disk requirements.
3. Run a construction gate and real bounded smoke.
4. Train Arm C with every Arm B setting held fixed except beta/reference model.
5. Compare rule retention, rare-class behavior, easy-slice retention and reward
   gain against Arm B.

Do not change reward and beta together. If the Arm C resource preflight fails,
record that result and close W2 without forcing the experiment.

## 14. Phase J: confirmation, reporting and W2 closeout

After selecting exactly one final recipe from development evidence:

1. lock the final model/checkpoint and evaluation command;
2. open the new untouched confirmation set once;
3. compare against the locked SFT baseline with paired uncertainty;
4. report macro-F1, selective macro-F1, coverage, validity, rules, class support,
   compute time, GPU memory and disk use;
5. retain the legacy frozen-300 result as explicitly non-confirmatory context;
6. update `W2_GRPO_BLOG_TRACKER.md` with all successful and failed arms;
7. commit manifests, metrics, predictions where appropriate, hashes and
   closeout records;
8. close W2 and move to W3 rather than beginning an open-ended sweep.

If a winning recipe emerges, repeat it with at least one additional seed before
making a broad scientific claim. A single seed can support an engineering case
study when that limitation is explicit.

## 15. Blog and publication track

The blog can progress in parallel without waiting for a positive Run 2 result.
Run 1 already supports a useful technical story about:

- a cleanly executed negative RL result;
- paired uncertainty instead of headline-only metrics;
- exact reward replay rejecting the easiest explanation;
- sparse within-group reward signal;
- overcommitment as a hypothesis rather than an assumed exploit;
- a train/validation boundary defect discovered by manifest-level auditing;
- how a corrective experiment is designed without reusing a burned test set.

Before making the repository, figures or W&B public, audit secrets, data rights,
external URLs, model licenses and artifact sizes. Keep scientific artifact
commits separate from rendered blog/hosting changes.

## 16. Decision log

| date | decision | reason |
|---|---|---|
| 2026-08-11 | Treat legacy frozen 300 as reporting-only | Diagnosis has exposed it to model-selection decisions |
| 2026-08-11 | Repair split boundary before Run 2 design | Existing pool includes 127 authoritative validation SKUs; Run 1 trained on 21 |
| 2026-08-11 | Replay original and dense rewards on identical training-only groups | Avoid comparing starting-policy offline data with changing-policy Run 1 data |
| 2026-08-11 | Require corrected original-reward control | Pool repair changes data/order, so historical Run 1 cannot isolate reward |
| 2026-08-11 | Evaluate greedy and sampled checkpoints | Greedy-only monitoring cannot test decoding-distribution mismatch |
| 2026-08-11 | Keep KL as a separate optional arm | Preserve causal attribution and require a separate memory preflight |
| 2026-08-11 | Require both SKU and family disjointness | Probe and legacy frozen data have zero pool SKU overlap but 37 and 33 overlapping normalized families |
| 2026-08-11 | Narrow H1 to modest, field-dependent overcommitment evidence | Abstain exits were 39 wrong vs 31 correct overall, but half the excess came from `collar_type` and `details` improved |
| 2026-08-11 | Replace the active-pool zero-variance reduction gate with ranking resolution | Original reward already varies in all selected starting-policy groups, but 94.7% have only two total values |
| 2026-08-11 | Lock ordinal payoff behavior before numeric reward tuning | Prevent replay outcomes from rationalizing arbitrary weights and expose empty/mixed-detail shortcuts first |
| 2026-08-12 | Freeze CB class weights before candidate scoring | Keep support choices independent of rollout outcomes and make later candidate differences attributable |
| 2026-08-12 | Reuse U field utilities inside CB | Isolate class-weighted aggregation as CB's only known-field change |
| 2026-08-12 | Reuse UA unknown semantics inside CB | Keep class weighting as the only UA/CB policy difference |
| 2026-08-12 | Reuse U/UA outer composition inside CB | Keep malformed and rule handling constant across all dense candidates |
| 2026-08-12 | Bind CB lookup once through an adapter factory | Keep hash/invariant validation outside the per-completion hot path |
| 2026-08-12 | Publish raw aligned replay before aggregation | Freeze the 1,438-group/11,504-completion denominator before calculating candidate comparisons |
| 2026-08-12 | Lock a separate full-training raw replay contract | Supply Gate G10 without mutating the frozen 1,438-product active replay or regenerating completions |
| 2026-08-12 | Stage and validate both full-replay files before linking either | Make collision or late-stream failure leave no plausible partial artifact |
| 2026-08-12 | Treat Gate G10 as eligibility, not ranking | U, UA and CB all pass; variation alone does not establish semantic direction or safety |
| 2026-08-12 | Do not interpret UA's 439 vs CB's 438 as superiority | A one-product zero-variance difference cannot justify CB complexity without active-pool D3 evidence |
| 2026-08-12 | Keep the D3 CLI preflight-only during launcher integration | Prove locked inputs and validate-before-publish wiring without authorizing the real active replay |

## 17. Live checklist

- [x] Run 1 diagnosis revised, committed and pushed.
- [x] Existing pool/SFT-manifest overlap discovered and quantified.
- [ ] Phase A: review and bank remaining intended local documentation.
- [x] Phase B1: implement and publish deterministic split-boundary audit.
- [x] Phase B2: fix authoritative split enforcement.
- [x] Phase B3: rebuild versioned train-only GRPO pool.
- [x] Phase B4: add fail-closed overlap tests and Run 2 pool preflight.
- [x] Phase C: implement SFT-to-GRPO prediction-transition analysis.
- [x] Phase D1: replay original reward on training-only difficulty groups.
- [x] Phase D2: define candidate reward family, payoff tables and bounded numeric scales.
- [x] Implement and adversarially test the shared strict semantic gate.
- [x] Implement and synthetically test Candidate U's known-field semantic scorer.
- [x] Compose and synthetically test Candidate U's single-record total reward.
- [x] Add and integration-test Candidate U's TRL-compatible batch wrapper.
- [x] Implement and synthetically test Candidate UA's gold-unknown component.
- [x] Combine and synthetically test Candidate UA's known/unknown semantics.
- [x] Compose and synthetically test Candidate UA's single-record total reward.
- [x] Add and integration-test Candidate UA's shared TRL-compatible batch wrapper.
- [x] Build, validate and hash-lock Candidate CB's training-only class-weight map.
- [x] Implement and synthetically test Candidate CB's weighted known-field scorer.
- [x] Combine and synthetically test Candidate CB's known/unknown semantics.
- [x] Compose and synthetically test Candidate CB's single-record total reward.
- [x] Add and integration-test Candidate CB's shared TRL-compatible batch adapter.
- [x] Publish deterministic raw original/U/UA/CB replay records on identical active groups.
- [x] Hash-lock exact D3 metrics, D4 numeric gates and the complexity-aware selection rule.
- [x] Implement and synthetically prove the file-I/O-free D3 analyzer core.
- [x] Implement and synthetically prove the nested replay-ledger adapter.
- [x] Implement and synthetically prove dominance and required segment summaries.
- [x] Implement and synthetically prove manifest-verified streaming orchestration.
- [x] Pass production preflight without decompressing or aggregating real evidence.
- [x] Lock the separate 3,240-product full-training raw replay contract for Gate G10.
- [x] Implement and synthetically test full-training scope validation and shared-scorer iteration.
- [x] Implement deterministic full-training records/manifest publication and synthetic failure paths.
- [x] Run the full-training builder's production scope preflight without scoring or publication.
- [x] Publish and independently verify the real full-training raw replay without aggregating candidate outcomes.
- [x] Preserve and diagnose the safely aborted first publication attempt.
- [x] Lock and test Candidate CB behavior for gold classes absent from the active-pool weight map.
- [x] Thread the audited CB diagnostic extension through the full-replay manifest and publication path.
- [x] Retry real full-training replay publication with the integrated audited extension.
- [x] Integrate and hash-lock the full-scope artifact in analysis preflight without opening either replay.
- [x] Implement and synthetically test the Gate G10 full-scope zero-variance calculator.
- [x] Adapt one nested full-replay group into Gate G10 candidate inputs and prove it synthetically.
- [x] Compose the adapter and calculator across an ordered synthetic multi-group stream.
- [x] Add synthetic manifest-verified Gate G10 streaming orchestration.
- [x] Lock and test the production Gate G10 result/publication contract.
- [x] Add and run the production Gate G10 launcher in preflight-only mode.
- [x] Add an explicit production Gate G10 execution mode and prove it with synthetic substitutes.
- [x] Calculate and independently verify Gate G10 from the verified full-scope replay.
- [x] Lock and synthetically test the production D3 aggregate result path and schema.
- [x] Integrate the D3 contract into a production-only launcher with preflight-only mode.
- [x] Phase D3: replay and compare all candidates on identical groups.
- [x] Phase D4: lock one candidate or stop if none passes.
- [ ] Phase E: lock development, easy-retention and confirmation roles.
- [x] Phase F: implement and smoke greedy plus sampled checkpoint monitoring.
- [x] Phase G: commit corrected-control/treatment run contract.
- [ ] Phase H: run corrected original-reward control and dense-reward treatment.
- [ ] Phase I: decide whether a separate KL arm is justified.
- [ ] Phase J: perform untouched confirmation and close W2.

## 18. Immediate next small step

Implement and CPU-prove the Arm A runtime/trainer composition behind the
validated bridge. Use injected fake model, trainer, config and callback types to
prove the 300-row dataset remains ordered, the original reward is attached, the
profiler and causal monitor callbacks are both retained, and every failure
prevents success publication. Keep model loading, a real `GRPOTrainer`, detached
process launch and GPU optimizer steps unavailable until that composition proof
is reviewed.
