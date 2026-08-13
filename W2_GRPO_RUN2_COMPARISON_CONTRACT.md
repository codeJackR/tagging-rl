# GRPO Run 2 comparison and acceptance contract

**Version:** `grpo-run2-comparison-acceptance-contract-v1`
**Status:** locked before aggregate candidate replay
**Created:** 2026-08-12
**Compute used:** CPU-only contract checks; no candidate aggregation
**Allowed evidence:** corrected authoritative SFT-training pool and its locked rollouts
**Prohibited evidence:** SFT validation, legacy frozen 300, probe 100, future confirmation data

## 1. Why this contract exists

The raw replay now contains original, U, UA and CB rewards for the same 11,504
completions in the same 1,438 eight-completion product groups. We deliberately
published those row-level ledgers without aggregating them. This document locks
the scoreboard before the aggregate outcomes are read.

The original total reward lies in `[0,4]`; dense rewards lie in `[-1.25,1]`.
Their raw variance magnitudes are therefore not comparable. A seven-inch ruler
does not become more informative than a ten-centimeter ruler because its number
is larger. Candidate selection instead uses scale-safe ties, pairwise ordering
and alignment with the already-declared semantic objectives. Raw mean, median,
standard deviation and quantiles are still reported within each reward.

## 2. Fixed unit and tie rule

The product/SKU is the unit of analysis. Each group has eight completions and
28 unordered completion pairs. Completions are not treated as 11,504
independent products.

Before any equality or direction comparison, finite values are rounded to 12
decimal places and negative zero becomes zero. This removes binary floating
point dust without erasing a meaningful reward difference.

Every candidate must report completion-level and group-level count, minimum,
p05, p25, median, mean, p75, p95, maximum and population standard deviation.
Both mean and median are required because group behavior may be skewed.

## 3. Primary metrics

### Ranking resolution

- `unique reward levels`: number of canonical reward values among eight;
- `largest tie`: largest number of completions sharing one reward;
- `pairwise discrimination`: non-tied reward pairs divided by 28;
- `zero variance`: all eight canonical rewards tie.

Population variance is reported only on each reward's own scale. It is not an
old-versus-dense acceptance metric.

### Directional alignment

For a target-different pair, reward and target can agree, disagree or tie.
Directional net alignment is:

`(concordant pairs - discordant pairs) / all target-different pairs`

Reward ties remain in the denominator and contribute zero. A coarse reward
therefore cannot appear accurate merely by refusing to distinguish outputs.

The targets are fixed before replay aggregation:

- canonical known utility: U's per-record correct/partial/abstain/wrong known
  semantic score before rule adjustment;
- known exact rate: `correct_labels / scorable_labels` from the source grade;
- known coverage: non-abstained gold-known fields divided by scorable fields;
- selective correctness: exact correct labels divided by committed gold-known
  fields, undefined when there are no commitments;
- unknown abstention rate: correct abstentions divided by gold-unknown fields;
- rule quality: negative rule-violation count;
- class-balanced known utility: CB's locked weighted known score before
  unknown/rule composition.

### Harmful coverage preference

For pairs where one completion covers more known fields but has lower canonical
known utility, preferring that completion scores 1, a reward tie scores 0.5 and
preferring the higher-utility completion scores 0. Lower is safer. Counting ties
as half-harm prevents the original coarse reward from receiving free credit.

Metrics are first calculated inside each product group and then averaged across
eligible groups, so a product with more pairwise contrasts cannot dominate.
Contributing group and pair counts are always published.

## 4. Uncertainty and segments

Candidate-minus-comparator intervals use 10,000 paired product-group bootstrap
replicates, seed `20260812`, with the entire k=8 group resampled as one cluster.
The interval is the 2.5th to 97.5th percentile. The same sampled group indices
are used for both rewards in every paired replicate. No p-values drive
selection.

Primary alignment claims require at least 200 contributing groups. Product
category, difficulty pass-rate band, gold-known field count, attribute and gold
class support band are mandatory views. A segment with fewer than 30 groups is
reported with support but receives no directional interpretation.

## 5. Universal candidate gates

Every selected candidate must pass all applicable gates:

1. all lineage, deterministic rebuild and CPU-test checks pass;
2. active-pool zero-variance share stays exactly 0%;
3. at least 50% of active groups have three or more reward levels, versus the
   original 76/1,438 or 5.3%;
4. at most 50% have a largest tie of six or more, versus the original
   954/1,438 or 66.3%;
5. mean pairwise discrimination improves over original by at least 0.10 and
   its paired 95% interval has a lower bound above zero;
6. canonical-known-utility net alignment is no worse in point estimate and its
   paired lower bound is at least -0.02;
7. harmful-coverage preference increases by no more than 0.02 in point estimate
   and no more than 0.05 at the paired upper bound;
8. no field supplies more than 20% of total absolute semantic contribution;
9. for CB, no gold class supplies more than 15% of total absolute known
   semantic contribution;
10. on the full 3,240-row authoritative training scope, zero-variance share is
    at most 40%, versus 1,571/3,240 or 48.5% for the original reward.

Gate 10 needs a separate raw full-training replay because the published D3a
file intentionally contains only the 1,438 active groups. That replay remains
training-only and may not regenerate completions; it must score the already
locked k=8 outputs. No candidate can be finally selected until this evidence
exists.

The 50%, 0.10, 2-point and 5-point thresholds are practical-effect gates, not
claims that those numbers are natural laws. They were chosen before opening
candidate aggregates: half the products is a material departure from the old
5.3% three-level exception; ten points is a visible resolution gain; and a
two-point alignment margin allows small reward-composition trade-offs without
accepting a meaningful semantic regression.

## 6. Complexity-aware selection

Passing the universal gates does not automatically justify a more complicated
reward.

1. Remove candidates that fail an applicable gate.
2. Prefer UA over U only if unknown-abstention alignment improves by at least
   0.10 with a paired lower bound above zero, while known alignment and
   discrimination lower bounds are no worse than -0.02 and harmful coverage's
   upper bound is no worse than +0.02.
3. Prefer CB over the surviving uniform candidate only if class-balanced known
   alignment improves by at least 0.03 with a lower bound above zero, while the
   same known, resolution, coverage and dominance safeguards pass.
4. Any unresolved tie goes to the simpler policy: U, then UA, then CB.
5. If nothing passes, stop Phase D and do not launch GPU training.

This chooses a reward for an offline design objective. It does not establish
that GRPO with that reward will improve model quality.

## 7. Contribution and sensitivity accounting

Field contribution is the absolute post-normalization semantic contribution
summed by field and divided by the total across fields. Malformed floors and
rule adjustment are separate channels. For CB, a field's absolute contribution
is allocated across its gold class keys in proportion to their locked class
weights. Gold-unknown `details` contributes to unknown behavior, not a known
class.

The aggregate report must also show CB with all class weights replaced by 1.0,
CB with a narrower `[0.75,1.5]` cap, and all primary metrics by gold-known field
count. These are sensitivity analyses, not additional candidates or hidden
winner-selection opportunities.

## 8. Audit boundary and next action

The executable artifact reads the original baseline, the payoff/scale contracts
and only the raw replay manifest. It records the gzip path, byte count and hash
from that manifest but does not open the gzip. Its explicit flags say candidate
aggregates, rankings, thresholds and winner selection remain undone.

Next, implement the D3 aggregator against this exact contract. Only then open
the candidate replay records once and publish U, UA, CB and original results in
one deterministic report.
