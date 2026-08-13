# GRPO Run 2 bounded reward-scale contract

**Version:** `grpo-run2-reward-scale-contract-v1`
**Status:** numeric scales selected and mechanically proved; candidate replay not run
**Created:** 2026-08-11
**Parent contract:** `grpo-run2-reward-payoffs-v1`
**Compute used:** CPU only
**Selection evidence:** authoritative SFT-training labels plus locked starting-policy rule-count structure

This document maps the previously locked payoff ordering onto finite numbers.
It does not implement the production parser/reward functions, score candidate
rewards on rollouts, choose a winner, or authorize GPU training.

## 1. Locked numeric scale

| outcome or component | value/range |
|---|---:|
| correct known-field prediction (`C`) | `+1` |
| abstention on a known field (`A`) | `0` |
| wrong known-field commitment (`W`) | `-1` |
| correct abstention on a gold-unknown field (`UA`) | `+1` |
| unsupported commitment on a gold-unknown field (`UC`) | `-1` |
| normalized semantic component | `[-1, +1]` |
| maximum rule cost | `-0.15` |
| valid complete record | `[-1.15, +1]` |
| malformed-record floor (`FLOOR`) | `-1.25` |
| all possible record rewards | `[-1.25, +1]` |

The centered `-1/0/+1` scale makes the intended direction visible: correct is
positive, abstention is neutral, and an unsupported guess is negative. Its
absolute units are arbitrary; candidate comparisons must use identical rollout
groups and report ranking/variance effects rather than claim improvement merely
because these numbers differ from Run 1's `0..4` range.

The floor is lower than the worst valid record: `-1 - 0.15 = -1.15`, which is
still greater than `-1.25`. Malformed output therefore cannot escape a semantic
penalty by receiving a less-negative score.

## 2. Field normalization

Candidate U takes the ordinary mean of utilities over gold-known fields.
Candidate CB takes a weighted mean. Dividing by the number or summed weight of
the observed fields keeps both semantic scores inside `[-1,+1]`; a product with
15 known labels does not receive a larger reward scale solely because another
product has only two.

For any positive field weight:

- replacing `A=0` with `C=+1` increases the mean;
- replacing `A=0` with `W=-1` decreases the mean;
- replacing `W=-1` with `C=+1` produces twice the movement of `W -> A`.

The CPU fixtures prove these directions for every uniform field count from one
through 15 and for the minimum, middle and maximum class weights.

## 3. Unknown-aware combination

Known and unknown fields are averaged separately. Candidate UA/CB then combine
those two averages using the fixed cell composition of the active training
pool:

- known: `12,533 / 21,570`, approximately `0.581`;
- unknown: `9,037 / 21,570`, approximately `0.419`.

So, when a product contains unknown fields:

`semantic = 0.581 * known_mean + 0.419 * unknown_mean`

The artifact stores the exact fractions; the rounded values above are for
readability. A product with no gold-unknown field uses its known mean directly.
This prevents an absent component from shrinking its score.

These weights are not tuned against reward outcomes. They preserve the observed
training-cell composition while separate normalization prevents products with
many unknowns from dominating merely through field count. Candidate U remains
the explicit unknown-neutral comparison.

## 4. Partial credit for `details`

The set quality `q` is order-insensitive set-F1, bounded from zero to one. Its
payoff is:

`P(q) = 2*q - 1`

Therefore:

- no overlap: `P(0) = -1`, equal to a wrong commitment;
- set-F1 `0.5`: `P(0.5) = 0`, equal to abstention;
- exact set: `P(1) = +1`, equal to correct;
- every quality increase strictly raises reward.

This explicitly settles the deferred crossover: a partial committed set must
exceed `0.5` set-F1 to beat honest abstention. In the active pool, 917 labeled
`details` targets contain one value and 38 contain two, so set-F1 is useful but
does not get a separate larger component scale.

## 5. Capped class balancing

Candidate CB derives one weight from active training support only:

`weight = clip(sqrt(attribute median positive support / class support), 0.5, 2.0)`

The median is calculated separately inside each attribute. A median-support
class gets weight `1`. Rarer classes get more weight, frequent classes get less,
but no raw field weight leaves `[0.5,2]`. Thus the largest possible raw ratio is
four-to-one, not the unbounded ratio produced by inverse frequency.

Why this form:

- the active pool has 116 observed attribute/class pairs;
- support ranges from 1 to 1,151;
- 17 classes have fewer than five examples and 30 have fewer than ten;
- square root softens the response to those very small counts;
- clipping prevents one rare label from controlling a record;
- re-normalizing by summed weights keeps the final known score in `[-1,+1]`.

For a multi-label gold `details` field, its field weight is the mean weight of
its gold labels. Gold-unknown fields are not class weighted.

### 5.1 Zero active-support classes in the broader diagnostic scope

CB's immutable base map remains defined only by the 1,438-product active pool.
The separate 3,240-product Gate G10 diagnostic contains 13 valid gold
attribute/class pairs with zero active-pool support. For this broader diagnostic
only, each such class receives the existing maximum weight `2.0`.

This is the clipped limit of the same formula: as active support approaches
zero, the raw inverse-square-root weight grows beyond the cap, so clipping fixes
it at `2.0`. The extension may add only valid gold classes absent from the base
map. It must not replace any of the 116 active weights, use full-scope counts to
retune them, repair a class missing on an active product, or affect U/UA.

The locked extension contains 13 pairs, 53 class observations and 50 affected
non-active training products. Its ordered entry-ledger SHA-256 is
`aeb089a1081d7efd1a99ccb2124e7b7412ec71f2362509f3df20dc2aa5837416`.
No candidate rollout reward was inspected to choose this rule.

## 6. Rule cost

The coherence adjustment is:

`RULE(v) = -0.15 * min(v, 3) / 3`

The first three violations therefore cost `0.05` each; additional violations
cannot push the component below `-0.15`. There is no positive clean-output
bonus. This keeps clean completions at zero rule adjustment instead of creating
another saturated reward channel.

The cap was selected from locked training-only structure: among 11,504 active
starting-policy rollouts, 11,418 had zero violations, 81 had one, one had two,
and four had three; none exceeded three. The pack nevertheless contains 34
possible rules (25 written and nine derived), so the cap is needed to protect
the total range if a future completion triggers many at once. The `0.15`
maximum is intentionally small relative to the semantic range of width two.

This is a design calibration, not evidence that the candidate reward performs
better. That question belongs to paired replay.

## 7. Mechanical proof results

The executable contract checks all of the following:

1. `C > A > W > FLOOR`;
2. `UA > UC`;
3. `P(0)=W`, `P(0.5)=A`, `P(1)=C`, with strict monotonicity between them;
4. each rule violation strictly lowers reward until the cap;
5. the worst valid total `-1.15` remains above malformed `-1.25`;
6. correct-for-abstain raises and wrong-for-abstain lowers both uniform and
   class-weighted means;
7. known/unknown combination weights sum to one;
8. class weights and every component remain bounded.

The code fails closed on out-of-range set quality, nonpositive class support,
negative rule counts, invalid semantic scores and mismatched weight vectors.

## 8. Statistical interpretation boundary

The scale uses complete deterministic training structure, so no p-value or
confidence interval is appropriate. Medians and percentiles describe skewed
support/count distributions without letting the largest class dictate a
typical value. Candidate completion rewards were deliberately not calculated
during scale selection, avoiding outcome-guided tuning.

## 9. Durable evidence

- executable scale contract: `training/reward_scale_contract.py`;
- synthetic and real-structure checks: `tests/test_reward_scale_contract.py`;
- generated artifact: `runs/grpo-run2-reward-scale-contract.json`;
- parent ordinal contract: `W2_GRPO_RUN2_REWARD_PAYOFFS.md`;
- original-reward structural evidence:
  `runs/grpo-run2-original-reward-training-replay.json`.

The generated artifact records source hashes, the exact formulas and constants,
training-only distribution summaries, class-support bounds, rule-count
histogram, proof booleans, and an explicit flag that no candidate completion
reward was evaluated.

## 10. Next gate

Implement the strict semantic gate and the three candidate reward functions
from these locked ordinal and numeric contracts. Exercise them first on
synthetic adversarial records. Do not yet select a candidate or run GPU
training; paired candidate replay is the following phase.
