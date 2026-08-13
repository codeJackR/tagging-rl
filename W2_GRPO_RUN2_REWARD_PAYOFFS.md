# GRPO Run 2 reward payoff contract

**Version:** `grpo-run2-reward-payoffs-v1`
**Status:** ordinal payoff contract complete; numeric scales and weights not yet selected
**Created:** 2026-08-11
**Compute used:** none
**Selection evidence allowed:** authoritative SFT-training rollouts only
**Selection evidence prohibited:** legacy frozen 300, SFT validation, future confirmation data

This document defines what candidate rewards must prefer before any numeric
constants are chosen or candidate code is written. It converts phrases such as
"reward partial correctness" into testable ordering constraints and exposes
strategic shortcuts while they are still cheap to fix.

## 1. Why an ordinal contract comes first

The original reward assigns only three observed totals on the active starting
policy: 1, 2 and 4. It varies in every selected k=8 group, but 94.7% of groups
contain only two total values. Run 2 therefore needs finer semantic ranking,
not merely another positive component.

Choosing weights first would make it easy to rationalize an accidental payoff
after seeing replay results. This document instead locks the required ordering:
for example, replacing an abstention with a correct answer must help, while
replacing it with a wrong answer must hurt. Numeric scales come later and must
satisfy every ordering here.

## 2. Symbols and required inequalities

The symbols are ordinal placeholders, not numbers.

| symbol | meaning |
|---|---|
| `FLOOR` | fixed payoff for an output that fails the candidate format/completeness gate |
| `C` | correct prediction on a gold-known field |
| `A` | explicit abstention on a gold-known field |
| `W` | committed wrong prediction on a gold-known field |
| `UA` | correct abstention on a gold-unknown field |
| `UC` | unsupported commitment on a gold-unknown field |
| `P(q)` | partial multi-value payoff at set quality `q` |
| `RULE(v)` | bounded coherence adjustment for `v` rule violations |

Required orderings:

1. `C > A > W > FLOOR`.
2. `UA > UC` for candidates that score gold-unknown behavior.
3. `P(q)` increases strictly as set quality `q` improves.
4. `P(0) = W` and `P(1) = C`.
5. Each additional rule violation strictly lowers an otherwise identical
   completion until the documented safety cap is reached.
6. The worst complete, vocabulary-valid payoff after the maximum bounded rule
   cost remains above `FLOOR`. Malformed output must never become an escape from
   semantic penalties.

The position of `A` within the continuous multi-value curve is deliberately
not numeric yet. The scale-design step must state what set quality is good
enough to beat abstention.

## 3. Candidate semantic gate

Semantic credit is calculated only if all gate conditions pass:

1. literal output is one JSON object, with no prose or Markdown wrapper;
2. duplicate JSON keys are rejected rather than silently keeping the last one;
3. the object contains exactly the pack's 15 expected fields;
4. there are no extra fields;
5. every field has the required scalar/list/null shape;
6. every non-null value is in the controlled vocabulary;
7. the multi-value `details` list is nonempty;
8. `details` is either exactly `["unknown"]` or contains canonical detail
   values with no `unknown` mixed into the list;
9. `details` contains no duplicate values and respects its maximum length.

Any gate failure receives `FLOOR`; field and rule components are not added.

This gate is intentionally stricter than the current verifier. Today `{}`,
`{"details": []}`, and `{"details": ["unknown", "none"]}` all pass the
verifier. They are strategically ambiguous for a ternary reward:

- `{}` could collect missing-field abstention credit without emitting the task;
- `[]` conflicts with the explicit target conventions `["none"]` and
  `["unknown"]`;
- mixing `unknown` with a committed detail lets one output claim and abstain at
  the same time.

Run 2 candidate code must reject all three at its semantic gate. The base
verifier is not changed by this design step; the stricter behavior belongs to
the versioned Run 2 reward contract and must receive direct unit tests.

Cross-field rule violations do **not** fail this gate. A structurally valid
completion can earn semantic credit and then receive a separate per-violation
coherence cost. This preserves information about partially useful outputs.

## 4. Gold-known scalar and not-applicable payoffs

Gold-known includes both ordinary labeled values and `not_applicable`. Null is
a real prediction of not-applicability, not an abstention.

| gold state | model output | classification | payoff |
|---|---|---|---|
| labeled value `x` | exactly `x` | correct | `C` |
| labeled value `x` | `unknown` | abstain | `A` |
| labeled value `x` | another canonical value | wrong commitment | `W` |
| labeled value `x` | `null` | wrong not-applicable claim | `W` |
| not applicable | `null` | correct | `C` |
| not applicable | `unknown` | abstain | `A` |
| not applicable | any canonical value | wrong applicability claim | `W` |
| either gold-known state | missing/OOV/wrong shape | gate failure | `FLOOR` for the record |

Local replacement tests implied by the table:

- `unknown -> correct` must increase total reward;
- `unknown -> wrong` must decrease total reward;
- `wrong -> correct` must increase it more than `wrong -> unknown`;
- emitting more non-unknown fields has no reward by itself.

The last condition directly blocks coverage gaming.

## 5. Gold-unknown payoff

Gold `unknown` means the listing does not support a truth value. It is not a
hidden class to guess. Two policies remain useful enough to replay as a small
ablation:

| model output on gold-unknown | unknown-neutral candidate | unknown-aware candidate |
|---|---|---|
| explicit `unknown` | excluded from semantic mean | `UA` |
| canonical value | excluded from semantic mean | `UC` |
| `null` | excluded from semantic mean | `UC` |
| missing/OOV/wrong shape | record-level `FLOOR` | record-level `FLOOR` |

Required for the unknown-aware candidate: `UA > UC`. A commitment on unsupported
text must not beat honest abstention.

Why retain the unknown-neutral candidate? The weak labels' unknown state partly
reflects the frontier labeler's evidence judgment. Scoring it may teach useful
evidence discipline, but it may also teach that labeler's abstention bias. The
neutral/aware pair isolates this decision without touching validation data.

Known-field and unknown-field components must be normalized separately before
combination. Otherwise products with many unknown labels would dominate the
reward merely because they contain more abstention opportunities.

## 6. Multi-value `details` payoff

For a gold-known detail set `G` and a committed predicted set `P`, quality `q`
is an order-insensitive set score. The exact formula—set-F1 or another bounded
monotonic set score—is selected in the scale-design step.

| model output | treatment | payoff requirement |
|---|---|---|
| exact set, any order | fully correct | `P(1) = C` |
| nonempty partial-overlap set | partial | `W < P(q) < C` for `0 < q < 1` |
| disjoint canonical set | committed wrong | `P(0) = W` |
| exactly `["unknown"]` | abstain | `A` |
| `null` with gold not-applicable | correct | `C` |
| `null` with labeled gold | wrong | `W` |
| empty list | contradictory/ambiguous form | record-level `FLOOR` |
| list mixing `unknown` and values | contradictory form | record-level `FLOOR` |
| duplicate/OOV/too many values | invalid form | record-level `FLOOR` |

Monotonicity fixtures must show:

- adding a missing true detail without adding a false detail increases reward;
- removing a false extra detail without removing a true detail increases reward;
- exact set beats every partial set;
- zero-overlap commitment scores below abstention;
- whether a low-overlap partial set beats abstention is fixed explicitly by the
  later numeric scale, not left accidental.

For gold-unknown `details`, `["unknown"]` is `UA` in the unknown-aware candidate;
any committed list or `null` is `UC`.

## 7. Rule-coherence payoff

Rule handling is dense rather than binary:

| comparison | required ordering |
|---|---|
| same semantic fields, zero vs one violation | zero violations ranks higher |
| same semantic fields, one vs two violations | one violation ranks higher |
| same semantic fields, `v` vs `v+1` below cap | `RULE(v) > RULE(v+1)` |
| count above safety cap | bounded at documented minimum |

The later formula must normalize or cap rule cost so a product capable of many
rules cannot dominate solely through rule count. The cap is a numerical safety
bound, not a return to the original one-bit compliance reward.

No separate positive "clean" bonus is required. Starting every clean output
with free reward would recreate a saturated component. Zero violations should
mean zero cost; violations subtract.

## 8. Candidate family to replay

All candidates share the strict semantic gate, known-field ternary ordering,
multi-value partial credit, per-rule cost, field-count normalization, bounded
total range and identical locked rollout groups.

### Candidate U: uniform, unknown-neutral

- Uniform contribution from each gold-known field.
- Gold-unknown fields excluded.
- Purpose: test whether dense known-field feedback and rule costs alone improve
  resolution without learning the weak labeler's abstention policy.

### Candidate UA: uniform, unknown-aware

- Candidate U plus a separately normalized unsupported-claim component.
- Gold-unknown abstention ranks above commitment.
- Purpose: test whether evidence-aware abstention reduces the overcommitment
  behavior suggested by Phase C.

### Candidate CB: capped class-balanced, unknown-aware

- Candidate UA plus bounded class weights derived only from authoritative
  training support.
- Weight normalization keeps the mean known-field scale comparable to U/UA.
- Rare-class weights are capped; no single class or attribute may dominate.
- Purpose: test alignment with attribute-macro-F1 after Phase C showed that
  total exact correctness can rise while macro-F1 falls.

This is the maximum candidate family for offline replay. Additional reward
ideas require a new predeclared rationale rather than being added after results
are visible.

## 9. Whole-record adversarial payoff table

| output strategy | required result | shortcut blocked |
|---|---|---|
| malformed JSON with some recognizable correct text | `FLOOR` | semantic credit from unparseable prose |
| `{}` | `FLOOR` | empty-object abstention farming |
| 15 explicit `unknown` values | complete record; `A` on known fields | free format/compliance bonus or hidden coverage reward |
| replace one abstention with the correct value | strictly higher | all-abstain local optimum |
| replace one abstention with a wrong value | strictly lower | guessing for coverage |
| predict one common class in every field | wrong where unmatched | majority-class flooding |
| all fields semantically identical, add one rule violation | strictly lower | binary clean/dirty saturation |
| improve one field while leaving all others unchanged | strictly higher | whole-record all-or-nothing ties |
| product with 2 known fields vs product with 15 known fields at same average quality | comparable normalized semantic scale | label-density domination |
| one rare correct class under CB | bounded increase | inverse-frequency explosion |
| exact known fields but unsupported commitments on unknown fields | equal under U; lower under UA/CB | concealed evidence overcommitment |
| partial `details` overlap vs disjoint details | partial ranks higher | exact-only multi-value tie |

## 10. Constraints the numeric scale must satisfy

The next substep may choose component ranges and weights only if every candidate
satisfies all of these:

1. total reward has a finite documented minimum and maximum;
2. `FLOOR` is the unique outcome for gate failure and is lower than every
   complete vocabulary-valid record after bounded rule cost;
3. one correct-for-abstain replacement always increases total reward;
4. one wrong-for-abstain replacement always decreases total reward;
5. adding a rule violation never increases total reward;
6. field-count normalization prevents high-density products from producing a
   larger scale solely because they have more scorable labels;
7. unknown behavior is separately normalized where scored;
8. class balancing preserves the candidate's overall component range;
9. affine scale choices are documented so apparent gains are not merely larger
   numerical units;
10. all candidates can be evaluated on the exact D1 ordered SKU and rollout-key
    hashes without regeneration.

## 11. Claims and decisions intentionally deferred

This document does **not** decide:

- numeric values for `C`, `A`, `W`, `UA`, `UC` or `FLOOR`;
- component weights;
- the rule cap or rule-cost magnitude;
- the multi-value partial-credit formula or abstention crossover;
- the class-weight formula or cap;
- which candidate wins;
- whether the active training pool should be widened;
- whether `beta > 0` should be tested.

Those decisions require explicit component ranges and then paired replay on the
training-only artifacts. No GPU work is authorized by this payoff contract.

## 12. Next decision gate

Convert the ordinal symbols into bounded component scales. Before replaying any
candidate, prove mechanically on synthetic fixtures that all inequalities and
adversarial rows above hold. Only then implement the reusable candidate reward
functions.
