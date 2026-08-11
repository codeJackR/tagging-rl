# GRPO run 1: formal failure hypotheses and evidence

**Status:** diagnosis document, written 2026-08-11 after the locked evaluation,
paired bootstrap, attribute/class/rule decomposition and training-reward replay
of `runs/grpo-first-300` were complete. Every number cited here traces to a
hash-pinned artifact listed in the provenance section. This document proposes
explanations; it does not modify the locked negative result recorded in
`runs/grpo-first-300-closeout.json`.

**Scope note.** The frozen 300-row set stopped being blind the moment these
analyses were run. Hypothesis tests proposed here must therefore be measured on
non-frozen validation data; the frozen set may confirm a final pre-declared
comparison only.

---

## 1. What happened

One GRPO training run was executed from the locked SFT baseline
(`runs/sft-combined-2epoch/checkpoint-406`) at training commit `19be0ed`:
300 optimizer steps, one prompt and eight sampled completions per step
(2,400 rollouts), over a seeded shuffle of the 1,565-row cap-four difficulty
pool. Rewards were three binary components weighted `1:1:2`: format validity,
verifier vocabulary/rule compliance, and whole-record golden agreement.
Key optimizer settings: lr `5e-6`, 30-step warmup, cosine decay, `beta=0.0`
(no KL term, no reference model), DAPO loss, temperature 0.7 / top-p 0.95,
`scale_rewards="group"`, seed 42.

Mechanically the run was clean: 300/300 steps, all logged values finite, no
truncated completions, no clipping, peak 4.70 GiB reserved, and a final adapter
byte-identical to an earlier execution of the same seed that completed training
but failed during publication
(`741a189a...d3c`). This is strong evidence of same-runtime reproducibility; it
does not by itself establish that the objective or optimization path was
correct.

The predeclared frozen comparison then failed:

| metric (frozen 300) | locked SFT | GRPO | delta |
|---|---:|---:|---:|
| macro-F1 | 0.6411 | 0.6223 | **−0.0188** |
| selective macro-F1 | 0.7170 | 0.6578 | **−0.0591** |
| coverage | 94.30% | 96.77% | **+2.47 pp** |
| schema validity | 100% | 100% | 0 |
| vocabulary validity | 88.67% | 89.33% | +0.67 pp |
| rule violations | 12 | 28 | **+16** |

The paired row bootstrap (5,000 replicates, seed 20260801, same resampled rows
for both models) puts the macro-F1 delta at `[−0.03117, −0.00581]` with 99.78%
of replicates below zero, the selective delta at `[−0.07094, −0.03378]` with
100% below zero, and the coverage delta at `[+0.01855, +0.03122]` with 100%
above zero. The regression is directional, not sampling noise against these
labels.

## 2. The constraining fact

`evalharness/replay_rewards.py` re-ran the exact production reward callbacks
on the frozen greedy outputs of both models:

| exact training reward on frozen outputs | SFT | GRPO | delta |
|---|---:|---:|---:|
| format passes (weight 1) | 300 | 300 | 0 |
| vocabulary/rule passes (weight 1) | 256 | 243 | −13 |
| golden-agreement passes (weight 2) | 59 | 50 | −9 |
| mean weighted total (max 4) | 2.2467 | 2.1433 | **−0.1033** |

Pass/fail transitions show churn, not a uniform shift: compliance gained 10
rows and lost 23; agreement gained 9 and lost 18.

Meanwhile, sampled training reward rose across the run's three 100-step blocks
(weighted 2.9625 → 3.2113 → 3.2325; agreement 0.5025 → 0.6200 → 0.6325),
although blocks contain different shuffled products and are not a controlled
learning curve.

**Constraint on all hypotheses:** later shuffled training blocks had higher
observed sampled reward, but this does not prove that the policy improved its
training objective because the blocks contain different products. Establishing
learning would require evaluating multiple checkpoints on the same prompts
under the same decoding and sampling conditions. What is established is that
the final policy got worse on held-out products under deterministic decoding,
both under macro-F1 and under the exact reward callbacks. Therefore simple
reward/metric disagreement is insufficient on its own, but the evidence does
not yet distinguish failed learning from a generalization or decoding-transfer
failure.

---

## 3. Hypotheses

Each hypothesis states a mechanism, the supporting evidence, what it does not
explain, and a confidence grade. The hypotheses are not mutually exclusive;
H1 and H2 are two views of the same reward design and likely compounded.

### H1. The reward offered upside for guessing without penalizing clean wrong commitments

**Confidence: medium. Behavioral change confirmed; exact incentive mechanism not yet isolated.**

**Mechanism.** Under the locked pass definition, a gold-`unknown` field is
excluded from scoring, but answering `"unknown"` on a gold-labeled field fails
whole-record agreement exactly as a wrong value does. For an otherwise valid,
rule-compliant record, both abstaining and making a clean wrong guess usually
earn format plus compliance reward (total 2), while a correct guess can raise
the total to 4 only when every other scorable field also agrees. The reward
therefore offers occasional upside for guessing while usually assigning no
additional penalty to a clean wrong guess. Guessing is not universally
dominant: a guess that creates a rule violation can lower the total to 1, and
the golden reward is record-level rather than field-level. This asymmetric but
sparse incentive could shift probability mass from `"unknown"` toward committed
answers.

**Evidence.**

1. Coverage rose +2.47 pp with 100% of 5,000 paired bootstrap replicates above
   zero. This is the most statistically unambiguous behavioral change in the
   run.
2. Selective macro-F1 (accuracy conditioned on committing) fell 5.91 points
   with 100% of replicates below zero. The additional commitments were wrong
   more often than the baseline's commitments.
3. `collar_type` is the extreme case and alone accounts for 47.4% of the total
   negative attribute-delta magnitude: coverage rose +3.96 pp to exactly
   100% (abstention eliminated), exact match fell 4.46 points, selective
   macro-F1 fell 19.98 points.
4. Rare-class F1 collapsed in a direction consistent with overcommitment to
   common answers: `polo` 1.000 → 0.500 (support 6), `notched lapel`
   0.250 → 0.000 (7), `hooded` 0.889 → 0.667 (8), `band` 0.364 → 0.182 (8);
   also `closure=lace_up` and `neckline=cowl` each −0.444 (7) and
   `pattern=camouflage` −0.350 (7). These class deltas do not by themselves
   prove that common-answer frequency increased; a prediction-transition
   analysis is still required.
5. Because macro-F1 averages per class, degraded rare classes are maximally
   expensive under exactly the headline metric. This explains why a modest
   behavioral shift produced a clear headline regression.
6. The direction contradicts the failure mode the plan predicted
   (empty-but-valid conservatism) and matches its mirror image. The incentive
   table makes that behavior plausible, but does not prove it caused the shift.

**Does not explain.** Whether the higher later-block sampled reward reflects
learning or product mix. Also does not by itself explain the rule-violation
doubling, though more committed fields mechanically create more opportunities
to violate cross-field rules.

**Falsification for run 2.** Add an explicit ordering correct > abstain > wrong
at field level (the plan's ternary shape). Before training, directly classify
gold-known field transitions from SFT to GRPO as abstain→correct,
abstain→wrong, correct→wrong and wrong→correct, and measure predicted-class
frequency shifts. If H1 is correct, the first run should show excess
abstain→wrong transitions; under the ternary reward, coverage should return
toward the SFT level, selective macro-F1 should recover, and `collar_type`
coverage should drop below 100% on non-frozen validation data.

### H2. Two saturated components left a sparse, all-or-nothing gradient signal

**Confidence: high. Primary signal-quality hypothesis.**

**Mechanism.** Component means across all 2,400 training rollouts were format
1.0000, compliance 0.9654, agreement 0.5850. A component with no within-group
variance contributes nothing to a group-normalized advantage. In practice the
gradient was driven almost entirely by the binary whole-record agreement bit,
which assigns identical reward (0) to a completion with 14 of 15 fields correct
and one with 3 of 15 correct. Such a signal can reinforce any token-level
behavior correlated with complete passes on the sampled rows, while field-level
accuracy improvements that do not complete a record earn nothing. The reward
statistics alone do not identify overcommitment as that learned correlate; H1
requires the separate behavioral transition test above.

**Evidence.**

1. Component saturation measured directly: 100.0% / 96.5% / 58.5% mean pass
   rates during training. More importantly, direct grouping of the 2,400
   durable rollout records shows within-group variation in format for 0/300
   groups, compliance for 33/300, golden agreement for 199/300, and weighted
   total for 202/300. Golden agreement varied in 199 of the 202
   gradient-bearing groups (98.5%); compliance varied in only 33 (16.3%).
2. 98 of 300 optimizer steps (32.7%) had zero within-group reward variance,
   and the zero-gradient steps align exactly with those 98 groups. One third
   of the paid updates carried no policy-gradient signal. There were 202
   gradient-bearing groups, although the optimizer and learning-rate schedule
   still advanced through all 300 steps.
3. Frozen replay transitions show undirected churn with net loss (compliance
   +10/−23, agreement +9/−18). This is consistent with a noisy sparse
   objective but does not uniquely identify it as the cause.
4. Rule violations more than doubled (12 → 28; `bodycon_is_tight` +7,
   `pants_length_subset` +4, spread over 18 newly-violating products, not one
   pathological row). The compliance component that should defend the rules was
   96.5% saturated in training and therefore nearly signal-free.
5. This was measurable offline: the difficulty-scoring artifact already
   contained 28,800 k=8 rollouts of the starting policy in exactly GRPO's
   group structure. Replaying the exact reward over those groups could have
   measured component saturation and zero-variance share before dispatch. That
   full replay has not yet been performed, so its result must not be assumed to
   equal the observed 32.7% training share.

**Does not explain.** On its own, sparsity predicts inefficiency and variance,
not necessarily a *directional* regression. Whether H1 supplied a directional
incentive and H3 allowed additional drift requires the proposed counterfactuals.

**Falsification for run 2.** Replace whole-record binary agreement with dense
per-field credit (fraction of scorable fields correct, optionally
class-balanced). Offline replay over the existing k=8 rollouts should first
confirm the zero-variance share drops well below 32.7%; the training run should
then show reduced churn on a held-out validation slice.

### H3. `beta = 0`: no KL anchor to the SFT policy, so weakly rewarded behaviors could drift

**Confidence: medium. Documented execution gap; causal contribution untested.**

**Mechanism.** With no KL term and no reference model, any behavior the reward
measures only weakly or sparsely can drift. Abstention calibration and
rare-class fidelity receive no direct dense reward. Cross-field rules are seen
by compliance reward, but that component varied in only 33 of 300 groups. A KL
anchor is the standard counterweight; it was absent.

**Evidence.**

1. Config provenance: the smoke config locked `beta=0.0` explicitly as a
   simplification, with the tracker stating the full-run value "must be locked
   separately after the smoke, and any change from zero must receive its own
   memory preflight." No such decision was ever recorded; the full-run
   contract table contains no beta row, and the construction gate confirms
   beta zero at dispatch. This is an acknowledged open item that shipped
   unresolved.
2. The drift signature is consistent with the mechanism: quality loss
   concentrated in weakly rewarded behaviors, while format remained perfect.
   Higher sampled reward in later training blocks is descriptive only because
   the product mix changed.
3. Rule-violation growth spread across 18 products rather than one pathological
   row. Some affected behavior belongs to categories the pool underweights
   (see H5), which is consistent with but does not establish unopposed drift
   away from SFT-era habits.

**Does not explain.** Why drift went in the specific direction of
overcommitment (H1 supplies the direction). Cannot be confirmed from this run
alone because no beta > 0 counterfactual exists.

**Falsification for run 2.** Train the same recipe with beta > 0 against the
SFT reference. If H3 contributes, rule-violation growth and rare-class F1 loss
should shrink at comparable training-reward gain.

### H4. Distribution mismatch between sampled training and greedy evaluation

**Confidence: low-medium. Objective distinction is real; causal contribution is unmeasured.**

**Mechanism.** GRPO optimized the temperature-0.7 / top-p 0.95 sampling
distribution; the deployment metric decodes greedily. Raising the expected
sampled reward does not have to improve, and can degrade, the argmax output.
The run had no instrument measuring greedy quality during training, so the
sampled-vs-greedy gap was invisible until the frozen evaluation.

**Evidence.**

1. Sampled training reward was higher in later blocks while greedy frozen
   reward fell. Because those blocks contain different products, this is only
   consistent with the hypothesis, not evidence that the sampled objective
   improved while greedy quality declined.
2. The replay artifact's own caveat records that greedy frozen replay is not
   an off-policy estimate of the sampled training return; the two objectives
   were never the same quantity.
3. No greedy validation curve exists for checkpoints 100/200/300; checkpoints
   were retained but never quality-evaluated before the final frozen run, so
   this mechanism was structurally undetectable during training.

**Falsification for run 2.** At every checkpoint, evaluate both greedy outputs
and repeated sampled outputs on the same non-frozen validation prompts. A
greedy-only curve can detect degradation but cannot isolate decoding mismatch
from general overfitting. If sampled expected reward rises on those fixed
prompts while greedy quality falls, H4 is operating and the run can be stopped
early.

### H5. Curriculum selection shifted the training distribution and abandoned always-solved behavior

**Confidence: low-medium. Plausible contributing factor; evidence is correlational.**

**Mechanism.** The pool admits only products with mixed outcomes under eight
samples of the SFT policy (0 < pass rate < 1), then family-caps them. Two
consequences: optimization pressure applies to a boundary-case distribution
that differs from the frozen set's distribution, and the 782 always-pass
products contribute zero reinforcement, so the behaviors that made them pass
(including correct abstention and rule compliance on easy rows) receive no
defense during training. With beta = 0 (H3), nothing else defends them either.

**Evidence.**

1. Measured distribution shift of the active pool versus the full catalog:
   category TVD 5.53%, store TVD 7.85%; Thursday Boots rises to ~17.6% of
   rows, dresses fall from 6.89% to 5.41%.
2. `bodycon_is_tight` (+7 violations) is a dress-family rule, and dresses are
   underweighted in the pool relative to the frozen set. This ecological
   correlation does not establish that category underweighting caused those
   seven transitions.
3. The always-pass exclusion is total: no replay or mixing fraction existed,
   so 782 products' worth of correct behavior went unreinforced for 300 steps.

**Does not explain.** The coverage increase or the reward's internal incentive
structure. TVD magnitudes are modest; this is a contributor, not the driver.

**Falsification for run 2.** Monitor an easy non-frozen validation slice and, in
a separate retention arm, preserve easy behavior through a KL term or a small
SFT replay objective. Simply adding always-pass rows to the GRPO pool may waste
steps because they can again produce zero within-group reward variance. If H5
contributes, explicit retention should reduce easy-slice regression.

### H6. The run had too few independent, gradient-bearing groups to estimate a reliable effect

**Confidence: low as a causal explanation; high as an uncertainty limitation.**

**Mechanism and evidence.**

1. 300 steps at one prompt per step visited 300 of 1,565 pool rows (19.2%)
   exactly once; no product was ever revisited.
2. The run contained 202 gradient-bearing single-group updates from a
   high-variance binary signal and only one training seed. The optimizer and
   scheduler nevertheless advanced through all 300 steps.
3. Difficulty rates guiding the pool were measured once from eight samples of
   the *starting* policy (0.125 quantization, single seed). They may become less
   representative as the policy moves, but this run does not establish how
   much, if any, of the 32.7% zero-variance rate came from staleness.
4. The rising block rewards cannot be read as a learning curve because blocks
   contain different shuffled products; no controlled within-run learning
   measurement exists.

**Falsification for run 2.** Repeat matched evaluations on fixed non-frozen
validation prompts across checkpoints and use multiple training seeds before
attributing a direction to run length. A longer run alone is not a clean test:
a mis-specified objective or absent KL anchor could make the regression worse.

---

## 4. Explanations ruled out by the evidence

| candidate explanation | ruled out by |
|---|---|
| Broken training mechanics (bad gradients, corrupt updates) | Zero-gradient steps align exactly with zero-variance groups; all 300 logs finite; adapter byte-identical across two independent executions of the same seed |
| Simple reward/metric mismatch as the primary story | Frozen replay: GRPO also lost on the exact training reward (2.2467 → 2.1433), so the frozen failure is not objective disagreement alone; whether training-objective learning or transfer failed remains unresolved |
| The predicted empty-JSON / conservatism hack | Format mean 1.0000, mean completion 108.86 tokens, coverage *rose*; the observed exploit is the opposite of the predicted one |
| Evaluation artifact or drift | Freeze checksum verified before scoring; 300/300 predictions, SKU sets equal, order equal; prediction and report hashes pinned |
| Truncation or length effects | No completion hit the cap in training or frozen generation; `mask_truncated_completions` never triggered |

---

## 5. Preventability analysis

Three of the six were preventable with checks adjacent to work that already
existed; the rest are covered by standard monitoring that was absent.

| hypothesis | prevention that would have caught it | cost |
|---|---|---|
| H1 | Incentive analysis from the policy's perspective: enumerate total format/compliance/golden payoffs for correct, clean-wrong, rule-violating and abstaining outputs. This would have exposed the unpenalized clean-wrong case without incorrectly assuming universal dominance | Pencil and paper |
| H2 | Replay the reward offline over the existing 28,800 k=8 difficulty rollouts; compute component means and within-group variance. This measures the starting-policy saturation and zero-variance share before dispatch; it does not guarantee the later training share will be identical | One CPU script |
| H3 | A launch gate that blocks while any "must be locked separately" deferral remains unresolved, plus a smoke-to-full config diff requiring re-justification of every inherited value | Process |
| H4, H6 | Greedy and repeated sampled evaluation on the same non-frozen validation prompts at every checkpoint. At observed speed, greedy generation is about 80 seconds for a 100-row slice; larger or repeated sampled evaluations cost proportionally more | Minutes |
| H5 | Easy-slice monitoring plus an explicit KL or SFT-replay retention arm; plain always-pass GRPO rows may remain zero-variance | Small |

The measurement discipline that existed (lock, one-shot frozen eval, paired
bootstrap, replay, decomposition) is what made this failure cheap to diagnose
and is retained unchanged. The gap was pre-dispatch analysis of incentives and
signal, not post-hoc rigor.

---

## 6. Predeclared implications for run 2

Consistent with the closeout's constraint, follow-up selects on non-frozen
validation evidence. First, replay both the original and proposed dense rewards
over the existing k=8 rollouts; no GPU run is dispatched until the measured
component and total group-variance profiles are recorded. The recommended
single primary training ablation is **reward densification** (H1 + H2 together,
since both live in the same reward definition): per-field credit with correct >
abstain > wrong ordering.

To preserve attribution, `beta > 0` must not be silently combined with the
primary reward change. The comparison sequence is:

1. Original reward, `beta=0`: existing run 1.
2. Dense reward, `beta=0`: isolates reward design.
3. Dense reward, `beta>0`: a separate matched arm isolating the added KL anchor.

Instrumentation shared by every new arm: greedy and repeated sampled evaluation
on the same non-frozen validation slice at every checkpoint (H4), offline
variance measurement before dispatch (H2), and an easy-slice monitor (H5).

Predicted observations if the hypotheses are correct, measurable without the
frozen set:

1. Offline replay of the dense reward over the existing k=8 rollouts shows
   zero-variance groups well below 32.7% (H2).
2. Under the dense ternary reward, validation coverage moves back toward the
   SFT level and selective macro-F1 recovers; `collar_type` coverage leaves
   100% (H1).
3. With beta > 0 at matched settings, rule-violation growth and rare-class
   F1 loss shrink (H3).
4. On fixed validation prompts, sampled and greedy checkpoint curves either
   improve together or diverge (H4 detected directly, run stopped cheaply).

---

## 7. Provenance

All numbers cited above derive from these artifacts (hashes as recorded in
`W2_GRPO_BLOG_TRACKER.md` and `runs/grpo-first-300-closeout.json`):

| artifact | SHA-256 |
|---|---|
| final adapter weights | `741a189a23f948248a6c9067c401d66dde9187328748ea815441307241ab9d3c` |
| frozen GRPO predictions | `f14f95ca0d5bde1bf8ece0927b2f02975fed89b1da1cf6da7ebc34ecd5a0573e` |
| frozen evaluation report | `478fec2c75ca1477772d45db44ce12beaa7e2410f7fb07d2ed860e63aed075f1` |
| paired bootstrap | `f40d0fe8a27a6ca76b6fed2ed0edc268f196026b6c71a35101fb521d2583d251` |
| bootstrap replicate stream | `7c6364be2f38153f6b4d6661cb655fb92eff30d9341651f86d5bb9c518689c07` |
| attribute/class/rule decomposition | `973ef2d6ca8b739cf26fd66ae09f9e437115d63545932c7cd93514956b94d638` |
| training-reward replay | `4ba819c1491677f210873d13025a36e2bf317e1991cea2dc2ecca16e49050aa4` |
| experiment closeout | `b5ba9f8a6e16736e0021afc89a5bb09ac06cee84abf0b40738fea86c799dba4d` |

Training evidence (rollouts, scalar logs, phase timings, lifecycle events) is
indexed by the completed-run manifest
(`f266641137e6303ddb781eda72b436163954ddf66516a82241ed34b4ac872247`) with a
durable second copy at `../tagging-rl-artifacts/grpo-first-300`.
