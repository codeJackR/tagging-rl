# GRPO Run 2 causal GPU experiment contract

**Version:** `grpo-run2-causal-gpu-experiment-v1`

**Status:** locked before either GPU training arm

**Question:** With the corrected training population, does replacing the
original coarse reward with selected dense Candidate UA improve the final
policy when everything else is held fixed?

## Causal comparison

| arm | role | reward functions | reward weights | beta |
|---|---|---|---|---:|
| A | corrected control | format validity, vocabulary/rule compliance, exact known-field agreement | `1,1,2` | 0 |
| B | treatment | Candidate UA | `1` | 0 |

Arm A runs first and must close successfully before Arm B starts. Both begin
from the same byte-identical SFT adapter. Run 1 is historical context only: its
training-pool boundary differs, so it cannot replace Arm A or identify the
effect of reward densification.

The intended causal difference is the reward definition and its necessarily
associated function/weight list. Output paths and run labels differ only to
keep evidence separate. No KL arm is included. A future `beta > 0` arm would be
a separate experiment because changing reward and KL together would destroy
attribution.

## Fixed training schedule

Both arms consume the exact 300-product JSONL and optimizer-step order in
`grpo-run2-causal-schedule-v1`. Products are selected without replacement from
the corrected 1,438-row, family-capped, authoritative SFT-training pool by
ascending SHA-256 of the public namespace, seed 42 and SKU ID. Trainer shuffling
is disabled. One product generates one group of eight completions and one
optimizer step.

This explicit schedule avoids relying on a library sampler whose order could
change across versions. It also makes the 300-step budget honest: the trial
tests reward behavior on 300 predeclared products, not all 1,438 eligible
products. Schedule composition and its distance from the source pool are
published, including categories/stores that receive zero rows by chance.

## Shared model and optimization configuration

- base model: `unsloth/Qwen2.5-1.5B-Instruct`;
- starting adapter: combined attention+MLP SFT checkpoint 406, exact adapter
  weight hash locked in the executable contract;
- LoRA: rank 16, alpha 16, dropout 0, bias none, targeting `q/k/v/o`,
  `gate/up/down` projections; 18,464,768 trainable parameters;
- seed and data seed: 42;
- optimizer: 8-bit AdamW, learning rate `5e-6`, betas `0.9/0.999`, epsilon
  `1e-8`, weight decay `0.001`, maximum gradient norm 1.0;
- schedule: cosine, 10% warmup (30 of 300 optimizer steps);
- generation: eight completions, temperature 0.7, top-p 0.95, repetition
  penalty 1.0, maximum prompt 600 and completion 170 tokens;
- batch: generation batch 8, per-device batch 8, gradient accumulation 1;
- objective: one GRPO iteration, DAPO loss, epsilon 0.2/0.28, group reward
  scaling, truncated-completion masking, explicit `beta=0`;
- precision/runtime: BF16, gradient checkpointing, no vLLM, one RTX 3090;
- duration: exactly 300 optimizer steps; checkpoints at 100, 200 and 300.

Scalar metrics and raw rollout evidence are retained at every step. Completion
tables are not printed to the process log. Checkpoints save model-only state,
with `save_total_limit=2`; step-100 evidence is exported before trainer
rotation, checkpoints 200/300 and the final adapter are retained. An interrupted
arm restarts from the beginning because exact optimizer-state resumption is not
available under the disk-safe model-only policy.

## Checkpoint monitoring and stop rule

Every retained checkpoint synchronously runs the Phase F monitor on the fixed
360-row non-frozen development population: one greedy pass and eight sampled
passes with seeds `20260813`–`20260820`. It reports representative, difficult,
middle and easy-retention views under the primary evaluator, original reward
replay and Candidate UA replay. The monitor must bind the exact checkpoint and
publish atomically before training may continue.

Evaluator launch failure, timeout, invalid publication or checkpoint mismatch
aborts immediately. Quality stopping uses a separate baseline-derived policy:

1. for higher-is-better metrics, threshold = baseline sampled mean minus the
   larger of two population standard deviations or a predeclared practical
   margin;
2. for rule-violation rate, threshold = mean plus the larger of two standard
   deviations or its practical margin;
3. greedy and sampled-mean observations are evaluated separately;
4. one breach warns; the same view/metric/decoding-mode must breach at two
   consecutive checkpoints to abort;
5. a clean intervening checkpoint resets that breach sequence;
6. rewards never trigger quality stopping, preventing either training reward
   from grading itself.

The exact numeric policy is embedded in the executable contract generated from
the full starting-SFT baseline. No single noisy checkpoint can stop an arm.

## Primary endpoint and treatment decision

The primary endpoint is Arm B minus Arm A at checkpoint 300 on representative
greedy macro-F1. Product/SKU is the paired unit. The comparison uses 10,000
paired product bootstrap replicates, seed `20260821`, and a 95% percentile
interval. Checkpoints 100 and 200 are trajectory diagnostics, not alternative
winner-selection opportunities.

Candidate UA replaces the original reward only if all conditions hold:

1. both arms and every required monitor close successfully;
2. primary macro-F1 improvement is at least 0.02 and its paired interval lower
   bound is above zero;
3. sampled representative macro-F1 mean is not lower;
4. greedy coverage, schema validity, vocabulary validity and easy-retention
   macro-F1 are each no worse by more than 0.03;
5. greedy rule-violation rate increases by no more than 0.02.

Otherwise retain Arm A. If either arm aborts or checkpoint-300 evidence is
missing, declare the causal experiment incomplete rather than selecting from a
partial trajectory. The exposed legacy frozen 300 and untouched confirmation
set cannot select a reward, checkpoint, threshold or beta.

## Resources, order and publication

The suite requires at least 3 GiB free before Arm A, at least 2.5 GiB before Arm
B and at least 2 GiB after each published arm. `save_total_limit=2` is
mandatory. Historical training peaked at about 4.92 GB allocated and the Phase
F evaluator at 3.39 GB, so their conservative additive estimate remains below
half of a 24 GB RTX 3090. The live callback must nevertheless require at least
6 GiB driver-free memory before launching each child evaluator.

Each arm uses detached, fail-closed execution with distinct final, staging,
control, monitor, failure and receipt paths. Existing paths are collisions, not
resume signals. Arm B cannot start until Arm A's final manifest, checkpoint-300
monitor receipt and post-run disk floor validate. Software, environment and
code identities may not change between arms.

## Interpretation boundary

This is a controlled single-seed engineering experiment, not a multi-seed
estimate of training-run variance. The paired product bootstrap measures
uncertainty across development products; it does not include variation from a
different optimization seed, GPU stack or schedule. A positive result supports
a causal claim for this locked run recipe and population. Broader claims require
replication. Final publishable performance still requires the separately
acquired untouched confirmation set.
