# Run 2 quality-policy amendment: compliance warns, quality aborts

**Version:** `grpo-run2-quality-policy-amendment-v1`
**Date:** 2026-08-14
**Status:** amended before any completed arm; evidence from an aborted attempt

## What changed

The checkpoint quality policy aborted training on any guardrail breached at two
consecutive checkpoints. It now aborts only on `macro_f1`, `selective_macro_f1`
and `coverage`. Rule-violation rate and vocabulary validity remain measured,
reported and warned on, in their own `recorded_only_breached_keys` field, but
cannot terminate a run.

The change applies identically to both arms. Applying it to one would make the
arms differ in more than their reward and destroy the causal claim.

## Why

Arm A's first production attempt was aborted at step 200. Its breaches were
rule-violation rate (greedy and sampled) and sampled vocabulary validity.
Its macro-F1 was **above** the SFT baseline at both checkpoints and rising:

| checkpoint | macro-F1 greedy | vs SFT baseline 0.8537 |
|---|---:|---|
| step 100 | 0.8577 | +0.004 |
| step 200 | 0.8616 | +0.008 |

Measured on the same 360 development prompts, the original `1:1:2` reward
raises the rule-violation rate well past what the policy allows:

| | rule violations vs SFT baseline |
|---|---|
| threshold permits | 1.72x |
| Arm A step 100 | 2.40x greedy, 2.54x sampled |
| Arm A step 200 | 2.00x greedy, 2.18x sampled |
| Run 1, completed 300 steps, frozen set | 2.33x |

Run 1 used this reward, completed all 300 steps and finished at 2.33x. Under
this policy it would have breached at every checkpoint. **The control arm was
disqualified by construction.**

## The defect being corrected

The guardrails were derived from the SFT baseline's own variability, as
`baseline + max(2 x population stddev, practical margin)`. That is a reasonable
way to ask "is this run behaving like the model it started from". It is the
wrong question for Arm A, whose entire purpose is to reproduce the degradation
the original reward causes, so that Arm B's dense reward can be shown to fix it.

The policy was written before any GRPO reference existed. Nothing available at
the time revealed that a control arm could not pass it.

For Arm B the same numbers carry the opposite meaning. Its dense reward prices
each rule violation directly, so failing to hold these metrics is that arm's
headline result rather than a reason to stop it early.

## What is deliberately unchanged

- Both arms remain bound to one reward difference and nothing else.
- Thresholds themselves are unchanged. Only the consequence of breaching two of
  them changed.
- A repeated `macro_f1`, `selective_macro_f1` or `coverage` breach still aborts.
- A run breaching compliance at every checkpoint remains stoppable on quality;
  a dedicated test covers exactly that case.
- Monitor failure and insufficient GPU headroom still abort immediately.
- The original contract, preflight and construction artifacts are retained
  unmodified as the original predeclaration.

## Honesty boundary

This is a predeclared policy changed after seeing data, which the project's
own rules resist. The justification is that the policy was under-specified
rather than unfavourable: it could not be satisfied by the experiment it was
written to govern, and that was demonstrable from Run 1's completed trajectory
without reference to Arm A's outcome. The amendment does not touch any metric
threshold, and it cannot make a failing arm look successful, because the
primary endpoint and its abort conditions are untouched.

The aborted attempt's evidence is retained at
`runs/grpo-run2-arm-a-aborted-attempt-1/` rather than deleted.

## Re-verification chain

Amending a pinned execution file invalidated the contract's lineage check, as
designed. The full chain was rebuilt at code commit `a08f8b6`:

| artifact | status |
|---|---|
| `runs/grpo-run2-causal-experiment-contract-v2.json` | `locked_no_gpu_training_dispatched` |
| `runs/grpo-run2-causal-preflight-v2.json` | `passed_read_only_no_training_dispatch` |
| `runs/grpo-run2-causal-construction-v2.json` | `both_arm_configs_constructed_no_trainer_no_dispatch` |

The v1 artifacts remain in place, unmodified.
