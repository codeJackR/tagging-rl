# GRPO Run 2 full-training raw replay contract

**Version:** `grpo-run2-full-training-replay-contract-v1`
**Locked:** 2026-08-12
**Purpose:** supply the separate raw evidence required by comparison Gate G10
**Status:** raw replay and Gate G10 result published and independently verified;
all candidates pass G10, with no ranking or selection

## 1. Question this artifact answers

For each dense candidate, does reward vary across the already-generated `k=8`
starting-policy completions for all 3,240 authoritative SFT-training products?

This is a scope-completeness artifact. It does not estimate on-policy GRPO
performance and does not justify widening the 1,438-product Run 2 training pool.

## 2. Why this must be a separate artifact

The existing candidate replay contains the 1,438 corrected active-pool products.
Gate G10 is defined over all 3,240 authoritative SFT-training products, including
always-fail, mixed and always-pass starting-policy groups. Relabeling the active
artifact or appending 1,802 products to it would destroy the frozen active-pool
denominator. The full-training replay therefore receives its own role, version,
manifest and records path.

| scope | products | completions | purpose |
|---|---:|---:|---|
| corrected active pool | 1,438 | 11,504 | primary within-group reward comparison |
| full authoritative SFT training | 3,240 | 25,920 | Gate G10 zero-variance evidence |
| SFT validation, excluded | 360 | 2,880 | development data; never replayed here |

The full scope contains all 1,438 active products plus 1,802 authoritative
training products outside the active pool.

## 3. Locked source evidence

| input | required identity |
|---|---|
| SFT split | `data/splits/sft-v1.json`; version `sft-v1`; SHA-256 `4d14d46fa4f7df95a24658c741940db64093e7798b5ccd1558f4faa29bbe9a3b` |
| weak-training source | `data/train_weak.jsonl`; 3,600 products; SHA-256 `1cbcbfba5ad379e7c66895d720a997edf913030ee1e76e4917101dfccb09530b` |
| difficulty manifest | `runs/sft-difficulty-k8/manifest.json`; version `sft-difficulty-v2`; mode `full`; SHA-256 `5c6fcc41bab65b36904cef256c56e747a19302b30b6da3c7208382e4dfdd3e5b` |
| locked rollout records | `runs/sft-difficulty-k8/rollouts.jsonl.gz`; 28,800 records; SHA-256 `f17360b157287caaea8d0f8e907f0a4bf4fd107977452442e2e447628e95bf8b` |
| CB class weights | `runs/grpo-run2-cb-class-weights.json`; SHA-256 `7b53323a7f1c170fa68c6b1a0d1356c67fd827f70f466ba2972b857418f4ab37` |

The builder must use the SFT manifest's `train` list as the canonical product
order. Its locked ordered-product hash is
`05e22c09120a63f9936473fd1adf8bf7639545cbe2a22bdbb28b8ab2d74906ee`.
Expanding every product to rollout indices `0` through `7` produces the locked
ordered-key hash
`a6acc3446db2102b95fdec7fc798f731969a473b053bc23fe0e5a7a1d9851d59`.

These hashes use UTF-8 values separated by one newline. A rollout key is
`<sku_id><TAB><rollout_index>`.

## 4. Output contract

The builder will publish two new files only:

- records: `runs/grpo-run2-full-training-candidate-replay-records.jsonl.gz`;
- manifest: `runs/grpo-run2-full-training-candidate-replay-manifest.json`.

The manifest version must be
`grpo-run2-full-training-candidate-replay-records-v1`, and its role must be
`full_authoritative_training_identical_group_candidate_replay_records`.

Each gzip JSONL line represents exactly one product and must preserve the same
record schema as the active replay:

- one canonical `group_position`;
- the product SKU, difficulty pass rate and complete gold record;
- gold-known and gold-unknown field counts summing to 15;
- exactly eight source completions in rollout-index order;
- raw output text and its SHA-256;
- the original `1:1:2` reward channels;
- U, UA and CB rewards plus their component ledgers.

The existing `build_replay_group` and reward implementations are the semantic
authority. The new builder may choose a different SKU scope, but it must not
copy or alter reward math.

For Candidate CB only, the full-training diagnostic uses a derived immutable
lookup. It copies every active-pool weight unchanged and adds the 13 locked valid
gold classes with zero active support at the existing maximum weight `2.0`.
This extension is version
`grpo-run2-cb-full-training-zero-active-support-extension-v1`; its ordered
entry-ledger SHA-256 is
`aeb089a1081d7efd1a99ccb2124e7b7412ec71f2362509f3df20dc2aa5837416`.
The output manifest must preserve its complete audit ledger and base-artifact
hash.

## 5. Required fail-closed invariants

Before publication, the builder must prove all of the following:

1. The locked split, source, difficulty, rollout and class-weight identities
   match Section 3.
2. The split contains 3,240 unique training SKUs and 360 unique validation
   SKUs; the sets are disjoint and cover all 3,600 source rows.
3. Training and validation normalized product families are disjoint.
4. The rollout source has 3,600 unique product groups and 28,800 unique keys;
   every group contains ordered indices `0` through `7`.
5. Selection uses exactly the 3,240 manifest training SKUs in manifest order,
   never the older embedded `split` field.
6. The output has exactly 3,240 groups, 25,920 completions and 25,920 unique
   rollout keys, matching both locked ordered hashes.
7. Every source completion is scored once by the original reward and all three
   dense candidates.
8. All 1,438 active-pool SKUs are present, exactly 1,802 additional training
   SKUs are present and zero validation SKUs are present.
9. Neither full-training output path aliases either active-replay output path.
10. Deterministic gzip uses `mtime=0`; records and manifest are written
    exclusively and atomically, with no overwrite and no partial publication.

## 6. Selection boundary

The manifest must record these facts explicitly:

- `candidate_rewards_calculated=true`;
- `aggregate_candidate_comparison_calculated=false`;
- `candidate_rankings_calculated=false`;
- `acceptance_thresholds_applied=false`;
- `winner_selected=false`;
- `model_generation_performed=false`;
- `validation_data_used=false`;
- `legacy_frozen_300_used=false`;
- `probe_100_used=false`;
- `cuda_imports_performed=false`.

The raw replay may calculate individual completion rewards because that is its
purpose. It may not calculate reward distributions, zero-variance shares,
bootstrap intervals, gate results, rankings or a winner.

## 7. Synthetic acceptance tests before real publication

The first implementation step must use only temporary invented evidence and
prove:

1. exact manifest-train selection and canonical order;
2. complete validation exclusion;
3. exactly eight ordered completions per selected product;
4. delegation to the shared group builder rather than duplicate reward math;
5. byte-identical deterministic output across repeated builds;
6. wrong hashes, duplicate keys, missing/reordered completions and family
   leakage fail before publication;
7. active and full output roles and paths cannot be interchanged;
8. a late bad group leaves neither a final nor partial output;
9. the manifest retains every boundary flag from Section 6.

Only after those tests and the full CPU suite pass may the real 3,240-product
artifact be built. Building it remains a separate, explicit step.

Completed synthetic implementation: manifest-train selection, exact source
coverage, train/validation SKU and family separation, complete k=8 validation,
ordered-hash and fixed-role enforcement, shared-scorer delegation, deterministic
gzip/manifest staging, active-path separation, collision protection and
two-file rollback now pass 17 focused tests. The real production scope has not
yet been preflighted through this builder at that synthetic milestone, and no
real output has been written.

Production scope preflight subsequently passed in 1.17 seconds. It verified all
3,600 source products and 28,800 source rollouts, selected the exact 3,240
authoritative training products in locked order, excluded all 360 validation
products, included all 1,438 active products plus 1,802 additional training
products, and reproduced both ordered hashes with zero SKU or family overlap.
The preflight calculated no candidate reward, staged no file and published no
artifact.

The first real scoring/publication attempt then stopped after 3.41 seconds at
`details/gathered`: Candidate CB's immutable class map was intentionally derived
from the 1,438-product active pool, while the broader 3,240-product diagnostic
scope contains gold classes absent from that map. Transactional cleanup left
both final outputs absent.

A gold-only support audit found 13 missing attribute/class pairs, 53 class
observations and 50 affected authoritative training products. None of the 1,438
active products is affected. No fallback policy may be added silently: its
behavior must be predeclared and tested before publication is retried.

That policy is now locked: a valid gold class with zero active-pool support gets
the existing clipped maximum `2.0` in the full-training diagnostic lookup only.
Five additional focused tests prove exact entry/count locking, no retuning of
existing weights, successful scoring of `details/gathered`, identical scoring
on a controlled active product, immutable lookup behavior, contract-drift
rejection and continued failure when any active class is missing. The full
suite passes 677 tests. At that policy-only milestone, the extension had not yet
been threaded into publication and both real output paths remained absent.

Publication integration is now complete. The publisher requires a
`FullTrainingCBExtension` rather than a bare lookup, verifies that the derived
lookup equals the base map plus exactly the audited ledger entries, and embeds
the complete audit plus its hash in the manifest. A ledger/lookup disagreement
fails before scoring. Twenty-five focused full-training tests and the complete
680-test CPU suite pass. Both real output paths remain absent pending an explicit
retry at that integration-only milestone.

The explicit retry succeeded in 17.19 seconds. It published 3,240 groups and
25,920 completions with the complete CB extension ledger and no aggregates.
Independent streaming verification reproduced both ordered hashes, found zero
validation SKUs and proved all 1,438 overlapping active groups are byte-equivalent
at record content after removing only their scope-specific `group_position`.

Published identities:

- records: 4,168,170 bytes, SHA-256
  `9b4a110910977b54e181c8f3d3452c555bdbb6c826e22bfec1c475045517fcf9`;
- manifest: 10,709 bytes, SHA-256
  `ad7f4b8b3749062b73b7b35f25c206cad8ef17ca55b248fbeff228e35a2bc9a0`.

The dual-scope analysis preflight now requires this pair in production mode,
pins both published byte counts and SHA-256 identities, recomputes the 13-entry
CB extension ledger hash and rejects active/full path or role aliasing. The real
preflight passed without decompressing either replay or calculating any
candidate aggregate. Focused preflight/full-builder tests pass 36 cases and the
complete CPU suite passes 684 tests. Gate G10, candidate rankings and a winner
remain uncalculated.

The file-I/O-free Gate G10 core is now implemented and proven on synthetic
groups. It reuses the locked 12-decimal canonical reward comparison, counts
each product once, and evaluates the 40% threshold as exact integer arithmetic:
`zero_variance_groups * 5 <= total_groups * 2`. Five focused tests prove that
exactly 40% passes, one additional zero-variance group fails, denominator and
identity drift fail closed, and the constants match the locked comparison and
original-reward artifacts. The complete CPU suite passes 689 tests. The real
full replay remains unopened by this calculator, so no real Gate G10 result is
available.

The next in-memory layer is also complete. The one-group Gate G10 adapter
validates a product's group position and SKU, exact completions 0–7, matching
source SKU/index keys, exact U/UA/CB ledger membership and identity, shared
eligibility, and finite saved final rewards. It then emits exactly one
`GateG10Group` per candidate without performing file I/O or an aggregate.
Five adapter tests include direct composition with the calculator and
fail-closed corruption cases. The related Gate/replay/preflight selection
passes 52 tests and the complete CPU suite passes 694 tests. No real replay
group was opened or adapted.

The in-memory multi-group collector is now proven. It requires an exact
denominator before adaptation, contiguous group positions from zero and unique
SKUs; adapts each group once; reconstructs and hashes the global ordered SKU
and rollout-key ledgers; and calls the Gate G10 calculator exactly once for U,
UA and CB on the same product order. Its output explicitly says that source
scope is not verified by the collector and therefore no real result is
authorized. Five focused collector tests, 57 related Gate/replay/preflight
tests and the complete 699-test CPU suite pass. The real replay remains
unopened.

Synthetic manifest-verified orchestration is now complete. It runs the proven
dual-scope preflight first, opens only the synthetic full gzip exactly once,
streams and adapts all groups, and requires observed group/completion counts
plus ordered SKU and rollout hashes to equal the preflight-verified manifest.
Five tests prove success, hash failure before decompression, position drift
before calculation, ordered-SKU drift after a valid physical re-hash and
invalid-JSON failure without publication. The related stack passes 62 tests
and the complete CPU suite passes 704 tests. No real gzip was opened.

The production Gate G10 result schema and publisher are now locked separately
from replay execution. The schema revalidates the pinned dual-scope preflight,
full lineage, 3,240 per-product rollout hashes, candidate arithmetic, exact
40% gate constants and a no-ranking/no-winner boundary. It may publish only to
`runs/grpo-run2-gate-g10-result.json`, exclusively and atomically. Nine focused
tests use fabricated inputs, 65 related tests pass, and the complete CPU suite
passes 713 tests. This milestone did not open the real replay or calculate a
real Gate G10 result.

The production launcher's preflight-only mode has now run against the real six
source files with gzip opening forcibly disabled. It revalidated the 3,240
groups, 25,920 completions and locked identities, confirmed the result path was
absent before and after hashing, and stopped without parsing or calculation.
The related stack passes 71 tests and the complete suite passes 719. Execution
mode was not yet implemented at that milestone, and Gate G10 remained unknown.

The explicit execution mode is now implemented and proven with synthetic
substitutes only. It streams the full scope once, requires the collector's
group/completion and ordered hashes to match this manifest, builds the locked
Gate G10 result and publishes exclusively. Lineage drift and a concurrent
output both fail closed. The related stack passes 75 tests and the complete
suite passes 723. The production execute flag has not run, so the real replay
remained unopened by this execution path and Gate G10 was still unknown at that
milestone.

The real Gate G10 execution subsequently opened this pinned gzip exactly once
and published the locked 8,126-byte result with SHA-256 `6a602e62…1e0d` in
2.32 seconds. U, UA and CB respectively have 860, 439 and 438 zero-variance
groups, so all pass the maximum of 1,296. Independent verification re-hashed
all six source files and recomputed the result arithmetic without decompressing
the replay again. This does not rank candidates or apply active-pool Gates
G1-G9.

## 8. What remains unestablished

This contract does not show that U, UA or CB reduces full-training zero
variance below the locked 40% Gate G10 threshold. It does not reveal a candidate
ranking, select a reward, or authorize GPU training. It only fixes what evidence
must exist before those decisions can be made.
