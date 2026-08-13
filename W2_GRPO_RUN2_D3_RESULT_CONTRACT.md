# GRPO Run 2 D3 production result contract

**Version:** `grpo-run2-d3-production-result-contract-v1`
**Locked:** 2026-08-12
**Status:** schema, launcher integration and production preflight proven;
real active replay not opened by these steps

## Purpose

This contract governs the durable active-pool D3 aggregate before Gates G1-G9
or candidate selection. It preserves the predeclared product-group metrics,
paired uncertainty, contribution dominance and product segments calculated over
the same 1,438 products and 11,504 completions.

It does not apply acceptance thresholds, rank U/UA/CB, select a reward or
authorize GPU training.

## Locked output

The only production path is:

`runs/grpo-run2-d3-candidate-analysis.json`

Publication is exclusive and atomic. An existing result is never overwritten;
a final-link failure leaves neither a final artifact nor a temporary file.

## Locked provenance and lineage

The artifact must identify exactly:

- active replay manifest: 6,435 bytes, SHA-256 `e10c3c47…163e`;
- active replay records: 1,921,202 bytes, SHA-256 `30e3ea86…e38a`;
- comparison contract: 12,079 bytes, SHA-256 `8692291a…1142`;
- CB class map: 27,446 bytes, SHA-256 `7b53323a…ab37`;
- 1,438 unique products and 11,504 completions;
- ordered SKU SHA-256 `97a96e77…b4c7`;
- ordered rollout-key SHA-256 `c1fe09c8…5500`;
- one stream and one adaptation per product;
- seed `20260812`, 10,000 paired product-group bootstrap replicates and 95%
  confidence.

The source-identity constants are immutable mappings. A red test found that a
mutable fixture could otherwise modify both the observed artifact and its
supposed expected value through the same dictionary reference.

## Analysis-core invariants

The contract requires:

- exact original/U/UA/CB reward membership;
- 1,438 unique ordered group IDs reproducing the active manifest hash;
- completion, group-mean, within-group variance and discrimination
  distributions with reconciled histogram counts;
- unique-reward-level and largest-tie histograms summing to 1,438;
- zero-variance, three-level and large-tie counts reproducing their shares;
- one pairwise-discrimination value for every product;
- all seven predeclared directional targets for every reward;
- pair counts that reconcile as concordant + discordant + reward ties;
- harmful-coverage counts that reconcile as harmful + safe + ties;
- U/UA/CB paired candidate-minus-original records for discrimination,
  canonical-known alignment and harmful coverage;
- whole-product pairing, no completion resampling, locked bootstrap settings,
  finite intervals and probability fractions summing to one.

Canonical JSON may reorder object keys, so candidate and metric dictionaries
are validated by exact membership rather than physical serialization order.

## Diagnostics and grain boundaries

The artifact must retain:

- field-contribution records for U, UA and CB with the 20% reference recorded
  but unapplied;
- CB class contributions with the 15% reference recorded but unapplied;
- proof that field/class child rows never become product denominators and that
  class allocations reconstruct CB known-field contributions;
- product-category, difficulty-band and gold-known-count dimensions;
- exactly one segment membership per active product in every dimension;
- segment hashes, completion counts and the fixed 30-product interpretation
  minimum;
- no segment bootstrap, acceptance gate or winner decision.

## Selection boundary

The production aggregate must state:

- candidate aggregate metrics calculated: true;
- real active candidate replay used: true;
- acceptance gates applied: false;
- candidate rankings calculated: false;
- winner selected: false.

D3 measurement and D4 decision remain separate artifacts and actions.

## Synthetic contract evidence

Eight focused tests use a fabricated production-shaped aggregate whose 1,438
group IDs come from the corrected training pool, not candidate replay outcomes.
They cover valid publication, immutable provenance, source/lineage/settings
drift, histogram and group-order drift, bootstrap drift, segment partition
drift, premature gate/selection flags, nonfinite data, unknown schema fields,
alternate paths, output collisions and atomic-link cleanup.

The original focused contract passes 8 tests. After launcher integration, the
focused orchestrator/contract/launcher set passes 29 tests, the related D3
stack passes 47 tests and the complete CPU suite passes 741 tests.

## Launcher and preflight integration

Aggregate construction now returns a complete in-memory artifact before any
publication call. The production launcher's future execution composition must
validate that complete artifact and only then call the exclusive publisher.
A synthetic trace proved the order `build -> validate -> publish`; a malformed
selection boundary never reached publication.

The launcher CLI currently exposes only `--preflight-only`. Its real preflight
was run with gzip opening patched to fail and still verified all four source
identities, the 1,438/11,504 lineage, both ordered hashes and locked bootstrap
settings. It created no output. The real active replay therefore remains
unopened and the locked D3 result remains absent.
