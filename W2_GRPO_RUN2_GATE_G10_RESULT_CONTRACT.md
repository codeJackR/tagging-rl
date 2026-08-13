# GRPO Run 2 Gate G10 production result contract

**Version:** `grpo-run2-gate-g10-production-result-v1`
**Locked:** 2026-08-12
**Status:** production result published and independently verified; U, UA and
CB pass; no ranking or winner selected

## Purpose

This artifact answers one question only: for each of U, UA and CB, what share
of the 3,240 authoritative SFT-training product groups has one canonical reward
level across its eight saved completions, and does that share satisfy the
inclusive 40% Gate G10 threshold?

It does not calculate active-pool D3 metrics, apply Gates G1–G9, rank
candidates, select a winner or authorize GPU training.

## Locked output

The only permitted production path is:

`runs/grpo-run2-gate-g10-result.json`

Publication is exclusive and atomic. An existing result is never overwritten.
A serialization or final-link failure must leave no final or temporary result.

## Required source authorization

Before the result can be built, the production dual-scope preflight must prove:

- active manifest SHA-256 `e10c3c47…163e`;
- active records SHA-256 `30e3ea86…e38a`;
- comparison contract SHA-256 `8692291a…1142`;
- CB class map SHA-256 `7b53323a…ab37`;
- full manifest: 10,709 bytes, SHA-256 `ad7f4b8b…c9a0`;
- full records: 4,168,170 bytes, SHA-256 `9b4a1109…fcf9`;
- CB extension ledger SHA-256 `aeb089a1…7416`, recomputed with no active
  weights changed;
- 3,240 products, 25,920 completions and the locked ordered SKU/rollout hashes;
- exactly 3,240 valid per-product rollout-key hashes from the shared collector;
- no replay parsing, aggregation, gate, ranking, winner or artifact publication
  occurred during preflight.

## Candidate result invariants

Each candidate result must contain:

- Gate core version `grpo-run2-gate-g10-core-v1`;
- candidate identity U, UA or CB;
- exactly 3,240 groups and 25,920 completions;
- the locked ordered product hash;
- zero-variance plus varying groups equal to 3,240;
- zero-variance share exactly equal to count divided by 3,240;
- a one-to-eight unique-level histogram summing to 3,240, whose level-one count
  equals the zero-variance count;
- 12-decimal tie canonicalization;
- threshold `2/5`, maximum passing count 1,296 and a pass/margin recomputed from
  integer counts;
- the locked original baseline 1,571/3,240.

The artifact repeats a compact candidate summary, but the publisher recomputes
it from the full candidate records and rejects any disagreement. It also
validates the exact gate constants and rejects unknown top-level or nested
candidate fields, so a hidden ranking or selection decision cannot be smuggled
into this artifact.

## Selection boundary

The durable result must state:

- real full-training replay used: true;
- Gate G10 calculated and threshold applied: true;
- active candidate aggregates calculated: false;
- Gates G1–G9 applied: false;
- candidate rankings calculated: false;
- winner selected: false;
- GPU training authorized: false.

Gate G10 alone cannot establish candidate superiority. Its next authorized use
is alongside the separately locked active-pool D3/G1–G9 analysis.

## Synthetic contract evidence

Fabricated production-shaped preflight and candidate results test the schema
without opening either real gzip. Tests cover exact-boundary pass behavior,
failing candidates, source-hash and preflight-boundary drift, denominator and
arithmetic drift, nonlocked output paths, existing-output preservation,
tampered summaries and final-link cleanup.

## Production launcher preflight

`training/run2_gate_g10_production.py` first exposed only `--preflight-only`.
It fixes all six source paths, calls the existing dual-scope production
preflight, validates that report against this result contract, and checks the
locked result path both before and after source hashing.

The real preflight passed with gzip opening patched to fail. It authorized the
pinned 3,240-product/25,920-completion lineage and left every calculation,
selection and publication flag false. Six launcher tests pass, the related
stack passes 71 tests, and the complete CPU suite passes 719 tests. At that
preflight-only milestone, the locked result file remained absent.

## Explicit execution path

Launcher version `grpo-run2-gate-g10-production-launcher-v2` now requires
exactly one of `--preflight-only` or `--execute`.
Execution performs the production preflight, checks the output again, streams
only the full-training gzip once, collects U/UA/CB on the shared denominator,
matches streamed lineage to the preflight, builds this locked schema and calls
the exclusive atomic publisher. It contains no active-replay aggregation,
Gates G1-G9, ranking or winner logic.

Synthetic replacements exercised the complete execution sequence and published
only inside a temporary repository. Tests also prove lineage drift prevents a
build and a result appearing during streaming is preserved rather than
overwritten. Twenty-four launcher/result/orchestrator tests pass, 75 related
tests pass, and the complete suite passes 723 tests. The real execute flag has
not been invoked at that implementation milestone and the production result
remained absent.

## Production result

The explicit CPU execution completed in 2.32 seconds with a runtime guard that
permitted exactly one gzip open at the pinned full-training records path. It
published:

- path: `runs/grpo-run2-gate-g10-result.json`;
- bytes: 8,126;
- SHA-256: `6a602e629a58e6a7c006fb9a86ff7fcee5c1821ed7505f0b88fdfdc89b661e0d`.

| candidate | zero-variance groups | share | maximum | margin | G10 |
|---|---:|---:|---:|---:|---|
| U | 860 | 26.5432% | 1,296 | 436 | pass |
| UA | 439 | 13.5494% | 1,296 | 857 | pass |
| CB | 438 | 13.5185% | 1,296 | 858 | pass |

A separate standard-library verifier imported no calculator or result-contract
code. It re-hashed the result and all six physical sources and independently
recomputed lineage, candidate count/share/histogram arithmetic, the exact
`zero * 5 <= 3,240 * 2` decision, margins, summaries and interpretation flags.
It did not decompress or parse either replay. All checks passed.

These results establish only that every candidate clears the full-training
variation gate. UA and CB differ by one zero-variance product, which is not a
quality ranking and cannot justify CB's added complexity. Gates G1-G9, active
candidate aggregates, ranking, winner selection and GPU authorization remain
false. The post-result complete CPU suite passes 723 tests.
