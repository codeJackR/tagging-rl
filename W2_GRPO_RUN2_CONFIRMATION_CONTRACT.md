# GRPO Run 2 untouched-confirmation acquisition contract

**Version:** `grpo-run2-confirmation-acquisition-contract-v1`
**Status:** locked before collection, labeling, or model inference
**Purpose:** acquire one final confirmation set that cannot be influenced by
Run-2 model behavior

## 1. Decision being protected

The confirmation set will be opened exactly once after one final Run-2 recipe
and checkpoint have been selected using development evidence. It may confirm or
reject that recipe; it may not choose the reward, beta, training arm, checkpoint,
decoding settings, stopping rule, or reporting threshold.

The existing local corpus cannot fill this role. All 4,000 labeled products are
already allocated to SFT, the probe, or the diagnosis-exposed frozen evaluation
set. The 65 family-disjoint probe rows are too small and remain a diagnostic
asset. This contract therefore requires a new product snapshot.

## 2. Locked size and precision rationale

- Final confirmation target: **400 products**.
- Family-clean candidate buffer before selection: **at least 800 products**.
- Each product is the statistical unit; attribute cells are not independent
  products.

Four hundred is a practical improvement over the old 300-row set. Under the
usual inverse-square-root approximation, row-sampling uncertainty scales by
`sqrt(300 / 400) = 0.866`, roughly a **13.4% narrower** interval, while k=5
labeling and complete human review remain tractable. This is a design estimate,
not a promised confidence-interval width.

The target may not be reduced after collection begins. If fewer than 800 clean
candidates are available, add eligible stores or take a later snapshot; do not
relax overlap rules or shrink the final set.

## 3. Acquisition boundary

Source listings may come from public Shopify `products.json` endpoints only
after the domain has passed the existing probe and its terms permit the planned
use. Fetch politely with the repository's descriptive user agent, one-second
minimum delay, round-robin store traversal, apparel filter, and a hard stop on
HTTP 403 or 429.

The snapshot must record:

- acquisition start/end time in UTC;
- requested domains and the exact accepted working-domain file;
- URL pattern, delay, page limit, and fetcher code commit/hash;
- per-store requests, rows seen/retained, errors, and final row counts;
- raw candidate file bytes and SHA-256.

No SFT, GRPO, or frontier-label output may be generated during acquisition or
candidate selection.

## 4. Exclusion universe

Before a listing is eligible, remove it if either its exact SKU or normalized
product family appears anywhere in the existing 4,000-product labeled universe:

- 3,600 SFT products, including Run-2 training and development rows;
- 100 probe products;
- 300 diagnosis-exposed evaluation products.

The family key is the existing `training.split_sft.group_key`: normalized brand
plus title before the spaced variant separator. Exclusion occurs before
selection and labeling. Missing brand/title information that prevents a stable
family key fails closed.

Also remove duplicate SKU IDs, duplicate family/variant listings beyond the
locked family cap, empty titles, and non-apparel rows. A final confirmation SKU
or family overlap count above zero invalidates the dataset.

## 5. Deterministic selection before labels

Exactly 400 rows are selected from the family-clean candidate buffer using only
listing metadata. The selector is fixed before labels:

- seed: `20260813`;
- stable tie-break: SHA-256 of `20260813\0<sku_id>` ascending;
- maximum four rows from one normalized family;
- maximum 60 rows (15%) from one store;
- at least eight stores represented;
- provisional garment-category strata are derived only from product type/title
  aliases in the locked pack;
- select for broad store and provisional-category coverage, then fill remaining
  positions by the stable hash order.

The selection manifest must contain all candidate rejection reasons, selected
SKU order, stratum counts, overlap checks, source identities, and selector code
identity. It must be published before frontier labeling begins.

True-label support is audited after labeling but cannot alter membership. Rare
or absent classes remain disclosed limitations; rows may not be swapped in to
improve a result or confidence interval.

## 6. Labeling protocol

Every selected product receives exactly five usable structured frontier labels:

- provider: OpenAI;
- model: `gpt-5.6-luna`;
- prompt: `prelabel-v1`;
- diversity: the five existing user-turn perturbations;
- controlled vocabulary/rules: the hash-locked Vastraa Taste v1 pack;
- consensus: existing `labeling.consensus.consensus_labels` implementation.

A failed or malformed request is retried for that same selected SKU. It is not
dropped and replaced with an easier product. All five usable samples and their
provider request lineage must be retained. Labels remain unavailable to SFT or
GRPO training.

## 7. Human review and label quality

All **6,000 product-attribute cells** (400 × 15) receive human review, not only
frontier-disagreement cells. Reviewers may inspect listing text/images,
controlled vocabulary, consensus labels, and individual frontier disagreement;
they may not inspect SFT/GRPO predictions or know which future model is being
confirmed.

Additionally, a deterministic 10% audit sample—40 products selected by
SHA-256 of `20260813-review\0<sku_id>`—receives a second independent review.
Disagreements are adjudicated and recorded. The frozen dataset must report:

- reviewed and corrected cells/rows;
- reviewer and adjudicator identifiers or anonymized stable IDs;
- agreement before adjudication;
- every correction with timestamp and rationale;
- verifier schema, vocabulary, and rule results;
- support for every attribute/status/value used by the evaluator.

Any unresolved review cell, invalid row, or unrecorded correction blocks freeze.

## 8. Freeze and secrecy boundary

Expected frozen location:
`data/confirmation_run2_v1/eval.jsonl`.

The freeze bundle must include a collision-protected manifest, source and label
provenance, exact row order, dataset bytes/hash, pack hashes, selection manifest
hash, review summary, and confirmation role. The data-role manifest must then be
updated from `confirmation required` to `confirmation assigned`.

Before final recipe selection:

- no SFT or GRPO predictions may be generated for these products;
- labels may be audited/frozen operationally but may not be used in model or
  threshold selection;
- no aggregate confirmation metric may be calculated;
- membership and labels may not be changed after freeze.

After one final recipe/checkpoint is locked, confirmation is opened once. The
primary comparison is paired product-level macro-F1 against the locked SFT
baseline, accompanied by selective macro-F1, coverage, validity, rule
violations, class support, and paired uncertainty. Unsupported classes are
reported; they are not silently removed after results are known.

## 9. Failure and stopping rules

Stop without freezing if:

- fewer than 800 family-clean candidates are available;
- the 400-row selector cannot satisfy store/family constraints;
- any selected row overlaps prior data by SKU or family;
- fewer than five usable frontier labels exist for any selected row;
- complete human review or the 10% independent audit is unfinished;
- any frozen row fails the verifier;
- source, selection, labeling, review, or pack lineage cannot be reconstructed.

Failures are retained as artifacts. No requirement may be weakened merely to
avoid another collection pass.

## 10. Current boundary

This document and its executable JSON counterpart lock the protocol only. No
network request, labeling call, human review, confirmation freeze, model
inference, or GPU training is authorized by this step.
