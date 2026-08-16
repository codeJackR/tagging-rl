# The corpus is not redistributed

Nine files in this directory hold **4,000 apparel listings** collected from
public Shopify endpoints: title, brand, category, description, tags and image
URL. They are not in this repository, on purpose.

## Why

The Run 2 terms audit examined 20 candidate stores. **16 prohibit the automated
access used to collect this data. 4 are unresolved. 0 approve it.**

That audit governed *future* collection. Publishing what was already collected
is a separate and larger question, because **redistribution is a stronger act
than access**. The W2 brief called this corpus "an experimental research
artifact rather than a redistributable benchmark" before the audit made the
position concrete; the audit turned a cautious sentence into a decision.

The labels are a different matter. Consensus annotations, the provenance trail
and the human review record are this project's own work, and they are published
in full.

## What is missing

| file | rows |
|---|---:|
| `train_weak.jsonl` | 3,600 |
| `train_weak_sft_scored.jsonl` | 3,600 |
| `train_weak_grpo_cap4.jsonl` | 1,565 |
| `train_weak_grpo_cap4_sft_train_v1.jsonl` | 1,438 |
| `eval_300/eval.jsonl` | 300 |
| `eval_candidates.jsonl` | 300 |
| `grpo_run2_causal_schedule_v1.jsonl` | 300 |
| `probe_100.jsonl` | 100 |
| `train_weak_grpo_smoke_v1.jsonl` | 5 |

## What is published instead

`eval_300/eval-labels.jsonl` — the frozen evaluation set with every label,
provenance record and split assignment, and none of the merchant prose. It is
produced by `tools/export_eval_labels.py`, which withholds `title`,
`description`, `raw_tags` and `image_url` and keeps `brand` and `category`,
which are one-word retailer taxonomy terms that several cross-field rules key
off.

`sku_id` is kept and embeds the store domain. It is the join key for every
prediction file under `runs/`, so removing it would sever the labels from the
evidence they exist to verify. A domain is not the copy.

This is a **derivative, not a substitute**. The frozen manifest
`eval_300/eval.jsonl.frozen.json` checksums the original file, so
`eval-labels.jsonl` must never be fed to a provenance check.

## What this costs a reader

Every number in blog post 1 still verifies: **17 of its 19 claim tests pass with
the corpus entirely absent**, and the remaining two read gold labels, which
`eval-labels.jsonl` supplies.

What does not run is the provenance layer. Roughly **170 tests** require the
corpus, concentrated in the checkpoint monitor, GRPO pool builders, preflight
checks and replay contracts, because those verify data-file checksums and a file
that is not present cannot be checksummed. That is the honest cost of the
decision and it is not worked around.

## Confirming you have the same data

If you collect an equivalent corpus under your own terms, these SHA-256 values
identify the exact files this project used. Matching them means matching the
inputs; not matching them means your numbers and this repository's numbers are
not comparable.

Run `shasum -a 256 <file>` and compare against the hashes recorded in
`W2_GRPO_RUN2_BLOG_TRACKER.md` and the contracts under `runs/`, which pin these
files by content rather than by name.

The collection tooling is published: `tools/fetch_shopify.py`. **Read the target
stores' terms before pointing it anywhere.** The audit that produced the numbers
at the top of this file is in `runs/grpo-run2-confirmation-terms-audit.json`.
