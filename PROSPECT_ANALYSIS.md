# Prospect analysis: 14 brands in the existing corpus

Internal working notes. Not a report, not outreach copy. Numbers computed from
`data/train_weak.jsonl` (3,600 products already collected) plus the pipeline's
existing labels. **No new API spend, no new collection.**

Reproduce: `scratchpad/brand_analysis.py`.

---

## How to read the columns

| column | meaning |
|---|---|
| **n** | products in our corpus for that brand. Not their full catalog. |
| **app%** | share the apparel pack can actually describe. Footwear and bags fall outside 15 attributes built for garments. |
| **junk tags** | share of their tags carrying no product attribute. Internal codes like `tempMD7.7.26`, `U/26`, `badge:`. |
| **gap** | of the values we confidently derived, the share absent from their category + tags. |
| **abstain** | share of cells where the pipeline declined to answer. **A proxy for how thin their product copy is.** |
| **addable** | confident values per product that are absent from their structured data. **The value we would actually deliver.** |

## Ranked by addable attributes per product

| # | brand | n | app% | junk tags | gap | abstain | **addable** |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | fahertybrand.com | 419 | 92% | 88% | 94% | 49% | **6.4** |
| 2 | www.untuckit.com | 236 | 99% | 81% | 80% | 56% | **4.7** |
| 3 | www.thursdayboots.com | 490 | 28% | 92% | 87% | 68% | **4.1** |
| 4 | www.everlane.com | 382 | 92% | 65% | 54% | 52% | **3.4** |
| 5 | www.taylorstitch.com | 332 | 97% | 63% | 58% | 59% | **3.3** |
| 6 | www.outdoorvoices.com | 102 | 98% | 93% | 80% | 70% | **3.3** |
| 7 | www.girlfriend.com | 25 | 100% | 69% | 64% | 66% | **2.9** |
| 8 | ministryofsupply.com | 51 | 94% | 65% | 92% | 81% | **2.5** |
| 9 | www.american-giant.com | 384 | 100% | 73% | 54% | 67% | **2.3** |
| 10 | naadam.co | 274 | 99% | 66% | 42% | 59% | **2.2** |
| 11 | www.tentree.com | 212 | 97% | 68% | 43% | 63% | **2.1** |
| 12 | www.rothys.com | 172 | 0% | 51% | 69% | 76% | **2.1** |
| 13 | www.allbirds.com | 215 | 3% | 100% | 68% | 79% | **2.0** |
| 14 | www.marinelayer.com | 306 | 99% | 97% | 68% | 82% | **1.7** |

## Natural-language query test

Products matching the query, and how many are findable from the brand's own
category + tags today. Sampled matches were checked against merchant copy;
see the caveat at the bottom.

| brand | loose linen shirt | oversized sweater | something for the beach | lined footwear | striped shirt | work-appropriate |
|---|---|---|---|---|---|---|
| fahertybrand.com | **60** / 0 | **3** / 0 | **28** / 3 | - | - | - |
| www.untuckit.com | **3** / 0 | - | **3** / 0 | - | - | **20** / 0 |
| www.thursdayboots.com | **3** / 0 | - | - | **208** / 0 | - | **2** / 0 |
| www.everlane.com | **8** / 0 | **9** / 0 | **17** / 0 | **2** / 0 | - | **4** / 0 |
| www.taylorstitch.com | - | - | - | - | - | **14** / 13 |
| www.outdoorvoices.com | - | **1** / 0 | **3** / 0 | - | - | - |
| www.girlfriend.com | - | - | - | - | - | - |
| ministryofsupply.com | - | - | - | - | - | **22** / 0 |
| www.american-giant.com | **1** / 0 | - | - | - | - | **1** / 0 |
| naadam.co | - | **17** / 14 | **1** / 1 | - | - | **4** / 4 |
| www.tentree.com | - | **6** / 0 | **3** / 0 | - | - | - |
| www.rothys.com | - | - | - | - | - | - |
| www.allbirds.com | - | - | **5** / 0 | - | - | - |
| www.marinelayer.com | **1** / 0 | **1** / 0 | **10** / 0 | - | - | - |

Read `60 / 0` as: 60 products match, 0 are findable from structured data.

## Per-attribute gap, all brands pooled

| attribute | confident values | findable today | gap |
|---|---:|---:|---:|
| occasion | 1,915 | 109 | 94% |
| garment_category | 3,600 | 1,908 | 47% |
| colour_primary | 2,407 | 837 | 65% |
| material | 2,629 | 1,181 | 55% |
| fit | 994 | 28 | 97% |
| details | 948 | 6 | 99% |
| pattern | 878 | 181 | 79% |
| closure | 834 | 150 | 82% |
| sleeve_length | 1,292 | 627 | 51% |
| collar_type | 782 | 205 | 74% |
| neckline | 678 | 287 | 58% |
| garment_length | 345 | 98 | 72% |
| silhouette | 214 | 8 | 96% |
| sleeve_style | 60 | 1 | 98% |
| waistline | 45 | 1 | 98% |

---

# Findings

## 1. Abstention is the variable nobody would have guessed mattered

The intuitive target is the brand with the worst structured data. That is
**Marine Layer**: 97% junk tags, 68% gap. It also has an **82% abstention rate**,
and ranks **last** on addable attributes at 1.7 per product.

The reason is that the two failures share a cause. A brand whose tags are junk
usually also writes thin product copy, and thin copy is what the pipeline reads.
Their catalog is bad in a way we cannot fix, because there is nothing in it to
extract.

**So worst-data is the wrong filter.** The right one is *large gap and low
abstention*: their structured data is poor but their prose is rich.

**Direct finding:** abstention ranges 49% to 82% across brands and inverts the
naive ranking. **Intuition:** we are not fixing catalogs, we are moving
information that already exists from prose into structure. If it was never
written down, we have nothing to move. **Limitation:** abstention here is
measured against our 15-attribute pack, so it partly reflects pack fit rather
than copy quality alone.

## 2. Faherty is the standout, and not narrowly

6.4 addable attributes per product, against 4.7 for the next brand. It has the
combination that matters: 92% apparel, 94% gap, and the **lowest abstention of
all 14 at 49%**. 419 products.

Its tags are the reason. `['Adult', 'getaway', 'July426', 'Responsible
Materials', 'tempMD7.7.26', 'U/26', 'women']` is a real example: campaign codes
and internal flags, 88% junk. Meanwhile the copy is unusually rich, which is why
abstention is low. Rich prose, empty structure, and the widest spread between
them in the set.

**Direct finding:** Faherty leads on addable attributes by 36% over second
place, with the lowest abstention. **Intuition:** they invested in
merchandising copy and never invested in the taxonomy behind it, which is the
exact shape our pipeline converts. **Limitation:** 419 products is our sample,
not their catalog; the real one is larger and the mix may differ.

## 3. Two brands the pack cannot serve at all

**Rothy's** is 0% apparel and **Allbirds** is 3%. Both are footwear and bags.
Fourteen of our fifteen attributes are garment concepts, so they score on
`details` and `colour_primary` and nothing else.

**Thursday Boots** is the interesting middle case: 490 products, but only 28%
apparel. Its strong finding, 208 of 245 boots lined and zero findable, comes
entirely from `details`, one of the few attributes that survives the crossover.

**Direct finding:** three of fourteen brands are majority non-apparel and the
pack does not describe them. **Intuition:** a footwear pack is a different
product, and the pack format was built to make that a config change rather than
a rewrite. Untested. **Limitation:** we have never built a second pack for a
real vertical, so pack portability is a design claim, not a demonstrated one.

## 4. The brands that tag well have the smallest gap, which is the honest bad news

**Naadam** (42% gap) and **Tentree** (43%) already carry 32.6 and 26.2 tags per
product. They are doing the work. Our value to them is roughly half what it is
to Faherty.

This cuts against the pitch. The brands most likely to understand why structured
attributes matter are the ones who already have them.

**Direct finding:** gap ranges 42% to 94%, and correlates inversely with tag
volume. **Intuition:** the buyer who gets it fastest needs it least, and the
buyer who needs it most has to be convinced the problem exists. That is the
sales problem in one line. **Limitation:** tag volume is not tag quality; Naadam
is 66% junk despite tagging heavily, so some of that 42% may be coincidence
rather than genuine coverage.

## 5. The query test survives checking; the contradiction test did not

Twelve sampled query matches were verified against the merchants' own product
copy. All twelve grounded: Faherty's linen shirts say linen, Thursday's lined
boots say lined.

An earlier attempt to find **contradictions** in merchant data produced false
positives on both examples checked. One matched "Short sleeves" as "shorts".
That angle is dead until the rules are far better, and it would have been
embarrassing in an email.

**Direct finding:** query matching verified 12/12; contradiction detection
failed 2/2. **Intuition:** claiming a brand is *missing* something is safe
because their own copy is the evidence. Claiming they are *wrong* puts our
judgement against theirs, and ours is not good enough yet. **Limitation:** 12 is
a spot check, not a measurement; the real precision of the query test is
unmeasured.

## 6. What this does not tell us

- **No accuracy number.** Every "confident value" here is the pipeline's own
  output. We can say a value is absent from their structured data; we cannot yet
  say it is right.
- **Corpus is not catalog.** 25 to 490 products per brand, collected months ago.
- **Merchant data is understated.** We read `category` and `raw_tags` only.
  Shopify metafields and variant options may carry attributes we are counting
  as missing, which would inflate every gap number here. **This is the single
  most likely way these numbers are wrong, and it is checkable on any brand's
  public product JSON.**
- **Provenance.** This was collected via access that 16 of 20 audited stores
  prohibit. Fine for internal analysis. Any external use needs an answer.

# Where I would look first

| tier | brands | why |
|---|---|---|
| **A** | Faherty, Untuckit | highest addable, low abstention, apparel-dominant |
| **B** | Everlane, Taylor Stitch | good confidence, moderate gap, sizeable catalogs |
| **C** | American Giant, Tentree, Naadam | they already tag; smaller delta |
| **D** | Thursday Boots, Outdoor Voices, Ministry of Supply, Girlfriend | mix or size problems |
| **out** | Rothy's, Allbirds | pack does not describe their products |

The check I would run before any of it: take one Faherty product, pull its
public product JSON, and see whether metafields carry attributes we are counting
as missing. If they do, the gap numbers shrink and tier A changes.

---

# Correction, after checking the collector

Section 6 flagged that merchant coverage was read from `category` and `raw_tags`
only, and that this was the most likely way these numbers were wrong. It was.
Two checks, both from local evidence, no fetching.

## Pruning does not inflate the gap

`tools/fetch_shopify.py:prune_ubiquitous_tags` drops tags carried by more than
90% of a store's products, which sounded like it could remove real attribute
tags. It cannot: the function **explicitly protects any tag appearing in the
pack's vocabulary**, matching across plural and singular. Ops noise is removed,
attribute tags survive. This concern is closed.

## Titles were never counted, and they carry a lot

`to_row` keeps title, description, tags and product_type, and drops `options`
and `variants`, which is where Shopify usually carries Size and Colour. I could
not check that directly, because **all four tier A and B brands are marked
`prohibited` in our own terms audit**, so fetching their product JSON to
validate our numbers is not available to us.

The title is a usable proxy, and it moves the number:

| | gap |
|---|---:|
| counting category + tags only | **68%** |
| also crediting the product title | **53%** |

Colour is most of the difference: 788 of 2,407 colour values appear in the
title and not in the tags. **Every colour-based claim in this document is
overstated** and should be treated as unreliable until options data is
available.

## What survives both checks

| attribute | values | in tags | in title | truly absent |
|---|---:|---:|---:|---:|
| occasion | 1,915 | 109 | 21 | **1,785 (93%)** |
| fit | 994 | 28 | 20 | **946 (95%)** |
| details | 948 | 6 | 103 | **839 (88%)** |
| garment_category | 3,600 | 1,908 | 220 | 1,472 (41%) |
| material | 2,629 | 1,181 | 321 | 1,127 (43%) |

## The distinction this forces

"In the title" is not the same as structured, and which number is honest depends
on the claim:

- **For on-site filtering and faceting**, a title is useless. The shopper cannot
  filter on prose. **68% stands.**
- **For agent search and feed attributes**, a title is parseable text and a
  model will read it. **53% is the honest floor.**

Leading with 71% or 68% against an agent-search pitch would be overclaiming, and
a technical buyer would catch it.

**Direct finding:** the gap is 53% once titles are credited, not 71%; colour is
the single most affected attribute and its numbers here are unreliable.
**Intuition:** the three attributes that survive every deflation are `occasion`,
`fit` and `details`, at 88 to 95% truly absent. Those are exactly the
"what it's like" attributes from the original observation, and the case is
stronger for being smaller and unable to be argued down. **Limitation:** the
title proxy is still not the real test. `options` and `variants` remain
unchecked, and our own audit prohibits the access that would check them on the
brands we most want to approach.
