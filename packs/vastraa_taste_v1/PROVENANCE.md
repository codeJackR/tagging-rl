# vastraa_taste_v1 — selection & provenance log

W1 · Step 1. Why these 15 fields, out of Fashionpedia's 294 attributes and 46 categories.

Verify with:

```bash
python tools/check_vocab_provenance.py
```

---

## Source

| | |
|---|---|
| Ontology | Fashionpedia (Jia et al., ECCV 2020) — [paper](https://arxiv.org/abs/2004.12276) |
| Snapshot | `data/raw/fashionpedia/ontology.json` — categories + attributes only, 51 KB |
| Extracted from | `instances_attributes_val2020.json` (14.5 MB, not committed) |
| Tooling | [`KMnP/fashionpedia-api`](https://github.com/KMnP/fashionpedia-api), BSD-3-Clause |
| Images | **not downloaded** — the W1–W3 pipeline is text-in. Vision is the optional W6+ track. |

The `val_freq` recorded per attribute is its instance count in `val2020`. It reflects
**Fashionpedia's editorial-photography distribution, not a retail catalogue's**, and is used
below only as a coarse rarity signal. See "Known gaps".

---

## Selection criteria

Each candidate had to pass all six.

1. **Closed, finite value set.** Required for exact-match scoring in W1 Step 4.
2. **Determinable from product text.** Title, bullets, description, structured feed fields.
   Fashionpedia is an *image* ontology; a large fraction of its attributes describe
   distinctions no merchant ever types. Those were cut regardless of how useful they look.
3. **Two careful humans would agree.** 300 rows get hand-corrected under a 6-hour timebox in
   Step 3. A field that makes you hesitate is a field that injects noise into the one artifact
   that must be trustworthy.
4. **Enough signal to survive a 300-row eval set.** Macro-F1 averages per class, so a class
   with 3 positives contributes a high-variance term to the headline metric. Rare classes were
   merged upward, not kept for completeness.
5. **Participates in a cross-field constraint.** Rule-violation count is one of the four eval
   metrics and the rules engine is part of the reusable artifact. Fields were chosen partly so
   that W1 Step 2 has real constraints to express, not busywork.
6. **Matters to Vastraa's memory layer.** Taste is silhouette, fit, formality, colour, pattern —
   not warehouse metadata.

**Landed at 15 fields — the floor of the plan's 15–25 band, deliberately.** Every added field
costs 300 more hand-corrections in Step 3 and one more column the model can learn to game in W2.
Widening is cheap later; the merges below are all reversible because the source ids are recorded.

---

## The three structural adaptations

These are the interesting decisions — the places where "adapt" did real work over "adopt".

### 1. `length` was overloaded → split into two fields

Fashionpedia ids 146–160 all sit under one `length` supercategory, but they describe two
different things, disambiguated only by which garment part the annotation attaches to:

- 146–155 → hem length (above-the-hip … floor)
- 156–160 → sleeve length (sleeveless … wrist-length)

Product text carries no such structure, so the supercategory is split into `garment_length`
and `sleeve_length`. Both are marked `ordered: true` — they are monotonic scales, which lets W2
give ordinal partial credit (predicting `midi` for a `knee` item should not score the same as
predicting `cropped`).

### 2. `silhouette` was overloaded → shape vs fit

Ids 114–138 mix shape (`a-line`, `pencil`, `wide leg`), fit (135–138: tight/regular/loose/oversized)
and symmetry (114–115). Shape stays in `silhouette`; fit becomes its own field; symmetry is
dropped — no merchant writes "symmetrical", and `symmetrical` has val_freq 1895, i.e. it is the
unmarked default rather than a real label.

### 3. Fashionpedia has **no fabric attributes** → `material` is a custom field

This was the most surprising finding, and it is worth stating precisely because it is the kind
of thing a plan written from memory gets wrong. Fashionpedia's material coverage is:

- `non-textile material type` (10) — plastic, rubber, metal, feather, gem, bone, ivory, fur, wood
- `leather` (4) — suede, shearling, crocodile, snakeskin

There is **no cotton, linen, silk, wool, polyester**. Those values are therefore marked `custom`
with justifications, not laundered through a fake ontology reference. `suede` (290) and the fur
family (289 + 291) are the two genuine ontology values inside an otherwise custom field.

Same story for `colour_primary` and `occasion` — Fashionpedia annotates shape, texture and
construction, not colour or use-context. Both are fully custom fields.

---

## Merge log — the deliberate information loss

Reversible in every case: the source ids are in `vocab.yaml`.

| Field | Merge | Why |
|---|---|---|
| `neckline` | crew + round → `crew` | contour distinction absent from copy |
| `neckline` | scoop + u-neck + oval → `scoop` | three depth variants, one retail word |
| `neckline` | boat + straight-across → `boat` | same horizontal line |
| `neckline` | surplice + crossover → `wrap` | tailoring terms; merchants write "wrap" |
| `garment_length` | micro + mini → `mini` | merchants don't distinguish; removes a coin-flip label |
| `garment_length` | below-knee + midi → `midi` | one retail band |
| `garment_length` | maxi + floor → `maxi` | re-split if gowns matter to the taste layer |
| `pattern` | 6 animal + crocodile + snakeskin → `animal` | all val_freq ≤ 7 individually — none would survive a 300-row eval set |
| `pattern` | cartoon + letters/numbers → `graphic` | one retail word |
| `pattern` | geometric + chevron + argyle → `geometric` | chevron 7, argyle 2 |
| `collar_type` | collar nicknames + lapel nicknames + `hood` category | retail copy has one collar line; Fashionpedia models three separate garment parts |
| `sleeve_style` | tulip + circular flounce → `flutter` | sold as one thing |
| `material` | fur + shearling → `fur` | copy doesn't separate real from faux consistently |

`neckline` is the most-collapsed field in the pack — 25 ontology values into 15. That trade
(resolution for label reliability) is the clearest concrete example for blog #1.

---

## Rejected — considered and cut

Recording these matters: the one question worth being able to answer about this step is
*"why isn't X in here?"*

| Candidate | Fashionpedia ids | Cut because |
|---|---|---|
| **Garment nicknames** (peplum top, biker jacket, dirndl skirt, …) | 0–113 (114 values) | Criterion 4. The single largest group in the ontology and the biggest temptation. Values like `jodhpur`, `bloomers`, `flamenco skirt` have val_freq 0. Their useful content is already captured by `garment_category` × `silhouette` × `garment_length`, so including them would be redundant surface area. |
| **Pocket type** | 217–224 | Criterion 2. Merchants essentially never describe pocket construction in copy, despite decent Fashionpedia frequencies (patch 119, curved 127) — those come from *looking at the photo*. |
| **Lapel as its own field** | 174–178 | Criterion 4. Applies only to tailored outerwear ≈ 15% of a general feed → ~45 eval rows split across 4 values. Folded into `collar_type` instead, where it survives. |
| **Non-textile materials** (plastic, metal, gem, wood, bone, ivory) | 281–288, 294 | Criterion 2 + 6. Relevant to accessories, not to apparel taste. Only `fur` survives, inside `material`. |
| **Symmetry** | 114–115 | `symmetrical` val_freq 1895 = unmarked default; nobody writes it. |
| **Basque waistline** | 144 | Criterion 3 + 4. val_freq 16 and merchants never use the word. |
| **`season`** | — (would be custom) | Criterion 3. Cannot be labelled consistently from text without inventing a rule, and any such rule is derivable from `material` + `sleeve_length` anyway. |
| **`layering_role`** (base/mid/outer) | — (would be custom) | Deterministically derivable from `garment_category`. A redundant field is free reward-hacking surface: the model can score on it without learning anything. |

---

## Things to watch in W2

Written down now, while the reasons are fresh — these are predictions the W2 run will test.

- **`garment_category: other` is a hack indicator.** It exists because real feeds contain
  non-apparel rows, but it is also the cheapest escape hatch in the pack. Track its rate per run.
  A rising rate is the model dumping hard rows to dodge harder fields.
- **`pattern: solid`** (val_freq 1548) and **`sleeve_style: set_in`** (846) are the two dominant
  defaults. Expect majority-class collapse here first. This is exactly the "safe-tag
  conservatism" the plan predicts, and macro-F1 is what will expose it while accuracy hides it.
- **`sleeve_length: sleeveless` looks rare and isn't.** val_freq is only 12 because sleeveless
  garments carry *no sleeve annotation at all* in Fashionpedia — that is absence, not rarity.
  In a retail feed it will be common. Do not prune it on frequency.
- **`bodycon` overlaps `fit: tight`** by construction. That's a consistency rule for Step 2, and
  a place the model can satisfy two rewards with one guess.
- **`details` is the only multi-label field.** W1 Step 4 must score it with set-F1. Running exact
  match on it would silently make the whole report incomparable across fields.
- **`occasion` is the weakest field on annotator agreement.** If you find yourself disagreeing
  with your own labels during Step 3, cut it to `{casual, formal, athletic}` *before* freezing
  the eval set, not after.

---

## Known gaps — what this step did NOT do

Stated plainly rather than left implicit.

1. **No distribution check against a real feed.** Criterion 4 was applied using Fashionpedia's
   `val_freq`, which is editorial-photography distribution, not affiliate-retail distribution.
   Before Step 3, pull ~200 Sovrn rows and check the marginal distribution of each field. Any
   field where one value exceeds ~85% is a field that will flatter the model; merge or cut it.

2. **No 20-row hand-tag sanity pass.** The cheapest insurance in this step: fill in the schema
   by hand for 20 real products before labelling 300. Any field that makes you pause repeatedly
   should be cut *now* — cutting it during the Step 3 timebox costs 10× more. `occasion`,
   `silhouette` and `fit` are the three most likely to fail this.

3. **Alias lists are seeded, not measured.** They were written from retail-copy conventions.
   Once real feed text is in hand, mine the unmatched strings — every product string that fails
   to normalise to a canonical value is either a missing alias or a missing value, and that
   distinction is worth logging separately.
