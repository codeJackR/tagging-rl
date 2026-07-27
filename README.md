# tagging-rl

One verifier, two consumers: the same schema-verification module serves as an **RL reward
function** (W2, W4–5) and as a **production QA gate** (W3).

Plan: `~/Downloads/rl-catalog-plan-stepwise.html`

---

## Status

| Week | Step | State |
|---|---|---|
| W1 | 1 · Adapt Fashionpedia's ontology | **done** — `packs/vastraa_taste_v1/vocab.yaml` |
| W1 | 2 · Schema pack format (Pydantic + vocab + rules) | **done** — `verifier/`, 71 tests green |
| W1 | 3 · Eval set (300, frozen) + train set (3–5k, weak) | next |
| W1 | 4 · Eval harness v0 | |

---

## W1 Step 1 — what exists

```
packs/vastraa_taste_v1/
  vocab.yaml         15 fields, 156 values, every value provenance-tagged
  PROVENANCE.md      why these 15, what was rejected, the merge log
data/raw/fashionpedia/
  ontology.json      frozen snapshot: 46 categories + 294 attributes (51 KB)
tools/
  check_vocab_provenance.py
```

**Done-when was "vocab.yaml exists, every value traceable to the ontology."** Traceability is
enforced, not asserted:

```bash
python tools/check_vocab_provenance.py
```

It resolves every `fp:` reference against the frozen ontology, fails on id *or name* mismatch,
requires a written justification on every `custom` value, and rejects ambiguous aliases. Current
result:

```
PASS — every value traceable, every custom value justified

  fields                 15
  values                156
    direct                98   62.8%
    derived               26   16.7%
    custom                32   20.5%

  Fashionpedia attributes consumed: 139 / 294
  Fashionpedia categories consumed:  16 / 46
```

The three `custom`-heavy fields are `material`, `colour_primary` and `occasion` — Fashionpedia
annotates shape, texture and construction, and contains no fabric, colour or use-context
attributes at all. See `PROVENANCE.md`.

### Before moving to Step 2

Two checks this step could not do, both cheap and both worth doing first:

1. Pull ~200 Sovrn rows and check each field's marginal distribution. Any field where one value
   exceeds ~85% will flatter the model — merge or cut it.
2. Hand-fill the schema for 20 real products. Cut any field that makes you hesitate repeatedly
   *now*, not during Step 3's 6-hour timebox.

---

## W1 Step 2 — the verifier

```
verifier/
  __init__.py    load_pack() · verify() · VerifierResult   <- the one import path
  schema.py      builds Pydantic models FROM vocab.yaml
  rules.py       three declarative rule types + derived applicability
packs/<name>/
  vocab.yaml     controlled vocabulary
  rules.yaml     cross-field rules
tests/
  test_pack_agnostic.py   identical suite, both packs
  test_vastraa_rules.py   one case per rule
```

```python
from verifier import load_pack, verify

pack = load_pack("packs/vastraa_taste_v1")
result = verify(model_output, pack)
# result.schema_valid · vocab_valid · rule_violations · parsed
#      + errors · abstentions · normalized
```

```bash
python -m pytest tests/ -q          # 71 passed
python tools/pack_info.py           # fields, rule inventory, per pack
python tools/check_vocab_provenance.py
```

**Packs are data, not code.** `build_model` lives in `verifier/schema.py`, not in each
pack — a pack that ships Python is not pack-agnostic, and `test_pack_dir_contains_no_python`
enforces that. `packs/demo_pack/` is three invented spacecraft fields and two rules, in two
YAML files and zero lines of Python. Every test in `test_pack_agnostic.py` runs against both
packs unchanged: that parametrised suite *is* the pack-agnosticism claim.

**Two models, not one.** `schema_valid` (is it JSON with the right keys and shapes) and
`vocab_valid` (are the values in the controlled vocabulary) are computed by separate models
so they move independently. W2 rewards format validity and vocab compliance separately; one
enum-typed model would collapse them into a single number and hide which one the model is
learning.

**`null` ≠ `"unknown"`.** `null` means the field does not apply to this item; `"unknown"` means
the model declined to answer. W4's reward is +1 correct / 0 abstain / −λ wrong, so abstention
has to be countable — `VerifierResult.abstentions` is that count, and W3's escalation queue
keys off the same list.

**No silent repair.** `verify()` does not strip markdown fences, fix trailing commas, or hunt
for the first `{`. In W2 the model trains unconstrained so you can watch it learn to emit clean
JSON; repairing output here would delete that reward signal before it was measured. Alias
leniency exists but is opt-in (`normalize=True`) and reports what it changed.

Rule inventory for `vastraa_taste_v1`: **34 total — 25 written + 9 derived.** The derived ones
come from `applies_to:` in vocab.yaml rather than being copied into rules.yaml, for the same
reason the Pydantic model is generated: two hand-maintained copies drift.

---

## Reproducing the ontology snapshot

`data/raw/fashionpedia/ontology.json` is committed (51 KB). It was extracted from the full
annotation file, which is not:

```bash
curl -L -o /tmp/fp_val.json \
  https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_val2020.json
```

then keep `categories` + `attributes`, attaching each attribute's `val2020` instance count as
`val_freq`. Images are not needed — the W1–W3 pipeline is text-in; vision is the optional
W6+ track.

Ontology: Fashionpedia (Jia et al., ECCV 2020) · [paper](https://arxiv.org/abs/2004.12276) ·
[api repo](https://github.com/KMnP/fashionpedia-api) (BSD-3-Clause).

---

## Layout

```
packs/        schema packs — nothing in the library may know which pack it is running
verifier/     the shared module: reward function (W2) + FastAPI service (W3)
evalharness/  per-attribute exact match, macro-F1, schema-validity, rule violations
training/     SFT baseline, GRPO runs
data/         raw/ · eval_300/ (frozen) · train_weak/
tools/        standalone scripts, no library imports
```
