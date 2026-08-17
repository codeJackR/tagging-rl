import json, re, collections
from verifier import load_pack

pack = load_pack("packs/vastraa_taste_v1")
FIELDS = list(pack.field_names)
vocab = {n: {v.lower() for v in pack.specs[n].values} for n in FIELDS}
allvocab = set().union(*vocab.values())
junk = re.compile(r"^\W*$|\d{3,}|^temp|^badge:|^[a-z]{1,2}/\d|^u/\d", re.I)

rows = [json.loads(l) for l in open("data/train_weak.jsonl")]
def store(s):
    m = re.match(r"shopify:([^:]+):", s or ""); return m.group(1) if m else "?"
by = collections.defaultdict(list)
for r in rows: by[store(r["sku_id"])].append(r)

def val(lab):
    if not lab or lab.get("status") != "labeled": return []
    v = lab["value"]; return [str(x).lower() for x in (v if isinstance(v, list) else [v])]
def merch_blob(r):
    return " ".join([(r["input"].get("category") or ""), *(r["input"].get("raw_tags") or [])]).lower()

QUERIES = {
    "loose linen shirt":      {"fit": "loose", "material": "linen", "garment_category": "shirt_blouse"},
    "oversized sweater":      {"fit": "oversized", "garment_category": "sweater"},
    "something for the beach":{"occasion": "beach"},
    "lined footwear":         {"details": "lined", "garment_category": "shoe"},
    "striped shirt":          {"pattern": "striped", "garment_category": "shirt_blouse"},
    "work-appropriate":       {"occasion": "work"},
}

out = {}
for b, rs in by.items():
    n = len(rs)
    cats = collections.Counter()
    for r in rs:
        for v in val(r["labels"].get("garment_category")): cats[v] += 1
    apparel = sum(v for k, v in cats.items() if k not in ("shoe", "bag", "other"))

    tags = [t for r in rs for t in (r["input"].get("raw_tags") or [])]
    attr_tags = sum(1 for t in tags if any(v in t.lower() for v in allvocab) and not junk.search(t.strip().lower()))

    # coverage: labeled vs findable in merchant structured data
    per_attr = {}
    for a in FIELDS:
        lab_n = fnd = 0
        for r in rs:
            vs = val(r["labels"].get(a))
            if not vs or vs == ["none"] or vs == ["unknown"]: continue
            lab_n += 1
            if any(v in merch_blob(r) for v in vs): fnd += 1
        per_attr[a] = (lab_n, fnd)

    # abstention: how often the pipeline declined
    st = collections.Counter()
    for r in rs:
        for a in FIELDS:
            st[(r["labels"].get(a) or {}).get("status", "missing")] += 1

    q = {}
    for name, cons in QUERIES.items():
        match = [r for r in rs if all(cons[a] in val(r["labels"].get(a)) for a in cons)]
        fnd = sum(1 for r in match if all(cons[a] in merch_blob(r) for a in cons))
        q[name] = (len(match), fnd)

    out[b] = dict(n=n, apparel=apparel, footwear=cats["shoe"] + cats["bag"],
                  tags=len(tags), attr_tags=attr_tags, per_attr=per_attr,
                  status=dict(st), queries=q)

json.dump(out, open("/private/tmp/claude-501/-Users-rp7-Documents-Study/7bd89980-1be0-40eb-b227-d24a5076e93b/scratchpad/brand_analysis.json", "w"), indent=1)
print("computed", len(out), "brands")
