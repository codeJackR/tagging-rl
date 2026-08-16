# Blog #1 pre-publication checklist

**Status:** everything except pressing publish.
**Post:** [`blog/01-an-sft-baseline-an-rl-result-can-stand-on.md`](blog/01-an-sft-baseline-an-rl-result-can-stand-on.md)

Publication is a human decision and deliberately out of scope for the automated
W3 work. This document exists so that decision is a single choice rather than a
project: everything mechanical is done, and everything requiring judgement is
stated below with its options.

## 1. Automated verification — done

`tests/test_blog_claims.py`, **14 tests**, extracts figures from the post by
regex and checks each against the artifact it came from. Numbers are pulled out
of the prose rather than hand-copied, so the test cannot drift from the text it
is checking: an edit that changes a figure without changing the evidence fails.

Verified: the frozen headline, the zero-shot comparison including its
conditional denominator, the bootstrap interval, both arms at checkpoints 203
and 406, trainable parameter counts, split sizes against the manifest, the lock
manifest's checkpoint and hashes, and the cost table as arithmetic on the stated
hourly rate.

Also verified as caveats rather than numbers, because they are the first thing
an editor trimming for length would cut: the 78-of-4,500 human-review figure,
the word "conditional" beside the zero-shot F1, and all five limitation bullets.

Every figure reference and every repo-relative evidence link resolves.

## 2. Figures — done

`blog/assets/make_figures.py` regenerates all five SVGs **byte-identically**,
confirmed by hashing before and after. The figures are therefore reproducible
from committed data rather than being artefacts of one session.

## 3. Secrets sweep — clean

| check | result |
|---|---|
| API-key patterns in tracked files | none |
| `.env` in git history | never committed |
| `.env` ignored | yes |
| GPU host address, port, or `root@` in tracked files | none |

## 4. Decisions that need a human

### 4.1 Local paths in four committed artifacts

Four provenance files contain absolute paths of the form
`/Users/<username>/Documents/Study/...`:

```text
runs/grpo-run2-cb-class-weights.json
runs/grpo-run2-data-boundary-audit.json
runs/grpo-run2-original-reward-training-replay.json
runs/grpo-run2-reward-scale-contract.json
```

This is not a credential leak. It exposes a local username and directory layout,
which is low sensitivity but not nothing.

It is **not** being fixed unilaterally, because these files are hash-pinned
evidence: their SHA-256 values are recorded in the Run 2 tracker and referenced
by later contracts. Rewriting the paths changes those hashes and invalidates the
provenance chain that the whole project's discipline rests on.

Three options, in increasing cost:

1. **Accept.** A username and a directory name. Most public ML repositories leak
   more than this in notebook outputs.
2. **Re-issue.** Rewrite the paths as repository-relative, regenerate the
   artifacts, and update every recorded hash. Correct, auditable, and touches
   the provenance chain.
3. **Exclude.** Keep those four files out of the public repository and note the
   omission. Cheapest, but a reader cannot then verify the claims that cite them.

### 4.2 Merchant text redistribution

`data/train_weak.jsonl` contains **3,600 product listings** with title, brand,
category, description, tags and image URL, collected from public Shopify
endpoints.

The Run 2 terms audit found **16 of 20 candidate stores prohibit** the automated
access used to collect this, and none approve it. That audit governed *future*
collection; publishing the already-collected corpus is a separate question and a
larger one, because redistribution is a stronger act than access.

The W2 brief already carries the caveat that this corpus "should be treated as
an experimental research artifact rather than a redistributable benchmark". That
sentence was written before the terms audit made the position concrete.

Options:

1. **Publish the repository without the corpus**, keeping labels, metrics,
   predictions and all code. Every number in the post remains verifiable from
   the artifacts; the raw merchant text does not travel. This is the option the
   evidence supports.
2. Publish the corpus and accept the redistribution question.
3. Publish hashes and schema of the corpus only, so a reader can confirm they
   have the same data if they collect it themselves under their own terms.

Option 1 is recommended. Note that the post's reproduction commands assume the
corpus is present, so choosing it means adding one line saying the corpus is not
redistributed and why.

### 4.3 W&B run visibility

The post links two runs, `iwsrgsn2` (attention arm) and `s0ar902g` (combined
arm). Both are currently private. The post's loss-curve claims are backed by
exported CSVs committed to the repository, so the links are corroboration rather
than load-bearing.

Options: make both runs public, or drop the links and rely on the committed
histories. Either is defensible; leaving private links in a published post is
not, because a reader clicking them gets a login wall rather than evidence.

### 4.4 Repository visibility itself

Making the repository public is the precondition for every relative link in the
post to resolve. It should be the last action before publishing, after 4.1 and
4.2 are settled, since both change what the public repository contains.

## 5. What publication actually requires

Once 4.1 through 4.4 are decided:

1. Apply the chosen data and path decisions.
2. Make the repository public.
3. Make the two W&B runs public, or remove their links.
4. Render `blog/01-*.md` on the chosen host with `blog/assets/*.svg` alongside.
5. Post it.

Nothing in steps 1 through 5 requires further engineering.

## 6. Explicitly not done

Publication. The post is finished, verified against its evidence, and waiting.
