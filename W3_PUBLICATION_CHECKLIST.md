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

> **DECIDED 2026-08-16: option 1, accept.** Re-issue was costed and rejected:
> `cb-class-weights.json` alone is hash-pinned in 12 places including four
> Python source files, and those pinning artifacts are themselves pinned by
> further contracts. The exposure is a macOS username and a directory layout.
> The repository is public with these paths intact.

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

> **DECIDED 2026-08-16: option 1, publish without the corpus, with one
> amendment.** Nine files are untracked and ignored; their labels ship as
> `data/eval_300/eval-labels.jsonl` via `tools/export_eval_labels.py`. All 19 of
> the post's claim tests pass against publishable data, and a clone reports
> 898 passed / 181 skipped / 0 failed.
>
> **The amendment, and it is a real one:** the corpus remains in git history
> from `e018a5a` onward, and the repository was published with that history
> intact rather than rewritten. A rewrite would have changed every commit hash
> from W1 forward, including the lock commits `8bff4c6` and `4c3e986` that the
> post cites as its proof of pre-registration. So the corpus is reachable by
> `git checkout e018a5a`. The untracking states the position and keeps the
> default clone clean; it does not restrict access, and this document should not
> be read as claiming otherwise.

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

> **OPEN, and the last item blocking publication.** Requires the W&B UI:
> project settings on `rushabhsp95-vastraa/tagging-rl`, Privacy, set Public.
> Visibility is per project, not per run, so this exposes every run in the
> project and the run list is worth reviewing first. The alternative is
> dropping the two links; the committed CSV histories keep the claims intact
> either way.

The post links two runs, `iwsrgsn2` (attention arm) and `s0ar902g` (combined
arm). Both are currently private. The post's loss-curve claims are backed by
exported CSVs committed to the repository, so the links are corroboration rather
than load-bearing.

Options: make both runs public, or drop the links and rely on the committed
histories. Either is defensible; leaving private links in a published post is
not, because a reader clicking them gets a login wall rather than evidence.

### 4.4 Repository visibility itself

> **DONE 2026-08-16.** https://github.com/codeJackR/tagging-rl is public.
> A full-history credential scan across all 152 commits was clean before the
> flip: no key patterns in any blob, and the only secret-shaped file ever
> committed is the `.env.example` placeholder.

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

## 7. One exposure this document originally missed

The W&B entity is `rushabhsp95-vastraa`, which embeds the local-part of the
author's email address. It appears in the post at lines 384 and 385, in
`W2_BLOG_BRIEF.md`, and in `runs/sft-attention-2epoch/README.md`.

It is worth separating from §4.1, which this document did cover. Git commit
metadata uses `rp7@mac.lan`, so the W&B string is the **only** place the real
address becomes derivable from this repository.

> **DECIDED 2026-08-16: accept.** The post is published under the author's own
> name, so attribution is intended; the realistic downside is scraper spam
> rather than any account exposure. Renaming the entity would break run URLs
> recorded inside committed artifacts, which is disproportionate to the risk.
