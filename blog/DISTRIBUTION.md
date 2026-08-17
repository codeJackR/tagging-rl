# Distributing blog post 1

Canonical URL: `https://www.vastraa.ai/engineering/sft-baseline.html`
Social card: `https://www.vastraa.ai/engineering/og-sft-baseline.png`
Exported figures and tables: `https://www.vastraa.ai/engineering/img/`

Everything below assumes the canonical stays the source of record. Post
elsewhere as a copy that points home, never as a second original.

---

## X thread

The thread has to promote the post that exists. Post 1 is the SFT baseline and
the measurement discipline around it; the GRPO result is post 2 and its material
is parked at the bottom of this file, unused until that post ships.

The hook is not "RL lost", because a reader who clicks will not find that. The
hook is the failure class the baseline exposes, which is concrete, visual, and
lands in one screenshot.

Nine posts. The link goes last, because X suppresses reach on posts carrying an
external link and the thread should be readable without clicking.

> **1/**
> A 1.5B model tagging apparel catalogs produced this, for a dress whose
> listing literally says "surplice neckline":
>
> `"neckline": "supraprise"`
>
> Structurally perfect JSON. Correct category, length, material, colour.
> And `supraprise` is not a word.

> **2/**
> That is the failure class worth naming: **semantically plausible,
> contract-invalid.**
>
> No schema check catches it. The JSON is valid. The field is a string. It
> just isn't one of the 156 values any downstream consumer will accept.

> **3/**
> So before touching a model, "correct" became a Python module rather than a
> vibe:
>
> a schema, a closed vocabulary of 156 values, and 34 cross-field rules.
>
> The same module is the RL reward and the production gate. One
> implementation, two consumers, no drift between them.

> **4/**
> Supervised fine-tune of Qwen2.5-1.5B, LoRA, on 300 frozen products it never
> trained on:
>
> macro-F1 0.197 → 0.641
> schema-valid 62.7% → 100%
> vocabulary-valid 0% → 88.7%
> rule violations 1,204 → 12
>
> Both arms trained: 4.4 cents.

> **5/**
> One honest reading note on that table.
>
> The zero-shot 0.197 is computed over 196 products, not 300. The base model
> produced nothing scorable for the other 104.
>
> The harness does not zero-fill failures, because that turns one format
> error into fifteen wrong answers.

> **6/**
> The sneakiest thing I hit was the learning-rate schedule.
>
> Cosine decay plans its descent over the total steps it expects. A 1-epoch
> run plans 203 steps and is at ~0 by step 203. A 2-epoch run plans 406, so at
> step 203 it is still near half its peak.
>
> Same step number. Different training state.

> **7/**
> So "load the epoch-1 checkpoint and train one more epoch" silently mixes two
> effects you can never separate afterward: seeing the data twice, and
> following a different LR curve from the start.
>
> The fix is to start over with one fresh 2-epoch run, saving at 203 on the
> way through.
>
> Nine minutes. Two cents.

> **8/**
> Then the part that mattered most later.
>
> The winning checkpoint was locked in a committed file, with its adapter
> SHA-256, before any frozen inference ran. The commit proves the choice
> predates the result.
>
> Frozen score came in at 0.641 against 0.854 on validation, and I did not
> explain that gap away.

> **9/**
> That refusal paid off. When the RL runs came, the same two sets disagreed
> about whether RL had helped at all, and I could tell which one to believe.
>
> Full writeup, with every number recomputing from committed artifacts and a
> test suite that enforces it:
>
> https://www.vastraa.ai/engineering/sft-baseline.html

### Notes

- Post 1 is the screenshot candidate. Consider attaching the `supraprise`
  JSON block as an image.
- Post 4's numbers all trace to
  `runs/sft-combined-2epoch/frozen-eval-300-metrics.json`. Keep them exact.
- Do not cut post 5. Publishing the conditional-F1 caveat next to the headline
  is the whole credibility argument, and it costs one post.
- Post 9 gestures at the RL disagreement without claiming this post analyses
  it, which it does not. Do not strengthen that wording until post 2 is live.
- If posting a single link instead of a thread, use post 1 plus the link. The
  card carries the numbers.

---

## X Article

Draft: `blog/x-article.html`. Open it in a browser, select all, copy, paste
into the X Articles composer. Requires Premium+ or a Verified Organization.

Pasting markdown yields literal hashes and backticks, so the draft is formatted
HTML instead; headings, bold, lists and quotes arrive already applied.

Regenerate with `uv run --with markdown-it-py python blog/make_x_article.py`.

| element | count | handling |
|---|---:|---|
| prose, headings, lists | - | carries over on paste |
| inline SVG figures | 5 | replaced with hosted PNG URLs |
| tables | 5 | replaced with hosted PNG URLs |
| code blocks | 8 | emitted as blockquotes, since X degrades them anyway |
| reproduction commands | 1 block | replaced with a link, nobody runs bash from a blockquote |

27,181 characters against a limit of about 100,000.

If the images do not survive the paste, upload them from `blog/assets/png/`.
Each one sits directly above its caption in the draft, so the positions are
unambiguous.

---

## Medium

**Use the importer, never paste.** `https://medium.com/p/import`, give it the
canonical URL. Medium fetches the page and sets `rel="canonical"` back to it,
so your site stays the source of record. Pasting the text creates a
duplicate-content penalty instead.

### What will survive and what will not

| element | count | expected |
|---|---:|---|
| prose, headings, blockquotes | - | fine |
| code blocks | 8 | usually survive, sometimes lose language hints |
| tables | 5 | **will break.** Medium has no table support |
| inline SVG figures | 5 | **will drop.** Medium does not accept inline SVG |

The canonical page carries its figures as *inline* SVG, so the importer drops
them regardless of the PNGs existing. The PNGs change the repair cost, not the
import behaviour.

### Repairing after import

This is now the cheaper option and the one to take. All ten images are already
rendered and hosted, so repair is ten uploads from `blog/assets/png/` at
positions the import leaves obvious:

- `fig1-frozen-eval.png` through `fig5-lora-arms.png`
- `table-1.png` through `table-5.png`, numbered in document order

Regenerate any of them with
`uv run python blog/assets/make_pngs.py`. Figures go through `npx sharp-cli`;
tables are drawn as SVG from the post's own markdown, so their contents cannot
drift from the article.

The alternative, an abridged Medium version that replaces each table with the
one sentence it supports, is still defensible and reads better on Medium. It
costs a second text to keep in sync, which is why it is no longer the
recommendation.

---

## Order

1. Publish the canonical, which is live.
2. Post the X thread. It drives the first readers and the card renders.
3. Import to Medium a day or two later, once the canonical has been crawled,
   so there is no ambiguity about which came first.
4. The X Article is optional and overlaps the thread. If both go out, space
   them, and let the thread lead.

---

## Held for post 2

The following was written as a thread for post 1 and cannot ship with it: every
number below is from the GRPO runs, and **none of them appear in post 1**. Post
1 defers the question explicitly ("whether a reward built on that verifier
actually fixes them is post 2's question") and spoils only the shape of the
answer.

Linking these posts to the SFT writeup would promise analysis the reader will
not find. Held here because the writing is good and the material is real, ready
when post 2 ships.

> **A/**
> I fine-tuned a 1.5B model to tag apparel catalogs, then ran RL on top with a
> pre-registered causal experiment.
>
> RL lost to the supervised baseline.
>
> That is the interesting result, and I only know it because of what I built
> before running it.

> **B/**
> GRPO samples 8 answers per product and learns from how they differ.
>
> If all 8 score the same there is nothing to compare and the step teaches
> nothing.
>
> My reward could only ever emit 3 distinct scores across 2,400 completions.
> 64% landed on the maximum.

> **C/**
> Consequence: 125 of 300 optimisation steps produced no gradient at all.
>
> Two fifths of the run bought nothing.
>
> A denser reward that scores all 15 fields separately gave 141 distinct
> values and cut dead steps to 22.

> **D/**
> I wrote down two predictions before looking.
>
> The first was right, by more than forecast.
>
> The second was wrong: I predicted the dead-step count would climb flatter.
> It climbed faster.
>
> Saturation is the policy converging, not the reward failing to discriminate.

> **E/**
> On the dev set I watched during training, RL looked like a small win.
> On the frozen set, both arms were significantly worse than the baseline.
>
> Same run. The two sets disagreed about whether RL helped at all.

> **F/**
> What RL actually bought was not accuracy.
>
> Rule violations: 0.67x the baseline, against the original reward's 2.75x.
> Answers less often, is more right when it does.
>
> Macro-F1 charges for abstention, so it read as a loss.

Note when reusing: E is the most useful and least common thing in the project,
and post 1 already sets it up. Do not soften it.
