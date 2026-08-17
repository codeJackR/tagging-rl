# Distributing blog post 1

Canonical URL: `https://www.vastraa.ai/engineering/sft-baseline.html`
Social card: `https://www.vastraa.ai/engineering/og-sft-baseline.png`

Everything below assumes the canonical stays the source of record. Post
elsewhere as a copy that points home, never as a second original.

---

## X thread

The post's own thesis is the hook: a proper experiment that returned a negative
result. That is rarer on X than another "we got 91%" post and it is the honest
version, so lead with it.

Nine posts. The link goes in the last one, because X suppresses reach on posts
that carry an external link and the thread should be readable without clicking.

> **1/**
> I fine-tuned a 1.5B model to tag apparel catalogs, then ran RL on top with a
> pre-registered causal experiment.
>
> RL lost to the supervised baseline.
>
> That is the interesting result, and I only know it because of what I built
> before running it.

> **2/**
> First, the boring part that made the rest legible.
>
> "Correct" is a Python module, not a vibe: schema, a closed vocabulary of 156
> values, 34 cross-field rules.
>
> The same module is the RL reward and the production gate. One implementation,
> two consumers.

> **3/**
> The supervised baseline, on 300 frozen products it never trained on:
>
> macro-F1 0.197 → 0.641
> schema-valid 62.7% → 100%
> vocabulary-valid 0% → 88.7%
> rule violations 1,204 → 12
>
> Total training compute: 4.4 cents.

> **4/**
> Then GRPO. It samples 8 answers per product and learns from how they differ.
>
> If all 8 score the same there is nothing to compare and the step teaches
> nothing.
>
> My reward could only ever emit 3 distinct scores across 2,400 completions.
> 64% landed on the maximum.

> **5/**
> Consequence: 125 of 300 optimisation steps produced no gradient at all.
>
> Two fifths of the run bought nothing.
>
> A denser reward that scores all 15 fields separately gave 141 distinct values
> and cut dead steps to 22.

> **6/**
> I wrote down two predictions before looking.
>
> The first was right, by more than forecast.
>
> The second was wrong: I predicted the dead-step count would climb flatter.
> It climbed faster.
>
> Saturation is the policy converging, not the reward failing to discriminate.
> A finer ruler postpones it; it does not prevent it.

> **7/**
> Now the part I did not expect.
>
> On the dev set I watched during training, RL looked like a small win.
> On the frozen set, both arms were significantly worse than the baseline.
>
> Same run. The two sets disagreed about whether RL helped at all.

> **8/**
> What RL actually bought was not accuracy.
>
> Rule violations: 0.67x the baseline, against the original reward's 2.75x.
> Answers less often, is more right when it does.
>
> Macro-F1 charges for abstention, so it read as a loss.

> **9/**
> Full writeup: how the split avoids leakage, why the checkpoint was locked
> before the frozen eval ran, and the arithmetic on every claim.
>
> Every number recomputes from committed artifacts, and a test suite enforces
> that.
>
> https://www.vastraa.ai/engineering/sft-baseline.html

### Notes

- Post 3 is the one most likely to be quoted; keep the numbers exactly as they
  are, they all trace to `runs/sft-combined-2epoch/frozen-eval-300-metrics.json`.
- Do not soften post 7. The dev-versus-frozen disagreement is the most useful
  thing in the whole project and the least common thing to admit.
- If posting a single link instead of a thread, use post 1 as the text. The
  card now carries the numbers.

---

## Medium

**Use the importer, never paste.** `https://medium.com/p/import`, give it the
canonical URL. Medium fetches the page and sets `rel="canonical"` back to it,
so your site stays the source of record and you avoid a duplicate-content
penalty. Pasting the text creates exactly that penalty.

### What will survive and what will not

| element | count | expected |
|---|---:|---|
| prose, headings, blockquotes | - | fine |
| code blocks | 16 | usually survive, sometimes lose language hints |
| tables | 6 | **will break.** Medium has no table support |
| inline SVG figures | 5 | **will drop.** Medium does not accept inline SVG |

So roughly a third of the post's evidence does not make the trip. Two ways to
handle it, and the second is better:

1. Repair by hand after import: screenshot each table and figure, upload as
   images. Slow, and the images stop being regenerable.
2. Publish an abridged Medium version: keep the narrative, keep the code, and
   replace each table with the one sentence it supports, then link to the
   canonical for the full evidence. Shorter, reads better on Medium, and stops
   pretending Medium can carry an evidence appendix.

### If you want the figures on Medium

They exist as SVG in `blog/assets/`. Converting them to PNG is the same
Pillow/fonttools path `make_og_image.py` already uses, so it is a small script
rather than manual work. Ask and it can be added.

---

## Order

1. Publish the canonical, which is live.
2. Post the X thread. It drives the first readers and X's card now renders.
3. Import to Medium a day or two later, once the canonical has been crawled, so
   there is no ambiguity about which came first.
