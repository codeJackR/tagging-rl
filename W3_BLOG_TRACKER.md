# W3 technical tracker: the production path, and the second consumer

Running record of W3 work, findings and intuitions. Each entry states what was
built or attempted, what it revealed, and what it does not prove. Failures are
recorded rather than tidied away, on the same basis as W2: a plan that only
records successes teaches the reader nothing about what the work was actually
like.

Plan: [`W3_PLAN.md`](W3_PLAN.md). Preceding trackers:
[`W2_GRPO_BLOG_TRACKER.md`](W2_GRPO_BLOG_TRACKER.md),
[`W2_GRPO_RUN2_BLOG_TRACKER.md`](W2_GRPO_RUN2_BLOG_TRACKER.md).

Every entry ends with three lines, kept from the W2 trackers because they force
the useful parts out: **Direct finding** is what was measured, **Intuition** is
what it feels like, and **Limitation** is what it does not establish.

---

## 1. W3 starts from a claim that is still half-proven

The project's architecture claim has always been that **one verifier serves two
consumers**: an RL reward during training, and a quality gate in production. W2
proved the first half thoroughly, twice, and at some expense. The second half
has never been built.

That asymmetry is easy to miss because the verifier is already imported by three
things: the reward functions, the eval harness and the test suite. It looks
dual-use in the import graph. But an eval harness and a reward function are the
same kind of consumer, both offline, both scoring finished predictions in bulk.
Neither is a production gate deciding, one record at a time, whether something
may enter a catalog.

Nothing in the repository currently proves that two consumers would agree with
each other, because there has only ever been one kind of consumer.

**Direct finding:** the architecture claim is currently supported by an import
graph rather than by a test. **Intuition:** the wiring diagram shows two
sockets, and only one has ever had anything plugged into it. **Limitation:**
this is an observation about the codebase, not evidence that the two consumers
would disagree; the point of W3.1 is that nobody presently knows.

## 2. The week's data source disappeared before the week began

The plan's step 4 is "run it on real mess", sourced from public Shopify
catalogs. The Run 2 terms audit had already established that this is not
available: of 20 candidate stores, **16 prohibit the access the plan assumes, 4
are unresolved, and 0 are approved.** The executable source gate requires at
least eight written merchant approvals before a single product request is sent,
and there are none.

The response is to run the production path against the 4,000 products already
collected, and to report the constraint rather than route around it. Role
discipline from W2 carries over unchanged: the 3,600 weak training rows are the
demo input, the frozen 300 is reporting-only because model selection has already
seen it, and the 100-row probe stays untouched as the last family-disjoint
diagnostic asset.

This costs less than it appears to. A production path is judged on throughput,
cost per SKU, escalation rate and quality, and every one of those is measurable
on data already in hand. What is lost is the demonstration that the pipeline
survives contact with an unfamiliar catalog, and that loss should be stated
plainly rather than glossed.

**Direct finding:** the planned data source is unavailable, and the demo will
run on the existing corpus with the constraint reported as a result.
**Intuition:** the restaurant was booked for a tasting menu and the supplier
never delivered, so the tasting happens with what is already in the larder.
**Limitation:** running on already-collected data cannot show how the pipeline
behaves on a catalog whose wording and category conventions it has never seen,
which is exactly what a real production deployment would face first.

## 3. The most likely way to lose this week is to enjoy it

The plan carries a warning that no other week's plan carries:

> Don't gold-plate; this is your comfort zone and it can eat the week.

It is worth taking literally. W1 and W2 were unfamiliar work: ontologies,
reward design, GRPO internals, adversarial review. W3 is a FastAPI service, a
retry loop and a CSV round-trip. That is ordinary backend engineering, and the
temptation is to build it properly: authentication, a database, a queue, a
container, a deployment. All of that is *good engineering* and none of it is
*this week's evidence*.

Every step in the plan therefore carries an explicit stop condition, and the
stop conditions are deliberately unsatisfying. One file. A JSONL table. No auth.
The deliverable is a report with five numbers in it, not a system.

**Direct finding:** every W3 step has a written stop condition, chosen to be
smaller than what would feel finished. **Intuition:** the risk this week is not
that the work is too hard, it is that it is pleasant. **Limitation:** a stop
condition written in a plan is not a stop condition observed under pressure;
whether it held is something the later entries will have to record honestly.

---

## 4. W3.1 — the second consumer exists, and it agrees

`service/verifier_service.py` is the verifier's production consumer: a FastAPI
gate with two endpoints, `POST /verify` and `GET /health`. It is 154 lines,
imports `verify` and `verify_record`, and contains no verification logic of its
own.

The interesting part is not the service. It is the test.

### The claim, made executable

`test_the_two_consumers_never_disagree` takes **120 literal model outputs from
the committed SFT and GRPO prediction files**, pushes each through both paths,
and requires the verdicts to be identical on schema validity, vocabulary
validity, the `ok` property, the rule-violation list and the parsed record.

The reward path calls `verify` in-process. The gate path crosses HTTP, with its
own request parsing, pack loading and response projection. Those are genuinely
different code paths around the same core, which is exactly what makes the
comparison worth running.

Real outputs were used rather than invented fixtures deliberately. Invented
records agree trivially, because they are written by someone who already knows
what both sides do. The committed files contain the failures a model actually
produced: the record that opened with prose instead of JSON, and the one that
invented `supraprise` as a neckline. Those are the cases where a divergence
would hide.

A second test compares the gate against the **reward callables themselves**
rather than against `verify`, because the reward functions wrap the verifier and
a wrapper that reinterpreted the result would slip past a comparison with the
core alone.

A third test reads the service source and fails if it contains `rules.yaml`,
`vocab.yaml`, `ValidationError` or `json.loads(`. Equivalence today does not
prevent divergence tomorrow; that test is what makes "adds no logic of its own"
a property rather than an intention.

**All 12 tests pass.** The architecture claim has stopped being an import graph.

### Three things found along the way

**The design anticipated this in W1.** `verifier/__init__.py` opens with:

```text
W2 / W4-5   reward function      verify(rollout_output, pack).rule_violations
W3          production QA gate   verify(frontier_output, pack).schema_valid
```

The docstring named this week's consumer before the reward function existed. The
interface needed no change to accept it, which is the useful evidence: a design
that anticipated a second consumer and then received one without modification.

**Three wrong assumptions about the Pack API, caught by tests.** `pack.fields`
is `pack.specs`, `pack.field_names` exists but `pack.fields` does not, and
`rule_inventory` is a method rather than an attribute. All three were written
from memory of the interface rather than from reading it, and all three failed
immediately. This is the same class of defect that cost a full GPU run in W2 —
asserting against an interface that was assumed rather than checked — appearing
again within an hour of being written up. The difference is only that a CPU test
catches it in a second.

**A miscount that would have shipped a wrong number.** `rule_inventory()`
returns `{"written": 25, "derived": 9, "total": 34}`. Summing its values gives
68, because `total` is a key in the dict rather than something to be derived
from it. The health endpoint briefly reported 68 rules where there are 34. The
fix reports written and derived counts separately, which is more useful anyway:
an auditor reading a gate's identity wants to know how much of the rule set was
hand-authored.

**Direct finding:** two independent consumers of the verifier return identical
verdicts on 120 real model outputs, including malformed ones, and a source-level
test prevents the service from acquiring verification logic later. **Intuition:**
the second socket now has something plugged into it, and a meter confirms both
are carrying the same current. **Limitation:** equivalence is demonstrated over
120 records from two runs of one model family on one pack. It shows the two
consumers agree; it does not show the verifier is *correct*, which is a separate
question the weak labels cannot answer.

## 5. W3.2 — the production path inverts every decoding choice W2 made

`production/tagger.py` tags products with **constrained decoding**, which is the
exact opposite of what W2 did, for the exact opposite reason.

W2 decoded unconstrained on purpose. Whether the model learns to emit valid JSON
was the behaviour under study, and a grammar would have made that reward free
and therefore unmeasurable. Section 120 of the Run 2 tracker records the price
of that choice: the format-validity component scored 1.0000 on every one of
2,400 completions, contributing nothing to any gradient.

In production the reasoning inverts. Nobody benefits from a malformed record
reaching a catalog, so the schema is enforced during generation and validity
stops being a question. The same fact that made the reward useless makes the
production path free of an entire failure mode.

### What a grammar cannot do, and why the gate still runs

A schema can guarantee the shape of a record and the membership of each enum. It
cannot guarantee that a `solid` pattern is not also `multicolour`, because that
is a relationship between two fields that are individually legal. Cross-field
rules are checked after generation by the same verifier the reward function
uses, and a test asserts exactly this case: schema valid, vocabulary valid, gate
failed.

That is the clearest statement of why the verifier is not replaceable by a
schema, and it is worth having as an executable example rather than an argument.

Schema validity is **recorded rather than assumed**, even though constrained
decoding should make it constant. "Guaranteed by construction" is a claim about
a vendor's implementation, and the recorded number is the one that would falsify
it.

### Cost is measured, not estimated

`complete_with_usage` was added to the OpenAI provider so the API's reported
token counts reach the caller. The existing `complete` now delegates to it, so
no existing caller changed.

Dollars and tokens are kept deliberately separate. Token counts are a
measurement returned by the API; the price table is an *input* stated in one
place, so a reader can substitute their own rates without re-running anything.
Earlier in this project a cost figure computed from a wrong scaling factor was
stated and had to be corrected; that class of error is avoided by never deriving
a token count.

**Every attempt is charged**, including retries that failed. A retry consumed
tokens whether or not it produced an answer, and a cost-per-SKU figure that
counts only successful calls understates what the pipeline costs to operate. A
test pins this: three attempts must cost exactly three times one attempt.

### Idempotency keyed on the question, not the request

The key is a hash of product, pack name and prompt text. Same three, same key,
so a retry is identifiable. Change the pack or the prompt and the key changes,
because the previous answer is no longer an answer to this question: a record
judged against a different vocabulary is a different record.

### Stop condition honoured

No caching, no concurrency, no persistence, no queue. Output is a JSONL file.
The plan warned that this week's work is the comfort zone and could eat the
week; the tagger is 250 lines and the temptation to add a worker pool was real
and declined.

**Direct finding:** the production path enforces the schema during decoding,
applies the same verifier afterwards for what a schema cannot express, and
reports cost from API-reported tokens with retries charged. 13 tests pass with a
fake provider, so every retry and failure path is reachable without spending
anything. **Intuition:** the RL run took the safety rails off to watch the model
learn to stay on the road; production bolts them back on and measures only the
things rails cannot prevent. **Limitation:** a fake provider cannot prove that
constrained decoding actually yields valid JSON at the vendor, which is why
schema validity is recorded; that number only becomes evidence on the real run
in W3.4.

## 6. W3.3 — the escalation rate is the product's unit economics

`production/escalation.py` samples the production path k=5 times per product,
measures per-attribute agreement across those samples, and routes cells below a
threshold to a review queue.

Confidence deliberately does not come from logprobs. The plan's revision 2 is
explicit: generative models have no well-calibrated confidence over structured
outputs, and token probabilities do not reliably track whether a value is right.
So confidence is measured behaviourally instead. Ask the same question five
times and see whether the answers agree.

The consensus mathematics is not reimplemented. `labeling.consensus` already
computes modal labels, agreement fractions and review queues, and was used for
the W1 labelling round; this module samples the production path and hands the
results to it.

### Why this number matters more than accuracy

Accuracy alone is not a claim about a pipeline, because a pipeline can reach any
accuracy you like by escalating everything, and one that escalates nothing is
only as good as its worst cell. **The escalation rate is the labour cost per
catalog**, which is the number a buyer of this system actually pays.

### Three ways the rate can be quietly wrong

**Collapsing the two kinds of blank.** `null` means the field cannot apply;
`unknown` means it could and the listing did not say. If those are merged, an
abstention looks like an inapplicability and drops out of the queue, which is
exactly the cell a human most needs to see. The label mapping keeps them
distinct and a test pins it.

**Counting an unmeasurable product as a confident one.** A product whose samples
all failed cannot be scored for agreement. It is not confident, it is unmeasured.
Counting it either way corrupts the rate, so it is excluded and the exclusion is
tested.

**Hiding one bad attribute inside a healthy average.** One attribute escalating
at 100% inside fifteen gives a 6.7% headline, which reads as fine. The report is
therefore per attribute as well as overall, sorted worst first, and a test
constructs exactly that case.

### Two rates, because a reviewer opens products

Cell rate and **product touch rate** are both reported. A single disagreeing
cell in fifteen is a 6.7% cell rate and a 100% chance of having to open the
product. Those are very different pieces of information for someone estimating
review labour, and quoting only the first would understate the work.

### The threshold is a decision, so it is reported as a curve

Escalation rate is computed across every achievable threshold rather than at one
chosen point. With k samples only multiples of 1/k are achievable, so the curve
uses exactly those: intermediate values would add resolution the sample size
cannot support. A single number would hide the trade-off it was selected from.
A test asserts the curve is monotonic, since a higher bar cannot escalate less.

### Stop condition honoured

The queue is a CSV, the same round-trip W1 used, sorted worst-agreement first so
a reviewer with limited time spends it where the model is least sure. No review
interface was built. The W1 review round was completed on a phone spreadsheet,
which is evidence that a CSV is sufficient rather than an excuse for skipping
the UI.

**Direct finding:** self-consistency escalation is implemented over the existing
consensus code, reports cell and product rates per attribute and overall, and
exposes the threshold as a curve rather than a chosen point; 11 tests pin the
arithmetic and the three ways the rate can be corrupted. **Intuition:** asking
the same question five times and counting the disagreements is a cruder measure
of doubt than reading the model's own probabilities, and it is the one that
survives contact with structured output. **Limitation:** agreement is not
correctness. Five samples can agree confidently on the same wrong answer, and
nothing here detects that; the escalation rate bounds review labour, it does not
bound error.

## 7. Constrained decoding is not free, and a two-product smoke found the bill

The run report gates itself on a projected cost before spending anything. The
projection reused W2's measured prompt budget of 600 tokens, which seemed safe:
that figure came from tokenising the real corpus with the real tokenizer, and
section 3 of the W2 brief makes a point of how much better that is than
guessing.

A two-product smoke, costing under a cent, measured the truth:

| | prompt tokens | completion tokens | $/request |
|---|---:|---:|---:|
| projected from W2's budget | 600 | 170 | 0.00245 |
| **measured against the API** | **1,214** | **220** | **0.00372** |

The estimate was low by a factor of two, and the reason is structural rather
than arithmetic. **W2 measured the prompt the model sees. The production request
also carries the JSON schema**, because that is how constrained decoding is
specified, and this pack's schema enumerates all 156 vocabulary values.

So the grammar that makes schema validity free is itself billed, as prompt
tokens, on every request — including every one of the k self-consistency
repeats. Constrained decoding trades a class of failure for a line on the
invoice, and the trade is invisible until someone looks at a usage field.

At the original plan of 500 products the corrected figure is **$9.31 against a
$10 ceiling**, which passes but leaves no headroom for retries, and retries are
charged. The run was sized to 300 products at $5.79 instead. That is a change
made because a measurement moved, not because a number was unwelcome, and 300
products still yields 4,500 cells and 300 per attribute, which is ample for
stable per-attribute escalation rates.

The smoke also caught something duller and more likely: the API key lives in
`.env`, which the shell does not export, so the provider raised on construction.
Half a cent to discover, versus discovering it part-way through a paid run.

**Direct finding:** the production prompt is 1,214 tokens against a 600-token
projection, because the JSON schema for constrained decoding is sent on every
request and this pack's schema carries 156 vocabulary values. **Intuition:** the
grammar that guarantees the answer's shape has to be shipped with the question,
every time, and it is longer than the question. **Limitation:** measured on two
products of one pack against one vendor's structured-output implementation; a
vendor that caches or precompiles schemas server-side would bill this
differently, and a smaller vocabulary would shrink it.

### The rule this reinforces

Two token estimates in this project have now been wrong in the same direction.
W1 estimated prompt lengths from character counts and was out by up to 2x, which
is why W2 measured with the real tokenizer. W3 reused that measurement in a new
context and was out by 2x again, because the context added something the
original measurement never contained.

A measurement is only valid for the thing that was measured. Carrying one across
a boundary — from training to production, from the model's prompt to the API's
request — needs re-measuring rather than reuse, and a two-product smoke is the
cheapest possible way to find out.

## 8. W3.5 — the post is verified by a test, not by a proofread

Blog #1 was drafted, evidence-complete and unpublished. The plan's step was to
prepare it to the point where publication is a single decision rather than a
project. Two things were built.

### A test that reads the post

`tests/test_blog_claims.py` extracts figures **from the post by regex** and
checks each against the artifact it came from. Fourteen tests cover the frozen
headline, the zero-shot comparison, the bootstrap interval, both arms at
checkpoints 203 and 406, parameter counts, split sizes, the lock manifest's
hashes, and the cost table as arithmetic on the stated hourly rate.

Extracting rather than hand-copying is the point. A test containing its own copy
of the numbers drifts from the prose it is meant to protect, and then passes
while the post is wrong. Pulling them out of the file means an edit that changes
a figure without changing the evidence fails.

Three tests check **caveats rather than numbers**: that "conditional" still
appears beside the zero-shot F1, that the 78-of-4,500 human-review figure
survives, and that all five limitation bullets are present. Those are the first
things an editor trimming for length would cut, and they are what the post's
honesty rests on.

The post claims "every table in this post recomputes from committed artifacts".
The only respectable way to make that claim is to recompute them, so now
something does, on every commit.

The figures also regenerate **byte-identically** from `make_figures.py`,
confirmed by hashing before and after. They are reproducible from committed
data rather than artefacts of one session.

### A sweep that found one real thing

Secrets: clean. No key patterns in tracked files, `.env` never committed and
ignored, no GPU host address, port or user anywhere in the repository.

Four committed provenance artifacts contain absolute paths of the form
`/Users/<username>/Documents/Study/...`. Not a credential, but it exposes a
username and a directory layout.

It was **not fixed**, and the reason is the interesting part. Those files are
hash-pinned evidence: their SHA-256 values are recorded in the Run 2 tracker and
referenced by later contracts. Rewriting a path changes the hash and invalidates
the provenance chain the entire project's discipline rests on. A tidy-up that
silently breaks an audit trail is worse than a username in a JSON file.

So it goes to the human with three costed options: accept, re-issue the
artifacts and every recorded hash, or exclude the four files and lose the
verifiability of the claims that cite them.

### The larger question the sweep surfaced

`data/train_weak.jsonl` holds 3,600 product listings with descriptions, tags and
image URLs, collected from public Shopify endpoints. The terms audit found 16 of
20 stores prohibit that access and none approve it.

That audit governed *future collection*. Publishing the corpus is a different
and larger question, because **redistribution is a stronger act than access**.
The W2 brief already called the corpus "an experimental research artifact rather
than a redistributable benchmark", written before the audit made the position
concrete; the audit turns a cautious sentence into a decision.

The recommendation in the checklist is to publish the repository without the
corpus. Every number in the post stays verifiable from the committed labels,
metrics and predictions; only the raw merchant text stays behind.

**Direct finding:** the post is verified against its evidence by 14 automated
tests, its figures regenerate byte-identically, the secrets sweep is clean, and
two publication decisions requiring human judgement are documented with costed
options. **Intuition:** the post has been handed a set of keys and a note saying
which doors are still locked and why, rather than being declared finished.
**Limitation:** the tests verify that the post's numbers match the artifacts;
they cannot verify that the artifacts are correct, and no test can check whether
a sentence is a fair characterisation of what a number means.

## 9. W3.4 — five numbers, and the two defects that had to be fixed to trust them

The corpus run is done: 300 products, five self-consistency passes, 1,500
requests, $6.18, no failed requests. The five numbers the plan asked for, each
with the caveat that makes it readable.

| number | value | what it does not mean |
|---|---:|---|
| gate pass rate | **67.3%** | not accuracy; the gate is schema, vocabulary and cross-field rules, with no reference to a gold label |
| escalation rate | **26.2% of cells** | at threshold 1.0, the most aggressive setting; the same run touches **96.7% of products**, so "26%" is not "26% of the work" |
| cost per SKU | **$0.0206** | that is all five passes; a single tagging pass is **$0.0041**, and confidence costs 4x more than the answer |
| throughput | **4.3 products/min** | sequential, one process, no concurrency; a floor, not a capability. 21.5 requests/min is the same number per request |
| p95 latency | **3.90 s** | p50 is 2.83 s and max is 6.38 s, over 1,500 requests |

Underneath: schema validity 97.7%, vocabulary validity 97.7%, 92 rule
violations across 300 records.

### The escalation rate has two denominators and they tell opposite stories

26.2% of cells sounds like a quarter of the work. 96.7% of products sounds like
all of it. Both are the same measurement at threshold 1.0, and the threshold
curve is what makes the choice legible:

| threshold | cells escalated | products touched |
|---:|---:|---:|
| 0.6 | 3.0% | 32.0% |
| 0.8 | 15.5% | 86.0% |
| 1.0 | 26.2% | 96.7% |

With fifteen fields per product, requiring unanimity on all of them means
almost every product has *something* to look at. If the review interface is
per product, threshold 1.0 is barely better than reviewing everything, and 0.6
is the only setting that leaves most products untouched.

Worst attributes are `details` (62.0%) and `waistline` (57.3%), not the
`closure` (35.3%) that a reader of the raw artifact would have named. That is a
defect, and it is the first of the two below.

### Defect 1: the worst-first ranking did not survive being written down

`EscalationReport.summary()` sorts attributes worst-first, deliberately, with a
comment saying why. The report is then written with `sort_keys=True`, and JSON
objects are unordered by spec regardless, so the artifact came out
alphabetically. `closure` appears first, and a reader takes the first entry for
the worst one when it is fifth.

The fix publishes the ranking as an explicit list rather than relying on dict
order, with a test that round-trips it through the writer.

### Defect 2: throughput was reported as 1,416,888 products per minute

Throughput was `len(items) / wall_seconds`. Every pass on this run resumed from
disk, so wall clock measured five file reads, and the number came out six orders
of magnitude too high.

It now derives from summed request latencies, which were recorded when the
requests actually ran and stay true across a resume, and the report records
which passes were resumed so a reader can tell whether `wall_seconds` means
anything. The test asserts the number is under 600 per minute, which is the
bound one sequential process against a hosted API cannot exceed.

**This is the second time a resume path produced a wrong number rather than a
crash.** A crash would have been better.

### What the run found about constrained decoding

**Schema validity is 97.7%, not 100%.** All seven invalid records hit exactly
`max_tokens = 400` with empty content. They are truncations, not model errors.
Constrained decoding guarantees the output matches the grammar *only if
generation runs to completion*; a token cutoff yields a truncated document that
satisfies no schema, and nothing in the API surface says so.

The retry did not fire on any of them (`attempts: 1`), because `tag_one` retries
when the provider returns `None` and this returned an empty string. An empty
string is not an answer, but the code treats it as one.

Both are recorded rather than fixed. Fixing means re-running to keep the report
consistent with the code that produced it, and $6.18 of the $10 W3 ceiling is
already spent. Named for W4: **raise `max_tokens`, record `finish_reason`, and
retry a truncation as a budget failure rather than scoring it as a model
error.**

### One rule is carrying the whole failure rate

Of 92 rule violations, **79 are `auto:applies_to:waistline`** and 78 of those
are the same mistake: the model emits the string `"none"` for a top, shirt or
sweater where the pack requires JSON `null`.

The convention is not ambiguous. In the training data, when `garment_category`
is outside waistline's `applies_to` set, the label is `not_applicable` with
`null` in **2,783 of 2,784 rows (99.96%)**, and the string `"none"` is never
used that way. But `none` is a legal member of the waistline enum, because a
dress genuinely can have no waistline, and the schema offers it on every
product. Applicability is a cross-field constraint; a JSON Schema enum is
per-field. So the schema cannot express the rule and actively supplies the
tempting wrong answer.

The same pathology shows up in `details`, where the model spells absence four
different ways: `["unknown"]`, `["none"]`, `null` and `[]`.

For contrast, the SFT model decoding **unconstrained** over 300 frozen products
produced 12 rule violations total, and `auto:applies_to:waistline` does not
appear among them. Fine-tuning learned the convention from data; the grammar
cannot represent it. Not a like-for-like comparison, different models and
different products, but the direction is the interesting part: **the constrained
path is worse on exactly the constraint the grammar does not cover.**

### Agreement is not correctness, and now there is a number for it

W3.3 shipped a caveat saying five samples can agree on the same wrong answer.
This run measured it. `production/confidence_audit.py` crosses agreement against
the gate:

- **15 of 80 attributable violating cells (18.75%) are unanimous across all five
  samples.** For `waistline` alone it is 13 of 74. Those cells never enter the
  review queue at any threshold, including 1.0.
- The product-level version of the same question returns **zero**: no product
  fails the gate while every one of its fifteen cells is unanimous.

The second number is nearly vacuous and it would have been the flattering one to
quote. With fifteen fields, 290 of 300 products have at least one disagreeing
cell, so a product-level blind spot is almost impossible to observe whether or
not one exists. The cell is the honest unit because `write_queue` emits one row
per flagged **cell**: a reviewer opens flagged cells, not whole records, so a
unanimous wrong cell stays unseen even inside a product that was flagged for
something else entirely.

**Direct finding:** the pipeline tags at $0.0041 per SKU and $0.0206 with
confidence, 4.3 products/min sequential, p95 3.90 s, 67.3% gate pass, 26.2% of
cells escalated at threshold 1.0 but 96.7% of products touched; one applicability
rule accounts for 79 of 92 violations, and 18.75% of attributable violating cells
are unanimously agreed and therefore unreviewable. **Intuition:** constrained
decoding and self-consistency each guarantee something real and neither
guarantees correctness. The grammar guarantees membership, not applicability;
agreement guarantees stability, not truth. The verifier is the only one of the
three that measures the thing the catalog cares about, which is the entire
argument for it serving both consumers. **Limitation:** one model, one pack, 300
products already seen during SFT; agreement is measured over k=5, so 15 is a
count of what five samples missed rather than a bound on what the model gets
wrong; and the truncation and retry defects above are documented, not fixed, so
the 97.7% schema validity figure describes a pipeline with a known bug in it.

## 10. W3 closes, and one of its own stop conditions did not survive contact

All five steps are done: the verifier serves two consumers and they provably
agree, the production path inverts every decoding choice W2 made, confidence
comes from self-consistency with a published threshold curve, the corpus run
produced its five numbers, and blog #1 is verified by tests and stopped one step
short of publishing.

The part worth recording is that **W3.4 falsified W3.2's stop condition.**

The plan said W3.2 was done when "schema validity is 100% by construction". The
corpus run measured 97.7%. Constrained decoding guarantees the grammar only if
generation runs to completion, and seven records hit the `max_tokens` ceiling
and came back empty.

The implementation was never confused about this. `TagResult.schema_valid`
carries a comment saying it is "recorded rather than assumed, because
'guaranteed by construction' is a claim about a vendor's implementation and this
is the number that would falsify it", and section 5 of this tracker said the
number "only becomes evidence on the real run in W3.4". The code was built to
catch exactly this and it did.

The plan sentence was the sloppy one. It has been amended in place with the
reason and a pointer, rather than quietly corrected, because a stop condition
that was wrong is more useful visible than erased.

### The shape of the week's defects

Four defects reached a committed artifact this week, and three share a shape:

| defect | how it presented |
|---|---|
| throughput on a resumed run | wrong number, no error |
| worst-first ranking lost to `sort_keys` | wrong order, no error |
| `max_tokens` truncation scored as a model error | wrong attribution, no error |
| `details: []` crashed the consensus mapping | **crash** |

Only the last one announced itself. The other three produced confident,
plausible, wrong output, and two of them were found by reading the artifact
rather than by any test. The crash was the cheapest defect of the four: it
stopped the run, pointed at its own line, and was fixed in minutes.

**Direct finding:** W3 is complete against its amended plan; one stop condition
was falsified by a later step and amended in place rather than removed.
**Intuition:** the failures that cost the most are the ones that keep going. A
pipeline that reports 1,416,888 products per minute is more dangerous than one
that stops, because only one of them asks to be looked at. **Limitation:** three
of the four defects were caught by reading an artifact, which is not a repeatable
process; each now has a test, but the class of defect that produces a plausible
wrong number has no general guard in this repo.

## 11. The accuracy number cannot be produced, and finding out why produced a better one

W3.4 was closed without the "accuracy" the plan asks for. The stated reason was
that the demo products had been seen during SFT training. That reason was
incomplete, and the real one is structural.

### Every label in this project came from the model we would be grading

`data/eval_300` carries `provenance.labeler = "gpt-5.6-luna@prelabel-v1"`.
`production/run_report.py` builds an `OpenAIProvider`, whose
`default_model` is `"gpt-5.6-luna"`. **The production model is the labeller.**

Worse, the frozen set is barely independent of it. 111 of 300 rows were
reviewed, and review changed **23**. So 277 of 300 gold records are verbatim
what this model already said.

The tell was available before spending anything: scoring the stored frontier
snapshots against the gold they were derived from returns **macro-F1 0.9915**.
That is not a ceiling. It is a model agreeing with itself, and any "accuracy"
computed the same way inherits the same circularity.

`probe_100`, `eval_candidates` and all 4,000 raw rows carry the same labeller.

### There is a human anchor, and it is 78 cells wide

An earlier draft of this section said no independently labelled data exists.
That was wrong, and `data/reliability.json` is the correction.

**78 cells across 111 rows and 11 attributes were adjudicated by a human**, with
a second model family used for cross-checking. That is 1.7% of the frozen set's
4,500 cells, and the file marks itself `usable: false` for exactly the reason
you would expect: per-attribute counts run as low as `fit` n=2 and `closure`
n=3, with confidence intervals like 0.21 to 0.94.

What those 78 cells do support is a single blunt number:

> **The label source is about 72% accurate against human judgement.**
> `macro_accuracy 0.72`, `micro_accuracy 0.7051`, over 78 cells.

That reframes the whole comparison. The production model agrees with gold at
macro-F1 0.894 — but gold is itself roughly 72% right. Agreement with a
reference that wrong cannot be read as accuracy, in either direction: the model
is penalised for being right where the label is wrong, and rewarded for
reproducing the label's mistakes, which it is unusually likely to do because it
wrote them.

So the accurate statement is narrower than "no accuracy number is possible". It
is: **the only human-anchored measurement in this project is 78 cells wide,
declares itself unusable, and says the yardstick is 72% accurate.** A production
accuracy figure worth publishing needs an eval set labelled without the model in
the loop, and that does not exist here.

This does not retroactively damage the SFT number. Qwen is a different model
family, so 0.641 measures agreement-with-luna, which is what blog #1 already
says it is. The circularity only bites when the model under test *is* the
labeller.

### What the run measured instead, and why it is worth more

Holding the model and the 300 products fixed and changing only the path:

| frozen 300 | SFT Qwen 1.5B, unconstrained | **luna unconstrained** | **luna constrained** |
|---|---:|---:|---:|
| vocabulary validity | 0.887 | **1.000** | **1.000** |
| schema validity | 1.000 | not recorded | 0.970 |
| rule violations | 12 | **6** | **79** |
| ... of which `applies_to:waistline` | 0 | **0** | **60** |
| macro-F1 | 0.641 | 0.9915 † | 0.8944 † |

† circular for luna, since the gold is largely its own output. Not accuracy, and
not quoted as such. **The rule-violation row is not circular**: violations are
computed by the verifier against the pack's rules and never touch a gold label.

**Turning constrained decoding on multiplied rule violations by 13x, in the same
model, on the same products.** The waistline applicability failure goes from
zero occurrences to 60.

And the benefit it was turned on for did not materialise here at all.
**Vocabulary validity was already 1.000 unconstrained.** For a frontier model
that emits legal vocabulary anyway, the grammar on this pack is pure cost: it
guarantees a property already held, and buys an applicability failure by putting
`none` inside the enum for every product including the ones where the field
cannot apply.

The earlier version of this finding compared against the SFT Qwen model, which
confounded model with decoding path. This is the controlled version.

One honest limitation on the attribution: the two paths differ in **both**
decoding mode and prompt (`prelabel-v1` against the production renderer), so the
13x belongs to the production path as a whole. The mechanism narrows it: 60 of
the 79 violations are one enum offering a value the schema cannot condition on,
which is a property of constrained decoding rather than of prompt wording.

### The rest of the run

Cost per SKU **$0.004169** and p95 **4.21 s**, both within a few percent of the
demo run's $0.004117 and 3.90 s, on a different product set. Schema validity
0.970, with 9 unparseable records, the same `max_tokens` truncation defect
section 9 recorded and did not fix. The frozen set's checksum verified
(`freeze_ok: true`).

The escalation rate reads 0.0 and **must not be quoted**: this run used k=1, so
every cell is trivially unanimous. Escalation needs k>1 and was measured on the
demo run.

Worst attribute by macro-F1 is `waistline` at 0.725 with exact match **0.448**,
which is the same defect surfacing in a third independent view.

**Direct finding:** no publishable production-accuracy number exists here,
because every labelled set was pre-labelled by the model under test, 277 of 300
frozen gold records are that model's verbatim output, and the only human anchor
is 78 cells that mark themselves unusable and put the label source at 72%
accurate; the run done to discover this instead produced a controlled comparison
showing constrained decoding raising rule violations from 6 to 79 in one model
on one product set, with vocabulary validity already at 1.000 without it.
**Intuition:** a measuring stick built by the thing being measured cannot measure
it, and the frontier's 0.9915 "ceiling" was that fact sitting in plain sight.
**Limitation:** the two paths differ in prompt as well as decoding, so the 13x
belongs to the production path as a whole; one model, one pack, 300 products;
the 72% anchor rests on 78 cells with intervals wide enough to be nearly
uninformative per attribute; and the accuracy gap is documented rather than
closed, since closing it needs an eval set labelled without the model in the
loop.

## 12. What section 11 means for W6, including one thing it does not mean

W6 plans a four-row table (distill arm A, distill arm B, the W2 GRPO
checkpoint, the teacher) scored on the frozen 300. Section 11 changes what that
table can say, and it is worth separating the part that carries from the part
that does not.

### It does not carry the circularity

Section 11's problem is that the production model **is** the labeller. W6 is an
all-Qwen affair: a 1.5B student, a Qwen teacher, and the plan makes matching
tokenizer families mandatory anyway. Qwen is not `gpt-5.6-luna`, so scoring
Qwen arms against luna-derived labels is the same arrangement W2's SFT number
already used, with the same caveat and no new defect. Nobody should read
section 11 and conclude the frozen 300 is unusable. It is unusable **for
grading luna**, which is a narrower claim.

### It does carry the yardstick

What follows W6 everywhere is the 72%. Every cell in that four-row table is
agreement with a reference that is itself roughly 72% right on the only 78
cells anyone checked. So the table can **rank** the arms against each other, and
cannot state how good any of them is. That is the same rule blog #1 set for
itself, applied one week later: a delta under a fixed evaluator, not an absolute.

### The teacher decision is currently undecidable

The plan's rule is: teacher = "your W2 GRPO checkpoint **if it beat SFT**, else
LoRA-SFT Qwen2.5-7B-Instruct". That test cannot be run today.

**No arm of run 2 has ever been scored on the frozen 300.** Only run 1
(`grpo-first-300-frozen-eval-300-*`) has. Every Arm A and Arm B number in
sections 121 to 124, including the whole paired comparison, comes from the
checkpoint monitor's 360 dev products. Section 124 flagged this as a limitation;
it is now also a blocker, because the plan's teacher branch depends on a
comparison nobody has made.

Applying the plan's own rule therefore needs one cheap step first: score Arm B's
checkpoint-300 on the frozen 300. The adapter is archived and hash-verified, the
harness exists, and the SFT frozen eval cost cents.

Two outcomes, both worth knowing before the week starts:

- **Arm B clearly beats SFT there** → the teacher is a 1.5B checkpoint already in
  hand, and W6 is the cheap week the plan describes.
- **It does not** → the teacher is a LoRA-SFT Qwen2.5-7B, which is training a
  second model the plan budgets loosely, on top of the distillation itself.

On the dev monitor the arms were tied on macro-F1 (0.8645 against a 0.8537
baseline, a gap of 0.011), so the second branch is at least as likely as the
first. That is a real cost sitting behind a measurement that has not been taken.

### The hardware does not match any configuration the plan sizes

The plan sizes VRAM for an 80GB A100 ("comfortably") with a 40GB fallback
(teacher 4-bit, or student down to 0.5B). The box is a **24GB RTX 3090**, below
both. Arm B's monitor peaked around 9.2GB with a 1.5B model, so there is real
headroom, but a 7B teacher held in-process alongside a student and its optimizer
is a different problem. The plan already requires a 50-step smoke run before
committing, and on this hardware that smoke run stops being a formality.

**Direct finding:** W6 is untouched by section 11's circularity because it is
all-Qwen, but inherits the 72% yardstick, so its four-row table ranks rather
than grades; its teacher rule is undecidable until Arm B is scored on the frozen
300, which has never been done for either run-2 arm; and the 24GB box sits below
the smallest configuration the plan sizes. **Intuition:** the causal experiment
answered the question it was designed for on the set it was monitored on, and
quietly never met the set it was supposed to be judged on. A monitoring signal
became the result because it arrived first. **Limitation:** the tie on the dev
monitor is 360 products and one seed, and a frozen-300 score could fall either
way; the VRAM concern is arithmetic rather than a failed smoke run, and the
plan's own smoke step is the thing that would settle it.

## 13. The teacher question from section 12 is answered, and it is the expensive branch

Section 12 recorded that W6's teacher rule was undecidable, because no run-2 arm
had ever been scored on the frozen 300. Both arms have now been scored, and the
result is in
[`W2_GRPO_RUN2_BLOG_TRACKER.md`](W2_GRPO_RUN2_BLOG_TRACKER.md) section 125.

| frozen 300, macro-F1 | value | vs SFT | 95% CI |
|---|---:|---:|---|
| SFT baseline | 0.6411 | - | - |
| Arm A (original reward) | 0.6247 | −0.0164 | [−0.0287, −0.0040] |
| Arm B (candidate UA) | 0.6315 | −0.0097 | [−0.0170, −0.0020] |

**Neither arm beat SFT, and both intervals exclude zero.** The plan's rule
therefore resolves to its `else`: the teacher is a LoRA-SFT Qwen2.5-7B-Instruct
trained on the 3,600 weak rows, not a checkpoint already in hand.

So the cost estimate in section 12 lands on the wrong side of its coin flip. W6
now needs a 7B trained before the distillation it was budgeted for, and held
in-process next to a 1.5B student and its optimizer on a 24GB card that is below
both configurations the plan sizes. The plan's 50-step smoke run stops being a
formality and becomes the first thing the week should do.

There is a consolation the section-12 estimate did not anticipate. The plan's
teacher gate is *"teacher beats the student baseline on the frozen eval by a
margin worth transferring"*, and the frozen numbers make that gate cheap to
apply: the student baseline is 0.6411, and that number now sits alongside two
GRPO arms measured identically. If the 7B does not clear it, the plan's own
advice is to fix the teacher before touching `GKDTrainer`, and the fixing can be
decided in one scoring run rather than after a week of distillation.

**Direct finding:** the frozen-300 scores resolve W6's teacher to the 7B branch,
because both GRPO arms are significantly below the SFT baseline on macro-F1.
**Intuition:** the cheap version of W6 depended on a number nobody had measured,
and measuring it removed the cheap version. **Limitation:** this decides the
teacher under the plan's stated rule only; a different rule, for instance
picking the teacher on rule compliance where Arm B beats SFT 8 to 12, would
choose differently, and nothing here argues the plan's rule is the right one.
