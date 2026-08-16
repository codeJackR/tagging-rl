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
