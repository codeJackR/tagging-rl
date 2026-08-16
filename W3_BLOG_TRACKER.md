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
