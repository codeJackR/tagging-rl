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
