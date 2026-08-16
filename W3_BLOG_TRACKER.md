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
