# W3: production path, escalation, and the architecture claim made true

**Status:** active execution plan
**Created:** 2026-08-16
**Working style:** one small auditable step at a time; each step ends with a
tracker entry in [`W3_BLOG_TRACKER.md`](W3_BLOG_TRACKER.md) recording what was
built, what was found, and what it cost.

## 1. What W3 is for

W2 asked whether a verifier can serve as an RL reward. W3 asks the other half
of the original claim: **can the same verifier serve as a production quality
gate?** Until a second consumer exists, "one verifier, two consumers" is a
sentence in a plan rather than a property of the code.

The plan's own framing: *this is the week the architecture claim becomes true
rather than aspirational.*

Secondary goal, and the one with the longest lead time: blog #1 is drafted and
evidence-complete but unpublished, and publishing is what starts the job-search
thread. W3 prepares everything for it. **Publication itself is performed
manually and is out of scope for this plan.**

## 2. What already exists

W3 is assembly, not greenfield. Surveyed before planning:

| asset | state | W3 use |
|---|---|---|
| `verifier/` | 601 lines, pack-agnostic, schema + vocab + 34 rules | the library both consumers import |
| `labeling/providers.py` | OpenAI and Gemini strict structured outputs, schema adaptation | constrained decoding, already built |
| `labeling/consensus.py` | k-sample consensus, `review_cells`, `escalation_rate`, `queue_savings` | self-consistency confidence |
| `labeling/review.py` | CSV review round-trip, correction import | escalation queue |
| `evalharness/` | per-attribute metrics, macro-F1, paired bootstrap | production accuracy reporting |
| 4,000 labelled products | 3,600 train / 300 frozen eval / 100 probe | the corpus to run against |

The verifier is already imported by the reward functions, the eval harness and
the test suite. What does not exist is a **service** consumer, and no test
currently proves that two consumers agree.

## 3. Constraints that shape this week

### 3.1 Data acquisition is closed

The Run 2 terms audit examined 20 candidate Shopify stores: **16 prohibited, 4
unresolved, 0 approved**. The plan's step 4 says "run it on real mess" and names
new Shopify catalogs as the source. That is not available.

Consequence: the production demo runs against the **corpus already collected**,
and the sourcing constraint is reported as a W3 finding rather than worked
around. This is not a downgrade of the deliverable. A production path is
measured by throughput, cost, escalation rate and quality, all of which are
measurable on data already in hand.

Role discipline carries over from W2:

| dataset | W3 role |
|---|---|
| 3,600 weak training rows | production demo input, since they are already exposed to training |
| frozen 300 | reporting only; already exposed to model selection |
| probe 100 | **untouched**; remains a reserved diagnostic asset |

### 3.2 The plan's own warning

> Don't gold-plate; this is your comfort zone and it can eat the week.

Backend engineering is the most familiar work in this project and therefore the
most likely to expand. Every step below has an explicit stop condition. No
authentication, no database, no deployment, no queue infrastructure, no
container orchestration. A FastAPI app in one file and a JSONL review table are
the correct scope.

### 3.3 Budget

Frontier API calls cost roughly $1 per 1,000 requests at the observed labeling
rate. A 500-product demo with k=5 self-consistency is 2,500 requests, about
$2.50. Any step whose projected cost exceeds $10 stops and reports first.

## 4. Execution overview

| step | purpose | stop condition |
|---|---|---|
| W3.1 | verifier library + service, two callers, one test suite | a test proves both consumers return identical verdicts |
| W3.2 | production tagging path with constrained decoding | schema validity is 100% by construction and cost per SKU is logged |
| W3.3 | self-consistency confidence and escalation queue | escalation percentage is a number that can be quoted |
| W3.4 | run on the real corpus and report | one report: accuracy, escalation %, cost/SKU, throughput, p95 |
| W3.5 | blog #1 prepared to the point of publication | pre-publication checklist complete; publication left to a human |

## 5. W3.1 — Two consumers, one verifier

**Build.** A FastAPI application exposing the verifier over HTTP. It imports
`verifier.load_pack` and `verifier.verify` and adds no verification logic of its
own.

Endpoints, and nothing more:

- `POST /verify` — one record in, a verdict out: schema valid, vocabulary valid,
  rule violations, normalised record.
- `GET /health` — pack identity and version, so a caller can tell which
  vocabulary it is being judged against.

**The load-bearing test.** A single test feeds the same records through the
reward function path and the service path and asserts the verdicts are
identical. If they can differ, there are two verifiers regardless of what the
import graph says. This test is the architecture claim; everything else in the
step is plumbing.

**Repo legibility.** The plan asks that dual use be visible in the tree without
reading the README. `service/` beside `training/` and `evalharness/`, all three
importing `verifier/`, satisfies that.

**Stop condition:** one import path, two callers, one test suite. No auth, no
persistence, no deployment.

## 6. W3.2 — Production tagging with constrained decoding

**Build.** A production tagging path that is deliberately the opposite of the RL
path: decoding is **constrained**, so schema validity is free.

W2's runs were unconstrained on purpose, because format learning was the
behaviour under study. In production that reasoning inverts: nobody benefits
from a malformed record reaching a catalog. `providers.py` already implements
OpenAI strict structured outputs and the schema adaptation they require.

Additions this step:

- **idempotency keys** so a retried SKU cannot be tagged twice;
- **retry with backoff** on transient failures, with a bounded attempt count;
- **per-SKU cost tracking** from actual token usage, not estimates.

**Trap to avoid.** Cost must come from the API's reported usage. Earlier in this
project a cost figure was computed from a wrong scaling factor and had to be
corrected publicly; token counts are reported by the API and there is no reason
to estimate them.

**Stop condition:** schema validity is 100% by construction and per-SKU cost is
logged from real usage. No caching layer, no parallelism beyond simple batching.

## 7. W3.3 — Confidence without logprobs, and an escalation queue

**Why not logprobs.** The plan's revision 2 is explicit: generative models have
no well-calibrated confidence over structured outputs, and token logprobs do not
reliably track correctness. Confidence comes from **self-consistency** instead.

**Build.** Sample k=5 per product, measure per-attribute agreement across the
samples, and route cells whose agreement falls below a threshold to a review
table. `labeling/consensus.py` already computes agreement, `review_cells` and
`escalation_rate`; this step wires them into the production path rather than
reimplementing them.

**The number that matters.** Escalation rate is the unit economics of the whole
product: it is the fraction of cells a human must look at, and therefore the
labour cost per catalog. It must be reported per attribute as well as overall,
because a single bad attribute can dominate it.

**Threshold selection.** The threshold must be chosen on data whose labels are
known, and reported as a curve rather than a single point, so the
accuracy-versus-escalation trade-off is visible instead of hidden inside one
arbitrary choice.

**Stop condition:** an escalation percentage that can be quoted, with the
threshold that produced it and the curve around it. Reviewing is a CSV
round-trip, as in W1. No review UI.

## 8. W3.4 — Run it on the corpus, and report

**Run.** The production path end to end over a sample of the existing corpus,
large enough for the numbers to mean something and small enough to stay inside
budget. Target 500 products, subject to the cost gate in §3.3.

**Report**, one document with five numbers:

| metric | source |
|---|---|
| accuracy | eval harness against known labels |
| escalation % | self-consistency agreement below threshold |
| cost per SKU | API-reported token usage |
| throughput | products per minute, measured |
| p95 latency | per-product wall time distribution |

**Honesty requirements**, carried from W1 and W2. Accuracy against weak labels
is a comparison, not a production-accuracy claim. The frozen 300 has been
exposed to model selection and is reporting-only. Any number derived from a
sample states its sample size.

**Stop condition:** the report exists with all five numbers and their caveats.

## 9. W3.5 — Blog #1 prepared, not published

Publication is manual and out of scope. This step makes it a single decision
rather than a project.

Checklist to complete:

- final read of `blog/01-*.md` against the committed artifacts, confirming every
  quoted number still matches its source file;
- figures regenerate cleanly from `blog/assets/make_figures.py`;
- a secrets and data-rights sweep before the repository can be public, covering
  API keys, personal paths, merchant text redistribution and model licences;
- the W&B run visibility decision documented, since the post links two runs;
- a written list of what publication requires, so the human step is mechanical.

**Stop condition:** everything except pressing publish.

## 10. Risks

| risk | mitigation |
|---|---|
| backend scope creep | explicit stop condition on every step; no auth, database, deployment or queue |
| the service quietly reimplementing verification | the equivalence test is written before the service is finished |
| cost overrun | projected cost stated before any run; anything above $10 stops and reports |
| API rate limits or outages | bounded retries with backoff; partial runs report what completed |
| accuracy numbers over-claimed | weak-label caveat repeated at the point of every number, not once in a footer |
| the corpus constraint being hidden | the terms audit result is reported as a W3 finding |

## 11. Decision log

| date | decision | reason |
|---|---|---|
| 2026-08-16 | Run the production demo on the existing corpus | 16 of 20 candidate stores prohibit the access the plan assumed; 0 approved |
| 2026-08-16 | Keep probe 100 untouched | it is the only family-disjoint diagnostic asset left |
| 2026-08-16 | Constrained decoding in production, unconstrained in RL | opposite goals: production wants validity free, RL needs to measure whether it is learned |
| 2026-08-16 | Confidence from self-consistency, not logprobs | plan revision 2; generative models lack calibrated confidence over structured outputs |
| 2026-08-16 | Blog #1 prepared but not published | publication is a human decision and was excluded from this scope |

## 12. Live checklist

- [ ] W3.1 verifier service, two callers, equivalence test
- [ ] W3.2 constrained-decoding production path with cost tracking
- [ ] W3.3 self-consistency confidence and escalation queue
- [ ] W3.4 corpus run and five-number report
- [ ] W3.5 blog #1 pre-publication checklist
