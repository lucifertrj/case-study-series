# Eval-Driven Development — Haystack × Qdrant × TruLens

A faithful, runnable replication of Airbnb's
[**Eval-driven development: Lessons from evaluating GenAI at scale**](https://medium.com/airbnb-engineering/eval-driven-development-lessons-from-evaluating-genai-at-scale-e817e5ae5788),
built on the walkthrough scenario from the post:

> *"You're building an AI assistant that answers questions about a travel
> platform's support policies."*

The assistant is a RAG pipeline over a corpus of **real Airbnb support policies**,
collected verbatim from the Airbnb Help Center:

- **Cancellation policies & refunds** — [article 475](https://www.airbnb.com/help/article/475)
- **AirCover for guests** — [article 3218](https://www.airbnb.com/help/article/3218),
  [article 170](https://www.airbnb.com/help/article/170)
- **Host payouts** — [article 92](https://www.airbnb.com/help/article/92),
  [article 425](https://www.airbnb.com/help/article/425)
- **Major Disruptive Events Policy** — [article 1320](https://www.airbnb.com/help/article/1320)
- **Service fees** — [article 1857](https://www.airbnb.com/help/article/1857)
- **Reviews** — [article 367](https://www.airbnb.com/help/article/367)
- **Changing a reservation** — [article 913](https://www.airbnb.com/help/article/913),
  [article 1505](https://www.airbnb.com/help/article/1505)

Every question in the golden set is grounded in these documents (with exact
numbers — 72-hour AirCover windows, the 1.5% / $15 Fast Pay fee, the Firm
7-to-30-day 50% rule, the 14.1%–16.5% guest service fee, the 14-day review window,
and so on), and the clean negatives ask about things these policies genuinely
don't cover (loyalty points, pet fees, crypto payments, cleaning-fee caps).
Evaluation is treated as a first-class engineering discipline, implemented as the
blog's three layered defenses and its keep/reject experiment loop.

## Stack

| Concern | Tool |
| --- | --- |
| Orchestration (indexing + RAG) | **Haystack 3.x** (deepset) |
| Vector database | **Qdrant Cloud** |
| Virtual judges, tracing, dashboard | **TruLens** |
| Generator under test / judge model | OpenAI (`gpt-5.6-luna` / `gpt-5.6-luna`) |

## The three evaluation layers (from the blog)

```
Layer 1  Programmatic & heuristic checks   fast, free, catches obvious failures   edd/programmatic.py
             ↓
Layer 2  Virtual judges (LLM-as-judge)     nuanced quality dimensions             edd/judges.py
             ↓
Layer 3  Human evaluation & calibration    validates & calibrates the stack       edd/calibration.py
```

- **Layer 1 — programmatic**: deterministic checks (presence, length bounds, no
  leaked internal terms, and correct *refusal behavior* on clean negatives). No
  LLM call, so it filters the obvious failures before any judge spends a token.
- **Layer 2 — virtual judges**: a small, sharp panel — **one evaluator per
  dimension**, no "God evaluator" — judged by a *different, stronger* model than
  the generator. The RAG triad (**groundedness**, **answer_relevance**,
  **context_relevance**) plus a product-specific **conciseness** judge. Every
  score carries a chain-of-thought reason.
- **Layer 3 — calibration**: a judge you haven't calibrated gives false
  confidence. `edd calibrate` runs the answerability judge against the human
  labels in the golden set and reports **% agreement** and **Cohen's kappa**,
  listing disagreements to analyze. Target the high 80s–90s before trusting a
  judge at scale.

### Why the app isn't a single `Pipeline.run()`

TruLens scores from the OTEL spans the app emits. `SupportPolicyRAG`
([src/edd/app.py](src/edd/app.py)) therefore exposes **separate instrumented
`retrieve()` and `generate()` methods** orchestrated by `query()`, so the
`RETRIEVAL` / `GENERATION` / `RECORD_ROOT` spans are populated and every judge
selector resolves to real data. `TRULENS_OTEL_TRACING=1` is set in
[src/edd/__init__.py](src/edd/__init__.py) **before** any `trulens` import — if it
isn't, spans are empty and scores come back null.

## The gold set

[evals/gold_set.jsonl](evals/gold_set.jsonl) contains **both** positives
(`label: 1`, answerable from the corpus) **and clean negatives** (`label: 0`, not
covered — the assistant must refuse honestly). *"You can't test discernment
without bad examples."* Validation (`edd golden`) fails if either class is
missing. The schema follows the blog, not any boilerplate — `input`,
`expected_output`, `label`, `critique` (the human domain expert's annotation); no
`metadata` block. The current set is 16 positives / 9 clean negatives.

---

## Setup

### 1. Install

```sh
cd airbnb-eval-driven-development
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Credentials

```sh
cp .env.example .env
```

Fill in your three credentials (and nothing else — these are the only values the
code reads from the environment):

```
OPENAI_API_KEY=sk-...
QDRANT_URL=https://<your-cluster>.cloud.qdrant.io:6333
QDRANT_API_KEY=<your-qdrant-cloud-api-key>
```

> `QDRANT_URL` and `QDRANT_API_KEY` come from your **Qdrant Cloud** cluster
> (Clusters → your cluster → *Details* for the URL, *API Keys* for the key).

### 3. Index the corpus into Qdrant

```sh
edd index
```

Re-run whenever `chunk_size`, `chunk_overlap`, the embedding model, or the
documents under [data/support_policies/](data/support_policies/) change.

---

## The eval-driven loop

### Step 1 — Explore & discover ("look at your data")

Before building any metric, read the outputs and traces.

```sh
edd golden          # validate the gold set (positives + clean negatives)
edd explore         # generate answers for every gold-set question and read them
edd ask "how long do refunds take?"
```

### Step 2 — Evaluate across all three layers

```sh
edd evaluate        # Layer 1 + Layer 2 over the gold set, with clean-negative recall
```

### Step 3 — Calibrate the virtual judge (Layer 3)

```sh
edd calibrate                 # % agreement + Cohen's kappa vs. human labels
edd calibrate --target 0.9    # raise the bar
```

If agreement is below target, analyze the printed disagreements, sharpen the
rubric / few-shots in [src/edd/calibration.py](src/edd/calibration.py), and re-run
until the judge is trustworthy.

### Step 4 — Define goals & gates upfront, then iterate one variable at a time

[evals/experiment_plan.json](evals/experiment_plan.json) encodes the blog's two
iteration rules directly:

1. **Goals & gates upfront** — a top-level `goals` statement and a `gates` block
   of absolute thresholds ("what must be true before you ship"), decided *before*
   iterating:

   ```json
   "gates": { "groundedness": 0.8, "answer_relevance": 0.8, "clean_negative_recall": 0.9 }
   ```

2. **Fix one variable at a time** — each experiment carries a `varies` tag and
   follows the blog's staged protocol: *fix the model and vary the prompt → fix
   the prompt and vary the model → fix both and vary the serving configuration.*
   Virtual-judge results narrow the candidate pool at each stage.

`edd run <name>` evaluates one experiment, compares its required metrics to the
baseline (relative, to narrow the pool), checks the absolute **ship gates**, and
lets **Git enforce the verdict**:

```sh
edd run baseline         # stage 0 — defines the benchmark
edd run prompt-tighten   # stage 1 — fix model, vary prompt
edd run model-gpt4o      # stage 2 — fix prompt, vary model
edd run retrieval-k2     # stage 3 — fix both, vary serving config
```

- **Keep** — every required metric flat-or-better vs. baseline → the source change
  + result snapshot are committed.
- **Reject** — any required metric regressed → `src/` is reverted to the last
  clean state; only the result record is committed.
- **Shippable** — a kept candidate that *also* clears every absolute ship gate.
  Keep/reject narrows the pool; the gates decide whether a winner is actually
  releasable.
- **Plateau** — two consecutive flat keeps → the loop warns that prompt tweaks
  have run out and a structural change is needed (reranking, chunking, hybrid
  search).

> Prompt experiments (`varies: prompt`) are edits to `SYSTEM_PROMPT` in
> [src/edd/app.py](src/edd/app.py); the Git keep/reject captures them automatically.

> **Note on Git**: `edd run` makes real commits to enforce keep/reject. This
> folder lives inside the shared `case-study-series` repo, so those commits land
> there (scoped to the `src/` and `evals/` paths — it won't touch sibling
> projects). If you want an isolated history, copy this folder out into its own
> repo before running experiments.

### Step 5 — Inspect failures

```sh
edd report retrieval-k2                      # worst cases + judge chain-of-thought
edd report retrieval-k2 --metric groundedness --worst 3
```

### Step 6 — Mirror the evals in production

Evaluation is never one-and-done. `edd monitor` samples a fraction of a
de-identified traffic log ([evals/production_traffic.jsonl](evals/production_traffic.jsonl)),
runs the Layer-1 checks plus the answerability judge over each sampled query, and
surfaces flagged outputs (including likely fabrications) for human review.

```sh
edd monitor --sample-rate 0.3
```

### Dashboard

```sh
edd dashboard        # TruLens UI at http://localhost:8501
```

Explore records, span timelines, feedback scores, and the experiment leaderboard
(each experiment carries its keep/reject `decision` as metadata).

---

## Project layout

```
airbnb-eval-driven-development/
├── data/support_policies/        # the RAG corpus — real Airbnb Help Center policies
│   ├── cancellation_and_refunds.md
│   ├── aircover_for_guests.md
│   ├── host_payouts.md
│   ├── major_disruptive_events_policy.md
│   ├── service_fees.md
│   ├── reviews.md
│   └── changing_a_reservation.md
├── evals/
│   ├── gold_set.jsonl            # gold set: positives + clean negatives
│   ├── experiment_plan.json      # goals + ship gates + staged one-variable experiments
│   ├── production_traffic.jsonl  # de-identified traffic for production monitoring
│   └── results/                  # per-experiment result snapshots
└── src/edd/
    ├── __init__.py               # enables TRULENS_OTEL_TRACING before imports
    ├── config.py                 # Settings (one experiment = one knob changed)
    ├── store.py                  # shared Qdrant Cloud document store
    ├── indexing.py               # Haystack indexing pipeline
    ├── app.py                    # instrumented Haystack RAG app (the seam)
    ├── dataset.py                # gold-set loader + validator
    ├── programmatic.py           # Layer 1 — programmatic checks
    ├── judges.py                 # Layer 2 — virtual judges (RAG triad + conciseness)
    ├── calibration.py            # Layer 3 — judge calibration (Cohen's kappa)
    ├── evals.py                  # run an experiment across all layers
    ├── report.py                 # keep/reject verdict + worst cases
    ├── results.py                # persist/load result snapshots
    ├── loop.py                   # the EDD loop with Git keep/revert
    ├── monitor.py                # production monitoring on sampled traffic
    └── cli.py                    # the `edd` command
```

## Key takeaways, mapped to code

| Blog takeaway | Where it lives |
| --- | --- |
| Look at your data first | `edd explore`, `edd ask` |
| Build evals for *real* failure modes | `programmatic.py`, `judges.py` |
| One evaluator per dimension | `judges.py` (triad + conciseness, no God evaluator) |
| Calibrate to high 80s–90s before trusting a judge | `edd calibrate` (Cohen's kappa) |
| Use all three methods as layered defenses | `programmatic` → `judges` → `calibration` |
| Include bad examples in the gold set | `gold_set.jsonl` clean negatives + `dataset.py` guard |
| Evaluate the system, not just the model | instrumented `retrieve`/`generate` spans |
| Mirror evals in production | `edd monitor` |
| Define goals & gates upfront; fix one variable at a time | `experiment_plan.json` (goals/gates/staged) + `report.decide` gates |
