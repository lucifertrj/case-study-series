# Eval-Driven Development — Haystack × Qdrant × TruLens

A faithful, runnable replication of Airbnb's
[**Eval-driven development: Lessons from evaluating GenAI at scale**](https://medium.com/airbnb-engineering/eval-driven-development-lessons-from-evaluating-genai-at-scale-e817e5ae5788),
built on the walkthrough scenario from the post:

> *"You're building an AI assistant that answers questions about a travel
> platform's support policies."*

The assistant is a RAG pipeline over a small corpus of **support-policy
documents** (refunds & cancellations, guest safety & claims, host payouts).
Evaluation is treated as a first-class engineering discipline, implemented as the
blog's three layered defenses and its keep/reject experiment loop.

## Stack

| Concern | Tool |
| --- | --- |
| Orchestration (indexing + RAG) | **Haystack 3.x** (deepset) |
| Vector database | **Qdrant Cloud** |
| Virtual judges, tracing, dashboard | **TruLens** |
| Generator under test / judge model | OpenAI (`gpt-4o-mini` / `gpt-4o`) |

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

## The golden dataset

[evals/golden.jsonl](evals/golden.jsonl) contains **both** positives (`label: 1`,
answerable from the corpus) **and clean negatives** (`label: 0`, not covered —
the assistant must refuse honestly). *"You can't test discernment without bad
examples."* Validation (`edd golden`) fails if either class is missing.

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
edd golden          # validate the golden set (positives + clean negatives)
edd explore         # generate answers for every golden question and read them
edd ask "how long do refunds take?"
```

### Step 2 — Evaluate across all three layers

```sh
edd evaluate        # Layer 1 + Layer 2 over the golden set, with clean-negative recall
```

### Step 3 — Calibrate the virtual judge (Layer 3)

```sh
edd calibrate                 # % agreement + Cohen's kappa vs. human labels
edd calibrate --target 0.9    # raise the bar
```

If agreement is below target, analyze the printed disagreements, sharpen the
rubric / few-shots in [src/edd/calibration.py](src/edd/calibration.py), and re-run
until the judge is trustworthy.

### Step 4 — Run experiments, one variable at a time

Each entry in [evals/experiments.json](evals/experiments.json) isolates **one**
change (a model, a `retrieval_k`, or a prompt in
[src/edd/app.py](src/edd/app.py)). `edd run` evaluates it, compares required
metrics to the baseline, and lets **Git enforce the verdict**:

```sh
edd run baseline      # always kept — it defines the benchmark
edd run topk-2        # compared against baseline; kept or reverted automatically
```

- **Keep** — every required metric flat-or-better → the source change + result
  snapshot are committed.
- **Reject** — any required metric regressed → `src/` is reverted to the last
  clean state; only the result record is committed.
- **Plateau** — two consecutive flat keeps → the loop warns that prompt tweaks
  have run out and a structural change is needed (reranking, chunking, hybrid
  search).

### Step 5 — Inspect failures

```sh
edd report topk-2                      # worst cases + judge chain-of-thought
edd report topk-2 --metric groundedness --worst 3
```

### Step 6 — Mirror the evals in production

Evaluation is never one-and-done. `edd monitor` samples a fraction of a
de-identified traffic log ([evals/traffic.jsonl](evals/traffic.jsonl)), runs the
Layer-1 checks plus the answerability judge over each sampled query, and surfaces
flagged outputs (including likely fabrications) for human review.

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
├── data/support_policies/        # the RAG corpus (markdown policy docs)
├── evals/
│   ├── golden.jsonl              # golden set: positives + clean negatives
│   ├── experiments.json          # one-variable-at-a-time experiments + verdicts
│   ├── configs.json              # multi-config sweep definitions
│   ├── traffic.jsonl             # de-identified traffic for production monitoring
│   └── results/                  # per-experiment result snapshots
└── src/edd/
    ├── __init__.py               # enables TRULENS_OTEL_TRACING before imports
    ├── config.py                 # Settings (one experiment = one knob changed)
    ├── store.py                  # shared Qdrant Cloud document store
    ├── indexing.py               # Haystack indexing pipeline
    ├── app.py                    # instrumented Haystack RAG app (the seam)
    ├── dataset.py                # golden-set loader + validator
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
| Include bad examples in the gold set | `golden.jsonl` clean negatives + `dataset.py` guard |
| Evaluate the system, not just the model | instrumented `retrieve`/`generate` spans |
| Mirror evals in production | `edd monitor` |
| Appoint a decision-maker / gates | `experiments.json` acceptance + `report.decide` |
