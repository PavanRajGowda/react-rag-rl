# FinanceRAG

Elasticsearch BM25 retrieval for the **FinQA configuration only** of
`G4KMU/t2-ragbench`. Each unique `context_id` is one indexed document. Text,
Markdown tables, pre-text and post-text are preserved without chunking or
truncation. Questions and answers are exported separately and never indexed.

For a file-by-file explanation and architecture guide, see the single
[technical documentation README](<technical documentation/README.md>).

## H100 pilot and full-test evaluation

The baseline uses pinned `Qwen/Qwen3.5-4B` in FP16 on one H100. The default
pilot selects the first 50 test rows, with K=2 and replace history. Questions
run sequentially in chunks of 10; each GPU invocation has a one-hour timeout.
Weights download on CPU before GPU allocation. Full output traces are saved,
but terminal reasoning streaming is off by default so the progress bar stays
readable. Add `--verbose` for live generated text.

Run the 50-example pilot from the project directory:

```powershell
$env:PYTHONIOENCODING = "utf-8"
.\.tools\uv.exe sync --frozen --group observability
.\.tools\uv.exe run modal deploy modal_app.py
try {
    .\.tools\uv.exe run --group observability modal run baseline_agent/modal_app.py --num-examples 50 --dashboard
} finally {
    .\.tools\uv.exe run modal app stop --yes finance-rag-search
}
```

A terminal bar shows processed questions out of 50, elapsed time, ETA, accuracy
and infrastructure-error count. Startup/download can take time before the first
question completes. Do not redirect output if you want the live terminal bar.
The experiment directory is printed before work starts. Results, raw responses,
per-turn diagnostics and an atomic run manifest are saved after each example.
Model outputs are never supplied gold answers or gold-context indicators.

Review `summary.json`: require 50 completed rows and zero infrastructure errors,
and inspect `status_counts` for turn/context limits before expanding the run.
Number Match ignores sign and powers of ten; passing scores do not guarantee
percentage-format compliance. Inspect raw predictions as well.

If interrupted, redeploy search and resume using the printed directory:

```powershell
$run = "data/baseline/experiments/YOUR_RUN_ID"
.\.tools\uv.exe run modal deploy modal_app.py
try {
    .\.tools\uv.exe run --group observability modal run baseline_agent/modal_app.py --num-examples 50 --resume $run --dashboard
} finally {
    .\.tools\uv.exe run modal app stop --yes finance-rag-search
}
```

Successful saved examples are skipped. To repeat saved infrastructure or local
postprocessing failures, add `--retry-errors`. Incorrect numeric answers are
never selectively rerun by this flag. Resume refuses changed model, GPU,
instructions, generation configuration or dataset provenance. Keep the same
`--example-index`, `--top-k`, `--context-mode` and `--max-searches` when resuming.

After reviewing the pilot, extend that same experiment to the full test split:

```powershell
.\.tools\uv.exe run modal deploy modal_app.py
try {
    .\.tools\uv.exe run --group observability modal run baseline_agent/modal_app.py --full-test --resume $run --dashboard
} finally {
    .\.tools\uv.exe run modal app stop --yes finance-rag-search
}
```

This selects all 1,147 rows, records the known blank question as skipped, and
continues with the remaining questions. The pilot's saved results are reused;
there are 1,146 nonblank questions overall. `--batch-size` defaults to 10 and
`--attempts` to 3. Each failed worker stream retries only unsaved questions. After
bounded attempts, missing results are recorded as infrastructure errors. A
whole chunk failing repeatedly opens a circuit breaker to avoid repeated GPU
allocation for a global failure. Per-example exceptions are recorded and the
worker proceeds; CUDA OOM ends its chunk so later questions can start a fresh
model invocation. This cannot prevent process kills or disk failures; durable
raw results/checkpoints support recovery, and failure states are not hidden.

For a selection check without allocating any GPU or calling search:

```powershell
.\.tools\uv.exe run modal run baseline_agent/modal_app.py --num-examples 50 --dry-run
.\.tools\uv.exe run modal run baseline_agent/modal_app.py --full-test --dry-run
```

TensorBoard remains local:

```powershell
.\.tools\uv.exe run --group observability tensorboard --logdir data/baseline/experiments --host 127.0.0.1 --port 6006
```

Open `http://127.0.0.1:6006/#text`. Each generation shows its gold-context hit,
whether that context was in its prompt, active evidence afterward and ever-hit
status. These exact context-ID diagnostics are computed locally, not fed to the
model; alternative valid evidence may exist. `turns.csv` provides the same data.
In replace mode all previous model outputs and formatting feedback remain,
while only the latest API information block is retained.

GPU dependencies remain separately pinned in `baseline_agent/uv.lock` and
`requirements.gpu.lock.txt`. The root uv environment handles orchestration,
progress and scoring without local Torch. Old experiment snapshots were removed
at user request; the existing five-question 4B result is a historical T4 run.

## Question-only (Pretrained-Only) baseline

Use the same pinned instruction-tuned 4B model without any retriever or supplied
documents. This is a no-retrieval evaluation setting, not a switch to base-model
weights. It allows exactly one generation, no repair turns, and no search calls.
The model must provide its best numerical answer or Yes/No even if uncertain;
abstention is not allowed. If it still produces unknown or nonnumeric text,
that is recorded as a policy violation, not replaced with an invented answer.
Gold answers remain local; invalid output counts as incorrect.
The model may use numbers already present in the question and its prior knowledge.

Start with five examples; no Elasticsearch deployment is needed:

```powershell
$env:PYTHONIOENCODING = "utf-8"
.\.tools\uv.exe run --group observability modal run baseline_agent/modal_app.py --num-examples 5 --pretrained-only --dashboard
```

The existing progress/checkpoint/retry controls apply. Keep `--pretrained-only`
when resuming; RAG and question-only runs cannot share a checkpoint directory.
Retrieval metrics are not applicable for question-only runs. To compare the same
first five against the existing 50-question RAG pilot:

```powershell
.\.tools\uv.exe run python -m baseline_agent.compare --rag data/baseline/experiments/b836aee0ff0d --pretrained data/baseline/experiments/b06a90daab88 --num-examples 5 --output reports/pretrained-only-first-five-comparison
```

The earlier abstention-allowed five-question check scored 1/5 Number Match without retrieval, versus 5/5
for RAG on those same questions. Masco was the sole question-only pass and its
question supplies the required numbers. This does not prove memorization or
rule out all pretraining contamination. The interrupted larger run is archived
outside the dashboard under `data/baseline/interrupted/`; the five completed
outputs were reused without regenerating them.

The fresh forced-answer check is experiment `41b9ecf7703e`: all five generated
numbers with zero search calls and no abstentions, but only Masco passed NM
(1/5). Predictions were 0, 0.85, 0, 0.00 and 1.1197. Forced guessing probes
answer production without retrieval; correct guesses do not prove memorization,
and low accuracy does not eliminate all training-contamination concerns.
Results are saved separately under `reports/pretrained-forced-first-five*`.
The policy change requires a new question-only run; old control checkpoints
cannot be resumed under the changed prompt. RAG behavior is unchanged.

## Explore with Kibana

Kibana is an optional, password-protected learning environment in the separate
Modal app `finance-rag-kibana`. It starts Elasticsearch and Kibana 8.19.3
together and indexes the same immutable FinQA corpus as the training service.
It is a separate copy, not a live view of a training container's cluster.

```powershell
$env:PYTHONIOENCODING = "utf-8"
.\.tools\uv.exe run python -m finance_rag.kibana_setup
.\.tools\uv.exe run modal deploy modal_kibana.py
```

Open the HTTPS dashboard URL printed by the deployment. Log in as `researcher`
using the password in `data/kibana-login.txt`. Credentials are generated
locally, ignored by Git, and supplied to Modal as a secret. Never share the
credential files. Elasticsearch is only accessible inside this container;
the browser uses Kibana's native login over Modal HTTPS.

In **Discover**, select **FinQA contexts**. This data view has no time field,
so there is no date window hiding older financial reports. You should see
**2,789 documents**. Add columns such as `company_name`, `ticker`,
`report_year`, `page_number`, and `context_id`. Expand a row to inspect the
complete `text`, `table_markdown`, `pre_text`, `post_text`, and source metadata.
Try the KQL filter `ticker: "AAPL" and report_year: 2002`.

In **Dev Tools → Console**, run:

```http
GET finqa-*/_count

GET finqa-*/_mapping

GET finqa-*/_search
{
  "size": 5,
  "query": {
    "bool": {
      "filter": [
        {"term": {"ticker": "AAPL"}},
        {"term": {"report_year": 2002}}
      ],
      "must": [{"match": {"text": "net sales"}}]
    }
  }
}
```

The researcher can read FinQA and manage Kibana views/dashboards, but cannot
modify FinQA documents. The corpus index is also write-blocked.

To verify every indexed source against the corpus, not just its count:

```powershell
.\.tools\uv.exe run python -m finance_rag.kibana_verify YOUR_KIBANA_HTTPS_URL
```

Resources: two CPU cores, 6 GB memory, zero minimum containers, one maximum,
60-second idle scale-down. Browsing and Kibana's background browser requests
can keep the container active and incur charges. Close the tab when finished;
for a complete stop, use:

```powershell
.\.tools\uv.exe run modal app stop --yes finance-rag-kibana
```

Redeploy `modal_kibana.py` to make it available again. Cold startup includes
both services and can take several minutes. FinQA and its default data view
are recreated on each start. Custom saved dashboards currently reset on a
container restart; durable dashboard storage can be added later.

Verification: all 2,789 indexed contexts match the prepared corpus byte for
byte after canonical JSON serialization. Native authentication protects the
document APIs. Results are recorded in `reports/kibana-verification.json`.
The dashboard was stopped after verification to avoid ongoing runtime charges;
use the deployment command above when you want to explore it.

## Environment

Python 3.11, managed by uv. Dependencies are pinned in `uv.lock`.
The repository-local uv executable is currently `.tools/uv.exe` on Windows;
replace `uv` below with `.\.tools\uv.exe` if uv is not on PATH.

```powershell
uv sync --frozen
uv run pytest -q
uv run ruff check finance_rag tests modal_app.py modal_external.py modal_kibana.py
```

## 1. Prepare the corpus

```powershell
uv run python -m finance_rag.prepare --revision adf7fe1541ac37351ce1142544d8e3b43010ed92
```

Output in `data/finqa`: `contexts.jsonl`, `labels.jsonl`, `manifest.json`.
The manifest records the dataset commit, content hash, counts, context lengths,
and split overlap. Conflicting rows for a shared context ID fail ingestion.
The audited manifest is also recorded in `reports/finqa-manifest.json`.
Do not treat whitespace word counts as model token counts.

## 2. Start Elasticsearch and index

Docker Compose is for local development only; it binds an unauthenticated
single-node Elasticsearch instance to localhost. Docker is not installed on
the current development machine. Alternatively, configure an existing
Elasticsearch 8.x endpoint using the environment variables below.

```powershell
docker compose up -d
$env:ELASTICSEARCH_URL = "http://localhost:9200"
$env:ES_REPLICAS = "0"
uv run python -m finance_rag.index
```

For Elastic Cloud, set `ELASTICSEARCH_URL` to its HTTPS endpoint and set
`ELASTIC_API_KEY`. Use separate indexing and read-only serving API keys.
For a VM, use authenticated TLS Elasticsearch; do not expose the development
Compose configuration publicly. A single-node VM needs `ES_REPLICAS=0`.

The corpus version remains `finqa-301676584646`; the current index name is
`finqa-301676584646-search-v4`, including the company-resolution schema version.
Existing indexes with keyword-only company names must be reindexed into the new
name. Modal containers build it automatically after a fresh deployment.
The indexer verifies the document count and blocks future writes. It refuses
to overwrite an existing version. A failed partial build must be inspected
and removed explicitly before retrying.

## 3. Serve and query

Environment variables are read from the process; `.env` is not auto-loaded.

```powershell
$env:SEARCH_API_TOKEN = "your-long-random-token"
uv run uvicorn finance_rag.api:app --host 127.0.0.1 --port 8000
```

`POST /v1/search`, with `Authorization: Bearer <SEARCH_API_TOKEN>`:

```json
{"ticker":"AAPL","year":2002,"query":"net sales 2001 2000","top_k":5}
```

`year` always means **report year**, not the year of a table value. Provide
company_name, ticker, or both. Both must identify the same company. Names use
Elasticsearch's standard analyzed text matching: `apple` can match `Apple Inc.`.
For multiple words, every analyzed word must match the issuer name. Tickers
remain case-insensitive exact matches. All resolution and retrieval use
Elasticsearch. If no issuer name matches, search document text using company
name plus query across all issuers and report years. The response explicitly
explains that both filters were relaxed and includes each result's actual issuer
and report year. No synonym extraction, synonym matching, or local text scan is
performed.

Ambiguous issuer names return candidate names, tickers and available years.
Known issuers lacking the requested year return `report_year_unavailable`;
they never silently switch to unrestricted retrieval. Responses include
`match_method`, `message`, requested identifiers, applied filters and full
contexts. Empty searches use `code: no_results`; backend failures remain errors.
`top_k` defaults to two (allowed range 1–10). Financial formatting is preserved.

Schema v4 removes the derived organization field. Existing indices remain
untouched; reindex external Elasticsearch or redeploy Modal to build v4:

```powershell
.\.tools\uv.exe sync --frozen
.\.tools\uv.exe run python -m finance_rag.index
# Temporary CPU-only integration check:
.\.tools\uv.exe run modal run modal_app.py --verify-resolution
# Deploy the search service and optional Kibana copy:
.\.tools\uv.exe run modal deploy modal_app.py
.\.tools\uv.exe run modal deploy modal_kibana.py
```

The index command requires a running Elasticsearch endpoint. Modal indexes the
original prepared contexts at startup. No organization-preparation command or
artifact is required. Stop deployments when finished to avoid further usage.

The API verifies local corpus integrity and remote index metadata at startup.
Protected `/health` checks backend connectivity; `/docs` exposes the schema.

## 4. Measure retrieval

```powershell
uv run python -m finance_rag.evaluate --split dev
```

This baseline uses **gold company and report year filters** and the question
as query. It reports Recall@5, Recall@10, MRR@10 and per-question ranks. These
are retrieval metrics under oracle metadata, not end-to-end agent scores.
Use dev for tuning; keep test for final evaluation. Actual audit: no shared
context IDs across splits, but many shared reports. A report-disjoint research
evaluation is still needed, alongside the official splits.
The audit also found one blank test question; labels are preserved, and the
evaluator explicitly reports its exclusion rather than inventing a query.

## 5. Run Elasticsearch on Modal CPU containers (default)

The default deployment uses the official Elasticsearch 8.19.3 image on Modal,
with Python added. No Docker daemon or VM is needed. Elasticsearch runs as
UID 1000, listens only on the container's loopback interface, and is accessed
through an authenticated Modal class method. There is no public HTTP endpoint
in this mode and no `finance-search` secret is required.

```powershell
uv export --frozen --no-dev --no-emit-project --output-file requirements.lock.txt --quiet
uv run modal run modal_app.py
uv run modal run modal_app.py --evaluate --split dev
uv run modal deploy modal_app.py
uv run modal run modal_app.py --verify-deployed
uv run modal run modal_app.py --verify-companies
```

On Windows, set `$env:PYTHONIOENCODING = "utf-8"` before using the Modal CLI.

Each CPU container starts Elasticsearch and indexes all 2,789 intact contexts
before accepting searches. Index files are local to that container; they are
rebuilt on cold start from the versioned bundled corpus. This deliberately
avoids sharing live Lucene files across containers or committing a writable
index to a Modal Volume. The search service bundles contexts and the manifest,
**not QA labels**. Evaluation sends only search arguments from the local caller.

Resources: two CPU cores, 4 GB RAM, 1 GB Java heap, up to 16 concurrent calls,
one container maximum, scale to zero after 300 idle seconds. Cold starts cost
startup/indexing time; training should issue a warmup call before GPU rollouts.
The container's CPU/memory runtime and idle grace period incur Modal charges.

Call from a Modal GPU function or a locally authenticated Modal client:

```python
import modal

SearchService = modal.Cls.from_name("finance-rag-search", "SearchService")
search = SearchService()
result = await search.search.remote.aio({
    "ticker": "AAPL", "year": 2002,
    "query": "net sales 2001 2000", "top_k": 5,
})
```

The response has `status="ok"` and complete evidence, or a structured argument
error. Infrastructure failures raise rather than impersonating empty results.
Deployment preserves the same company/year/query schema as the HTTP API.

## Alternative: external Elasticsearch and HTTP API

The original deployment is retained in `modal_external.py` under the separate
app name `finance-rag-search-http`. For Elastic Cloud or a VM, index using
`finance_rag.index`, configure the `finance-search` Modal secret with
`ELASTICSEARCH_URL`, `ELASTIC_API_KEY` and `SEARCH_API_TOKEN`, then deploy
`modal_external.py`. This uses an always-warm HTTP wrapper around a persistent
external index. Do not deploy it when using the private default above.

## Current status and remaining steps

- Complete: uv environment, locked dependencies, real dataset audit/export,
  indexer, API, retrieval evaluator, local Compose and Modal deployment code.
- Modal search and Kibana have been deployed and verified, then stopped to
  avoid ongoing runtime charges. Deployment definitions and historical
  verification reports are retained in the repository.
- Real Elasticsearch indexing and source-preservation checks passed on Modal.
  Development baseline (883 questions, gold company/year filters): Recall@5
  99.43%, Recall@10 100%, MRR@10 0.9363. Local-to-Modal warm request latency at
  concurrency eight: median 177 ms, p95 212 ms. First call including startup
  took 42 seconds. These are retrieval metrics, not LLM answer accuracy.
- A second Modal CPU container successfully called the deployed search class:
  first call 41.0 seconds, repeated warm call 212 ms. Verification is recorded
  in `reports/modal-deployed.json`; no GPU was required for this connectivity
  check. Actual GPU rollout throughput remains to be measured.
- Forty-four component tests and lint checks pass with observability dependencies. Startup requires
  `-XX:+StartAttachListener` for Java self-attach under Modal's gVisor runtime;
  Elasticsearch's entitlement protections remain enabled.
- The default hosting target is Modal CPU containers; VM provisioning is no
  longer required. Review measured cold-start and retrieval latency before
  increasing rollout concurrency.
- Later: company aliases, batch search, report-disjoint evaluation, model
  token-budget checks and Search-R1-style training integration. The model
  should generate tool arguments; hidden gold metadata must not fill them in
  during end-to-end experiments. Keep retrieved observations out of policy
  generated-token loss.

## Local evaluation dashboard

Start the read-only dashboard from the project root:

```powershell
.\.tools\uv.exe run --frozen python -m baseline_agent.dashboard
```

Open http://127.0.0.1:8501. Stop with Ctrl+C in that terminal. This uses saved
experiment files only; no Modal, model inference, or Elasticsearch is started.
No extra dependencies are needed. Select a primary run and optionally a comparison
run. Click an evidence/outcome matrix cell to inspect its questions, then open a
question for model outputs, exact generation inputs, API responses and context.
Use Refresh runs after new checkpoints are saved. Export diagnostics as CSV.

Number Match retains saved benchmark scores. Signed numeric and exact text match
are supplementary diagnostics. Reference retrieval means canonical context-ID
match, not proof of complete evidence. Active-at-answer is N/A without a final
answer. Format checks use the current parser, not a replay of historical parser
code. Reference-excluding issuer/year filters compare against optional local
corpus metadata; these are evaluation diagnostics, not live retrieval fallbacks.
The dashboard never modifies original runs or sends diagnostics to the model.
Use `--port 8502`, `--experiments PATH`, or `--corpus PATH` for alternate locations.

### Original FinQA reasoning labels

The dashboard now groups performance by original gold-program operation count:
1, 2, and 3+ operations, with clickable question filters and a paired-run table.
Question details include the original question, program, source ID and support IDs.
These describe numerical reasoning steps, not multiple retrieval hops.

Rebuild the evaluation-only join (public downloads, no model inference):

```powershell
.\.tools\uv.exe run --frozen python -m baseline_agent.finqa_reasoning --download
```

Original files are pinned under `data/finqa_original/`; the sidecar is
`data/finqa/reasoning_annotations.json` and the tracked audit is
`reports/finqa-reasoning-join.json`. All 8,281 rows match by split/index, exact
source report/page and original answer, with null/blank answers normalized.
The join checks one-to-one IDs and agreement between program operation count and
annotated steps. It does not certify reformulated question meaning. Dashboard
labels require matching dataset revision, question and gold answer; otherwise
they display unavailable. Existing prediction files and scores remain unchanged.

### Numerical dependency depth and search recovery

The dashboard separates three labels: total gold-program operation count,
numerical reasoning dependency depth, and later-search reference recovery.
Depth is the longest chain of operation nodes ending at the final program
result. Direct calculation is depth 1; depth 2+ is dependent multistep numerical
reasoning. References #0, #1, etc. connect operations. Parallel operations do not
increase depth until their results feed another operation; unused branches are
excluded from final-answer depth. Invalid/self/forward references are rejected.
These are labels from the original gold program, not minimal reasoning depth or
proof that multiple searches are required. Question details show dependencies.

Later-search reference recovery means first search missed the canonical context
and a later search found it. The denominator is searched questions with a first-
search miss; no-search questions are excluded. Recovery accuracy uses recovered
questions only. Neither metric establishes true multihop retrieval requirements.

## GRPO implementation review: Commit 1 only

The `grpo_training/` package currently provides configuration, train/dev data
boundaries and local preflight provenance. It does not contain a trainer or
launch GPU jobs. Run the CPU-only inspection from the project root:

```powershell
.\.tools\uv.exe run --frozen python -m grpo_training.inspect
.\.tools\uv.exe run --frozen python -m grpo_training.inspect --config grpo_training/configs/smoke.json --train-count 4 --dev-count 4
```

Add `--output-root data/grpo/preflight` to save a local review record. It refuses
to overwrite an existing record of the same identity. All records say
`execution_status: not_started`; these are not training checkpoints.

The root environment runs inspection and CPU tests. The separate
`grpo_training/pyproject.toml` + `uv.lock` target Linux x86_64 Python 3.11 for
future Modal training. Do not sync that GPU environment on Windows. Check its
lock without installing it using:

```powershell
.\.tools\uv.exe lock --project grpo_training --check
```

The training lock resolves TRL 1.15.0, vLLM 0.22.0, Torch 2.11.0,
Transformers 5.5.4, PEFT 0.18.1, Accelerate 1.12.0 and datasets 4.8.5.
Resolution is not an import/kernel/adapter-compatibility test; those checks
remain in later reviewed commits. The baseline environment is unchanged.
See `technical documentation/README.md` for the file-by-file review guide.

### GRPO review Commit 2: outcome reward

`grpo_training/rewards.py` wraps the existing Number Match scorer without changing
its epsilon, sign/scale handling or Yes/No mapping. Accepted final answers earn
1 for a match and 0 otherwise. Unfinished model episodes earn 0; infrastructure
and postprocessing failures return an unscorable `None`. Invalid gold labels or
mismatched question IDs raise errors. No additional format or retrieval reward is
included. See the technical README for the status table and review details.

```powershell
.\.tools\uv.exe run --frozen python -m pytest tests/test_grpo_rewards.py -q
```

This module does not call a model, query Elasticsearch, retry groups or run GRPO.
