# FinanceRAG

Elasticsearch BM25 retrieval for the **FinQA configuration only** of
`G4KMU/t2-ragbench`. Each unique `context_id` is one indexed document. Text,
Markdown tables, pre-text and post-text are preserved without chunking or
truncation. Questions and answers are exported separately and never indexed.

For a file-by-file explanation and architecture guide, see the single
[technical documentation README](<technical documentation/README.md>).

## Multi-turn RAG baseline and evaluation

`baseline_agent/` contains a Search-R1-style inference controller for
`Qwen/Qwen3.5-2B`. It loads one question from the pinned FinQA **test** split,
without putting gold context, answers, company metadata, or year metadata in
the model prompt. The model generates company/year/query tags and receives
evidence from the existing private Modal search service. Scoring is performed
locally using T2-RAGBench Number Match (epsilon 0.01). Yes/No predictions are
converted to 1/0 before comparison with the numeric gold answers.

```powershell
$env:PYTHONIOENCODING = "utf-8"
.\.tools\uv.exe run modal deploy modal_app.py
.\.tools\uv.exe run modal run baseline_agent/modal_app.py
```

The default selects test index 0 and permits up to six searches. It prints
the model's generated reasoning blocks, search arguments, and full information
blocks. The model runs on a temporary **T4 in FP16**; its exact revision is
recorded. Weights download on CPU into the `finance-rag-qwen-cache` Modal Volume
before a GPU is allocated. This is inference only, not RL training.

Results are saved under a unique ignored `data/baseline/experiments/<run-id>/`
directory as per-question JSON and text files, `results.csv`, and `summary.json`.
The JSON includes each actual model input, output, API response, timing, and
token counts. The text file is a readable complete interaction transcript.
CSV and summary files are updated after each question, preserving partial runs.
Historical one-example transcripts under `data/baseline/runs/` remain unchanged.
Only the newest information block remains in the active prompt by default;
generated reasoning/search history is retained. To use Search-R1's accumulating
history, add `--context-mode append`. To select another single question later,
use `--example-index N`; this command never runs the full test set.

The GPU job stops when the run completes. Stop the deployed CPU service too
when finished to prevent new calls from waking it:

```powershell
.\.tools\uv.exe run modal app stop --yes finance-rag-search
```

To run the first five questions with a local dashboard:

```powershell
.\.tools\uv.exe run --group observability modal run baseline_agent/modal_app.py --num-examples 5 --dashboard
.\.tools\uv.exe run --group observability tensorboard --logdir data/baseline/experiments --host 127.0.0.1 --port 6006
```

Open `http://127.0.0.1:6006`. Scalars show scores, latency, token counts, and
search attempts. Text shows expected/predicted answers, generated reasoning,
and tool requests. TensorBoard runs locally without a hosted account.
The model loads once per batch; each question uses fresh history and seed
`42 + example_index`. The CLI still defaults to one question.

CSV includes `correct_answer`, `our_answer`, normalized numeric values, score,
status, sign agreement, and timings. `inference_seconds` includes generation
and tool waits, but excludes download, model loading and readiness warmup.
Number Match intentionally ignores sign and power-of-ten differences; inspect
raw answers and `sign_agrees` as well. Unsupported gold is explicitly unscored.

Retrieval diagnostics are calculated locally after inference using the test
example's gold `context_id`; this identifier and the hit indicators never enter
the model prompt or GPU payload. `turns.csv`, per-example JSON/text, and
TensorBoard Text show whether each turn retrieved the gold context, whether it
was present in that generation's evidence, and whether it was ever retrieved.
Turns without a search have a blank hit field and `search_performed=False`.
These are exact context-ID checks; other contexts may contain valid evidence.
The completed v4 run has an annotated review copy under
`data/baseline/experiments/81c16ea03897/retrieval-review/`; its original predictions
and prompts were preserved. Select its `tensorboard` run to inspect the indicators.

To resume without rerunning saved questions:

```powershell
.\.tools\uv.exe run --group observability modal run baseline_agent/modal_app.py --num-examples 5 --dashboard --resume data/baseline/experiments/RUN_ID
```

Resume verifies model revision, selected questions and generation configuration.

GPU dependencies have a separate `baseline_agent/pyproject.toml` and `uv.lock`;
they are not installed in the local search environment. Regenerate the GPU
export with:

```powershell
.\.tools\uv.exe lock --project baseline_agent
.\.tools\uv.exe export --project baseline_agent --frozen --no-dev --no-emit-project --emit-index-url --output-file baseline_agent/requirements.gpu.lock.txt --quiet
```

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
