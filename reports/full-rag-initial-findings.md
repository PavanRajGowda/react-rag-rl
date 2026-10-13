# Initial findings: full multi-turn FinQA baseline

Experiment `b836aee0ff0d` evaluated Qwen3.5-4B on H100 FP16 with K=2,
replace history, at most six searches and eight generations. There were 1,146
scored questions and one blank skipped row. No model, prompt, scores or labels
were changed during this analysis.

## Overall outcome

- Number Match: **729/1,146 = 63.61%**.
- Forced-answer question-only: **95/1,146 = 8.29%** on identical questions.
- Paired results: 662 RAG-only passes, 28 question-only-only passes, 67 both
  passes and 389 both failures. The net NM gap is 55.32 percentage points.
- 1,006 questions produced final answers; 729 of those passed (72.47%).
- 126 reached the search limit and 14 the generation-turn limit.
- Zero infrastructure failures and zero context-limit outcomes were recorded.

This establishes the pre-RL baseline comparison, not an RL improvement or
a directly matched replication of the paper's evaluation population/budget.

## Retrieval versus answer production

| Canonical gold context retrieved at least once | NM pass | Answered but NM fail | No final answer |
|---|---:|---:|---:|
| Yes | 700 | 232 | 35 |
| No | 29 | 45 | 105 |

The canonical context was found for **967 questions (84.38%)**. Of 417 NM
failures, 267 had retrieved that context. This suggests a substantial opportunity
in interpreting evidence and deciding when to answer, rather than retrieval
alone. However, context-ID matching is a diagnostic: alternative evidence can
exist, and question/gold consistency is not guaranteed.

The first search found the gold ID for 897 questions (78.27%). Later searches
found it for another 70 (6.11 percentage points); 49 of those eventually passed.
This demonstrates observed recoveries, not a causal single-turn ablation.
907 questions used one API call; most cases therefore behave like a single-lookup
agent, with multi-turn exploration concentrated in the difficult tail.

## Unproductive looping and report-year selection

120/126 search-limit cases repeated an identical normalized request. Across all
questions, 129 repeated a request and 148 repeated a nonempty result set.
The 140 no-answer cases are 12.22% of scored questions but consume **31.47%**
of episode inference time. Simply increasing the search budget is not supported
by these observations: loops often need a different query, report year or stop
decision. Difficulty confounds any association between search count and accuracy.

For 109 questions the first filtered request excluded the canonical context.
In 58 eventual limit cases the agent never changed its requested year. Under
Armour (`finqa_test_57`) is a representative event-year versus report-year case:
it kept searching the 2005 report for IPO details. The companion JSON records
the gold report year and query history. Ninety-three questions received report-
year-unavailable feedback; 43 eventually passed NM.

Thirty-five limit cases had found the gold context at some point, with 31 still
having it active at the last generation. AES (`finqa_test_68`) kept looking for
refinancing/LIBOR information although the reference calculation was the retired
principal times its coupon. Scope interpretation matters as well as stopping.

## Financial correctness and reward design

The benchmark deliberately ignores signs and powers of ten. A separate signed,
same-scale diagnostic matched **491/1,146 = 42.84%**, using
`abs(pred-gold) <= max(1e-6, abs(gold)*0.01)`. This is not the paper's metric
and should not replace official NM silently.

Among NM passes, 215 have scale discrepancies and 28 have opposite signs;
five overlap. Of the scale-discrepancy questions, 183 mention percentages,
ratios or proportions. These are diagnostic discrepancies, not automatically
proof of incorrect financial interpretation: units and conventions may differ.
They nevertheless show that NM alone does not enforce our requested signed,
fraction-scale output behavior. Reward design should monitor these separately.

## Dataset consistency needs review

Manual review identified likely question/gold misalignment:

- **Fortune Brands, `finqa_test_56`:** the rewritten question asks for change
  from 2015 to 2017. The source states $46.21 (2015), $57.37 (2016), and $63.43
  (2017). The asked calculation is about 37.26%; the gold fraction 0.10563
  instead matches 2016 to 2017. The model's calculation follows the asked years,
  although its percent-scale output violates our formatting instruction.
- **E*TRADE, `finqa_test_51`:** the question specifies 2010. The source identifies
  $19.3 million of counterparty collateral for 2010, but the gold ratio 4.6 uses
  the 2011 $2.3 million numerator divided by the 2010 $0.5 million denominator.
  The model also makes its own scope error, using $0.8 million of securities
  pledged by the company instead of collateral received from counterparties.
- **American Water Works, `finqa_test_83`:** the question's gross-liabilities
  wording may be broader than the source/gold's gross unrecognized-tax-benefit
  liabilities, encouraging a search for a company-wide denominator.

These examples came from a failure-focused sample; they do not estimate dataset
noise prevalence. Official results remain unchanged. Before RL reward design,
review corresponding patterns in train/dev so correct reasoning is not penalized
for a mismatched reference question. Keep any cleaned diagnostic analysis
separate from the frozen benchmark score.

## Operational profile

Median episode time was 12.91 seconds; mean 19.19 seconds and p95 54.37 seconds.
Generation consumed about 97.9% of episode time, versus about 2.0% in search calls.
Median search-call time was 0.18 seconds, with p95 0.55 seconds. There is no
evidence here that Elasticsearch capacity is the principal latency bottleneck.
Only 37 questions (3.23%) received invalid-protocol feedback, totaling 95 events;
formatting is a smaller remaining issue than the completion/scoring gaps.

## Priorities for subsequent RL experiments

1. Reward evidence-consistent value/period/scope selection and correct arithmetic.
2. Learn useful query/year revisions instead of repeating unchanged requests.
3. Learn to answer once sufficient evidence is available, preserving easy cases.
4. Monitor sign/scale compliance alongside NM and audit noisy training references.

These are hypotheses from baseline analysis. Preserve the evaluated baseline;
develop and validate training choices on train/dev before measuring held-out gains.

Machine-readable aggregates are in `full-rag-initial-analysis.json`, per-question
diagnostics in `full-rag-case-diagnostics.csv`, and paired outcomes in
`full-test-baseline-comparison.csv`. Original traces remain in the experiment
folder for direct review.
