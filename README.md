**Instamart Customer Intelligence & Support Copilot**

# Overview

This repository implements a reproducible, four-part internal customer intelligence and support workflow using synthetic fictional data only. The Swiggy Instamart name is scenario framing only; no real customer data, pricing, policy, or production system is represented. Monetary values are INR. All model calls use the supplied offline `mock_llm.py`; there are no API keys, paid services, or network model calls.

The supplied `generate_data.py` and `mock_llm.py` are included unmodified. Part A cleans and stores the generated source data, Part B segments customers and trains a lapse model, Part C redacts and triages support tickets, and Part D retrieves policy text and runs a database-backed simulated refund agent.

# Reproduce

Run these numbered commands from the repository root in **Windows PowerShell**. Use Python 3.11 or newer. The commands regenerate the synthetic inputs and all clean tables, models, triage outputs, policy evaluations, agent logs, and checkpoints.

1. `python -m venv .venv`
2. `.\.venv\Scripts\Activate.ps1`
3. `python -m pip install --upgrade pip`
4. `python -m pip install -r requirements.txt`
5. `python generate_data.py`
6. `python run_part_a.py`
7. `python run_validation.py`
8. `python run_part_b.py`
9. `python run_part_c.py`
10. `python run_part_d.py`
11. `python -m unittest discover -s tests`

On macOS/Linux, activate the environment with `source .venv/bin/activate` instead of command 2. The generator's final line must be exactly `generated: customers=1200 products=40 orders_raw=13084 tickets=164`. Part A's validation script prints `ALL CHECKPOINTS PASSED`. Commands use fixed random seeds where applicable; the agent ledger includes timestamps, so its timestamp text naturally changes between runs.

# Part A

### Raw-data audit

The raw files were read with pandas and `keep_default_na=False`, preserving blank strings. Shapes and raw pandas dtypes are shown here; missing means an empty/whitespace string (rather than pandas converting it to NaN).

| Table | Shape | Column | Raw dtype | Missing count | Missing % |
|---|---:|---|---|---:|---:|
| orders | `(13084, 9)` | `order_id` | `object` | 0 | 0.00% |
| orders | `(13084, 9)` | `customer_id` | `object` | 0 | 0.00% |
| orders | `(13084, 9)` | `product_id` | `object` | 0 | 0.00% |
| orders | `(13084, 9)` | `quantity` | `int64` | 0 | 0.00% |
| orders | `(13084, 9)` | `order_ts` | `object` | 0 | 0.00% |
| orders | `(13084, 9)` | `status` | `object` | 0 | 0.00% |
| orders | `(13084, 9)` | `discount_pct` | `int64` | 0 | 0.00% |
| orders | `(13084, 9)` | `order_amount` | `object` | 0 | 0.00% |
| orders | `(13084, 9)` | `delivery_minutes` | `object` | 2541 | 19.42% |
| customers | `(1200, 5)` | `customer_id` | `object` | 0 | 0.00% |
| customers | `(1200, 5)` | `city` | `object` | 0 | 0.00% |
| customers | `(1200, 5)` | `signup_date` | `object` | 0 | 0.00% |
| customers | `(1200, 5)` | `age` | `object` | 61 | 5.08% |
| customers | `(1200, 5)` | `gender` | `object` | 30 | 2.50% |
| products | `(40, 4)` | `product_id` | `object` | 0 | 0.00% |
| products | `(40, 4)` | `product_name` | `object` | 0 | 0.00% |
| products | `(40, 4)` | `category` | `object` | 0 | 0.00% |
| products | `(40, 4)` | `price_inr` | `int64` | 0 | 0.00% |

Specific raw audit checkpoints:

| Audit item | Result |
|---|---:|
| Raw order rows | 13,084 |
| Exact duplicate order rows | 119 |
| Rows with `pending` status | 656 |
| Distinct raw `order_id` values | 12,309 |
| Non-numeric `order_amount` entries | 120 |
| Negative `delivery_minutes` | 106 |
| Blank `delivery_minutes` | 2,541 |
| Blank customer ages | 61 |
| Ages outside 10–100 | 6 |
| Distinct raw city spellings | 24 |
| Distinct nonblank raw gender spellings | 10 |
| Blank gender values | 30 |

### Cleaning rules and changes

- **Customers:** trim city whitespace and apply title case, reducing 24 raw city spellings to the six supported cities. Map all gender variants to `male`, `female`, or `unknown`; final counts are female 610, male 560, unknown 30. Convert ages to numeric, treat the six ages outside 10–100 as missing, and fill missing/invalid ages (61 blanks plus 6 invalid values) with the city median using `groupby(...).transform(...)`; final missing ages: 0 and final age range: 18–60. Convert `signup_date` to datetime. Final city counts: Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184.
- **Orders, in order:** parse `order_ts` as datetime; remove 119 exact duplicate rows; sort by timestamp and keep the latest row for each `order_id`, removing 656 further rows and resolving earlier `pending` records; parse amounts numerically and impute 112 remaining missing/unparseable amounts using `round(price_inr × quantity × (1 − discount_pct / 100), 2)`; set negative delivery durations to missing, then impute 774 missing completed-order delivery durations using the median for that product category among completed orders. Cancelled orders are not imputed: their 1,199 delivery durations remain missing.
- **Products:** preserve the 40-row catalogue and convert `price_inr` to numeric.
- Write clean CSVs to `data/clean/` and SQLite tables `customers`, `products`, `orders` to `data/instamart.db`. `order_ts` is saved as `YYYY-MM-DD HH:MM:SS`.

Final clean checkpoints: 12,309 orders, one per order ID; 11,110 completed and 1,199 cancelled; no pending rows; zero missing order amounts; zero missing delivery durations among completed orders; 1,199 missing durations among cancelled orders; completed revenue **INR 2,998,668.80**. The validator asserts 14 customer/order checkpoints and prints `ALL CHECKPOINTS PASSED`.

### SQL analysis

Seven runnable queries are stored in `sql/queries.sql` and each is executed with `pandas.read_sql_query`:

| Query | Result |
|---|---|
| Q1 — `WHERE` then `HAVING` | 30 customers have completed orders above INR 500 whose qualifying spend exceeds INR 8,000. |
| Q2 — three-table join | 30 city/category rows; revenue sums to INR 2,998,668.80. Largest cell: Bengaluru / Electronics, INR 233,909.45. |
| Q3 — scalar subquery | 11 products priced above the average catalogue price. |
| Q4 — `LEFT JOIN` | 25 customers have no completed order. |
| Q5 — CTEs | 550 customers are above average completed spend; average spend is INR 2,552.06 across customers with completed spend. |
| Q6 — window rankings | Top customers: Bengaluru C0082 (INR 12,856.20), Chennai C0930 (INR 10,151.40), Delhi C0291 (INR 13,742.60), Hyderabad C0743 (INR 12,730.05), Mumbai C0693 (INR 14,766.55), Pune C0426 (INR 10,377.10). Exactly 8 rows have `RANK()` different from `ROW_NUMBER()`: ties share a rank, while row numbers still uniquely order tied rows. |
| Q7 — running total | 183 daily rows; final cumulative revenue is INR 2,998,668.80. |

The pandas `pivot_table(..., margins=True)` grand total is also INR 2,998,668.80. City summaries: Bengaluru INR 523,643.40 total, INR 276.47 mean, 1,894 completed orders; Mumbai INR 479,058.50 total, INR 297.00 mean, 1,613 completed orders. Full city summaries and the pivot are in `outputs/city_spend_summary.csv` and `outputs/pandas_city_category_pivot.csv`.

### Exploratory analysis (completed order amounts)

- Mean INR 269.91; median INR 135.00; skewness 4.61.
- Q1 INR 64.80; Q3 INR 297.00; IQR INR 232.20; lower fence INR -283.50; upper fence INR 645.30.
- Outliers beyond fences: 0 below the lower fence and 897 above the upper fence.
- Largest absolute correlation is `order_amount` with `discount_pct`, approximately -0.095; `order_amount` with `quantity` is +0.087.

The data describes synthetic completed grocery and other product orders across six cities. Amounts are strongly right-skewed: the mean (INR 269.91) is well above the median (INR 135.00), and 897 amounts exceed the upper IQR fence. The right tail is plausible for a mixed catalogue that includes expensive electronics, although the planted dirty values are handled during cleaning. The strongest absolute correlation among the requested variables is the small negative relationship between amount and discount percentage (about -0.095). Correlation of this size does not establish causation. The correlations are weak overall and should not be treated as causal evidence.

# Part B

### RFM and segmentation

RFM uses completed orders as of `2025-10-01 00:00:00`. There are 1,175 customers with at least one completed order. Recency median 19 days, minimum 0, maximum 180; frequency mean 9.46 and maximum 33; monetary mean INR 2,552.06 and maximum INR 14,766.55.

K-means uses `StandardScaler` and `KMeans(n_clusters=K, n_init=10, random_state=42)`:

| K | Inertia | Silhouette |
|---:|---:|---:|
| 2 | 1,820.8 | 0.465 |
| 3 | 1,268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.481 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

K=4 is selected: it has the strongest silhouette score in this run and yields actionable profiles. Inertia always falls as K increases because adding clusters can only keep or reduce within-cluster squared distances; K=8 therefore has lower inertia than K=2 by construction. That alone does not select K because more clusters also add complexity and can create less useful, fragmented business segments.

| Segment ID | Size | Mean recency (days) | Mean frequency | Mean monetary (INR) | Segment name | Recommended action |
|---:|---:|---:|---:|---:|---|---|
| 0 | 436 | 28.86 | 3.77 | 1,057.62 | Steady Value | Use relevant replenishment reminders and value bundles. |
| 1 | 138 | 24.12 | 6.77 | 7,129.70 | Premium | Prioritise service recovery and personalised retention offers. |
| 2 | 419 | 8.77 | 19.40 | 3,410.02 | Frequent Essentials | Encourage repeat baskets and cross-category add-ons. |
| 3 | 182 | 109.04 | 2.20 | 685.99 | Dormant Re-engagement | Test a targeted win-back campaign. |

Segment ID is only a cluster label and has no inherent ordering. Segment 1 is Premium because it has the highest mean monetary value. PCA on the scaled RFM matrix explains 0.634 and 0.218 of variance in its first two components (total 0.852). The complete segment output is `outputs/customer_segments.csv`; customers with no completed order receive segment ID -1 and `No Completed Orders`, and are not marked Premium.

### Lapse definition, leakage prevention, and model results

The cutoff is `2025-09-01`. The cohort includes customers with at least one completed order before the cutoff; lapsed means no completed order during 1–30 September. There are 1,156 cohort customers and 443 lapsed (38.32%); the stratified split has 924 training and 232 test rows. Features use only orders dated before the cutoff and include recency, frequency, monetary, average discount, cancellation rate, average delivery duration, favourite product category, city, age, gender and tenure. Segment labels are not model features. The feature table is asserted to contain no missing values.

All preprocessing is inside a scikit-learn pipeline: the one-hot encoder (`drop="first"`) and numeric scaler are fitted only on training data (and on training folds during cross-validation), then applied to test rows. No encoder, scaler or resampler saw test rows. No data from 2025-09-01 onwards entered the features. The fitted scaler's training mean for frequency is 8.115; the full-cohort feature mean is 8.029. They differ because the scaler sees only the training split, whereas the full-cohort mean includes held-out test customers.

Baseline (always predict not lapsed): accuracy 61.6%, lapsed precision 0, recall 0, F1 0.

| Model | Grid | Best parameters | Best CV F1 | Test confusion matrix `[[TN, FP], [FN, TP]]` | Lapsed precision | Lapsed recall | Lapsed F1 |
|---|---|---|---:|---|---:|---:|---:|
| Logistic regression | `C ∈ (0.01, 0.1, 1, 10, 100)`, `max_iter=1000`, 5-fold F1 | C=100 | 0.651 | `[[111, 32], [31, 58]]` | 0.644 | 0.652 | 0.648 |
| Random forest | `max_depth ∈ (3, 5, 8)`, `min_samples_leaf ∈ (5, 10, 20)`, 100 trees, seed 42, 5-fold F1 | depth=8, leaf=10 | 0.681 | `[[113, 30], [31, 58]]` | 0.659 | 0.652 | 0.655 |

The Random Forest is selected because it has the better cross-validated and test F1 in this run and slightly better precision, with the same test recall as logistic regression. Missing a lapsing customer means losing an opportunity to retain them, while wrongly targeting a loyal customer wastes incentives and may annoy them; the retention team should choose its operating threshold based on the relative costs. Test F1 is 0.655, above the required 0.55.

Class-balance experiment (oversampling training rows only): the original training counts are 570 not-lapsed and 354 lapsed; the balanced training set has 570 in each class. On the unchanged test set, logistic regression moves from precision 0.644, recall 0.652, F1 0.648 to precision 0.589, recall 0.854, F1 0.697. Recall moves most, while precision falls. This finds more at-risk customers at the cost of more false-positive retention outreach.

Top Random Forest features by importance:

| Feature | Importance | Business reading |
|---|---:|---|
| Frequency | 0.280 | Order cadence is the strongest signal in this model. |
| Monetary | 0.271 | Historical spend also helps separate likely lapse behaviour. |
| Recency | 0.115 | Time since the last order contributes materially. |
| Average delivery duration | 0.061 | Delivery experience adds a smaller signal. |
| Cancellation rate | 0.052 | Previous cancellations add a smaller signal. |

Logistic-regression coefficient signs: frequency is negative (more historical orders associate with lower predicted lapse, holding other features fixed); monetary is positive in this fitted model; the Electronics favourite-category indicator is negative relative to the dropped reference category. These are model associations, not causal effects. The selected pipeline is saved to `outputs/lapse_model.joblib`, reloaded, and verified to reproduce test predictions exactly. `outputs/retention_list.csv` contains the top 10 Premium-flagged test customers by predicted lapse probability; 1 of those 10 truly lapsed. This retention list is illustrative and not a leakage-free evaluation because the Premium flag is a post-hoc whole-window label that includes the September window used to define lapse.

# Part C

### Guardrails, redaction, and classification

The ticket input is processed in file order with `keep_default_na=False`. Four blank tickets are `skipped_empty`; four long tickets estimate to 734 tokens and are `flagged_long` because they exceed `MAX_INPUT_TOKENS=400`. Neither group triggers a model call.

Redaction is applied in the specified order: email to `[EMAIL]`, phone to `[PHONE]`, order ID to `[ORDER_ID]`. Across all 164 tickets there are 55 email substitutions, 91 phone substitutions and 132 order-ID substitutions, 278 total. Reapplying the patterns to the redacted text finds 0 remaining matches. An assertion runs before each model call to ensure no PII pattern remains, and the user message contains only the redacted ticket text.

Both system prompts contain `CLASSIFY`, the four allowed categories, and JSON-only instructions. The zero-shot prompt omits `EXAMPLES:`; the few-shot prompt includes labelled examples, including fraud. Every model call is wrapped in a maximum-three-retry handler with exponential backoff and jitter (`0.05 × 2**n × uniform(0.9, 1.15)`). `mock_llm.reset()` runs before each independent full pipeline. The full run recovered all 19 transient failures; each required retry was retry number 0 and the logged wait was within 0.045–0.0575 seconds.

| Prompt | Classified tickets | Accuracy on classified tickets | Total response tokens | Tokens per classified ticket |
|---|---:|---:|---:|---:|
| Zero-shot | 147 | 85.7% (126/147) | 11,740 | 79.86 |
| Few-shot | 147 | 100.0% (147/147) | 18,292 | 124.44 |

Zero-shot confusion table (true category by predicted category):

| True \ Predicted | Billing | Delivery | Fraud | Technical |
|---|---:|---:|---:|---:|
| Billing | 38 | 0 | 0 | 0 |
| Delivery | 0 | 36 | 0 | 0 |
| Fraud | 11 | 0 | 23 | 0 |
| Technical | 10 | 0 | 0 | 29 |

Both runs produced 135 `classified_json`, 12 `classified_keyword`, 9 `flagged_invalid_output`, 4 `flagged_long` and 4 `skipped_empty`. Keyword recovery uses `re.findall(r"[a-z]+", text.lower())`, which correctly separates words such as `billing-related`; invalid JSON or an out-of-list category is never guessed and goes to human review.

The few-shot prompt costs an additional 6,552 tokens overall, or 44.57 extra tokens per classified ticket (124.44 vs 79.86), but increases accuracy by 14.3 percentage points in this synthetic run. That cost is worth paying when avoiding misrouting fraud and technical issues is more important than minimising token use; a higher-volume deployment should compare that benefit with its latency and capacity constraints.

The zero-shot budget demo uses a token budget of 6,000 and marks 76 tickets `budget_exhausted`; it stops making calls when accumulated usage has reached the budget before the next call. The final `outputs/triage_results.csv` is the few-shot run with budget 100,000 and contains all 164 tickets. Routing rules: empty/long/invalid/budget-exhausted/permanent-error → `human_review`, NORMAL; fraud → `security_team`, HIGH; otherwise Premium → `premium_desk`, HIGH; otherwise `support_queue`, NORMAL. Rule-validated categories are routed automatically; empty, long, invalid and budget-stopped cases are sent to humans. The budget demo is `outputs/triage_budget_demo.csv`.

# Part D

### Policy retrieval and grounded answers

The five policy documents are split into 40-word chunks with a 10-word overlap, yielding 15 chunks: refund 4, delivery 3, payment 3, escalation 3, privacy 2. `TfidfVectorizer()` is fitted on chunk text; query vectors are ranked by cosine similarity. TF measures term frequency within a text, while IDF downweights terms common across many texts. TF-IDF can miss semantically equivalent phrasings that use different words.

For the query `refund policy late delivery`, the top three chunks all come from `refund_policy.txt`. With overlap 10, 12/12 answerable questions retrieve the expected document in the top three, 12/12 answers contain the expected fact as a case-sensitive substring, and 2/2 unanswerable questions return the fallback beginning `I could not find`. With zero overlap, expected-document retrieval remains 12/12 and the two unanswerable fallbacks remain 2/2, but expected facts fall to 10/12. Two failures are: “What delivery time makes a completed order eligible for a full refund?” and “How soon is money returned when a payment fails but is deducted?” Without overlap, the key eligibility phrase and the payment-return timing sentence are split across chunk boundaries, so the retrieved context can contain only a fragment. Evaluation files are `outputs/policy_eval_overlap10.csv` and `outputs/policy_eval_overlap0.csv`.

### Refund tools, scenarios, and safe stopping

The `TOOLS_SPEC` list and `TOOL_MAP` dictionary define five tools: `get_order`, `search_policy`, `calculate_refund`, `issue_refund`, and `escalate_to_human`. The agent reads order data from `data/instamart.db`, reuses policy retrieval, and appends simulated issue/escalation actions to `outputs/refund_ledger.jsonl`. Only completed orders are eligible; delivery over 45 minutes gets a full refund; delivery in 45 minutes or less gets INR 0; a refund above INR 1,000 is escalated to a human. The scenarios are treated as raised within the policy's seven-day window.

| Scenario | Status | Refund amount (INR) | Model calls | Tokens used |
|---|---|---:|---:|---:|
| S1 | REFUND | 128.25 | 5 | 927 |
| S2 | REFUND | 135.00 | 5 | 927 |
| S3 | ESCALATE | 1,999.00 | 5 | 914 |
| S4 | DENIED | 0.00 | 4 | 629 |
| S5 | DENIED | 0.00 | 3 | 355 |
| S6 | NOT_FOUND | 0.00 | 2 | 84 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 3,036 |

Stop-safely demonstrations: S1 with `token_limit=200` ends as `FALLBACK_TOKEN_LIMIT`; S1 with `time_limit_seconds=0` ends as `FALLBACK_TIMEOUT` before the first model call. S7 reaches `FALLBACK_MAX_ITERATIONS` after six model calls under the default 6,000-token limit.

For the crash-and-resume demonstration, the simulated crash occurs after the second tool observation has been checkpointed. The tools executed across the original run and resume, in order, are exactly `get_order → search_policy → calculate_refund → issue_refund`; none is repeated, and the resumed result is `REFUND: INR 128.25 issued.` Checkpoints are stored under `state/`; `outputs/agent_runs.jsonl` contains one record per scenario, with tool arguments and observations.

### ReAct versus plan-and-execute (under 250 words)

This ReAct implementation uses 5 model calls for S1, 3 for S5, 2 for S6, and 6 for the intentionally looping S7. Each call can inspect a new observation before deciding what to do, which makes it straightforward to stop after a missing order (`NOT_FOUND`) or a cancelled order (`DENIED`). The trade-off is repeated prompt context and more model calls/tokens for ordinary cases. A plan-and-execute design could ask for a complete action sequence once, reducing calls and often token overhead when the path is predictable; it must still validate every tool result and stop or replan if the order is missing, cancelled, or otherwise unexpected. For refunds, I would keep a constrained ReAct flow or a hybrid: the decision path is short, but order status, delivery time, refund threshold and escalation are safety-critical. Deterministic tool-side checks and a human handoff are more important than minimising one or two calls.

# Limitations and Next Steps

- All customers, orders, tickets and policies are synthetic. The results are a reproducibility exercise, not evidence about real Instamart operations.
- The supplied offline model is deterministic and deliberately injects transient errors and invalid outputs. Real-world model quality, latency and cost are not evaluated.
- The lapse model is trained on one synthetic six-month order window and its performance may not generalise to real customer behaviour. Thresholds should be chosen based on the relative cost of missed lapses and unnecessary outreach.
- The Premium flag uses whole-window RFM including September, which overlaps the lapse-label period. Therefore, the retention list is illustrative and not a leakage-free evaluation; the model features themselves use only pre-cutoff orders and do not include segment labels.
- The refund ledger is a simulation and does not move money. A production implementation would need authenticated access, idempotency keys, audit controls, policy versioning, time-window validation, payment-provider integration and human approval enforcement.
- Further work could add monitoring for drift, calibrated lapse probabilities, threshold analysis, structured privacy testing and more robust semantic retrieval while retaining offline reproducibility.
