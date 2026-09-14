# Movie Recommendation System — MovieLens 1M

A recommendation-system case study built as a portfolio piece for
product/ML analyst roles: it compares popularity, content-based, item-based
collaborative, matrix-factorization and hybrid recommenders on MovieLens-1M
under one honest, leakage-free evaluation protocol, then turns the offline
result into a product decision and an online test design.

This is a rebuild of an earlier version of this project. The rebuild started
from a full audit of that version's code and conclusions rather than from a
blank page — several of its numbers turned out to be wrong or not
reproducible, and the audit findings below are as much a part of the
deliverable as the final metrics.

## Executive Summary

- **A matrix-orientation bug had made ALS look broken.** The previous
  version fit `implicit`'s ALS on a `(items, users)` matrix instead of
  `(users, items)`; the call did not error, it silently scored every user
  against the wrong factors, producing NDCG@10 ≈ 0. Fixing the orientation
  brings ALS to **NDCG@10 = 0.091** — a competitive model, not a broken one.
  See `notebooks/02_evaluation_setup.ipynb`, Section 4.4.
- **Popularity is a real, competitive baseline** (NDCG@10 = **0.074**) that
  the previous version's headline comparison omitted entirely. It still
  loses to every personalized model here, but it beats the ranking-loss
  factorization model (BPR, NDCG@10 = 0.066) — a claim that can only be made
  once popularity is actually in the same table.
- **Item-kNN and a validation-tuned hybrid are the best single models**
  (NDCG@10 ≈ **0.104 / 0.103**, effectively tied), while a pure
  **content-based (genre) model is the weakest on accuracy** (NDCG@10 =
  0.017) but reaches **83% of the catalog** across users, vs. 13% for
  Item-kNN — accuracy and catalog coverage pull in different directions.
- **Routing users to a model chosen per history-size segment beats every
  single model on every offline metric measured**: NDCG@10 = **0.107**
  (best single model: 0.104) and catalog coverage of 21% (vs. 13% for
  Item-kNN alone) — a concrete, validation-selected case for a segment-aware
  policy, proposed here as an A/B test, not a guaranteed production win.
- **LightFM could not be installed** on this environment (its legacy build
  script fails under modern `setuptools` on Windows/Python 3.12); `implicit`'s
  BPR was substituted as the ranking-loss model. It underperformed ALS and
  Item-kNN here — reported as-is rather than hidden.

## Recommendation Problem

Generate a ranked list of movies a user has not yet seen, using their rating
history and each movie's genre/year metadata, such that the movies they go
on to rate highly actually appear near the top of that list. Models are
compared as **rankers**, using Recall/Precision/MAP/NDCG@10, not as rating
predictors (RMSE was intentionally not used — see `02_evaluation_setup.ipynb`
for why ranking metrics fit this task better).

## Dataset

MovieLens-1M: 1,000,209 ratings from 6,040 users on 3,706 rated movies
(3,883 in the full catalog; 177 have never been rated), ratings 1–5,
timestamps, and movie genres. The user-movie matrix is **95.5%** sparse
(`1 - 1,000,209 / (6,040 × 3,706)`) — the previous version of this README
stated "~99.9%", which does not match what its own notebook computed; this
is corrected here.

Full analysis: `notebooks/01_data_and_eda.ipynb`.

## Evaluation Setup

**Split.** Chronological, per user: each user's interactions are sorted by
timestamp and cut into contiguous **train (~70%) / validation (~15%) / test
(~15%)** blocks. MovieLens-1M guarantees at least 20 ratings per user, so
every user contributes to all three blocks (minimum 14 train interactions).
A random split was rejected because rating activity is strongly non-uniform
over time (`01_data_and_eda.ipynb`) — it would let a model see the future of
some of its own training users.

**Relevance.** `rating >= 4` counts as a relevant recommendation (57.5% of
all ratings). The distribution is skewed toward high ratings (mode = 4,
median = 4); `rating >= 3` would mark 83.6% of interactions "relevant" —
too lenient to be a useful target. This definition is applied identically
to every model, in every split. A robustness check against `rating >= 3`
(`03_models.ipynb`, Section 4) confirms it does not change which model wins.

**Candidates & exclusion.** Every model ranks the same catalog — items seen
at least once in the data available at that point (train only while tuning;
train+validation for the final test run) — and has each user's own
already-interacted items (any rating) removed before ranking. Items that
only appear later are excluded from every model's candidate set alike
("item cold start", see Limitations).

**Model selection discipline.** All hyperparameters, the content-based
history threshold, and the hybrid weight are chosen on **validation only**,
maximizing NDCG@10 (`notebooks/02_evaluation_setup.ipynb`). The **test set
is touched exactly once**, in `notebooks/03_models.ipynb`, to report the
final numbers below.

## Models Compared

| Model | What it does |
|---|---|
| **Popularity** | Ranks by raw train interaction count. Non-personalized. |
| **Item-kNN** | Cosine similarity between movies' train rating vectors; scores = rating-weighted similarity to a user's own history. |
| **Content-Based** | Cosine similarity over genre + normalized release-year vectors; user profile = train movies rated ≥4 (threshold chosen on validation). |
| **ALS** | `implicit` Alternating Least Squares on a confidence-weighted implicit matrix (`1 + 2·(rating-1)`), correctly oriented `(users, items)` — see the orientation-bug writeup in `02`. |
| **BPR** | `implicit` Bayesian Personalized Ranking — pairwise ranking loss on the same confidence matrix; stands in for LightFM (see Limitations). |
| **Hybrid** | Content-Based + Item-kNN scores, each min-max normalized per user, blended `0.1·content + 0.9·Item-kNN` (weight and partner model chosen on validation, `02`). |

## Final Test Results (K = 10)

| Model | Recall@10 | Precision@10 | MAP@10 | NDCG@10 | Coverage@10 | Novelty@10 | Diversity@10 | Avg. Popularity@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Popularity | 0.048 | 0.062 | 0.035 | 0.074 | 0.036 | 8.50 | 0.645 | 2379 |
| **Item-kNN** | 0.082 | 0.079 | 0.051 | **0.104** | 0.127 | 8.94 | 0.635 | 1847 |
| Content-Based | 0.017 | 0.013 | 0.007 | 0.017 | **0.832** | **13.14** | 0.085 | **259** |
| ALS | 0.081 | 0.068 | 0.043 | 0.091 | 0.355 | 9.67 | 0.602 | 1237 |
| BPR | 0.064 | 0.048 | 0.030 | 0.066 | 0.761 | 11.13 | 0.485 | 620 |
| **Hybrid** (Content + Item-kNN) | **0.083** | 0.078 | 0.050 | 0.103 | 0.144 | 9.04 | 0.581 | 1734 |
| **Routing strategy** (Section below) | **0.088** | **0.081** | **0.052** | **0.107** | 0.214 | 9.11 | 0.618 | 1682 |

*(Coverage/Novelty/Diversity/Avg. Popularity are defined in
`02_evaluation_setup.ipynb`; full numbers in `reports/tables/test_metrics.csv`.)*

![Model comparison](reports/figures/model_comparison.png)

**Reading it:** Item-kNN and the Hybrid are effectively tied for best
ranking accuracy; ALS is a close third and is a legitimate model once its
orientation bug is fixed; Popularity is a real baseline that only
personalized models beat; Content-Based is the weakest on accuracy but the
strongest on catalog reach; BPR (the LightFM substitute) is the weakest
collaborative model on this dataset.

## Relevance vs. Coverage/Diversity Trade-offs

![Accuracy vs coverage trade-off](reports/figures/accuracy_vs_coverage_tradeoff.png)

Two trade-offs came out of the numbers (not assumed beforehand):

1. **Accuracy vs. catalog coverage.** The two most accurate models
   (Item-kNN, Hybrid) show the *least* catalog coverage among personalized
   models (13–14%); Content-Based has the *most* coverage (83%) and the
   worst accuracy. There is no model in this comparison that wins on both.
2. **Catalog coverage vs. within-list diversity are not the same thing.**
   Content-Based has the highest coverage **and** the lowest intra-list
   diversity (0.085): it reaches many different corners of the catalog
   *across users*, but for any one user it returns a genre-homogeneous
   cluster. Popularity is the mirror image: near-zero coverage (the same
   handful of movies for everyone) but comparatively high intra-list
   diversity (0.645), because the most popular movies overall span many
   genres. A model can score well on one "diversity-shaped" metric and
   poorly on another — they need to be checked separately.

## Results by User-History Segment

Users are split into quartiles by how many interactions they have at
prediction time (train count for validation-phase numbers, train+val for
test-phase numbers) — MovieLens-1M has no true zero-history users, so this
is a *relative* notion of "how little/much do we know about this person",
not literal cold start (see Limitations).

![Segment heatmap](reports/figures/segment_heatmap.png)

| Segment | Best model (test NDCG@10) | Notes |
|---|---|---|
| Sparse (Q1, ≤37 interactions) | **ALS** (0.101) | Factorization generalizes better than similarity counts with little history. |
| Low (Q2) | **Hybrid** (0.086), Item-kNN essentially tied (0.086) | |
| Medium (Q3) | **Item-kNN** (0.082) | |
| High (Q4, power users) | **Item-kNN** (0.156) | Popularity is a strong second here (0.142) — large rated histories make even a non-personalized list likely to overlap. |

**The common hypothesis "content-based helps most for low-history users" is
not supported by this data** — Content-Based is the weakest model in *every*
segment, including the sparsest one (NDCG@10 = 0.018 there, vs. 0.101 for
ALS). Genre similarity alone does not reliably predict which specific movie
a given user will rate highly next, regardless of how little history they have.

### Routing strategy

A router picks, per segment, the model with the best **validation** NDCG@10
(sparse→ALS, low→Hybrid, medium/high→Item-kNN), then is evaluated **once**
on test using each user's live segment. It beats every single model on
every offline metric (table above) and nearly doubles catalog coverage vs.
Item-kNN alone (21% vs. 13%). The margin over Item-kNN alone is real but
modest (NDCG@10 0.107 vs 0.104) — a good A/B test candidate, not proof of a
large production win. Full derivation: `notebooks/04_product_analysis.ipynb`.

## Cold Start

- **User cold start** cannot be observed in MovieLens-1M — every user has
  ≥20 ratings by construction of the dataset. In a real product, a
  brand-new user has none of that, and Item-kNN/ALS/BPR/Hybrid all require
  ≥1 training interaction to say anything. Popularity and an onboarding
  content-preference flow (which lets Content-Based work from interaction
  zero) are the only usable strategies at true zero history.
- **Item cold start is real and measured**: ~7.8% of rated movies have
  fewer than 5 ratings, and 177 catalog movies have never been rated. Every
  collaborative/factorization model here is structurally unable to
  recommend such an item (the candidate catalog is restricted to
  train-observed items for exactly this reason); Content-Based is the only
  model that can score a brand-new item from its metadata alone.

## Product Decision

1. **Ship Item-kNN as the single-model default.** It is strongest or tied
   strongest in 3 of 4 segments, simplest to explain, and cheapest to
   maintain among the competitive options.
2. **Treat the segment router as an A/B test candidate, not a day-one
   requirement.** It beat every single model offline, but the gain over
   Item-kNN alone is modest; the operational cost of running four models
   plus a routing layer should be justified online, not assumed from an
   offline delta.
3. **Keep Popularity as a zero-data fallback**, not as a segment winner —
   it never wins a segment outright but is the only strategy that needs no
   trained model and no user history at all.
4. **The default optimizes accuracy over exploration.** Item-kNN has the
   lowest catalog coverage of any personalized model (13%) and leans on
   already-popular items (avg. recommended popularity ≈ 1,847 vs.
   Content-Based's 259). If catalog health / long-tail exposure matters as
   a product goal, blending in 1–2 higher-novelty slots is a deliberate,
   measurable trade a product team could choose — this project does not
   have the data to say users would tolerate it.
5. **None of this is a causal engagement claim.** Offline Recall/NDCG
   measure whether a model would have ranked a movie the user *happened* to
   rate later near the top — not whether showing it changes behavior. That
   is exactly what the A/B test below is for.

## Online A/B-Test Design

A **design**, not a simulated result — MovieLens has no CTR/watch-time data,
so the sample-size table below is an explicit sensitivity analysis, not a forecast.

- **Hypothesis:** routing users to a segment-appropriate model increases
  engagement with recommendations vs. today's single-model policy, without
  hurting latency or over-concentrating recommendations.
- **Randomization unit:** user.
- **Control:** current single-model policy (e.g., Item-kNN for everyone).
- **Treatment:** the segment router (Section above), segment assigned from
  each user's live interaction count.
- **Primary metric:** recommendation click-through rate (clicks or
  play-starts ÷ impressions).
- **Secondary metrics:** play-through/completion rate of clicked
  recommendations, watch time from recommended titles, save/add-to-list rate.
- **Guardrails:** serving latency (p50/p95), catalog concentration of what
  is actually shown, overall session length/bounce rate.

**Sample size sensitivity** (two-proportion z-test, α=0.05, power=80%, per arm):

| Baseline CTR | +5% relative | +10% relative | +20% relative |
|---:|---:|---:|---:|
| 5% | 122,124 | 31,234 | 8,158 |
| 10% | 57,763 | 14,751 | 3,841 |
| 15% | 36,310 | 9,257 | 2,402 |

Real required sample size depends entirely on the platform's actual
baseline CTR and traffic — plug that in before committing to a test
duration. Suggested rollout: 5–10% of traffic, evenly split, for at least
one full weekly cycle (activity has clear day-of-week/hour-of-day patterns,
`01_data_and_eda.ipynb`), reading the primary metric only at the
pre-registered sample size.

Full derivation: `notebooks/04_product_analysis.ipynb`, Section 7.

## Limitations

- No true user cold start is observable in this dataset (see Cold Start).
- Item features are genres + release year only; richer metadata (cast,
  synopsis embeddings, tags) was out of scope and would likely help
  Content-Based and the Hybrid specifically.
- The underlying signal is explicit 1–5 ratings, not implicit behavior
  (clicks/plays/skips); the `rating >= 4` relevance rule and the
  confidence-weighting used for ALS/BPR are reasonable adaptations, not a
  substitute for validating against real implicit feedback.
- LightFM was not used — its build fails on this environment (Windows,
  Python 3.12, unmaintained since 2020); BPR is a reasonable but not
  identical substitute, and does not consume genre features directly.
- Offline ranking metrics are not a causal engagement estimate (see Product Decision, point 5).
- The routing strategy's offline margin over Item-kNN alone is small — a
  legitimate, validation-selected result, but not a large effect.

## Repository Structure

```text
rec-sys/
├── data/                          # raw MovieLens-1M .dat files
├── notebooks/
│   ├── 01_data_and_eda.ipynb          # data facts behind every methodology choice
│   ├── 02_evaluation_setup.ipynb      # split, relevance, metrics, validation-only tuning (incl. ALS bug repro)
│   ├── 03_models.ipynb                # one-time final test evaluation + trade-off plots
│   └── 04_product_analysis.ipynb      # segments, routing strategy, cold start, product decision, A/B design
├── src/
│   ├── data.py                    # loading, temporal split, relevance/seen/history builders
│   ├── metrics.py                 # Recall/Precision/MAP/NDCG + coverage/novelty/diversity/popularity
│   ├── recommenders.py            # Popularity, Item-kNN, Content-Based, ALS/BPR wrapper, Hybrid
│   ├── evaluation.py               # shared ranking/exclusion/segment/routing logic
│   └── pipeline.py                # orchestration + validation-selected hyperparameters
├── reports/
│   ├── figures/                   # saved plots used above
│   └── tables/                    # saved result tables (csv/json)
├── requirements.txt
└── README.md
```

## How to Run

```bash
git clone https://github.com/avgrosheva/rec-sys.git
cd rec-sys
python -m venv .venv
source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install -r requirements.txt
jupyter notebook
```

Run the notebooks in order — `01` → `02` → `03` → `04`. Each notebook
re-loads and re-splits the data itself (no hidden state between notebooks),
so they can also be reproduced headlessly:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_and_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_evaluation_setup.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_models.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/04_product_analysis.ipynb
```

Full run time is a few minutes on a laptop CPU.
