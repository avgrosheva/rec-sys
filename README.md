# Movie Recommendation System

A recommendation system project comparing collaborative filtering, content-based, and hybrid approaches on the MovieLens 1M dataset.

The project focuses not only on building recommendation algorithms, but also on evaluating how different approaches behave under highly sparse user-item interaction data.

## Recommendation Problem

The goal is to generate personalized movie recommendations based on users' historical ratings and available movie information.

Three approaches are compared:

1. **Item-based collaborative filtering** using similarity between movies derived from user interactions.
2. **Content-based filtering** using movie genre information.
3. **Hybrid recommendation** combining collaborative and content-based signals.

The models are evaluated as ranking systems rather than rating predictors.

## Dataset

The project uses the **MovieLens 1M** dataset.

It contains approximately:

- 1 million ratings;
- 6,000 users;
- 4,000 movies;
- ratings on a 1–5 scale;
- movie titles and genres;
- timestamps for user interactions.

The interaction matrix is highly sparse — approximately **99.9% of possible user-item interactions are missing**.

This makes the dataset useful for exploring one of the central challenges in recommendation systems: generating relevant recommendations from limited interaction history.

## Data Analysis

Before modeling, the dataset was explored to understand:

- rating distribution;
- user activity;
- movie popularity;
- interaction sparsity;
- genre distribution;
- temporal structure of the ratings.

The analysis showed that interactions are strongly concentrated around a relatively small subset of popular movies, while the overall user-item matrix remains extremely sparse.

## Train / Test Strategy

A temporal split is used instead of randomly separating individual ratings.

Earlier user interactions are used for training, while later interactions are held out for evaluation.

This better reflects the real recommendation scenario:

> Given what was known about a user in the past, how well can the system recommend items they interact with later?

## Recommendation Approaches

### Item-kNN

The collaborative filtering model represents movies through user interactions and recommends items similar to those a user has previously rated.

Item similarity is calculated using cosine similarity.

This approach relies entirely on collaborative information and therefore depends heavily on sufficient overlap between user histories.

### Content-Based Filtering

The content-based model represents movies using their genres.

Recommendations are generated based on similarity between movie content and the user's historical preferences.

Unlike collaborative filtering, this approach does not require strong overlap between different users' interaction histories.

### Hybrid Model

The hybrid model combines collaborative and content-based recommendation scores.

The goal is to use both:

- behavioral information from user-item interactions;
- content information from movie genres.

The experiment tests whether combining the two signals improves ranking quality over either component independently.

## Evaluation

The models are evaluated using ranking metrics at `K = 10`.

**Recall@10**  
Measures how many relevant held-out movies appear among the top 10 recommendations.

**MAP@10**  
Measures both whether relevant items are retrieved and how highly they are ranked.

**NDCG@10**  
Evaluates ranking quality while giving greater weight to relevant items appearing near the top of the recommendation list.

These metrics are more appropriate for a recommendation task than standard classification or regression metrics because the product ultimately needs to produce a ranked list of items.

## Results

| Model | Recall@10 | MAP@10 | NDCG@10 |
|---|---:|---:|---:|
| Item-kNN | 0.00158 | 0.00138 | 0.00666 |
| **Content-Based** | **0.01512** | **0.01209** | **0.03107** |
| Hybrid | 0.01130 | 0.00888 | 0.02459 |

The **content-based model achieved the strongest performance across all three ranking metrics**.

The hybrid approach improved substantially over pure Item-kNN, but did not outperform the content-based model.

## Interpretation

The results illustrate an important property of recommendation systems: **a more complex model does not automatically produce better recommendations**.

The collaborative Item-kNN model performs poorly in this setup, which is consistent with the extremely sparse interaction matrix. With limited overlap between user histories, collaborative similarity provides a relatively weak recommendation signal.

The content-based model is less affected by interaction sparsity because it can use movie genre information directly.

Adding collaborative information to the content signal improves performance compared with Item-kNN alone, but the weak collaborative component does not provide enough additional information for the hybrid model to outperform content-based recommendations.

The experiment therefore suggests that, for this particular feature set and evaluation setup, **available content information provides a stronger signal than neighborhood-based collaborative filtering**.

## Product Perspective

Recommendation quality is not determined only by the algorithm used.

The experiment highlights several product and data considerations:

- the amount and quality of behavioral data directly affect collaborative approaches;
- sparse interactions can make similarity-based recommendations unreliable;
- useful item metadata can partially compensate for limited behavioral information;
- hybrid systems only improve performance when their additional signals contribute meaningful information;
- offline ranking metrics should be interpreted alongside the data available to the system.

In a real product, model selection would also depend on factors such as cold-start performance, recommendation diversity, novelty, latency, and online user behavior.

## Limitations

The project intentionally uses relatively simple recommendation approaches and a limited set of movie metadata.

The content model primarily relies on genres, which provide only a coarse representation of movie similarity.

The project also uses explicit ratings, while many production recommendation systems rely heavily on implicit feedback such as:

- clicks;
- views;
- watch time;
- saves;
- skips;
- repeat interactions.

Offline ranking metrics also do not directly measure whether users would actually engage with the recommendations in a live product.

## Possible Next Steps

Several extensions could improve the recommendation system:

- richer movie metadata such as descriptions, actors, directors, and tags;
- text embeddings for semantic movie representations;
- matrix factorization methods;
- implicit-feedback recommendation models;
- LightFM or similar hybrid approaches;
- neural collaborative filtering;
- learning-to-rank models;
- two-stage retrieval and ranking architecture;
- evaluation of diversity, novelty, and catalog coverage;
- online evaluation through A/B testing.

## Tech Stack

- Python
- pandas
- NumPy
- SciPy
- scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

## Repository Structure

```text
rec-sys/
├── data/
├── notebooks/
├── requirements.txt
└── README.md
```

## How to Run

Clone the repository:

```bash
git clone https://github.com/avgrosheva/rec-sys.git
cd rec-sys
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Open the project notebook in Jupyter and run the analysis from top to bottom to reproduce the data exploration, recommendation models, and evaluation.

## Key Takeaway

The strongest model in this experiment was not the most complex one.

Content-based filtering outperformed both Item-kNN and the hybrid approach, while the hybrid model still substantially improved over collaborative filtering alone.

The results demonstrate why recommendation systems should be selected based on **data characteristics and measured ranking performance rather than model complexity**.
