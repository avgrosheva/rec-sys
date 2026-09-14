"""Recommenders. Every model exposes the same interface:

    model.fit(...)
    model.score_all(user_idx) -> np.ndarray of shape (n_catalog,)

Scores are aligned to a shared `catalog_items` array (position i of the
score vector corresponds to item id `catalog_items[i]`). Keeping every
model's output in this common form lets a single ranking / exclusion /
evaluation routine (see src/evaluation.py) be applied identically to every
model: same candidate universe, same already-seen exclusion, same K.
"""
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


def rank_topk(scores, exclude_positions, k):
    """Return the top-k catalog positions, excluding given positions."""
    scores = scores.copy()
    if exclude_positions:
        idx = np.fromiter(exclude_positions, dtype=int, count=len(exclude_positions))
        idx = idx[idx < len(scores)]
        scores[idx] = -np.inf
    k = min(k, len(scores))
    top = np.argpartition(-scores, k - 1)[:k]
    return top[np.argsort(-scores[top])]


class PopularityRecommender:
    """Ranks items by raw train interaction count (most-rated first)."""

    def fit(self, catalog_items, pop_count):
        self.catalog_items = catalog_items
        self.scores = np.array([pop_count.get(i, 0) for i in catalog_items], dtype=float)
        return self

    def score_all(self, user_idx):
        return self.scores


class ItemKNNRecommender:
    """Item-based collaborative filtering: cosine similarity between items'
    train rating vectors, scored as the rating-weighted sum of similarity to
    a user's own train history."""

    def fit(self, train_df, catalog_items, item_pos):
        n_items = len(catalog_items)
        rows, cols, vals = [], [], []
        for user_idx, item_idx, rating in zip(
            train_df["user_idx"].values, train_df["item_idx"].values, train_df["rating"].values
        ):
            if item_idx in item_pos:
                rows.append(item_pos[item_idx])
                cols.append(user_idx)
                vals.append(float(rating))
        n_users = train_df["user_idx"].max() + 1
        item_user = csr_matrix((vals, (rows, cols)), shape=(n_items, n_users))
        self.sim = cosine_similarity(item_user, dense_output=True)
        np.fill_diagonal(self.sim, 0.0)

        self.user_history = {}
        for user_idx, g in train_df.groupby("user_idx"):
            pos_rating = [(item_pos[i], r) for i, r in zip(g["item_idx"], g["rating"]) if i in item_pos]
            if pos_rating:
                self.user_history[user_idx] = pos_rating
        self.catalog_items = catalog_items
        return self

    def score_all(self, user_idx):
        history = self.user_history.get(user_idx, [])
        if not history:
            return np.zeros(len(self.catalog_items))
        positions = np.array([p for p, _ in history])
        weights = np.array([r for _, r in history])
        return weights @ self.sim[positions]


class ContentBasedRecommender:
    """Cosine similarity over genre + year features, scored as the sum of
    similarity to items in a user's positive (rating >= min_rating) history."""

    def fit(self, train_df, catalog_items, item_pos, feature_matrix, min_rating, fallback_scores):
        self.catalog_items = catalog_items
        self.sim = cosine_similarity(feature_matrix)
        np.fill_diagonal(self.sim, 0.0)
        self.fallback_scores = fallback_scores
        self.min_rating = min_rating

        sub = train_df[train_df["rating"] >= min_rating] if min_rating is not None else train_df
        self.user_history = {}
        for user_idx, g in sub.groupby("user_idx"):
            positions = [item_pos[i] for i in g["item_idx"] if i in item_pos]
            if positions:
                self.user_history[user_idx] = positions
        return self

    def score_all(self, user_idx):
        positions = self.user_history.get(user_idx)
        if not positions:
            return self.fallback_scores.copy()
        return self.sim[positions].sum(axis=0)


class FactorModelRecommender:
    """Thin wrapper around an `implicit` matrix-factorization model
    (ALS or BPR), fit on a correctly-oriented (n_users, n_catalog) confidence
    matrix. Scores are the raw user-factor . item-factor dot product over the
    full catalog, so the same rank_topk / exclusion logic used everywhere
    else applies unchanged."""

    def __init__(self, model):
        self.model = model

    def fit(self, user_item_csr):
        self.model.fit(user_item_csr, show_progress=False)
        self.catalog_size = user_item_csr.shape[1]
        return self

    def score_all(self, user_idx):
        if user_idx >= self.model.user_factors.shape[0]:
            return np.zeros(self.catalog_size)
        uf = self.model.user_factors[user_idx]
        return np.asarray(self.model.item_factors @ uf).reshape(-1)


class HybridRecommender:
    """Weighted fusion of two score vectors after per-user min-max scaling,
    so neither model's raw score scale dominates the combination."""

    def __init__(self, model_a, model_b, alpha):
        self.model_a = model_a
        self.model_b = model_b
        self.alpha = alpha

    @staticmethod
    def _norm(scores):
        lo, hi = scores.min(), scores.max()
        if hi - lo < 1e-12:
            return np.zeros_like(scores)
        return (scores - lo) / (hi - lo)

    def score_all(self, user_idx):
        a = self._norm(self.model_a.score_all(user_idx))
        b = self._norm(self.model_b.score_all(user_idx))
        return self.alpha * a + (1 - self.alpha) * b
