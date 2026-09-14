"""Ranking and beyond-accuracy metrics, all evaluated at a fixed cutoff K.

Per-user metrics (recall / precision / average precision / ndcg) take an
already-ranked, already-seen-item-filtered list of recommended item ids and
a set of relevant ("ground truth") item ids.

Aggregate metrics (coverage / novelty / diversity / popularity) take the
full collection of per-user recommendation lists produced by a model.
"""
import numpy as np


def recall_at_k(pred, truth, k):
    if not truth:
        return 0.0
    pred_k = pred[:k]
    hits = len(set(pred_k) & truth)
    return hits / len(truth)


def precision_at_k(pred, truth, k):
    if not truth:
        return 0.0
    pred_k = pred[:k]
    hits = len(set(pred_k) & truth)
    return hits / k


def average_precision_at_k(pred, truth, k):
    if not truth:
        return 0.0
    pred_k = pred[:k]
    hits, score = 0, 0.0
    for i, item in enumerate(pred_k):
        if item in truth:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(truth), k)


def ndcg_at_k(pred, truth, k):
    if not truth:
        return 0.0
    pred_k = pred[:k]
    dcg = sum(1.0 / np.log2(i + 2) for i, item in enumerate(pred_k) if item in truth)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(truth), k)))
    return dcg / idcg if idcg > 0 else 0.0


def catalog_coverage_at_k(rec_lists, catalog_size):
    """Share of the catalog that ever appears in a top-K list across all users."""
    recommended = set()
    for items in rec_lists.values():
        recommended.update(items)
    return len(recommended) / catalog_size


def novelty_at_k(rec_lists, item_pop_count, n_train_interactions):
    """Mean self-information (bits) of recommended items: -log2(p(item)).

    Popular items have low self-information (low novelty); long-tail items
    have high self-information (high novelty). Averaged per user, then
    across users.
    """
    per_user = []
    for items in rec_lists.values():
        if not items:
            continue
        vals = []
        for item in items:
            p = item_pop_count.get(item, 0) / n_train_interactions
            p = max(p, 1e-12)
            vals.append(-np.log2(p))
        per_user.append(np.mean(vals))
    return float(np.mean(per_user)) if per_user else 0.0


def intra_list_diversity_at_k(rec_lists, item_feature_matrix, item_pos):
    """Mean pairwise (1 - cosine similarity) between items in each user's list.

    item_feature_matrix: (n_catalog, n_features) array.
    item_pos: dict item_idx -> row position in item_feature_matrix.
    Users with fewer than 2 recommended items are skipped.
    """
    norm = item_feature_matrix / (
        np.linalg.norm(item_feature_matrix, axis=1, keepdims=True) + 1e-12
    )
    per_user = []
    for items in rec_lists.values():
        positions = [item_pos[i] for i in items if i in item_pos]
        if len(positions) < 2:
            continue
        vecs = norm[positions]
        sim = vecs @ vecs.T
        n = len(positions)
        off_diag_sum = sim.sum() - np.trace(sim)
        mean_sim = off_diag_sum / (n * (n - 1))
        per_user.append(1.0 - mean_sim)
    return float(np.mean(per_user)) if per_user else 0.0


def average_recommended_popularity(rec_lists, item_pop_count):
    """Mean training-set popularity (interaction count) of recommended items.

    A simple, easy-to-explain popularity-bias indicator: higher means a
    model leans more heavily on already-popular items.
    """
    vals = []
    for items in rec_lists.values():
        for item in items:
            vals.append(item_pop_count.get(item, 0))
    return float(np.mean(vals)) if vals else 0.0


def evaluate_recommendations(rec_lists, truth, k):
    """Aggregate Recall/Precision/MAP/NDCG@k over users present in truth."""
    recalls, precisions, maps, ndcgs = [], [], [], []
    for user, truth_items in truth.items():
        pred = rec_lists.get(user, [])
        recalls.append(recall_at_k(pred, truth_items, k))
        precisions.append(precision_at_k(pred, truth_items, k))
        maps.append(average_precision_at_k(pred, truth_items, k))
        ndcgs.append(ndcg_at_k(pred, truth_items, k))
    return {
        "Recall@K": float(np.mean(recalls)),
        "Precision@K": float(np.mean(precisions)),
        "MAP@K": float(np.mean(maps)),
        "NDCG@K": float(np.mean(ndcgs)),
        "n_users": len(recalls),
    }
