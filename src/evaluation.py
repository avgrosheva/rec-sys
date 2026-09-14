"""Shared evaluation machinery applied identically to every model: same
candidate universe (the train-observed catalog), same K, same already-seen
exclusion rule, same relevance definition."""
import numpy as np
from scipy.sparse import csr_matrix

from src import metrics as M
from src.recommenders import rank_topk


def item_position_map(catalog_items):
    return {item: pos for pos, item in enumerate(catalog_items)}


def item_pop_count(df, split_names):
    sub = df[df["split"].isin(split_names)]
    return sub.groupby("item_idx").size().to_dict()


def build_confidence_matrix(df, split_names, n_users, catalog_items, item_pos, alpha=2.0):
    """(n_users, n_catalog) confidence matrix for implicit-feedback models.

    confidence = 1 + alpha * (rating - 1), following Hu et al. (2008); every
    interaction counts as an implicit positive signal regardless of its
    rating value (the model observes "the user watched/rated this"), while
    the *evaluation* relevance definition (rating >= 4) is applied
    separately and only when scoring ranking quality.
    """
    sub = df[df["split"].isin(split_names)]
    sub = sub[sub["item_idx"].isin(item_pos)]
    rows = sub["user_idx"].values
    cols = np.array([item_pos[i] for i in sub["item_idx"].values])
    vals = 1.0 + alpha * (sub["rating"].values.astype(float) - 1.0)
    return csr_matrix((vals, (rows, cols)), shape=(n_users, len(catalog_items)))


def generate_rec_lists(model, users, seen, catalog_items, item_pos, k):
    rec_lists = {}
    for u in users:
        scores = np.asarray(model.score_all(u), dtype=float)
        exclude_items = seen.get(u, ())
        exclude_pos = {item_pos[i] for i in exclude_items if i in item_pos}
        top_pos = rank_topk(scores, exclude_pos, k)
        rec_lists[u] = catalog_items[top_pos].tolist()
    return rec_lists


def full_evaluation(rec_lists, truth, catalog_items, pop_count, n_train_interactions,
                     feature_matrix, item_pos, k):
    """Ranking metrics + beyond-accuracy metrics for one model's recommendations."""
    ranking = M.evaluate_recommendations(rec_lists, truth, k)
    ranking["Coverage@K"] = M.catalog_coverage_at_k(rec_lists, len(catalog_items))
    ranking["Novelty@K"] = M.novelty_at_k(rec_lists, pop_count, n_train_interactions)
    ranking["Diversity@K"] = M.intra_list_diversity_at_k(rec_lists, feature_matrix, item_pos)
    ranking["AvgPopularity@K"] = M.average_recommended_popularity(rec_lists, pop_count)
    return ranking


def segment_of_users(counts, edges, labels):
    """Map user_idx -> segment label based on an interaction-count dict and
    ascending cutoff edges, e.g. edges=[30, 66, 145] with 4 labels."""
    seg = {}
    for u, n in counts.items():
        i = 0
        while i < len(edges) and n > edges[i]:
            i += 1
        seg[u] = labels[i]
    return seg


def segment_metrics_table(rec_lists_by_model, truth, segment_of_user, k):
    """Long-format rows of {model, segment, n_users, Recall/MAP/NDCG@K}."""
    rows = []
    segments = sorted(set(segment_of_user.values()))
    for model_name, rec_lists in rec_lists_by_model.items():
        for seg in segments:
            seg_users = {u for u, s in segment_of_user.items() if s == seg}
            seg_truth = {u: t for u, t in truth.items() if u in seg_users}
            if not seg_truth:
                continue
            m = M.evaluate_recommendations(rec_lists, seg_truth, k)
            rows.append({"model": model_name, "segment": seg, **m})
    return rows


def build_router(rec_lists_by_model, truth, segment_of_user, k):
    """Pick, per segment, the model with the highest NDCG@K (on this split)."""
    best_model = {}
    for seg in sorted(set(segment_of_user.values())):
        seg_users = {u for u, s in segment_of_user.items() if s == seg}
        seg_truth = {u: t for u, t in truth.items() if u in seg_users}
        if not seg_truth:
            continue
        scores = {
            name: M.evaluate_recommendations(rec_lists, seg_truth, k)["NDCG@K"]
            for name, rec_lists in rec_lists_by_model.items()
        }
        best_model[seg] = max(scores, key=scores.get)
    return best_model


def apply_router(router, rec_lists_by_model, segment_of_user, users):
    routed = {}
    for u in users:
        seg = segment_of_user.get(u)
        model_name = router.get(seg)
        if model_name is None:
            continue
        routed[u] = rec_lists_by_model[model_name].get(u, [])
    return routed
