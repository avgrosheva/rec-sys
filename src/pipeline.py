"""Orchestration used by both notebooks 03 and 04, so every notebook that
needs fitted models rebuilds them the same way from the same functions
instead of depending on saved notebook state.

Hyperparameters below (BEST_PARAMS) are the outcome of the validation-only
grid search performed in notebooks/02_evaluation_setup.ipynb. They are
hard-coded here to keep 03/04 fast to re-run; the search itself, with its
full results table, lives in 02.
"""
import numpy as np
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking

from src import data as D
from src import evaluation as E
from src.recommenders import (
    PopularityRecommender, ItemKNNRecommender, ContentBasedRecommender,
    FactorModelRecommender, HybridRecommender,
)

K = 10
CONFIDENCE_ALPHA = 2.0

BEST_PARAMS = {
    "content_min_rating": 4,
    "als_factors": 32,
    "als_regularization": 0.01,
    "als_iterations": 20,
    "bpr_factors": 128,
    "bpr_iterations": 100,
    "hybrid_partner": "Item-kNN",
    "hybrid_alpha": 0.1,
}


def load_and_split(data_dir):
    users, movies, ratings = D.load_raw(data_dir)
    ratings, user_id_to_idx, item_id_to_idx = D.add_index_columns(ratings)
    df = D.temporal_split(ratings)
    return users, movies, df, user_id_to_idx, item_id_to_idx


def fit_all_models(df, movies, item_id_to_idx, train_splits, params=BEST_PARAMS, random_state=42):
    """Fit every model on the given train split(s) and return everything
    downstream code needs (models, catalog, exclusion sets, features)."""
    n_users = df["user_idx"].nunique()
    catalog_items = D.catalog_from_split(df, train_splits)
    item_pos = E.item_position_map(catalog_items)
    train_df = df[df["split"].isin(train_splits) & df["item_idx"].isin(item_pos)]
    pop_count = E.item_pop_count(df, train_splits)
    feature_matrix, genre_names = D.build_movie_features(movies, catalog_items, item_id_to_idx)
    conf_matrix = E.build_confidence_matrix(df, train_splits, n_users, catalog_items, item_pos, CONFIDENCE_ALPHA)

    pop_model = PopularityRecommender().fit(catalog_items, pop_count)
    knn_model = ItemKNNRecommender().fit(train_df, catalog_items, item_pos)
    cb_model = ContentBasedRecommender().fit(
        train_df, catalog_items, item_pos, feature_matrix, params["content_min_rating"], pop_model.scores
    )
    als_model = FactorModelRecommender(
        AlternatingLeastSquares(
            factors=params["als_factors"], regularization=params["als_regularization"],
            iterations=params["als_iterations"], random_state=random_state,
        )
    ).fit(conf_matrix)
    bpr_model = FactorModelRecommender(
        BayesianPersonalizedRanking(
            factors=params["bpr_factors"], iterations=params["bpr_iterations"], random_state=random_state,
        )
    ).fit(conf_matrix)

    partner = {"Item-kNN": knn_model, "ALS": als_model, "BPR": bpr_model}[params["hybrid_partner"]]
    hybrid_model = HybridRecommender(cb_model, partner, params["hybrid_alpha"])

    models = {
        "Popularity": pop_model,
        "Item-kNN": knn_model,
        "Content-Based": cb_model,
        "ALS": als_model,
        "BPR": bpr_model,
        "Hybrid": hybrid_model,
    }
    return {
        "models": models,
        "catalog_items": catalog_items,
        "item_pos": item_pos,
        "pop_count": pop_count,
        "feature_matrix": feature_matrix,
        "genre_names": genre_names,
        "n_train_interactions": len(train_df),
    }


def recommend_all(fitted, users, seen, k=K):
    return {
        name: E.generate_rec_lists(model, users, seen, fitted["catalog_items"], fitted["item_pos"], k)
        for name, model in fitted["models"].items()
    }


def evaluate_all(rec_lists_by_model, truth, fitted, k=K):
    rows = []
    for name, rec_lists in rec_lists_by_model.items():
        m = E.full_evaluation(
            rec_lists, truth, fitted["catalog_items"], fitted["pop_count"],
            fitted["n_train_interactions"], fitted["feature_matrix"], fitted["item_pos"], k,
        )
        rows.append({"model": name, **m})
    return rows
