"""Data loading, indexing and temporal train/validation/test split for MovieLens-1M.

Design decisions (see README.md "Evaluation setup" for the full rationale):

- The split is chronological and computed independently per user: a user's
  earliest interactions go to train, the next block to validation, and the
  most recent block to test. This mirrors the real recommendation setting
  ("given the past, predict the future") instead of a random split, which
  would leak future behaviour into training.
- A rating is treated as a "relevant" (positive) interaction for ranking
  evaluation only if rating >= RELEVANT_RATING_THRESHOLD. Every interaction
  (regardless of rating) still counts as "seen" and is excluded from a
  user's candidate list, because a user cannot be re-shown something they
  already rated, liked or not.
- The candidate/catalog universe for every model is restricted to items
  that appear at least once in the TRAIN split. Items that only appear in
  validation/test are unseen ("item cold start") and are excluded from
  every model's candidate set alike, so no model is unfairly penalised or
  helped by items nobody could have recommended in the first place.
"""
import re

import numpy as np
import pandas as pd

RELEVANT_RATING_THRESHOLD = 4
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15


def load_raw(data_dir="../data"):
    users = pd.read_csv(
        f"{data_dir}/users.dat", sep="::", engine="python", encoding="latin-1",
        names=["user_id", "gender", "age", "occupation", "zip_code"],
    )
    movies = pd.read_csv(
        f"{data_dir}/movies.dat", sep="::", engine="python", encoding="latin-1",
        names=["movie_id", "title", "genres"],
    )
    ratings = pd.read_csv(
        f"{data_dir}/ratings.dat", sep="::", engine="python", encoding="latin-1",
        names=["user_id", "movie_id", "rating", "timestamp"],
    )
    ratings["timestamp"] = pd.to_datetime(ratings["timestamp"], unit="s")
    return users, movies, ratings


def extract_year(title):
    m = re.search(r"\((\d{4})\)", str(title))
    return int(m.group(1)) if m else np.nan


def add_index_columns(ratings):
    """Add user_idx / item_idx integer positions (0-based, dense) to ratings."""
    user_ids = np.sort(ratings["user_id"].unique())
    item_ids = np.sort(ratings["movie_id"].unique())
    user_id_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    item_id_to_idx = {iid: i for i, iid in enumerate(item_ids)}
    ratings = ratings.copy()
    ratings["user_idx"] = ratings["user_id"].map(user_id_to_idx)
    ratings["item_idx"] = ratings["movie_id"].map(item_id_to_idx)
    return ratings, user_id_to_idx, item_id_to_idx


def _split_sizes(n, train_frac, val_frac):
    n_test = max(1, round(n * (1 - train_frac - val_frac)))
    n_val = max(1, round(n * val_frac))
    n_train = n - n_val - n_test
    if n_train < 1:
        n_train = 1
        remaining = n - n_train
        n_val = remaining // 2
        n_test = remaining - n_val
    return n_train, n_val, n_test


def temporal_split(ratings, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC):
    """Chronological per-user split into contiguous train / val / test blocks.

    Requires ratings to already have a user_idx column. Returns a copy of
    ratings with a new "split" column. Because MovieLens-1M guarantees at
    least 20 ratings per user, every user contributes at least one
    interaction to each of the three blocks.
    """
    df = ratings.sort_values(["user_idx", "timestamp"]).reset_index(drop=True)
    splits = np.empty(len(df), dtype=object)
    for _, positions in df.groupby("user_idx").indices.items():
        n = len(positions)
        n_train, n_val, n_test = _split_sizes(n, train_frac, val_frac)
        labels = np.array(["train"] * n_train + ["val"] * n_val + ["test"] * n_test)
        splits[positions] = labels
    df["split"] = splits
    return df


def get_truth(df, split_name, threshold=RELEVANT_RATING_THRESHOLD):
    """user_idx -> set of item_idx rated >= threshold within a given split.

    Users with zero qualifying items in the split simply do not appear as
    keys; they form no evaluation signal for that split (see cohort size
    reporting in 02_evaluation_setup.ipynb).
    """
    sub = df[(df["split"] == split_name) & (df["rating"] >= threshold)]
    return sub.groupby("user_idx")["item_idx"].apply(set).to_dict()


def get_seen(df, split_names):
    """user_idx -> set of item_idx interacted with (any rating) in given splits."""
    sub = df[df["split"].isin(split_names)]
    return sub.groupby("user_idx")["item_idx"].apply(set).to_dict()


def get_history(df, split_names, min_rating=None):
    """user_idx -> list of (item_idx, rating) tuples across given splits.

    Used to build a user's taste profile for content-based / item-kNN
    scoring. If min_rating is set, only interactions at or above it are kept.
    """
    sub = df[df["split"].isin(split_names)]
    if min_rating is not None:
        sub = sub[sub["rating"] >= min_rating]
    out = {}
    for uid, g in sub.groupby("user_idx"):
        out[uid] = list(zip(g["item_idx"].values, g["rating"].values))
    return out


def build_movie_features(movies, catalog_item_idx, item_id_to_idx):
    """Genre multi-hot + normalised year matrix, ordered by catalog position.

    catalog_item_idx: array of item_idx values defining catalog order
    (position i in the returned matrix corresponds to catalog_item_idx[i]).
    """
    idx_to_item_id = {v: k for k, v in item_id_to_idx.items()}
    movie_ids_in_order = [idx_to_item_id[i] for i in catalog_item_idx]

    movies_indexed = movies.set_index("movie_id")
    genres_list = movies_indexed.loc[movie_ids_in_order, "genres"].str.split("|")
    all_genres = sorted({g for gs in genres_list for g in gs})
    genre_to_col = {g: i for i, g in enumerate(all_genres)}

    genre_mat = np.zeros((len(movie_ids_in_order), len(all_genres)), dtype=np.float32)
    for row, gs in enumerate(genres_list):
        for g in gs:
            genre_mat[row, genre_to_col[g]] = 1.0

    titles = movies_indexed.loc[movie_ids_in_order, "title"]
    years = titles.apply(extract_year).astype(float)
    years = years.fillna(years.median())
    year_norm = ((years - years.mean()) / years.std()).values.reshape(-1, 1).astype(np.float32)

    features = np.hstack([genre_mat, year_norm])
    return features, all_genres


def catalog_from_split(df, split_names):
    """Sorted array of item_idx observed in the given split(s) -> the candidate universe."""
    sub = df[df["split"].isin(split_names)]
    return np.sort(sub["item_idx"].unique())
