
"""
Data loading and feature engineering for the Movies Success Prediction project.

This module:
- Loads the movies_metadata.csv file.
- Cleans and converts numeric columns.
- Creates financial features (profit, ROI, success label).
- Extracts features from genres, language and release date.
"""

from __future__ import annotations

import json
from typing import Tuple, List

import numpy as np
import pandas as pd

from .config import DATA_PATH


def _to_numeric(series: pd.Series, default_value: float | None = None) -> pd.Series:
    """
    Safely convert a Pandas Series to numeric.
    Non-convertible values are set to default_value (or NaN if default_value is None).
    """
    numeric = pd.to_numeric(series, errors="coerce")
    if default_value is not None:
        numeric = numeric.fillna(default_value)
    return numeric


def _parse_release_date(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Parse release_date column to year and month.
    Returns two Series: (year, month).
    """
    dates = pd.to_datetime(series, errors="coerce")
    return dates.dt.year, dates.dt.month


def _add_financial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add profit, ROI and success label to the DataFrame.
    Assumes budget and revenue are already numeric.
    success = 1 if ROI > 0.5, else 0.
    """
    df = df.copy()

    df["profit"] = df["revenue"] - df["budget"]

    # Avoid division by zero
    df["roi"] = np.where(
        df["budget"] > 0,
        (df["revenue"] - df["budget"]) / df["budget"],
        np.nan,
    )

    # Define success: ROI > 0.5
    df["success"] = (df["roi"] > 0.5).astype(int)

    return df


def _extract_genres_list(genres_str: str) -> List[str]:
    """
    Convert the 'genres' JSON-like string into a list of genre names.
    If parsing fails, return an empty list.
    """
    if pd.isna(genres_str):
        return []
    try:
        genres = json.loads(genres_str)
        # Each item is expected to be a dict with a 'name' field
        return [g.get("name", "") for g in genres]
    except Exception:
        return []


def _add_genre_features(df: pd.DataFrame, top_k: int = 8) -> pd.DataFrame:
    """
    Add n_genres and one-hot columns for the top_k most frequent genres.
    """
    df = df.copy()
    df["genres_list"] = df["genres"].apply(_extract_genres_list)
    df["n_genres"] = df["genres_list"].apply(len)

    # Count genre frequencies
    all_genres = df["genres_list"].explode()
    top_genres = (
        all_genres.value_counts()
        .head(top_k)
        .index
        .tolist()
    )

    for g in top_genres:
        col_name = f"genre_{g.lower().replace(' ', '_')}"
        df[col_name] = df["genres_list"].apply(lambda lst, genre=g: int(genre in lst))

    return df


def _add_language_features(df: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    """
    Add one-hot features for the top_k most common original languages.
    All other languages go to 'lang_other'.
    """
    df = df.copy()
    lang_counts = df["original_language"].value_counts()
    top_langs = lang_counts.head(top_k).index.tolist()

    for lang in top_langs:
        df[f"lang_{lang}"] = (df["original_language"] == lang).astype(int)

    # 'Other' language indicator
    df["lang_other"] = (~df["original_language"].isin(top_langs)).astype(int)
    return df


def _add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add release_year and release_month features.
    """
    df = df.copy()
    year, month = _parse_release_date(df["release_date"])
    df["release_year"] = year
    df["release_month"] = month
    return df


def _select_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select only the columns that will be used as input features (X) for the models.
    """
    feature_cols = [
        # numeric features available BEFORE release
        "budget",
        "popularity",
        "runtime",
        "n_genres",
        "release_year",
        "release_month",
    ]

    # Add all genre_ one-hot columns
    genre_cols = [c for c in df.columns if c.startswith("genre_")]

    # Add all lang_ one-hot columns
    lang_cols = [c for c in df.columns if c.startswith("lang_")]

    all_features = feature_cols + genre_cols + lang_cols

    df_model = df.dropna(subset=all_features + ["success"]).copy()

    return df_model[all_features + ["success"]]




def load_and_prepare_movies_dataset(
    path: str | None = None,
    drop_missing_financials: bool = True,
) -> pd.DataFrame:
    """
    Load raw movies metadata and return a cleaned DataFrame
    with engineered features and a 'success' label.

    Parameters
    ----------
    path : optional path to the CSV file. If None, use config.DATA_PATH.
    drop_missing_financials : if True, drop rows with non-positive budget or revenue.
    """
    csv_path = DATA_PATH if path is None else path

    # 1. Load raw CSV
    df = pd.read_csv(csv_path, low_memory=False)

    # 2. Keep only relevant columns for now
    cols = [
        "id",
        "title",
        "budget",
        "revenue",
        "popularity",
        "runtime",
        "genres",
        "release_date",
        "original_language",
    ]
    df = df[cols].copy()

    # 3. Convert budget, revenue, popularity, runtime to numeric
    df["budget"] = _to_numeric(df["budget"], default_value=0.0)
    df["revenue"] = _to_numeric(df["revenue"], default_value=0.0)
    df["popularity"] = _to_numeric(df["popularity"])
    df["runtime"] = _to_numeric(df["runtime"])

    # 4. Remove rows with zero/very small budget or revenue if requested
    if drop_missing_financials:
        df = df[(df["budget"] > 0) & (df["revenue"] > 0)]

    # 5. Add financial features and success label
    df = _add_financial_features(df)

    # 6. Add genres-based features
    df = _add_genre_features(df, top_k=8)

    # 7. Add language and date features
    df = _add_language_features(df, top_k=5)
    df = _add_date_features(df)

    # 8. Prepare final modeling DataFrame
    df_model = _select_model_features(df)

    return df_model
