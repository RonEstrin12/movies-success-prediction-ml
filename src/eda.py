
"""
Exploratory Data Analysis (EDA) utilities for the Movies Success Prediction project.

These functions take a prepared DataFrame (after feature engineering)
and print useful statistics to understand the data before training models.
"""

from __future__ import annotations

import pandas as pd


def print_basic_info(df: pd.DataFrame) -> None:
    """
    Print basic information about the prepared movies dataset.
    """
    print("=== Basic Dataset Info ===")
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {df.shape[1]}")
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df.head())


def summarize_success_distribution(df: pd.DataFrame) -> None:
    """
    Print the distribution of the 'success' label.
    """
    print("\n=== Success Distribution ===")
    counts = df["success"].value_counts().sort_index()
    total = len(df)
    for label, count in counts.items():
        ratio = count / total
        print(f"success = {label}: {count} movies ({ratio:.2%})")
    print(f"\nOverall success rate: {df['success'].mean():.2%}")


def summarize_financials(df: pd.DataFrame) -> None:
    """
    Print descriptive statistics for budget, revenue and ROI.
    """
    print("\n=== Financial Summary ===")

    for col in ["budget", "revenue", "profit", "roi"]:
        if col not in df.columns:
            continue
        print(f"\n--- {col.upper()} ---")
        desc = df[col].describe(percentiles=[0.25, 0.5, 0.75])
        print(desc)


def summarize_by_year(df: pd.DataFrame) -> None:
    """
    Show how many movies we have per release year.
    Assumes 'release_year' column exists.
    """
    if "release_year" not in df.columns:
        print("release_year column not found.")
        return

    print("\n=== Movies per Year ===")
    counts = df["release_year"].value_counts().sort_index()
    for year, c in counts.items():
        print(f"{int(year)}: {c} movies")


def summarize_by_decade(df: pd.DataFrame) -> None:
    """
    Show how many movies we have per decade.
    Assumes 'release_year' column exists.
    """
    if "release_year" not in df.columns:
        print("release_year column not found.")
        return

    print("\n=== Movies per Decade ===")
    decade = (df["release_year"] // 10) * 10
    counts = decade.value_counts().sort_index()
    for d, c in counts.items():
        print(f"{int(d)}s: {c} movies")
