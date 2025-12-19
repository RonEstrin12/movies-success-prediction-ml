
"""
Model training and evaluation utilities for the Movies Success Prediction project.

This module provides:
- Functions to split the data into train/test sets.
- Functions to train Decision Tree, Random Forest and AdaBoost models.
- Functions to evaluate models using accuracy and classification report.
"""

from __future__ import annotations

from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from .config import RANDOM_SEED


def train_test_split_movies(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the prepared movies dataset into train and test sets.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    feature_cols = [c for c in df.columns if c != "success"]
    X = df[feature_cols].values
    y = df["success"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def train_decision_tree(
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    random_state: int = RANDOM_SEED,
) -> DecisionTreeClassifier:
    """
    Train a Decision Tree classifier on the given training data.
    """
    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    clf.fit(X_train, y_train)
    return clf


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
    max_depth: int | None = None,
    random_state: int = RANDOM_SEED,
) -> RandomForestClassifier:
    """
    Train a Random Forest classifier on the given training data.
    """
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf


def train_adaboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
    learning_rate: float = 0.5,
    random_state: int = RANDOM_SEED,
) -> AdaBoostClassifier:
    """
    Train an AdaBoost classifier on the given training data.
    By default it uses a DecisionTreeClassifier as the base estimator.
    """
    clf = AdaBoostClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state,
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "model",
) -> Dict[str, Any]:
    """
    Evaluate a trained model on the test set and print metrics.

    Returns a dictionary with accuracy and the raw classification report string.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=3)

    print(f"\n=== Evaluation for {model_name} ===")
    print(f"Accuracy: {acc:.4f}")
    print("Classification report:")
    print(report)

    return {
        "model_name": model_name,
        "accuracy": acc,
        "report": report,
    }
