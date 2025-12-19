
"""
Main entry point for the Movies Success Prediction project.

Steps:
1. Load and prepare the movies dataset.
2. Run basic EDA (optional).
3. Train Decision Tree, Random Forest and AdaBoost models.
4. Evaluate all models on the test set.
"""

from __future__ import annotations

from src.data_preparation import load_and_prepare_movies_dataset
from src import eda
from src import models


def run_eda(df):
    """
    Run a basic EDA pipeline on the prepared DataFrame.
    """
    eda.print_basic_info(df)
    eda.summarize_success_distribution(df)
    eda.summarize_financials(df)
    eda.summarize_by_year(df)
    eda.summarize_by_decade(df)


def run_training(df):
    """
    Train three models (Decision Tree, Random Forest, AdaBoost)
    and evaluate them on a held-out test set.
    """
    X_train, X_test, y_train, y_test = models.train_test_split_movies(df)

    # Decision Tree
    dt_clf = models.train_decision_tree(
        X_train,
        y_train,
        max_depth=8,
        min_samples_leaf=10,
    )
    models.evaluate_model(dt_clf, X_test, y_test, model_name="Decision Tree")

    # Random Forest
    rf_clf = models.train_random_forest(
        X_train,
        y_train,
        n_estimators=200,
        max_depth=12,
    )
    models.evaluate_model(rf_clf, X_test, y_test, model_name="Random Forest")

    # AdaBoost
    ab_clf = models.train_adaboost(
        X_train,
        y_train,
        n_estimators=200,
        learning_rate=0.3,
    )
    models.evaluate_model(ab_clf, X_test, y_test, model_name="AdaBoost")


if __name__ == "__main__":
    # 1. Load and prepare dataset
    df_model = load_and_prepare_movies_dataset()

    # 2. Run EDA (you can comment this out if you only want training)
    run_eda(df_model)

    # 3. Train and evaluate models
    run_training(df_model)
