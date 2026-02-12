"""Train baseline housing price models and save the best pipeline artifact."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

# Optional external gradient boosting libraries.
try:
    from xgboost import XGBRegressor  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    LGBMRegressor = None

try:
    from catboost import CatBoostRegressor  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    CatBoostRegressor = None

RANDOM_STATE = 42
TARGET_COL = "price"

# High-cardinality text/id-like columns that make v1 harder to generalize.
EXCLUDE_COLS = {
    TARGET_COL,
    "Building Name",
    "Address",
    "Facilities",
    "Developer",
    "Railway Station",
    "Bus Stop",
    "School",
    "Nearby Mall",
}

DATA_PATH = Path("data/final/house_model_ready.csv")
MODEL_PATH = Path("models/best_model.joblib")
META_PATH = Path("models/model_metadata.json")
COMPARISON_PATH = Path("reports/model/model_comparison.csv")
TEST_METRICS_PATH = Path("reports/model/test_metrics.json")


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "median_abs_error": float(np.median(np.abs(y_true - y_pred))),
    }


def build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=10,
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor, numeric_cols, categorical_cols


def build_candidates(preprocessor: ColumnTransformer) -> dict[str, Pipeline]:
    candidates: dict[str, Pipeline] = {
        "dummy_median": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", DummyRegressor(strategy="median")),
            ]
        ),
        "linear_regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", LinearRegression()),
            ]
        ),
        "ridge": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "lasso": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", Lasso(alpha=0.001, max_iter=10000, random_state=RANDOM_STATE)),
            ]
        ),
        "elastic_net": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", ElasticNet(alpha=0.01, l1_ratio=0.2, max_iter=10000, random_state=RANDOM_STATE)),
            ]
        ),
        "decision_tree": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", DecisionTreeRegressor(max_depth=18, min_samples_leaf=3, random_state=RANDOM_STATE)),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=400,
                        min_samples_leaf=2,
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "extra_trees": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=600,
                        min_samples_leaf=2,
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "bagging": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    BaggingRegressor(
                        estimator=DecisionTreeRegressor(max_depth=18, random_state=RANDOM_STATE),
                        n_estimators=250,
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", GradientBoostingRegressor(random_state=RANDOM_STATE)),
            ]
        ),
        "adaboost": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    AdaBoostRegressor(
                        estimator=DecisionTreeRegressor(max_depth=4, random_state=RANDOM_STATE),
                        n_estimators=300,
                        learning_rate=0.03,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        random_state=RANDOM_STATE,
                        learning_rate=0.05,
                        max_depth=8,
                        max_iter=500,
                    ),
                ),
            ]
        ),
        "knn": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", KNeighborsRegressor(n_neighbors=15, weights="distance")),
            ]
        ),
        "svr_rbf": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", SVR(kernel="rbf", C=50.0, epsilon=0.05, gamma="scale")),
            ]
        ),
    }

    if XGBRegressor is not None:
        candidates["xgboost"] = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    XGBRegressor(
                        n_estimators=1000,
                        learning_rate=0.03,
                        max_depth=6,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_alpha=0.0,
                        reg_lambda=1.0,
                        objective="reg:squarederror",
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        )

    if LGBMRegressor is not None:
        candidates["lightgbm"] = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    LGBMRegressor(
                        n_estimators=1200,
                        learning_rate=0.03,
                        num_leaves=31,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        )

    if CatBoostRegressor is not None:
        candidates["catboost"] = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    CatBoostRegressor(
                        iterations=1200,
                        learning_rate=0.03,
                        depth=6,
                        loss_function="RMSE",
                        random_seed=RANDOM_STATE,
                        verbose=False,
                    ),
                ),
            ]
        )

    return candidates


def build_schema_defaults(X: pd.DataFrame) -> tuple[dict, dict, dict]:
    defaults: dict[str, object] = {}
    numeric_ranges: dict[str, dict[str, float]] = {}
    categorical_values: dict[str, list[str]] = {}

    for col in X.columns:
        series = X[col]
        if pd.api.types.is_numeric_dtype(series):
            defaults[col] = float(series.median()) if series.notna().any() else 0.0
            numeric_ranges[col] = {
                "min": float(series.min()) if series.notna().any() else 0.0,
                "max": float(series.max()) if series.notna().any() else 1.0,
            }
        else:
            mode = series.mode(dropna=True)
            defaults[col] = str(mode.iloc[0]) if not mode.empty else "Unknown"
            vals = series.dropna().astype(str).value_counts().head(25).index.tolist()
            categorical_values[col] = vals

    return defaults, numeric_ranges, categorical_values


def find_constant_columns(X: pd.DataFrame) -> list[str]:
    """Return columns with <=1 unique non-null value (no predictive signal)."""
    constant_cols: list[str] = []
    for col in X.columns:
        unique_count = X[col].nunique(dropna=True)
        if unique_count <= 1:
            constant_cols.append(col)
    return constant_cols


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in {DATA_PATH}")

    df = df.copy()
    df = df[df[TARGET_COL] > 0].reset_index(drop=True)

    initial_feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X = df[initial_feature_cols].copy()
    dropped_constant_cols = find_constant_columns(X)
    if dropped_constant_cols:
        X = X.drop(columns=dropped_constant_cols)

    feature_cols = X.columns.tolist()
    if not feature_cols:
        raise ValueError("No usable features remain after dropping constant columns.")

    y = df[TARGET_COL].astype(float).to_numpy()
    y_log = np.log1p(y)

    X_trainval, X_test, y_trainval_log, _y_test_log, y_trainval_raw, y_test_raw = train_test_split(
        X, y_log, y, test_size=0.15, random_state=RANDOM_STATE
    )

    X_train, X_valid, y_train_log, y_valid_log, y_train_raw, y_valid_raw = train_test_split(
        X_trainval,
        y_trainval_log,
        y_trainval_raw,
        test_size=0.1764705882,
        random_state=RANDOM_STATE,
    )

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X_train)
    candidates = build_candidates(preprocessor)
    print(f"Candidate models: {', '.join(candidates.keys())}")

    comparison_rows: list[dict[str, float | str]] = []
    best_name = ""
    best_valid_rmse = float("inf")
    best_pipeline: Pipeline | None = None

    for name, pipeline in candidates.items():
        pipeline.fit(X_train, y_train_log)
        valid_pred_raw = np.expm1(pipeline.predict(X_valid))
        valid_metrics = evaluate_predictions(y_valid_raw, valid_pred_raw)

        row = {
            "model": name,
            "valid_rmse": valid_metrics["rmse"],
            "valid_mae": valid_metrics["mae"],
            "valid_r2": valid_metrics["r2"],
            "valid_median_abs_error": valid_metrics["median_abs_error"],
        }
        comparison_rows.append(row)

        if valid_metrics["rmse"] < best_valid_rmse:
            best_valid_rmse = valid_metrics["rmse"]
            best_name = name
            best_pipeline = pipeline

    assert best_pipeline is not None

    # Refit best model on train+valid for final test evaluation.
    best_pipeline.fit(X_trainval, y_trainval_log)
    test_pred_raw = np.expm1(best_pipeline.predict(X_test))
    test_metrics = evaluate_predictions(y_test_raw, test_pred_raw)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEST_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_pipeline, MODEL_PATH)

    comparison_df = pd.DataFrame(comparison_rows).sort_values("valid_rmse", ascending=True)
    comparison_df.to_csv(COMPARISON_PATH, index=False)

    defaults, numeric_ranges, categorical_values = build_schema_defaults(X_trainval)

    metadata = {
        "target": TARGET_COL,
        "transform": "log1p",
        "best_model": best_name,
        "train_rows": int(len(X_train)),
        "valid_rows": int(len(X_valid)),
        "test_rows": int(len(X_test)),
        "feature_columns": feature_cols,
        "dropped_constant_columns": dropped_constant_cols,
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "excluded_columns": sorted(list(EXCLUDE_COLS)),
        "defaults": defaults,
        "numeric_ranges": numeric_ranges,
        "categorical_values": categorical_values,
        "validation_metrics": {
            "rmse": float(best_valid_rmse),
        },
        "test_metrics": test_metrics,
    }

    with META_PATH.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with TEST_METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    print("Training complete")
    print(f"Dropped constant columns: {dropped_constant_cols if dropped_constant_cols else 'None'}")
    print(f"Best model: {best_name}")
    print(f"Validation RMSE: {best_valid_rmse:,.2f}")
    print(f"Test RMSE: {test_metrics['rmse']:,.2f}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metadata: {META_PATH}")
    print(f"Saved comparison: {COMPARISON_PATH}")
    print(f"Saved test metrics: {TEST_METRICS_PATH}")


if __name__ == "__main__":
    main()
