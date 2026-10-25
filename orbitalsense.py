"""Reusable OrbitalSense ML pipeline.

The bundled .joblib files are used for fast prediction. Training is available
through train_models() and the command-line options in run_project.py.
"""
from pathlib import Path
from typing import Dict, Iterable, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

FEATURES = [
    "latency_ms", "packet_loss_rate", "signal_quality_score",
    "available_bandwidth_mbps", "congestion_level_pct",
    "remaining_contact_time_s", "relative_velocity_km_s",
    "representative_orbit_altitude_km", "inter_satellite_link_distance_km",
    "traffic_load_pct", "queue_delay_ms", "position_uncertainty_km",
    "link_available",
]
MODEL_FILES = {
    "bandwidth": "M1_RandomForest_Bandwidth.joblib",
    "quality": "M2_SVM_LinkQuality.joblib",
    "lifetime": "M3_ExtraTrees_Lifetime.joblib",
    "congestion": "M4_KNN_Congestion.joblib",
    "route": "OrbitalSense_Stacking.joblib",
}
DEFAULT_SAMPLE = {
    "latency_ms": 35.0, "packet_loss_rate": 0.01,
    "signal_quality_score": 85.0, "available_bandwidth_mbps": 120.0,
    "congestion_level_pct": 20.0, "remaining_contact_time_s": 180.0,
    "relative_velocity_km_s": 6.2, "representative_orbit_altitude_km": 15.0,
    "inter_satellite_link_distance_km": 900.0, "traffic_load_pct": 20.0,
    "queue_delay_ms": 18.0, "position_uncertainty_km": 4.0,
    "link_available": 1.0,
}


def project_dir() -> Path:
    return Path(__file__).resolve().parent


def make_preprocessor() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


def load_models(directory: Optional[Path] = None) -> Dict[str, object]:
    directory = Path(directory or project_dir())
    missing = [name for name in MODEL_FILES.values() if not (directory / name).exists()]
    if missing:
        raise FileNotFoundError("Missing model file(s): " + ", ".join(missing))
    return {key: joblib.load(directory / filename) for key, filename in MODEL_FILES.items()}


def validate_input(data: pd.DataFrame) -> pd.DataFrame:
    missing = [feature for feature in FEATURES if feature not in data.columns]
    if missing:
        raise ValueError("Input is missing required feature columns: " + ", ".join(missing))
    result = data.copy()
    for feature in FEATURES:
        result[feature] = pd.to_numeric(result[feature], errors="raise")
    return result


def predict_new_links(data: pd.DataFrame, models: Optional[Dict[str, object]] = None) -> pd.DataFrame:
    data = validate_input(data)
    models = models or load_models()
    x = data[FEATURES]
    predictions = pd.DataFrame({
        "predicted_future_bandwidth_mbps": models["bandwidth"].predict(x),
        "predicted_link_quality": models["quality"].predict(x),
        "predicted_remaining_lifetime_s": models["lifetime"].predict(x),
        "predicted_congestion_level": models["congestion"].predict(x),
        "predicted_route_performance": models["route"].predict(x),
    })
    return pd.concat([data.reset_index(drop=True), predictions], axis=1)


def train_models(data_path: Optional[Path] = None, output_dir: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
    """Train the five models and save models plus metric CSVs."""
    output_dir = Path(output_dir or project_dir())
    data_path = Path(data_path or output_dir / "OrbitalSense_AI_Kaggle_Style_Dataset.csv")
    df = pd.read_csv(data_path).drop_duplicates().reset_index(drop=True)
    x = df[FEATURES]
    y_bw = df["available_bandwidth_mbps"]
    y_quality = df["link_suitability"]
    y_lifetime = df["remaining_contact_time_s"]
    y_congestion = pd.cut(df["congestion_level_pct"], bins=[-np.inf, 33.33, 66.67, np.inf], labels=["Low", "Medium", "High"])
    y_route = df["link_success"]

    xtr, xte, ytr, yte = train_test_split(x, y_bw, test_size=.2, random_state=42)
    m1 = Pipeline([("preprocessing", make_preprocessor()), ("model", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))]).fit(xtr, ytr)
    p = m1.predict(xte)
    reg = [{"Model": "M1 Random Forest - Bandwidth", "MAE": mean_absolute_error(yte, p), "RMSE": mean_squared_error(yte, p) ** .5, "R2": r2_score(yte, p)}]

    xtr, xte, ytr, yte = train_test_split(x, y_quality, test_size=.2, random_state=42, stratify=y_quality)
    m2 = Pipeline([("preprocessing", make_preprocessor()), ("model", SVC(kernel="rbf", C=2, class_weight="balanced"))]).fit(xtr, ytr)
    pred = m2.predict(xte)
    cls = [{"Model": "M2 SVM - Link Quality", "Accuracy": accuracy_score(yte, pred), "Precision": precision_score(yte, pred, average="macro", zero_division=0), "Recall": recall_score(yte, pred, average="macro", zero_division=0), "F1-score": f1_score(yte, pred, average="macro", zero_division=0)}]

    xtr, xte, ytr, yte = train_test_split(x, y_lifetime, test_size=.2, random_state=42)
    m3 = Pipeline([("preprocessing", make_preprocessor()), ("model", ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1))]).fit(xtr, ytr)
    p = m3.predict(xte)
    reg.append({"Model": "M3 Extra Trees - Link Lifetime", "MAE": mean_absolute_error(yte, p), "RMSE": mean_squared_error(yte, p) ** .5, "R2": r2_score(yte, p)})

    xtr, xte, ytr, yte = train_test_split(x, y_congestion, test_size=.2, random_state=42, stratify=y_congestion)
    m4 = Pipeline([("preprocessing", make_preprocessor()), ("model", KNeighborsClassifier(n_neighbors=9, weights="distance"))]).fit(xtr, ytr)
    pred = m4.predict(xte)
    cls.append({"Model": "M4 KNN - Congestion", "Accuracy": accuracy_score(yte, pred), "Precision": precision_score(yte, pred, average="macro", zero_division=0), "Recall": recall_score(yte, pred, average="macro", zero_division=0), "F1-score": f1_score(yte, pred, average="macro", zero_division=0)})

    xtr, xte, ytr, yte = train_test_split(x, y_route, test_size=.2, random_state=42, stratify=y_route)
    stack = StackingClassifier(estimators=[
        ("svm", Pipeline([("preprocessing", make_preprocessor()), ("model", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42))])),
        ("knn", Pipeline([("preprocessing", make_preprocessor()), ("model", KNeighborsClassifier(n_neighbors=9, weights="distance"))])),
    ], final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced"), cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), stack_method="predict_proba", n_jobs=-1).fit(xtr, ytr)
    pred = stack.predict(xte)
    cls.append({"Model": "Stacking - Route Performance", "Accuracy": accuracy_score(yte, pred), "Precision": precision_score(yte, pred, average="macro", zero_division=0), "Recall": recall_score(yte, pred, average="macro", zero_division=0), "F1-score": f1_score(yte, pred, average="macro", zero_division=0)})

    models = {"bandwidth": m1, "quality": m2, "lifetime": m3, "congestion": m4, "route": stack}
    for key, model in models.items():
        joblib.dump(model, output_dir / MODEL_FILES[key])
    reg_df, cls_df = pd.DataFrame(reg), pd.DataFrame(cls)
    reg_df.to_csv(output_dir / "OrbitalSense_Regression_Metrics.csv", index=False)
    cls_df.to_csv(output_dir / "OrbitalSense_Classification_Metrics.csv", index=False)
    return {"regression": reg_df, "classification": cls_df}


def format_prediction(result: pd.DataFrame) -> str:
    row = result.iloc[0]
    return (f"Future bandwidth: {row['predicted_future_bandwidth_mbps']:.2f} Mbps\n"
            f"Link quality: {row['predicted_link_quality']}\n"
            f"Remaining lifetime: {row['predicted_remaining_lifetime_s']:.2f} s\n"
            f"Congestion: {row['predicted_congestion_level']}\n"
            f"Route performance: {row['predicted_route_performance']}")
