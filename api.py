"""Render-friendly FastAPI wrapper for the OrbitalSense prediction models."""
from typing import Any, Dict, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
from orbitalsense import FEATURES, format_prediction, load_models, predict_new_links

app = FastAPI(title="OrbitalSense AI", version="1.0.0", description="Satellite-link prediction API")
_models = None

class LinkRequest(BaseModel):
    latency_ms: float = Field(ge=0)
    packet_loss_rate: float = Field(ge=0)
    signal_quality_score: float
    available_bandwidth_mbps: float = Field(ge=0)
    congestion_level_pct: float = Field(ge=0, le=100)
    remaining_contact_time_s: float = Field(ge=0)
    relative_velocity_km_s: float = Field(ge=0)
    representative_orbit_altitude_km: float = Field(ge=0)
    inter_satellite_link_distance_km: float = Field(ge=0)
    traffic_load_pct: float = Field(ge=0, le=100)
    queue_delay_ms: float = Field(ge=0)
    position_uncertainty_km: float = Field(ge=0)
    link_available: float = Field(ge=0, le=1)

class BatchRequest(BaseModel):
    records: List[LinkRequest]

def get_models():
    global _models
    if _models is None:
        _models = load_models()
    return _models

@app.get("/")
def root() -> Dict[str, Any]:
    return {"service": "OrbitalSense AI", "status": "ok", "docs": "/docs"}

@app.get("/health")
def health() -> Dict[str, str]:
    try:
        get_models()
        return {"status": "ok", "models": "loaded"}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Models unavailable: {exc}")

@app.post("/predict")
def predict(request: LinkRequest) -> Dict[str, Any]:
    try:
        result = predict_new_links(pd.DataFrame([request.model_dump()]), get_models())
        row = result.iloc[0]
        return {"input": {feature: row[feature] for feature in FEATURES}, "prediction": {column: row[column] for column in result.columns if column.startswith("predicted_")}}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

@app.post("/predict/batch")
def predict_batch(request: BatchRequest) -> Dict[str, Any]:
    try:
        result = predict_new_links(pd.DataFrame([item.model_dump() for item in request.records]), get_models())
        return {"predictions": result.to_dict(orient="records")}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
