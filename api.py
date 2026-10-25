"""Render-friendly FastAPI wrapper for the OrbitalSense prediction models."""
import json
from pathlib import Path
from typing import Any, Dict, List
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import pandas as pd
from orbitalsense import FEATURES, format_prediction, load_models, predict_new_links

app = FastAPI(title="OrbitalSense AI", version="1.0.0", description="Satellite-link prediction API")
_models = None
WEB_ROOT = Path(__file__).resolve().parent / "static"

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

@app.get("/", include_in_schema=False)
def root():
    return FileResponse(WEB_ROOT / "index.html")

@app.get("/manus-routes.json", include_in_schema=False)
def routes_manifest() -> Dict[str, Any]:
    return {"routes": [{"path": "/", "title": "OrbitalSense AI Link Intelligence"}]}

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
        return json.loads(result.to_json(orient="records"))[0] | {
            "prediction": {
                column: json.loads(result[[column]].to_json(orient="records"))[0][column]
                for column in result.columns if column.startswith("predicted_")
            }
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

@app.post("/predict/batch")
def predict_batch(request: BatchRequest) -> Dict[str, Any]:
    try:
        result = predict_new_links(pd.DataFrame([item.model_dump() for item in request.records]), get_models())
        return {"predictions": json.loads(result.to_json(orient="records"))}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
