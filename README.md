# OrbitalSense ML Project

This is a cleaned, standalone version of the uploaded project. Repository internals and unrelated generated files were removed. The bundled `.joblib` files let you run predictions without retraining.

## Install

Use Python 3.9 or newer.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Run the browser UI locally

```bash
uvicorn api:app --reload
```

Open `http://127.0.0.1:8000/` in a browser. Enter the 13 telemetry values and click **Predict link**.

## Run the Render-ready API locally

```bash
uvicorn api:app --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive API. Endpoints are `GET /health`, `POST /predict`, and `POST /predict/batch`.

For Render, create a new Web Service from this repository. Render can use `render.yaml` automatically, or use:

- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn api:app --host 0.0.0.0 --port $PORT`
- Health check path: `/health`

The same FastAPI process serves the browser dashboard at `/` and the JSON API endpoints.

## Run without the UI

```bash
python run_project.py --input new_links_sample.csv
```

The input CSV must contain these columns:

`latency_ms`, `packet_loss_rate`, `signal_quality_score`, `available_bandwidth_mbps`, `congestion_level_pct`, `remaining_contact_time_s`, `relative_velocity_km_s`, `representative_orbit_altitude_km`, `inter_satellite_link_distance_km`, `traffic_load_pct`, `queue_delay_ms`, `position_uncertainty_km`, `link_available`.

## Retrain

```bash
python run_project.py --train
```

Training uses `OrbitalSense_AI_Kaggle_Style_Dataset.csv`, overwrites the five model files, and updates the two metric CSVs.

## Important note

The project uses synthetic data and is for demonstration/academic use. Its predictions do not validate real-world satellite routing or collision avoidance.
