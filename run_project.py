"""Command-line utilities for the OrbitalSense project.

The browser UI is served by `uvicorn api:app`; this file remains useful for
CSV inference and retraining from a terminal.
"""
import argparse
from pathlib import Path
import pandas as pd
from orbitalsense import predict_new_links, train_models

parser = argparse.ArgumentParser(description="OrbitalSense satellite-link prediction")
mode = parser.add_mutually_exclusive_group()
mode.add_argument("--input", type=Path, help="Predict one or more rows from a CSV file")
mode.add_argument("--train", action="store_true", help="Retrain all models from the bundled dataset")
mode.add_argument("--web", action="store_true", help="Start the FastAPI browser UI")
args = parser.parse_args()

if args.web:
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
elif args.train:
    results = train_models()
    print("Training complete.")
    print("\nRegression metrics:\n", results["regression"].round(4).to_string(index=False))
    print("\nClassification metrics:\n", results["classification"].round(4).to_string(index=False))
elif args.input:
    result = predict_new_links(pd.read_csv(args.input))
    output = Path(__file__).resolve().parent / "OrbitalSense_New_Predictions.csv"
    result.to_csv(output, index=False)
    print(result.to_string(index=False))
    print(f"\nSaved predictions to {output}")
else:
    parser.print_help()
    print("\nBrowser UI: uvicorn api:app --reload")
