"""Entry point for the standalone OrbitalSense project."""
import argparse
from pathlib import Path
import pandas as pd
from orbitalsense import format_prediction, predict_new_links, train_models

parser = argparse.ArgumentParser(description="OrbitalSense satellite-link prediction")
mode = parser.add_mutually_exclusive_group()
mode.add_argument("--gui", action="store_true", help="Open the Tkinter input form")
mode.add_argument("--input", type=Path, help="Predict one or more rows from a CSV file")
mode.add_argument("--train", action="store_true", help="Retrain all models from the bundled dataset")
args = parser.parse_args()

if args.gui or not (args.input or args.train):
    from tkinter_ui import main
    main()
elif args.train:
    results = train_models()
    print("Training complete.")
    print("\nRegression metrics:\n", results["regression"].round(4).to_string(index=False))
    print("\nClassification metrics:\n", results["classification"].round(4).to_string(index=False))
else:
    result = predict_new_links(pd.read_csv(args.input))
    output = Path(__file__).resolve().parent / "OrbitalSense_New_Predictions.csv"
    result.to_csv(output, index=False)
    print(result.to_string(index=False))
    print(f"\nSaved predictions to {output}")
