# OrbitalSense report replacement guide

These values come from the final run on the bundled project dataset using the cleaned `generate_report_assets.py` evaluation script.

## Yellow text replacements

### Paragraph 57 — key measured results

Use:

> On the held-out test set, the stacked route-performance model achieved **0.8910 accuracy**, **0.8317 macro precision**, **0.8790 macro recall**, and **0.8512 macro F1-score**. The M1 bandwidth model achieved **MAE 0.1475 Mbps, RMSE 0.4249 Mbps, R² 0.999997**, while the M3 link-lifetime model achieved **MAE 0.0226 s, RMSE 0.0998 s, R² 0.999999**. These results are from synthetic data and should not be interpreted as real-constellation performance.

### Paragraph 341 — evaluation protocol

Use:

> The evaluation uses `OrbitalSense_AI_Kaggle_Style_Dataset.csv`, a synthetic Kaggle-style dataset included in the supplied project archive. It contains **10,000 records and 24 columns**. Each model uses an **80% training / 20% held-out test split with random seed 42**. The stacking classifier uses **5-fold StratifiedKFold cross-validation with shuffle=True and random_state=42**. Missing numeric values are median-imputed and numeric features are standardized inside each model pipeline. Classification precision, recall and F1-score are **macro averages** with `zero_division=0`.

### Table 4.2 — regression metrics

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| M1 Random Forest Regressor | **0.1475** | **0.4249** | **0.999997** |
| M3 Extra Trees Regressor | **0.0226** | **0.0998** | **0.999999** |

### Table 4.3 — classification metrics

| Model | Accuracy | Precision (macro) | Recall (macro) | F1-score (macro) |
|---|---:|---:|---:|---:|
| M2 SVM classifier | **0.8615** | **0.8486** | **0.8969** | **0.8640** |
| M4 KNN classifier | **0.8665** | **0.8864** | **0.7768** | **0.8132** |
| Stacked model | **0.8910** | **0.8317** | **0.8790** | **0.8512** |

### Paragraph 396 — results discussion

Use:

> The M1 and M3 scatter plots lie very close to the identity line, which is consistent with their very small errors on this synthetic dataset. For M2, the most frequent confusion is **Good predicted as Degraded** (219 test cases), followed by **Degraded predicted as Good** (57 cases); the Poor class is almost completely separated. For M4, the most frequent confusion is **Medium predicted as Low** (72 cases) or **Medium predicted as High** (9 cases), while Low is confused with Medium in 94 cases and High with Medium in 92 cases. The stacked model improves accuracy over the best individual base classifier (**0.8910 vs 0.8665**) but does not improve macro F1 over M2 (**0.8512 vs 0.8640**). These results come from one fixed synthetic-data split and are not proof of real constellation performance.

### Paragraph 417 — stacking contribution

Replace the placeholder with:

> The measured stacked-model accuracy is **0.8910** and its macro F1-score is **0.8512**.

### Reference [16] — dataset source and license

Do **not** invent a Kaggle URL or license. The uploaded archive contains no public source URL, author metadata, or license file. Use this honest wording unless you have the dataset’s actual provenance:

> [16] OrbitalSense AI Kaggle-style synthetic dataset, file `OrbitalSense_AI_Kaggle_Style_Dataset.csv`, supplied with the project archive by the project author. No external public source URL or redistribution license was included in the supplied files; verify ownership and add an explicit license before public redistribution.

## Generated figures

Insert these files at the corresponding yellow plot placeholders:

- Fig. 4.3: [fig_4_3_regression_actual_vs_predicted.png](report_assets/fig_4_3_regression_actual_vs_predicted.png)
- Fig. 4.4: [fig_4_4_classification_confusion_matrices.png](report_assets/fig_4_4_classification_confusion_matrices.png)
- Fig. 4.5: [fig_4_5_stacked_confusion_matrix.png](report_assets/fig_4_5_stacked_confusion_matrix.png)
- Fig. 4.6: [fig_4_6_model_comparison.png](report_assets/fig_4_6_model_comparison.png)

## UI screenshots

The delivered project is a **Tkinter desktop UI**, not the React/FastAPI/CesiumJS dashboard described in several report paragraphs. Use only screenshots that match the software that actually exists:

- Dashboard/input overview: [screenshot_tkinter_input_form.png](report_assets/screenshot_tkinter_input_form.png)
- Predicted metrics and route rating: [screenshot_tkinter_prediction_result.png](report_assets/screenshot_tkinter_prediction_result.png)
- Collision-risk alert page: **Not implemented in the delivered Tkinter project. Do not fabricate a screenshot.** Replace the placeholder with “Not implemented in the current prototype” or remove it.
- CesiumJS 3D satellite view: **Not implemented in the delivered Tkinter project. Do not fabricate a screenshot.** Replace the placeholder with “Not implemented in the current prototype” or remove it.

If the report is intended to describe a separate React/FastAPI/CesiumJS application, that application’s actual source and running deployment are needed before those four dashboard screenshots can be truthfully produced.

## End-to-end verification log

The cleaned application was tested with the supplied sample CSV:

```text
Command: python3 run_project.py --input new_links_sample.csv
Status: PASS
Input: 1 row, 13 required telemetry features
Output: prediction row written to OrbitalSense_New_Predictions.csv
Predicted future bandwidth: 118.981315 Mbps
Predicted link quality: Good
Predicted remaining lifetime: 179.9671 s
Predicted congestion: Low
Predicted route performance: 1
```

Additional checks:

- Python syntax compilation: **PASS**
- Import of `orbitalsense.py` and `tkinter_ui.py`: **PASS**
- Tkinter form launch under a virtual display: **PASS**
- Model-file integrity and sample prediction: **PASS**
- Invalid/missing input columns are rejected by `validate_input()` with a clear `ValueError`.

The project does not currently expose a FastAPI service or a separate collision-screening endpoint, so those report claims should be removed or marked as future work rather than presented as implemented features.
