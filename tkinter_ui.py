"""Tkinter UI for OrbitalSense link prediction."""
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
from orbitalsense import DEFAULT_SAMPLE, FEATURES, format_prediction, load_models, predict_new_links

COLORS = {"bg": "#0b1726", "panel": "#12263a", "field": "#18344b", "text": "#f4f8fb", "muted": "#9db2c5", "accent": "#35d0ba", "border": "#27465d"}
HINTS = {"latency_ms": "milliseconds", "packet_loss_rate": "ratio", "signal_quality_score": "0 - 100", "available_bandwidth_mbps": "Mbps", "congestion_level_pct": "percent", "remaining_contact_time_s": "seconds", "relative_velocity_km_s": "km/s", "representative_orbit_altitude_km": "km", "inter_satellite_link_distance_km": "km", "traffic_load_pct": "percent", "queue_delay_ms": "milliseconds", "position_uncertainty_km": "km", "link_available": "1 = yes, 0 = no"}

class OrbitalSenseApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("OrbitalSense | Link Prediction")
        self.configure(bg=COLORS["bg"])
        self.resizable(False, False)
        self.entries = {}
        self.models = None
        self.last_result = None
        self._configure_style()
        self._build_ui()

    def _configure_style(self):
        self.option_add("*Font", ("Segoe UI", 10))
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Input.TEntry", fieldbackground=COLORS["field"], foreground=COLORS["text"], insertcolor=COLORS["accent"], bordercolor=COLORS["border"], lightcolor=COLORS["border"], darkcolor=COLORS["border"], padding=7)
        style.configure("Primary.TButton", background=COLORS["accent"], foreground="#06231f", font=("Segoe UI", 10, "bold"), padding=(18, 9), borderwidth=0)
        style.map("Primary.TButton", background=[("active", "#62e4d2")])
        style.configure("Secondary.TButton", background=COLORS["field"], foreground=COLORS["text"], padding=(12, 8), borderwidth=0)

    def _build_ui(self):
        form = tk.Frame(self, bg=COLORS["bg"], padx=24, pady=22); form.pack()
        tk.Label(form, text="ORBITALSENSE", fg=COLORS["accent"], bg=COLORS["bg"], font=("Segoe UI", 10, "bold")).grid(row=0, column=0, columnspan=2, sticky="w")
        tk.Label(form, text="New link prediction", fg=COLORS["text"], bg=COLORS["bg"], font=("Segoe UI", 22, "bold")).grid(row=1, column=0, columnspan=2, sticky="w")
        tk.Label(form, text="Enter satellite link telemetry to estimate route performance.", fg=COLORS["muted"], bg=COLORS["bg"]).grid(row=2, column=0, columnspan=2, sticky="w", pady=(2, 18))
        panel = tk.Frame(form, bg=COLORS["panel"], padx=18, pady=16, highlightthickness=1, highlightbackground=COLORS["border"]); panel.grid(row=3, column=0, columnspan=2, sticky="ew")
        tk.Label(panel, text="LINK TELEMETRY", fg=COLORS["muted"], bg=COLORS["panel"], font=("Segoe UI", 9, "bold")).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))
        for index, feature in enumerate(FEATURES):
            col, row = divmod(index, 7); cell = tk.Frame(panel, bg=COLORS["panel"]); cell.grid(row=row + 1, column=col, sticky="ew", padx=(0, 22 if col == 0 else 0), pady=4)
            tk.Label(cell, text=feature.replace("_", " ").title(), fg=COLORS["text"], bg=COLORS["panel"], anchor="w", width=28).pack(anchor="w")
            tk.Label(cell, text=HINTS[feature], fg=COLORS["muted"], bg=COLORS["panel"], font=("Segoe UI", 8)).pack(anchor="w")
            entry = ttk.Entry(cell, width=22, style="Input.TEntry"); entry.insert(0, str(DEFAULT_SAMPLE[feature])); entry.pack(anchor="w", pady=(3, 0)); self.entries[feature] = entry
        panel.columnconfigure(0, weight=1); panel.columnconfigure(1, weight=1)
        actions = tk.Frame(form, bg=COLORS["bg"]); actions.grid(row=4, column=0, columnspan=2, sticky="e", pady=(18, 0))
        ttk.Button(actions, text="Load CSV", command=self.load_csv, style="Secondary.TButton").pack(side="left", padx=(0, 8))
        ttk.Button(actions, text="Reset", command=self.reset, style="Secondary.TButton").pack(side="left", padx=(0, 8))
        ttk.Button(actions, text="Predict link", command=self.predict, style="Primary.TButton").pack(side="left")
        self.status = tk.Label(form, text="Bundled models are ready for prediction.", fg=COLORS["muted"], bg=COLORS["bg"]); self.status.grid(row=5, column=0, columnspan=2, sticky="w", pady=(12, 0))

    def reset(self):
        for feature, entry in self.entries.items():
            entry.delete(0, tk.END); entry.insert(0, str(DEFAULT_SAMPLE[feature]))
        self.status.config(text="Values reset to the sample defaults.")

    def _read_form(self):
        values = {}
        for feature, entry in self.entries.items():
            try: values[feature] = float(entry.get().strip())
            except ValueError: raise ValueError(f"{feature} must contain a number.")
        return pd.DataFrame([values])

    def predict(self):
        try:
            data = self._read_form()
            if self.models is None: self.models = load_models()
            self.last_result = predict_new_links(data, self.models)
            self.last_result.to_csv(Path(__file__).resolve().parent / "OrbitalSense_New_Predictions.csv", index=False)
            self._show_result()
        except Exception as exc:
            messagebox.showerror("Prediction failed", str(exc), parent=self)

    def load_csv(self):
        path = filedialog.askopenfilename(title="Select a link input CSV", filetypes=[("CSV files", "*.csv")])
        if not path: return
        try:
            data = pd.read_csv(path)
            if len(data) != 1: raise ValueError("The form UI accepts exactly one CSV row. Use run_project.py --input for multiple rows.")
            for feature in FEATURES: self.entries[feature].delete(0, tk.END); self.entries[feature].insert(0, str(data.iloc[0][feature]))
            self.status.config(text=f"Loaded {Path(path).name}. Click Predict link to continue.")
        except Exception as exc: messagebox.showerror("Could not load CSV", str(exc), parent=self)

    def _show_result(self):
        dialog = tk.Toplevel(self); dialog.title("Prediction result"); dialog.configure(bg=COLORS["bg"]); dialog.resizable(False, False)
        tk.Label(dialog, text="PREDICTION RESULT", fg=COLORS["accent"], bg=COLORS["bg"], font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=24, pady=(22, 4))
        tk.Label(dialog, text=format_prediction(self.last_result), justify="left", fg=COLORS["text"], bg=COLORS["panel"], padx=20, pady=18, font=("Segoe UI", 12)).pack(padx=24, pady=(0, 18))
        ttk.Button(dialog, text="Close", command=dialog.destroy, style="Primary.TButton").pack(pady=(0, 20))
        self.status.config(text="Prediction saved to OrbitalSense_New_Predictions.csv.")

def main():
    OrbitalSenseApp().mainloop()

if __name__ == "__main__": main()
