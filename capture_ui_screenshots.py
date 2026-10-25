from pathlib import Path
import tkinter as tk
from PIL import ImageGrab
from tkinter_ui import OrbitalSenseApp

OUT = Path(__file__).resolve().parent / 'report_assets'
OUT.mkdir(exist_ok=True)
app = OrbitalSenseApp()
app.update_idletasks()

def bbox(widget):
    return (widget.winfo_rootx(), widget.winfo_rooty(), widget.winfo_rootx()+widget.winfo_width(), widget.winfo_rooty()+widget.winfo_height())

def capture_root():
    app.update_idletasks()
    ImageGrab.grab(bbox=bbox(app)).save(OUT/'screenshot_tkinter_input_form.png')
    app.predict()
    app.after(1000, capture_result)

def capture_result():
    dialogs=[w for w in app.winfo_children() if isinstance(w, tk.Toplevel)]
    if dialogs:
        d=dialogs[0]; d.update_idletasks(); ImageGrab.grab(bbox=bbox(d)).save(OUT/'screenshot_tkinter_prediction_result.png')
    app.destroy()

app.after(1000, capture_root)
app.mainloop()
print('screenshots generated')
