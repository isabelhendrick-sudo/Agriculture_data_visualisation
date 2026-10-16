# ISE vs YOLOv8 early-warning kit

```
pip install pandas numpy matplotlib
python simulate_practice_data.py   # writes SYNTHETIC practice CSVs to ./data
python make_figures.py             # reads ./data, writes ./figures/*.png + alarm_table.csv
```

When your real data is ready, replace the files in `./data` with your own (same column names, no `synthetic` column) and rerun `make_figures.py`.

> The practice data is made up. Use it to build and debug the pipeline. Do not report it as a result.

## The six files your experiment must produce

| File | One row per | Key columns | How to get it |
|---|---|---|---|
| `1_units_and_plants.csv` | plant | plant_id, reservoir_id, treatment, bench_row, position_col | Write it once at set-up |
| `2_ise_readings.csv` | reservoir × ion × time | timestamp, ion, mV_raw, temp_C, conc_mM, calibration_id | Logger every 1–6 h (NO3⁻ and K⁺ at minimum) |
| `3_ise_calibrations.csv` | calibration standard | standard_mM, mV, slope_mV_per_decade, r2 | Daily 2–3-point calibration; log the check standard |
| `4_yolo_detections.csv` | detection | image_id, plant_id, timestamp, class, confidence, bbox | Daily photo per plant → run `best.pt` with `conf=0.05` and save **all** detections |
| `5_reference_truth.csv` | plant × scoring day | blinded visible score (0–4), tissue N/K %, lab-confirmed disease | Blinded scorer every 2 days; tissue test every 4–7 days |
| `6_events_log.csv` | event | solution changes, top-ups, corrections, inoculations | Note it every time it happens |

The fusion data doesn't need its own sensor. It's calculated from files 2 and 4 by `make_figures.py`, using the earliest alarm ("either") and both-sensor agreement ("both").

## Minimum experiment

* **Reservoirs are the replicates.** Plants sharing one reservoir share one ISE trace. Aim for at least 4 reservoirs per treatment: control, −N, −K, and a disease treatment (or natural infection) as a negative case for ISE.
* Run for 4–6 weeks, or until visible symptoms appear in every stressed reservoir.
* Photograph the same plants from a fixed angle under fixed lighting each day.
* Set the alarm rules at the top of `make_figures.py` before you look at the outcomes.

## Exporting YOLOv8 detections to file 4

```python
from ultralytics import YOLO
import pandas as pd, pathlib
m = YOLO("runs/detect/train/weights/best.pt")
rows = []
for r in m.predict("daily_photos/", conf=0.05, stream=True):
    name = pathlib.Path(r.path).stem              # e.g. IMG_1105_P017  (date_plant)
    for b in r.boxes:
        x, y, w, h = b.xywhn[0].tolist()
        rows.append(dict(image_id=name, plant_id=name.split("_")[-1], cls=r.names[int(b.cls)],
                         confidence=float(b.conf), x_center=x, y_center=y, width=w, height=h))
pd.DataFrame(rows).to_csv("4_yolo_detections_raw.csv", index=False)  # then add reservoir_id + timestamp
```
