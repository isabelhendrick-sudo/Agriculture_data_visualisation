""
Run YOUR trained YOLOv8 model on YOUR merged dataset's test split and record, for every image and class,
the highest confidence the model gave that class. Output: yolo_profile.csv (used by simulate_practice_data.py).
 
    pip install ultralytics pyyaml
    python yolo_confidence_profile.py
 
This is the only real-data step in the semi-synthetic mode: it captures how your model actually scores
images that have a disease (hits and misses) and images that don't (false positives).
Use the TEST split only - never images the model trained or validated on.
"""
from pathlib import Path
import yaml, pandas as pd
from ultralytics import YOLO
 
# ── edit these ──────────────────────────────────────────────────────────────
WEIGHTS = r"C:\Users\isabe\runs\detect\train\weights\best.pt"          # your Run 2 best.pt
DATASET = r"C:\Users\isabe\OneDrive\Documents\strawberry_expert\merged_dataset"
SPLIT = "test"
# ────────────────────────────────────────────────────────────────────────────
 
root = Path(DATASET)
names = yaml.safe_load(open(root / "data.yaml"))["names"]
names = list(names.values()) if isinstance(names, dict) else names
model = YOLO(WEIGHTS)
if list(model.names.values()) != names:
    raise SystemExit(f"Class order differs!\n model: {list(model.names.values())}\n yaml:  {names}")
 
rows = []
imgs = sorted(p for p in (root / SPLIT / "images").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
for r in model.predict([str(p) for p in imgs], conf=0.01, stream=True, verbose=False):
    img = Path(r.path)
    lab = root / SPLIT / "labels" / (img.stem + ".txt")
    present = {int(l.split()[0]) for l in lab.read_text().splitlines() if l.strip()} if lab.exists() else set()
    best = {}
    for c, s in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
        best[int(c)] = max(best.get(int(c), 0.0), float(s))
    for k, n in enumerate(names):
        rows.append(dict(image=img.name, **{"class": n}, gt_present=int(k in present), max_conf=round(best.get(k, 0.0), 4)))
 
df = pd.DataFrame(rows)
df.to_csv(Path(__file__).parent / "yolo_profile.csv", index=False)
# quick summary at the 0.5 alarm threshold
s = df.assign(hit=df.max_conf >= 0.5).groupby(["class", "gt_present"]).hit.mean().unstack()
s.columns = ["false_alarm_rate (no disease)", "detection_rate (disease present)"]
print(f"{len(imgs)} {SPLIT} images\n", s.round(3).to_string())