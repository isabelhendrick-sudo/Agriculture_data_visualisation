"""
Generate a SYNTHETIC practice dataset in the same CSV format your real experiment should produce.
 
!! Every row is made up. Use it only to build and test your analysis code and figure layouts.
!! Never report these numbers as results. Replace the CSVs with your own measurements.
 
Design: 16 independent reservoirs (6 plants each), 30 days
  R01-R04 control | R05-R08 minus_N | R09-R12 minus_K | R13-R16 control + powdery mildew inoculation
 
Usage:  python simulate_practice_data.py            -> writes ./data/*.csv
"""
import numpy as np, pandas as pd
from pathlib import Path
 
rng = np.random.default_rng(42)
OUT = Path(__file__).parent / "data"; OUT.mkdir(exist_ok=True)
START = pd.Timestamp("2026-11-01 09:00")
DAYS = 30
TARGET = {"NO3": 12.0, "K": 6.0}           # mM, full-strength target in the solution
SLOPE = {"NO3": -56.5, "K": 57.0}          # mV per decade (Nernstian, anion negative)
E0 = {"NO3": 245.0, "K": 52.0}             # mV offset at 1 mM
sig = lambda x: 1 / (1 + np.exp(-x))
 
# ---------- 1. units and plants ----------
treat = (["control"] * 4) + (["minus_N"] * 4) + (["minus_K"] * 4) + (["mildew"] * 4)
res = [f"R{i+1:02d}" for i in range(16)]
plants = []
for r, (rid, t) in enumerate(zip(res, treat)):
    for p in range(6):
        plants.append(dict(plant_id=f"P{r*6+p+1:03d}", reservoir_id=rid, treatment=t,
                           bench_row=r // 4 + 1, position_col=(r % 4) * 6 + p + 1,
                           variety="Seolhyang", planted_date="2026-10-25", synthetic="yes"))
plants = pd.DataFrame(plants)
 
# reservoir-level hidden "truth" used to generate everything else
depl_day = {rid: rng.uniform(5, 9) for rid in res}       # when the solution starts falling
onset_day = {}                                           # visible symptom day (reservoir median)
for rid, t in zip(res, treat):
    if t in ("minus_N", "minus_K"): onset_day[rid] = depl_day[rid] + rng.uniform(11, 15)
    elif t == "mildew": onset_day[rid] = rng.uniform(12, 17)
    else: onset_day[rid] = np.nan
 
# ---------- 2. ISE readings (every 6 h, NO3 and K) + 3. calibrations ----------
rows, cals = [], []
times = START + pd.to_timedelta(np.arange(0, DAYS * 24, 6), unit="h")
for rid, t in zip(res, treat):
    drift = {ion: rng.normal(0, 0.15) for ion in TARGET}       # mV/day electrode drift
    for ti, ts in enumerate(times):
        d = ti * 6 / 24
        temp = 21 + 1.5 * np.sin(2 * np.pi * (d % 1)) + rng.normal(0, 0.3)
        for ion in TARGET:
            c = TARGET[ion] * (1 + rng.normal(0, 0.03))
            if (t == "minus_N" and ion == "NO3") or (t == "minus_K" and ion == "K"):
                c = TARGET[ion] * (1 - 0.8 * sig((d - depl_day[rid] - 4) / 2.0)) * (1 + rng.normal(0, 0.04))
            if t == "mildew" and ion == "NO3":   # sick plants take up a little less -> slight rise
                c *= 1 + 0.05 * sig((d - onset_day[rid]) / 2)
            mv = E0[ion] + SLOPE[ion] * np.log10(c) + drift[ion] * d + rng.normal(0, 0.8)
            # calibrated concentration (daily recalibration removes most drift)
            mv_corr = mv - drift[ion] * (d % 1)
            conc = 10 ** ((mv_corr - E0[ion]) / SLOPE[ion])
            rows.append(dict(reservoir_id=rid, timestamp=ts.isoformat(timespec="minutes"), ion=ion,
                             mV_raw=round(mv, 2), temp_C=round(temp, 2), conc_mM=round(conc, 3),
                             calibration_id=f"CAL-{ts:%m%d}", ec_mS_cm=round(1.4 * (0.5 + 0.5 * c / TARGET[ion]), 3),
                             ph=round(5.8 + rng.normal(0, 0.05), 2), synthetic="yes"))
for day in range(DAYS):
    ts = START.normalize() + pd.Timedelta(days=day, hours=8, minutes=30)
    for ion in TARGET:
        for std in (1, 10, 100):
            cals.append(dict(calibration_id=f"CAL-{ts:%m%d}", timestamp=ts.isoformat(timespec="minutes"), ion=ion,
                             standard_mM=std, mV=round(E0[ion] + SLOPE[ion] * np.log10(std) + rng.normal(0, 0.5), 2),
                             slope_mV_per_decade=round(SLOPE[ion] + rng.normal(0, 0.6), 2),
                             r2=round(0.998 + rng.uniform(-0.002, 0.0015), 4), synthetic="yes"))
ise = pd.DataFrame(rows); cal = pd.DataFrame(cals)
 
# ---------- 4. YOLOv8 detections (one image per plant per day) ----------
cls_for = {"minus_N": "N_deficiency", "minus_K": "K_deficiency", "mildew": "powdery_mildew"}
 
# Optional SEMI-SYNTHETIC mode: if yolo_profile.csv exists (made by yolo_confidence_profile.py from YOUR
# model on YOUR test images), YOLOv8 confidences are sampled from your model's real score distributions.
# Map each treatment to a class name exactly as it appears in your data.yaml. Treatments with no matching
# class in your model stay fully synthetic.
PROFILE = Path(__file__).parent / "yolo_profile.csv"
REAL_CLASS_FOR = {"mildew": "Powdery Mildew Leaf"}      # e.g. add "minus_N": "N_deficiency" if your model has it
prof = pd.read_csv(PROFILE) if PROFILE.exists() else None
if prof is not None:
    pos = {c: g.max_conf.to_numpy() for c, g in prof[prof.gt_present == 1].groupby("class")}
    neg = {c: g.set_index("image").max_conf for c, g in prof[prof.gt_present == 0].groupby("class")}
    neg_images = sorted(set(prof.loc[prof.gt_present == 0, "image"]))
    REAL_CLASS_FOR = {t: c for t, c in REAL_CLASS_FOR.items() if c in pos}
    print("Semi-synthetic mode: real YOLOv8 scores used for", REAL_CLASS_FOR or "nothing (check class names)")
    def neg_draw():
        """Strongest wrong-class score on a random real image without that class -> realistic false positives."""
        im = neg_images[rng.integers(len(neg_images))]
        scores = {c: s.get(im, 0.0) for c, s in neg.items()}
        c = max(scores, key=scores.get); return c, float(scores[c])
FLAG = "semi" if prof is not None and REAL_CLASS_FOR else "yes"
det, ref = [], []
for _, p in plants.iterrows():
    rid, t = p.reservoir_id, p.treatment
    plant_onset = onset_day[rid] + rng.normal(0, 1.0) if t != "control" else np.nan
    for day in range(DAYS):
        ts = START.normalize() + pd.Timedelta(days=day, hours=10)
        img = f"IMG_{ts:%m%d}_{p.plant_id}"
        if t == "control":
            fp = rng.random() < 0.015          # occasional false positive
            cls = rng.choice(list(cls_for.values())) if fp else "healthy"
            conf = rng.uniform(0.5, 0.7) if fp else rng.uniform(0.8, 0.97)
        else:
            lead = 2.5 if t == "mildew" else 1.5  # YOLO sees subtle signs a little before human scoring
            s = sig((day - (plant_onset - lead)) / 1.1)
            cls = cls_for[t] if s > 0.35 else "healthy"
            conf = float(np.clip(s + rng.normal(0, 0.06), 0.05, 0.99)) if cls != "healthy" else float(np.clip(1 - s + rng.normal(0, 0.05), 0.4, 0.98))
        if FLAG == "semi" and (t == "control" or t in REAL_CLASS_FOR):
            s_now = 0.0 if t == "control" else sig((day - (plant_onset - 1.5)) / 1.1)
            if rng.random() < s_now:      # plant is showing symptoms -> score drawn from real positive images
                cls, conf = REAL_CLASS_FOR[t], float(rng.choice(pos[REAL_CLASS_FOR[t]]))
            else:                         # not yet -> score drawn from real images without the disease
                cls, conf = neg_draw()
            if conf < 0.05: cls, conf = "healthy", 0.9
        det.append(dict(image_id=img, plant_id=p.plant_id, reservoir_id=rid, timestamp=ts.isoformat(timespec="minutes"),
                        model_version="best.pt", **{"class": cls}, confidence=round(conf, 3),
                        x_center=round(rng.uniform(0.35, 0.65), 3), y_center=round(rng.uniform(0.35, 0.65), 3),
                        width=round(rng.uniform(0.15, 0.4), 3), height=round(rng.uniform(0.15, 0.4), 3), synthetic=FLAG))
        # ---------- 5. reference truth: blinded score every 2 days, tissue test every 4 days ----------
        if day % 2 == 0:
            score = 0 if np.isnan(plant_onset) or day < plant_onset else int(min(4, 1 + (day - plant_onset) // 3))
            row = dict(plant_id=p.plant_id, reservoir_id=rid, date=f"{ts:%Y-%m-%d}", scorer_id="S1", blinded="yes",
                       visible_symptom_score_0to4=score,
                       symptom_type="none" if score == 0 else {"minus_N": "interveinal_yellowing", "minus_K": "marginal_scorch", "mildew": "white_powder"}[t],
                       tissue_N_pct="", tissue_K_pct="", disease_lab_confirmed="", synthetic="yes")
            if day % 4 == 0:
                n = 2.8 - (1.3 * sig((day - plant_onset + 4) / 2) if t == "minus_N" else 0) + rng.normal(0, 0.08)
                k = 2.0 - (1.0 * sig((day - plant_onset + 4) / 2) if t == "minus_K" else 0) + rng.normal(0, 0.07)
                row.update(tissue_N_pct=round(n, 2), tissue_K_pct=round(k, 2))
            if t == "mildew" and score > 0: row["disease_lab_confirmed"] = "powdery_mildew"
            ref.append(row)
det = pd.DataFrame(det); ref = pd.DataFrame(ref)
 
# ---------- 6. events ----------
ev = [dict(reservoir_id=rid, timestamp=START.isoformat(timespec="minutes"), event="solution_change",
           detail={"control": "full Hoagland", "minus_N": "N withheld from top-ups", "minus_K": "K withheld from top-ups",
                   "mildew": "full Hoagland"}[t], synthetic="yes") for rid, t in zip(res, treat)]
ev += [dict(reservoir_id=rid, timestamp=(START + pd.Timedelta(days=3)).isoformat(timespec="minutes"),
            event="inoculation", detail="Podosphaera aphanis spore suspension", synthetic="yes")
       for rid, t in zip(res, treat) if t == "mildew"]
 
for name, df in [("1_units_and_plants", plants), ("2_ise_readings", ise), ("3_ise_calibrations", cal),
                 ("4_yolo_detections", det), ("5_reference_truth", ref), ("6_events_log", pd.DataFrame(ev))]:
    df.to_csv(OUT / f"{name}.csv", index=False)
    print(f"{name}.csv  {len(df):>6} rows")
print(f"\n{'SEMI-SYNTHETIC' if FLAG == 'semi' else 'SYNTHETIC'} practice data written to", OUT)
 