"""
Find near-identical images that ended up in different splits of the merged dataset
(e.g. the same photo in ds1_train and ds2_test). Those leak test data into training and inflate mAP.
 
    pip install imagehash pillow numpy
    python check_duplicates.py          -> prints a summary, writes duplicates_across_splits.csv
 
Then remove/move the listed test (and valid) copies, and re-run training and evaluation.
"""
from pathlib import Path
import csv, imagehash
import numpy as np
from PIL import Image
 
DATASET = Path(r"C:\Users\isabe\OneDrive\Documents\strawberry_expert\merged_dataset")
MAX_DIST = 4          # perceptual-hash distance; 0 = identical, <=4 = same photo resized/re-encoded
 
items = []
for split in ["train", "valid", "test"]:
    for p in (DATASET / split / "images").iterdir():
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            items.append((split, p, imagehash.phash(Image.open(p))))
print(f"hashed {len(items)} images")
 
# compare every pair (vectorised; a few thousand images takes seconds)
H = np.array([int(str(h), 16) for _, _, h in items], dtype=np.uint64)
pairs = []
for i in range(len(items)):
    x = np.bitwise_xor(H[i + 1:], H[i])
    d = np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(1)
    for j in np.nonzero(d <= MAX_DIST)[0] + i + 1:
        a, b = items[i], items[j]
        if a[0] != b[0]:
            pairs.append((a[0], a[1].name, b[0], b[1].name, int(d[j - i - 1])))
 
with open(DATASET / "duplicates_across_splits.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["split_a", "image_a", "split_b", "image_b", "hash_distance"]); w.writerows(pairs)
test_hit = {p[1] for p in pairs if p[0] == "test"} | {p[3] for p in pairs if p[2] == "test"}
print(f"{len(pairs)} cross-split duplicate pairs; {len(test_hit)} test images also appear in another split")
print("details:", DATASET / "duplicates_across_splits.csv")
 