# Session Log — 2026-04-14 (Part 2)
**Project:** USB Cable Classification (Stripped vs Unstripped)
**Working directory:** `/Users/paulrydrickpuri/Documents/code/script/USB-classification`
**Duration context:** Continuation of Part 1 same day — auto-label analysis, balanced dataset, COCO packaging

---

## Summary

Following the dataset cleanup and auto-labelling work from Part 1, this session investigated the auto-label results (flagging a critical model bias finding), resolved the SAM crop filename–annotation linkage problem, built a perfectly balanced 1:1 dataset by selecting the highest-resolution stripped images to match all 1,219 unstripped images, and packaged the result in the exact same COCO partition format used by the existing training dataset. The balanced dataset is ready to upload to the platform as a standalone training partition.

---

## What Was Built

### 1. Auto-Label Analysis & Bias Flag
- Reviewed `autolabel_report.json` produced by Part 1's `autolabel.py`
- Discovered: **6,056 stripped / 2 unstripped** out of 6,060 images accepted
- Stripped mean confidence: **1.0000** (6,047 images at ≥0.99 confidence)
- Identified this as model bias — not a genuine class distribution in the unlabelled set
- Root cause: model trained without `sampler: true` on 19:1 imbalance has overwhelming prior toward `stripped`
- After preprocessing (black removal + center square crop), unlabelled images resemble stripped SAM crops → model defaults to stripped

### 2. SAM Hash → Annotation UUID Linkage Discovery
- Investigated why only 540 stripped + 601 unstripped images were found in initial balance scan (only originals, not SAM crops)
- Root cause: SAM crop filenames (`UUID_N_sam_HASH.jpg`) — the `HASH` component is the **first 8 hex chars of the parent annotation UUID** (not the crop UUID)
- Example: `ce8a4701-..._0_sam_2b380ae4.jpg` → parent ann UUID `2b380ae4-267d-4fe2-ac71-c1e1dca1c3e5`
- Confirmed 20/20 test cases matched — 100% reliable lookup
- This unlocked correct class resolution for all 25,760 stripped + 1,219 unstripped disk images

### 3. `balance_dataset.py` — High-Resolution Balanced Dataset
- **File:** `model/dataprep/balance_dataset.py`
- Scans all 6 partitions × 3 splits using the UUID/hash linkage
- Collects all 1,219 unstripped images (kept entirely)
- From 25,760 stripped images, selects top-1,219 by pixel count (w×h) descending
- Copies to `dataset/balanced/stripped/` and `dataset/balanced/unstripped/`
- No black border filtering — images kept as-is
- **Output:** 1,219 stripped + 1,219 unstripped = **2,438 total, 1:1 ratio**
- Selected stripped resolution range: **722,700 – 2,217,680 px²** (mean ~894,540 px²)
- Unstripped resolution range: 15,400 – 912,848 px² (mean ~273,693 px²)

### 4. `package_balanced_coco.py` — COCO Format Packaging
- **File:** `model/dataprep/package_balanced_coco.py`
- Takes `dataset/balanced/` and packages into full COCO partition format
- Output folder: `(COCO)USB Classification(14Apr2026-12_57_31)(balanced)/`
- Stratified 80/10/10 train/val/test split by class (seed=42)
- Builds `annotations.json` per split with `info`, `licenses`, `categories`, `images`, `annotations`
- Globally unique image IDs and annotation IDs across all splits
- Copies images into each split folder alongside `annotations.json`

**Final balanced partition:**

| Split | Total | Stripped | Unstripped |
|-------|:-----:|:--------:|:----------:|
| train | 1,950 | 975      | 975        |
| val   | 242   | 121      | 121        |
| test  | 246   | 123      | 123        |
| **Total** | **2,438** | **1,219** | **1,219** |

---

## Key Decisions & Rationale

| Decision | Choice Made | Why |
|---|---|---|
| Balance strategy | Keep ALL unstripped, select top-N stripped by resolution | Unstripped is precious (only 1,219 exist); stripped pool has 25,760 to choose from — pick best quality |
| Resolution metric | Pixel count (w × h) | Simple, size-independent proxy for detail; higher-res = more visual information for the model |
| Black border handling | Keep as-is, no filtering | User explicitly requested — black backgrounds are SAM masking artefacts, not defects; model can learn from them |
| Train/val/test ratio | 80/10/10 stratified | Matches standard practice; stratification ensures both classes are equally represented in every split |
| SAM hash linkage | `sam_HASH` = first 8 hex chars of parent annotation UUID (no dashes) | Confirmed empirically across 20 test cases — reliable and fast O(1) lookup |
| COCO format | Identical to existing partitions (same `info`, `licenses`, `categories` structure) | Drop-in compatible with the platform — no format changes needed at upload |
| Auto-label confidence | 0.90 (kept from Part 1) | User requested high confidence; result correctly reflects model bias rather than being a config problem |

---

## Findings & Observations

1. **Auto-label result (6,056 stripped / 2 unstripped) is a model bias artefact**, not the true distribution of the unlabelled dataset. The model predicts stripped with mean confidence 1.0000 — this is a textbook symptom of training on severely imbalanced data without a weighted sampler.
2. **SAM crop filename encodes parent UUID**: `UUID_N_sam_HASH.jpg` where `HASH` = first 8 hex characters of the parent annotation UUID (dashes stripped). This is the key to resolving class labels for the 95% of disk images that are SAM crops.
3. **Full stripped pool is 25,760 images** across all 6 partitions × 3 splits (vs initial wrong count of 540 due to incorrect resolution code).
4. **Only 1,219 unique unstripped images exist** across the entire training dataset — confirmed by both annotation scan and the balance script.
5. **Unstripped resolution is lower on average** (mean 273,693 px²) vs selected stripped (mean 894,540 px²) — the unstripped set includes a mix of the cropped originals and smaller SAM crops; all are kept regardless.
6. **Balanced partition is a clean 1:1 ratio** — 975/975 train, 121/121 val, 123/123 test. Training on this should directly address the recall gap (0.855 test recall) that was present in the imbalanced model.
7. **Re-running autolabel after retraining** with corrected config + balanced data is expected to produce a much more realistic class distribution for the 6,060 unlabelled images.
8. **Unlabelled images should be run through autolabel on non-processed (raw) versions** — the preprocessing pipeline (black removal + center crop) may have degraded some images; running on originals with the new balanced model will give more reliable pseudo-labels.

---

## Problems & Fixes

| Problem | Root Cause | Fix Applied |
|---|---|---|
| Balance scan found only 540 stripped (not 25,760) | Code resolved class via exact UUID match only — missed all `_sam_` crop files whose parent UUIDs weren't in annotations by exact match | Rewrote scan using `sam_HASH` → parent UUID prefix lookup; all 25,760 stripped images resolved correctly |
| SAM crops not linkable to annotation class | `_sam_` filenames use a different UUID than the annotation parent; relationship wasn't documented | Reverse-engineered: `HASH` after `_sam_` = first 8 hex chars of the parent annotation UUID. Confirmed 20/20 empirically |

---

## ML / Training Config (Recommended — Next Run)

```yaml
# hyp.yaml — apply before next training run on balanced partition
sampler: true          # CRITICAL — 19:1 imbalance fix (less impactful on balanced data, but still good practice)
lr: 0.0005             # Lower LR — best checkpoint was epoch 12; slow down to avoid overfitting
lr_scheduler: cosine   # No scheduler currently; cosine decay prevents late-epoch divergence
warmup_epochs: 3
epochs: 15             # Stop at epoch 15 — best checkpoint was epoch 12
hsv_h: 0.015           # Minimal hue jitter for lighting variation

# inference_config.yaml
confidence: 0.35       # Lower threshold to recover unstripped recall (was 0.855 at 0.5 threshold)
```

**Next model to trial: YOLOv8m (classification mode)**
- Larger receptive field than ResNet34
- Built-in augmentation pipeline
- Expected to generalise better on the balanced 1:1 dataset
- Will be trained on `(COCO)USB Classification(14Apr2026-12_57_31)(balanced)/`

---

## Files Created / Modified

| File | Location | Purpose |
|---|---|---|
| `balance_dataset.py` | `model/dataprep/` | Scan all partitions, select top-N stripped by resolution, copy to `balanced/` |
| `package_balanced_coco.py` | `model/dataprep/` | Package `balanced/` into COCO partition format with train/val/test splits |
| `balanced/stripped/` | `dataset/balanced/` | 1,219 highest-resolution stripped images |
| `balanced/unstripped/` | `dataset/balanced/` | 1,219 unstripped images (all available) |
| `balanced/balance_report.json` | `dataset/balanced/` | Resolution stats and selection summary |
| `(COCO)…(balanced)/train/` | `dataset/` | 1,950 images + annotations.json (975 per class) |
| `(COCO)…(balanced)/val/` | `dataset/` | 242 images + annotations.json (121 per class) |
| `(COCO)…(balanced)/test/` | `dataset/` | 246 images + annotations.json (123 per class) |

---

## Next Steps / Open Questions

- [ ] **Train YOLOv8m (classification)** on `(COCO)USB Classification(14Apr2026-12_57_31)(balanced)/` with corrected config
- [ ] **Re-run autolabel** (`autolabel.py`) on the **non-processed (raw) unlabelled images** after retraining — the balanced+corrected model will give a more realistic stripped/unstripped distribution
- [ ] **Human spot-check** auto-label results for unstripped class before adding to training set
- [ ] Verify YOLOv8m accepts COCO classification format on the platform (may need conversion to YOLOv8 flat class-folder format)
- [ ] Monitor whether balanced training closes the unstripped recall gap (target: ≥0.93 from current 0.855)
- [ ] Consider adding auto-labelled stripped images (6,056) from `dataset/labelled/stripped/` to augment next training run after human validation

---

## Session Metadata
- **Date:** 2026-04-14
- **Model:** Claude (claude-sonnet-4-6)
- **Working directory:** `/Users/paulrydrickpuri/Documents/code/script/USB-classification`
- **Key packages used:** Python 3.12, PyTorch 2.6, torchvision, Pillow 12.1.1, NumPy 2.4.3
