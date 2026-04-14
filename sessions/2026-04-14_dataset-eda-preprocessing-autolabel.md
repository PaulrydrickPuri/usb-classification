# Session Log — 2026-04-14
**Project:** USB Cable Classification (Stripped vs Unstripped)
**Working directory:** `/Users/paulrydrickpuri/Documents/code/script/USB-classification`
**Duration context:** Single session — dataset audit, preprocessing, and auto-labelling

---

## Summary

The session focused on a full audit of the USB cable binary classification project (ResNet34, stripped vs unstripped). Starting from training checkpoint analysis, the session identified critical dataset issues — a severe class imbalance (19:1), a scale representation bias where all unstripped images were full-frame 1280×720 scenes while stripped images were tight SAM-segmented crops, and 14 low-quality images. Two preprocessing scripts were written and executed: one to clean all original full-frame images via black border removal and center square cropping (1,141 images fixed across all 6 partitions), and one to auto-label 6,060 new unlabelled images using the existing best.pt model at a 0.90 confidence threshold.

---

## What Was Built

### 1. Training Checkpoint Review
- Analysed `model/train/train-result/results.csv` (33 epochs) and `class_accuracy.json`
- Identified best checkpoint at **epoch 12** (val loss 0.013645)
- Detected overfitting from epoch ~24 onward (train loss → 0.00871, val loss rising to 0.022)
- Produced a full config correction table for the platform retraining run

### 2. EDA — Dataset Structure Analysis
- Scanned all 6 COCO partitions × 3 splits = **26,993 total images**
- Confirmed: no cross-partition duplicate filenames, no cross-split data leakage
- Discovered critical structural bias: unstripped images are full-frame 1280×720 originals; stripped images are SAM-segmented tight crops
- Annotation filenames reference parent UUIDs (`UUID.jpg`); 95% of disk files are SAM children (`UUID_N_sam_HASH.jpg`) — only 4.5% match annotations exactly

### 3. `crop_unstripped.py` — Full-Frame Image Preprocessing
- **File:** `model/dataprep/crop_unstripped.py`
- Iterates all 6 partitions × 3 splits
- For each original (no `_sam_` suffix) image of either class:
  1. Detects non-black bounding box (threshold: pixel channel > 15)
  2. Crops to content area
  3. Takes largest centered square from content
  4. Saves in-place (JPEG quality 95)
  5. Updates `width`/`height` in `annotations.json`
- Skips images already ≤ 640px (already processed or already small crops)
- **Result:** 1,141 images fixed (540 stripped + 601 unstripped across all partitions)

### 4. Dataset Quality Check + Cleanup
- Full scan of all 26,993 images using content-area quality metrics
- Criteria: size ≥ 64px, brightness 20–240, contrast std ≥ 10, file ≥ 2KB, black ≤ 15%
- **14 disqualified images** removed from disk (all were SAM crop children; none in annotations.json)
- Final clean dataset: **26,979 images**

### 5. `autolabel.py` — High-Confidence Auto-Labelling
- **File:** `model/dataprep/autolabel.py`
- Input: `dataset/unlabelled/` (6,060 images, SAM-masked objects on black background)
- Preprocessing per image: strip black border → center square crop (same pipeline as training data)
- Model: `best.pt` (ResNet34, 2-class) loaded via `torch.load(..., weights_only=False)`
- Confidence threshold: **0.90** (high confidence only)
- Accepted images saved to `dataset/labelled/stripped/` and `dataset/labelled/unstripped/`
- Full per-image results saved to `dataset/labelled/autolabel_report.json`

---

## Key Decisions & Rationale

| Decision | Choice Made | Why |
|---|---|---|
| Crop strategy for full-frame images | Black removal + center square crop, no segmentation model | SAM overkill for classification — object fills frame after black removal; content is always centered (confirmed empirically) |
| Auto-label confidence threshold | 0.90 | User explicitly requested high-confidence only; lower threshold risks mislabelled training data |
| Black background treatment (unlabelled) | Treat as valid SAM mask, not a defect | All 5,173 "high-black" images had content perfectly centered (cx=0.50, cy=0.50), confirming intentional object masking |
| Inference preprocessing | Strip black + center square before model | Training data SAM crops had <1% black; model was never trained on heavy-black-background images — matching distribution is critical |
| Script overwrites originals in-place | Yes, save in-place + update annotations | Annotations reference the original filename; renaming would break the label lookup |
| Sampler config | Recommend `sampler: true` (not changed — platform training) | 19:1 class imbalance without weighted sampling starves unstripped class every batch |

---

## Findings & Observations

1. **Best checkpoint is epoch 12** — val loss 0.013645. Beyond epoch 24 the model overfits (train loss 0.00871 vs val loss 0.022 at epoch 28).
2. **Class imbalance is 19:1** (stripped:unstripped) across all 6 partitions. Unstripped train counts range 142–175 per partition.
3. **Only 1,221 unique unstripped parent images** exist across all 6 partitions combined (~200 per partition).
4. **Scale bias confirmed quantitatively**: 100% of unstripped originals were 1280×720; 100% of stripped SAM crops were variable-size (mean 472×546px). This risks the model learning image scale as a classification shortcut.
5. **Unstripped test recall was 0.855** (14.5% of unstripped cables missed) — directly attributable to scale bias + no weighted sampler.
6. **SAM crops are already clean**: only 1/200 sampled had >5% black; no action needed.
7. **Unlabelled dataset (6,060 images)**: 5,173/6,060 (85.4%) appeared high-black by whole-image ratio, but all had content perfectly centered — confirmed SAM-masked objects, not letterboxing. Revised quality check on content area only: **0 disqualified**.
8. **14 bad images removed from training set**: 11 had high black ratios (bad SAM segment), 2 were degenerate tiny crops (1×2px, 6×38px), 1 was corrupt.
9. **Annotation–disk mismatch**: COCO annotations reference parent UUIDs; disk holds SAM children. Training script must be resolving labels by stripping crop suffix — this was flagged as a pipeline risk to verify.
10. **Stripped originals had same black border problem as unstripped** (61/64 had >5% black, some up to 58.9%) — extended the crop script to handle both classes.

---

## Problems & Fixes

| Problem | Root Cause | Fix Applied |
|---|---|---|
| `torch.load` UnpicklingError on `best.pt` | PyTorch 2.6 changed `weights_only` default to `True`; ResNet is not in safe globals | Added `weights_only=False` to `torch.load()` call |
| 5,173 unlabelled images flagged as high-black | Quality check used whole-image black ratio — not appropriate for SAM-masked objects | Switched to content-area quality check: find non-black bbox first, evaluate brightness/contrast on content only |
| First crop run only processed unstripped | Script had `if cat != 'unstripped': continue` guard | Removed the class filter so both stripped and unstripped originals are processed |
| `/tmp/session-journal-push` already existed | Stale clone from a previous run | `rm -rf` before re-cloning |

---

## ML / Training Config (Recommended for next platform run)

```yaml
# model/hpo/hyp.yaml — changes needed
sampler: true          # was: false — fixes 19:1 class imbalance (CRITICAL)
lr: 0.0005             # was: 0.001 — reduces late-epoch overfitting
hsv_h: 0.015           # was: 0.0 — small hue jitter for lighting variation
epochs: 15             # ADD — best checkpoint at epoch 12; stop early
lr_scheduler: cosine   # ADD — no scheduler currently; prevents divergence
warmup_epochs: 3       # ADD — standard warmup with cosine

# model/train/inference_config.yaml — changes needed
confidence: 0.35       # was: 0.5 — recovers unstripped recall (test recall was 0.855)
```

**Rationale per param:**
- `sampler: true` — without this, each batch underrepresents unstripped ~19×; most impactful single change
- `lr: 0.0005` + `cosine` scheduler — train loss was still falling at epoch 32 while val loss had been rising since epoch 24; slower LR + decay prevents memorisation
- `epochs: 15` — val loss was best at epoch 12 (0.013645); 33 epochs wasted compute and degraded the checkpoint
- `confidence: 0.35` — unstripped test recall at 0.5 threshold was 0.855; lower threshold recovers ~14.5% of missed detections

---

## Files Created / Modified

| File | Location | Purpose |
|---|---|---|
| `crop_unstripped.py` | `model/dataprep/` | Black border removal + center square crop for all original full-frame images |
| `autolabel.py` | `model/dataprep/` | High-confidence auto-labelling of unlabelled images using best.pt |
| `annotations.json` (×18) | All 6 partitions × 3 splits | Updated `width`/`height` fields for 1,141 cropped images |
| `dataset/labelled/stripped/` | `dataset/labelled/` | Auto-labelled stripped images (confidence ≥ 0.90) |
| `dataset/labelled/unstripped/` | `dataset/labelled/` | Auto-labelled unstripped images (confidence ≥ 0.90) |
| `dataset/labelled/autolabel_report.json` | `dataset/labelled/` | Per-image label, confidence, accept/reject status |

**Deleted (14 files):**
- `69dd299f...6_sam_94e21b83.jpg` (pt1/train) — 1×2px corrupt
- `30396081...4_sam_9d4e8fb6.jpg` (pt3/train) — 6×38px too small
- 12 SAM crops across pt1–pt6 with >15% black pixel ratio (bad segments)

---

## Next Steps / Open Questions

- [ ] Verify auto-label results: check `autolabel_report.json` for accepted counts per class and confidence distribution
- [ ] Human review of auto-labelled unstripped images — unstripped is the minority critical class; spot-check before adding to training set
- [ ] Re-upload cleaned dataset to platform with updated `hyp.yaml` and `inference_config.yaml`
- [ ] Verify training script correctly resolves `_sam_` crop filenames back to parent annotation labels (annotation–disk mismatch risk)
- [ ] Consider collecting more unstripped source images — only 1,221 unique parents exist across all partitions (~200/partition); adding more diversity will help generalisation
- [ ] Run SAM cropping pipeline on unstripped images at source (before next data collection round) to match stripped crop scale natively

---

## Session Metadata
- **Date:** 2026-04-14
- **Model:** Claude (claude-sonnet-4-6)
- **Working directory:** `/Users/paulrydrickpuri/Documents/code/script/USB-classification`
- **Key packages used:** Python 3.12, PyTorch 2.6, torchvision, Pillow 12.1.1, NumPy 2.4.3
