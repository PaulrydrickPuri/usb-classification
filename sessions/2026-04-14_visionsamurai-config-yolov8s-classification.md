# Session Log — 2026-04-14 (Part 3)
**Project:** USB Cable Classification (Stripped vs Unstripped)
**Working directory:** `/Users/paulrydrickpuri/Documents/code/script/USB-classification`
**Duration context:** Continuation — VisionSamurai platform config + skill corrections

---

## Summary

With the balanced COCO dataset uploaded to VisionSamurai (USB-v2 project, 2,438 images), this session focused on producing the correct training configuration for the platform. The intended model was YOLOv8m-cls but the platform only exposes YOLOv8s-cls for classification — so the config was adapted accordingly. The platform's actual UI constraints were also discovered and corrected: SGD is the only available optimizer (not AdamW), the three exact sampler names differ from generic documentation, and Blur/Dropout each have Min/Max/Prob parameters (not just probability). These corrections were written back into the `visionsamurai-training-config` skill reference files so future sessions use accurate platform values.

---

## What Was Built

### 1. YOLOv8s-cls Training Config for USB-v2
Full recommended configuration for the VisionSamurai 5-step Advanced Training flow, covering:
- Model selection (yolov8s-cls — forced by platform availability)
- Data settings, training settings, solver settings
- Optimizer (SGD + Nesterov, corrected from AdamW)
- LR Scheduler (WarmupCosineLR)
- Data Sampler (TrainingSampler — balanced dataset)
- Full augmentation set with Min/Max/Prob values for all parameters

### 2. VisionSamurai Skill — Platform Constraint Corrections
Three reference files updated with confirmed platform constraints:

**`SKILL.md`:**
- Optimizer: changed from "SGD or Adam/AdamW" → SGD only
- Data Sampling: replaced vague "default uniform sampler" with exact 3 platform options
- Universal baseline table: added Blur and Dropout with Min/Max/Prob parameter structure
- Flags section: split sampler guidance into 3 separate lines (balanced / imbalanced / very large)

**`references/model-config.md`:**
- Optimizer section: rewrote entirely as SGD-only with LR guidance by task type
- Data Sampling section: replaced with named platform samplers + decision table

**`references/augmentation-guide.md`:**
- Blur: added parameter descriptions (Min/Max = intensity bounds), updated defaults to Min 0.1 / Max 1.5 / Prob 40%, added macro-specific warning (keep Max ≤ 1.5)
- Dropout: added parameter descriptions (Min/Max = pixel drop fraction), confirmed defaults Min 0.05 / Max 0.10 / Prob 30%, added macro-specific cap guidance

---

## Key Decisions & Rationale

| Decision | Choice Made | Why |
|---|---|---|
| Model | yolov8s-cls (not yolov8m-cls) | Platform only exposes yolov8s for classification; yolov8m-cls not available |
| Optimizer | SGD + Nesterov | Only option on platform; Nesterov always enabled to reduce oscillation |
| LR | 0.001 | SGD fine-tune range for medium dataset; 0.0005 considered but 0.001 standard for SGD |
| LR Scheduler | WarmupCosineLR | Smoother decay than MultiStep for classification fine-tuning; avoids abrupt drops that destabilise SGD |
| Patience | 30 (raised from default 20) | 13.44% train duplicates add noise to val signal; model needs more room before early stopping |
| Weight Decay | 0.0001 (raised from default 0.00001) | Counteracts overfitting risk from 262 duplicate images in train split |
| Eval Interval | 5 (lowered from default 10) | Catch overfitting earlier given duplicate-inflated train signal |
| Sampler | TrainingSampler | Dataset is perfectly 1:1 balanced — uniform sampling is correct choice |
| Blur Max | 1.5 (not 2.0+) | Macro close-up classification — high blur destroys fiber texture detail the model needs to distinguish stripped vs unstripped |
| Dropout Max | 0.10 (not 0.20+) | Keeping drop fraction ≤ 10% preserves class-defining surface texture in each image |
| No Perspective | Excluded | Close-up macro classification — perspective warp distorts the class signal |
| No weather augmentations | Excluded | Indoor/controlled industrial environment; Rain/Fog/Snow not realistic |
| Flip U/D | Included | Macro close-up shots have no gravity-fixed orientation |

---

## Findings & Observations

1. **YOLOv8m-cls is not available** on VisionSamurai for classification — only yolov8s-cls is shown. User confirmed this. Config adapted for yolov8s-cls throughout.
2. **SGD is the only optimizer** on the platform. AdamW is not available despite being theoretically preferred for classification fine-tuning. SGD with Nesterov Momentum and WarmupCosineLR compensates.
3. **Three exact sampler names on platform**: `TrainingSampler`, `RepeatFactorTrainingSampler`, `RandomSubsetTrainingSampler` — prior skill documentation was using incorrect/generic names.
4. **Blur and Dropout have Min/Max/Prob** — not just probability. This is a platform-specific UI detail not previously captured in the skill.
5. **262 train duplicates (13.44%)** flagged from health tab — same-parent SAM crops landing in the same split. Config accounts for this via higher weight decay (0.0001), lower eval interval (5), and higher patience (30).
6. **Images are industrial macro close-ups** — fibrous stripped cable ends (look like palm fruit fibers) vs clean unstripped cable tips. Black SAM backgrounds present on a subset. Model must learn fine surface texture differences.
7. **TrainingSampler is correct for this dataset** — RepeatFactorTrainingSampler is for imbalanced data; since the balanced partition is 1:1, uniform sampling is appropriate.
8. **WarmupCosineLR preferred over WarmupMultiStepLR** for this task — smooth decay is better for fine-tuning on a medium dataset; MultiStep's abrupt drops can cause instability with SGD on classification.

---

## Problems & Fixes

| Problem | Root Cause | Fix Applied |
|---|---|---|
| Config recommended AdamW | Platform only supports SGD | Revised config to SGD + Nesterov; adjusted LR to 0.001 (SGD range) |
| Sampler names were generic/wrong | Skill was written without platform verification | Updated all 3 skill files with exact platform sampler names and decision table |
| Blur/Dropout shown as single probability | Platform UI has Min/Max/Prob for these two | Updated SKILL.md baseline table + augmentation-guide.md with full parameter structure |

---

## ML / Training Config (Final — USB-v2, YOLOv8s-cls)

```
▸ MODEL
  Architecture : yolov8s-cls
  Fine-Tune    : ✅ Yes (ImageNet pretrained)

▸ DATA SETTINGS
  Input Shape  : 640
  Batch Size   : 16

▸ TRAINING SETTINGS
  Base LR      : 0.001
  Epochs       : 200
  Patience     : 30

▸ SOLVER SETTINGS
  Warmup Epochs       : 5
  Evaluation Interval : 5
  Event Interval      : 5
  Checkpoint Interval : 10

▸ OPTIMIZER
  Type              : SGD
  Weight Decay      : 0.0001
  Nesterov Momentum : ✅
  Gradient Clipping : ☐

▸ LR SCHEDULER
  Type    : WarmupCosineLR
  Final LR: 0

▸ DATA SAMPLING
  Sampler : TrainingSampler

▸ AUGMENTATION
  ✅ Brightness  (Min 0.7, Max 1.3, Prob 50%)
  ✅ Contrast    (Min 0.8, Max 1.2, Prob 50%)
  ✅ Saturation  (Min 0.7, Max 1.3, Prob 40%)
  ✅ HSV         (Hue ±10, Sat ±20, Val ±20, Prob 50%)
  ✅ Flip L/R    (Prob 50%)
  ✅ Flip U/D    (Prob 50%)
  ✅ Rotation    (Min 0°, Max 30°, Prob 50%)
  ✅ Blur        (Min 0.1, Max 1.5, Prob 40%)
  ✅ Dropout     (Min 0.05, Max 0.10, Prob 30%)
  ✅ Shadow      (Prob 30%)
  ✅ Lighting    (Prob 40%)
  ☐  Perspective / Scale / Rain / Fog / Snow / Sunflare / Motion Blur
```

---

## Files Created / Modified

| File | Location | Purpose |
|---|---|---|
| `SKILL.md` | `~/.claude/plugins/.../visionsamurai-training-config/` | Updated optimizer (SGD only), sampler names, Blur/Dropout parameter structure |
| `model-config.md` | `.../references/` | Rewrote optimizer section (SGD only), replaced sampler section with named platform options + decision table |
| `augmentation-guide.md` | `.../references/` | Added Min/Max/Prob parameters for Blur and Dropout, updated defaults, added macro-specific guidance |

---

## Next Steps / Open Questions

- [ ] Launch training run on VisionSamurai with the config above (yolov8s-cls, 200 epochs, patience 30)
- [ ] Monitor train vs val loss curve — watch for overfitting signal from the 262 duplicates
- [ ] After training: re-run `autolabel.py` on **raw (non-processed) unlabelled images** using the new yolov8s-cls model
- [ ] Patch `package_balanced_coco.py` to enforce parent-level deduplication (no two SAM crops from same parent in same split) — removes the 13.44% duplicate rate
- [ ] Evaluate whether yolov8s-cls achieves the unstripped recall target (≥ 0.93 vs ResNet34's 0.855)

---

## Session Metadata
- **Date:** 2026-04-14
- **Model:** Claude (claude-sonnet-4-6)
- **Working directory:** `/Users/paulrydrickpuri/Documents/code/script/USB-classification`
- **Key packages used:** Python 3.12, PyTorch 2.6, Pillow 12.1.1, NumPy 2.4.3
- **Platform:** VisionSamurai (app.visionsamur.ai)
