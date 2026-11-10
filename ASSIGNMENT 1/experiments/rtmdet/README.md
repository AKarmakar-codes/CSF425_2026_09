# GraphPCB instance segmentation in Google Colab

Open `RTMDet_Ins_tiny_GraphPCB_Colab.ipynb` in Colab and run its cells in order.
The notebook is self-contained: it writes every source file, installs an isolated
Python environment, mounts Drive, prepares the supplied ZIP, runs a GPU preflight,
trains three configurations, freezes validation selections, evaluates all test
boards, generates plots, and packages the outputs. No GitHub download of this
project is required. Enable a GPU runtime before installation.

## Scope and limits

This implements Exercise 2 (segmentation), as requested. It does **not** implement
Exercise 1 classification or constitute a complete 30-mark assignment submission.
The code has not been trained or benchmarked here: run the notebook to obtain
actual checkpoints, parameter counts, losses and scores. No scores are fabricated.

The supplied ZIP has 115/47 Graph-F train/test images and 37/10 Graph-W train/test
images. Graph-F provides RGB semantic masks, not individual instance identities.
The code recovers instances as 8-connected regions of each exact class color.
Touching objects of the same class can remain merged, and separate fragments can
become separate instances. Consequently, report these as **reconstructed-instance
metrics**, not metrics against original per-instance annotations.

Graph-W provides Pascal VOC XML **boxes only** (`segmented=0`). It supplies class
and box losses in a mixed-supervision training batch. Rectangular masks exist only
as loader placeholders, and the custom head excludes every Graph-W sample from
mask loss. There is no pseudo-mask pixel supervision. True Graph-W mask AP/F1
and full Graph-F+Graph-W segmentation scores are `null`, with the reason recorded
in every report. The model still predicts masks and renders overlays on **all 57
official test boards**; Graph-W evaluation reports only box AP. Original pixel
annotations would be needed to satisfy the full mask-evaluation requirement.

## Architecture and parameter efficiency

The model uses MMDetection 3.3.0's official RTMDet-Ins-tiny: CSPNeXt-P5 with
depth factor 0.167, width factor 0.375, CSPNeXtPAFPN with 96 output channels, and
RTMDetInsSepBNHead with a dynamic instance-mask branch. The final class head has
25 outputs rather than COCO's 80. The official 80-class model is approximately
5.6 million parameters; the notebook measures the exact 25-class total and
trainable counts. The routing subclass adds **zero** parameters and is replaced
by the standard head in the portable checkpoint.

Three trials isolate practical choices:

| Trial | Network input | Frozen backbone | Purpose |
|---|---:|---|---|
| tiny_640 | 640 | None | Baseline |
| tiny_960 | 960 | None | Preserve more small-component detail without adding parameters |
| tiny_960_frozen2 | 960 | Stem + stages 1 and 2 | Reduce trainable parameters; measure any accuracy tradeoff |

All use COCO-pretrained RTMDet-Ins-tiny weights, AdamW, a warmup/cosine schedule,
mixed precision, accumulated gradients, gradient clipping, EMA, training-only
class-balanced tile repetition, random resizing/cropping, HSV augmentation, and
horizontal/vertical/diagonal flips. Mosaic and MixUp are omitted because they
would mix valid-mask and box-only examples. Frozen layers stay frozen throughout
that trial; total and trainable parameters are both reported. Resolution and
freezing are ablations, not promised performance improvements.

Pretrained COCO weights are disclosed separately from additional PCB training
data. **No external PCB images** are added. Geometric/color augmentation and
overlapping tiles are transformations of the original boards. The manifest records
the actual distinct training boards out of 152 official training boards; the
remainder are held out for validation by board group. The split retains at least
one training group for every class present in each source. If the instructor
counts all official boards available rather than only those used, the ratio is
1.0. Either interpretation is below the 1.5 augmentation-penalty boundary.

The grading efficiency is only a modest multiplier:
`E = clip(1 + 0.05*(log10(P_median)-log10(P_trainable)), 0.90, 1.10)`.
Do not sacrifice a large amount of accuracy merely to freeze more parameters.
The cohort median and other groups' scores are unknown, so exact marks cannot be
predicted. Trial selection uses validation S_seg, with smaller trainable parameter
count breaking a tie; this does not claim to optimize the unknown cohort utility.

## Data mapping and leakage prevention

Canonical IDs 1..25 are `R,C,U,Q,J,L,RA,D,RN,TP,IC,P,CR,M,BTN,FB,CRA,SW,T,F,V,LED,S,QA,JP`.
The **source** RGB mask palette comes from PCBSegClassNet's `create_mask.py`;
the **display** palette uses the assignment's hex colors. These are different.
Every ID, source RGB tuple and display hex value is written to `data_mapping.json`.
Unknown mask colors cause an error rather than a nearest-color guess.

Graph-W uses source component types, not the coarse four-class graph labels.
Examples: resistor -> R; electrolytic capacitor -> C; ic -> IC; transistor -> Q;
connector -> J; resistor network -> RN; test point -> TP; ferrite bead -> FB;
switch -> SW; jumper -> JP. U and IC remain separate, as do S and SW. Unmapped or
ambiguous types (text, pins, unknown, clock, capacitor jumper, etc.) are excluded
from the 25 target classes and listed in the mapping audit. Ambiguous source labels
are never merged on visual intuition. Boxes use documented 1-based inclusive VOC
to 0-based half-open coordinate conversion. Stored image colors are preserved.

Official membership is read from `.pt` **filenames only**, without unpickling the
graphs. Their labels are only four-class labels and are unsuitable for this task.
Within official training, validation is split by board group before tiling, keeping
front/back views together when names identify them. A training-label-only scan
prevents validation from taking the last training group supporting any class.
The manifest records class support and actual fold sizes. Official test membership is
unchanged. No test labels select checkpoints, trials, thresholds or cleanup rules.
Exact byte duplicates across official train/test cause preparation to stop.
Any same-hardware relationships already present in the official partition are
preserved, not silently reassigned.

## Colab use

1. Put `archive.zip` in Google Drive (about 6.09 GB; prefer Drive over browser upload).
2. Upload/open the `.ipynb` in Colab and choose a GPU runtime (for example T4).
3. Edit `ZIP_PATH` and `DRIVE_RUN_ROOT` in the settings cell. Leave `RUN_NAME`
   unchanged only when resuming the same experiment and settings.
4. Run installation, extraction/conversion, and preflight. The setup uses a
   separate Python 3.10 interpreter so it does not depend on Colab's kernel Python
   version. It installs a precompiled MMCV CUDA wheel; no MMCV source build or
   kernel restart is needed. All training commands use that interpreter.
5. Run all three trials. The default is 60 epochs **per trial**; this can take many
   hours and exceed one free Colab GPU quota. Start with 5 epochs to validate the
   workflow in a separate run directory; such a smoke run is not a strong final
   experiment. Use the same epoch budget across trials for a controlled comparison.
6. If GPU memory runs out, set batch=1 and accumulation=8 for **all trials** in a
   new run. Do not change the input resolution silently. Stop and restart the
   process, then resume from its saved training checkpoint.
7. After all trials validate, run test evaluation and reporting. Once the test set
   has been evaluated, do not use the results to retune the selected model.
8. Download the generated segmentation package; it excludes the dataset archive.

Results, resumable optimizer checkpoints and source snapshots are stored under
`DRIVE_RUN_ROOT/RUN_NAME`; prepared tiles are kept on Colab local disk for speed.
On a new runtime, rerun setup and preparation from the same ZIP, settings and seed,
then rerun training with `--resume`. Already validated trials are skipped.
For changed seeds/settings, always choose a new run name.

Equivalent Linux commands after setup (replace paths):

```bash
/content/pcb-venv/bin/python src/segmentation/prepare_data.py --zip /path/archive.zip --out /content/pcb-data
/content/pcb-venv/bin/python src/segmentation/smoke_test.py --data /content/pcb-data
/content/pcb-venv/bin/python src/segmentation/train.py --data /content/pcb-data --out /path/results --epochs 60 --resume
/content/pcb-venv/bin/python src/segmentation/evaluate.py --data /content/pcb-data --out /path/results
/content/pcb-venv/bin/python src/segmentation/report.py --data /content/pcb-data --out /path/results
```

## Checkpoint and threshold selection

Each trial retains the epoch with highest held-out Graph-F **tile** COCO mask mAP.
That checkpoint is then evaluated with tiled inference on full validation boards.
A global confidence threshold is chosen on the 101-point grid [0,1] to maximize
validation micro instance-mask F1. The largest threshold breaks a tie. The three
trials and their thresholds, the selected trial and the validation rule are written
to `selection_frozen.json` **before test evaluation**. The selected model is not
chosen by test scores. No ensembles or test-time training are used.

## Metric definitions and outputs

Mask AP50 and mAP@[.50:.95] use pycocotools' COCO definitions, maxDets=100, and
only Graph-F recovered instances. AP for an absent class is N/A (`null`), not an
invented zero. A separate maxDets=1000 result shows the effect of dense boards;
it is **not substituted** for standard COCO AP in S_seg. Classwise AP is COCO's
supported-class convention; fixed-25-class macro F1 uses zero for undefined classes.

`F1_mask` is **micro instance F1**, matching predictions to same-class ground-truth
masks once, in descending score order, at mask IoU >= .50. Unmatched predictions
are FP; unmatched instances are FN. It is not pixel Dice. A fixed-25-class macro
F1 is reported too. `S_seg = .20*AP50 + .50*mAP50:95 + .30*F1_mask` is reported
with the explicit Graph-F-only reconstructed-instance scope. It is **not** the
unavailable full-assignment segmentation score. If the instructor specifies a
different meaning of mask F1, agree on that definition before final evaluation.

All input tiles are 1024 pixels with 256 overlap. Network inputs are resized per
trial. Inference assigns overlapping tile predictions to tile-center ownership
regions, applies class-aware mask NMS at .50, and caps at 5000 detections/board.
The model's candidate floor is .001 and its per-tile cap is 300. PR/F1 curves at
threshold zero therefore mean all **retained** candidates, not unlimited logits.
Overlays show predictions above the validation-frozen threshold; prediction JSON
retains lower-confidence candidates for AP. No boxes/masks are manually cleaned.

Generated files include:

- `segmentation_stats.json`: consistent trial schema with seed, epochs, parameter
  counts, losses, validation/test metrics, threshold, data ratio, runtime and checkpoint.
- `<trial>/validation/` and `<trial>/test/`: prediction JSON, metric JSON, curves,
  COCO precision-recall/F1 plots, and classwise AP/F1 plots.
- `<trial>/test/overlays/*.png`: translucent predicted masks on every test board.
- `<trial>/losses.png`, `trial_comparison.png`, `completion_audit.json`.
- `checkpoints/<trial>.ckpt`, `checkpoints/best_segment.ckpt`: portable inference
  checkpoints after training. They are generated artifacts, not included before training.
- `work_dirs/<trial>/`: exact resolved config, history, hardware/software logs,
  resumable training snapshots. Validation loss uses the entire Graph-F tile
  validation set with EMA weights and without updating BN statistics. Its task
  scope is recorded; it need not numerically match mixed-domain training loss.

For multiple seeds, use separate run directories and report mean +/- standard
deviation of validation/test metrics without picking a favorable test seed.

## Standalone inference

The exported `.ckpt` embeds the full model and preprocessing configuration. The
single `inference_segmentation.py` contains construction, forward orchestration,
tile preprocessing/postprocessing, RLE masks, metrics and overlays, and imports
only standard-library/external packages. It has no imports from project modules.

```bash
PCB_PYTHON=/content/pcb-venv/bin/python bash scripts/run_segmentation.sh \
  --checkpoint /path/checkpoints/best_segment.ckpt \
  --images /path/new_board.jpg --out /path/predictions
```

For a hidden inference run, only the installed environment, that Python file,
the shell script, the checkpoint and input images are needed. Annotation files and
the training repository are unnecessary. `--coco` with `--data-root` optionally
evaluates a documented COCO file, using the already frozen threshold.

## Assignment items still outstanding

The assignment also requires classification source, at least three classification
trials, classification plots/JSON/checkpoint and standalone inference. These are
outside this segmentation request. The all-or-nothing organization marks require
both exercises, exact group-specific naming, a private repository, both readable
checkpoints, a combined submission archive, and the prescribed email process.
The provided packaging cell creates a **segmentation package**, not a falsely
complete assignment submission. Set the group ID and combine both exercises
before applying the PDF's exact final naming convention.

## Sources and attribution

- Assignment PDF supplied by the user (CS F425, Project Assignment I, Semester I 2026-27).
- [GraphPCB dataset and official partitions](https://github.com/BlueSkyLT/GraphPCB).
- [PCBSegClassNet source mask palette](https://github.com/CandleLabAI/PCBSegClassNet/blob/main/src/data/create_mask.py).
- [Official RTMDet configurations, weights and paper citation](https://github.com/open-mmlab/mmdetection/tree/v3.3.0/configs/rtmdet).
- [RTMDet paper](https://arxiv.org/abs/2212.07784).
- [MMDetection](https://github.com/open-mmlab/mmdetection), Apache-2.0.
- [MMCV binary installation documentation](https://mmcv.readthedocs.io/en/2.x/get_started/installation.html).
- [Official CUDA 12.1 / PyTorch 2.1 MMCV wheel index](https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html).
- [pycocotools COCO evaluation implementation](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py).

The model implementation, operators, losses and base recipe come from installed
OpenMMLab packages; the project adds source-aware loss routing, data conversion,
the three trials, full-board inference, audits, and reporting. Cite pretrained
weights and adapted framework configuration in the final submission.
