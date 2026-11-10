# YOLOv8m-seg + augmentation on Kaggle

**Publication fix:** Kaggle was extracting uploaded checkpoint ZIPs, explaining
why only `backup_manifest.json` and `run_settings.json` appeared. Backups now use
an armored `.zip.pcb64` object, automatically decoded to the exact original ZIP.
The publisher logs/checks staging contents and runs `datasets files` before
read-back; missing payloads get a specific error. See
[KAGGLE_PUBLICATION_FIX.md](KAGGLE_PUBLICATION_FIX.md) for wire format, CLI 2.x /
`KAGGLE_API_TOKEN` setup and the opt-in live nonempty-ZIP integration test.
Model/training/data/evaluation code is unchanged. Live private-Dataset verification
still requires your authenticated Kaggle runtime.

Open `YOLOv8m_Seg_GraphPCB_Kaggle.ipynb` in Kaggle using **Import Notebook**.
The notebook contains every source file and can run independently of the RTMDet
folder. Add the GraphPCB Kaggle dataset (or your uploaded `archive.zip`) as an input,
enable a GPU accelerator and Internet, then run the cells in order.

**Persistent backup setup is now required by the notebook.** Read
[PERSISTENT_BACKUPS.md](PERSISTENT_BACKUPS.md): configure a dedicated private
Kaggle Dataset and Secrets, or an explicit private S3 destination. The notebook
performs a real upload/read-back preflight and restores verified state
automatically. `/kaggle/working` alone is not persistent protection.

Use `Backup_Recovery_Smoke_Kaggle.ipynb` to verify two epochs, remote backup,
fresh-session restore and continuation from epoch 3 before a long experiment.
**Live Kaggle/S3 persistence has not been verified by the author.** Local/mock
failure tests and a real CPU model recovery test passed; see TESTING.md.

The persistent read-back wrapper now captures sanitized CLI diagnostics, uses
non-quiet downloads by default, and retries transient/propagation failures within
explicit limits. It still requires exact snapshot/manifest checksums before
confirming a backup. For a metadata-200/manual-download-success but preflight
failure, see [KAGGLE_DOWNLOAD_TROUBLESHOOTING.md](KAGGLE_DOWNLOAD_TROUBLESHOOTING.md)
and the notebook's optional read-only diagnostic cell. That cell compares the
exact old `-o -q` invocation with `-o`, without uploading or deleting versions.

This is a second **segmentation** experiment for comparison with the existing
RTMDet-Ins-tiny experiment. Classification remains outside this folder's scope.
No accuracy, relative marks or ranking is promised before training.

## Training speed update

The Kaggle notebook keeps **YOLOv8m-seg**, all three trials, their original
resolutions, 60 epochs each, augmentation, effective batch size and validation
protocol. The three trials are separate experiments, not three model layers.
The parameter counts are unchanged; this update improves training execution.

- Exact, unaugmented uint16 instance masks are decoded once and cached in
  `/kaggle/temp/graphpcb_prepared/mask_cache_v1`. Annotation content hashes
  invalidate old cache entries. The first epoch fills the cache. Cache files
  consume temporary disk space and are not included in the saved results.
- Instance boxes and surviving mask IDs are reconstructed with one image scan
  and an ID lookup, replacing a full image scan for every component. Mask holes
  and small regions are retained exactly.
- Training workers persist between epochs and prefetch batches. The sampler
  sends the current epoch to each worker, so random augmentation changes every
  epoch and uses the same per-sample seeds after resume.
- CUDA uses fused AdamW, existing AMP, cuDNN autotuning and TF32 where supported.
  Loss totals stay on the GPU until logging or epoch completion, reducing CPU
  synchronisation. TF32/autotuning can change rounding and repeatability;
  this is not a guarantee of identical trained weights or accuracy.
- `work_dirs/<trial>/history.json` includes training seconds and images/second
  per epoch. Validation and checkpoint time are excluded from this throughput;
  total runtime still includes the full trial. Compare after the first epoch.

Use the **updated notebook**: it embeds the updated source, so editing a separate
Python file beside an older imported notebook does not update that notebook.
For a fresh run the default name ends in `fast_60e_seed42`. To continue an existing
medium-model run, attach its saved outputs, set `RESUME_FROM` and use its original
`RUN_NAME`, epochs, batch size, accumulation and seed. The updated notebook retains
the old source, records a source-upgrade snapshot, and restores optimizer/model/
EMA/scaler state from `last.pt`. Old source checksums can change for this upgrade;
experiment settings and the trainer's saved config still must match.

Start with `WORKERS=2`; try 4 if CPU loading limits GPU utilisation. Increase
physical batch size only if memory permits and reduce accumulation to keep their
product at 8. A changed batch setting requires a new run. GPU speedup depends on
the bottleneck and hardware; no Kaggle runtime or accuracy improvement is claimed
without measuring a full GPU run.

## Model details

The code constructs the official Ultralytics **YOLOv8m-seg** model with a 25-class
segmentation head, initialized from `yolov8m-seg.pt` COCO weights. Its backbone is
YOLOv8's CSP-style **Conv/C2f/SPPF** design, commonly described as a modified
CSPDarknet backbone. It does not replace that backbone with a separately defined
Darknet-53 network. The medium scale is checked explicitly, preventing an
accidental nano/small model. `ultralytics==8.3.161` pins the exact model/loss APIs.

The model is substantially larger than RTMDet-Ins-tiny. The 25-class model has
27,254,123 total parameters and 27,254,107 trainable parameters before additional
backbone freezing (16 DFL projection weights are inherently fixed). Freezing the
backbone leaves 15,398,251 trainable parameters. The scripts measure these counts
for every trial; augmentation does not add
parameters. A frozen-backbone ablation reduces trainable parameters while
retaining the same medium architecture. It is not an architecture compression.

| Trial | Input size | Freeze policy | Augmentation |
|---|---:|---|---|
| yolov8m_640_aug | 640 | None, except YOLO's inherently fixed DFL projection | Yes |
| yolov8m_960_aug | 960 | Same | Yes |
| yolov8m_960_aug_frozen_backbone | 960 | First 10 modules, covering the backbone | Yes |

All use synchronized random rotation +/-15 degrees, scale 0.75–1.25, translation
up to 10%, horizontal/vertical flips, and modest HSV changes. Original stored
image colors are retained before augmentation. Mosaic and MixUp are disabled
because this experiment mixes pixel-annotated and box-only samples. No external
PCB images, ensembles or test-time training are added.

## Exact masks and source-aware training

The supplied GraphPCB archive has Graph-F 115/47 and Graph-W 37/10 official
train/test boards. Graph-F provides RGB semantic masks, from which instances are
recovered as 8-connected regions of each exact source class color. Touching
same-class objects can remain merged and disconnected fragments can become
separate instances. These are **reconstructed-instance metrics**, not evaluation
against original instance identities that are absent from the ZIP.

Graph-W has VOC XML boxes only. It contributes the YOLO box, class and DFL losses;
its segmentation loss is disabled. Its box rectangles are not treated as true
pixel masks. The custom loss subclasses the official YOLOv8 segmentation loss
and masks out Graph-W positives **only for the segmentation term**, adding zero
parameters. Tests check that changing Graph-W mask pixels cannot change mask loss.
Graph-W true mask AP/F1 and the full-assignment combined segmentation score stay
`null`. Masks and overlays are still predicted on every official test board.

Training uses the exact COCO RLE masks, including holes and small regions, through
a custom PyTorch loader. There is no lossy conversion to YOLO polygon text labels.
Nearest-neighbor transforms preserve mask IDs. The YOLO prototype loss internally
uses its standard lower-resolution prototype masks. Instances that disappear
under a training resize/crop are removed from that transformed sample; full-board
evaluation keeps all supplied recovered ground-truth regions.

The standard model performs the forward pass; the custom loop controls only data,
mixed supervision, optimizer, EMA, validation and checkpointing. Loss terms are
the official box/seg/class/DFL terms, using gains 7.5/7.5/0.5/1.5. Optimization uses
AdamW, learning rate 0.0002 with three-epoch warmup and cosine decay, weight decay
0.05 on matrix/kernel weights (zero on bias and normalization affine parameters),
AMP on GPU, effective batch size 8 through accumulation, clipping at 10,
and Ultralytics EMA. The summed official loss vector is divided by physical batch
size before gradient accumulation. These choices are recorded in each trial config.

## Matching the RTMDet data and evaluation protocol

The data converter, canonical taxonomy/source mapping, class-preserving board
split and COCO/F1 metric functions are reused from the RTMDet implementation.
Official membership comes from graph filenames, without unpickling graph files.
The graph files' four-class labels are not used as the 25-class targets.

With seed 42 and the supplied ZIP, the split is 100 Graph-F + 31 Graph-W training
boards, 15 + 6 validation boards, and the unchanged 47 + 10 test boards. Every
available Graph-F class retains training mask support. Front/back views are grouped
before tiling when board names identify them. Test annotations never select
hyperparameters, checkpoints, thresholds or cleanup rules.

Training tiles are 1024 with 256 overlap. The source RGB mask palette is decoded
exactly; output overlays use the assignment's separate 25-color display palette.
The full mapping and ignored Graph-W source labels are written to
`data_mapping.json`. U and IC remain distinct, as do S and SW. Ambiguous/unmapped
source types are excluded rather than guessed.

Every fifth epoch evaluates Graph-F validation tiles. The best epoch by **COCO
tile mask mAP** is retained. Its global confidence threshold is then tuned on full
validation boards using micro instance F1 on the 101-point [0,1] grid. Trial
selection uses validation `S_seg`, breaking ties by fewer trainable parameters.
`selection_frozen.json` is saved before the entire test partition is evaluated.

Mask AP50/mAP50:95 use pycocotools with standard maxDets=100. A separate dense
maxDets=1000 diagnostic is reported. F1 is one-to-one same-class, score-ordered
instance matching at mask IoU >= .50; both micro F1 and fixed-25-class macro F1
are reported. Missing-class AP is `null`, not a fabricated zero. The reported
`S_seg=.2*AP50+.5*mAP50:95+.3*F1_mask` is explicitly **Graph-F only**.

Inference uses YOLO's official NMS/native mask reconstruction per tile, then the
same tile-center ownership and board-level mask NMS used for RTMDet. Candidate
floor .001, per-tile cap 300 and board cap 5000 are recorded. F1 at threshold zero
means all retained candidates. Predicted mask bounding boxes support the
additional box metrics, matching the existing RTMDet evaluation convention.

## Kaggle setup, saved outputs and resume

1. Import the `.ipynb`. Enable a single NVIDIA GPU and Internet in notebook settings.
   A dual-GPU allocation is supported, but this script uses GPU 0 only.
2. Add the GraphPCB dataset as an input. `DATA_SOURCE=''` auto-detects a directory
   directly containing Graph-F/Graph-W, or a ZIP containing the same structure.
   If multiple candidates exist, set DATA_SOURCE explicitly to the printed path.
3. Leave `PRETRAINED_WEIGHTS='yolov8m-seg.pt'` for the official online download,
   or point it at a checkpoint added as a Kaggle input. These are initialization
   weights, not a completed assignment checkpoint.
4. Setup retains Kaggle's existing matched CUDA PyTorch/torchvision pair and tests
   a real CUDA NMS operation. Unlike RTMDet, no MMCV wheel or separate Colab Python
   environment is required. Training/evaluation run as separate Python processes.
5. Run preparation and the preflight. The preflight exercises the actual medium
   model with random weights at 128 pixels to check the pipeline quickly; final
   trials use the configured 640/960 inputs and pretrained weights.
6. Run all three trials, normally 60 epochs each. `SESSION_HOURS=6` and
   `SHUTDOWN_BUFFER_SECONDS=900` reserve time for final upload. Checks occur
   between batches and during validation. An unfinished epoch is discarded and
   repeated from the last committed checkpoint after restore. Set a budget below
   the actual quota/session time Kaggle shows for your account.
7. At an incomplete run, confirmed remote snapshots enable automatic recovery.
   Saving notebook outputs is optional for that path, but useful for plots and
   legacy copies. For unattended execution use Save Version / Save & Run All.
   Session limits and quota still apply. Check the explicit remote verified
   status rather than assuming browser disconnect or local files imply a backup.
8. In a later run, the configured provider is downloaded and verified
   automatically; keep RUN_NAME/destination and set `ALLOW_NEW_RUN=False`.
   Settings are restored automatically. As an optional legacy path, add the
   previous notebook outputs as an input and set
   `RESUME_FROM` to the previous run directory (the directory containing
   `results/segmentation`). Keep RUN_NAME, epochs, seed and batch settings
   unchanged. Trainer source upgrades are recorded as described above. The notebook copies the saved run to `/kaggle/working` and resumes
   model/optimizer/scaler/EMA state at the next epoch. Already validated trials
   are skipped. The incomplete current epoch may need to be repeated after an
   abrupt runtime termination; completed epochs have atomic checkpoint saves.
9. After training is complete, run test evaluation and reporting once. Do not
   retune using the test scores. Download the portable results ZIP from the output.

Prepared boards/tiles are stored in `/kaggle/temp`, so dataset copies do not inflate
saved notebook outputs. The run, sources, manifests, optimizer checkpoints and
results live in `/kaggle/working`. Preparation is repeated after a fresh session.
The background worker uploads committed training state hourly to the configured
remote provider and retains previous verified generations. Save/download notebook
outputs for convenient access to plots and test overlays as well.

If batch 2 at 960 runs out of GPU memory, choose **a new run name**, batch 1 and
accumulation 8 for all trials. Do not silently reduce resolution only for one
model. Use the same seed, epochs and corresponding input resolution when assessing
the models. For a five-epoch pipeline check, use a separate run name and record
it as a smoke experiment rather than a strong final accuracy result.

## Outputs and inference

After real training and evaluation, `results/segmentation` contains the consistent
`segmentation_stats.json`, frozen selection, model checkpoints, loss history,
PR/F1/classwise plots, full test predictions and translucent overlays. The report
also writes `completion_audit.json` for missing assignment requirements.
`work_dirs/<trial>/last.pt` is the resumable optimizer checkpoint. The portable
`checkpoints/best_segment.ckpt` contains a standard model YAML and state dictionary,
without custom pickled model/loss classes. It is not an ordinary `YOLO('...pt')`
checkpoint; use the supplied standalone inference script.

```bash
python inference_segmentation.py --checkpoint /path/best_segment.ckpt \
  --images /path/board.jpg --out /path/predictions
# Or:
bash scripts/run_segmentation.sh --checkpoint /path/best_segment.ckpt \
  --images /path/images --out /path/predictions
```

Only that Python file, the checkpoint and the installed external packages are
needed for inference. No project-specific modules are imported by the file.
The final portable ZIP excludes datasets and `last.pt` optimizer snapshots; keep
the saved Kaggle notebook outputs when you need to resume.

## Compare against RTMDet

Add the existing RTMDet `segmentation_stats.json` and `manifest.json` as Kaggle
inputs, then fill the optional comparison cell. The comparison script verifies
the exact train/validation/test board membership, seed, tile size and overlap,
then emits CSV/JSON with accuracy, parameter counts, thresholds and runtime for
all trials. It marks each architecture's **validation-selected** trial; do not
select the highest test-scoring configuration after the fact.

The recipes use different optimizers/losses/augmentation details, so this compares
complete training recipes rather than isolating architecture alone. Runtime from
Colab versus Kaggle, or different GPUs/epoch budgets, is not a controlled speed
comparison. Run both on the same hardware for credible speed comparisons. The
cohort parameter median is unknown; larger YOLO models do not automatically earn
more marks. Compare S_seg and trainable parameters together. The comparison cell
does not train or port the RTMDet implementation to Kaggle.

## Assignment limits

This implements segmentation only. True Graph-W pixel annotations and original
Graph-F instance identities remain unavailable. The classification exercise,
group-specific private repository, combined submission ZIP, both task checkpoints
and final submission organization still require separate work. No submission email
or repository changes are performed by this notebook.

## Sources

- [Official YOLOv8 model documentation](https://docs.ultralytics.com/models/yolov8/).
- [Pinned model YAML / CSP-style C2f backbone](https://github.com/ultralytics/ultralytics/blob/v8.3.161/ultralytics/cfg/models/v8/yolov8-seg.yaml).
- [Official segmentation loss implementation](https://github.com/ultralytics/ultralytics/blob/v8.3.161/ultralytics/utils/loss.py).
- [Ultralytics model and checkpoint code](https://github.com/ultralytics/ultralytics/blob/v8.3.161/ultralytics/nn/tasks.py).
- [Ultralytics licensing](https://www.ultralytics.com/license): package release uses AGPL-3.0.
- [GraphPCB dataset / official partitions](https://github.com/BlueSkyLT/GraphPCB).
- [PCBSegClassNet source mask palette](https://github.com/CandleLabAI/PCBSegClassNet/blob/main/src/data/create_mask.py).
- [COCO metric implementation](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py).
- [Kaggle notebook documentation](https://www.kaggle.com/docs/notebooks).

Model layers, heads, loss implementations, pretrained weights, assignment and EMA
come from Ultralytics. Project code adds exact-mask data loading/augmentation,
source-aware loss routing, resumable experiments, comparison and assignment reports.
Data conversion and metric definitions match the existing RTMDet code in this workspace.
