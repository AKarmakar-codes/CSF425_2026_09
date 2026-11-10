# CS F425 Deep Learning - Assignment I

**Group G09** | BITS Pilani, Hyderabad Campus | Semester I, 2026-27

| Member | Student ID |
|---|---|
| Yashvardhan Singh | 2024A7PS0116H |
| Agnibh Karmakar | 2024B4A70837H |

This package combines PCB component classification and instance segmentation over the assignment's fixed 25 classes. It contains the available source, historical classification evidence, checkpoint-backed segmentation experiments, inference scripts, plots and provenance.

**Checkpoint update:** all four classification model weights and the YOLOv8s weights have now been incorporated. Both mandatory best-checkpoint files exist. Classification exports preserve every original model tensor while omitting optimizer state; the standalone loader supports the explicitly declared historical preprocessing contract. The supplied best classifier is ConvNeXt-Tiny with saved threshold **0.57**. Its saved validation F1/threshold differ from the older report (F1 0.820882 / threshold 0.48), so historical performance is not certified for these supplied weights. This remaining experimental discrepancy is detailed in `SUBMISSION_STATUS.md`; it does not prevent archiving the complete set of available files.

## Submission naming

- Final private repository: `CSF425_2026_G09`. Its URL has not been supplied. The reference repository `AKarmakar-codes/DeepLearning` is source material, not the final repository.
- Final archive: **`CSF425_2026_G09_Assignment-I.zip`**, containing one top-level folder `CSF425_2026_G09/`.
- One group member sends one email with subject **`[CSF425_2026_G09_Assignment-I]`** and exactly one archive attachment.
- No email has been sent and no final repository has been created or updated during preparation.

The underscores above are verified against the rendered assignment PDF, `docs/assignment_guidelines.pdf`. The user-provided group ID is G09.

## Organization

```text
README.md, config.yml, data_mapping.json, submission_metadata.json
classification_stats.json, segmentation_stats.json
selection_classification.json, selection_segmentation.json
inference_classification.py, inference_segmentation.py
run_classification.sh, run_segmentation.sh
src/classification/             Repaired classification train/evaluate/report code
src/segmentation/               Medium-model prepare/train/evaluate/report code
scripts/                       Launchers, verification, reporting and packaging
checkpoints/                   Actual available inference checkpoints
results/classification/        Four historical trial reports and supplied plots
results/segmentation/          Saved metrics, loss histories, curves and overlays
experiments/source_repository/ Historical source matching the repository reports
experiments/rtmdet/             RTMDet baseline source, separate runtime
data/                          Derived annotation manifests and official split identities
evidence/                      Original conflicting reports and recorded environments
docs/                          Assignment, mapping audit and provenance
```

No raw dataset archive, dataset image dump, private SSH key, backup credential, local dependency installation or synthetic trained checkpoint is included. Derived annotation manifests are retained for split reproducibility. Large checkpoints are excluded by `.gitignore` from the private code repository but included in this local submission folder.

## Selection and results

Model selection uses validation only. Classification: `S_cls = .20 P_macro + .20 R_macro + .60 F1_macro`. Segmentation: `S_seg = .20 AP50 + .50 mAP50:95 + .30 F1_mask`. Cohort median model size and other submissions are unknown; no efficiency-adjusted grade or rank is claimed.

### Segmentation: checkpoint-backed trials

| Configuration | Total / trainable parameters | Epoch budget / chosen checkpoint | Validation S_seg | Saved test S_seg | t* |
|---|---:|---:|---:|---:|---:|
| YOLOv8m, augmented, 640 | 27,254,123 / 27,254,107 | 60 / 25 | 0.420117 | 0.240452 | 0.39 |
| YOLOv8m, augmented, 960 | 27,254,123 / 27,254,107 | 60 / 45 | 0.391282 | 0.252894 | 0.48 |
| RTMDet-Ins-tiny, 640 | 5,622,182 / 5,622,182 | 5 / 5 | 0.354716 | 0.243259 | 0.43 |

`checkpoints/best_segment.ckpt` is an exact copy of the validation-winning medium-640 export. Both medium checkpoint hashes match the original experiment records. Medium-960 has the higher saved test score, but test results are not used for selection. RTMDet is a five-epoch baseline, not a matched-budget speed comparison.

The selected primary three remain medium-640, medium-960 and RTMDet-tiny. The newly supplied YOLOv8s checkpoint contains a **25-epoch** history and 11,799,771 parameters; it is not the missing 70-epoch model behind the local reported score 0.355455. The source-repository 25-epoch report has S_seg 0.286543; its final embedded training metrics also differ slightly from that report. The actual YOLO-s checkpoint and embedded history are retained as supporting evidence under `checkpoints/supporting/` and `evidence/received_yolov8s*`. The archive's file called best_segment.ckpt is a duplicate of these YOLO-s weights and has not overwritten the validation-winning medium-640 checkpoint. YOLO-n weights were not present. Historical constant losses, duplicated AP/F1 arrays and zero trainable-parameter fields are not promoted to verified primary results. `selection_segmentation.json` preserves the reported comparisons and exclusions.

The medium frozen-backbone trial is excluded from the primary three: training was incomplete, and its checkpoint threshold is 0.0 while the saved fast-test report used 0.25. No matching frozen validation record was supplied. Its historical summary remains documented in the provenance; optimizer snapshots and duplicate checkpoints are not needed for submission inference.

For each primary run, root `segmentation_stats.json` includes seed, epochs, parameter counts, train/val losses where available, validation metrics, saved test metrics, threshold, data ratio, runtime in seconds and relative checkpoint path. Medium-model test records were supplied under `fast_test/`; their filenames and evaluation notes are preserved. Prediction image IDs cover all 57 official test boards, as checked during assembly. This verifies saved prediction coverage, not a fresh quantitative rerun. RTMDet train/val losses were null in its source JSON and remain null.

### Classification: supplied weights and historical reports

| Trial | Parameters | Reported validation S_cls | Reported test S_cls | t* |
|---|---:|---:|---:|---:|
| EfficientNet-B0 | 4,039,573 | 0.690599 | 0.314110 | 0.75 |
| MobileNetV3-Large | 4,234,057 | 0.777797 | 0.391596 | 0.71 |
| ConvNeXt-Tiny | 27,839,353 | 0.828935 (historical) | 0.396280 (historical) | 0.48 reported / 0.57 in supplied weights |
| EfficientNet-B2 | 7,736,219 | 0.807337 | 0.392038 | 0.53 |

These are the four historical reports and plots from reference repository commit `bf79077695478c216491d259705a0942198b2b88`; they were not quantitatively rerun during this update. Supplied EfficientNet-B0, MobileNetV3-Large and EfficientNet-B2 checkpoint validation F1 values match their reported values. **ConvNeXt-Tiny differs:** checkpoint epoch 15, val_f1 0.7799875 and threshold 0.57, versus the report's F1 0.8208817 and threshold 0.48. The supplied best checkpoint is byte-identical to the supplied ConvNeXt checkpoint, so that identity is retained without claiming that the historical top score has been reverified. Root statistics retain the historical metrics, record the actual inference threshold, preserve the historical threshold separately, and attach explicit checkpoint metadata. Runtime values are hours as established by the original training source; additional runtime_seconds fields are derived by multiplication by 3600.

The root classifier loader supports both the repaired v2 schema and a separate declared historical export schema. Supplied legacy exports use RGB, integer-truncated valid component boxes, 224x224 bilinear resize and ImageNet normalization, as specified by the original source. They are not mislabeled as repaired training outputs. All four models strictly load, match reported parameter counts and produce finite 25-class outputs. Exact source/export hashes and tensor equality are recorded in `evidence/classification_checkpoint_verification.json`. The matching historical implementation is preserved in `experiments/source_repository/`; the unexecuted alternative notebook is not experimental evidence.

## Environment

Use Python 3.12 in a dedicated environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\Activate.ps1`. Use Git Bash/WSL for shell launchers, or execute the Python files directly. For GPU training, use a matched Torch/torchvision CUDA wheel pair and verify the environment. The local assembly checks used Python 3.12.5, Torch 2.6.0+cpu, torchvision 0.21.0+cpu and Ultralytics 8.3.161 with existing local packages. A brand-new dependency installation was not performed.

Recorded medium training environment: Python 3.13.15, Torch 2.11.0+cu128, Ultralytics 8.3.161, Tesla T4. Exact recorded packages are in `evidence/yolov8m/environment_freeze.txt`. RTMDet used Python 3.10.21, Torch 2.1.2+cu121, mmcv 2.1.0, mmengine 0.10.7 and mmdet 3.3.0 on Tesla T4. Follow `experiments/rtmdet/README.md` in a separate environment; the primary requirements do not install MMDetection. Historical classification training software/hardware versions are not precisely recorded; generic descriptions in the reference README are not substituted as measurements.

## Data and exact 25-class mapping

Obtain the original GraphPCB dataset separately. `archive.zip` is not part of the submission. Official partitions have Graph-F 115 train / 47 test and Graph-W 37 train / 10 test boards. The medium-model manifest reserves 21 of the 152 official training boards for validation, leaving 131 distinct training boards, ratio `131/152 = 0.861842`. There are no additional PCB images in these runs. Training-only transforms do not increase the distinct-data ratio.

`data_mapping.json` contains classification IDs/names, required display colors, aliases and ignore rules. IDs are 1..25 in exported annotations and 0..24 at model interfaces where documented. Canonical order is:

`R C U Q J L RA D RN TP IC P CR M BTN FB CRA SW T F V LED S QA JP`.

Do not merge U with IC or SW with S. Unknown/unresolved source labels are ignored rather than remapped using test performance. Classification mapping ambiguities are documented in `docs/classification_mapping_audit.md`. Segmentation's original source RGB palette and mapping are in `configs/segmentation_data_mapping.json`; source mask colors differ from the assignment display palette. Overlays use the assignment palette.

Graph-F provides class-colored masks, from which 8-connected regions are reconstructed as instances. Touching same-class objects may remain merged, and disconnected fragments may split. Graph-W supplies box annotations without true pixel masks. Medium/RTMDet training disables Graph-W mask loss, and mask evaluation covers Graph-F only. All-board and per-domain box metrics remain available. These limitations prevent claiming the full assignment's true-instance, both-domain mask result.

Prepare medium-model data from a ZIP or expanded directory:

```bash
python src/segmentation/prepare_data.py --source /path/to/archive.zip --out /path/to/prepared_seg --seed 42
```

The prepared directory contains images/tiles and COCO files and must stay outside the submission archive. Official test membership is never moved into training. `data/segmentation_manifest.json` preserves the supplied medium run's exact split/image hashes.

## Training, evaluation and reporting

Medium-640 and medium-960 use pretrained YOLOv8m-seg, 25-class heads, class-balanced tile repetition, EMA, AdamW, batch 2 with accumulation 4, initial LR .0002 and weight decay .05. Losses include classification, box, mask and DFL terms. Tilings use size 1024 and overlap 256. Exact augmentation, inference/NMS settings and per-epoch losses are in each `work_dirs/<trial>/config.json` and `history.json`.

Reproduction commands (new GPU training; not run during assembly):

```bash
python src/segmentation/train.py --data /path/to/prepared_seg --out /path/to/new_seg_run --trials yolov8m_640_aug yolov8m_960_aug --epochs 60 --batch 2 --accumulation 4 --seed 42 --hours 0
python src/segmentation/evaluate.py --data /path/to/prepared_seg --out /path/to/new_seg_run
```

Training selects checkpoint epochs and confidence thresholds from validation. Test evaluation requires frozen selections. Keep a new output path to avoid mixing runs. To recreate figures from the supplied saved metrics without training:

```bash
python scripts/plot_saved_results.py
python scripts/render_saved_overlays.py --archive /path/to/archive.zip
```

Overlay source hashes are checked. Representative images are selected deterministically by official test ID and recorded in `results/segmentation/overlay_manifest.json`. Overlay generation uses existing predictions; it does not tune thresholds or compute new performance.

For the repaired classification path, expand GraphPCB under a separate data directory, derive and validate the manifest, then train/evaluate in new output directories:

```bash
python -m src.data_prep --legacy-manifest data/dataset_manifest.json --source-root /path/to/GraphPCB --cache --output-dir /path/to/classification_ready
python -m src.data_prep --validate /path/to/classification_ready/manifest.json
python train_experiments.py --manifest /path/to/classification_ready/manifest.json --config config.yml --run-dir /path/to/new_cls_run
python scripts/select_classification.py --runs /path/to/new_cls_run/trial_1_efficientnet_b0 /path/to/new_cls_run/trial_2_mobilenet_v3_large /path/to/new_cls_run/trial_3_convnext_tiny --output-dir /path/to/cls_selection
python scripts/evaluate_all_test.py --manifest /path/to/classification_ready/manifest.json --checkpoints /path/to/new_cls_run/trial_1_efficientnet_b0/selected.ckpt /path/to/new_cls_run/trial_2_mobilenet_v3_large/selected.ckpt /path/to/new_cls_run/trial_3_convnext_tiny/selected.ckpt --output-dir /path/to/cls_test
```

Resolve mapping/source/bounds validation errors against the original annotations. These commands are prepared for a repaired rerun and do not reproduce the historical statistics byte-for-byte. The fourth historical B2 configuration remains in `experiments/source_repository/config.yml`. New classification tests use a fixed 25-label macro average, zero for absent classes, primary argmax metrics and supplementary validation-frozen rejection metrics. Test predictions are never used to tune t*.

## Standalone inference

Each launcher invokes one Python file containing construction, preprocessing, forward, postprocessing and visualization with only external-package/standard-library imports.

Segmentation, available now:

```bash
bash scripts/run_segmentation.sh --checkpoint checkpoints/best_segment.ckpt --images /path/to/board.png --out /path/to/predictions --device cpu
# Direct equivalent:
python inference_segmentation.py --checkpoint checkpoints/best_segment.ckpt --images /path/to/board.png --out /path/to/predictions --device cpu
```

Use `--device cuda:0` on a compatible GPU. Images may be a file or directory. Frozen confidence is loaded from checkpoint metadata. `--coco /path/to/test_boards.json --data-root /path/to/prepared_seg` enables the included metrics path on annotated data.

Classification, using the supplied ConvNeXt-Tiny weights and their saved threshold 0.57:

```bash
bash scripts/run_classification.sh --image /path/to/board.png --checkpoint checkpoints/best_classification.ckpt --boxes-json /path/to/boxes.json --output /path/to/cls_prediction.png
# For an already isolated component crop:
python inference_classification.py --image /path/to/crop.png --checkpoint checkpoints/best_classification.ckpt --single-crop --output /path/to/crop_prediction.png
```

`boxes.json` is a list of `[x1,y1,x2,y2]` component boxes. Full-board classification requires annotated/derived boxes; treating the whole PCB as one component is not the task. The exported supplied checkpoints declare the historical integer-truncation crop contract. Repaired v2 checkpoints retain their distinct strict floor/ceil contract. The classification launcher was tested on all 374 supplied component boxes of the first official test board, Arty_Bottom; its new PNG/JSON output is in `results/classification/supplied_convnext_test_Arty_Bottom*`. This is an inference check, not a full-test quantitative rerun.

## Final checks and archive creation

```bash
python scripts/check_submission.py --load-checkpoints --output submission_audit.json
python scripts/package_submission.py --output-dir /path/to/output
```

The packager checks all mandatory files, model hashes and declared checkpoint metadata, creates the exact required archive, tests ZIP CRCs and prints SHA256. The audit distinguishes `artifact_checks_pass` / `ready_to_zip` from `ready_to_submit`: historical ConvNeXt metric provenance still needs reconciliation and the final private repository must be handled separately. A structurally complete archive does not certify historical performance or private repository settings. Keep only one final archive when submitting, and exclude assembly workspace/backup folders.

## Sources and adapted software

- Assignment guidelines: `docs/assignment_guidelines.pdf`.
- Original group source: https://github.com/AKarmakar-codes/DeepLearning (private, commit recorded above).
- GraphPCB data/official partition source: https://github.com/BlueSkyLT/GraphPCB .
- Canonical FPIC-component taxonomy reference: https://datasetninja.com/fpic-component .
- Ultralytics YOLOv8 architecture and adapted losses: https://github.com/ultralytics/ultralytics/tree/v8.3.161 . Pretrained medium segmentation weights: `yolov8m-seg.pt`; trained exported heads use 25 classes.
- Torchvision classification architectures/ImageNet pretrained weights: https://github.com/pytorch/vision . Trained tensors are supplied; exact original training package versions and pretrained-weight enum provenance still require the original training logs.
- RTMDet/MMDetection: https://github.com/open-mmlab/mmdetection/tree/v3.3.0 . Its adapted source/config is under `experiments/rtmdet/`.
- COCO mask/AP tooling: https://github.com/cocodataset/cocoapi .

Full input-to-output provenance and conflicting-result handling are documented in `docs/PROVENANCE.md`.
