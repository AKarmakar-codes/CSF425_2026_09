# Submission status - G09

All mandatory file types and both required best-checkpoint files are now present. The package can be archived. A historical metric discrepancy and the final private repository still require attention before claiming full submission compliance.

| Requirement | Status |
|---|---|
| G09, both names and student IDs | Included |
| Source, configuration, mapping, README, shell/Python inference | Included |
| Primary segmentation configurations | YOLOv8m-640, YOLOv8m-960, RTMDet-tiny-640 |
| best_segment.ckpt | Validation-winning medium-640; source hash matched and CPU inference checked |
| best_classification.ckpt | Supplied ConvNeXt-Tiny identity, all tensors preserved, strict load/25-class forward checked |
| Classification trial weights | EfficientNet-B0, MobileNetV3-Large, ConvNeXt-Tiny, EfficientNet-B2 included |
| Classification launcher | Ran successfully on all 374 supplied boxes of official test board Arty_Bottom |
| Classification metrics | Historical reports retained; B0/MobileNet/B2 checkpoint F1 values match; ConvNeXt differs |
| YOLOv8s checkpoint | Supporting weights and embedded 25-epoch history included; not the older reported 70-epoch model |
| YOLOv8n checkpoint | Not supplied; no invented weights |
| Segmentation plots, tables, losses and overlays | Included |
| Both standalone launchers | Executed on real images with the supplied trained models |
| Complete true-mask metrics across both domains | Unavailable: Graph-W lacks true masks; Graph-F instances are reconstructed |
| Final private repository CSF425_2026_G09 | Not created/updated; final URL has not been supplied |
| Exact named ZIP | Created by the packager after artifact checks; CRC and SHA256 verified |

## Remaining experimental discrepancy

The supplied best_classification.ckpt is identical to the supplied trial_3_convnext_tiny.ckpt. Its saved epoch is 15, val_f1 is 0.779987500000434 and frozen threshold is 0.5700000000000001. The historical report instead gives validation F1 0.820881684325103 and threshold 0.48. Inference uses the actual checkpoint threshold. Historical scores and plots remain explicitly historical; no fabricated replacement metrics or full-test rerun is claimed. See evidence/classification_checkpoint_verification.json and selection_classification.json.

MobileNet lacked a saved threshold. Its export uses 0.71 from its matching historical validation report, recording that source. B0 and B2 retain saved thresholds 0.75 and 0.53.

## Segmentation selection

The supplied YOLO-s has a 25-epoch history and cannot support the older local 70-epoch score that initially placed s third. The primary set remains medium-640, medium-960 and RTMDet-tiny. The newly received file named best_segment.ckpt duplicates YOLO-s; it has not overwritten the stronger validation-selected medium checkpoint. No nano weights were in this archive.

## Artifact handling

Inference exports omit optimizer state but preserve every model tensor and saved model identity, epoch, validation F1 and threshold. Exact source/export hashes and tensor comparisons are recorded. Original archives remain untouched outside the submission folder. Large checkpoints are ignored by Git and included in the local email archive.

The checker separates artifact completeness (artifact_checks_pass / ready_to_zip) from submission review (ready_to_submit / submission_review_required). Reconcile the ConvNeXt report before claiming its old score for these weights, and create/verify the final private repository separately. No repository upload or email was performed.
