# Checkpoint inventory

- best_classification.ckpt: supplied best ConvNeXt-Tiny, exported without optimizer state. Identical to exported trial_3_convnext_tiny.ckpt; all tensors match the source. Saved threshold 0.57.
- trial_1_efficientnet_b0.ckpt, trial_2_mobilenet_v3_large.ckpt, trial_3_convnext_tiny.ckpt, trial_4_efficientnet_b2.ckpt: all supplied classification trials, with explicit historical preprocessing metadata and source hashes. All strictly load and produce finite 25-class outputs.
- best_segment.ckpt: exact copy of yolov8m_640_aug.ckpt, selected on validation S_seg. Source hash matches the recorded experiment export.
- yolov8m_960_aug.ckpt: medium resolution ablation, source hash matched.
- supporting/rtmdet_tiny_640.ckpt: genuine five-epoch RTMDet baseline; requires the separate environment in experiments/rtmdet/README.md.
- supporting/trial_1_yolov8s_seg.pt: supplied native YOLO-s checkpoint. Embedded metadata: Ultralytics 8.4.182, 25 epochs, 11,799,771 parameters. The source archive's best_segment.ckpt duplicates it and is not separately included. It does not match the older 70-epoch local report.

Classification optimizer states are unnecessary for inference and were omitted. Source archives remain unchanged. Exact hashes, tensor equality, counts and report differences are in evidence/classification_checkpoint_verification.json.

ConvNeXt checkpoint val_f1 0.7799875 / threshold 0.57 differ from its historical report F1 0.8208817 / threshold 0.48. No quantitative rerun was performed. MobileNet lacked a saved threshold; its export explicitly records the matching historical validation report as the source of 0.71. These are declared legacy exports, not repaired training runs or new performance measurements.

The root loader accepts the separate classification-checkpoint-legacy-export-v1 schema and repaired v2 schema. Unknown raw legacy metadata is rejected. Keep large checkpoint files out of GitHub commits; include them in the submission archive or a separately documented checkpoint location.
