# Source and evidence provenance

- Assignment: the exact user-designated desktop PDF, copied to assignment_guidelines.pdf. Its visual text specifies underscores in repository/archive/email names.
- Source reference: private AKarmakar-codes/DeepLearning, read at commit bf79077695478c216491d259705a0942198b2b88. This is not the final submission repository. Only source and necessary annotation manifests are copied; Git history/credentials are excluded.
- Primary segmentation source: the existing consolidated medium-model project, originally provided in GraphPCB_Report_Results.zip. Exact exported checkpoint hashes match GraphPCB_Model_Checkpoints.zip.
- Medium results: GraphPCB_Report_Results.zip. Completed medium-640 and medium-960 training have 60-epoch histories; checkpoint epochs are 25 and 45. Their saved fast_test metrics were not renamed to imply a new full evaluation.
- RTMDet: GraphPCB_T4_segmentation_results.zip; five-epoch tiny-640 baseline. Source, metadata, metrics, plots and its actual checkpoint are retained separately. Train/val scalar losses are null in its report and remain null.
- Classification: four historical source-repository reports and their matching plots. A separately repaired classification implementation is at root; historical source remains under experiments/source_repository. The subsequently supplied classifier weights use a separate declared legacy export schema, not the repaired v2 schema.
- Older local historical metrics and plots are preserved under evidence/raw and evidence/legacy_classification_plots. They conflict with the repository for some runs and are not treated as a new result.
- CSF425_Tested_Source_Code.zip is a separate reference implementation without experiment checkpoints; its untrained alternative models are not promoted to trained experimental evidence.
- archive.zip is the raw dataset and is excluded from submission. Official membership and derived annotation manifests are retained to reproduce splits; no source image dump or dataset archive is included.
- Synthetic tests, local dependency folders, remote backup credentials, optimizer states, duplicate model snapshots, and development scratch files are excluded.

No quantitative model training or full test rerun was performed during packaging. Numerical summaries and plots derive from saved results. No test scores were used to select the primary checkpoint. Efficiency-adjusted marks cannot be computed without the cohort median and competing submissions.

The RTMDet scalar logger emits newline-delimited records and NaN gradient-norm tokens in files originally suffixed `.json`. Their exact bytes are preserved with `.original_jsonlines.log` extensions so those raw text logs are not misrepresented as valid JSON documents. Primary statistics and all retained `.json` documents use valid JSON.

## Subsequently supplied checkpoint archive

checkpoints-20261018T165317Z-1-001.zip supplies all four classifier weights and YOLOv8s. Every classifier model tensor was preserved in inference exports; optimizer states were omitted. Exact source/export hashes, tensor equality, counts and 25-class forward checks are in evidence/classification_checkpoint_verification.json.

The supplied best classifier is identical to source ConvNeXt, so its identity was retained. Its metadata (val_f1 0.7799875 / t* 0.57) differs from the historical report (F1 0.8208817 / t* 0.48). This is explicit; no unmeasured P/R/S scores were fabricated. MobileNet's absent threshold is declared from its matching historical validation report (0.71), with the source recorded.

YOLO-s and the supplied file named best_segment.ckpt are duplicates. Their embedded history is 25 epochs, not the older local 70-epoch report. The actual YOLO-s weights/history are supporting evidence; primary best segmentation remains medium-640. No nano weights were supplied. See evidence/received_yolov8s*.json for native metadata/history.

The root classification launcher processed all 374 supplied boxes of Arty_Bottom with the provided ConvNeXt weights. Its PNG/probability JSON are included as an inference check, not a new full-test score. No test scores were used for selection or threshold tuning.
