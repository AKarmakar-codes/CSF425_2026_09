# Classification evidence and supplied weights

The original trial plots and quantitative fields in root classification_stats.json come from reference repository commit bf79077695478c216491d259705a0942198b2b88. All four classifier architectures now have supplied trained weights. Parameter counts and strict model loading were checked; each produces finite 25-class outputs.

B0, MobileNet and B2 saved validation F1 values match their reports. ConvNeXt differs: checkpoint F1 0.7799875 and t* 0.57 versus report F1 0.8208817 and t* 0.48. Historical scores/plots are retained without claiming they were rerun. Root statistics distinguish the actual inference threshold from historical_report_threshold and attach checkpoint_metadata. This run identity mismatch still needs reconciliation.

supplied_convnext_test_Arty_Bottom.png and its JSON sidecar are a new real inference check on all 374 supplied boxes of the first official Graph-W test board, using the frozen threshold from the supplied checkpoint. This is not a full-test quantitative result and was not used to select a model or threshold.

Other historical runs remain separately in evidence/raw/classification_legacy.json. Historical plot palette and whole-test coverage were not newly certified.
