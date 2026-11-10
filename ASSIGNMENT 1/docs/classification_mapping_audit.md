# Classification mapping audit

This audit applies one fixed lexical transformation to derived labels. It does not inspect model performance. Official files and the shared segmentation manifest remain unchanged.

Only the retained legacy annotations are available. All 209 source images, Graph-W XML files, source Graph-F masks, official graph lists, and external source-label documentation are absent. Previously dropped source objects cannot be recovered. Official membership is unverified; the original 130 train, 22 validation, and 57 test boards are frozen exactly.

| Split | Original eligible crops | Derived eligible crops |
|---|---:|---:|
| train | 22436 | 22407 |
| val | 4213 | 4205 |
| test | 5765 | 5757 |

| Transition | Count | Representative raw labels |
|---|---:|---|
| 2->ignore | 32 | clock U4, clock Y3, clock Y2, clock Y1 |
| 5->25 | 78 | jumper J21, jumper J25, jumper JP2, jumper JP1 |
| 8->ignore | 7 | "diode zener array" DA1, "diode zener array" DA3, display P5, display DSP1 |
| 12->1 | 7 | potentiometer SW2, potentiometer R536, potentiometer SW9, potentiometer SW6 |
| 1->ignore | 6 | "resistor jumper" JT6, "resistor jumper" JT4, "resistor jumper" JT5, "resistor jumper" RJ1 |

QA and QA1 map to class 24; JP, JP1, and jumper map to class 25. Graph-F canonical IDs remain authoritative. Named source types take precedence over markings: `ic U1` follows the project WACV mapping to IC (11), while a bare `U1` maps to U (3). This distinction still needs external source documentation confirmation. Potentiometer-to-Resistor (1) is explicitly declared in the original project mapping; independent source documentation is unavailable.

Clock/display and the compound types resistor jumper/diode zener array are unresolved and ignored conservatively. Their counts and raw examples remain in the JSON audit. Specific compound rules precede generic types. Bare designators match an entire symbol plus optional digits, so arbitrary words cannot be classified by their first letter.

Rebuild from original source XML/images/graph lists into a new directory before final training/evaluation. The current derived manifest is a structural audit artifact, with missing image identities explicitly null. Final test evaluation refuses unverified official membership.

Full class counts and representative labels for every transition are in `data/classification_repaired_v2_verified/mapping_audit.json`.
