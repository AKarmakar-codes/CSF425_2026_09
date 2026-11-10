"""All tables, plots, and notebook summaries derive from the same prediction file."""
import csv
import json
from pathlib import Path
import numpy as np
from src.classification.policy import evaluate_probabilities, predictions, sweep_threshold, POLICY
from src.classification.provenance import write_json, file_hash

def save_records(path, samples, probabilities, targets, threshold, metadata, split):
    a, t = predictions(probabilities, threshold)
    if len(samples) != len(a) or len({s['sample_id'] for s in samples}) != len(samples):
        raise ValueError('Prediction record count/identity mismatch')
    rows = []
    for i, s in enumerate(samples):
        if int(targets[i]) != s['class_id'] - 1:
            raise ValueError('Target order mismatch')
        rows.append({k: s[k] for k in ['sample_id','board_id','dataset','image_path','bbox']} |
                    {'target': int(targets[i]), 'probabilities': np.asarray(probabilities[i]).tolist(),
                     'argmax_prediction': int(a[i]), 'thresholded_prediction': int(t[i])})
    write_json(path, {'schema': 'classification-predictions-v2', 'split': split, 'threshold': threshold,
                      'metadata': metadata, 'decision_policy': POLICY, 'records': rows})

def summarize_records(path):
    data = json.loads(Path(path).read_text(encoding='utf8'))
    rows = data['records']
    p, y = np.array([r['probabilities'] for r in rows]), np.array([r['target'] for r in rows])
    a, t = predictions(p, data['threshold'])
    if [r['argmax_prediction'] for r in rows] != a.tolist() or [r['thresholded_prediction'] for r in rows] != t.tolist():
        raise ValueError('Saved decision mismatch')
    summary = evaluate_probabilities(p, y, data['threshold'])
    summary.update(split=data['split'], prediction_sha256=file_hash(path), metadata=data['metadata'])
    return summary

def render_records(path, output_dir):
    from src.classification.eval import plot_confusion_matrix_heatmap, plot_threshold_curves
    import matplotlib
    matplotlib.use('Agg')
    summary = summarize_records(path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    data = json.loads(Path(path).read_text(encoding='utf8'))
    names = data['metadata']['class_names']
    write_json(out/'metrics.json', summary)
    for policy in ['argmax','frozen_threshold']:
        m = summary[policy]
        plot_confusion_matrix_heatmap(np.array(m['confusion_matrix']), names, str(out/(policy+'_confusion.png')))
    with (out/'per_class.csv').open('x', newline='', encoding='utf8') as f:
        w = csv.writer(f)
        w.writerow(['policy','class_id','class_name','support','precision','recall','f1'])
        for policy in ['argmax','frozen_threshold']:
            m = summary[policy]
            for i in range(25):
                w.writerow([policy,i+1,names[i],m['support'][i],m['P_per_class'][i],m['R_per_class'][i],m['F1_per_class'][i]])
    # Test predictions never enter a threshold sweep.
    if data['split'] == 'val':
        p = np.array([r['probabilities'] for r in data['records']])
        y = np.array([r['target'] for r in data['records']])
        ts, ps, rs, fs, _ = sweep_threshold(p, y)
        plot_threshold_curves(ts, ps, rs, fs, data['threshold'], str(out/'validation_threshold_sweep.png'))
    return summary
