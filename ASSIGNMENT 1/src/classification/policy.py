"""Fixed 25-class decision contract; internal targets 0..24, reject = -1."""
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

LABELS = list(range(25))
POLICY = {'version': 'fixed25-v2', 'primary': 'argmax', 'supplementary': 'frozen_threshold',
          'reject_label': -1, 'acceptance': 'max_probability >= threshold',
          'threshold_objective': 'validation fixed-25 macro F1',
          'threshold_grid': '0.00..1.00 inclusive, step 0.01', 'tie_break': 'lowest threshold',
          'zero_division': 0, 'macro_labels': LABELS}

def predictions(probabilities, threshold=0.0):
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 25 or len(p) == 0:
        raise ValueError('Expected nonempty N-by-25 probability matrix')
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(1), 1, atol=1e-5):
        raise ValueError('Invalid probabilities')
    if not 0 <= threshold <= 1:
        raise ValueError('Threshold must be in [0,1]')
    a = p.argmax(1)
    return a, np.where(p.max(1) >= threshold, a, -1)

def compute_s_cls(p_macro, r_macro, f1_macro):
    return float(.2 * p_macro + .2 * r_macro + .6 * f1_macro)

def compute_metrics_dict(y_true, y_pred, num_classes=25):
    if num_classes != 25:
        raise ValueError('Canonical taxonomy has exactly 25 classes')
    y, p = np.asarray(y_true), np.asarray(y_pred)
    if y.ndim != 1 or y.shape != p.shape or not len(y) or not np.isin(y, LABELS).all() or not np.isin(p, [-1] + LABELS).all():
        raise ValueError('Invalid targets or predictions')
    pc, rc, fc, support = precision_recall_fscore_support(y, p, labels=LABELS, average=None, zero_division=0)
    pm, rm, fm, _ = precision_recall_fscore_support(y, p, labels=LABELS, average='macro', zero_division=0)
    assert np.allclose([pm, rm, fm], [pc.mean(), rc.mean(), fc.mean()])
    rejected = int((p == -1).sum())
    cm = np.zeros((25, 26 if rejected else 25), dtype=int)
    np.add.at(cm, (y.astype(int), np.where(p == -1, 25, p).astype(int)), 1)
    return dict(accuracy=float((y == p).mean()), P_macro=float(pm), R_macro=float(rm), F1_macro=float(fm),
                S_cls=compute_s_cls(pm, rm, fm), P_per_class=pc.tolist(), R_per_class=rc.tolist(),
                F1_per_class=fc.tolist(), support=support.tolist(), zero_support_classes=np.where(support == 0)[0].tolist(),
                confusion_matrix=cm.tolist(), coverage=float((p != -1).mean()), rejected_count=rejected,
                count=len(y))

def evaluate_probabilities(probabilities, targets, threshold):
    a, t = predictions(probabilities, threshold)
    am, tm = compute_metrics_dict(targets, a), compute_metrics_dict(targets, t)
    if len(tm['confusion_matrix'][0]) == 25:
        tm['confusion_matrix'] = [r + [0] for r in tm['confusion_matrix']]
    return {'primary_policy': 'argmax', 'test_metrics': am, 'argmax': am, 'frozen_threshold': tm,
            'threshold': float(threshold), 'decision_policy': POLICY}

def sweep_threshold(all_probs, all_targets, num_classes=25, steps=101):
    if num_classes != 25 or steps != 101:
        raise ValueError('Frozen policy uses fixed 25 classes and 101 thresholds')
    ts = np.linspace(0, 1, steps)
    metrics = [compute_metrics_dict(all_targets, predictions(all_probs, float(t))[1]) for t in ts]
    ps, rs, fs = [np.array([m[k] for m in metrics]) for k in ['P_macro', 'R_macro', 'F1_macro']]
    return ts, ps, rs, fs, float(ts[int(fs.argmax())])
