from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def classification_metrics(y_true, y_prob, threshold=0.5):
    import numpy as np
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    # AUC only if both classes present
    try:
        metrics["auc"] = roc_auc_score(y_true, y_prob)
    except Exception:
        metrics["auc"] = float("nan")
    return metrics

def mean_std(rows, key):
    import numpy as np
    vals = np.array([r[key] for r in rows], dtype=float)
    return float(vals.mean()), float(vals.std(ddof=1) if len(vals) > 1 else 0.0)
