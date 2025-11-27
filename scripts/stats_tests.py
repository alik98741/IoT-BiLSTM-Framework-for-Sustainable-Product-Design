import json, argparse
from scipy import stats
from src.utils.metrics import mean_std

def load_results():
    with open("results/bilstm_kfold_results.json") as f:
        bilstm = json.load(f)
    with open("results/baselines_kfold_results.json") as f:
        base = json.load(f)
    return bilstm, base

def paired_t(metric, bilstm_rows, baseline_rows):
    import numpy as np
    a = np.array([r[metric] for r in bilstm_rows], dtype=float)
    b = np.array([r[metric] for r in baseline_rows], dtype=float)
    t, p = stats.ttest_rel(a, b)
    return t, p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--metric", type=str, default="f1")
    args = ap.parse_args()

    bilstm, base = load_results()
    for name, rows in base.items():
        t, p = paired_t(args.metric, bilstm, rows)
        print(f"[{name}] paired t-test on {args.metric}: t={t:.3f}, p={p:.4f} -> {'SIGNIFICANT' if p < args.alpha else 'ns'}")

if __name__ == "__main__":
    main()
