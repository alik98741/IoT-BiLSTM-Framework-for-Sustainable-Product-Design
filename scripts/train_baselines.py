import argparse, os, json, numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier as _XGB if False else None  # placeholder (avoid import error)
from src.data.datasets import make_synthetic, load_npz
from src.utils.metrics import classification_metrics, mean_std
from src.utils.seed import set_seed

def aggregate_features(X):
    # Simple aggregation: mean, std across time
    mu = X.mean(axis=1)
    sd = X.std(axis=1)
    return np.concatenate([mu, sd], axis=1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--data_npz", type=str, default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    if args.synthetic:
        X, y = make_synthetic(n_samples=2000, seq_len=60, n_features=17, seed=args.seed, task="classification")
    elif args.data_npz:
        X, y = load_npz(args.data_npz)
    else:
        raise SystemExit("Provide --synthetic or --data_npz path")

    Xagg = aggregate_features(X)
    models = {
        "logreg": LogisticRegression(max_iter=1000),
        "rf": RandomForestClassifier(n_estimators=200, random_state=args.seed),
        "svm": SVC(probability=True, random_state=args.seed),
        "knn": KNeighborsClassifier(n_neighbors=7),
        "gnb": GaussianNB(),
    }

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    os.makedirs("results", exist_ok=True)
    out = {}
    for name, clf in models.items():
        rows = []
        for fi, (tr, va) in enumerate(skf.split(Xagg, y)):
            Xtr, ytr = Xagg[tr], y[tr]
            Xva, yva = Xagg[va], y[va]
            clf.fit(Xtr, ytr)
            prob = clf.predict_proba(Xva)[:,1]
            m = classification_metrics(yva, prob)
            m['fold'] = fi
            rows.append(m)
        out[name] = rows

    with open("results/baselines_kfold_results.json", "w") as f:
        json.dump(out, f, indent=2)

    for name, rows in out.items():
        print(f"== {name} ==")
        for k in ["accuracy","precision","recall","f1","auc"]:
            mu, sd = mean_std(rows, k)
            print(f"{k}: {mu:.4f} ± {sd:.4f}")

if __name__ == "__main__":
    main()
