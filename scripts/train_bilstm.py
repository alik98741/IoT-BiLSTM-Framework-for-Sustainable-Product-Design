import argparse, os, json
import numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from src.models.bilstm import BiLSTMModel
from src.data.datasets import make_synthetic, SequenceDataset, load_npz
from src.utils.metrics import classification_metrics, mean_std
from src.utils.seed import set_seed
from tqdm import tqdm

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X).squeeze(-1)
        loss = criterion(logits, y) if logits.dim()==y.dim() else criterion(logits, y.long())
        loss.backward()
        optimizer.step()
        total += loss.item() * X.size(0)
    return total / len(loader.dataset)

def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            p = model.predict_prob(X).detach().cpu().numpy()
            ps.append(p)
            ys.append(y.numpy())
    y_true = np.concatenate(ys)
    y_prob = np.concatenate(ps)
    return classification_metrics(y_true, y_prob)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--data_npz", type=str, default=None)
    ap.add_argument("--seq_len", type=int, default=60)
    ap.add_argument("--features", type=int, default=17)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.synthetic:
        X, y = make_synthetic(n_samples=2000, seq_len=args.seq_len, n_features=args.features, seed=args.seed, task="classification")
    elif args.data_npz:
        X, y = load_npz(args.data_npz)
    else:
        raise SystemExit("Provide --synthetic or --data_npz path")

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_rows = []
    for fi, (tr, va) in enumerate(skf.split(X, y)):
        Xtr, ytr = X[tr], y[tr]
        Xva, yva = X[va], y[va]
        tr_ds = SequenceDataset(Xtr, ytr)
        va_ds = SequenceDataset(Xva, yva)
        tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True)
        va_dl = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False)

        model = BiLSTMModel(input_size=X.shape[-1], hidden_size=args.hidden, num_layers=args.layers, dropout=args.dropout, num_classes=1, task="classification").to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        for epoch in range(args.epochs):
            train_one_epoch(model, tr_dl, criterion, optimizer, device)

        m = evaluate(model, va_dl, device)
        m['fold'] = fi
        fold_rows.append(m)

    os.makedirs("results", exist_ok=True)
    out_json = "results/bilstm_kfold_results.json"
    with open(out_json, "w") as f:
        json.dump(fold_rows, f, indent=2)

    # Print summary
    for k in ["accuracy","precision","recall","f1","auc"]:
        mu, sd = mean_std(fold_rows, k)
        print(f"{k}: {mu:.4f} ± {sd:.4f}")
    print(f"Saved: {out_json}")

if __name__ == "__main__":
    main()
