import json, os
import matplotlib.pyplot as plt
from src.utils.metrics import mean_std

def load_json(p):
    with open(p) as f:
        return json.load(f)

def bar_with_error(ax, labels, means, stds, title, ylabel):
    x = range(len(labels))
    ax.bar(x, means, yerr=stds)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_title(title)
    ax.set_ylabel(ylabel)

def main():
    os.makedirs("figures", exist_ok=True)
    bilstm = load_json("results/bilstm_kfold_results.json")
    base = load_json("results/baselines_kfold_results.json")

    # Figure 1: F1 means±SD across models
    labels, means, stds = ["BiLSTM"], [], []
    mu, sd = mean_std(bilstm, "f1")
    means.append(mu); stds.append(sd)
    for name, rows in base.items():
        m,s = mean_std(rows, "f1")
        labels.append(name.upper())
        means.append(m); stds.append(s)

    fig, ax = plt.subplots(figsize=(8,5))
    bar_with_error(ax, labels, means, stds, "F1 Score (Mean ± SD, 5-Fold)", "F1")
    fig.tight_layout()
    fig.savefig("figures/f1_barplot.png", dpi=200)
    print("Saved figures/f1_barplot.png")

if __name__ == "__main__":
    main()
