import argparse, os, numpy as np
import pandas as pd
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, required=True, help="Directory with raw hydraulic files")
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--seq_len", type=int, default=60)
    ap.add_argument("--features", type=int, default=17)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    # Placeholder: users should implement actual parsing of UCI files.
    # We create a dummy npz to illustrate format expected by training scripts.
    n = 5000
    X = np.random.randn(n, args.seq_len, args.features).astype(np.float32)
    y = (X[:,:,:3].mean(axis=(1,2)) > 0).astype(np.float32)
    np.savez_compressed(os.path.join(args.output_dir, "hydraulic_prepared.npz"), X=X, y=y)
    print("Wrote", os.path.join(args.output_dir, "hydraulic_prepared.npz"))
if __name__ == "__main__":
    main()
