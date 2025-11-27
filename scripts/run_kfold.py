import os, subprocess, sys

def run(cmd):
    print(">>>", cmd)
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        sys.exit(r.returncode)

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    run("python scripts/train_bilstm.py --synthetic --epochs 5")
    run("python scripts/train_baselines.py --synthetic")
    print("Done.")
