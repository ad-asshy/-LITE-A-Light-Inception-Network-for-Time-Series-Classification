import os, subprocess, sys, time
import pandas as pd

BUDGET_H = float(os.environ.get("SESSION_BUDGET_H", 8))
PASS = int(os.environ.get("PASS", 1))
STORE = "/kaggle/working/results_store"
OUT = f"{STORE}/results_pass{PASS}/"
CSV = f"{OUT}LITE/results_ucr.csv"
FAILED = f"{STORE}/failed.txt"
t0 = time.time()

datasets = [d.strip() for d in open(f"{STORE}/datasets_65_128.txt") if d.strip()]

def finished():
    return set(pd.read_csv(CSV)["dataset"]) if os.path.exists(CSV) else set()

def failed():
    return set(open(FAILED).read().split()) if os.path.exists(FAILED) else set()

def push(msg):
    for c in (["git", "add", "-A"], ["git", "commit", "-m", msg],
              ["git", "pull", "--rebase"], ["git", "push"]):
        subprocess.run(c, cwd=STORE)

for ds in datasets:
    if ds in finished() or ds in failed():
        continue
    if (time.time() - t0) / 3600 > BUDGET_H:
        print("Time budget reached, stopping cleanly."); break
    r = subprocess.run([sys.executable, "main.py", "--dataset", ds, "--runs", "1",
                        "--track-emissions", "True", "--output-directory", OUT])
    if r.returncode != 0:
        open(FAILED, "a").write(ds + "\n")
    push(f"{ds}: exit code {r.returncode}")
print("Finished:", len(finished()), "of", len(datasets))