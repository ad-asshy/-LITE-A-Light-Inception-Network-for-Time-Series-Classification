import os, subprocess, sys, time
import pandas as pd

STORE = os.environ.get("STORE", "/content/results_store")
BUDGET_H = float(os.environ.get("SESSION_BUDGET_H", 3))
RUNS = os.environ.get("RUNS", "1")
LISTFILE = os.environ.get("DATASET_FILE", "datasets_improve.txt")
ARMS = {"A_base": ("0", "100"), "B_split": ("1", "100000"), "C_es": ("1", "100")}
t0 = time.time()
os.makedirs("/content/logs", exist_ok=True)
datasets = [d.strip() for d in open(f"{STORE}/{LISTFILE}") if d.strip()]

def done(arm, ds):
    csv = f"{STORE}/results_{arm}/LITE/results_ucr.csv"
    return os.path.exists(csv) and ds in set(pd.read_csv(csv)["dataset"])

def push(msg):
    for c in (["git", "add", "-A"], ["git", "commit", "-m", msg],
              ["git", "pull", "--rebase"], ["git", "push"]):
        r = subprocess.run(c, cwd=STORE, capture_output=True, text=True)
        if r.returncode != 0 and c[1] in ("pull", "push"):
            print("GIT PROBLEM:", c[1], r.stderr[-300:].replace("\n", " "), flush=True)

for ds in datasets:
    for arm, (es, pat) in ARMS.items():
        if done(arm, ds):
            continue
        if (time.time() - t0) / 3600 > BUDGET_H:
            print("Time budget reached, stopping cleanly.", flush=True); sys.exit()
        env = dict(os.environ, LITE_EARLY_STOPPING=es, LITE_PATIENCE=pat, LITE_VAL_SPLIT="0.2")
        t1 = time.time()
        log = f"/content/logs/{arm}_{ds}.log"
        with open(log, "w") as lf:
            r = subprocess.run([sys.executable, "main.py", "--dataset", ds, "--runs", RUNS,
                                "--track-emissions", "True",
                                "--output-directory", f"{STORE}/results_{arm}/"],
                               stdout=lf, stderr=subprocess.STDOUT, env=env)
        print(f"{arm} {ds}: exit {r.returncode}, {(time.time() - t1) / 60:.1f} min", flush=True)
        if r.returncode != 0:
            print(open(log).read()[-800:], flush=True)
        push(f"{arm} {ds}: exit {r.returncode}")
print("All done.", flush=True)