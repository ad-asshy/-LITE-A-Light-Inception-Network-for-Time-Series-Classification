import pandas as pd
from scipy.stats import wilcoxon
ST = "/content/results_store"
ARMS = {"Original": "A_base", "Control": "B_split", "ES100": "C_es", "ES300": "D_es_p300"}
acc = pd.DataFrame({k: pd.read_csv(f"{ST}/results_{v}/LITE/results_ucr.csv").set_index("dataset")["LITE-mean"]
                    for k, v in ARMS.items()})
for a, b in (("ES100", "Original"), ("ES300", "Original"), ("ES100", "Control"), ("ES300", "Control")):
    d = acc[a] - acc[b]
    w, t, l = int((d > 0).sum()), int((d == 0).sum()), int((d < 0).sum())
    p = wilcoxon(d).pvalue if (d != 0).any() else float("nan")
    print(f"{a} vs {b}: wins/ties/losses = {w}/{t}/{l}, Wilcoxon p = {p:.3f}")
