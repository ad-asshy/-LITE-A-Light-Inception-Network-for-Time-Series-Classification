import glob, json, os, pandas as pd
from scipy.stats import wilcoxon
ST = os.environ.get("ST", "/content/results_store")
rows = []
for arm in ["A_base", "B_split", "C_es"]:
    c = pd.read_csv(f"{ST}/results_{arm}/LITE/results_ucr.csv")
    for _, r in c.iterrows():
        js = [json.load(open(f)) for f in glob.glob(f"{ST}/results_{arm}/LITE/run_*/{r['dataset']}/dict_emissions.json")]
        rows.append(dict(arm=arm, dataset=r["dataset"], acc=r["LITE-mean"], time_s=r["Training Time-mean"],
                         co2_g=r["CO2 Consumption-mean"] * 1000,
                         epochs=sum(j.get("epochs_run", float("nan")) for j in js) / max(len(js), 1)))
w = pd.DataFrame(rows).pivot(index="dataset", columns="arm", values=["acc", "time_s", "co2_g", "epochs"])
d = pd.DataFrame({
    "acc_A": w["acc"]["A_base"], "acc_B": w["acc"]["B_split"], "acc_C": w["acc"]["C_es"],
    "epochs_C": w["epochs"]["C_es"],
    "time_saved_vs_A_%": 100 * (1 - w["time_s"]["C_es"] / w["time_s"]["A_base"]),
    "time_saved_vs_B_%": 100 * (1 - w["time_s"]["C_es"] / w["time_s"]["B_split"]),
    "co2_saved_vs_A_%": 100 * (1 - w["co2_g"]["C_es"] / w["co2_g"]["A_base"]),
}).dropna()
print(d.round(3).to_string())
print("mean acc change  C-A: %+.4f   C-B: %+.4f" % ((d.acc_C - d.acc_A).mean(), (d.acc_C - d.acc_B).mean()))
print(d[[c for c in d if "saved" in c]].mean().round(1).to_string())
diff = d.acc_C - d.acc_A
print("Wilcoxon p (C vs A accuracy):", wilcoxon(diff).pvalue if (diff != 0).any() else "all equal")
d.to_csv("improvement_table.csv")
