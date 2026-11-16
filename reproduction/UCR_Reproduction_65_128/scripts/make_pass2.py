import pandas as pd
d = pd.read_csv("results_ucr.csv")[["dataset", "Training Time-mean"]]
d = d.rename(columns={"Training Time-mean": "t"}).sort_values("t").reset_index(drop=True)
d["cum_h"] = (d["t"].cumsum() / 3600).round(1)
print(d.tail(12).to_string())
open("datasets_pass2.txt", "w").write("\n".join(d["dataset"]))
print(len(d), "datasets,", d["cum_h"].iloc[-1], "h of training")