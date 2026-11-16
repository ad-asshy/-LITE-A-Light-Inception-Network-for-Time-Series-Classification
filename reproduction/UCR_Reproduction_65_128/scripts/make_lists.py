import pandas as pd

COL = "dataset"

names = sorted(set(pd.read_csv("results.csv")[COL].astype(str).str.strip()))
print("number of datasets:", len(names))
open("datasets_1_64.txt", "w").write("\n".join(names[:64]))
open("datasets_65_128.txt", "w").write("\n".join(names[64:]))
print(names[:3], names[64:67])
