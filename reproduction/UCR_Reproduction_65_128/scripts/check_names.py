import numpy as np
from aeon.datasets import load_classification

bad = []
for name in open("datasets_65_128.txt").read().split():
    try:
        X, y = load_classification(name=name, split="train")
        if not isinstance(X, np.ndarray) or np.isnan(X).any():
            bad.append((name, "variable length or NaN"))
        else:
            print("ok", name, X.shape)
    except Exception as e:
        bad.append((name, str(e)[:80]))
print("PROBLEMS:", bad)