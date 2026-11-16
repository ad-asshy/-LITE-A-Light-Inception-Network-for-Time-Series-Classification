import os, subprocess
os.chdir("/content/LITE")
P = "src/classifiers/lite.py"
if "LITE_EARLY_STOPPING" not in open(P).read():
    r = subprocess.run(["python", "/content/results_store/patch_lite_es.py"], capture_output=True, text=True)
    print(r.stdout, r.stderr[-500:])
s = open(P).read()
s = s.replace("n_epochs=2,", "n_epochs=1500,")
if "min_delta" not in s:
    assert s.count("patience=self.patience,") == 1, "EarlyStopping line not found"
    s = s.replace("patience=self.patience,",
                  'patience=self.patience, min_delta=float(os.environ.get("LITE_MIN_DELTA", "0")),')
if "LITE_SEED_TAG" not in s:
    old = 'seed = zlib.crc32("/".join(os.path.normpath(self.output_directory).split(os.sep)[-2:]).encode()) % (2**31 - 1)'
    new = 'seed = zlib.crc32(("/".join(os.path.normpath(self.output_directory).split(os.sep)[-2:]) + os.environ.get("LITE_SEED_TAG", "")).encode()) % (2**31 - 1)'
    assert s.count(old) == 1, "seed line not found"
    s = s.replace(old, new)
open(P, "w").write(s)
s = open(P).read()
compile(s, P, "exec")
assert "n_epochs=1500," in s and "LITE_EARLY_STOPPING" in s and s.count("min_delta") == 1 and "LITE_SEED_TAG" in s
print("lite.py ready")
