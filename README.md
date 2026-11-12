#  LITE: A Light Inception Network for Time Series Classification


# Reproducing and extending LITE (Light Inception with boosTing tEchniques)

Course project (COMP41850, AI for Time Series). We rerun the code of

> Ismail-Fawaz, Devanne, Berretti, Weber, Forestier. *Look into the LITE in deep learning for time series classification.* International Journal of Data Science and Analytics 20 (2025) 4029-4049. [doi:10.1007/s41060-024-00708-5](https://doi.org/10.1007/s41060-024-00708-5)

(authors' code: <https://github.com/MSD-IRIMAS/LITE>), compare our numbers with the ones the authors publish, and test two extensions: **channel mixing for the multivariate model (LITEMV) on datasets with many channels**, and **LITE for time series extrinsic regression**.

**Report:** `paper/main.tex` (IEEE format). **TODO:** add the PDF link here.

> **Status (update before submitting).** UCR: 74 of 128 datasets compared at the time of the last merge (Nov 8). UEA: 16 of 30 datasets at the full protocol. Channel mixing: 4 of 6 planned datasets finished (Table 5); Heartbeat and FaceDetection pending. Search this file for **TODO**.

## Contents

```
README.md                 this file
requirements.txt          package versions (see "Environment")
notebooks/                Kaggle notebooks, one per experiment (see "How to reproduce")
scripts/                  merge_ucr.py, compare_uea.py, compare_without.py (run on your own computer)
data/uea_table3_paper.csv Table 3 of the paper (UEA accuracies, all competitors), copied by hand and checked against the paper's averages
results/                  result files of our runs (CSV)
paper/                    main.tex, refs.bib
```

## Environment

All runs were made in **Kaggle notebooks** (Python 3.13, TensorFlow 2.20.0, Keras 3.13.2, two Tesla T4 GPUs). Training needs a GPU; a laptop CPU works only for the smallest datasets.

1. Create a Kaggle notebook, set **Accelerator: GPU T4 x2** and **Internet: On**.
2. Import one of the notebooks from `notebooks/` (File, Import Notebook). Section 0 of each notebook clones the authors' repository, installs `aeon` and `codecarbon`, and patches the authors' `main.py` (see "Problems in the released code").
3. For the scripts in `scripts/`: `pip install pandas scipy openpyxl` and run them with `python3`.

`requirements.txt` lists the versions we know. Versions of `aeon`, `codecarbon` and the plotting packages were not recorded. **TODO:** pin them (command in `requirements.txt`).

## Data

Nothing has to be downloaded by hand. The authors' loader (`aeon.datasets.load_classification`) downloads the UCR and UEA datasets the first time they are used, and `aeon.datasets.load_regression` downloads the regression datasets from the Monash archive. Each dataset has its official train/test split; there is no validation set. The Kimore experiment of the paper (Table 4) was **not** reproduced: we did not obtain the data.

| Archive | Datasets in the paper | Channels | Used for |
|---|---|---|---|
| UCR | 128 | 1 | reproduction of LITE |
| UEA | 30 | 2 to 1,345 | reproduction of LITEMV, channel mixing |
| Monash regression | 4 used (Covid3Month, FloodModeling1-3) | 1 | regression extension |

## How to reproduce each result

Protocol everywhere, as in the paper: **5 runs per dataset, 1,500 epochs, batch size 64**, checkpoint with the lowest training loss, ensemble = average of the 5 runs' probabilities. Every notebook has a settings cell (section 1) and a live progress display. Runs are resumable.

| # | Result | Notebook and setting | Output | Report/README table |
|---|---|---|---|---|
| R1 | 9,814 parameters | cell below, in any notebook after section 0 | printed number | claims table of the paper |
| R2 | UCR reproduction | `lite_ucr_all_two_parts.ipynb`: `N_PARTS` people share the datasets, each sets `MY_PART`. `lite_ucr_rest.ipynb` runs the datasets that were missing after the first round | `ucr_part<N>_results.csv` | Tables 1, 2 |
| R3 | UEA reproduction | `lite_uea_part0.ipynb`, `MIX = 0` (extra datasets: the same notebook with `ONLY`, `PART_LABEL` and `RUNS` changed in section 1, see the next table) | `uea_part<N>_orig_results.csv` | Table 3 |
| R4 | Channel mixing (improvement) | `lite_uea_improvement_mix16.ipynb`, `MIX = 16`; runs only datasets with more than 16 channels | `uea_part<N>_mix16_results.csv` | Table 5 |
| R5 | Parameters against channels | counts printed by every UEA run in `params.txt`, collected in the result files; formula check below | | Table 4 |
| R6 | Regression extension and ablation | `lite_regression.ipynb` (sections 6 to 9) | `regression_results.csv`, `ablation_results.csv`, `covid_curves.png` | Tables 6, 7 |

**Result files and the settings that produced them** (`results/`; keep the file names, the scripts find the files by name):

| File | Notebook | Settings in section 1 | Content |
|---|---|---|---|
| `UEA_Reproduction.csv` | `lite_uea_part0.ipynb` | `MIX = 0`, 5 runs, 1,500 epochs, work split in three parts | 16 datasets, five-model ensembles |
| `uea_partrest_orig_results.csv` | `lite_uea_part0.ipynb` | `MIX = 0`, **`RUNS = 1`**, `ONLY = [...]`, `PART_LABEL = "rest"` | 7 datasets, single runs (too slow for five), marked `runs = 1` |
| `UEA_improvement_4.csv` | `lite_uea_improvement_mix16.ipynb` | `MIX = 16`, 5 runs, 1,500 epochs | DuckDuckGeese, FingerMovements, MotorImagery, NATOPS |
| `uea_partlast2_mix16_results.csv` | `lite_uea_improvement_mix16.ipynb` | `MIX = 16`, `ONLY = ["FaceDetection", "Heartbeat"]`, `PART_LABEL = "last2"` | Heartbeat (FaceDetection: **TODO**) |
| `improvement_4_datasets_vs_authors.csv` | `scripts/compute_improvement.py` | | the improvement table against the authors |

**R1, parameter count** (run in a notebook where the authors' repository is cloned and on the path):
```python
import numpy as np
from src.classifiers import LITE
m = LITE(output_directory="/tmp/x/", length_TS=286, n_channels=1, n_classes=2)
total = sum(int(np.prod(w.shape)) for w in m.model.trainable_weights)
print(total, total - (32 * 2 + 2))      # 9880 with the 2-class output layer, 9814 without it (the paper's number)
```

**Merging and comparing on your own computer.** Download the result files from the notebooks' Output panel into one folder, then:
```bash
python3 scripts/merge_ucr.py          # merges ucr_part*_results*.csv, compares with the authors' results.csv, lists datasets not run
python3 scripts/compare_without.py    # the same comparison with and without the three Dodger datasets (see "Findings")
python3 scripts/compare_uea.py        # UEA results against every model of the paper's Table 3
```
`merge_ucr.py` downloads the authors' `results.csv` from their repository if it is not in the folder.

## Results: ours against the paper

### Table 1. UCR, the first four datasets (accuracy as a fraction; ours = mean ± std of 5 runs, LITETime = 5-model ensemble)

| Dataset | Test size | LITE ours | LITE authors | LITETime ours | LITETime authors |
|---|---|---|---|---|---|
| Coffee | 28 | 1.0000 ± 0 | 1.0000 | 1.0000 | 1.0000 |
| ItalyPowerDemand | 1029 | 0.9685 ± 0.0018 | 0.9685 | 0.9689 | 0.9670 |
| GunPoint | 150 | 1.0000 ± 0 | 0.9973 | 1.0000 | 1.0000 |
| ECG200 | 100 | 0.8960 ± 0.0150 | 0.8960 | 0.9000 | 0.9200 |

We also reloaded all 20 saved models of these four datasets and recomputed every run's test accuracy independently of `main.py`: the five runs differ from each other (ECG200: 0.91, 0.90, 0.91, 0.89, 0.87) and their means equal the reported ones. The exact agreement of the LITE mean on ECG200 and ItalyPowerDemand is, we think, coincidence.

### Table 2. UCR, all datasets merged so far (as of Nov 8). **TODO: update with the final merge**

| Same 74 datasets | Ours | Authors | Difference | Wins / ties / losses | Wilcoxon p |
|---|---|---|---|---|---|
| LITE | 0.8368 | 0.8490 | -0.0121 | 33 / 3 / 38 | 0.514 |
| LITETime | 0.8461 | 0.8632 | -0.0171 | 25 / 9 / 40 | 0.035 |

The paper's averages (0.8304 for LITE and 0.8462 for LITETime) are over all 128 datasets and are not comparable with a partial average, so we compare with the authors' per-dataset values on the same datasets.

### Table 3. UEA, LITEMV reproduction: ensemble accuracy in %, 16 datasets (5 runs, 1,500 epochs)

| Dataset | Ours: ensemble | Ours: single run, mean ± std | Paper (LITEMVTime) | Difference |
|---|---|---|---|---|
| StandWalkJump | 33.33 | 32.00 ± 2.67 | 66.67 | -33.34 |
| AtrialFibrillation | 6.67 | 6.67 ± 0.00 | 13.33 | -6.66 |
| NATOPS | 87.22 | 87.00 ± 0.97 | 90.00 | -2.78 |
| Libras | 88.33 | 86.00 ± 1.84 | 89.44 | -1.11 |
| PenDigits | 98.37 | 98.00 ± 0.33 | 98.86 | -0.49 |
| BasicMotions | 100.00 | 100.00 ± 0.00 | 100.00 | 0 |
| Epilepsy | 99.28 | 99.42 ± 0.29 | 99.28 | 0 |
| Cricket | 98.61 | 98.61 ± 0.00 | 98.61 | 0 |
| ArticularyWordRecognition | 98.67 | 99.00 ± 0.30 | 97.33 | +1.34 |
| HandMovementDirection | 37.84 | 34.05 ± 3.01 | 35.14 | +2.70 |
| SelfRegulationSCP1 | 79.52 | 78.29 ± 1.07 | 73.04 | +6.48 |
| FingerMovements | 63.00 | 61.00 ± 3.52 | 56.00 | +7.00 |
| UWaveGestureLibrary | 92.19 | 90.69 ± 1.16 | 84.68 | +7.51 |
| ERing | 93.70 | 92.74 ± 0.90 | 84.44 | +9.26 |
| RacketSports | 87.50 | 85.53 ± 1.72 | 73.68 | +13.82 |
| Handwriting | 66.12 | 57.74 ± 1.46 | 40.00 | +26.12 |

On these 16 datasets our mean is 76.90 against 75.03 in the paper (difference +1.87; wins/ties/losses 8 / 3 / 5; Wilcoxon p = 0.244). The average agrees, but several datasets differ from the paper by much more than our run-to-run spread (Handwriting, RacketSports, ERing, UWaveGestureLibrary: higher; StandWalkJump: lower, with only 15 test samples, one sample = 6.7 points). **TODO:** compare with the authors' own `results_multivariate.csv` (the paper copies its competitors' numbers from the ConvTran paper, and the protocols may differ).

### Table 4. LITEMV size grows linearly with the number of channels

Trainable parameters of our UEA runs fit exactly: **parameters = 1,084 x C + 6,656 + 33 x (number of classes)**, with C input channels. We checked it on the 16 datasets of Table 3 (2 to 28 channels, 0 mismatches), and it reproduces the count measured on DuckDuckGeese (1,464,801 with 1,345 channels). The formula is a fit to counts from the authors' code as run; we did not read `litemv.py` to explain the 1,084.

| Channels C | Parameters without the output layer |
|---|---|
| 2 | 8,824 |
| 10 | 17,496 |
| 28 (FingerMovements) | 37,008 |
| 1,345 (DuckDuckGeese) | 1,464,636 |

### Table 5. Channel mixing (improvement): LITEMV with a learned 1x1 convolution mapping C channels to 16, against the authors (ensemble accuracy in %, 5 runs, 1,500 epochs)

Mixing only applies to the 8 UEA datasets with more than 16 channels; 6 were planned and 4 are finished. "Original" is our rerun of the authors' LITEMV (Table 3). Where it is "not run" (DuckDuckGeese: about a day per run; MotorImagery: no reproduction finished), the effect of mixing cannot be separated from the difference between our setup and the authors'.

| Dataset | Channels | Authors (LITEMVTime) | Our original | Our mixing (16) | Mixing - authors | Mixing - our original |
|---|---|---|---|---|---|---|
| DuckDuckGeese | 1,345 | 18.00 | not run | 44.00 | +26.00 | no baseline |
| MotorImagery | 64 | 53.00 | not run | 51.00 | -2.00 | no baseline |
| FingerMovements | 28 | 56.00 | 63.00 | 53.00 | -3.00 | **-10.00** |
| NATOPS | 24 | 90.00 | 87.22 | 88.33 | -1.67 | +1.11 |
| Heartbeat | 61 | 61.46 | **TODO** | **TODO** | | |
| FaceDetection | 144 | 61.01 | **TODO** | **TODO** | | |

Spread, size and cost (single runs: mean ± std; time in seconds per run with parallel workers, so not clean):

| Dataset | Single runs, original | Single runs, mixing | Parameters, original | Parameters, mixing | Time, original | Time, mixing |
|---|---|---|---|---|---|---|
| DuckDuckGeese | not run | 40.0 ± 3.6 | 1,464,801 | 45,685 | not run | 3,253 |
| MotorImagery | not run | 50.6 ± 2.7 | 76,098 (formula) | 25,090 | not run | 4,602 |
| FingerMovements | 61.0 ± 3.5 | 51.4 ± 2.1 | 37,074 | 24,514 | 710 | 783 |
| NATOPS | 87.0 ± 1.0 | 86.3 ± 2.5 | 32,870 | 24,582 | 408 | 540 |

For context, the authors' other models on the same four datasets (Table 3 of the paper; `results/improvement_4_datasets_vs_authors.csv` has all columns): ConvTran 62.00 / 56.00 / 56.00 / 94.44 (DuckDuckGeese / MotorImagery / FingerMovements / NATOPS).

Reading: mixing makes the model much smaller on very wide data (32 times on DuckDuckGeese) but shows **no accuracy gain**: a tie on NATOPS (+1.1 points, 2 test samples of 180), a clear loss on FingerMovements (-10 points, with run-to-run spread of 2 to 3.5), and no usable baseline on the other two. One preliminary measurement exists: on DuckDuckGeese the unchanged model reached 48% in a single 50-epoch run (50 test samples), above the 44% with mixing, so the +26 over the paper's 18% is not an effect of mixing.

### Table 6. Regression extension: RMSE relative to the mean predictor (below 1 beats it)

LITE with the output layer replaced by one linear unit (+33 parameters, MSE loss, standardised targets), 3 seeds, **300 epochs** (not 1,500), one official split per dataset, no paper result to compare with:

| Dataset | Mean-predictor RMSE | MiniRocket | LITE (mean ± std, 3 seeds) | LITETime-R (3 seeds) |
|---|---|---|---|---|
| Covid3Month | 0.0447 | 0.916 | 1.667 ± 0.650 | 1.047 |
| FloodModeling3 | 0.0228 | 0.259 | 0.196 ± 0.011 | 0.180 |
| FloodModeling2 | 0.0185 | 0.473 | 0.216 ± 0.020 | 0.211 |
| FloodModeling1 | 0.0189 | 0.381 | 0.269 ± 0.015 | 0.253 |

The three Flood datasets are related and count as about one problem.

### Table 7. Regression ablation: relative RMSE, mean ± std over 3 seeds

| Variant | Covid3Month | FloodModeling3 |
|---|---|---|
| Full model | 1.688 ± 1.005 | 0.197 ± 0.006 |
| No custom filters | 2.325 ± 1.119 | 0.213 ± 0.010 |
| No dilation | 2.015 ± 0.197 | 0.214 ± 0.008 |
| No target standardisation | 1.282 ± 0.547 | 0.517 ± 0.093 |

Removing target standardisation hurts most on FloodModeling3; the other differences are about two seed standard deviations with three seeds, so they are not conclusive. On Covid3Month the seed spread is as large as the means.

## Problems in the released code (and what we changed)

The notebooks patch the authors' `main.py` when they run; the original file is not edited in this repository.

- `main.py` imports `distutils`, which Python 3.13 no longer has: replaced by a small local `strtobool`.
- The model constructor defaults to `n_epochs=2` and `main.py` never sets it, so the published command trains for **2 epochs**, not 1,500: we added an `--epochs` argument (default 1,500). On Coffee, 50 epochs gave 0.536 accuracy (chance) and 1,500 gave 1.0.
- The README of the authors says to edit a `folder_path` variable that does not exist; data are downloaded by `aeon`.
- `use_multiplexing` is stored by the model class but never passed to the layer builder, so the multiplexing ablation cannot be rerun with the release (from reading the code). **TODO:** confirm with parameter counts.
- The test set is evaluated after every epoch for plotting only (selection uses the training loss), which inflates the measured time and energy. Our parallel workers make timings unclean as well: **do not use our times or CO2 for efficiency claims.** CodeCarbon also placed our server in the Netherlands.
- No random seed is set, and GPU training is not bit-wise repeatable (the same seed gave test RMSE 0.1046 and 0.1124 in two runs): agreement with the paper must be judged against run-to-run spread.
- The authors' loader fails on datasets with unequal series lengths with current `aeon` (CharacterTrajectories: "inhomogeneous shape"; SpokenArabicDigits: "'list' object has no attribute 'shape'").
- Three UCR datasets (DodgerLoopDay, DodgerLoopGame, DodgerLoopWeekend) gave accuracies far below the authors' (0.15 against 0.54, 0.52 against 0.80, 0.74 against 0.97) with zero spread over five runs, which looks like a constant prediction. Without them the LITE mean on the other 71 datasets is about 0.8523 for both us and the authors. **TODO:** confirm the cause (missing values?) with the check in the notebook, and report the comparison with and without them.
- The Kimore results (Table 4 of the paper) are all multiples of 1/75, which does not fit a single 80/20 split of 71 recordings (about 15 test samples per exercise); the paper does not say how the numbers were averaged.

## Datasets not run (every one has a reason)

| Group | Datasets | Reason |
|---|---|---|
| UEA, left out | DuckDuckGeese (original model only), InsectWingbeat, PEMS-SF | original model about a day per run; about 25,000 training samples; download failed |
| UEA, loader fails | CharacterTrajectories, SpokenArabicDigits, JapaneseVowels (probably) | unequal series lengths |
| UEA, too slow | **TODO** (Heartbeat, PhonemeSpectra, FaceDetection, MotorImagery, EigenWorms: confirm which ones did not finish) | estimated compute too high |
| UCR | **TODO:** list from the final merge | loader failure (unequal length or missing values), or not finished |

## Limitations

Every number rests on the official train/test split of its dataset, so differences smaller than the run-to-run spread are not findings. The regression study uses 3 seeds and 300 epochs. Our setup (GPU, library versions) differs from the authors', so the rerun of the original model is the fair baseline for the effect of any change. The UCR run was divided between several people on separate Kaggle accounts, each producing its own result file, merged by dataset name.

## Credits and notes

- The model and the data loaders are the authors'; see their repository for its license. Datasets: UCR and UEA archives, Monash extrinsic regression archive, downloaded through `aeon`.
- **TODO:** team members and who did what; acknowledgments for the people who ran parts of the UCR runs; the course's rule on disclosing AI assistance, and the statement if it is required.
