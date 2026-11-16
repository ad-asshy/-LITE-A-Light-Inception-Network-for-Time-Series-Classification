# Validation-based early stopping

Question: can LITE's fixed 1500-epoch budget be shortened? Four arms on one Colab T4:
A original, B 80/20 split without stopping (control), C patience 100, D patience 300
(tolerance 0.001), on 10 UCR datasets; C and D repeated 5 times on Symbols and ShapeletSim.

Main results: C saves 65.7% of training time (67.3% energy) against A but collapses to chance
accuracy on Symbols (1 of 5 runs) and ShapeletSim (3 of 5); D collapses in 0 of 5 and 4 of 5.

## Reproduce (from a clone of the upstream repo next to this one)
```bash
cd LITE
E=../<this repo folder>/improvements/early_stopping
python $E/code/prepare_lite.py              # patch, 1500 epochs, tolerance and seed switches
export STORE=$E/work NO_PUSH=1; mkdir -p $STORE; cp $E/code/datasets_improve.txt $STORE/
python $E/code/run_arms.py                  # 40 runs, about 3.5 h on a T4
python $E/code/run_replicates.py            # 20 runs, about 1.5 h
ST=$E/results python $E/code/analyze_arms.py
ST=$E/results python $E/code/analyze_replicates.py
python $E/code/make_figures.py
```
Files: `code/` scripts, `results/` per-run CSV and JSON, `figures/` PDFs used in the report.
