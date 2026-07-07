# Generated Graphs

This folder contains summary charts generated from the datasets in this repository.

Run:

```bash
python3 generate_graphs.py
```

to regenerate the PNG files.

For poster-ready figures based on experiment results, run:

```bash
python3 src/generate_poster_figures.py \
  --results-csv results/master_results_table.csv \
  --results-dir results \
  --reliability-csv results/*reliability*.csv \
  --output-dir graphs/poster_figures
```

The poster figures are saved in `graphs/poster_figures/` and may include cross-dataset accuracy, average model summaries, ECE/Brier before-and-after plots, calibration improvement, and reliability plots.
