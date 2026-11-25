# Datavidia 11.0 Task 1 — shipment status classification

This project builds shipment features with Polars, writes Zstandard Parquet, and trains one multiclass CatBoost model. Run from PowerShell with Python 3.11 or newer.

```powershell
Set-Location D:\TIF\arkavidia
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item config/default.toml config/local.toml
.\.venv\Scripts\python.exe run.py --config config/local.toml inspect
```

## Configure the dataset

The default filenames and column names are placeholders. The project folder had no dataset when inspected. Set `data.root` relative to the TOML file, or use an absolute path with forward slashes. Put CSV or Parquet files there and use the inspection output to configure train, test, sample submission, ID, and target. Set `sample = ""` only if a sample is unavailable, then confirm Kaggle expects the configured ID and target headers in that order.

Set `features` to an explicit list of fields known at prediction time. List numeric-looking identifiers in `categorical` to retain them as strings. Declare date fields under `dates` with a format such as `"%Y-%m-%d %H:%M:%S"`. Timestamps are interpreted as UTC when no offset is supplied.

Add one `[[relations]]` block per auxiliary table. A lookup joins selected columns from a unique key. An aggregate computes a row count and means of selected numeric columns. For timestamped history, set `event_time` and `cutoff` so each shipment uses events at or before its cutoff. The source event timestamp can have its own format via `dates = { event_at = "%Y-%m-%d %H:%M:%S" }` inside its relation block. Use unfiltered aggregates only for tables already scoped to prediction time. For high-volume event joins, check history size per key because the join expands matching events before filtering. For CSVs with mixed-type or zero-padded columns, prefer explicit categorical declarations or typed upstream Parquet.

Exclude outcome fields, delivery or claim updates recorded after prediction, and derived statistics built from labels. CatBoost learns internal categorical statistics from fit rows. There is no shared external target-encoding table. Keep the seed and dataset row order fixed to reproduce the split, and choose experiments using validation scores before consulting holdout scores.

## Run

```powershell
.\.venv\Scripts\python.exe run.py --config config/local.toml prepare
.\.venv\Scripts\python.exe run.py --config config/local.toml train
.\.venv\Scripts\python.exe run.py --config config/local.toml train --profile large --device gpu
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

`train` prepares features in its own output directory. The baseline uses 450 trees at most and the large profile uses 1,600. GPU-first `auto` retries CPU on recognized CUDA/device failures and logs the reason. `--device gpu` requires GPU and propagates errors. `--device cpu` goes directly to CPU. CatBoost uses class weights during fitting and unweighted Macro F1 for early stopping. CatBoost documents GPU support for `MultiClass` and `TotalF1`. See the [official metric support table](https://catboost.ai/docs/en/concepts/loss-functions-multiclassification#used-for-optimization).

The split reserves 15% of all labeled shipments as a stratified holdout, then takes 20% of the remaining rows for one stratified validation set. The model fits the remaining approximately 68%. The validation set selects the tree count and probability multipliers. The holdout is scored afterward. The submission uses this same model and the frozen multipliers. `outputs/<run>` contains features, model, split membership, validation and holdout metrics, raw test probabilities, and the sample-ordered submission.

## Optional LightGBM blend

Use this only after confirming the competition permits your collaboration. Supply a CSV with one row per test shipment, the configured ID header, and exactly one probability column for each original target label. Rows must sum to one. IDs and class columns can arrive in any order. The command explicitly aligns them and checks for duplicates, missing IDs, and unexpected classes. The blend uses raw probabilities with a fixed teammate share and predicts by argmax. Choose the share using permitted validation predictions for the same split if available.

```powershell
.\.venv\Scripts\python.exe run.py --config config/local.toml ensemble --run outputs/MODEL_RUN --teammate data/teammate_probabilities.csv --weight 0.4 --out outputs/blend.csv --rules-permit
```

