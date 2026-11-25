import csv

import numpy as np
import pandas as pd

from .features import scan, unique


def probabilities(ids, classes, values, key):
    frame = pd.DataFrame(values, columns=classes)
    frame.insert(0, key, ids)
    return frame


def aligned(path, key, ids, classes):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), [])
    if len(header) != len(set(header)) or set(header) != {key, *classes}:
        raise ValueError(f"{path}: expected a unique ID column and exactly these class columns: {list(classes)}")
    frame = pd.read_csv(path, dtype={key: str}, keep_default_na=False)
    if (frame[key] == "").any() or frame[key].duplicated().any() or set(frame[key]) != set(ids):
        raise ValueError(f"{path}: duplicate, missing, or extra shipment IDs")
    values = frame.set_index(key).loc[list(ids), list(classes)].to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError(f"{path}: probabilities must be finite and between zero and one")
    if not np.allclose(values.sum(axis=1), 1, atol=1e-5):
        raise ValueError(f"{path}: each probability row must sum to one")
    return values


def write_submission(config, ids, predictions, path):
    settings = config["data"]
    key, target = settings["id"], settings["target"]
    if not settings.get("sample"):
        pd.DataFrame({key: ids, target: predictions}).to_csv(path, index=False)
        return
    sample = scan(config["root"] / settings["sample"], [key]).collect()
    if set(sample.columns) != {key, target}:
        raise ValueError("Sample submission must contain exactly the configured ID and target columns")
    sample = sample.with_columns(sample[key].cast(str))
    unique(sample, key, "sample submission")
    if set(sample[key]) != set(ids):
        raise ValueError("Sample submission IDs differ from test IDs")
    result = pd.DataFrame({key: sample[key].to_list()})
    result[target] = result[key].map(dict(zip(ids, predictions)))
    result[sample.columns].to_csv(path, index=False)

