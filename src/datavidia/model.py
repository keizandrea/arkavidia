import json
import logging

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostError, Pool
from sklearn.model_selection import train_test_split

METRIC = "TotalF1:average=Macro"


def partition(y, fraction, seed):
    rows = np.arange(len(y))
    development, holdout = train_test_split(rows, test_size=.15, stratify=y, random_state=seed)
    train, validation = train_test_split(
        development, test_size=fraction, stratify=y[development], random_state=seed + 1,
    )
    if any(np.unique(y[part]).size != np.unique(y).size for part in (train, validation, holdout)):
        raise ValueError("Every split must contain every class. Check rare class counts.")
    return train, validation, holdout


def macro_f1(expected, predicted, count):
    correct = expected == predicted
    hits = np.bincount(expected[correct], minlength=count)
    totals = np.bincount(expected, minlength=count) + np.bincount(predicted, minlength=count)
    return float(np.divide(2 * hits, totals, out=np.zeros(count), where=totals != 0).mean())


def tune(expected, probabilities):
    count = probabilities.shape[1]
    weights = np.ones(count)
    best = macro_f1(expected, probabilities.argmax(axis=1), count)
    for _ in range(2):
        for index in range(count):
            candidate = weights.copy()
            for multiplier in (.5, .7, .85, 1., 1.2, 1.5, 2., 3.):
                candidate[index] = multiplier
                score = macro_f1(expected, (probabilities * candidate).argmax(axis=1), count)
                if score > best:
                    best = score
                    weights[index] = multiplier
    return weights


def fit(params, training, validation, device, gpu_devices):
    if device not in {"auto", "gpu", "cpu"}:
        raise ValueError("Device must be auto, gpu, or cpu")
    for task in (("GPU", "CPU") if device == "auto" else (device.upper(),)):
        settings = dict(params, task_type=task)
        if task == "GPU":
            settings["devices"] = gpu_devices
        model = CatBoostClassifier(**settings)
        try:
            model.fit(training, eval_set=validation)
        except CatBoostError as error:
            message = str(error).lower()
            gpu_error = any(token in message for token in (
                "cuda", "cudnn", "cublas", "gpu memory", "gpu device",
                "no device", "device count", "driver version",
            ))
            if device != "auto" or task != "GPU" or not gpu_error:
                raise
            logging.warning("GPU failed, retrying on CPU: %s", error)
        else:
            return model, task
    raise RuntimeError("Training exited without a model")


def inputs(train, test):
    if list(train.columns) != list(test.columns):
        raise ValueError("Feature columns differ")
    categorical = []
    for column in train.columns:
        kind = train[column].dtype
        if pd.api.types.is_numeric_dtype(kind):
            try:
                test[column] = pd.to_numeric(test[column], errors="raise").astype(float)
                train[column] = train[column].astype(float)
            except (ValueError, TypeError) as error:
                raise ValueError(f"Incompatible test feature {column}") from error
            for frame in (train, test):
                frame[column] = frame[column].replace([np.inf, -np.inf], np.nan)
        else:
            categorical.append(column)
            for frame in (train, test):
                values = frame[column].astype("string")
                frame[column] = ("v:" + values).fillna("m:").astype(object)
    return categorical


def train_model(config, train, test, profile, destination, device):
    data, settings = config["data"], config["training"]
    key, target = data["id"], data["target"]
    labels = train[target].to_list()
    if any(label is None or not label.strip() for label in labels):
        raise ValueError("Training labels must be nonempty")
    classes, y = np.unique(labels, return_inverse=True)
    if len(classes) < 2 or target in classes or key in classes:
        raise ValueError("Require at least two classes and no class name matching ID or target")
    train_rows, val_rows, hold_rows = partition(
        y, settings["validation_fraction_of_development"], settings["seed"],
    )
    x = train.drop(key, target).to_pandas()
    x_test = test.drop(key).to_pandas()
    if x.shape[1] == 0:
        raise ValueError("No model features were produced")
    categorical = inputs(x, x_test)
    weighting = settings["class_weights"]
    if weighting not in {"None", "Balanced", "SqrtBalanced"}:
        raise ValueError("class_weights must be None, Balanced, or SqrtBalanced")
    counts = np.bincount(y[train_rows], minlength=len(classes))
    weighted = weighting != "None" and counts.min() != counts.max()
    params = dict(
        config["profiles"][profile], loss_function="MultiClass", eval_metric=METRIC,
        boosting_type="Plain", max_ctr_complexity=1, thread_count=6,
        random_seed=settings["seed"],
        early_stopping_rounds=settings["early_stopping"], use_best_model=True,
        allow_writing_files=False, verbose=50,
    )
    if weighted:
        params.update(auto_class_weights=weighting, eval_metric=METRIC + ";use_weights=false")
    model, task = fit(
        params, Pool(x.iloc[train_rows], y[train_rows], cat_features=categorical),
        Pool(x.iloc[val_rows], y[val_rows], cat_features=categorical),
        device or settings["device"], settings["gpu_devices"],
    )
    order = np.argsort(model.classes_)
    validation = model.predict_proba(x.iloc[val_rows])[:, order]
    multipliers = tune(y[val_rows], validation)
    holdout = model.predict_proba(x.iloc[hold_rows])[:, order]
    test_probabilities = model.predict_proba(x_test)[:, order]
    model.save_model(str(destination / "model.cbm"))
    metrics = {
        "device": task, "tree_count": model.tree_count_,
        "rows": dict(train=len(train_rows), validation=len(val_rows), holdout=len(hold_rows)),
        "classes": classes.tolist(), "id": key, "target": target,
        "multipliers": dict(zip(classes.tolist(), multipliers.tolist())),
    }
    split = np.empty(len(y), dtype=object)
    for name, rows, probabilities in (
        ("validation", val_rows, validation), ("holdout", hold_rows, holdout),
    ):
        metrics[name] = {
            "macro_f1": macro_f1(y[rows], probabilities.argmax(axis=1), len(classes)),
            "tuned_macro_f1": macro_f1(y[rows], (probabilities * multipliers).argmax(axis=1), len(classes)),
        }
    for name, rows in (("train", train_rows), ("validation", val_rows), ("holdout", hold_rows)):
        split[rows] = name
    pd.DataFrame({key: train[key].to_list(), "split": split}).to_csv(
        destination / "splits.csv", index=False,
    )
    (destination / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return classes, test_probabilities, multipliers

