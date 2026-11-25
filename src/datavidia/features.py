from pathlib import Path
import tomllib

import polars as pl


def load_config(path):
    path = Path(path).resolve()
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    config["root"] = path.parent / config["data"]["root"]
    return config


def scan(path, strings=()):
    if not path.is_file():
        raise FileNotFoundError(f"Data file missing: {path}")
    if path.suffix.lower() == ".parquet":
        return pl.scan_parquet(path)
    if path.suffix.lower() == ".csv":
        return pl.scan_csv(path, schema_overrides=dict.fromkeys(strings, pl.String))
    raise ValueError(f"Expected CSV or Parquet: {path}")


def require(frame, columns, context):
    missing = set(columns) - set(frame.collect_schema().names())
    if missing:
        raise ValueError(f"{context}: missing columns {sorted(missing)}")


def unique(frame, key, context):
    values = frame[key]
    if values.null_count() or (values == "").any() or values.n_unique() != len(values):
        raise ValueError(f"{context}: {key} must be nonempty and unique")


def inspect(config):
    root = config["root"]
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() in {".csv", ".parquet"} and path.is_file():
            print(path.relative_to(root), scan(path).collect_schema())
    if not root.is_dir():
        print(f"Data directory missing: {root}")


def timestamp(column, formats):
    value = pl.col(column).cast(pl.String)
    return value.str.to_datetime(format=formats.get(column), strict=True, time_zone="UTC")


def apply_relation(base, config, relation):
    data = config["data"]
    ident, target = data["id"], data["target"]
    left, right = relation["left_key"], relation["right_key"]
    columns = relation["columns"]
    name = relation["name"]
    event, cutoff = relation.get("event_time"), relation.get("cutoff")
    kind = relation["kind"]
    if (
        kind not in {"lookup", "aggregate"} or bool(event) != bool(cutoff)
        or (event and kind != "aggregate") or {target, ident, right} & set(columns)
        or target in {left, right, event, cutoff} or event in columns
        or len(set(columns)) != len(columns)
    ):
        raise ValueError(f"Invalid relation definition: {name}")
    source = scan(config["root"] / relation["path"], [right])
    require(base, [left, *([cutoff] if cutoff else [])], name)
    require(source, [right, *columns, *([event] if event else [])], name)
    prefix = f"{name}__"
    source_key = f"{prefix}key"
    if kind == "lookup":
        extra = source.select(
            pl.col(right).cast(pl.String).alias(source_key),
            *(pl.col(col).alias(prefix + col) for col in columns),
        ).collect()
        unique(extra, source_key, name)
        extra = extra.lazy()
        join_key = left
    else:
        schema = source.collect_schema()
        if any(not schema[col].is_numeric() for col in columns):
            raise ValueError(f"{name}: aggregate columns must be numeric")
        extra = source.select(
            pl.col(right).cast(pl.String).alias(source_key),
            *columns,
            *([timestamp(event, relation.get("dates", {})).alias("__event")] if event else []),
        ).filter(pl.col(source_key).is_not_null())
        join_key = left
        if event:
            extra = base.select(
                ident, pl.col(left).cast(pl.String).alias(source_key),
                timestamp(cutoff, data.get("dates", {})).alias("__cutoff"),
            ).join(extra, on=source_key).filter(pl.col("__event") <= pl.col("__cutoff"))
            source_key = ident
            join_key = ident
        extra = extra.group_by(source_key).agg(
            pl.len().alias(prefix + "count"),
            *(pl.col(col).mean().alias(prefix + col + "__mean") for col in columns),
        )
    overlap = (set(extra.collect_schema().names()) - {source_key}) & set(base.collect_schema().names())
    if overlap:
        raise ValueError(f"{name}: feature name collision {sorted(overlap)}")
    return base.join(
        extra, left_on=join_key, right_on=source_key,
        how="left", maintain_order="left", coalesce=True,
    )


def prepare(config, destination):
    data = config["data"]
    ident, target = data["id"], data["target"]
    features = data["features"]
    relations = config.get("relations", [])
    if ident == target or {ident, target} & set(features) or len(set(features)) != len(features):
        raise ValueError("Configure distinct ID, target and feature columns")
    if not features and not relations:
        raise ValueError("Configure prediction-time features or relations. Run inspect first.")
    if not set(data.get("categorical", []) + list(data.get("dates", {}))) <= set(features):
        raise ValueError("Categorical and date columns must also be listed in data.features")
    keys = [relation["left_key"] for relation in relations]
    strings = set([ident, target, *data.get("categorical", []), *data.get("dates", {}), *keys])
    frames = {}
    for name in ("train", "test"):
        role = [target] if name == "train" else []
        source = scan(config["root"] / data[name], strings)
        require(source, [ident, *features, *role], name)
        base = source.select([ident, *features, *role]).with_columns(
            pl.col([col for col in [ident, *features, *role] if col in strings]).cast(pl.String)
        )
        for relation in relations:
            base = apply_relation(base, config, relation)
        for column, fmt in data.get("dates", {}).items():
            date = timestamp(column, {column: fmt})
            base = base.with_columns(
                date.dt.hour().alias(column + "__hour"),
                date.dt.weekday().alias(column + "__weekday"),
                date.dt.month().alias(column + "__month"),
            ).drop(column)
        frame = base.collect(engine="streaming")
        unique(frame, ident, name)
        if frame.is_empty():
            raise ValueError(f"{name} has no rows")
        frames[name] = frame
    if frames["train"].select(ident).join(frames["test"].select(ident), on=ident).height:
        raise ValueError("Train and test shipment IDs overlap")
    if frames["train"].drop(target).columns != frames["test"].columns:
        raise ValueError("Train and test feature columns differ")
    for name, frame in frames.items():
        frame.write_parquet(destination / f"{name}_features.parquet", compression="zstd")
    return frames["train"], frames["test"]

