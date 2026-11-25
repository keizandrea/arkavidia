import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
import pandas as pd

from datavidia.features import inspect, load_config, prepare
from datavidia.model import train_model
from datavidia.submission import aligned, probabilities, write_submission


def main():
    parser = argparse.ArgumentParser(description="Datavidia shipment classification")
    parser.add_argument("--config", type=Path, default=Path("config/default.toml"))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("inspect")
    for command in ("prepare", "train"):
        commands.add_parser(command).add_argument("--out", type=Path)
    trainer = commands.choices["train"]
    trainer.add_argument("--profile", default="baseline", choices=("baseline", "large"))
    trainer.add_argument("--device", choices=("auto", "gpu", "cpu"))
    blend = commands.add_parser("ensemble")
    blend.add_argument("--run", type=Path, required=True)
    blend.add_argument("--teammate", type=Path, required=True)
    blend.add_argument("--weight", type=float, default=.5, help="Teammate share")
    blend.add_argument("--out", type=Path, required=True)
    blend.add_argument("--rules-permit", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    key, target = config["data"]["id"], config["data"]["target"]
    if args.command == "inspect":
        inspect(config)
        return
    if args.command == "ensemble":
        if not args.rules_permit or not 0 <= args.weight <= 1:
            parser.error("Ensembling requires --rules-permit and a weight between 0 and 1")
        metadata = json.loads((args.run / "metrics.json").read_text(encoding="utf-8"))
        if (key, target) != (metadata["id"], metadata["target"]):
            parser.error("Configuration ID and target differ from the model run")
        classes = np.array(metadata["classes"])
        cat_path = args.run / "probabilities.csv"
        ids = pd.read_csv(cat_path, usecols=[key], dtype=str, keep_default_na=False)[key].tolist()
        cat = aligned(cat_path, key, ids, classes)
        other = aligned(args.teammate, key, ids, classes)
        mixed = (1 - args.weight) * cat + args.weight * other
        if args.out.exists():
            parser.error(f"Output already exists: {args.out}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_submission(config, ids, classes[mixed.argmax(axis=1)], args.out)
        print(args.out.resolve())
        return
    destination = args.out or Path("outputs") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination.mkdir(parents=True, exist_ok=False)
    train, test = prepare(config, destination)
    if args.command == "train":
        classes, values, multipliers = train_model(
            config, train, test, args.profile, destination, args.device,
        )
        ids = test[key].to_list()
        probabilities(ids, classes, values, key).to_csv(destination / "probabilities.csv", index=False)
        write_submission(config, ids, classes[(values * multipliers).argmax(axis=1)], destination / "submission.csv")
        print((destination / "metrics.json").read_text(encoding="utf-8"))
    print(destination.resolve())


if __name__ == "__main__":
    main()

