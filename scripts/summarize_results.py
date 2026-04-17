import argparse
import glob
import json
import os

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize experiment run folders.")
    parser.add_argument("--runs-root", default="outputs/runs", help="Root directory of run folders.")
    parser.add_argument("--output", default="outputs/summaries/experiment_summary.csv", help="Output csv path.")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    for metrics_path in glob.glob(os.path.join(args.runs_root, "*", "metrics.json")):
        with open(metrics_path, "r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
        payload["run_dir"] = os.path.dirname(metrics_path)
        rows.append(payload)

    if not rows:
        raise FileNotFoundError("No metrics.json files found under {}".format(args.runs_root))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    pd.DataFrame(rows).sort_values(by=["experiment_name", "best_val_acc"], ascending=[True, False]).to_csv(
        args.output, index=False, encoding="utf-8-sig"
    )
    print("Summary saved to {}".format(args.output))


if __name__ == "__main__":
    main()
