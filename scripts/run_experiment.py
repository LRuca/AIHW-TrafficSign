import argparse
import os
import sys

from traffic_signs.config import load_config
from traffic_signs.train import run_experiment


def parse_args():
    parser = argparse.ArgumentParser(description="Run one traffic sign experiment.")
    parser.add_argument("--config", required=True, help="Path to experiment yaml.")
    parser.add_argument("--base-config", default="configs/base.yaml", help="Path to base yaml.")
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    config = load_config(project_root, args.base_config, args.config)
    run_experiment(config, project_root)


if __name__ == "__main__":
    main()
