import argparse
import glob
import os
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Run all experiments in a folder.")
    parser.add_argument("--config-dir", required=True, help="Directory containing yaml configs.")
    parser.add_argument("--base-config", default="configs/base.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    config_dir = os.path.abspath(args.config_dir)
    files = sorted(glob.glob(os.path.join(config_dir, "*.yaml")))
    if not files:
        raise FileNotFoundError("No yaml files found under {}".format(config_dir))

    script_dir = os.path.dirname(os.path.abspath(__file__))
    runner = os.path.join(script_dir, "run_experiment.py")

    for config_path in files:
        print("=== Running {} ===".format(config_path))
        command = [sys.executable, runner, "--base-config", args.base_config, "--config", config_path]
        subprocess.check_call(command)

    summary_script = os.path.join(script_dir, "summarize_results.py")
    subprocess.check_call([sys.executable, summary_script])


if __name__ == "__main__":
    main()
