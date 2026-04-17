import copy
import os

import yaml


def _deep_update(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def _load_yaml(path):
    with open(path, "r", encoding="utf-8") as file_obj:
        data = yaml.safe_load(file_obj) or {}
    return data


def _resolve_paths(project_root, config):
    for section, key in [("data", "train_dir"), ("data", "val_dir"), ("data", "test_dir"), ("model", "checkpoint_path"), ("output_root", None)]:
        if key is None:
            value = config.get(section)
            if value and not os.path.isabs(value):
                config[section] = os.path.join(project_root, value)
            continue

        value = config.get(section, {}).get(key)
        if value and not os.path.isabs(value):
            config[section][key] = os.path.join(project_root, value)
    return config


def load_config(project_root, base_config_path, experiment_config_path):
    base_path = base_config_path if os.path.isabs(base_config_path) else os.path.join(project_root, base_config_path)
    exp_path = experiment_config_path if os.path.isabs(experiment_config_path) else os.path.join(project_root, experiment_config_path)
    config = _deep_update(_load_yaml(base_path), _load_yaml(exp_path))
    config["project_root"] = project_root
    config["resolved_base_config"] = base_path
    config["resolved_experiment_config"] = exp_path
    return _resolve_paths(project_root, config)
