import os
import yaml
import logging
import argparse
from face_smoothing.detector.backend import _resource_path

def _positive_int(value):
    try:
        int_value = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if int_value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return int_value

def _get_env_int(name, default):
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return _positive_int(raw_value)
    except argparse.ArgumentTypeError:
        logging.warning(f"Invalid {name}={raw_value!r}. Using default {default}.")
        return default

def _get_env_str(name, default):
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    raw_value = raw_value.strip()
    return raw_value if raw_value else default

def load_configs():
    cfg_path = _resource_path(os.path.join("configs", "configs.yaml"))
    with open(cfg_path, "r", encoding="utf-8") as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)

    # Migrate old config structure if detected
    if "net" in cfg:
        if "model_file" in cfg["net"] and "model_name" not in cfg["net"]:
            logging.info("Migrating old config: switching from 'model_file' to 'model_name'")
            cfg["net"]["model_name"] = "buffalo_l"
            cfg["net"]["det_size"] = [640, 640]
    return cfg
