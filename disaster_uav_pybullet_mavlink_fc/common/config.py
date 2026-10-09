"""Neutral configuration loader for autonomy code (does not import the simulator)."""
import os

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(ROOT, "config", "drone.yaml")


def load_config(path=None) -> dict:
    with open(path or os.environ.get("UAV_CONFIG") or DEFAULT_CONFIG, encoding="utf-8") as f:   # UAV_CONFIG: alternate YAML for controlled experiments
        return yaml.safe_load(f)
