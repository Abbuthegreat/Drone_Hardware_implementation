import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "logs", "calibration")


def write_report(name, data):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name + ".json")
    with open(path, "w") as f:
        json.dump(data, f, indent=1, default=float)
    print(f"[{name}] " + json.dumps(data, default=float)[:1500])
    print(f"[{name}] report -> {path}")
    return path


def yaml_snippet(d: dict, indent="  "):
    return "\n".join(f"{indent}{k}: {v}" for k, v in d.items())
