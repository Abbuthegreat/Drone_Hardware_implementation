#!/usr/bin/env bash
cd "$(dirname "$0")/.." && exec "${UAV_PYTHON:-python3}" scripts/run_all_tests.py --ros-python "${UAV_PYTHON:-python3}"
