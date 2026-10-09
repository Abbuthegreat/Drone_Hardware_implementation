"""Kept for backward compatibility: identical to `python -m sim.run_fc_sim` (see that module for options)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim.run_fc_sim import main  # noqa: E402

if __name__ == "__main__":
    main()
