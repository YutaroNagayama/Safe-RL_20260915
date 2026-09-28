import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["UAV_SAFE_MARL_DISABLE_LIVE"] = "1"
