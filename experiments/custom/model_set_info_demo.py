"""Display a saved Model Set's name and comparison specification only."""

from __future__ import annotations

import argparse

from uav_safe_marl.experiments import ModelSet


def show_model_set(model_set_path: str) -> dict:
    model_set = ModelSet.open(model_set_path)
    model_set.print_summary()
    return model_set.describe()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_set", help="Path to a finalized Model Set directory")
    arguments = parser.parse_args()
    show_model_set(arguments.model_set)
