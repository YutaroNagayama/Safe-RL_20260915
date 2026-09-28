from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
BOOK_ROOT = ROOT / "notebooks" / "windows"


def _common_module():
    spec = importlib.util.spec_from_file_location("windows_notebook_common", BOOK_ROOT / "common.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_windows_notebooks_are_valid_and_code_parses():
    paths = sorted(BOOK_ROOT.glob("*.ipynb"))
    assert [path.name for path in paths] == [
        "00_generate_scenarios.ipynb",
        "01_A_full_prior.ipynb",
        "02_B_full_reactive.ipynb",
        "03_C_residual_prior_no_hocbf.ipynb",
        "04_D_residual_reactive_no_hocbf.ipynb",
        "05_E_hocbf_only.ipynb",
    ]
    for path in paths:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                ast.parse(cell.source, filename=str(path))


def test_windows_method_matrix_resolves_exactly():
    common = _common_module()
    settings = common.load_settings(ROOT)
    profile = settings["profiles"][settings["experiment"]["scale"]]
    context = {"settings": settings, "profile": profile, "max_steps": 100}
    configs = {
        method: common.resolved_method_config(ROOT, context, method, 0)
        for method in common.METHOD_LABELS
    }
    assert configs["A"].communication.information_mode == "prior_share"
    assert configs["A"].safety.enabled and configs["A"].objective.intervention_cost_enabled
    assert configs["B"].communication.information_mode == "reactive_share" and configs["B"].safety.enabled
    assert not configs["C"].safety.enabled and configs["C"].communication.information_mode == "prior_share"
    assert configs["D"].communication.information_mode == "reactive_share" and not configs["D"].safety.enabled
    assert configs["E"].control.actor_mode == "reference_only"
    assert configs["E"].communication.information_mode == "none" and configs["E"].safety.enabled


def test_windows_settings_keep_fixed_primary_conditions():
    common = _common_module()
    settings = common.load_settings(ROOT)
    scenario = settings["scenario"]
    config = settings["base_config"]
    assert (scenario["longitudinal_length"], scenario["transverse_size"]) == (280.0, 280.0)
    assert (scenario["altitude_min"], scenario["altitude_max"]) == (-28.0, 28.0)
    assert scenario["interaction_threshold"] == 75.0
    assert scenario["require_predicted_candidate"] is True
    assert (config["safety"]["d_safe"], config["safety"]["d_warn"], config["safety"]["d_eng"]) == (10.0, 30.0, 50.0)
    assert config["objective"]["absolute_distance_scale"] == 280.0
    assert config["objective"]["absolute_distance_enabled"] is False
    assert config["objective"]["progress_mode"] == "normalized_goal_delta"
    assert config["objective"]["w_residual"] == 0.01
    assert config["control"]["residual_acceleration_fraction"] == 0.5
    assert config["learning"]["d_cost"] == 3.0
    assert config["training"]["total_environment_steps_budget"] == 80000
    assert config["training"]["learning_starts"] == 2000
    assert config["training"]["updates_per_environment_step"] == 1


def test_training_bank_round_robin_balances_every_n_prefix():
    common = _common_module()
    bank = [
        {"actual_agent_count": count, "scenario_id": f"{count}-{index}"}
        for count in (2, 4, 8, 16) for index in range(3)
    ]
    ordered = common._round_robin_by_agent_count(bank, seed=7)
    assert [item["actual_agent_count"] for item in ordered[:8]] == [2, 4, 8, 16, 2, 4, 8, 16]
