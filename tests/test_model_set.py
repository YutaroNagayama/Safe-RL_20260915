import json

import numpy as np
import pytest

from uav_safe_marl import build_env, load_config
from uav_safe_marl.evaluation import save_scenario_bank
from uav_safe_marl.experiments import ModelSet
from uav_safe_marl.runners.trainer import SafeRLTrainer


def trainer_for(model_set, seed=3, agent_count=2):
    config = load_config(overrides={
        "seed": seed,
        "environment": {"agent_count": agent_count, "max_steps": 1},
        "learning": {"batch_size": 8},
    })
    return SafeRLTrainer(config, build_env(config), model_set.training_dir("full", seed))


def test_model_set_roundtrip_preserves_model_and_specification(tmp_path):
    model_set = ModelSet.create(
        "main_comparison",
        tmp_path,
        expected_methods=["full"],
        expected_seeds=[3],
        timestamp="20260917_120000",
    )
    trainer = trainer_for(model_set)
    trainer.env.reset(seed=3)
    observation = trainer._policy_observation({})
    before = trainer.backend.act(observation, deterministic=True)
    checkpoint = model_set.save_trainer(trainer, method="full", training_seed=3)
    bank = save_scenario_bank(tmp_path / "bank.json", [{
        "scenario_id": "fixed",
        "positions": [[0, 0, 0], [20, 0, 0]],
        "goals": [[10, 0, 0], [-10, 0, 0]],
        "reference_route_lengths": [10, 30],
    }])
    model_set.attach_scenario_bank("test", bank)
    model_set.finalize()

    opened = ModelSet.open(model_set.root)
    specification = opened.describe()
    assert specification["name"] == "main_comparison"
    assert specification["methods"] == ["full"]
    assert specification["training_seeds"] == [3]
    assert specification["scenario_banks"] == ["test"]
    assert specification["method_specs"][0]["actor_aggregation"] == "graph"
    assert specification["method_specs"][0]["saved_seeds"] == [3]
    loaded = opened.load_trainer("full", 3, output_dir=tmp_path / "loaded")
    loaded.env.reset(seed=3)
    after = loaded.backend.act(loaded._policy_observation({}), deterministic=True)
    np.testing.assert_allclose(before, after)
    assert checkpoint.is_file()

    with pytest.raises(RuntimeError, match="immutable"):
        opened.save_trainer(loaded, method="full", training_seed=3, checkpoint="other")


def test_model_set_refuses_finalize_when_primary_matrix_is_incomplete(tmp_path):
    model_set = ModelSet.create(
        "incomplete",
        tmp_path,
        expected_methods=["full", "pooled"],
        expected_seeds=[0],
        timestamp="20260917_120001",
    )
    with pytest.raises(RuntimeError, match="missing primary checkpoints"):
        model_set.finalize()
    with pytest.raises(RuntimeError, match="not finalized"):
        ModelSet.open(model_set.root)


def test_model_set_detects_checkpoint_corruption(tmp_path):
    model_set = ModelSet.create(
        "checksum",
        tmp_path,
        expected_methods=["full"],
        expected_seeds=[3],
        timestamp="20260917_120002",
    )
    checkpoint = model_set.save_trainer(trainer_for(model_set), method="full", training_seed=3)
    model_set.finalize()
    checkpoint.write_bytes(checkpoint.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        ModelSet.open(model_set.root)


def test_model_set_manifest_is_human_readable(tmp_path):
    model_set = ModelSet.create(
        "readable",
        tmp_path,
        expected_methods=["full"],
        expected_seeds=[0],
        timestamp="20260917_120003",
    )
    manifest = json.loads((model_set.root / "set_manifest.json").read_text())
    assert manifest["model_set_id"] == "20260917_120003_readable"
    assert manifest["status"] == "building"


def test_model_set_requires_same_non_seed_config_within_a_method(tmp_path):
    model_set = ModelSet.create(
        "config_consistency",
        tmp_path,
        expected_methods=["full"],
        expected_seeds=[0, 1],
        timestamp="20260917_120004",
    )
    model_set.save_trainer(trainer_for(model_set, seed=0, agent_count=2), method="full", training_seed=0)
    model_set.save_trainer(trainer_for(model_set, seed=1, agent_count=3), method="full", training_seed=1)
    with pytest.raises(RuntimeError, match="configs differ"):
        model_set.finalize()
