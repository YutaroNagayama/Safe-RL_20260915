"""Immutable comparison-set registry built around existing Trainer checkpoints."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Iterable

from uav_safe_marl import build_env, load_config
from uav_safe_marl.runners.trainer import SafeRLTrainer


_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _component(value: str, label: str) -> str:
    value = str(value)
    if not _SAFE_COMPONENT.fullmatch(value) or value in {".", ".."}:
        raise ValueError(f"{label} must be a safe path component")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _registered_path(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("manifest path escapes the model-set root")
    return candidate


class ModelSet:
    """Group comparable methods/seeds without replacing the current Trainer API."""

    MANIFEST = "set_manifest.json"
    SCHEMA_VERSION = 1

    def __init__(self, root: Path, manifest: dict[str, Any]) -> None:
        self.root = root
        self.manifest = manifest

    @classmethod
    def create(
        cls,
        name: str,
        root: str | Path = "model_sets",
        *,
        expected_methods: Iterable[str],
        expected_seeds: Iterable[int],
        primary_checkpoint: str = "final",
        timestamp: str | None = None,
    ) -> "ModelSet":
        safe_name = _component(name, "name")
        methods = [_component(item, "method") for item in expected_methods]
        seeds = [int(item) for item in expected_seeds]
        checkpoint = _component(primary_checkpoint, "primary_checkpoint")
        if not methods or len(set(methods)) != len(methods):
            raise ValueError("expected_methods must be non-empty and unique")
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("expected_seeds must be non-empty and unique")
        created = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        _component(created, "timestamp")
        model_set_id = f"{created}_{safe_name}"
        destination = Path(root).expanduser().resolve() / model_set_id
        if destination.exists():
            raise FileExistsError(destination)
        destination.mkdir(parents=True)
        manifest = {
            "schema_version": cls.SCHEMA_VERSION,
            "model_set_id": model_set_id,
            "name": safe_name,
            "created_at": datetime.now().astimezone().isoformat(),
            "status": "building",
            "expected_methods": methods,
            "expected_seeds": seeds,
            "primary_checkpoint": checkpoint,
            "models": [],
            "scenario_banks": {},
        }
        instance = cls(destination, manifest)
        instance._save_manifest()
        return instance

    @classmethod
    def open(cls, path: str | Path, *, allow_incomplete: bool = False, verify: bool = True) -> "ModelSet":
        root = Path(path).expanduser().resolve()
        manifest_path = root / cls.MANIFEST
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != cls.SCHEMA_VERSION:
            raise ValueError("unsupported model-set schema")
        if manifest.get("status") != "completed" and not allow_incomplete:
            raise RuntimeError("model set is not finalized")
        instance = cls(root, manifest)
        if verify:
            instance.verify()
        return instance

    @property
    def is_finalized(self) -> bool:
        return self.manifest["status"] == "completed"

    def _ensure_mutable(self) -> None:
        if self.is_finalized:
            raise RuntimeError("finalized model sets are immutable")

    def _save_manifest(self) -> None:
        _write_json(self.root / self.MANIFEST, self.manifest)

    def training_dir(self, method: str, training_seed: int) -> Path:
        """Return the canonical existing-Trainer output directory."""
        method = _component(method, "method")
        return self.root / "models" / method / f"seed_{int(training_seed):03d}" / "training"

    def save_trainer(self, trainer: SafeRLTrainer, *, method: str, training_seed: int, checkpoint: str = "final") -> Path:
        """Save and register a Trainer using its current checkpoint implementation."""
        self._ensure_mutable()
        method = _component(method, "method")
        tag = _component(checkpoint, "checkpoint")
        seed = int(training_seed)
        if method not in self.manifest["expected_methods"]:
            raise ValueError(f"unexpected method: {method}")
        if seed not in self.manifest["expected_seeds"]:
            raise ValueError(f"unexpected training seed: {seed}")
        if trainer.config.seed != seed:
            raise ValueError("training_seed must match trainer.config.seed")
        key = (method, seed, tag)
        if any((item["method"], item["training_seed"], item["checkpoint_tag"]) == key for item in self.manifest["models"]):
            raise ValueError(f"duplicate model checkpoint: {key}")

        model_root = self.root / "models" / method / f"seed_{seed:03d}"
        checkpoint_path = model_root / "checkpoints" / f"{tag}.pt"
        config_path = model_root / "resolved_config.json"
        config_value = trainer.config.to_dict()
        if config_path.exists():
            existing = json.loads(config_path.read_text(encoding="utf-8"))
            if existing != config_value:
                raise ValueError("all checkpoints for one method/seed must use the same resolved config")
        else:
            _write_json(config_path, config_value)
        if checkpoint_path.exists():
            raise FileExistsError(checkpoint_path)
        temporary_checkpoint = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
        trainer.save_checkpoint(temporary_checkpoint)
        temporary_checkpoint.replace(checkpoint_path)

        relative_checkpoint = checkpoint_path.relative_to(self.root).as_posix()
        relative_config = config_path.relative_to(self.root).as_posix()
        record = {
            "model_id": f"{method}_seed_{seed:03d}_{tag}",
            "method": method,
            "training_seed": seed,
            "checkpoint_tag": tag,
            "checkpoint": relative_checkpoint,
            "config": relative_config,
            "config_sha256": _sha256(config_path),
            "training_dir": (model_root / "training").relative_to(self.root).as_posix(),
            "sha256": _sha256(checkpoint_path),
            "saved_at": datetime.now().astimezone().isoformat(),
        }
        self.manifest["models"].append(record)
        canonical_training = model_root / "training"
        source_training = Path(trainer.logger.output_dir).resolve()
        if source_training != canonical_training.resolve() and source_training.is_dir():
            shutil.copytree(source_training, canonical_training, dirs_exist_ok=True)
        model_manifest = {
            "method": method,
            "training_seed": seed,
            "checkpoints": [
                item for item in self.manifest["models"]
                if item["method"] == method and item["training_seed"] == seed
            ],
        }
        _write_json(model_root / "model_manifest.json", model_manifest)
        self._save_manifest()
        return checkpoint_path

    def attach_scenario_bank(self, name: str, path: str | Path) -> Path:
        """Copy a fixed scenario bank into the set and record its digest."""
        self._ensure_mutable()
        bank_name = _component(name, "scenario bank name")
        if bank_name in self.manifest["scenario_banks"]:
            raise ValueError(f"duplicate scenario bank: {bank_name}")
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = self.root / "scenario_banks" / f"{bank_name}{source.suffix or '.json'}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source != destination.resolve():
            shutil.copy2(source, destination)
        self.manifest["scenario_banks"][bank_name] = {
            "path": destination.relative_to(self.root).as_posix(),
            "sha256": _sha256(destination),
        }
        self._save_manifest()
        return destination

    def finalize(self) -> None:
        """Verify the full method/seed matrix and make model records immutable."""
        self._ensure_mutable()
        primary = self.manifest["primary_checkpoint"]
        available = {
            (item["method"], item["training_seed"])
            for item in self.manifest["models"]
            if item["checkpoint_tag"] == primary
        }
        expected = {
            (method, seed)
            for method in self.manifest["expected_methods"]
            for seed in self.manifest["expected_seeds"]
        }
        missing = sorted(expected - available)
        if missing:
            raise RuntimeError(f"missing primary checkpoints: {missing}")
        for method in self.manifest["expected_methods"]:
            signatures = []
            for seed in self.manifest["expected_seeds"]:
                record = next(
                    item for item in self.manifest["models"]
                    if item["method"] == method and item["training_seed"] == seed and item["checkpoint_tag"] == primary
                )
                config = json.loads(_registered_path(self.root, record["config"]).read_text(encoding="utf-8"))
                config = dict(config)
                config.pop("seed", None)
                signatures.append(config)
            if any(value != signatures[0] for value in signatures[1:]):
                raise RuntimeError(f"resolved configs differ across training seeds for method: {method}")
        self.verify()
        self.manifest["status"] = "completed"
        self.manifest["finalized_at"] = datetime.now().astimezone().isoformat()
        self._save_manifest()

    def verify(self) -> None:
        """Verify every registered checkpoint/config/scenario path and checksum."""
        for record in self.manifest["models"]:
            checkpoint = _registered_path(self.root, record["checkpoint"])
            config = _registered_path(self.root, record["config"])
            if not checkpoint.is_file() or not config.is_file():
                raise FileNotFoundError(checkpoint if not checkpoint.is_file() else config)
            if _sha256(checkpoint) != record["sha256"]:
                raise ValueError(f"checkpoint checksum mismatch: {record['model_id']}")
            if _sha256(config) != record["config_sha256"]:
                raise ValueError(f"config checksum mismatch: {record['model_id']}")
        for name, record in self.manifest["scenario_banks"].items():
            path = _registered_path(self.root, record["path"])
            if not path.is_file() or _sha256(path) != record["sha256"]:
                raise ValueError(f"scenario-bank checksum mismatch: {name}")

    def model_record(self, method: str, training_seed: int, checkpoint: str | None = None) -> dict[str, Any]:
        tag = checkpoint or self.manifest["primary_checkpoint"]
        matches = [
            item for item in self.manifest["models"]
            if item["method"] == method and item["training_seed"] == int(training_seed) and item["checkpoint_tag"] == tag
        ]
        if len(matches) != 1:
            raise KeyError((method, training_seed, tag))
        return dict(matches[0])

    def load_trainer(self, method: str, training_seed: int, *, checkpoint: str | None = None, output_dir: str | Path | None = None, device: str | None = None) -> SafeRLTrainer:
        """Reconstruct the exact saved architecture and load a trusted local checkpoint."""
        record = self.model_record(method, training_seed, checkpoint)
        checkpoint_path = _registered_path(self.root, record["checkpoint"])
        if _sha256(checkpoint_path) != record["sha256"]:
            raise ValueError(f"checkpoint checksum mismatch: {record['model_id']}")
        config_values = json.loads(_registered_path(self.root, record["config"]).read_text(encoding="utf-8"))
        if device is not None:
            config_values["device"] = device
        # Model sets created before the absolute-distance reward was introduced
        # must retain their original reward when resumed.
        objective = config_values.setdefault("objective", {})
        if "absolute_distance_enabled" not in objective:
            objective.update({
                "absolute_distance_enabled": False,
                "absolute_distance_scale": 280.0,
                "w_absolute_distance": 0.0,
            })
        config = load_config(overrides=config_values)
        runtime_output = Path(output_dir) if output_dir is not None else Path(tempfile.mkdtemp(prefix="uav_safe_marl_loaded_"))
        trainer = SafeRLTrainer(config, build_env(config), output_dir=runtime_output)
        trainer.load_checkpoint(checkpoint_path, resume_training=False)
        return trainer

    def describe(self) -> dict[str, Any]:
        """Return the small, stable specification used by the demo and notebooks."""
        method_specs = []
        primary = self.manifest["primary_checkpoint"]
        for method in self.manifest["expected_methods"]:
            records = [item for item in self.manifest["models"] if item["method"] == method and item["checkpoint_tag"] == primary]
            if records:
                config = json.loads(_registered_path(self.root, records[0]["config"]).read_text(encoding="utf-8"))
                method_specs.append({
                    "method": method,
                    "saved_seeds": sorted(item["training_seed"] for item in records),
                    "policy_backend": config["policy"]["backend"],
                    "actor_aggregation": config["graph"]["actor_aggregation_backend"],
                    "critic_aggregation": config["graph"]["critic_aggregation_backend"],
                    "agent_count": config["environment"]["agent_count"],
                    "max_agent_count": config["environment"].get("max_agent_count") or config["environment"]["agent_count"],
                    "information_mode": config["communication"]["information_mode"],
                    "safety_enabled": config["safety"]["enabled"],
                    "intervention_cost_enabled": config["objective"]["intervention_cost_enabled"],
                    "spatial_scaling_mode": config.get("scenario", {}).get("spatial_scaling_mode", "legacy"),
                    "spatial_dimensions": [
                        config.get("scenario", {}).get("longitudinal_length"),
                        config.get("scenario", {}).get("transverse_size"),
                        (
                            config.get("scenario", {}).get("altitude_max", 0.0)
                            - config.get("scenario", {}).get("altitude_min", 0.0)
                        ),
                    ],
                    "goal_reward": config["objective"]["goal_reward"],
                    "absolute_distance_scale": config["objective"].get("absolute_distance_scale"),
                    "w_absolute_distance": config["objective"].get("w_absolute_distance", 0.0),
                    "cal_mode": config["learning"]["cal_mode"],
                    "cal_variant": config["learning"].get("cal_variant", "episodic_dual_replay_gradient"),
                    "constraint_semantics": config["learning"].get("constraint_semantics", "episodic_start_state"),
                })
        return {
            "model_set_id": self.manifest["model_set_id"],
            "name": self.manifest["name"],
            "status": self.manifest["status"],
            "methods": list(self.manifest["expected_methods"]),
            "training_seeds": list(self.manifest["expected_seeds"]),
            "primary_checkpoint": self.manifest["primary_checkpoint"],
            "registered_checkpoint_count": len(self.manifest["models"]),
            "scenario_banks": sorted(self.manifest["scenario_banks"]),
            "method_specs": method_specs,
            "root": str(self.root),
        }

    def print_summary(self) -> None:
        specification = self.describe()
        print(f"Model Set: {specification['name']} ({specification['model_set_id']})")
        print(f"Status: {specification['status']}")
        print(f"Methods: {', '.join(specification['methods'])}")
        print(f"Training seeds: {specification['training_seeds']}")
        print(f"Primary checkpoint: {specification['primary_checkpoint']}")
        print(f"Registered checkpoints: {specification['registered_checkpoint_count']}")
        print(f"Scenario banks: {specification['scenario_banks']}")
        for item in specification["method_specs"]:
            print(
                f"- {item['method']}: seeds={item['saved_seeds']}, "
                f"policy={item['policy_backend']}, "
                f"aggregation={item['actor_aggregation']}/{item['critic_aggregation']}, "
                f"N={item['agent_count']}, communication={item['information_mode']}, "
                f"safety={item['safety_enabled']}, intervention={item['intervention_cost_enabled']}, "
                f"space={item['spatial_scaling_mode']}:{item['spatial_dimensions']}, "
                f"goal_reward={item['goal_reward']}, w_abs={item['w_absolute_distance']}, "
                f"CAL={item['cal_mode']}:{item['cal_variant']}"
            )
        print(f"Root: {specification['root']}")
