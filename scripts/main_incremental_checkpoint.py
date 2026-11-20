"""Exact Replay persistence for additional main runs, outside engine source.

Immutable chunks contain changed physical ring slots. The small checkpoint
is committed last; interrupted writes cannot invalidate its predecessor.
"""
from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
from time import perf_counter
from unittest.mock import patch
from uuid import uuid4
from weakref import WeakMethod, proxy

import numpy as np

from scripts.main_notebook_runner import NotebookTrainer
from uav_safe_marl.experiments import main
from uav_safe_marl.policies.base import require_torch

FORMAT = "main_replay_chunks_v1"


class IncrementalNotebookTrainer(NotebookTrainer):
    def __init__(self, config, env, output_dir, **kwargs):
        super().__init__(config, env, output_dir, **kwargs)
        directory = Path(output_dir).resolve()
        self.resume_root = directory.parents[1] if directory.name == "collection" else directory.parent
        self.replay_chunks = []
        self.dirty_slots = set()
        trainer = proxy(self)
        add = WeakMethod(self.buffer.add)

        def tracked_add(transition):
            buffer = trainer.buffer
            slot = (buffer._start + buffer._size) % buffer.capacity if len(buffer) < buffer.capacity else buffer._start
            add()(transition)
            trainer.dirty_slots.add(slot)

        self.buffer.add = tracked_add

    def _chunk_path(self, name):
        path = (self.resume_root / name).resolve()
        if not path.is_relative_to(self.resume_root) or path.suffix != ".pt":
            raise ValueError("Replay chunk lies outside its run directory")
        return path

    def save_checkpoint(self, path, *, include_replay=True, extra_metadata=None):
        path = Path(path).resolve()
        torch = require_torch()
        original_save = torch.save

        def save_protocol5(payload, destination, *args, **kwargs):
            kwargs["pickle_protocol"] = 5
            return original_save(payload, destination, *args, **kwargs)

        if not include_replay:
            with patch.object(torch, "save", save_protocol5):
                return super().save_checkpoint(path, include_replay=False, extra_metadata=extra_metadata)
        if not path.is_relative_to(self.resume_root):
            raise ValueError("Resume checkpoint lies outside its run directory")
        if self.collecting and path.name == "collection.pt" and self.saved_collection == self._collection_state(path):
            return
        online_state = (self.total_environment_steps, self.config.training.total_environment_steps_budget)
        if path.name == "latest.pt" and self.last_online_save == online_state:
            return
        started = perf_counter()
        self.report(detail=f"Replay差分保存中 {path.name}")
        buffer = self.buffer
        # With no known durable base, capture every occupied physical slot.
        slots = sorted(self.dirty_slots) if self.replay_chunks else [
            index for index, item in enumerate(buffer._data) if item is not None]
        chunks = list(self.replay_chunks)
        if slots:
            directory = path.parent / f"{path.stem}_replay"
            directory.mkdir(parents=True, exist_ok=True)
            chunk = directory / f"{uuid4().hex}.pt"
            temporary = chunk.with_suffix(".pt.tmp")
            original_save({"format": FORMAT, "slots": [(i, buffer._data[i]) for i in slots]},
                          temporary, pickle_protocol=5)
            os.replace(temporary, chunk)
            chunks.append({"file": str(chunk.relative_to(self.resume_root)),
                           "sha256": main.file_hash(chunk), "slots": len(slots)})
            self.record_io(chunk, started, "save_replay_delta")
        metadata = dict(extra_metadata or {})
        if "incremental_resume" in metadata:
            raise ValueError("Reserved incremental checkpoint metadata")
        metadata["incremental_resume"] = {
            "format": FORMAT, "chunks": chunks,
            "replay": {"capacity": buffer.capacity, "storage_dtype": buffer.storage_dtype.str,
                       "size": buffer._size, "start": buffer._start,
                       "rng_state": buffer.sampling_state_dict(),
                       "occupancy_by_n": dict(buffer._occupancy_by_n)},
            "initial_state_buffer": self.initial_state_buffer.state_dict(),
        }
        # Original save owns model/config/source identity, optimizer, progress,
        # global RNG and atomic replacement. No training state is advanced.
        with patch.object(torch, "save", save_protocol5):
            super().save_checkpoint(path, include_replay=False, extra_metadata=metadata)
        self.replay_chunks = chunks
        self.dirty_slots.clear()

    def load_checkpoint(self, path, **kwargs):
        path = Path(path).resolve()
        torch = require_torch()
        started = perf_counter()
        self.report("保存点読込中", detail=path.name)
        payload = torch.load(path, map_location=self.backend.device, weights_only=False)
        resume = payload.get("extra_metadata", {}).get("incremental_resume")
        chunks = []
        if resume is not None and kwargs.get("resume_training", True):
            if resume.get("format") != FORMAT:
                raise ValueError("Unknown incremental Replay format")
            state = deepcopy(resume["replay"])
            capacity = int(state["capacity"])
            if capacity != self.buffer.capacity or np.dtype(state["storage_dtype"]) != self.buffer.storage_dtype:
                raise ValueError("Incremental Replay shape/precision differs")
            state["data"] = [None] * capacity
            chunks = resume["chunks"]
            for ref in chunks:
                chunk = self._chunk_path(ref["file"])
                if main.file_hash(chunk) != ref["sha256"]:
                    raise ValueError(f"Replay chunk hash differs: {chunk.name}")
                block = torch.load(chunk, map_location="cpu", weights_only=False)
                if block.get("format") != FORMAT or len(block["slots"]) != ref["slots"]:
                    raise ValueError("Replay chunk metadata differs")
                for index, transition in block["slots"]:
                    if not 0 <= index < capacity:
                        raise ValueError("Invalid Replay slot")
                    state["data"][index] = transition
            if not 0 <= state["size"] <= capacity or not 0 <= state["start"] < capacity:
                raise ValueError("Invalid Replay ring metadata")
            occupied = {(state["start"] + i) % capacity for i in range(state["size"])}
            if any(state["data"][i] is None for i in occupied):
                raise ValueError("Incomplete Replay chunks")
            payload["replay_buffer"] = state
            payload["initial_state_buffer"] = resume["initial_state_buffer"]
        original_load = torch.load

        def already_loaded(filename, *args, **options):
            if isinstance(filename, (str, Path)) and Path(filename).resolve() == path:
                return payload
            return original_load(filename, *args, **options)

        # Reuse all original config/source/budget checks and RNG restoration,
        # including for old full checkpoints, without loading them twice.
        with patch.object(torch, "load", already_loaded):
            metadata = super().load_checkpoint(path, **kwargs)
        self.replay_chunks = list(chunks)
        self.dirty_slots.clear()
        self.record_io(path, started, "load_with_replay")
        return metadata
