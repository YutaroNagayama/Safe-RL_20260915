"""Synthetic, tiny-checkpoint integration checks; never main research results."""
from io import StringIO
import json
from pathlib import Path
from unittest.mock import patch
import zipfile
import numpy as np
import pytest
import yaml
from tqdm.std import tqdm
from scripts import main_confirmation_runner as confirmation
from scripts import main_route_extension_runner as route
from scripts import main_additional_runner as additional
from uav_safe_marl import build_env
from uav_safe_marl.evaluation import load_scenario_bank
from uav_safe_marl.experiments import main
from uav_safe_marl.runners.trainer import SafeRLTrainer


def bars(**kw): return tqdm(file=StringIO(), **kw)


@pytest.fixture
def source(tmp_path):
    settings = yaml.safe_load(Path('notebooks/main/settings.yaml').read_text())
    settings.update(training_seeds=[0], agent_counts=[2], online_steps=6, checkpoint_steps=[6],
                    cost_warmup_updates=1, development_per_n=1, calibration_per_n=1)
    for spec in settings['banks'].values(): spec['per_n'] = 2
    cfg = settings['base_config']; cfg['learning'].update(batch_size=2, buffer_size=128, cost_target_n_step=2)
    cfg['training'].update(learning_starts=3, dual_learning_starts=3, dual_batch_size=2)
    cfg['graph'].update(actor_embedding_dim=4, actor_hidden_dim=8, actor_attention_heads=1,
                        critic_embedding_dim=8, critic_hidden_dim=8, critic_layers=1, critic_attention_heads=1)
    cfg['monitoring']['enabled'] = False; cfg['device'] = 'cpu'
    path = tmp_path / 'settings.yaml'; path.write_text(yaml.safe_dump(settings))
    parent = tmp_path / 'parent'; data = main.prepare(path, parent)
    # Keep the fixture's original evaluation limit short, before taking source snapshots.
    data['max_steps'] = 3; main.write(parent / 'experiment.json', data)
    output = tmp_path / 'additional'; additional.prepare_additional(parent, output, target=6)
    for method in route.METHODS:
        cfg = main.config_for(data, 'A' if method == 'A' else 'B', 0, 6)
        if method == 'B50': cfg.communication.reactive_radius = 50.
        path = (parent / 'training' / ('A' if method == 'A' else 'B') / 'seed_000/model_6.pt') if method != 'B50' else output / 'experiments/B50/training/B/seed_000/model_6.pt'
        trainer = SafeRLTrainer(cfg, build_env(cfg), tmp_path / f'fixture_{method}')
        trainer.total_environment_steps = 6; trainer.save_checkpoint(path, include_replay=False); trainer.env.close()
    return output


def test_original_bank_freeze_and_tamper_rejection(source, tmp_path):
    output = tmp_path / 'confirm'
    d = confirmation.prepare(source, output, per_n=2, counts=(2,))
    assert confirmation.prepare(source, output, per_n=2, counts=(2,)) == d
    assert main.file_hash(source / 'banks/confirmation_test.json') == main.file_hash(output / 'banks/confirmation_test.json')
    assert d['max_steps'] == 3 and d['methods'] == ['A', 'B75', 'B50', 'E']
    assert len(confirmation.select_scenarios(output, 1)) == 1
    assert len(confirmation.select_scenarios(output, 2)) == 2
    with pytest.raises(ValueError, match='Not enough'): confirmation.prepare(source, tmp_path / 'too_many', per_n=3, counts=(2,))
    with pytest.raises(ValueError, match='Immutable'): confirmation.prepare(source, output, per_n=1, counts=(2,))
    with (output / 'banks/confirmation_test.json').open('a') as f: f.write(' ')
    with pytest.raises(ValueError, match='bank changed'): confirmation.context(output)


def test_frozen_paired_rollout_resume_expand_export(source, tmp_path):
    output = tmp_path / 'confirm'; confirmation.prepare(source, output, per_n=2, counts=(2,))
    ids = confirmation.select_scenarios(output, 1)
    state = confirmation.evaluate_grid(output, seeds=[0], scenario_ids=ids, progress_factory=bars)
    assert state['completed_cases'] == state['requested_cases'] == 4
    paths = list((output / 'evaluation').rglob('rollout_0000.npz'))
    hashes = {str(p): main.file_hash(p) for p in paths}
    with patch.object(confirmation, 'load_trainer_checkpoint', side_effect=AssertionError('must resume')):
        confirmation.evaluate_grid(output, seeds=[0], scenario_ids=ids, progress_factory=bars)
    assert hashes == {str(p): main.file_hash(p) for p in paths}
    state = confirmation.evaluate_grid(output, seeds=[0], progress_factory=bars)
    assert state['completed_cases'] == state['expected_all_seeds_cases'] == 8
    e = next(g for g in state['groups'] if g['method'] == 'E')
    assert e['episodes'] == e['source_geometries'] == 2 and e['seeds'] == [0]
    # Same initial state for every method, with no horizon or geometry extension.
    scenario = next(s for s in load_scenario_bank(output / 'banks/confirmation_test.json') if s['scenario_id'] == ids[0])
    for method in confirmation.METHODS:
        case = output / 'evaluation/mean' / method / 'seed_000/cases' / main.digest(ids[0])[:24]
        with np.load(case / 'rollouts/rollout_0000.npz', allow_pickle=False) as z:
            np.testing.assert_array_equal(z['positions'][0], scenario['positions'])
    # Progress instrumentation must not change controls/dynamics.
    from uav_safe_marl.experiments.loading import load_trainer_checkpoint
    from uav_safe_marl.runners.evaluator import evaluate
    trainer = load_trainer_checkpoint(confirmation.checkpoints(output, [0])['A', 0], tmp_path / 'direct', device='cpu')
    main.seed_rollout(0, scenario['scenario_id'], 0)
    ref = evaluate(trainer, reset_options={'scenario': scenario}, policy_execution_mode='mean')['rollouts'][0]; trainer.env.close()
    case = output / 'evaluation/mean/A/seed_000/cases' / main.digest(ids[0])[:24]
    with np.load(case / 'rollouts/rollout_0000.npz', allow_pickle=False) as z:
        for key in ['positions', 'velocities', 'nominal_actions', 'safe_actions']:
            np.testing.assert_array_equal(z[key], ref[key])
    archive = confirmation.export_results(output)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert len([p for p in z.namelist() if p.endswith('.npz')]) == 8
        assert len([p for p in z.namelist() if p.endswith('.pt')]) == 4
        assert not any('runtime/' in p for p in z.namelist())
        manifest = json.loads(z.read('export_manifest.json'))
        assert manifest['methods'] == list(confirmation.METHODS)
    model = confirmation.checkpoints(output, [0])['A', 0]
    with model.open('ab') as f: f.write(b'changed')
    with pytest.raises(ValueError, match='model changed'):
        confirmation.evaluate_grid(output, seeds=[0], progress_factory=bars)


def test_combined_notebook_all_cells(source, tmp_path):
    notebook = json.loads(Path('notebooks/main/main_route_extension.ipynb').read_text()); ns = {}; first = True
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code': continue
        exec(compile(''.join(cell['source']), 'main_route_extension.ipynb', 'exec'), ns)
        if first:
            ns.update(SOURCE=source, OUTPUT=tmp_path / 'extension', CONFIRM_OUTPUT=tmp_path / 'confirmation',
                      CONFIRM_BANK_PER_N=2, CONFIRM_EVAL_PER_N=2, BANK_PER_N=1, EVAL_PER_N=1,
                      COUNTS=(2,), LEADS=(1.,), SEEDS=(0,), tqdm=bars, RUN_V11=False)
            first = False
        # Keep the stress fixture short; the production extension limit is unchanged.
        if 'experiment' in ns and (tmp_path / 'extension/experiment.json').exists():
            d = route.READ(tmp_path / 'extension/experiment.json'); d['max_steps'] = 3
            main.write(tmp_path / 'extension/experiment.json', d)
    assert ns['confirm_partial']['completed_cases'] == 8
    assert ns['partial']['completed_cases'] == 3
    assert ns['confirm_archive'] != ns['archive']
    for key, count in [('confirm_archive', 8), ('archive', 3)]:
        with zipfile.ZipFile(ns[key]) as z:
            assert len([p for p in z.namelist() if p.endswith('.npz')]) == count


def test_git_line_endings_restore_exact_original_hash(source, tmp_path):
    bank = source / 'banks/confirmation_test.json'
    original = bank.read_bytes()
    bank.write_bytes(original.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n'))
    output = tmp_path / 'portable'
    d = confirmation.prepare(source, output, per_n=2, counts=(2,))
    assert (output / 'banks/confirmation_test.json').read_bytes() == original
    assert main.file_hash(output / 'banks/confirmation_test.json') == d['original_bank_sha256']
    bank.write_bytes(bank.read_bytes() + b' ')
    with pytest.raises(ValueError, match='bank changed'):
        confirmation.prepare(source, tmp_path / 'changed', per_n=2, counts=(2,))
