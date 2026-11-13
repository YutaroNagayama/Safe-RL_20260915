from __future__ import annotations

from copy import deepcopy
import inspect
import json
from pathlib import Path
import textwrap
from types import MethodType

import numpy as np
import pytest
import torch
import yaml

from uav_safe_marl import build_env, load_config
from uav_safe_marl.runners.trainer import SafeRLTrainer
from uav_safe_marl.experiments.main import (prepare, config_for, train_method, evaluate_condition,
    capture_rng, restore_rng, context)
from uav_safe_marl.experiments.main_analysis import analyze


def small_settings(tmp_path):
    s = yaml.safe_load(Path('notebooks/main/settings.yaml').read_text())
    s.update(training_seeds=[0], agent_counts=[2], online_steps=6, checkpoint_steps=[3,6],
             cost_warmup_updates=1, development_per_n=1, calibration_per_n=1,
             calibration_repeats=2)
    for spec in s['banks'].values():
        spec['per_n'] = 2
    b=s['base_config']
    b['learning'].update(batch_size=2, buffer_size=128, cost_target_n_step=2)
    b['training'].update(learning_starts=3, dual_learning_starts=3, dual_batch_size=2)
    b['graph'].update(actor_embedding_dim=4, actor_hidden_dim=8, actor_attention_heads=1,
                      critic_embedding_dim=8, critic_hidden_dim=8, critic_layers=1,critic_attention_heads=1)
    b['monitoring']['enabled']=False
    b['device']='cpu'
    p=tmp_path/'settings.yaml'; p.write_text(yaml.safe_dump(s))
    root=tmp_path/'run'
    data=prepare(p,root)
    data['max_steps']=3  # Test-only short timeout, all methods get identical value.
    (root/'experiment.json').write_text(json.dumps(data))
    return root, data


def assert_tree(a,b):
    if hasattr(a, "__dataclass_fields__"):
        assert_tree({k:getattr(a,k) for k in a.__dataclass_fields__}, {k:getattr(b,k) for k in b.__dataclass_fields__})
    elif isinstance(a,torch.Tensor):
        assert torch.equal(a,b)
    elif isinstance(a,np.ndarray):
        np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: assert_tree(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b): assert_tree(x,y)
    else:
        assert a==b


def test_radius_is_independent_of_hocbf(tmp_path):
    root,data=small_settings(tmp_path)
    cfg=config_for(data,'B',0,6)
    t=SafeRLTrainer(cfg,build_env(cfg),tmp_path/'radius')
    try:
        assert t.communication.d_eng==75
        assert t.config.safety.d_eng==50
        states=t.env.reset()[0]
        assert load_config().communication.reactive_radius is None
        a=t.env.states[0]; b=t.env.states[1]
        a.position=np.zeros(3); b.position=np.array([60.,0,0])
        assert t.communication._linked(0,1,a,b)
    finally: t.env.close()


def test_main_end_to_end_and_budget_extension(tmp_path):
    root,data=small_settings(tmp_path)
    first=train_method(root,'A',0,6)
    assert first.exists()
    before=torch.load(root/'training/A/seed_000/latest.pt',weights_only=False)
    extended=train_method(root,'A',0,9)
    after=torch.load(root/'training/A/seed_000/latest.pt',weights_only=False)
    assert extended.exists()
    assert after['trainer_progress']['total_environment_steps']>before['trainer_progress']['total_environment_steps']
    assert after['trainer_progress']['scenario_bank_cursor']>before['trainer_progress']['scenario_bank_cursor']
    assert (root/'training/A/seed_000/warmup/audit.json').exists()
    rows=evaluate_condition(root,'A',0,checkpoint_step=9)
    off=evaluate_condition(root,'C',0,checkpoint_step=9)
    assert len(rows)==len(off)==2
    cal=evaluate_condition(root,'A',0,checkpoint_step=9,stochastic=True)
    assert len(cal)==2
    assert all(r['paired_initial_action_matches_rollout'] for r in cal)
    report=analyze(root,target=9,bootstrap_samples=4)
    assert report.exists()
    saved=np.load(next((root/'evaluation/decision/step_9/mean/A').rglob('*.npz')),allow_pickle=False)
    for key in ['reference_actions','residual_actions','safe_actions','nominal_actions','pure_cbf_actions',
                'safety_diagnostics_json','safety_wall_times','actor_adjacencies','information_ages']:
        assert key in saved
    assert len(saved['safety_diagnostics_json'])==len(saved['nominal_actions'])
    np.testing.assert_array_equal(saved['positions'][0],saved['initial_positions'])
    assert rows==evaluate_condition(root,'A',0,checkpoint_step=9)  # Reuse durable per-case results.
    with pytest.raises(ValueError,match='explicit'):
        evaluate_condition(root,'A',0,split='confirmation_test',checkpoint_step=9)
    with pytest.raises(ValueError,match='Only A/B/F'):
        train_method(root,'C',0,9)
    cfg=config_for(data,'A',0,12)
    cfg.learning.gamma_cost=.9
    t=SafeRLTrainer(cfg,build_env(cfg),tmp_path/'reject')
    try:
        with pytest.raises(ValueError,match='config differs'):
            t.load_checkpoint(root/'training/A/seed_000/latest.pt',allow_budget_extension=True)
    finally: t.env.close()


def test_exact_resume_preserves_next_update_and_rng(tmp_path):
    root,data=small_settings(tmp_path)
    cfg=config_for(data,'A',0,3)
    t=SafeRLTrainer(cfg,build_env(cfg),tmp_path/'first')
    try:
        bank=json.loads((root/'banks/training.json').read_text())['scenarios']
        t.train(None,scenarios=bank,live_display=False)
        ck=tmp_path/'resume.pt';t.save_checkpoint(ck)
        t.config.training.total_environment_steps_budget=6
        t.train(None,scenarios=bank,live_display=False)
        expected=deepcopy(t.backend.state_dict()); expected_replay=t.buffer.state_dict()
        expected_rng=capture_rng()
    finally:t.env.close()
    cfg2=config_for(data,'A',0,6)
    restored=SafeRLTrainer(cfg2,build_env(cfg2),tmp_path/'restored')
    try:
        restored.load_checkpoint(ck,allow_budget_extension=True)
        restored.train(None,scenarios=bank,live_display=False)
        assert_tree(expected,restored.backend.state_dict())
        assert_tree(expected_replay,restored.buffer.state_dict())
        assert_tree(expected_rng,capture_rng())
    finally:restored.env.close()


def test_n_step_optimization_preserves_update_exactly(tmp_path):
    root,data=small_settings(tmp_path)
    cfg=config_for(data,'A',0,None)
    t=SafeRLTrainer(cfg,build_env(cfg),tmp_path/'collect')
    try:
        bank=json.loads((root/'banks/warmup.json').read_text())['scenarios']
        t.train(2,scenarios=bank,update_mode='collect_only',live_display=False)
        batch=t.buffer.sample(2); cost=t.buffer.sample_n_step(2,2,.99)
        baseline=deepcopy(t.backend.state_dict()); rng=capture_rng()
        optimized=t.backend.update(batch,cost_batch=cost)
        expected=deepcopy(t.backend.state_dict()); expected_rng=capture_rng()
        t.backend.load_state_dict(baseline)
        source=textwrap.dedent(inspect.getsource(type(t.backend).update))
        start=source.index('        if cost_batch is None:\n            next_cost_members')
        end=source.index('    reward_loss =', start)
        reference_block=source[start:end].splitlines()[1:]
        source=source[:start]+'\n'.join(line[4:] for line in reference_block)+'\n'+source[end:]
        namespace=dict(type(t.backend).update.__globals__)
        exec(source,namespace)
        restore_rng(rng)
        reference=MethodType(namespace['update'],t.backend)(batch,cost_batch=cost)
        assert_tree(optimized,reference)
        assert_tree(expected,t.backend.state_dict())
        assert_tree(expected_rng,capture_rng())
    finally:t.env.close()


def test_tuning_branch_preserves_parent_and_moments(tmp_path):
    from uav_safe_marl.experiments.main import fork_tuning
    root,data=small_settings(tmp_path)
    train_method(root,'F',0,6)
    checkpoint=root/'training/F/seed_000/latest.pt'
    parent=torch.load(checkpoint,weights_only=False)
    branch=tmp_path/'branch'
    fork_tuning(root,branch,'F',0,learning_rate=.0001)
    child=torch.load(branch/'training/F/seed_000/latest.pt',weights_only=False)
    assert parent['config']['policy']['learning_rate']==.0003
    assert child['config']['policy']['learning_rate']==.0001
    assert_tree(parent['backend']['actor'],child['backend']['actor'])
    assert_tree(parent['replay_buffer'],child['replay_buffer'])
    assert_tree(parent['rng_state'],child['rng_state'])
    assert_tree(parent['backend']['actor_optimizer']['state'],child['backend']['actor_optimizer']['state'])
    assert child['backend']['actor_optimizer']['param_groups'][0]['lr']==.0001
    assert context(branch)['parent']['method']=='F'
    train_method(branch,'F',0,9)
    row=evaluate_condition(branch,'F',0,checkpoint_step=9)[0]
    assert row['learning_intervention_cost'] is False
    assert row['discounted_common_cost']>=row['discounted_episode_cost']-1e-8
    off=evaluate_condition(branch,'F_OFF',0,checkpoint_step=9)
    assert len(off)==2


def test_four_face_banks_follow_main_layout(tmp_path):
    settings = yaml.safe_load(Path('notebooks/main/settings.yaml').read_text())
    for spec in settings['banks'].values():
        spec['per_n'] = 2
    path = tmp_path / 'four_faces.yaml'
    path.write_text(yaml.safe_dump(settings))
    root = tmp_path / 'four_faces'
    prepare(path, root)
    axes = {'x': 0, 'y': 1}
    saw_repeated_entry = saw_adjacent_exit = False
    for bank in (root / 'banks').glob('*.json'):
        for scene in json.loads(bank.read_text())['scenarios']:
            entries, exits = scene['entry_faces'], scene['exit_faces']
            assert all(entry != exit_face for entry, exit_face in zip(entries, exits))
            saw_repeated_entry |= len(set(entries)) < len(entries)
            saw_adjacent_exit |= any(entry[0] != exit_face[0] for entry, exit_face in zip(entries, exits))
            assert scene['predicted_interaction_edges']
            for faces, points in [(entries, scene['positions']), (exits, scene['goals'])]:
                for face, point in zip(faces, points):
                    assert point[axes[face[0]]] == (-140 if face.endswith('min') else 140)
                assert all(np.linalg.norm(np.subtract(a, b)) >= 30
                           for i, a in enumerate(points) for b in points[i+1:])
    assert saw_repeated_entry and saw_adjacent_exit


def test_notebook_device_probe_reports_training_and_evaluation_separately(tmp_path):
    import sys
    from scripts.main_notebook_runner import probe_training_device

    settings = tmp_path / 'settings.yaml'
    settings.write_text('base_config: {device: cpu}\nevaluation_device: cpu\n')
    info = probe_training_device(settings)
    assert Path(info['python']) == Path(sys.executable)
    assert info['requested_device'] == info['training_device'] == info['evaluation_device'] == 'cpu'
    assert info['cuda_device_name'] is None
    assert info['torch_cuda'] == torch.version.cuda
    assert list(tmp_path.iterdir()) == [settings]


def test_notebook_reports_collection_save_and_avoids_duplicate(monkeypatch, tmp_path):
    from scripts.main_notebook_runner import NotebookTrainer

    trainer = NotebookTrainer.__new__(NotebookTrainer)
    trainer.collecting = True
    trainer.completed_episodes = 100
    trainer.total_environment_steps = 30000
    trainer.scenario_bank_hash = "warmup-bank"
    trainer.saved_collection = None
    events = []
    trainer.report = lambda *args, **fields: events.append(fields["detail"])
    trainer.record_io = lambda *args: None
    monkeypatch.setattr(SafeRLTrainer, 'save_checkpoint', lambda self, path: events.append(('save', path.name)))
    monkeypatch.setattr(SafeRLTrainer, 'load_checkpoint', lambda self, path: {})
    trainer.save_checkpoint(tmp_path/'collection.pt')
    trainer.save_checkpoint(tmp_path/'collection.pt')
    assert events == ['保存中 collection.pt', ('save', 'collection.pt'), '']
    trainer.saved_collection = None
    events.clear()
    trainer.load_checkpoint(tmp_path/'collection.pt')
    trainer.save_checkpoint(tmp_path/'collection.pt')
    assert events == ['collection.pt']


def test_notebook_grid_runs_serially_and_recovers_missing_development(tmp_path, monkeypatch, capsys):
    from io import StringIO
    from tqdm.std import tqdm
    from scripts import main_notebook_runner as launcher

    root, _ = small_settings(tmp_path)
    old_run = root/'training/B/seed_000'
    old_collection = old_run/'warmup/collection'
    old_collection.mkdir(parents=True)
    old_status = {'phase': 'collection', 'token': 'old-worker', 'steps': 987654, 'episodes': 2}
    (old_run/'notebook_progress.json').write_text(json.dumps(old_status))
    (old_collection/'metrics.jsonl').write_text(
        '{"episode":1,"cumulative_environment_steps":987654,"completed_episodes":2}\n')
    started = []
    original = launcher._run_with_progress

    def record(output, method, seed, target, *args):
        started.append((method, seed))
        if method != 'A':
            previous = 'A' if method == 'B' else 'B'
            assert (launcher._development_path(output, previous, seed, target)/'completed.json').is_file()
        return original(output, method, seed, target, *args)

    monkeypatch.setattr(launcher, '_run_with_progress', record)
    display = StringIO()
    bars = []

    def make_bar(**kwargs):
        bar = tqdm(file=display, **kwargs)
        bars.append(bar)
        return bar

    completed = launcher.run_training_grid(tmp_path/'settings.yaml', root, target=3,
                                           progress_factory=make_bar)
    assert completed == started == [('A', 0), ('B', 0), ('F', 0)]
    assert len(bars) == 1
    for method in ('A', 'B', 'F'):
        assert f'{method}/0' in display.getvalue()
    assert '987,654' not in display.getvalue()
    assert '2/2 ep' in display.getvalue()
    assert '1/1 更新' in display.getvalue()
    assert '3/3 step, ep 1' in display.getvalue()
    assert 'development完了' in display.getvalue()
    assert '1/1 件' in display.getvalue()
    status = json.loads((old_run/'notebook_progress.json').read_text())
    assert status == old_status
    assert not capsys.readouterr().out
    model = root/'training/A/seed_000/model_3.pt'
    before = model.stat().st_mtime_ns
    evaluation = root/'evaluation/development/step_3/mean/A/seed_000/completed.json'
    evaluation.unlink()
    count = len(started)
    assert launcher.run_training_grid(tmp_path/'settings.yaml', root,
                                      target=3, progress_factory=make_bar) == completed
    assert started[count:] == [('A', 0)]
    assert evaluation.exists()
    assert model.stat().st_mtime_ns == before


def test_notebook_resumes_legacy_warmup_then_small_checkpoint_exactly(tmp_path):
    from scripts import main_notebook_runner as notebook
    from uav_safe_marl.evaluation import load_scenario_bank

    root, data = small_settings(tmp_path)
    bank = load_scenario_bank(root / "banks/warmup.json")
    warm = root / "training/B/seed_000/warmup"
    cfg = config_for(data, "B", 0, None)

    def update(trainer):
        trainer.backend.update_cost_critic_only_n_step(
            trainer.buffer.sample_n_step(2, 2, cfg.learning.gamma_cost))

    old = SafeRLTrainer(cfg, build_env(cfg), warm / "collection")
    try:
        old.train(len(bank), scenarios=bank, update_mode="collect_only", live_display=False)
        update(old)
        old.save_checkpoint(warm / "progress.pt", extra_metadata={"main_warmup_updates": 1})
        update(old)
        update(old)
        expected = deepcopy(old.backend.state_dict())
        replay_rng = old.buffer.sampling_state_dict()
        expected_rng = capture_rng()
    finally:
        old.env.close()

    # A stale interrupted file is irrelevant, and legacy progress carries its own Replay.
    (warm / "progress.pt.tmp").touch()
    for updates in (2, 3):
        trainer = notebook.NotebookTrainer(cfg, build_env(cfg), warm / "collection",
                                           report=lambda *a, **kw: None, warmup_updates=updates)
        try:
            notebook._warmup(trainer, root, warm, updates)
            if updates == 3:
                assert_tree(expected, trainer.backend.state_dict())
                assert_tree(replay_rng, trainer.buffer.sampling_state_dict())
                assert_tree(expected_rng, capture_rng())
        finally:
            trainer.env.close()
    delta = torch.load(warm / "cost_resume.pt", weights_only=False)
    assert delta["extra_metadata"]["base_file"] == "progress.pt"
    assert delta["extra_metadata"]["main_warmup_updates"] == 3
    assert "replay_buffer" not in delta and "initial_state_buffer" not in delta


def test_notebook_handoff_and_reduced_batch_match_existing_runner(tmp_path, monkeypatch):
    from io import StringIO
    from tqdm.std import tqdm
    from scripts import main_notebook_runner as notebook

    old_dir, new_dir = tmp_path / "old", tmp_path / "new"
    old_dir.mkdir()
    new_dir.mkdir()
    old_root, _ = small_settings(old_dir)
    new_root, _ = small_settings(new_dir)
    monkeypatch.setattr(notebook.main, "TRAINED", ("A",))
    original_config_for = notebook.main.config_for

    def frequent_progress(*args, **kwargs):
        cfg = original_config_for(*args, **kwargs)
        cfg.monitoring.step_refresh_interval = 1
        return cfg

    monkeypatch.setattr(notebook.main, "config_for", frequent_progress)
    train_method(old_root, "A", 0, 6)
    display = StringIO()
    assert notebook.run_training_grid(new_dir / "settings.yaml", new_root, target=6,
                                      progress_factory=lambda **kw: tqdm(file=display, **kw)) == [("A", 0)]
    assert "ep内 1/3 step" in display.getvalue()
    assert "1/3 step, 完了ep 0" in display.getvalue()
    checkpoint = "training/A/seed_000/latest.pt"
    expected = torch.load(old_root / checkpoint, weights_only=False)
    actual = torch.load(new_root / checkpoint, weights_only=False)
    assert_tree(expected, actual)

    def unexpected_training(*args, **kwargs):
        pytest.fail("A completed legacy run should be reused")

    monkeypatch.setattr(notebook, "_run_with_progress", unexpected_training)
    assert notebook.run_training_grid(old_dir / "settings.yaml", old_root, target=6,
                                      progress_factory=lambda **kw: tqdm(file=StringIO(), **kw)) == [("A", 0)]


@pytest.mark.parametrize("interrupted_file", ["model_3.pt", "completed.json"])
def test_notebook_restores_interrupted_milestone_before_extending(tmp_path, monkeypatch, interrupted_file):
    from io import StringIO
    from tqdm.std import tqdm
    from scripts import main_notebook_runner as notebook

    root, _ = small_settings(tmp_path)
    monkeypatch.setattr(notebook.main, "TRAINED", ("A",))
    save = notebook.NotebookTrainer.save_checkpoint
    write = notebook.main.write

    def interrupt_save(self, path, **kwargs):
        if Path(path).name == interrupted_file:
            raise KeyboardInterrupt()
        return save(self, path, **kwargs)

    def interrupt_write(path, data):
        if path.name == interrupted_file:
            raise KeyboardInterrupt()
        return write(path, data)

    def run(target):
        return notebook.run_training_grid(tmp_path / "settings.yaml", root, target=target,
                                          progress_factory=lambda **kw: tqdm(file=StringIO(), **kw))

    with monkeypatch.context() as interrupted:
        interrupted.setattr(notebook.NotebookTrainer, "save_checkpoint", interrupt_save)
        interrupted.setattr(notebook.main, "write", interrupt_write)
        with pytest.raises(KeyboardInterrupt):
            run(6)
    assert run(6) == [("A", 0)]
    for point in (3, 6):
        assert (root / f"training/A/seed_000/model_{point}.pt").is_file()
        assert (notebook._development_path(root, "A", 0, point) / "completed.json").is_file()
