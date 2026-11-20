from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
from unittest.mock import patch
import zipfile

import numpy as np
import pytest
import torch
import yaml
from tqdm.std import tqdm

from scripts import main_additional_runner as additional
from scripts.main_incremental_checkpoint import IncrementalNotebookTrainer
from uav_safe_marl import build_env
from uav_safe_marl.experiments import main
from uav_safe_marl.evaluation import load_scenario_bank
from uav_safe_marl.runners.trainer import SafeRLTrainer


def equal_tree(a, b):
    if hasattr(a, '__dataclass_fields__'):
        equal_tree({k:getattr(a,k) for k in a.__dataclass_fields__}, {k:getattr(b,k) for k in b.__dataclass_fields__})
    elif isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a: equal_tree(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b): equal_tree(x, y)
    else:
        assert a == b


def bars(**kwargs):
    return tqdm(file=StringIO(), **kwargs)


@pytest.fixture
def parent(tmp_path):
    settings = yaml.safe_load(Path('notebooks/main/settings.yaml').read_text())
    settings.update(training_seeds=[0], agent_counts=[2], online_steps=6, checkpoint_steps=[3,6],
                    cost_warmup_updates=1, development_per_n=1, calibration_per_n=1, calibration_repeats=2)
    for spec in settings['banks'].values(): spec['per_n'] = 2
    cfg = settings['base_config']
    cfg['learning'].update(batch_size=2, buffer_size=128, cost_target_n_step=2)
    cfg['training'].update(learning_starts=3, dual_learning_starts=3, dual_batch_size=2)
    cfg['graph'].update(actor_embedding_dim=4, actor_hidden_dim=8, actor_attention_heads=1,
                        critic_embedding_dim=8, critic_hidden_dim=8, critic_layers=1, critic_attention_heads=1)
    cfg['monitoring']['enabled'] = False
    cfg['device'] = 'cpu'
    source = tmp_path/'settings.yaml'; source.write_text(yaml.safe_dump(settings))
    root = tmp_path/'parent'; data = main.prepare(source, root)
    data['max_steps'] = 3
    main.write(root/'experiment.json', data)
    return root, data


def make_suite(parent, tmp_path):
    output = tmp_path/'additional'
    additional.prepare_additional(parent[0], output, target=6)
    return output


def make_legacy_models(parent):
    for method in ('A','B','F'):
        main.train_method(parent[0], method, 0, 6)


def test_shared_banks_and_distinct_method_identity(parent, tmp_path):
    original_hash = SafeRLTrainer._source_hash()
    output = make_suite(parent, tmp_path)
    suite = additional.context(output)
    for method in ('F','B50'):
        child = output/'experiments'/method
        data = main.context(child)
        cfg = main.config_for(data, 'F' if method=='F' else 'B', 0, 6)
        assert cfg.safety.d_eng == 50
        if method=='F':
            assert cfg.learning.d_cost == 1.5 and not cfg.objective.intervention_cost_enabled
            assert cfg.communication.information_mode == 'prior_share'
        else:
            assert cfg.learning.d_cost == 6 and cfg.objective.intervention_cost_enabled
            assert cfg.communication.information_mode == 'reactive_share'
            assert cfg.communication.reactive_radius == 50
        for name in suite['bank_hashes']:
            assert (child/'banks'/f'{name}.json').read_bytes() == (parent[0]/'banks'/f'{name}.json').read_bytes()
    assert SafeRLTrainer._source_hash() == original_hash == suite['source_sha256']
    assert additional.prepare_additional(parent[0], output, target=6) == suite
    mirror=output/'parent_experiment.json';original=mirror.read_bytes();mirror.write_text('{}')
    with pytest.raises(ValueError,match='Parent snapshot changed'):additional.context(output)
    mirror.write_bytes(original)
    with pytest.raises(ValueError, match='identity changed'):
        additional.prepare_additional(parent[0], output, target=9)
    with pytest.raises(ValueError, match='Only new F'):
        additional.train_additional(output, methods=('D50',), progress_factory=bars)
    with pytest.raises(FileNotFoundError, match='評価モデル'):
        additional.evaluate_grid(output, methods=('C',), progress_factory=bars)


def test_incremental_ring_resume_matches_full_checkpoint_and_next_update(parent, tmp_path):
    root, data = parent
    cfg = main.config_for(data, 'A', 0, 3)
    cfg.learning.buffer_size = 4  # Overwrite the physical ring between saves.
    bank = load_scenario_bank(root/'banks/training.json')
    output = tmp_path/'resume'; checkpoint = output/'latest.pt'
    t = IncrementalNotebookTrainer(cfg, build_env(cfg), output/'online', report=lambda *a,**k:None, warmup_updates=1)
    try:
        t.train(None, scenarios=bank, live_display=False)
        t.save_checkpoint(checkpoint)
        first = torch.load(checkpoint, weights_only=False)
        first_chunk = deepcopy(first['extra_metadata']['incremental_resume']['chunks'][0])
        assert 'replay_buffer' not in first
        cfg.training.total_environment_steps_budget = 6
        t.train(None, scenarios=bank, live_display=False)
        t.save_checkpoint(checkpoint)
        second = torch.load(checkpoint, weights_only=False)
        refs = second['extra_metadata']['incremental_resume']['chunks']
        assert len(refs)==2 and refs[0]==first_chunk and refs[1]['slots']==3
        t.save_checkpoint(checkpoint)
        assert len(list(output.rglob('*_replay/*.pt')))==2
        full = output/'reference_full.pt'
        SafeRLTrainer.save_checkpoint(t, full)
        cfg.training.total_environment_steps_budget = 9
        t.train(None, scenarios=bank, live_display=False)
        expected = deepcopy(t.backend.state_dict()); replay = deepcopy(t.buffer.state_dict())
        initial = deepcopy(t.initial_state_buffer.state_dict()); rng = main.capture_rng()
    finally:
        t.env.close()
    for kind, path in [('incremental',checkpoint), ('legacy',full)]:
        config = deepcopy(cfg)
        restored = IncrementalNotebookTrainer(config, build_env(config), output/f'online_{kind}',
            report=lambda *a,**k:None, warmup_updates=1)
        try:
            restored.load_checkpoint(path, allow_budget_extension=True)
            restored.train(None, scenarios=bank, live_display=False)
            equal_tree(expected, restored.backend.state_dict())
            equal_tree(replay, restored.buffer.state_dict())
            equal_tree(initial, restored.initial_state_buffer.state_dict())
            equal_tree(rng, main.capture_rng())
        finally:
            restored.env.close()


def test_incremental_atomic_commit_and_corrupt_chunk(parent, tmp_path):
    root, data = parent
    cfg = main.config_for(data,'A',0,3);bank=load_scenario_bank(root/'banks/training.json')
    output=tmp_path/'atomic';checkpoint=output/'latest.pt'
    t=IncrementalNotebookTrainer(cfg,build_env(cfg),output/'online',report=lambda *a,**k:None,warmup_updates=1)
    try:
        t.train(None,scenarios=bank,live_display=False);t.save_checkpoint(checkpoint)
        before=checkpoint.read_bytes();cfg.training.total_environment_steps_budget=6
        t.train(None,scenarios=bank,live_display=False)
        import os
        replace=os.replace
        def fail_commit(src,dst):
            if Path(dst)==checkpoint:raise OSError('interrupted commit')
            return replace(src,dst)
        with patch('scripts.main_incremental_checkpoint.os.replace',fail_commit):
            with pytest.raises(OSError,match='interrupted'):
                t.save_checkpoint(checkpoint)
        assert checkpoint.read_bytes()==before
        t.save_checkpoint(checkpoint)
        payload=torch.load(checkpoint,weights_only=False)
        first=output/payload['extra_metadata']['incremental_resume']['chunks'][0]['file']
        with first.open('ab') as stream:stream.write(b'corrupted')
        fresh=IncrementalNotebookTrainer(deepcopy(cfg),build_env(cfg),output/'online_restored',report=lambda *a,**k:None,warmup_updates=1)
        try:
            with pytest.raises(ValueError,match='hash differs'):fresh.load_checkpoint(checkpoint)
        finally:fresh.env.close()
    finally:t.env.close()


def test_incremental_cost_warmup_resume_is_exact(parent, tmp_path):
    from scripts.main_notebook_runner import _warmup
    root,data=parent;cfg=main.config_for(data,'F',0,None)
    def trainer(folder):
        return IncrementalNotebookTrainer(deepcopy(cfg),build_env(cfg),folder/'warmup/collection',
            report=lambda *a,**k:None,warmup_updates=3)
    initial_rng=main.capture_rng()
    reference=trainer(tmp_path/'reference')
    try:
        _warmup(reference,root,tmp_path/'reference/warmup',3)
        expected=deepcopy(reference.backend.state_dict()); replay=deepcopy(reference.buffer.state_dict())
        initial=deepcopy(reference.initial_state_buffer.state_dict());rng=main.capture_rng()
    finally:reference.env.close()
    main.restore_rng(initial_rng)
    interrupted=trainer(tmp_path/'interrupted')
    try:_warmup(interrupted,root,tmp_path/'interrupted/warmup',1)
    finally:interrupted.env.close()
    restored=trainer(tmp_path/'interrupted')
    try:
        _warmup(restored,root,tmp_path/'interrupted/warmup',3)
        equal_tree(expected,restored.backend.state_dict())
        equal_tree(replay,restored.buffer.state_dict())
        equal_tree(initial,restored.initial_state_buffer.state_dict())
        equal_tree(rng,main.capture_rng())
    finally:restored.env.close()


def test_additional_recovers_interrupt_before_development(parent,tmp_path,monkeypatch):
    output=make_suite(parent,tmp_path)
    reference=tmp_path/'reference_suite'
    additional.prepare_additional(parent[0],reference,target=6)
    rng=main.capture_rng()
    additional.train_additional(reference,methods=('F',),progress_factory=bars)
    main.restore_rng(rng)
    original=main.evaluate_condition
    with monkeypatch.context() as patcher:
        def interrupt(*args,**kwargs):raise RuntimeError('evaluation interrupted')
        patcher.setattr(main,'evaluate_condition',interrupt)
        with pytest.raises(RuntimeError,match='evaluation interrupted'):
            additional.train_additional(output,methods=('F',),progress_factory=bars)
    assert additional.train_additional(output,methods=('F',),progress_factory=bars)==[('F',0)]
    relative='experiments/F/training/F/seed_000/latest.pt'
    expected=torch.load(reference/relative,weights_only=False)
    actual=torch.load(output/relative,weights_only=False)
    equal_tree(expected['backend'],actual['backend'])
    equal_tree(expected['trainer_progress'],actual['trainer_progress'])
    equal_tree(expected['rng_state'],actual['rng_state'])
    assert main.evaluate_condition is original


def test_b50_links_do_not_change_hocbf_monitoring(parent,tmp_path):
    output=make_suite(parent,tmp_path)
    data=main.context(output/'experiments/B50')
    cfg=main.config_for(data,'B',0,6)
    t=SafeRLTrainer(cfg,build_env(cfg),tmp_path/'radius')
    try:
        t.env.reset();a,b=t.env.states[:2];a.position=np.zeros(3);b.position=np.array([60.,0,0])
        assert t.communication.d_eng==50 and t.config.safety.d_eng==50
        assert not t.communication._linked(0,1,a,b)
        assert not t.env.safety_filter.constraints_for(0,t.env.states,np.zeros(3),t.env.active_mask)
        b.position=np.array([40.,0,0])
        assert t.communication._linked(0,1,a,b)
        assert len(t.env.safety_filter.constraints_for(0,t.env.states,np.zeros(3),t.env.active_mask))==1
    finally:t.env.close()


def test_additional_training_all_rollouts_calibration_reuse_and_export(parent, tmp_path, monkeypatch):
    make_legacy_models(parent)
    output=make_suite(parent,tmp_path)
    old_models={m:main.file_hash(parent[0]/f'training/{m}/seed_000/model_6.pt') for m in ('A','B','F')}
    assert additional.train_additional(output,progress_factory=bars)==[('F',0),('B50',0)]
    # New raw F is budget 1.5; legacy raw F remains HBA with budget 6.
    assert main.context(parent[0])['settings']['base_config']['learning']['d_cost']==6
    before=main.capture_rng()
    result=additional.evaluate_grid(output,progress_factory=bars)
    equal_tree(before,main.capture_rng())
    assert len(result)==12 and all(r['episodes']==1 for r in result)
    cases={m:next((output/f'evaluation/development/step_6/mean/{m}').rglob('cases/*/completed.json')) for m in additional.ALL_METHODS}
    records={m:json.loads(p.read_text()) for m,p in cases.items()}
    for on,off in [('A','C'),('B75','D75'),('B50','D50'),('HBA','HBA_OFF'),('F','F_OFF')]:
        assert records[on]['identity']['checkpoint_sha256']==records[off]['identity']['checkpoint_sha256']
        assert records[on]['identity']['runtime_hocbf'] and not records[off]['identity']['runtime_hocbf']
    assert records['A']['provenance']['kind']=='adopted'
    assert records['HBA']['provenance']['original_identity']['method']=='F'
    assert records['F']['metrics']['learning_budget']==1.5
    assert records['HBA']['metrics']['learning_budget']==6
    assert records['D50']['identity']['reactive_radius']==50
    assert records['D75']['identity']['reactive_radius']==75
    signatures=[]
    for case in cases.values():
        with np.load(case.parent/'rollouts/rollout_0000.npz') as z:
            signatures.append(z['initial_positions'].tobytes()+z['initial_velocities'].tobytes()+z['goals'].tobytes())
    assert len(set(signatures))==1
    cal=additional.evaluate_condition(output,'F',0,split='decision',stochastic=True)
    assert len(cal)==2 and all(r['paired_initial_action_matches_rollout'] for r in cal)
    assert all(r['discounted_episode_cost']==pytest.approx(r['discounted_distance_cost']) for r in cal)
    with pytest.raises(ValueError,match='Calibration'):
        additional.evaluate_condition(output,'F_OFF',0,split='decision',stochastic=True)
    with pytest.raises(ValueError,match='confirmation'):
        additional.evaluate_condition(output,'A',split='confirmation_test')
    # Per-case reuse works even if the grid completion record was interrupted.
    (cases['D50'].parents[2]/'completed.json').unlink()
    with monkeypatch.context() as patcher:
        patcher.setattr(additional,'evaluate',lambda *a,**k:(_ for _ in ()).throw(AssertionError('unexpected re-rollout')))
        assert len(additional.evaluate_grid(output,progress_factory=bars))==12
        assert additional.train_additional(output,progress_factory=bars)==[('F',0),('B50',0)]
    assert all(main.file_hash(parent[0]/f'training/{m}/seed_000/model_6.pt')==h for m,h in old_models.items())
    status=additional.status(output)
    assert len(status)==13 and all(r['complete'] for r in status)
    archive=additional.export_results(output)
    with zipfile.ZipFile(archive) as zip:
        names=zip.namelist()
        assert 'models/HBA/seed_000/model_6.pt' in names and 'models/F/seed_000/model_6.pt' in names
        assert not any('_replay/' in n or n.endswith('latest.pt') or n.endswith('collection.pt') for n in names)
        manifest=json.loads(zip.read('export_manifest.json'))
        import hashlib
        assert all(hashlib.sha256(zip.read(n)).hexdigest()==v['sha256'] for n,v in manifest['files'].items())
    # Reusing a wrong/mutated bank or incompatible checkpoint must fail loudly.
    raw=cases['A'].parent/'rollouts/rollout_0000.npz'
    original=raw.read_bytes();raw.unlink()
    with pytest.raises(ValueError,match='missing its raw'):
        additional.evaluate_condition(output,'A',0)
    raw.write_bytes(original)
    bank=output/'banks/development.json';bank.write_text('{}')
    with pytest.raises(ValueError,match='bank changed'):
        additional.context(output)


def test_notebook_default_flow_executes_on_tiny_common_experiment(parent,tmp_path):
    import nbformat
    make_legacy_models(parent)
    book=nbformat.read('notebooks/main/main_additional.ipynb',as_version=4)
    nbformat.validate(book)
    cells=[c.source for c in book.cells if c.cell_type=='code']
    scope={}
    exec(compile(cells[0],'notebook_setup','exec'),scope)
    output=tmp_path/'notebook_suite'
    scope.update(PARENT=parent[0],OUTPUT=output,TARGET=6,tqdm=bars)
    for index,cell in enumerate(cells[1:],1):
        exec(compile(cell,f'notebook_cell_{index}','exec'),scope)
    status=json.loads((output/'result_status.json').read_text())
    assert status['expected_development_cases']==status['completed_development_cases']==12
    assert not status['missing_development']
    assert len(scope['completed_runs'])==2
    assert len(scope['f_calibration'])==1 and scope['f_calibration'][0]['episodes']==2
    assert len(scope['result_status'])==13
