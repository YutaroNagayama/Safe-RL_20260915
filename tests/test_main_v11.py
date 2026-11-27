from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
import torch
from scripts import main_v11_runner as v11
from scripts.main_v11_backend import V11Backend
from uav_safe_marl.experiments import main
from uav_safe_marl.evaluation import load_scenario_bank
from test_main_confirmation import source, bars


def same(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b


def prepare(source,output,critic_updates=2):
    return v11.prepare(source,output,critic_updates=critic_updates,train_single=2,train_benign_per_n=2,
        dev_per_n=1,test_per_n=1,benign_counts=(2,),replay_capacity=128)


def test_banks_disjoint_and_benign_exact(source,tmp_path):
    out=tmp_path/'v11';d=prepare(source,out)
    assert prepare(source,out)==d
    assert d['replay_strategy'].startswith('recollect')
    singles=load_scenario_bank(out/'banks/train_single.json')
    assert all(s['actual_agent_count']==1 and not s['predicted_los_edges'] for s in singles)
    benign=load_scenario_bank(out/'banks/train_benign.json')
    assert all(p['predicted_cpa_distance']>=75 for s in benign for p in s['predicted_pair_data'])
    banks=[load_scenario_bank(out/f'banks/{name}.json') for name in d['bank_digests']]
    keys=[main.digest({k:s[k] for k in ['positions','velocities','goals']}) for bank in banks for s in bank]
    assert len(keys)==len(set(keys))
    with pytest.raises(ValueError,match='Immutable'):prepare(source,out,critic_updates=3)
    file=out/'banks/train_single.json';raw=json.loads(file.read_text());raw['scenarios'][0]['positions'][0][0]+=1
    file.write_text(json.dumps(raw))
    with pytest.raises(ValueError,match='bank changed'):v11.context(out)


@pytest.mark.parametrize('force_active_cal',[False,True])
def test_stage_masks_and_multiplayer_update_parity(source,tmp_path,force_active_cal):
    out=tmp_path/'v11';d=prepare(source,out);t=v11._make_trainer(out,d,0,'cpu')
    try:
        s=load_scenario_bank(out/'banks/train_risk.json')[0]
        t.buffer.domain='risk';t.train(1,scenarios=lambda _:s,update_mode='collect_only',live_display=False)
        b=t.buffer.buffers['risk'];batch=b._collate([b._at(0),b._at(1)]);cost=b.sample_n_step(2,2,.99);mu=v11._mu(t,load_scenario_bank(out/'banks/train_risk.json')).sample(2,{2:1})
        before=deepcopy(t.backend.state_dict())
        t.backend.update_partition(batch,initial_state_batch=mu,train_actor=False,train_cost=False)
        after=t.backend.state_dict()
        for k in ['actor','actor_optimizer','cost_critics','cost_targets','alpha_optimizer','log_alpha','dual']:
            assert same(before[k],after[k]),k
        assert not same(before['reward_critics'],after['reward_critics'])
        assert not same(before['reward_targets'],after['reward_targets'])
        # Explicitly copy the legacy backend and compare one complete multiplayer update.
        from uav_safe_marl.algorithms.sac_backend import SACBackend
        args=(t.config,t.encoder.output_dim,t.config.environment.capacity,np.ones((t.config.environment.capacity,3)))
        legacy=SACBackend(*args);new=V11Backend(*args)
        legacy.load_state_dict(before);new.load_state_dict(before)
        if force_active_cal:
            def active_constraint(_):return torch.full((2,2,1),10.),torch.full((2,1),10.)
            legacy._initial_constraint_values=active_constraint;new._initial_constraint_values=active_constraint
        rng=main.capture_rng();legacy_metrics=legacy.update(batch,cost_batch=cost,initial_state_batch=mu,allow_dual_update=False)
        assert legacy_metrics['actor_update_applied']==1.
        if force_active_cal:assert legacy_metrics['effective_cal_multiplier']>0
        main.restore_rng(rng);new.update_partition(batch,cost_batch=cost,initial_state_batch=mu,train_actor=True,train_cost=True)
        assert same(legacy.state_dict(),new.state_dict())
        # Even under a positive constraint estimate, lambda remains fixed.
        new.dual.value=2.
        initial=deepcopy(new.state_dict());rng=main.capture_rng()
        new.update_partition(batch,initial_state_batch=mu,train_actor=False,train_cost=True)
        assert new.dual.value==2. and same(initial['actor'],new.state_dict()['actor'])
    finally:t.env.close()


def test_resume_extension_exact_and_eval_export(source,tmp_path):
    out=tmp_path/'resume';direct=tmp_path/'direct';prepare(source,out);prepare(source,direct)
    v11.train_seed(out,0,target=3,device='cpu',progress_factory=bars,checkpoint_steps=1)
    model=v11.train_seed(out,0,target=6,device='cpu',progress_factory=bars,checkpoint_steps=1)
    other=v11.train_seed(direct,0,target=6,device='cpu',progress_factory=bars,checkpoint_steps=1)
    a=torch.load(model,weights_only=False,map_location='cpu');b=torch.load(other,weights_only=False,map_location='cpu')
    assert same(a['backend'],b['backend'])
    assert a['extra_metadata']['main_v11']['online_steps']>=6
    assert a['extra_metadata']['main_v11']['version']=='main_v1.1'
    state=v11.evaluate_grid(out,target=6,progress_factory=bars)
    assert state['completed_evaluation_cases']==9  # one seed => A0/A1/E, three scenario domains
    # Completed results can be extracted and resumed without resimulation.
    with patch.object(v11,'evaluate',side_effect=AssertionError('must reuse completed case')):
        assert v11.evaluate_grid(out,target=6,progress_factory=bars)['completed_evaluation_cases']==9
    import zipfile
    archive=v11.export_results(out,target=6)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert len([n for n in z.namelist() if n.endswith('.npz')])==9
        assert len([n for n in z.namelist() if n.endswith('.pt')])==3
        assert not any('resume.pt' in n or 'critic_progress.pt' in n for n in z.namelist())


def test_notebook_all_three_stages(source,tmp_path):
    import nbformat
    from uav_safe_marl.runners.trainer import SafeRLTrainer
    original_source=SafeRLTrainer._source_hash()
    n=nbformat.read('notebooks/main/main_route_extension.ipynb',4);ns={};first=True
    for cell in n.cells:
        if cell.cell_type!='code':continue
        exec(compile(cell.source,'main_route_extension.ipynb','exec'),ns)
        if first:
            ns.update(SOURCE=source,OUTPUT=tmp_path/'extension',CONFIRM_OUTPUT=tmp_path/'confirmation',
                CONFIRM_BANK_PER_N=2,CONFIRM_EVAL_PER_N=1,BANK_PER_N=1,EVAL_PER_N=1,
                COUNTS=(2,),LEADS=(1.,),SEEDS=(0,),tqdm=bars,V11_OUTPUT=tmp_path/'v11',
                V11_TARGET=3,V11_DEVICE='cpu',V11_DEV_PER_N=1,V11_TEST_PER_N=1,
                V11_SETTINGS=dict(critic_updates=2,train_single=2,train_benign_per_n=2,
                                  dev_per_n=1,test_per_n=1,benign_counts=(2,),replay_capacity=128))
            first=False
        if 'experiment' in ns and (tmp_path/'extension/experiment.json').exists():
            d=v11.READ(tmp_path/'extension/experiment.json');d['max_steps']=3;main.write(tmp_path/'extension/experiment.json',d)
    assert ns['confirm_partial']['completed_cases']==4
    assert ns['partial']['completed_cases']==3
    assert ns['v11_partial']['completed_evaluation_cases']==18
    assert ns['v11_archive'].exists()
    assert SafeRLTrainer._source_hash()==original_source


def test_single_actor_update_preserves_cost_optimizer_and_targets(source,tmp_path):
    out=tmp_path/'v11';d=prepare(source,out);t=v11._make_trainer(out,d,0,'cpu')
    try:
        # Put momentum into Cost parameters first; a reward-only update must not move it.
        s=load_scenario_bank(out/'banks/train_risk.json')[0];t.buffer.domain='risk'
        t.train(1,scenarios=lambda _:s,update_mode='collect_only',live_display=False)
        b=t.buffer.buffers['risk'];batch=b._collate([b._at(0),b._at(0)])
        t.backend.update_partition(batch,train_actor=False,train_cost=True)
        cost=deepcopy(t.backend.cost_critics.state_dict());targets=deepcopy(t.backend.cost_targets.state_dict())
        params=list(t.backend.cost_critics.parameters());moments=[deepcopy(t.backend.critic_optimizer.state[p]) for p in params]
        actor=deepcopy(t.backend.actor.state_dict())
        t.backend.update_partition(batch,train_actor=True,train_cost=False)
        assert same(cost,t.backend.cost_critics.state_dict()) and same(targets,t.backend.cost_targets.state_dict())
        assert all(same(a,t.backend.critic_optimizer.state[p]) for a,p in zip(moments,params))
        assert not same(actor,t.backend.actor.state_dict())
    finally:t.env.close()


def test_small_critic_snapshot_resume_matches_uninterrupted(source,tmp_path):
    out=tmp_path/'interrupted';direct=tmp_path/'direct';prepare(source,out,critic_updates=101);prepare(source,direct,critic_updates=101)
    original=V11Backend.update_partition;calls=0
    def interrupt(self,*args,**kwargs):
        nonlocal calls
        if not kwargs['train_actor']:
            calls+=1
            if calls==101:raise RuntimeError('synthetic interruption after durable update 100')
        return original(self,*args,**kwargs)
    with patch.object(V11Backend,'update_partition',interrupt),pytest.raises(RuntimeError,match='synthetic interruption'):
        v11.train_seed(out,0,target=3,device='cpu',progress_factory=bars)
    assert (out/'training/A/seed_000/critic_progress.pt').exists()
    a=v11.train_seed(out,0,target=3,device='cpu',progress_factory=bars)
    b=v11.train_seed(direct,0,target=3,device='cpu',progress_factory=bars)
    aa=torch.load(a,map_location='cpu',weights_only=False);bb=torch.load(b,map_location='cpu',weights_only=False)
    assert same(aa['backend'],bb['backend'])
    rows=[json.loads(line) for line in (out/'training/A/seed_000/updates.jsonl').read_text().splitlines() if line]
    assert [row['update'] for row in rows]==list(range(1,aa['extra_metadata']['main_v11']['updates']+1))


def test_eval_can_expand_with_same_frozen_bank(source,tmp_path):
    out=tmp_path/'v11';v11.prepare(source,out,critic_updates=0,train_single=2,train_benign_per_n=2,
                                  dev_per_n=2,test_per_n=2,benign_counts=(2,),replay_capacity=128)
    first=v11.evaluate_grid(out,target=3,versions=('E',),per_n=1,progress_factory=bars)
    assert first['completed_evaluation_cases']==3
    paths=list((out/'evaluation').rglob('rollout_0000.npz'));hashes={str(p):main.file_hash(p) for p in paths}
    second=v11.evaluate_grid(out,target=3,versions=('E',),per_n=2,progress_factory=bars)
    assert second['completed_evaluation_cases']==6
    assert hashes=={str(p):main.file_hash(p) for p in paths}
    with pytest.raises(ValueError,match='exceeds frozen bank'):
        v11.evaluate_grid(out,target=3,versions=('E',),per_n=3,progress_factory=bars)
    parent=Path(v11.context(out)['parent_models']['0']['path'])
    with pytest.raises(ValueError,match='Wrong v1.1'):
        v11._validate_child(parent,v11.context(out),0,3)
