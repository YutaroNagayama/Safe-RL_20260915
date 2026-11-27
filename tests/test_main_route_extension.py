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
from scripts import main_route_extension_runner as route
from scripts import main_additional_runner as additional
from uav_safe_marl import build_env
from uav_safe_marl.evaluation import load_scenario_bank
from uav_safe_marl.evaluation.scenarios import predicted_interaction_data
from uav_safe_marl.experiments import main
from uav_safe_marl.runners.trainer import SafeRLTrainer

def bars(**kw):return tqdm(file=StringIO(),**kw)

@pytest.fixture
def source(tmp_path):
    settings=yaml.safe_load(Path('notebooks/main/settings.yaml').read_text())
    settings.update(training_seeds=[0],agent_counts=[2],online_steps=6,checkpoint_steps=[6],cost_warmup_updates=1,development_per_n=1,calibration_per_n=1)
    for spec in settings['banks'].values():spec['per_n']=2
    cfg=settings['base_config'];cfg['learning'].update(batch_size=2,buffer_size=128,cost_target_n_step=2)
    cfg['training'].update(learning_starts=3,dual_learning_starts=3,dual_batch_size=2)
    cfg['graph'].update(actor_embedding_dim=4,actor_hidden_dim=8,actor_attention_heads=1,critic_embedding_dim=8,critic_hidden_dim=8,critic_layers=1,critic_attention_heads=1)
    cfg['monitoring']['enabled']=False;cfg['device']='cpu'
    path=tmp_path/'settings.yaml';path.write_text(yaml.safe_dump(settings));parent=tmp_path/'parent';data=main.prepare(path,parent)
    output=tmp_path/'additional';additional.prepare_additional(parent,output,target=6)
    return output,parent,data


def test_exact_time_shift_and_initial_separation():
    p=np.array([[-100.,0,0],[100.,5,0]]);g=-p;g[:,1]=p[:,1];v=np.array([[5.,0,0],[-5.,0,0]])
    edges,pp=predicted_interaction_data(p,v,g,75,50)
    s={'scenario_id':'pair','positions':p.tolist(),'velocities':v.tolist(),'goals':g.tolist(),'maximum_accelerations':[3.,3.],
       'maximum_velocities':[12.,12.],'minimum_initial_separation':30.,'prediction_horizon':50.,'los_prediction_threshold':75.,
       'spatial_dimensions':[280.,280.,56.],'predicted_pair_data':pp,'predicted_los_edges':edges}
    before=deepcopy(s);x=route.extend_route(s,40)
    assert s==before
    assert x['predicted_pair_data'][0]['predicted_cpa_distance']==pytest.approx(5.)
    assert x['predicted_pair_data'][0]['predicted_cpa_time']==pytest.approx(60.)
    np.testing.assert_allclose(x['positions'],p-40*v)
    np.testing.assert_array_equal(x['goals'],g)
    assert x['prediction_horizon']==90
    for lead in [-1,float('nan')]:
        with pytest.raises(ValueError):route.extend_route(s,lead)
    bad=deepcopy(s);bad['minimum_initial_separation']=1000
    with pytest.raises(ValueError,match='separation'):route.extend_route(bad,0)


def test_bank_reproducible_paired_and_immutable(source,tmp_path):
    output=tmp_path/'route';source_hash=SafeRLTrainer._source_hash()
    d=route.prepare(source[0],output,per_n=1,counts=[2],leads=[0,20],max_attempts=5000)
    assert d['methods']==['A','B75','B50'] and d['base_geometries']==1 and d['scenarios']==2
    bank=load_scenario_bank(output/'banks/route_extension.json');assert len({s['source_geometry_id'] for s in bank})==1
    assert bank[0]['designated_risk_pairs'] and bank[0]['goals']==bank[1]['goals']
    assert route.prepare(source[0],output,per_n=1,counts=[2],leads=[0,20],max_attempts=5000)==d
    mirror=tmp_path/'mirror';route.prepare(source[0],mirror,per_n=1,counts=[2],leads=[0,20],max_attempts=5000)
    assert main.digest(load_scenario_bank(mirror/'banks/route_extension.json'))==d['bank_digest']
    assert SafeRLTrainer._source_hash()==source_hash
    with pytest.raises(ValueError,match='Immutable'):route.prepare(source[0],output,per_n=2,counts=[2],leads=[0,20],max_attempts=5000)
    path=output/'banks/route_extension.json';raw=path.read_text();path.write_text(raw.replace('"prediction_horizon": 50.0','"prediction_horizon": 51.0'))
    with pytest.raises(ValueError,match='Bank'):route.context(output)


def test_frozen_rollout_resume_export_and_wrong_model_rejection(source,tmp_path):
    src,parent,data=source;output=tmp_path/'route';route.prepare(src,output,per_n=1,counts=[2],leads=[0,1])
    # Tiny untrained test fixtures, not research checkpoints or reported results.
    for m in route.METHODS:
        cfg=main.config_for(data,'A' if m=='A' else 'B',0,6)
        if m=='B50':cfg.communication.reactive_radius=50.
        path=(parent/'training'/('A' if m=='A' else 'B')/'seed_000/model_6.pt') if m!='B50' else src/'experiments/B50/training/B/seed_000/model_6.pt'
        t=SafeRLTrainer(cfg,build_env(cfg),tmp_path/f'fixture_{m}');t.total_environment_steps=6;t.save_checkpoint(path,include_replay=False);t.env.close()
    manifest=route.READ(output/'experiment.json');manifest['max_steps']=3;main.write(output/'experiment.json',manifest)
    state=route.evaluate_grid(output,seeds=[0],progress_factory=bars,scenario_limit=1)
    assert state['completed_cases']==3 and state['expected_all_seeds_cases']==6
    files=list((output/'evaluation').rglob('rollout_0000.npz'));assert len(files)==3
    original={str(p):main.file_hash(p) for p in files}
    with patch.object(route,'load_trainer_checkpoint',side_effect=AssertionError('completed cases must not reexecute')):
        assert route.evaluate_grid(output,seeds=[0],progress_factory=bars,scenario_limit=1)['completed_cases']==3
    assert original=={str(p):main.file_hash(p) for p in files}
    archive=route.export_results(output)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        names=z.namelist();assert len([x for x in names if x.endswith('.pt')])==3
        assert len([x for x in names if x.endswith('.npz')])==3
        assert not any('runtime/' in x or 'latest.pt' in x for x in names)
        assert json.loads(z.read('export_manifest.json'))['methods']==['A','B75','B50']
    # The step progress wrapper must leave control/dynamics numerically unchanged.
    from uav_safe_marl.experiments.loading import load_trainer_checkpoint
    from uav_safe_marl.runners.evaluator import evaluate
    first=load_scenario_bank(output/'banks/route_extension.json')[0]
    direct=load_trainer_checkpoint(route.checkpoints(output,[0])['A',0],tmp_path/'direct',device='cpu',config_overrides={'environment':{'max_steps':3}})
    main.seed_rollout(0,first['scenario_id'],0)
    ref=evaluate(direct,reset_options={'scenario':first},policy_execution_mode='mean')['rollouts'][0];direct.env.close()
    saved=next((output/'evaluation/mean/A').rglob('rollout_0000.npz'))
    with np.load(saved,allow_pickle=False) as z:
        for key in ['positions','velocities','nominal_actions','safe_actions','normalized_residual_actions']:
            np.testing.assert_array_equal(z[key],ref[key])
    # Execute every real Notebook code cell on the tiny frozen common bank.
    notebook=json.loads(Path('notebooks/main/main_route_extension.ipynb').read_text());ns={};first_code=True
    for cell in notebook['cells']:
        if cell['cell_type']!='code':continue
        exec(compile(''.join(cell['source']),'main_route_extension.ipynb','exec'),ns)
        if first_code:
            ns.update(SOURCE=src,OUTPUT=output,BANK_PER_N=1,EVAL_PER_N=1,COUNTS=(2,),LEADS=(0.,1.),SEEDS=(0,),tqdm=bars,RUN_CONFIRMATION=False,RUN_V11=False)
            first_code=False
    assert ns['partial']['completed_cases']==6 and ns['partial']['requested_cases']==6
    with zipfile.ZipFile(ns['archive']) as z:assert len([x for x in z.namelist() if x.endswith('.npz')])==6
    model=route.checkpoints(output,[0])['A',0]
    with model.open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='model changed'):route.evaluate_grid(output,seeds=[0],progress_factory=bars,scenario_limit=1)
    with pytest.raises(ValueError,match='missing/changed'):route.export_results(output)
