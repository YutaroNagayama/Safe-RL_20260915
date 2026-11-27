"""A_main_v1.1: resumable fixed-lambda navigation refinement outside the v1.0 engine."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import csv
import hashlib
import json
import zipfile
import numpy as np
from uav_safe_marl import build_env
from uav_safe_marl.evaluation import generate_random_traffic_bank, load_scenario_bank, save_scenario_bank
from uav_safe_marl.evaluation.scenarios import interaction_descriptor
from uav_safe_marl.evaluation.artifacts import save_evaluation, combine_evaluation_results
from uav_safe_marl.experiments import main
from uav_safe_marl.experiments.loading import load_trainer_checkpoint
from uav_safe_marl.replay import NumpyReplayBuffer, InitialStateBuffer
from uav_safe_marl.runners.trainer import SafeRLTrainer
from uav_safe_marl.runners.evaluator import evaluate
from uav_safe_marl.runtime import runtime_manifest
from scripts import main_route_extension_runner as route
from scripts import main_confirmation_runner as confirmation
from scripts.main_additional_runner import _validate_rollout
from scripts.main_v11_backend import V11Backend

READ = route.READ
DOMAINS = ('single', 'benign', 'risk')
VERSIONS = ('A_v1.0', 'A_v1.1', 'E')


def runner_hash():
    return main.digest([Path(__file__).read_text(encoding='utf-8'),
                        Path(__file__).with_name('main_v11_backend.py').read_text(encoding='utf-8'),
                        route.runner_hash(), Path(confirmation.__file__).read_text(encoding='utf-8')])


def _original_bank(source, suite, name):
    path = Path(source) / f'banks/{name}.json';raw = path.read_bytes();lf = raw.replace(b'\r\n', b'\n')
    if suite['bank_hashes'][name] not in {hashlib.sha256(b).hexdigest() for b in (raw, lf, lf.replace(b'\n', b'\r\n'))}:
        raise ValueError(f'Original {name} bank changed')
    return load_scenario_bank(path)


def _generate(parent, n, size, seed, *, benign=False, max_draws=20000):
    sc = parent['settings']['scenario'];result=[];drawn=0;batch=0
    while len(result)<size and drawn<max_draws:
        kwargs=dict(base_size=sc['longitudinal_length'],transverse_size=sc['transverse_size'],
            altitude_range=(sc['altitude_min'],sc['altitude_max']),minimum_initial_distance=sc['minimum_initial_separation'],
            d_safe=10.,d_warn=30.,d_eng=50.,los_prediction_threshold=sc['interaction_threshold'],prediction_horizon=sc['prediction_horizon'],
            initial_speed_fraction_range=(sc['initial_speed_fraction_min'],sc['initial_speed_fraction_max']),
            acceleration_range=(sc['acceleration_min'],sc['acceleration_max']),velocity_limit_range=(sc['velocity_limit_min'],sc['velocity_limit_max']),
            require_predicted_conflict=not benign and n>1,entry_layout='four_faces')
        batch_seed=int(np.random.SeedSequence([seed,n,batch]).generate_state(1)[0]);batch+=1
        for original in generate_random_traffic_bank(max(2,n),min(32,max_draws-drawn),batch_seed,**kwargs):
            drawn+=1;s=deepcopy(original)
            if n==1:
                for key in ['positions','velocities','goals','entry_faces','exit_faces','maximum_accelerations','maximum_velocities',
                            'initial_speed_fractions','reference_route_lengths','reference_routes','reference_arrival_times']:
                    s[key]=s[key][:1]
                s.update(scenario_id='single_'+original['scenario_id'],derived_from_scenario_id=original['scenario_id'],
                    generator='main_v11_single_from_four_face_route',generator_version=1,agent_count=1,actual_agent_count=1,
                    predicted_los_edges=[],predicted_interaction_edges=[],predicted_pair_data=[])
                s['scenario_descriptor']=interaction_descriptor(1,[],s['spatial_volume'],np.asarray(s['maximum_velocities']),np.asarray(s['maximum_accelerations']),10.,30.,50.)
            elif benign and any(p['predicted_cpa_distance']<75. for p in s['predicted_pair_data']):
                continue
            s['v11_domain']='single' if n==1 else ('benign' if benign else 'risk')
            result.append(s)
            if len(result)==size:break
    if len(result)!=size:raise RuntimeError(f'Insufficient exact benign/traffic scenarios N={n}; no relaxed fallback distribution')
    return result


def prepare(source, output, *, critic_updates=500, train_single=8, train_benign_per_n=8,
            dev_per_n=2, test_per_n=4, benign_counts=(2,4), replay_capacity=4000, bank_seed=2026112711):
    source,suite,parent=route._source(source);output=Path(output).resolve();counts=parent['settings']['agent_counts']
    if output==source or output.is_relative_to(source) or source.is_relative_to(output):raise ValueError('Use a separate v1.1 OUTPUT')
    for v in [train_single,train_benign_per_n,dev_per_n,test_per_n,replay_capacity]:
        if not isinstance(v,int) or v<1:raise ValueError('Bank/replay sizes must be positive integers')
    if not isinstance(critic_updates,int) or critic_updates<0:raise ValueError('Invalid critic update count')
    benign_counts=tuple(n for n in benign_counts if n in counts)
    if not benign_counts:raise ValueError('Choose benign N from original counts')
    batch=parent['settings']['base_config']['learning']['batch_size']
    if replay_capacity<batch+parent['settings']['base_config']['learning']['cost_target_n_step']:raise ValueError('Replay too small for batch/n-step')
    models={}
    for seed in suite['training_seeds']:
        paths=[source/f'models/A/seed_{seed:03d}/model_{suite["target"]}.pt',Path(suite['parent_root'])/f'training/A/seed_{seed:03d}/model_{suite["target"]}.pt']
        path=next((p for p in paths if p.is_file()),None)
        if path is None:raise FileNotFoundError('A_v1.0 model missing: '+' OR '.join(map(str,paths)))
        models[str(seed)]={'path':str(path),'sha256':main.file_hash(path)}
    spec=dict(critic_updates=critic_updates,train_single=train_single,train_benign_per_n=train_benign_per_n,
              dev_per_n=dev_per_n,test_per_n=test_per_n,benign_counts=list(benign_counts),replay_capacity=replay_capacity,bank_seed=bank_seed)
    expected=dict(schema_version=1,version='main_v1.1',parent_version='main_v1.0',kind='A_fixed_lambda_navigation_refinement',
        source_root=str(source),source_sha256=suite['source_sha256'],source_suite_digest=main.digest(suite),source_parent_digest=main.digest(parent),
        runner_sha256=runner_hash(),training_target=suite['target'],training_seeds=suite['training_seeds'],parent_models=models,
        bank_spec=spec,max_steps=parent['max_steps'],replay_strategy='recollect_frozen_v1.0_not_exact_parent_replay_resume',
        update_cycle=['single','benign','risk','risk'],constraint_mu='all_original_training_initial_states_equal_N',
        smoothing='future_work_no_control_or_reward_change')
    if (output/'experiment.json').exists():
        old=context(output)
        if any(old[k]!=v for k,v in expected.items()):raise ValueError('Immutable v1.1 specification/parent changed')
        return old
    if output.exists() and any(output.iterdir()):raise FileExistsError('Nonempty v1.1 output')
    risk=_original_bank(source,suite,'training')
    for s in risk:s['v11_domain']='risk'
    banks={'train_risk':risk,'train_single':_generate(parent,1,train_single,bank_seed),
           'train_benign':sum([_generate(parent,n,train_benign_per_n,bank_seed+1,benign=True) for n in benign_counts],[])}
    for split,size,offset in [('development',dev_per_n,10),('confirmation',test_per_n,20)]:
        banks[split]=(_generate(parent,1,size,bank_seed+offset)+
                      sum([_generate(parent,n,size,bank_seed+offset+1,benign=True) for n in benign_counts],[])+
                      sum([_generate(parent,n,size,bank_seed+offset+2) for n in counts],[]))
    # Check full geometry disjointness against all original banks, plus the new banks.
    key=lambda s:main.digest({k:s[k] for k in ['positions','velocities','goals','maximum_accelerations','maximum_velocities']})
    seen={key(s) for name in suite['bank_hashes'] for s in _original_bank(source,suite,name)}
    for name,bank in banks.items():
        if name=='train_risk':continue
        for s in bank:
            k=key(s)
            if k in seen:raise ValueError('Duplicate train/evaluation geometry')
            seen.add(k)
    output.mkdir(parents=True)
    for name,bank in banks.items():save_scenario_bank(output/f'banks/{name}.json',bank)
    main.write(output/'source_suite.json',suite);main.write(output/'source_parent_experiment.json',parent)
    expected['bank_digests']={name:main.digest(load_scenario_bank(output/f'banks/{name}.json')) for name in banks}
    main.write(output/'experiment.json',expected)
    # Validate using the original resolved A config, not only filenames.
    for seed,model in models.items():route._validate_model(Path(model['path']),'A',int(seed),expected,output)
    return expected


def context(output):
    output=Path(output);d=READ(output/'experiment.json')
    if d['runner_sha256']!=runner_hash() or d['source_sha256'] not in route.compatible_source_hashes():raise ValueError('v1.1 implementation changed')
    for name,h in d['bank_digests'].items():
        if main.digest(load_scenario_bank(output/f'banks/{name}.json'))!=h:raise ValueError('v1.1 bank changed')
    for name,k in [('source_suite.json','source_suite_digest'),('source_parent_experiment.json','source_parent_digest')]:
        if main.digest(READ(output/name))!=d[k]:raise ValueError('Source snapshot changed')
    return d


class DomainReplay:
    """Three independent episode-safe replays, with deterministic 1:1:2 update scheduling."""
    def __init__(self,capacity,seed,dtype):
        self.buffers={k:NumpyReplayBuffer(capacity,seed+i,dtype) for i,k in enumerate(DOMAINS)}
        self.domain='risk';self.callback=None
    def add(self,t):
        self.buffers[self.domain].add(t)
        if self.callback:self.callback()
    def __len__(self):return sum(map(len,self.buffers.values()))
    def occupancy_by_agent_count(self):
        result={}
        for b in self.buffers.values():
            for n,v in b.occupancy_by_agent_count().items():result[n]=result.get(n,0)+v
        return result
    def state_dict(self):return {k:b.state_dict() for k,b in self.buffers.items()}
    def load_state_dict(self,state):
        for k,b in self.buffers.items():b.load_state_dict(state[k])


def _atomic_torch(path,payload):
    import torch
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    torch.save(payload,tmp);tmp.replace(path)


def _make_trainer(output,d,seed,device):
    path=Path(d['parent_models'][str(seed)]['path'])
    if main.file_hash(path)!=d['parent_models'][str(seed)]['sha256']:raise ValueError('Parent model changed')
    t=load_trainer_checkpoint(path,Path(output)/f'training/A/seed_{seed:03d}',device=device,
        config_overrides={'training':{'total_environment_steps_budget':None}})
    if not t.graph_mode or t.config.policy.backend!='SAC':raise ValueError('v1.1 requires original graph SAC')
    backend=V11Backend(t.config,t.encoder.output_dim,t.config.environment.capacity,
                       np.full((t.config.environment.capacity,3),t.config.environment.acceleration_limit))
    backend.load_state_dict(t.backend.state_dict());t.backend=backend
    t.buffer=DomainReplay(d['bank_spec']['replay_capacity'],seed,np.float32 if t.config.performance.profile=='fast' else np.float64)
    import torch
    parent_progress=torch.load(path,map_location='cpu',weights_only=False)['trainer_progress']
    t.total_environment_steps=parent_progress['total_environment_steps']
    t.total_gradient_updates=parent_progress['total_gradient_updates']
    t.config.monitoring.enabled=False
    return t


def _mu(t,bank):
    """Only original training starts; no easy/single additions enter CAL's mu."""
    mu=InitialStateBuffer(len(bank),t.config.seed+710)
    for s in bank:
        t.env.reset(options={'scenario':s});t.communication.reset(s['predicted_los_edges'])
        mu.add(t._policy_observation({}),s['actual_agent_count'],s['scenario_id'])
    return mu


def _validate_child(path,d,seed,target):
    import torch
    payload=torch.load(path,map_location='cpu',weights_only=False)
    m=payload.get('extra_metadata',{}).get('main_v11',{})
    if (m.get('version')!='main_v1.1' or m.get('runner_sha256')!=d['runner_sha256']
        or m.get('parent_sha256')!=d['parent_models'][str(seed)]['sha256']
        or m.get('online_steps',-1)<target or m.get('requested_target')!=target or payload['config']['seed']!=seed
        or payload['backend']['dual']!=m.get('frozen_lambda')
        or payload['experiment_identity']['source_sha256'] not in route.compatible_source_hashes()):
        raise ValueError('Wrong v1.1 checkpoint version/parent/lambda/target')
    return m


def train_seed(output,seed,*,target=5000,device='auto',progress_factory=None,checkpoint_steps=1000):
    import torch
    output=Path(output);d=context(output)
    if seed not in d['training_seeds'] or not isinstance(target,int) or target<1 or checkpoint_steps<1:raise ValueError('Invalid seed/target/checkpoint interval')
    if progress_factory is None:
        from tqdm.auto import tqdm
        progress_factory=tqdm
    run=output/f'training/A/seed_{seed:03d}';run.mkdir(parents=True,exist_ok=True);latest=run/'resume.pt'
    published=run/f'completed_{target}.json';final=run/f'model_{target}.pt'
    if published.exists():
        if not final.exists() or main.file_hash(final)!=READ(published)['model_sha256']:raise ValueError('Published v1.1 model changed')
        _validate_child(final,d,seed,target)
        return final
    rng=main.capture_rng();t=_make_trainer(output,d,seed,device);frozen_lambda=t.backend.dual.value
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer=SummaryWriter(str(run/'tensorboard'))
    except ImportError:writer=None
    banks={k:load_scenario_bank(output/f'banks/train_{k}.json') for k in DOMAINS};mu=_mu(t,banks['risk'])
    meta=dict(version='main_v1.1',parent_sha256=d['parent_models'][str(seed)]['sha256'],
              runner_sha256=d['runner_sha256'],device=str(t.backend.device),frozen_lambda=frozen_lambda,
              collected_tasks=0,critic_updates=0,online_steps=0,online_episodes=0,updates=0,base_steps=t.total_environment_steps)
    def save():
        _atomic_torch(latest,dict(identity={k:meta[k] for k in ['version','parent_sha256','runner_sha256','device','frozen_lambda']},
            meta=meta,backend=t.backend.state_dict(),replay=t.buffer.state_dict(),mu=mu.state_dict(),
            collection_starts=t.initial_state_buffer.state_dict(),rng=main.capture_rng(),
            trainer=dict(total_environment_steps=t.total_environment_steps,total_gradient_updates=t.total_gradient_updates,
                         completed_episodes=t.completed_episodes,scenario_bank_cursor=t.scenario_bank_cursor,
                         transitions_collected_by_n=t.transitions_collected_by_n)))
        main.write(run/'progress.json',meta)
    try:
        if latest.exists():
            q=torch.load(latest,map_location='cpu',weights_only=False)
            if q['identity']!={k:meta[k] for k in q['identity']}:raise ValueError('v1.1 resume identity changed')
            meta=q['meta'];t.backend.load_state_dict(q['backend']);t.buffer.load_state_dict(q['replay']);mu.load_state_dict(q['mu']);t.initial_state_buffer.load_state_dict(q['collection_starts'])
            for k,v in q['trainer'].items():setattr(t,k,v)
            main.restore_rng(q['rng']);t.logger.truncate_after_episode(t.completed_episodes)
        def truncate_updates():
            log=run/'updates.jsonl'
            if log.exists():
                log.write_text('\n'.join(line for line in log.read_text().splitlines()
                    if line.strip() and json.loads(line)['update']<=meta['updates'])+'\n',encoding='utf-8')
        if meta['online_steps']>target:raise ValueError('Cannot relabel a later model as an earlier target')
        # Original starts are retained separately from collection starts.
        if not mu.supports(t.config.scenario.training_agent_count_distribution):raise ValueError('Original CAL mu lacks configured N')
        # Frozen v1.0 collection: real Reference actions and real stochastic Actor actions.
        tasks=[('single',s,reference) for s in banks['single'][:4] for reference in (True,False)]
        tasks += [('benign',s,False) for s in main.subset(banks['benign'],1)]
        tasks += [('risk',s,False) for s in main.subset(banks['risk'],1)]
        if meta['collected_tasks']<len(tasks):
            before=deepcopy(t.backend.state_dict());bar=progress_factory(total=len(tasks),initial=meta['collected_tasks'],desc=f'A_v1.1/{seed} collect',unit='ep')
            try:
                for index in range(meta['collected_tasks'],len(tasks)):
                    domain,s,reference=tasks[index];t.buffer.domain=domain
                    original=t.backend.act
                    if reference:t.backend.act=lambda obs,deterministic:np.zeros(t.env.action_space.shape)
                    flight=progress_factory(total=None,desc=f'{domain} collect',unit='step',leave=False)
                    old_step=t.env.step
                    def collect_step(action):
                        result=old_step(action);flight.update(1);return result
                    t.env.step=collect_step
                    try:t.train(1,scenarios=lambda _:s,update_mode='collect_only',live_display=False)
                    finally:t.backend.act=original;t.env.step=old_step;flight.close()
                    meta['collected_tasks']=index+1;bar.update(1)
                    if (index+1)%4==0:save()
                # No optimizer or model changes during collection.
                for k in ['actor','reward_critics','cost_critics','reward_targets','cost_targets']:
                    for name,value in before[k].items():
                        if not torch.equal(value,t.backend.state_dict()[k][name]):raise RuntimeError('Frozen collection changed model')
                meta['collection_steps']=t.total_environment_steps-meta['base_steps']
                save()
            finally:bar.close()
        batch=t.config.learning.batch_size;nstep=t.config.learning.cost_target_n_step
        if any(b._n_step_candidate_count(nstep)<batch for b in t.buffer.buffers.values()):
            raise ValueError('Frozen collection too short for a full batch; increase collection or reduce batch explicitly')
        def update(train_actor):
            domain=d['update_cycle'][meta['updates']%4];b=t.buffer.buffers[domain]
            rb=b.sample(batch);cb=b.sample_n_step(batch,nstep,t.config.learning.gamma_cost) if domain!='single' else None
            initial=mu.sample(t.config.training.dual_batch_size,t.config.scenario.training_agent_count_distribution)
            values=t.backend.update_partition(rb,cost_batch=cb,initial_state_batch=initial,train_actor=train_actor,train_cost=domain!='single')
            if t.backend.dual.value!=frozen_lambda:raise RuntimeError('Frozen lambda changed')
            meta['updates']+=1;t.total_gradient_updates+=1
            with (run/'updates.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(update=meta['updates'],domain=domain,stage='actor' if train_actor else 'critic',**values))+'\n')
            if writer is not None and meta['updates']%10==0:
                for key,value in values.items():writer.add_scalar(key,value,meta['updates'])
            return values
        # Small critic checkpoints refer to a fixed full replay snapshot, avoiding huge replay rewrites.
        small=run/'critic_progress.pt'
        if meta['critic_updates']<d['bank_spec']['critic_updates']:
            base_hash=main.file_hash(latest)
            if small.exists():
                q=torch.load(small,map_location='cpu',weights_only=False)
                if q['base_sha256']==base_hash:
                    meta=q['meta'];t.backend.load_state_dict(q['backend']);main.restore_rng(q['rng'])
                    for k,b in t.buffer.buffers.items():b.load_sampling_state_dict(q['replay_rng'][k])
                    mu.load_state_dict(q['mu']);t.total_gradient_updates=q['gradient_updates']
            truncate_updates()
            bar=progress_factory(total=d['bank_spec']['critic_updates'],initial=meta['critic_updates'],desc=f'A_v1.1/{seed} critic',unit='update')
            frozen=main.frozen_hashes(t)
            actor_hash=frozen['actor'];alpha=float(t.backend.alpha.detach())
            try:
                while meta['critic_updates']<d['bank_spec']['critic_updates']:
                    update(False);meta['critic_updates']+=1;bar.update(1)
                    if meta['critic_updates']%100==0:
                        _atomic_torch(small,dict(base_sha256=base_hash,meta=meta,backend=t.backend.state_dict(),rng=main.capture_rng(),
                            replay_rng={k:b.sampling_state_dict() for k,b in t.buffer.buffers.items()},mu=mu.state_dict(),gradient_updates=t.total_gradient_updates))
                if main.frozen_hashes(t)['actor']!=actor_hash or float(t.backend.alpha.detach())!=alpha:raise RuntimeError('Critic stage changed Actor/alpha')
                save()
            finally:bar.close()
        truncate_updates()
        meta['requested_target']=target
        # Online gradients run at the same causal transition-commit boundary as the original Trainer.
        steps=progress_factory(total=target,initial=min(target,meta['online_steps']),desc=f'A_v1.1/{seed} online',unit='step')
        old_step=t.env.step
        def observed(action):
            result=old_step(action);steps.update(1);return result
        t.env.step=observed
        def online_update():
            for _ in range(t.config.training.updates_per_environment_step):update(True)
        t.buffer.callback=online_update
        checkpoint_at=meta['online_steps'];cycle=d['update_cycle']
        try:
            while meta['online_steps']<target:
                idx=meta['online_episodes'];domain=cycle[idx%4];pool=banks[domain]
                offset=(idx//4)*(2 if domain=='risk' else 1)+(1 if domain=='risk' and idx%4==3 else 0)
                s=pool[offset%len(pool)]
                t.buffer.domain=domain;before=t.total_environment_steps
                history=t.train(1,scenarios=lambda _:s,update_mode='collect_only',live_display=False)
                meta['online_steps']+=t.total_environment_steps-before;meta['online_episodes']+=1
                steps.set_postfix(ep=meta['online_episodes'],refresh=True)
                # Compact model every episode, full replay only at coarse safe boundaries.
                t.save_checkpoint(run/'model_latest.pt',include_replay=False,extra_metadata={'main_v11':dict(meta)})
                if meta['online_steps']-checkpoint_at>=checkpoint_steps:
                    save();checkpoint_at=meta['online_steps']
            save()
            final=run/f'model_{target}.pt'
            if not final.exists():t.save_checkpoint(final,include_replay=False,extra_metadata={'main_v11':dict(meta)})
            main.write(run/f'completed_{target}.json',dict(target=target,actual_online_steps=meta['online_steps'],model_sha256=main.file_hash(final),**meta))
            return final
        finally:t.buffer.callback=None;t.env.step=old_step;steps.close()
    finally:
        t.env.close()
        if writer is not None:writer.close()
        main.restore_rng(rng)


def train_grid(output,*,target=5000,device='auto',progress_factory=None):
    d=context(output)
    return {s:str(train_seed(output,s,target=target,device=device,progress_factory=progress_factory)) for s in d['training_seeds']}


def _motion_metrics(rollout,cfg,scenario):
    active=np.asarray(rollout['active_masks'][:-1],bool);dt=cfg.environment.dt_base;n=int(rollout['reset_info']['actual_agent_count'])
    velocity=np.asarray(rollout['velocities']);a=np.diff(velocity,axis=0)/dt;norm=np.linalg.norm(a,axis=-1)
    pure=np.asarray(rollout['pure_cbf_actions'])-np.asarray(rollout['nominal_actions']);pure2=(pure*pure).sum(-1)
    positions=np.asarray(rollout['positions']);start=positions[0];goals=np.asarray(rollout['reset_info']['goals']);route=goals-start
    length2=(route*route).sum(-1);proj=((positions-start[None])*route[None]).sum(-1)/np.maximum(length2,1e-12)[None]
    lateral=np.linalg.norm(positions-start[None]-proj[...,None]*route[None],axis=-1)
    reward=np.asarray(rollout['rewards'])
    i,j=np.triu_indices(n,1)
    rel0=positions[:-1,i]-positions[:-1,j];rel1=positions[1:,i]-positions[1:,j];dr=rel1-rel0
    frac=np.clip(-(rel0*dr).sum(-1)/np.maximum((dr*dr).sum(-1),1e-20),0,1)
    distance=np.linalg.norm(rel0+frac[...,None]*dr,axis=-1)
    valid=active[:,i]&active[:,j];minimum=np.where(valid,distance,np.inf).min(axis=0)
    risks=[]
    initial={(p['first'],p['second']):p['predicted_cpa_distance'] for p in scenario['predicted_pair_data']}
    for k,(first,second) in enumerate(zip(i,j)):
        if initial[first,second]<30:risks.append(dict(first=int(first),second=int(second),initial_cpa=initial[first,second],minimum_distance=float(minimum[k])))
    reached=np.asarray(rollout['reached_step'])[:n]

    return dict(continuous_violation_pairs=int((minimum<10-1e-9).sum()),
                continuous_minimum_distance=float(minimum.min()) if len(minimum) else None,
                initial_risk_pairs=len(risks),initial_risk_pairs_below20=sum(r['minimum_distance']<20 for r in risks),
                initial_risk_pair_results=risks,mean_agent_arrival=float(reached.mean()*dt) if np.all(reached>=0) else None,
                actual_acceleration_norm=float(norm[active].mean()),acceleration_squared_integral_per_uav=float((norm[active]**2).sum()*dt/n),
                maximum_line_deviation=float(lateral[1:][active].max(initial=0)),pure_seconds_per_uav=float(((pure2>1e-12)&active).sum()*dt/n),
                discounted_environment_reward=float((cfg.learning.gamma_reward**np.arange(len(reward)))@reward),
                infeasible_agent_ticks=sum(bool(row[i].get('infeasible',False)) for row,mask in zip(rollout['safety_diagnostics'],active) for i in np.flatnonzero(mask)))


def evaluate_grid(output,*,target=5000,split='development',versions=VERSIONS,device='cpu',progress_factory=None,per_n=None):
    output=Path(output);d=context(output)
    if split not in ['development','confirmation'] or any(v not in VERSIONS for v in versions):raise ValueError('Unknown split/version')
    if progress_factory is None:
        from tqdm.auto import tqdm
        progress_factory=tqdm
    bank=load_scenario_bank(output/f'banks/{split}.json')
    if per_n is not None:
        if not isinstance(per_n,int) or per_n<1:raise ValueError('Invalid evaluation per-N size')
        selected=[]
        for domain in DOMAINS:
            pool=[s for s in bank if s['v11_domain']==domain]
            if any(sum(s['actual_agent_count']==n for s in pool)<per_n for n in {s['actual_agent_count'] for s in pool}):raise ValueError('Evaluation exceeds frozen bank')
            selected.extend(main.subset(pool,per_n))
        bank=selected
    tasks=[(v,s) for v in versions for s in ([0] if v=='E' else d['training_seeds'])]
    paths={}
    for version,seed in tasks:
        path=(confirmation._baseline(output,d) if version=='E' else Path(d['parent_models'][str(seed)]['path']) if version=='A_v1.0'
              else output/f'training/A/seed_{seed:03d}/model_{target}.pt')
        if not path.is_file():raise FileNotFoundError(str(path))
        if version=='A_v1.0' and main.file_hash(path)!=d['parent_models'][str(seed)]['sha256']:raise ValueError('Parent model changed')
        if version=='A_v1.1':
            _validate_child(path,d,seed,target)
            marker=output/f'training/A/seed_{seed:03d}/completed_{target}.json'
            if not marker.exists() or READ(marker)['model_sha256']!=main.file_hash(path):raise ValueError('v1.1 model completion/hash differs')
        elif version=='A_v1.0':route._validate_model(path,'A',seed,d,output)
        else:confirmation._validate_model(path,'E',0,d,output)
        paths[version,seed]=path
    bar=progress_factory(total=len(bank)*len(tasks),desc=f'v1.1 {split}',unit='case');rng=main.capture_rng()
    try:
        for version,seed in tasks:
            t=None
            try:
                identity=dict(version=version,seed=seed,model_sha256=main.file_hash(paths[version,seed]),
                    bank_digest=d['bank_digests'][split],runner_sha256=d['runner_sha256'],device=device,system=runtime_manifest()['system'])
                for s in bank:
                    case=output/f'evaluation/{split}/step_{target}/{version}/seed_{seed:03d}/cases'/main.digest(s['scenario_id'])[:24]
                    done=case/'completed.json';bar.set_description(f"{version}/{seed} {s['v11_domain']} N={s['actual_agent_count']}")
                    if done.exists():
                        if READ(done)['identity']!=identity:raise ValueError('Cached v1.1 evaluation identity changed')
                        _validate_rollout(case,s)
                    else:
                        if t is None:t=load_trainer_checkpoint(paths[version,seed],output/f'runtime/evaluation/{version}/{seed}',device=device)
                        main.seed_rollout(seed,s['scenario_id'],0)
                        steps=progress_factory(total=None,desc='flight',unit='step',leave=False);original=t.env.step
                        def step(action):
                            v=original(action);steps.update(1);return v
                        t.env.step=step
                        try:r=evaluate(t,reset_options={'scenario':s},policy_execution_mode='mean')
                        finally:t.env.step=original;steps.close()
                        row=r['episode_metrics'][0];row.update(version=version,training_seed=seed,domain=s['v11_domain'],split=split,
                            **main.rollout_diagnostics(r['rollouts'][0],t.config),**_motion_metrics(r['rollouts'][0],t.config,s))
                        save_evaluation(combine_evaluation_results([r],t.config.environment.dt_base),case);_validate_rollout(case,s)
                        main.write(done,dict(identity=identity,metrics=row,runtime=runtime_manifest()))
                    bar.update(1)
            finally:
                if t is not None:t.env.close()
    finally:bar.close();main.restore_rng(rng)
    return status(output)


def status(output):
    output=Path(output);d=READ(output/'experiment.json');rows=[]
    for path in sorted((output/'evaluation').glob('*/*/*/seed_*/cases/*/completed.json')):
        q=READ(path)
        if q['identity']['runner_sha256']!=d['runner_sha256']:raise ValueError('Foreign v1.1 result')
        row=dict(q['metrics']);row['target_label']=int(path.parents[4].name.split('_')[1]);rows.append(row)
    groups=[]
    for key in sorted({(r['target_label'],r['split'],r['domain'],r['actual_agent_count'],r['version']) for r in rows}):
        target,split,domain,n,version=key;rr=[r for r in rows if (r['target_label'],r['split'],r['domain'],r['actual_agent_count'],r['version'])==key]
        metrics=['mean_agent_arrival','episode_duration','path_stretch','actual_acceleration_norm','acceleration_squared_integral_per_uav','maximum_line_deviation','pure_seconds_per_uav','discounted_environment_reward','discounted_distance_cost','discounted_pure_cost_common']
        groups.append(dict(target=target,split=split,domain=domain,n=n,version=version,episodes=len(rr),
            success_rate=float(np.mean([r['episode_success'] for r in rr])),sample_violation_episodes=sum(r['separation_violation_pair_ticks']>0 for r in rr),
            continuous_violation_episodes=sum(r['continuous_violation_pairs']>0 for r in rr),
            initial_risk_pairs=sum(r['initial_risk_pairs'] for r in rr),
            initial_risk_pairs_below20=sum(r['initial_risk_pairs_below20'] for r in rr),
            infeasible_agent_ticks=sum(r['infeasible_agent_ticks'] for r in rr),
            **{k:float(np.mean([r[k] for r in rr if r.get(k) is not None])) if any(r.get(k) is not None for r in rr) else None for k in metrics}))
    result=dict(version='main_v1.1',groups=groups,completed_evaluation_cases=len(rows),
        training={str(s):READ(output/f'training/A/seed_{s:03d}/progress.json') for s in d['training_seeds'] if (output/f'training/A/seed_{s:03d}/progress.json').exists()},
        interpretation='Pilot descriptive comparison, not a noninferiority proof. Sampled violations only; use raw NPZ for continuous minimum distances. Smoothing remains future work.')
    diagnostics=[]
    for seed in d['training_seeds']:
        aa={(r['split'],r['target_label'],r['scenario_id']):r for r in rows if r['version']=='A_v1.0' and r['training_seed']==seed and r['domain']=='single'}
        ee={(r['split'],r['target_label'],r['scenario_id']):r for r in rows if r['version']=='E' and r['domain']=='single'}
        for key in sorted(aa.keys() & ee.keys()):
            delta=ee[key]['discounted_environment_reward']-aa[key]['discounted_environment_reward']
            diagnostics.append(dict(seed=seed,split=key[0],target=key[1],scenario_id=key[2],reference_minus_actor_environment_return=delta))
    comparisons=[]
    keys=sorted({(r['target_label'],r['split'],r['domain'],r['actual_agent_count']) for r in rows})
    for target,split,domain,n in keys:
        rr=[r for r in rows if (r['target_label'],r['split'],r['domain'],r['actual_agent_count'])==(target,split,domain,n)]
        for left,right in [('A_v1.1','A_v1.0'),('A_v1.1','E'),('A_v1.0','E')]:
            rhs={(r['training_seed'],r['scenario_id']):r for r in rr if r['version']==right}
            pairs=[(a,rhs[(0 if right=='E' else a['training_seed'],a['scenario_id'])]) for a in rr if a['version']==left
                   and (0 if right=='E' else a['training_seed'],a['scenario_id']) in rhs]
            if not pairs:continue
            item=dict(target=target,split=split,domain=domain,n=n,left=left,right=right,
                paired_rollouts=len(pairs),unique_geometries=len({a['scenario_id'] for a,b in pairs}),
                learned_seeds=sorted({a['training_seed'] for a,b in pairs}),metrics={})
            for metric in ['mean_agent_arrival','episode_duration','path_stretch','actual_acceleration_norm',
                           'acceleration_squared_integral_per_uav','maximum_line_deviation','pure_seconds_per_uav',
                           'discounted_distance_cost','discounted_pure_cost_common','infeasible_agent_ticks','continuous_violation_pairs']:
                valid=[(a[metric],b[metric]) for a,b in pairs if a.get(metric) is not None and b.get(metric) is not None]
                if valid:
                    a,b=np.asarray(valid).T
                    item['metrics'][metric]=dict(left=float(a.mean()),right=float(b.mean()),difference=float((a-b).mean()),
                        change_percent=float((a.mean()/b.mean()-1)*100) if b.mean()!=0 else None)
            comparisons.append(item)
    main.write(output/'paired_comparisons.json',dict(comparisons=comparisons,
        note='Descriptive paired differences. E is shared, not independent per learned seed. No noninferiority claim from this pilot.'))
    main.write(output/'navigation_reward_diagnostic.json',dict(pairs=diagnostics,note='Environment reward only, not SAC soft-Q calibration. No automatic guarantee of reference optimality.'))
    main.write(output/'result_status.json',result)
    if rows:
        keys=list(dict.fromkeys(k for r in rows for k in r))
        with (output/'episodes.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)
    return result


def export_results(output,*,target=5000,destination=None):
    output=Path(output).resolve();d=context(output);status(output)
    destination=Path(destination).resolve() if destination else output.with_name(output.name+'_results.zip')
    if destination.is_relative_to(output):raise ValueError('Export outside live output')
    files={str(p.relative_to(output)).replace('\\','/'):p for p in output.rglob('*') if p.is_file() and 'runtime' not in p.relative_to(output).parts and p.suffix in ['.json','.jsonl','.npz','.csv']}
    for seed in d['training_seeds']:
        p=output/f'training/A/seed_{seed:03d}/model_{target}.pt'
        if not p.exists():p=p.with_name('model_latest.pt')
        if p.exists():files[f'models/A_v1.1/seed_{seed:03d}/{p.name}']=p
        p=Path(d['parent_models'][str(seed)]['path'])
        if main.file_hash(p)!=d['parent_models'][str(seed)]['sha256']:raise ValueError('Export parent changed')
        files[f'models/A_v1.0/seed_{seed:03d}/model_{d["training_target"]}.pt']=p
    p=output/'baselines/E.pt'
    if p.exists():files['models/E.pt']=p
    tmp=destination.with_name(destination.name+'.tmp');destination.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
        manifest={}
        for name,p in sorted(files.items()):z.write(p,name);manifest[name]=dict(sha256=main.file_hash(p),bytes=p.stat().st_size)
        z.writestr('export_manifest.json',json.dumps(dict(version='main_v1.1',files=manifest,includes_replay=False),indent=2))
    tmp.replace(destination);return destination
