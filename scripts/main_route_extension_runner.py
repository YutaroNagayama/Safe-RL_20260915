"""A/B75/B50 frozen-policy route-extension evaluation. Never trains or edits engine code."""
from __future__ import annotations
from copy import deepcopy
from functools import lru_cache
from pathlib import Path, PureWindowsPath
import csv
import hashlib
import json
import math
from time import perf_counter
import zipfile
import numpy as np
from uav_safe_marl.evaluation import generate_random_traffic_bank, load_scenario_bank, save_scenario_bank
from uav_safe_marl.evaluation.scenarios import predicted_interaction_data, interaction_descriptor
from uav_safe_marl.evaluation.artifacts import combine_evaluation_results, save_evaluation
from uav_safe_marl.experiments import main
from uav_safe_marl.experiments.loading import load_trainer_checkpoint
from uav_safe_marl.runners.evaluator import evaluate
from uav_safe_marl.runners.trainer import SafeRLTrainer
from uav_safe_marl.runtime import runtime_manifest
from scripts.main_additional_runner import _validate_rollout

METHODS = ('A', 'B75', 'B50')
READ = lambda p: json.loads(Path(p).read_text(encoding='utf-8'))

def runner_hash():
    return hashlib.sha256(Path(__file__).read_text(encoding='utf-8').encode()).hexdigest()

@lru_cache(maxsize=1)
def compatible_source_hashes():
    """Account for exact POSIX/Windows path separators and LF/CRLF, not code changes."""
    root = Path(__file__).resolve().parents[1] / 'uav_safe_marl'
    files = list(root.rglob('*.py')); hashes = set()
    for windows in (False, True):
        order = sorted(files, key=lambda p: PureWindowsPath(str(p.relative_to(root))) if windows else p.relative_to(root))
        for crlf in (False, True):
            h = hashlib.sha256()
            for p in order:
                rel = p.relative_to(root)
                h.update((str(PureWindowsPath(str(rel))) if windows else rel.as_posix()).encode())
                data = p.read_bytes().replace(b'\r\n', b'\n')
                h.update(data.replace(b'\n', b'\r\n') if crlf else data)
            hashes.add(h.hexdigest())
    return hashes

def extend_route(scenario, lead_seconds):
    """Back-extend collinear routes by v_initial * lead. Preserve future CPA geometry."""
    if not np.isfinite(lead_seconds) or lead_seconds < 0:
        raise ValueError('lead_seconds must be finite and nonnegative')
    s = deepcopy(scenario);p = np.asarray(s['positions'], float);v = np.asarray(s['velocities'], float);g = np.asarray(s['goals'], float)
    p = p - v * lead_seconds
    n = len(p);i,j = np.triu_indices(n, 1)
    if np.linalg.norm(p[i]-p[j],axis=-1).min() < s['minimum_initial_separation']-1e-8:
        raise ValueError('Route extension violates minimum initial separation')
    horizon = float(s['prediction_horizon']) + lead_seconds
    edges, pairs = predicted_interaction_data(p, v, g, s['los_prediction_threshold'], horizon)
    source_id = s['scenario_id'];lengths = np.linalg.norm(g-p,axis=-1)
    # Full three-dimensional collinear extension; outer starts need not lie on one rectangle.
    extent = np.maximum(np.abs(p).max(0), np.abs(g).max(0))*2
    extent = np.maximum(extent, np.asarray(s['spatial_dimensions'],float))
    descriptor = interaction_descriptor(n, pairs, float(np.prod(extent)),s['maximum_velocities'],s['maximum_accelerations'],10.,30.,50.)
    descriptor.update(spatial_scaling_mode='route_extension',spatial_dimensions=extent.tolist(),spatial_volume=float(np.prod(extent)),initial_cbf_edge_count=sum(x['initial_distance']<=50 for x in pairs))
    s.update(scenario_id=f"{source_id}__lead_{lead_seconds:g}s",source_geometry_id=source_id,
             route_extension_seconds=float(lead_seconds),geometry='four_face_collinear_route_extension',
             generator='main_route_extension_v1',generator_version='main_route_extension_v1',
             positions=p.tolist(),reference_route_lengths=lengths.tolist(),reference_routes=np.stack([p,g],1).tolist(),
             reference_arrival_times=(lengths/np.linalg.norm(v,axis=1)).tolist(),prediction_horizon=horizon,
             predicted_los_edges=edges,predicted_interaction_edges=edges,predicted_pair_data=pairs,
             spatial_dimensions=extent.tolist(),spatial_volume=float(np.prod(extent)),scenario_descriptor=descriptor,
             spatial_scaling_mode='route_extension')
    return s

def _source(source):
    source = Path(source).resolve();suite=READ(source/'suite.json');parent=READ(source/'parent_experiment.json')
    if suite['source_sha256']!=parent['source_sha256'] or suite['source_sha256'] not in compatible_source_hashes():
        raise ValueError('Engine source differs from original main; no blanket bypass is allowed')
    cfg=parent['settings']['base_config']
    if (cfg['learning']['d_cost']!=6 or not cfg['objective']['intervention_cost_enabled']
            or [cfg['safety'][k] for k in ['d_safe','d_warn','d_eng']]!=[10,30,50]):
        raise ValueError('Source is not the original main A/B configuration')
    return source,suite,parent

def prepare(source,output,*,per_n=10,leads=(0.,20.,40.),bank_seed=2026112701,counts=(2,4,8,16),cpa_max=10.,far_distance=150.,max_attempts=20000):
    source,suite,parent = _source(source);output=Path(output).resolve()
    if output==source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Use a separate output directory outside the original experiment')
    leads=tuple(float(x) for x in leads);counts=tuple(int(n) for n in counts)
    if (not isinstance(per_n,int) or per_n<1 or not leads or len(set(leads))!=len(leads)
            or any(not np.isfinite(x) or x<0 for x in leads) or not counts or len(set(counts))!=len(counts)
            or any(n not in parent['settings']['agent_counts'] for n in counts)
            or not 0<cpa_max<=10 or not np.isfinite(far_distance) or far_distance<=0 or max_attempts<1):
        raise ValueError('Invalid bank size, lead, N, CPA threshold or attempt limit')
    spec={'per_n':per_n,'leads':list(leads),'bank_seed':bank_seed,'counts':list(counts),'cpa_max':cpa_max,
          'max_attempts':max_attempts,'minimum_designated_distance_at_max_lead':far_distance,'minimum_risk_pairs':{str(n):max(1,n//8) for n in counts}}
    expected={'schema_version':1,'kind':'A_Bs_route_extension','methods':list(METHODS),'source_root':str(source),
              'source_suite_digest':main.digest(suite),'source_parent_digest':main.digest(parent),
              'source_sha256':suite['source_sha256'],'runner_sha256':runner_hash(),'training_target':suite['target'],
              'training_seeds':suite['training_seeds'],'bank_spec':spec}
    manifest=output/'experiment.json'
    if manifest.exists():
        old=context(output)
        if any(old[k]!=v for k,v in expected.items()):raise ValueError('Immutable experiment changed; choose another OUTPUT')
        return old
    if output.exists() and any(output.iterdir()):raise FileExistsError('Incomplete or nonempty output; choose another directory')
    settings=parent['settings'];sc=settings['scenario'];base=settings['base_config'];sources=[];variants=[];attempts={}
    for n in counts:
        accepted=0;drawn=0;batch=0
        while accepted<per_n and drawn<max_attempts:
            seed=int(np.random.SeedSequence([bank_seed,n,batch]).generate_state(1)[0]);batch+=1
            pool=generate_random_traffic_bank(n,min(64,max_attempts-drawn),seed,
                base_size=sc['longitudinal_length'],transverse_size=sc['transverse_size'],altitude_range=(sc['altitude_min'],sc['altitude_max']),
                minimum_initial_distance=sc['minimum_initial_separation'],d_safe=10,d_warn=30,d_eng=50,
                los_prediction_threshold=sc['interaction_threshold'],prediction_horizon=sc['prediction_horizon'],
                initial_speed_fraction_range=(sc['initial_speed_fraction_min'],sc['initial_speed_fraction_max']),
                acceleration_range=(sc['acceleration_min'],sc['acceleration_max']),velocity_limit_range=(sc['velocity_limit_min'],sc['velocity_limit_max']),
                require_predicted_conflict=False,entry_layout='four_faces')
            for s in pool:
                drawn+=1
                far_positions=np.asarray(s['positions'])-max(leads)*np.asarray(s['velocities'])
                risks=[x for x in s['predicted_pair_data'] if x['predicted_cpa_distance']<cpa_max and x['predicted_cpa_time']>1.
                       and np.linalg.norm(far_positions[x['first']]-far_positions[x['second']])>=far_distance]
                if len(risks)<max(1,n//8):continue
                s['designated_risk_pairs']=[[x['first'],x['second']] for x in risks]
                try:vv=[extend_route(s,lead) for lead in leads]
                except ValueError:continue
                for v in vv:
                    pp={(x['first'],x['second']):x for x in v['predicted_pair_data']}
                    for x in risks:
                        y=pp[x['first'],x['second']]
                        if not np.isclose(y['predicted_cpa_distance'],x['predicted_cpa_distance'],atol=1e-7):raise RuntimeError('CPA geometry changed')
                        if not np.isclose(y['predicted_cpa_time'],x['predicted_cpa_time']+v['route_extension_seconds'],atol=1e-7):raise RuntimeError('CPA time shift changed')
                sources.append(s);variants.extend(vv);accepted+=1
                if accepted==per_n:break
        if accepted<per_n:raise RuntimeError(f'Could only generate {accepted}/{per_n} low-CPA geometries for N={n}; no fallback distribution used')
        attempts[str(n)]=drawn
    # Lead conditions share the same base geometry; count inference clusters accordingly.
    variants.sort(key=lambda s:(s['route_extension_seconds'],s['actual_agent_count'],s['source_geometry_id']))
    output.mkdir(parents=True)
    save_scenario_bank(output/'banks/base.json',sources);save_scenario_bank(output/'banks/route_extension.json',variants)
    maximum=max(max(s['reference_arrival_times']) for s in variants)
    expected.update(bank_digest=main.digest(load_scenario_bank(output/'banks/route_extension.json')),
                    base_bank_digest=main.digest(load_scenario_bank(output/'banks/base.json')),
                    max_steps=max(parent['max_steps'],math.ceil(2*maximum/base['environment']['dt_base'])),
                    base_geometries=len(sources),scenarios=len(variants),generation_draws=attempts)
    main.write(output/'source_suite.json',suite);main.write(output/'source_parent_experiment.json',parent);main.write(manifest,expected)
    return expected

def context(output):
    output=Path(output);d=READ(output/'experiment.json')
    if d['runner_sha256']!=runner_hash():raise ValueError('Evaluation runner changed; choose a new output or audit migration')
    if d['source_sha256'] not in compatible_source_hashes():raise ValueError('Engine changed')
    for filename,key in [('route_extension','bank_digest'),('base','base_bank_digest')]:
        if main.digest(load_scenario_bank(output/f'banks/{filename}.json'))!=d[key]:raise ValueError('Bank changed')
    for filename,key in [('source_suite.json','source_suite_digest'),('source_parent_experiment.json','source_parent_digest')]:
        if main.digest(READ(output/filename))!=d[key]:raise ValueError('Source snapshot changed')
    return d

def select_scenarios(output,per_n):
    """Choose a phase from an already fixed large bank; expand later without changing identity."""
    output=Path(output);d=context(output)
    if not isinstance(per_n,int) or not 1<=per_n<=d['bank_spec']['per_n']:raise ValueError('Evaluation per-N count outside frozen bank')
    base=load_scenario_bank(output/'banks/base.json');chosen=set()
    for n in d['bank_spec']['counts']:
        ids=sorted(s['scenario_id'] for s in base if s['actual_agent_count']==n)
        chosen.update(ids[:per_n])
    return [s['scenario_id'] for s in load_scenario_bank(output/'banks/route_extension.json') if s['source_geometry_id'] in chosen]

def checkpoints(output,seeds):
    """Find only exact model-only checkpoints, using canonical export or Windows source paths."""
    d=context(output);source=Path(d['source_root']);suite=READ(output/'source_suite.json');paths={};missing=[]
    for method in METHODS:
        for seed in seeds:
            if seed not in d['training_seeds']:raise ValueError('Unknown training seed')
            portable=source/'models'/method/f'seed_{seed:03d}'/f"model_{d['training_target']}.pt"
            if method=='B50':original=source/'experiments/B50/training/B'/f'seed_{seed:03d}'/f"model_{d['training_target']}.pt"
            else:original=Path(suite['parent_root'])/'training'/('A' if method=='A' else 'B')/f'seed_{seed:03d}'/f"model_{d['training_target']}.pt"
            path=next((p for p in [portable,original] if p.is_file()),None)
            if path is None:missing.append(str(portable)+' OR '+str(original))
            else:paths[method,seed]=path
    if missing:raise FileNotFoundError('Evaluation models missing; no substitute Actor is used:\n'+'\n'.join(missing))
    return paths

def _validate_model(path,method,seed,d,output):
    import torch
    payload=torch.load(path,map_location='cpu',weights_only=False);cfg=payload['config'];identity=payload['experiment_identity']
    if identity['source_sha256']!=d['source_sha256'] or cfg['seed']!=seed:raise ValueError('Model source/seed differs')
    if (payload['trainer_progress']['total_environment_steps']<d['training_target'] or not cfg['safety']['enabled']
        or cfg['control']['actor_mode']!='residual' or not cfg['objective']['intervention_cost_enabled'] or cfg['learning']['d_cost']!=6):
        raise ValueError('Not the intended trained A/B model')
    mode='prior_share' if method=='A' else 'reactive_share';radius=50 if method=='B50' else 75
    if cfg['communication']['information_mode']!=mode or cfg['communication']['reactive_radius']!=radius:raise ValueError('Model communication mode/radius differs')
    parent=READ(output/'source_parent_experiment.json');expected=main.config_for(parent,'A' if method=='A' else 'B',seed,d['training_target']).to_dict()
    if method=='B50':expected['communication']['reactive_radius']=50.
    for name in ['environment','scenario','safety','graph','control','communication','objective','policy','learning','training']:
        if cfg[name]!=expected[name]:raise ValueError(f'Model config differs: {method}/{name}')
    sha=main.file_hash(path)
    # If original rollout records are present, verify the exact evaluated checkpoint too.
    cases=list((Path(d['source_root'])/f'evaluation/development/step_{d["training_target"]}/mean/{method}/seed_{seed:03d}/cases').glob('*/completed.json'))
    hashes={READ(p)['identity']['checkpoint_sha256'] for p in cases}
    if not hashes and (Path(d['source_root'])/'models_manifest.json').is_file():
        hashes={r['sha256'] for r in READ(Path(d['source_root'])/'models_manifest.json')['models'] if r['method']==method and r['seed']==seed}
    if hashes and hashes!={sha}:raise ValueError('Model differs from the completed main rollout checkpoint')
    return sha

def evaluate_grid(output,*,seeds=None,device='cpu',progress_factory=None,scenario_limit=None,scenario_ids=None):
    output=Path(output).resolve();d=context(output);seeds=list(d['training_seeds'] if seeds is None else seeds)
    if not seeds or len(set(seeds))!=len(seeds):raise ValueError('Choose unique seeds')
    bank=load_scenario_bank(output/'banks/route_extension.json')
    if scenario_ids is not None:
        selected=set(scenario_ids)
        if not selected or not selected.issubset({s['scenario_id'] for s in bank}):raise ValueError('Unknown/empty scenario selection')
        bank=[s for s in bank if s['scenario_id'] in selected]
    if scenario_limit is not None:
        if not isinstance(scenario_limit,int) or scenario_limit<1:raise ValueError('scenario_limit must be positive')
        bank=bank[:scenario_limit]
    paths=checkpoints(output,seeds);hashes={(m,s):_validate_model(p,m,s,d,output) for (m,s),p in paths.items()}
    model_records={}
    if (output/'models_manifest.json').exists():
        model_records={(r['method'],r['seed']):r for r in READ(output/'models_manifest.json')['models']}
        for key,sha in hashes.items():
            if key in model_records and model_records[key]['sha256']!=sha:raise ValueError('Previously evaluated model changed')
    model_records.update({(m,s):{'method':m,'seed':s,'path':str(p),'sha256':hashes[m,s]} for (m,s),p in paths.items()})
    main.write(output/'models_manifest.json',{'models':list(model_records.values())})
    main.write(output/'evaluation_request.json',{'seeds':seeds,'scenario_ids':[s['scenario_id'] for s in bank],'expected_cases':len(bank)*len(paths),'device':device})
    if progress_factory is None:
        from tqdm.auto import tqdm
        progress_factory=tqdm
    bar=progress_factory(total=len(bank)*len(paths),desc='A/B75/B50 rollout',unit='case');rng=main.capture_rng()
    try:
        # One model in RAM, no multiprocessing or nested safety workers.
        for method in METHODS:
            for seed in seeds:
                trainer=None
                try:
                    identity={'method':method,'training_seed':seed,'step_label':d['training_target'],'runtime_hocbf':True,'stochastic':False,
                              'checkpoint_sha256':hashes[method,seed],'source_sha256':d['source_sha256'],'runner_sha256':d['runner_sha256'],
                              'bank_digest':d['bank_digest'],'max_steps':d['max_steps'],'device':device,'execution_system':runtime_manifest()['system']}
                    for scenario in bank:
                        case=output/'evaluation/mean'/method/f'seed_{seed:03d}'/'cases'/main.digest(scenario['scenario_id'])[:24]
                        completed=case/'completed.json';started=perf_counter()
                        bar.set_description(f"{method}/{seed} N={scenario['actual_agent_count']} +{scenario['route_extension_seconds']:g}s")
                        if completed.exists():
                            cached=READ(completed)
                            if cached['identity']!=identity:raise ValueError('Cached evaluation identity changed')
                            _validate_rollout(case,scenario)
                        else:
                            if trainer is None:
                                trainer=load_trainer_checkpoint(paths[method,seed],output/f'runtime/{method}/seed_{seed:03d}',device=device,
                                    config_overrides={'environment':{'max_steps':d['max_steps']}})
                                frozen=main.frozen_hashes(trainer)
                            # Evaluation-only predictor horizon follows the time-origin extension.
                            trainer.config.scenario.prediction_horizon=scenario['prediction_horizon']
                            main.seed_rollout(seed,scenario['scenario_id'],0)
                            steps=progress_factory(total=None,desc=f'{method}/{seed} flight',unit='step',leave=False)
                            original_step=trainer.env.step
                            def observed_step(action):
                                value=original_step(action)
                                steps.update(1)
                                return value
                            trainer.env.step=observed_step
                            try:
                                result=evaluate(trainer,reset_options={'scenario':scenario},policy_execution_mode='mean')
                            finally:
                                trainer.env.step=original_step
                                steps.close()
                            rollout=result['rollouts'][0];row=result['episode_metrics'][0]
                            active=np.asarray(rollout['active_masks'][:-1],bool);pure=np.asarray(rollout['pure_cbf_actions'])-np.asarray(rollout['nominal_actions'])
                            pure2=(pure*pure).sum(-1);diag=rollout['safety_diagnostics']
                            row.update(method=method,training_seed=seed,source_geometry_id=scenario['source_geometry_id'],
                                route_extension_seconds=scenario['route_extension_seconds'],prediction_horizon=scenario['prediction_horizon'],
                                designated_risk_pairs=scenario['designated_risk_pairs'],elapsed_seconds=perf_counter()-started,
                                pure_intervention_squared_sum=float(pure2[active].sum()),pure_nonzero_agent_ticks=int(((pure2>1e-12)&active).sum()),
                                infeasible_agent_ticks=sum(bool(diag[t][i]['infeasible']) for t,i in zip(*np.nonzero(active))),
                                emergency_agent_ticks=sum(bool(diag[t][i]['emergency']) for t,i in zip(*np.nonzero(active))),
                                **main.rollout_diagnostics(rollout,trainer.config))
                            save_evaluation(combine_evaluation_results([result],trainer.config.environment.dt_base),case)
                            _validate_rollout(case,scenario)
                            if main.frozen_hashes(trainer)!=frozen:raise RuntimeError('Evaluation changed trained model state')
                            main.write(completed,{'identity':identity,'metrics':row,'runtime':runtime_manifest()})
                        bar.update(1)
                        bar.set_postfix(seconds=f'{perf_counter()-started:.1f}',refresh=True)
                finally:
                    if trainer is not None:trainer.env.close()
    finally:bar.close();main.restore_rng(rng)
    return status(output)

def status(output):
    """Read-only model-independent partial result extraction; works after copying to Mac."""
    output=Path(output);d=READ(output/'experiment.json');rows=[]
    for p in sorted((output/'evaluation').glob('mean/*/seed_*/cases/*/completed.json')):
        q=READ(p)
        if q['identity']['bank_digest']!=d['bank_digest'] or q['identity']['runner_sha256']!=d['runner_sha256']:raise ValueError('Foreign result in output')
        rows.append(q['metrics'])
    def average(rr,key):
        values=[r[key] for r in rr if r.get(key) is not None]
        return float(np.mean(values)) if values else None
    groups=[]
    for method in METHODS:
        for lead in d['bank_spec']['leads']:
            for n in d['bank_spec']['counts']:
                rr=[r for r in rows if r['method']==method and r['route_extension_seconds']==lead and r['actual_agent_count']==n]
                if not rr:continue
                groups.append({'method':method,'lead_seconds':lead,'n':n,'episodes':len(rr),'source_geometries':len({r['source_geometry_id'] for r in rr}),
                    'seeds':sorted({r['training_seed'] for r in rr}),'success_rate':float(np.mean([r['episode_success'] for r in rr])),
                    'sample_violation_episodes':sum(r['separation_violation_pair_ticks']>0 for r in rr),
                    **{k:average(rr,k) for k in ['episode_duration','path_stretch','distance_cost','discounted_distance_cost','discounted_pure_cost_common']},
                    'path_stretch_valid_cases':sum(r.get('path_stretch') is not None for r in rr),
                    'successful_completion_seconds':average([r for r in rr if r['episode_success']], 'episode_duration'),
                    'infeasible_agent_ticks':sum(r['infeasible_agent_ticks'] for r in rr),'emergency_agent_ticks':sum(r['emergency_agent_ticks'] for r in rr)})
    result={'completed_cases':len(rows),'expected_all_seeds_cases':d['scenarios']*len(METHODS)*len(d['training_seeds']),'groups':groups,
            'note':'Partial means are descriptive. Source geometry, N and seed must be paired; lead variants are not independent geometries. Path stretch/completion means are success-only; episode_duration includes timeouts. Violations here are sampled; raw NPZ supports continuous-distance analysis.'}
    if (output/'evaluation_request.json').exists():
        request=READ(output/'evaluation_request.json');selected=set(request['scenario_ids'])
        result.update(requested_cases=request['expected_cases'],requested_completed_cases=sum(r['scenario_id'] in selected and r['training_seed'] in request['seeds'] for r in rows))
    main.write(output/'result_status.json',result)
    if rows:
        keys=list(dict.fromkeys(k for r in rows for k in r))
        with (output/'episodes.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)
    return result

def export_results(output,destination=None,*,include_models=True):
    output=Path(output).resolve();d=context(output);status(output)
    destination=Path(destination).resolve() if destination else output.with_name(output.name+'_results.zip')
    if destination.is_relative_to(output):raise ValueError('Export must be outside the live output')
    files={}
    for p in output.rglob('*'):
        rel=p.relative_to(output)
        if p.is_file() and 'runtime' not in rel.parts and p.suffix in ('.json','.jsonl','.npz','.csv') and not p.name.endswith('.tmp'):files[rel.as_posix()]=p
    # Include the original source metadata under canonical names for later portable re-evaluation.
    files['suite.json']=output/'source_suite.json'
    files['parent_experiment.json']=output/'source_parent_experiment.json'
    if include_models:
        if not (output/'models_manifest.json').exists():raise FileNotFoundError('Run model preflight/evaluation before exporting models')
        for m in READ(output/'models_manifest.json')['models']:
            p=Path(m['path'])
            if not p.is_file() or main.file_hash(p)!=m['sha256']:raise ValueError('Export model missing/changed')
            files[f"models/{m['method']}/seed_{m['seed']:03d}/model_{d['training_target']}.pt"]=p
    destination.parent.mkdir(parents=True,exist_ok=True);temporary=destination.with_name(destination.name+'.tmp')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
        manifest={}
        for name,p in sorted(files.items()):z.write(p,name);manifest[name]={'sha256':main.file_hash(p),'bytes':p.stat().st_size}
        z.writestr('export_manifest.json',json.dumps({'files':manifest,'methods':list(METHODS),'kind':d['kind'],'includes_models':include_models},indent=2))
    temporary.replace(destination);return destination
