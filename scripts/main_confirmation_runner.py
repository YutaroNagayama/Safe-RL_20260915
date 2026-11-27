"""Frozen A/B75/B50/E evaluation on the original confirmation bank; no training."""
from pathlib import Path
import csv
import hashlib
import json
from time import perf_counter
import zipfile
import numpy as np
from uav_safe_marl import build_env
from uav_safe_marl.evaluation import load_scenario_bank
from uav_safe_marl.evaluation.artifacts import combine_evaluation_results, save_evaluation
from uav_safe_marl.experiments import main
from uav_safe_marl.experiments.loading import load_trainer_checkpoint
from uav_safe_marl.runners.evaluator import evaluate
from uav_safe_marl.runners.trainer import SafeRLTrainer
from uav_safe_marl.runtime import runtime_manifest
from scripts import main_route_extension_runner as route
from scripts.main_additional_runner import _validate_rollout

METHODS = ('A', 'B75', 'B50', 'E')
READ = route.READ


def runner_hash():
    # Include the reused model-validation implementation as well.
    return main.digest([hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), route.runner_hash()])


def prepare(source, output, *, per_n=100, counts=(2, 4, 8, 16)):
    source, suite, parent = route._source(source)
    output = Path(output).resolve()
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Use a separate output outside the source experiment')
    counts = tuple(counts)
    if (not isinstance(per_n, int) or per_n < 1 or not counts or len(set(counts)) != len(counts)
            or any(n not in parent['settings']['agent_counts'] for n in counts)):
        raise ValueError('Invalid per-N count or N selection')
    bank_path = source / 'banks/confirmation_test.json'
    expected_hash = suite['bank_hashes']['confirmation_test']
    if parent['bank_hashes']['confirmation_test'] != expected_hash:
        raise ValueError('Source bank manifests disagree')
    raw = bank_path.read_bytes()
    lf = raw.replace(b'\r\n', b'\n')
    # Git may convert LF/CRLF across Windows and Mac. Restore only bytes whose
    # exact SHA256 matches the original manifest; never bypass a content change.
    original_bytes = next((b for b in (raw, lf, lf.replace(b'\n', b'\r\n'))
                           if hashlib.sha256(b).hexdigest() == expected_hash), None)
    if original_bytes is None:
        raise ValueError('Original confirmation bank changed')
    bank = load_scenario_bank(bank_path)
    if any(sum(s['actual_agent_count'] == n for s in bank) < per_n for n in counts):
        raise ValueError('Not enough confirmation geometries; do not silently generate a replacement')
    selected = main.subset(bank, per_n)
    selected = [s for s in selected if s['actual_agent_count'] in counts]
    plan = {
        'comparison_methods': list(METHODS), 'policy_execution_mode': 'mean',
        'aggregation': 'Report each N and equal-N means; pair geometry and cluster training seeds; E once per geometry',
        'metrics': ['initial-CPA-stratified closest distance', 'below 10/20/30m pair rates',
                    'pure HOCBF correction time/intensity/discounted and undiscounted cost',
                    'QP infeasibility/fallback', 'arrival time/path length/timeouts', 'communication'],
        'cpa_strata_m': [10, 30, 75], 'close_distance_thresholds_m': [10, 20, 30],
        'note': 'Frozen confirmation of existing checkpoints. Keep low-CPA route-extension stress results separate.'}
    expected = {'schema_version': 1, 'kind': 'main_original_confirmation', 'methods': list(METHODS),
        'source_root': str(source), 'source_suite_digest': main.digest(suite),
        'source_parent_digest': main.digest(parent), 'source_sha256': suite['source_sha256'],
        'runner_sha256': runner_hash(), 'training_target': suite['target'],
        'training_seeds': suite['training_seeds'], 'bank_spec': {'per_n': per_n, 'counts': list(counts)},
        'original_bank_sha256': expected_hash, 'bank_digest': main.digest(selected),
        'max_steps': parent['max_steps'], 'scenarios': len(selected), 'analysis_plan': plan}
    if (output / 'experiment.json').exists():
        old = context(output)
        if old != expected:
            raise ValueError('Immutable confirmation experiment changed; choose another OUTPUT')
        return old
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Incomplete or nonempty output')
    (output / 'banks').mkdir(parents=True)
    (output / 'banks/confirmation_test.json').write_bytes(original_bytes)
    main.write(output / 'source_suite.json', suite)
    main.write(output / 'source_parent_experiment.json', parent)
    main.write(output / 'experiment.json', expected)
    return expected


def context(output):
    output = Path(output); d = READ(output / 'experiment.json')
    if d['runner_sha256'] != runner_hash():
        raise ValueError('Confirmation runner changed')
    if d['source_sha256'] not in route.compatible_source_hashes():
        raise ValueError('Engine changed')
    if main.file_hash(output / 'banks/confirmation_test.json') != d['original_bank_sha256']:
        raise ValueError('Confirmation bank changed')
    if main.digest(_bank(output, d)) != d['bank_digest']:
        raise ValueError('Selected bank changed')
    for filename, key in [('source_suite.json', 'source_suite_digest'), ('source_parent_experiment.json', 'source_parent_digest')]:
        if main.digest(READ(output / filename)) != d[key]:
            raise ValueError('Source snapshot changed')
    return d


def _bank(output, d):
    return [s for s in main.subset(load_scenario_bank(Path(output) / 'banks/confirmation_test.json'),
                                  d['bank_spec']['per_n']) if s['actual_agent_count'] in d['bank_spec']['counts']]


def select_scenarios(output, per_n):
    d = context(output)
    if not isinstance(per_n, int) or not 1 <= per_n <= d['bank_spec']['per_n']:
        raise ValueError('Evaluation count outside frozen confirmation bank')
    return [s['scenario_id'] for s in main.subset(_bank(output, d), per_n)]


def _baseline(output, d):
    path = Path(output) / 'baselines/E.pt'
    if not path.exists():
        rng = main.capture_rng(); trainer = None
        try:
            cfg = main.config_for(READ(Path(output) / 'source_parent_experiment.json'), 'E', 0, None)
            cfg.device = 'cpu'; cfg.monitoring.enabled = False
            trainer = SafeRLTrainer(cfg, build_env(cfg), Path(output) / 'runtime/baseline_E')
            trainer.save_checkpoint(path, include_replay=False)
        finally:
            if trainer is not None: trainer.env.close()
            main.restore_rng(rng)
    return path

def checkpoints(output,seeds):
    """Find only exact model-only checkpoints, using canonical export or Windows source paths."""
    d=context(output);source=Path(d['source_root']);suite=READ(output/'source_suite.json');paths={};missing=[]
    for method in route.METHODS:
        for seed in seeds:
            if seed not in d['training_seeds']:raise ValueError('Unknown training seed')
            portable=source/'models'/method/f'seed_{seed:03d}'/f"model_{d['training_target']}.pt"
            if method=='B50':original=source/'experiments/B50/training/B'/f'seed_{seed:03d}'/f"model_{d['training_target']}.pt"
            else:original=Path(suite['parent_root'])/'training'/('A' if method=='A' else 'B')/f'seed_{seed:03d}'/f"model_{d['training_target']}.pt"
            path=next((p for p in [portable,original] if p.is_file()),None)
            if path is None:missing.append(str(portable)+' OR '+str(original))
            else:paths[method,seed]=path
    if missing:raise FileNotFoundError('Evaluation models missing; no substitute Actor is used:\n'+'\n'.join(missing))
    paths['E', 0] = _baseline(output, d)
    return paths


def _validate_model(path, method, seed, d, output):
    if method != 'E':
        return route._validate_model(path, method, seed, d, output)
    import torch
    payload = torch.load(path, map_location='cpu', weights_only=False)
    expected = main.config_for(READ(Path(output) / 'source_parent_experiment.json'), 'E', 0, None).to_dict()
    if payload['experiment_identity']['source_sha256'] not in route.compatible_source_hashes():
        raise ValueError('E baseline source differs')
    cfg = payload['config']
    if cfg['seed'] != 0: raise ValueError('E must be evaluated once per geometry')
    for name in ['environment', 'scenario', 'safety', 'graph', 'control', 'communication', 'objective', 'policy', 'learning', 'training']:
        if cfg[name] != expected[name]: raise ValueError(f'E baseline config differs: {name}')
    return main.file_hash(path)

def evaluate_grid(output,*,seeds=None,device='cpu',progress_factory=None,scenario_limit=None,scenario_ids=None):
    output=Path(output).resolve();d=context(output);seeds=list(d['training_seeds'] if seeds is None else seeds)
    if not seeds or len(set(seeds))!=len(seeds):raise ValueError('Choose unique seeds')
    bank=_bank(output, d)
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
    bar=progress_factory(total=len(bank)*len(paths),desc='A/B75/B50/E confirmation',unit='case');rng=main.capture_rng()
    try:
        # One model in RAM, no multiprocessing or nested safety workers.
        for method in METHODS:
            for seed in ([0] if method == 'E' else seeds):
                trainer=None
                try:
                    identity={'method':method,'training_seed':seed,'step_label':d['training_target'],'runtime_hocbf':True,'stochastic':False,
                              'checkpoint_sha256':hashes[method,seed],'source_sha256':d['source_sha256'],'runner_sha256':d['runner_sha256'],
                              'bank_digest':d['bank_digest'],'max_steps':d['max_steps'],'device':device,'execution_system':runtime_manifest()['system']}
                    for scenario in bank:
                        case=output/'evaluation/mean'/method/f'seed_{seed:03d}'/'cases'/main.digest(scenario['scenario_id'])[:24]
                        completed=case/'completed.json';started=perf_counter()
                        bar.set_description(f"{method}/{seed} N={scenario['actual_agent_count']} confirmation")
                        if completed.exists():
                            cached=READ(completed)
                            if cached['identity']!=identity:raise ValueError('Cached evaluation identity changed')
                            _validate_rollout(case,scenario)
                        else:
                            if trainer is None:
                                trainer=load_trainer_checkpoint(paths[method,seed],output/f'runtime/{method}/seed_{seed:03d}',device=device,
                                    config_overrides={'environment':{'max_steps':d['max_steps']}})
                                frozen=main.frozen_hashes(trainer)
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
                            row.update(method=method,training_seed=seed,source_geometry_id=scenario['scenario_id'],
                                evaluation_split='confirmation_test',prediction_horizon=scenario['prediction_horizon'],
                                elapsed_seconds=perf_counter()-started,
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
    """Model-independent partial extraction; E is never replicated across training seeds."""
    output = Path(output); d = READ(output / 'experiment.json'); rows = []
    for p in sorted((output / 'evaluation').glob('mean/*/seed_*/cases/*/completed.json')):
        q = READ(p)
        if q['identity']['bank_digest'] != d['bank_digest'] or q['identity']['runner_sha256'] != d['runner_sha256']:
            raise ValueError('Foreign result in output')
        rows.append(q['metrics'])
    groups = []
    for method in METHODS:
        for n in d['bank_spec']['counts']:
            rr = [r for r in rows if r['method'] == method and r['actual_agent_count'] == n]
            if not rr: continue
            groups.append({'method': method, 'n': n, 'episodes': len(rr),
                'source_geometries': len({r['scenario_id'] for r in rr}),
                'seeds': sorted({r['training_seed'] for r in rr}),
                'success_rate': float(np.mean([r['episode_success'] for r in rr])),
                'sample_violation_episodes': sum(r['separation_violation_pair_ticks'] > 0 for r in rr),
                'infeasible_agent_ticks': sum(r['infeasible_agent_ticks'] for r in rr),
                'emergency_agent_ticks': sum(r['emergency_agent_ticks'] for r in rr)})
    result = {'completed_cases': len(rows), 'expected_all_seeds_cases': d['scenarios'] * (3 * len(d['training_seeds']) + 1),
              'groups': groups, 'note': 'Original confirmation bank; pair by geometry and cluster training seeds. E once per geometry. Raw NPZ supports continuous-distance and full cost/efficiency analyses.'}
    if (output / 'evaluation_request.json').exists():
        request = READ(output / 'evaluation_request.json'); selected = set(request['scenario_ids'])
        result.update(requested_cases=request['expected_cases'], requested_completed_cases=sum(
            r['scenario_id'] in selected and (r['method'] == 'E' or r['training_seed'] in request['seeds']) for r in rows))
    main.write(output / 'result_status.json', result)
    if rows:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with (output / 'episodes.csv').open('w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
            w.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in r.items()} for r in rows)
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
