"""Staged main experiment; learning and control remain in the pre-main engine.

Run ``python -m uav_safe_marl.experiments.main --help``. Importing this module
never starts training, creates banks, or opens the confirmation test.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
from time import perf_counter
from typing import Any

import numpy as np
import yaml

from uav_safe_marl import build_env, load_config
from uav_safe_marl.evaluation import generate_random_traffic_bank, load_scenario_bank, save_scenario_bank
from uav_safe_marl.evaluation.artifacts import _jsonable, combine_evaluation_results, save_evaluation
from uav_safe_marl.experiments.loading import load_trainer_checkpoint
from uav_safe_marl.runners.evaluator import evaluate
from uav_safe_marl.runners.trainer import SafeRLTrainer
from uav_safe_marl.runtime import runtime_manifest

TRAINED = ('A', 'B', 'F')
CONDITIONS = {'A': ('A', True), 'B': ('B', True), 'C': ('A', False),
              'D': ('B', False), 'E': ('E', True), 'F': ('F', True),
              'R': ('R', False), 'F_OFF': ('F', False)}


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(_jsonable(value), sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(_jsonable(value), ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def subset(bank: list[dict], per_n: int) -> list[dict]:
    return [s for n in sorted({int(s['actual_agent_count']) for s in bank})
            for s in sorted((s for s in bank if int(s['actual_agent_count']) == n),
                            key=lambda s: digest(s['scenario_id']))[:per_n]]


def context(root: Path) -> dict:
    data = json.loads((root / 'experiment.json').read_text())
    if data['source_sha256'] != SafeRLTrainer._source_hash():
        raise ValueError('Implementation changed since prepare; use a new experiment or an audited migration')
    for name, expected in data['bank_hashes'].items():
        if file_hash(root / 'banks' / f'{name}.json') != expected:
            raise ValueError(f'Bank changed: {name}')
    return data


def prepare(settings_path: Path, root: Path) -> dict:
    settings = yaml.safe_load(settings_path.read_text())
    manifest = root / 'experiment.json'
    if manifest.exists():
        data = context(root)
        if data['settings'] != settings:
            raise ValueError('Immutable experiment differs; choose another output directory')
        return data
    if root.exists() and any(root.iterdir()):
        raise FileExistsError('Incomplete/nonempty experiment directory; do not overwrite')
    scenario = settings['scenario']
    base = settings['base_config']
    banks = {}
    seen = set()
    for name, spec in settings['banks'].items():
        bank = []
        for n in settings['agent_counts']:
            bank.extend(generate_random_traffic_bank(
                n, spec['per_n'], seed=spec['seed'] + n,
                base_size=scenario['longitudinal_length'], transverse_size=scenario['transverse_size'],
                spatial_scaling_mode='fixed', spatial_volume_scale_range=(1., 1.),
                altitude_range=(scenario['altitude_min'], scenario['altitude_max']),
                minimum_initial_distance=scenario['minimum_initial_separation'],
                d_safe=base['safety']['d_safe'], d_warn=base['safety']['d_warn'], d_eng=base['safety']['d_eng'],
                los_prediction_threshold=scenario['interaction_threshold'], prediction_horizon=scenario['prediction_horizon'],
                initial_speed_fraction_range=(scenario['initial_speed_fraction_min'], scenario['initial_speed_fraction_max']),
                acceleration_range=(scenario['acceleration_min'], scenario['acceleration_max']),
                velocity_limit_range=(scenario['velocity_limit_min'], scenario['velocity_limit_max']),
                require_predicted_conflict=True,
                entry_layout=scenario.get("entry_layout", "two_faces")))
        rng = np.random.default_rng(spec['seed'])
        groups = [[s for s in bank if s['actual_agent_count'] == n] for n in settings['agent_counts']]
        for group in groups:
            rng.shuffle(group)
        bank = [group[i] for i in range(spec['per_n']) for group in groups]
        for s in bank:
            key = digest({k: s[k] for k in ('positions','velocities','goals','maximum_accelerations','maximum_velocities')})
            if key in seen:
                raise ValueError('Duplicate initial condition across banks')
            seen.add(key)
        (root / 'banks').mkdir(parents=True, exist_ok=True)
        save_scenario_bank(root / 'banks' / f'{name}.json', bank)
        banks[name] = bank
    maximum = max(float(np.max(np.asarray(s['reference_route_lengths']) /
                  np.maximum(np.linalg.norm(s['velocities'], axis=1), 1e-6)))
                  for bank in banks.values() for s in bank)
    data = {'schema_version': 1, 'settings': settings, 'settings_sha256': digest(settings),
            'source_sha256': SafeRLTrainer._source_hash(), 'hardware': runtime_manifest(),
            'max_steps': int(np.ceil(2 * maximum / base['environment']['dt_base'])),
            'bank_hashes': {name: file_hash(root / 'banks' / f'{name}.json') for name in banks},
            'development_ids': [s['scenario_id'] for s in subset(banks['development'], settings['development_per_n'])],
            'calibration_ids': [s['scenario_id'] for s in subset(banks['decision'], settings['calibration_per_n'])]}
    write(manifest, data)
    return data


def config_for(data: dict, method: str, seed: int, budget: int | None):
    values = deepcopy(data['settings']['base_config'])
    values.update(seed=seed)
    counts = data['settings']['agent_counts']
    values['environment'].update(agent_count=min(counts), max_agent_count=max(counts), max_steps=data['max_steps'])
    values['scenario']['training_agent_count_distribution'] = {n: 1 / len(counts) for n in counts}
    values['training']['total_environment_steps_budget'] = budget
    if method == 'B':
        values['communication']['information_mode'] = 'reactive_share'
    if method == 'F':
        values['objective']['intervention_cost_enabled'] = False
    if method in ('E', 'R'):
        values['control']['actor_mode'] = 'reference_only'
        values['communication']['information_mode'] = 'none'
        values['safety']['enabled'] = method == 'E'
    return load_config(overrides=values)


def frozen_hashes(trainer) -> dict:
    state = trainer.backend.state_dict()
    def h(value):
        return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for _, v in sorted(value.items()))).hexdigest()
    return {**{k: h(state[k]) for k in ('actor', 'reward_critics', 'reward_targets')},
            'alpha': float(state['log_alpha'].detach().cpu()), 'lambda': state['dual']}


def train_method(root: Path, method: str, seed: int, target: int | None = None) -> Path:
    if method not in TRAINED:
        raise ValueError('Only A/B/F learn; C/D reuse A/B checkpoints')
    data = context(root)
    settings = data['settings']
    target = settings['online_steps'] if target is None else int(target)
    if target <= 0:
        raise ValueError('target must be positive')
    run = root / 'training' / method / f'seed_{seed:03d}'
    latest = run / 'latest.pt'
    warm = run / 'warmup' / 'full.pt'
    if not latest.exists():
        cfg = config_for(data, method, seed, None)
        trainer = SafeRLTrainer(cfg, build_env(cfg), run / 'warmup' / 'collection')
        try:
            if warm.exists():
                trainer.load_checkpoint(warm)
            else:
                bank = load_scenario_bank(root / 'banks' / 'warmup.json')
                collection = run / 'warmup' / 'collection.pt'
                progress = run / 'warmup' / 'progress.pt'
                if collection.exists():
                    trainer.load_checkpoint(collection)
                    trainer.logger.truncate_after_episode(trainer.completed_episodes)
                trainer.train(max(0, len(bank)-trainer.completed_episodes), scenarios=bank,
                              update_mode='collect_only', live_display=False, checkpoint_path=collection)
                trainer.save_checkpoint(collection)
                before = frozen_hashes(trainer)
                completed_updates = 0
                if progress.exists():
                    metadata = trainer.load_checkpoint(progress)
                    completed_updates = metadata['main_warmup_updates']
                for index in range(completed_updates, settings['cost_warmup_updates']):
                    batch = trainer.buffer.sample_n_step(cfg.learning.batch_size, cfg.learning.cost_target_n_step, cfg.learning.gamma_cost)
                    trainer.backend.update_cost_critic_only_n_step(batch)
                    if (index + 1) % 500 == 0:
                        # Count and model state must be durable in the same file.
                        trainer.save_checkpoint(progress, extra_metadata={'main_warmup_updates': index + 1})
                if before != frozen_hashes(trainer):
                    raise RuntimeError('Non-Cost parameters changed during warm-up')
                write(run / 'warmup' / 'audit.json', {'frozen_before': before, 'frozen_after': frozen_hashes(trainer),
                      'updates': settings['cost_warmup_updates'], 'collection_steps': trainer.total_environment_steps})
                trainer.save_checkpoint(warm)
        finally:
            trainer.env.close()
        # Exact load under the warm-up config; then explicitly hand off state.
        trainer = SafeRLTrainer(cfg, build_env(cfg), run / 'online')
        trainer.load_checkpoint(warm)
        trainer.total_environment_steps = trainer.total_gradient_updates = trainer.completed_episodes = 0
        trainer.transitions_collected_by_n = {}
        trainer.scenario_bank_cursor = 0
        trainer.scenario_bank_hash = None
        trainer.logger.truncate_after_episode(0)
        cfg.training.total_environment_steps_budget = min(target, settings['checkpoint_steps'][0])
        trainer.save_checkpoint(latest)
    else:
        cfg = config_for(data, method, seed, target)
        trainer = SafeRLTrainer(cfg, build_env(cfg), run / 'online')
        trainer.load_checkpoint(latest, allow_budget_extension=True)
        trainer.logger.truncate_after_episode(trainer.completed_episodes)
    try:
        bank = load_scenario_bank(root / 'banks' / 'training.json')
        milestones = sorted(set([v for v in settings['checkpoint_steps'] if v <= target] + [target]))
        for point in milestones:
            if trainer.total_environment_steps >= point:
                continue
            trainer.config.training.total_environment_steps_budget = point
            trainer.train(None, scenarios=bank, checkpoint_path=latest)
            trainer.save_checkpoint(latest)
            snapshot = run / f'model_{point}.pt'
            trainer.save_checkpoint(snapshot, include_replay=False)
            # Evaluate with a separate trainer and restore training RNG afterwards.
            rng = capture_rng()
            try:
                evaluate_condition(root, method, seed, snapshot, split='development', checkpoint_step=point)
            finally:
                restore_rng(rng)
        if trainer.total_environment_steps < target:
            raise RuntimeError('Target was not reached')
        final = run / f'model_{target}.pt'
        if not final.exists():
            trainer.save_checkpoint(final, include_replay=False)
        write(run / f'stage_{target}.json', {'target': target, 'actual_steps': trainer.total_environment_steps,
              'episodes': trainer.completed_episodes, 'checkpoint': str(final), 'source_sha256': trainer.source_hash,
              'resumable_checkpoint': str(latest), 'settings_sha256': data['settings_sha256']})
        return final
    finally:
        trainer.env.close()


def capture_rng():
    import torch
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    import torch
    random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2])
    if state[3] is not None:
        torch.cuda.set_rng_state_all(state[3])


def fork_tuning(parent: Path, root: Path, method: str, seed: int, *, learning_rate: float):
    """Explicit compatible LR branch: preserve replay/optimizer moments and RNG.

    Architecture, Reward/Cost definitions and scenario distributions stay fixed.
    This is deliberately not a generic override/hash-bypass interface.
    """
    import shutil
    import torch
    if method not in TRAINED or learning_rate <= 0:
        raise ValueError('Tuning requires A/B/F and a positive learning rate')
    data = context(parent)
    checkpoint = parent / 'training' / method / f'seed_{seed:03d}' / 'latest.pt'
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError('Tuning output must be empty')
    cfg = load_config(overrides=payload['config'])
    run = root / 'training' / method / f'seed_{seed:03d}'
    trainer = SafeRLTrainer(cfg, build_env(cfg), run / 'online')
    try:
        trainer.load_checkpoint(checkpoint)
        cfg.policy.learning_rate = learning_rate
        for optimizer in (trainer.backend.actor_optimizer, trainer.backend.critic_optimizer):
            for group in optimizer.param_groups:
                group['lr'] = learning_rate
        branch = deepcopy(data)
        branch['settings']['base_config']['policy']['learning_rate'] = learning_rate
        branch['settings_sha256'] = digest(branch['settings'])
        branch['parent'] = {'experiment': str(parent.resolve()), 'checkpoint_sha256': file_hash(checkpoint),
                            'method': method, 'seed': seed, 'actual_steps': trainer.total_environment_steps,
                            'changes': {'policy.learning_rate': learning_rate}}
        shutil.copytree(parent / 'banks', root / 'banks')
        write(root / 'experiment.json', branch)
        trainer.save_checkpoint(run / 'latest.pt')
    finally:
        trainer.env.close()
    return root


def seed_rollout(seed: int, scenario_id: str, repeat: int):
    import torch
    key = int(digest([seed, scenario_id, repeat])[:8], 16)
    random.seed(key); np.random.seed(key); torch.manual_seed(key)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(key)
    return key


def rollout_diagnostics(rollout: dict, cfg) -> dict:
    active = np.asarray(rollout['active_masks'][:-1], dtype=bool)
    nominal = np.asarray(rollout['nominal_actions'])
    correction = np.asarray(rollout['safe_actions']) - nominal
    pure = np.asarray(rollout['pure_cbf_actions']) - nominal
    amount = (correction ** 2).sum(-1)
    pure_amount = (pure ** 2).sum(-1)
    count = active.sum(-1).clip(min=1)
    weights = cfg.learning.gamma_cost ** np.arange(len(active))
    pure_cost = (pure_amount * active).sum(-1) / count
    distance = np.asarray([x['distance_cost'] for x in rollout['cost_components']])
    post = np.asarray(rollout['positions'][1:])
    pre = np.asarray(rollout['positions'][:-1])
    pair = np.linalg.norm(pre[:, :, None] - pre[:, None, :], axis=-1)
    valid_pair = active[:, :, None] & active[:, None, :] & ~np.eye(active.shape[1], dtype=bool)[None]
    nearest = np.where(valid_pair, pair, np.inf).min(-1)
    nz = active & (amount > 1e-12)
    diagnostics = {'discounted_distance_cost': float(weights @ distance),
        'discounted_pure_cost_common': float(weights @ pure_cost),
        'discounted_common_cost': float(weights @ (distance + pure_cost)),
        'runtime_correction_squared_sum': float(amount[active].sum()),
        'runtime_nonzero_agent_ticks': int(nz.sum()),
        'runtime_active_agent_ticks': int(active.sum()),
        'runtime_nonzero_squared_mean': float(amount[nz].mean()) if nz.any() else 0.,
        'warning_agent_tick_fraction': float(np.sum(active & (nearest < cfg.safety.d_warn)) / max(1, active.sum())),
        'max_abs_active_altitude': float(np.abs(post[..., 2][active]).max(initial=0)),
        'extent_outside_agent_ticks': int(np.sum(active & ((np.abs(post[..., 0]) > cfg.scenario.longitudinal_length/2) |
           (np.abs(post[..., 1]) > cfg.scenario.transverse_size/2) | (post[..., 2] < cfg.scenario.altitude_min) | (post[..., 2] > cfg.scenario.altitude_max))))}
    for name in ('safety_wall_times','evaluation_tick_times'):
        values = np.asarray(rollout.get(name, []))
        if values.size:
            diagnostics[name] = {**{f'p{q}': float(np.percentile(values, q)) for q in (50,95,99)},
                                 'mean': float(values.mean()), 'max': float(values.max())}
    diagnostics['pair_role_diagnostics'] = pair_roles(rollout, cfg)
    return diagnostics


def pair_roles(rollout, cfg):
    """Pre-main-style early CPA-opening and later safety-condition repair.

    Windows use tick-start positions/velocities, prior to first engagement and
    realized closest approach. These are observational, not causal credits.
    """
    p = np.asarray(rollout['positions'])
    v = np.asarray(rollout['velocities'])
    active = np.asarray(rollout['active_masks'][:-1], bool)
    actor = np.asarray(rollout['residual_actions'])
    nominal = np.asarray(rollout['nominal_actions'])
    safe = np.asarray(rollout['safe_actions'])
    adjacency = np.asarray(rollout['actor_adjacencies'])
    ticks = np.arange(len(actor))
    rows = []
    n = int(rollout['reset_info']['actual_agent_count'])
    for i in range(n):
        for j in range(i+1, n):
            valid = active[:,i] & active[:,j]
            if not valid.any():
                continue
            r = p[:-1,i]-p[:-1,j]; w = v[:-1,i]-v[:-1,j]
            d = np.linalg.norm(r,axis=-1)
            post_d = np.linalg.norm(p[1:,i]-p[1:,j],axis=-1)
            closest = int(np.argmin(np.where(valid, post_d, np.inf)))
            engaged = valid & (d <= cfg.safety.d_eng)
            first = int(np.flatnonzero(engaged)[0]) if engaged.any() else len(ticks)
            early = valid & (ticks < first) & (ticks < closest+1)
            tau = np.clip(-(r*w).sum(-1)/np.maximum((w*w).sum(-1),1e-12),0,cfg.scenario.prediction_horizon)
            q = r+tau[:,None]*w
            direction = q/np.maximum(np.linalg.norm(q,axis=-1,keepdims=True),1e-12)
            opening = {}
            for name,a in [('actor',actor),('nominal',nominal)]:
                values=tau*(direction*(a[:,i]-a[:,j])).sum(-1)
                opening[name] = float(values[early].mean()) if early.any() else None
            k1,k2 = cfg.safety.k1,cfg.safety.k2
            h=(r*r).sum(-1)-cfg.safety.d_safe**2
            base=2*(w*w).sum(-1)+2*(k1+k2)*(r*w).sum(-1)+k1*k2*h
            bn=base+2*(r*(nominal[:,i]-nominal[:,j])).sum(-1)
            bs=base+2*(r*(safe[:,i]-safe[:,j])).sum(-1)
            insufficient=engaged & (bn < -1e-7)
            rows.append({'first':i,'second':j,'early_ticks':int(early.sum()),
                'early_cpa_opening':opening,'minimum_distance':float(post_d[valid].min()),
                'first_engagement_tick':first if engaged.any() else None,
                'first_received_peer_tick':int(np.flatnonzero(adjacency[:,i,j] | adjacency[:,j,i])[0])
                    if np.any(adjacency[:,i,j] | adjacency[:,j,i]) else None,
                'nominal_insufficient_ticks':int(insufficient.sum()),
                'repaired_ticks':int(np.sum(insufficient & (bs >= -1e-7))),
                'warning_pair_seconds':float(np.sum(valid & (d<cfg.safety.d_warn))*cfg.environment.dt_base)})
    return rows


_EVALUATOR = None
_EVAL_ROOT = None
_EVAL_META = None


def _worker_init(checkpoint, output, meta, device):
    global _EVALUATOR, _EVAL_ROOT, _EVAL_META
    _EVAL_ROOT, _EVAL_META = Path(output), meta
    _EVALUATOR = load_trainer_checkpoint(checkpoint, Path(output) / 'runtime', device=device,
                                        config_overrides={'safety': {'enabled': meta['runtime_hocbf']}})


def _worker_case(item):
    scenario, repeat = item
    meta = _EVAL_META
    case = _EVAL_ROOT / 'cases' / digest([scenario['scenario_id'], repeat])[:24]
    completed = case / 'completed.json'
    if completed.exists():
        stored = json.loads(completed.read_text())
        if stored['identity'] != meta:
            raise ValueError('Evaluation identity changed')
        return stored['metrics']
    key = seed_rollout(meta['training_seed'], scenario['scenario_id'], repeat)
    result = evaluate(_EVALUATOR, reset_options={'scenario': scenario},
                      policy_execution_mode='stochastic' if meta['stochastic'] else 'mean',
                      pair_initial_action_with_cost=meta['stochastic'])
    row = result['episode_metrics'][0]
    row.update(method=meta['method'], training_seed=meta['training_seed'], repeat=repeat, rollout_rng_key=key,
               runtime_hocbf=meta['runtime_hocbf'], learning_budget=_EVALUATOR.config.learning.d_cost,
               learning_intervention_cost=_EVALUATOR.config.objective.intervention_cost_enabled,
               **rollout_diagnostics(result['rollouts'][0], _EVALUATOR.config))
    if meta['stochastic'] and not (row['paired_initial_action_matches_rollout'] and row['paired_initial_nominal_matches_rollout']):
        raise RuntimeError('Calibration action mismatch')
    save_evaluation(combine_evaluation_results([result], _EVALUATOR.config.environment.dt_base), case)
    write(completed, {'identity': meta, 'metrics': row})
    return row


def evaluate_condition(root: Path, method: str, seed: int, checkpoint: Path | None = None, *,
                       split='decision', stochastic=False, checkpoint_step=None, confirm=False):
    data = context(root)
    if split == 'confirmation_test' and not confirm:
        raise ValueError('Confirmation test requires explicit --confirm-test')
    source, enabled = CONDITIONS[method]
    settings = data['settings']
    step = checkpoint_step or settings['online_steps']
    if checkpoint is None:
        if source in TRAINED:
            checkpoint = root / 'training' / source / f'seed_{seed:03d}' / f'model_{step}.pt'
        else:
            seed = 0
            checkpoint = root / 'baselines' / f'{source}.pt'
            if not checkpoint.exists():
                cfg = config_for(data, source, 0, None)
                trainer = SafeRLTrainer(cfg, build_env(cfg), root / 'baselines' / source)
                try:
                    trainer.save_checkpoint(checkpoint, include_replay=False)
                finally:
                    trainer.env.close()
    if stochastic and method not in TRAINED:
        raise ValueError('Formal calibration applies to A/B/F ON only')
    bank = load_scenario_bank(root / 'banks' / f'{split}.json')
    if split == 'development':
        bank = subset(bank, settings['development_per_n'])
    if stochastic:
        bank = subset(bank, settings['calibration_per_n'])
    meta = {'method': method, 'training_method': source, 'training_seed': seed,
            'runtime_hocbf': enabled, 'stochastic': stochastic,
            'checkpoint_sha256': file_hash(checkpoint), 'source_sha256': data['source_sha256'],
            'bank_sha256': data['bank_hashes'][split], 'step_label': step}
    output = root / 'evaluation' / split / f'step_{step}' / ('stochastic' if stochastic else 'mean') / method / f'seed_{seed:03d}'
    repeats = settings['calibration_repeats'] if stochastic else 1
    tasks = [(s, repeat) for s in bank for repeat in range(repeats)]
    started = perf_counter()
    workers = settings.get('evaluation_workers', 1)
    if workers == 1:
        _worker_init(checkpoint, output, meta, settings['evaluation_device'])
        try:
            rows = [_worker_case(task) for task in tasks]
        finally:
            _EVALUATOR.env.close()
    else:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'),
              initializer=_worker_init, initargs=(checkpoint, output, meta, settings['evaluation_device'])) as pool:
            rows = list(pool.map(_worker_case, tasks))
    write(output / 'completed.json', {'identity': meta, 'episodes': len(rows), 'elapsed_seconds': perf_counter()-started})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare','train','evaluate','analyze','fork'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--settings', type=Path, default=Path('notebooks/main/settings.yaml'))
    parser.add_argument('--method', choices=list(CONDITIONS), default='A')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--target', type=int)
    parser.add_argument('--split', choices=['development','decision','confirmation_test'], default='decision')
    parser.add_argument('--stochastic', action='store_true')
    parser.add_argument('--confirm-test', action='store_true')
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--learning-rate', type=float)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == 'prepare':
        prepare(args.settings, root)
    elif args.action == 'train':
        print(train_method(root, args.method, args.seed, args.target))
    elif args.action == 'evaluate':
        evaluate_condition(root, args.method, args.seed, split=args.split, stochastic=args.stochastic,
                           checkpoint_step=args.target, confirm=args.confirm_test)
    elif args.action == 'fork':
        if args.parent is None or args.learning_rate is None:
            parser.error('fork requires --parent and --learning-rate')
        fork_tuning(args.parent.resolve(), root, args.method, args.seed, learning_rate=args.learning_rate)
    else:
        from uav_safe_marl.experiments.main_analysis import analyze
        print(analyze(root, args.split, args.target))


if __name__ == '__main__':
    main()
