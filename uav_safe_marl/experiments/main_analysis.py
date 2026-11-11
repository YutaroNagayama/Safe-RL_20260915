"""Offline main analysis from immutable per-case logs; never reruns policies."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from uav_safe_marl.evaluation.statistics import _seed_metrics, aggregate_primary_by_method_and_n
from .main import write


def calibration(rows):
    output = {}
    for method in sorted({r['method'] for r in rows}):
        output[method] = {}
        for n in ['all', *sorted({int(r['actual_agent_count']) for r in rows if r['method'] == method})]:
            group = [r for r in rows if r['method'] == method and (n == 'all' or r['actual_agent_count'] == n)]
            actual = np.array([r['discounted_episode_cost'] for r in group])
            prediction = np.array([r['start_state_cost_ucb'] for r in group])
            complete = np.array([not r['truncated'] for r in group])
            threshold = group[0]['learning_budget']
            high = actual > threshold
            def correlation(a,b):
                return float(np.corrcoef(a,b)[0,1]) if len(a)>1 and np.std(a)>0 and np.std(b)>0 else None
            output[method][str(n)] = {'count': len(group), 'truncated_count': int((~complete).sum()),
                'mc_mean_observed': float(actual.mean()), 'mc_median_observed': float(np.median(actual)),
                'mc_p95_observed': float(np.quantile(actual, .95)), 'mc_max_observed': float(actual.max()),
                'ucb_mae_observed': float(np.abs(actual-prediction).mean()),
                'ucb_bias_observed': float((prediction-actual).mean()),
                'empirical_coverage_observed': float(np.mean(actual <= prediction)),
                'pearson_observed': correlation(actual,prediction),
                'budget_false_negative_observed': float(np.mean(prediction[high]<=threshold)) if high.any() else None,
                'budget_false_positive_observed': float(np.mean(prediction[~high]>threshold)) if (~high).any() else None,
                'note': 'Truncated MC is censored; pooled correlation does not imply within-N ranking'}
    return output


def paired_differences(rows, samples=10000, seed=20261103):
    """Crossed paired seed x geometry bootstrap, preserving ratio denominators."""
    groups = {}
    for r in rows:
        groups.setdefault(r['method'], {})[(int(r['training_seed']), r['scenario_id'])] = r
    rng = np.random.default_rng(seed)
    output = {}
    for left, right in [('A','B'),('A','E'),('A','F'),('A','C'),('B','D'),('C','D'),('F','F_OFF')]:
        if left not in groups or right not in groups:
            continue
        seeds = sorted({s for s, _ in groups[left]})
        if right not in ('E','R'):
            seeds = sorted(set(seeds) & {s for s, _ in groups[right]})
        if not seeds:
            continue
        strata = {}
        for n in sorted({r['actual_agent_count'] for r in rows}):
            ids = set.intersection(*[
                {key[1] for key, row in groups[left].items() if key[0] == s and row['actual_agent_count'] == n}
                & {key[1] for key, row in groups[right].items() if key[0] == (0 if right in ('E','R') else s) and row['actual_agent_count'] == n}
                for s in seeds])
            if ids:
                strata[n] = sorted(ids)
        if not strata:
            continue
        def value(selected_seeds, selected_ids):
            diffs = {}
            for s in selected_seeds:
                for ids in selected_ids.values():
                    a = _seed_metrics([groups[left][(s, i)] for i in ids])
                    b = _seed_metrics([groups[right][(0 if right in ('E','R') else s, i)] for i in ids])
                    for metric in a:
                        if a[metric] is not None and b[metric] is not None:
                            diffs.setdefault(metric, []).append(a[metric]-b[metric])
            return {k: float(np.mean(v)) for k,v in diffs.items()}
        observed = value(seeds, strata)
        draws = {k: [] for k in observed}
        for _ in range(samples):
            ss = rng.choice(seeds, len(seeds), replace=True).tolist()
            ii = {n: rng.choice(ids, len(ids), replace=True).tolist() for n,ids in strata.items()}
            for k,v in value(ss, ii).items():
                draws[k].append(v)
        output[f'{left}-{right}'] = {'seed_count': len(seeds), 'geometry_count_by_n': {n:len(ids) for n,ids in strata.items()},
            'metrics': {k: {'difference': v, 'ci95': np.quantile(draws[k],[.025,.975]).tolist() if draws[k] else None} for k,v in observed.items()}}
    return output


def analyze(root: Path, split='decision', target=None, bootstrap_samples=10000):
    manifest = json.loads((root/'experiment.json').read_text())
    step = target or manifest['settings']['online_steps']
    source = root/'evaluation'/split/f'step_{step}'
    mean, stochastic, detailed = [], [], []
    for path in sorted(source.glob('*/*/seed_*/cases/*/completed.json')):
        case = json.loads(path.read_text())
        row = case['metrics']
        (stochastic if case['identity']['stochastic'] else mean).append(row)
        detailed.append({'case':str(path.parent.relative_to(root)), 'identity':case['identity'], 'metrics':row})
    if not mean:
        raise ValueError('No mean-policy evaluation cases found')
    output = root/'analysis'/split/f'step_{step}'
    summary = {'primary':aggregate_primary_by_method_and_n(mean),
               'paired':paired_differences(mean, bootstrap_samples), 'calibration':calibration(stochastic),
               'mean_episode_count':len(mean), 'stochastic_episode_count':len(stochastic),
               'note':'Partial results are allowed and explicitly counted. C/D reuse A/B. Zero-event bootstrap is not a safety guarantee.'}
    write(output/'summary.json',summary)
    write(output/'episodes.json',detailed)
    try:
        from .main_plots import render
        figures = render(root,output,detailed,summary)
    except ImportError:
        figures = []
    methods = sorted({r['method'] for r in mean})
    lines = ['# Main実験の保存結果分析', '', f'対象：{split}、checkpoint目標{step}step。mean評価{len(mean)}本、確率評価{len(stochastic)}本。', '',
             '|方式|episode数|全機通過率|違反episode数|平均観測割引Cost|実補正非ゼロ率|', '|---|---:|---:|---:|---:|---:|']
    for m in methods:
        rr=[r for r in mean if r['method']==m]
        nonzero=sum(r['runtime_nonzero_agent_ticks'] for r in rr)/max(1,sum(r['runtime_active_agent_ticks'] for r in rr))
        lines.append(f"|{m}|{len(rr)}|{np.mean([r['episode_success'] for r in rr]):.4f}|{sum(r['separation_violation_pair_ticks']>0 for r in rr)}|{np.mean([r['discounted_episode_cost'] for r in rr]):.4f}|{nonzero:.4f}|")
    lines += ['', 'N別・seed別・対応差・Cost校正はsummary.json、全episode診断と元軌道参照はepisodes.jsonを参照。',
              'Full軌道には位置・速度、Reference/Residual/nominal/pure/safe、活動mask、通信/CBF graph、情報age、Cost、毎tickの安全制約診断を保存。',
              'timeout例を削除しない。保存時刻は位置がtick境界、入力・制約診断はその区間に対応する。未評価方式・seedは未完了として扱う。']
    lines += ['', *[f'![{name}]({name})' for name in figures]]
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return output/'REPORT.md'
