"""Offline figures from main case artifacts."""
from pathlib import Path
import numpy as np


def render(root: Path, output: Path, detailed, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=[]
    by_n=summary['primary']['by_n']
    metrics=['episode_success_rate','separation_violation_rate','path_stretch',
             'pure_safety_intervention_magnitude','communication_load']
    fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    for ax,metric in zip(axes.flat,metrics):
        for method in sorted(summary['primary']['overall']):
            ns=[int(n) for n,v in by_n.items() if method in v and v[method][metric]['mean'] is not None]
            ys=[by_n[str(n)][method][metric]['mean'] for n in ns]
            ax.plot(ns,ys,marker='o',label=method)
        ax.set_title(metric);ax.set_xlabel('N');ax.grid(alpha=.2)
    axes.flat[-1].axis('off');axes.flat[0].legend(ncol=3)
    fig.savefig(output/'primary_by_n.png',dpi=150);plt.close(fig);names.append('primary_by_n.png')
    mean=[d for d in detailed if not d['identity']['stochastic']]
    for n in sorted({d['metrics']['actual_agent_count'] for d in mean}):
        cases=[d for d in mean if d['metrics']['actual_agent_count']==n and d['metrics']['method']=='A']
        if not cases:continue
        def distance(d):
            return min((p['actual_minimum_distance'] for p in d['metrics']['rollout_pair_results']),default=np.inf)
        cases.sort(key=distance)
        for label,item in [('strict',cases[0]),('median',cases[len(cases)//2])]:
            file=next((root/item['case']/'rollouts').glob('*.npz'))
            with np.load(file,allow_pickle=False) as z:
                p=z['positions'];active=z['active_masks'];t=np.arange(len(p))*.1
                fig,ax=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
                for i in range(n):
                    q=np.where(active[:,i,None],p[:,i],np.nan)
                    ax[0,0].plot(q[:,0],q[:,1]);ax[0,1].plot(q[:,0],q[:,2]);ax[1,0].plot(t,q[:,2])
                for i in range(n):
                    for j in range(i+1,n):
                        valid=active[:-1,i]&active[:-1,j]
                        dist=np.linalg.norm(p[1:,i]-p[1:,j],axis=-1)
                        ax[1,1].plot(t[1:],np.where(valid,dist,np.nan),alpha=.35)
                ax[0,0].set(xlabel='x (m)',ylabel='y (m)');ax[0,1].set(xlabel='x (m)',ylabel='z (m)')
                ax[1,0].set(xlabel='time (s)',ylabel='z (m)');ax[1,1].set(xlabel='time (s)',ylabel='pair distance (m)')
                for threshold in (10,30,50):ax[1,1].axhline(threshold,ls='--',color='grey',lw=.6)
                fig.suptitle(f'A / N={n} / {label} / seed={item["metrics"]["training_seed"]}')
                filename=f'trajectory_A_n{n}_{label}.png'
                fig.savefig(output/filename,dpi=150);plt.close(fig);names.append(filename)
    return names
