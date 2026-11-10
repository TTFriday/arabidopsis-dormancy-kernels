"""Audit existing resampling results; no fitting or search for favorable settings.

All comparisons are retained. The three extra Holm families summarize exploratory
sensitivity analyses, not newly prespecified hypotheses. Original files are read-only.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "results" / "sensitivity"

def load(name):
    return json.loads((OUT / name).read_text(encoding='utf-8'))

def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adjusted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    out = np.empty_like(p)
    out[order] = np.minimum(1., adjusted)
    return out

def annotate(items, pkey):
    labels = list(items)
    adj = holm([items[k][pkey] for k in labels])
    return {k: {**items[k], 'p_holm': float(adj[i])} for i,k in enumerate(labels)}

def nb(d):
    d = np.asarray(d)
    sd = d.std(ddof=1)
    t = d.mean()/np.sqrt((1/len(d)+0.25)*sd**2)
    return {'mean_difference':float(d.mean()), 'sd_difference':float(sd),
            't_corrected':float(t), 'p_corrected':float(2*stats.t.sf(abs(t),len(d)-1)),
            'shapiro_w':float(stats.shapiro(d).statistic),
            'shapiro_p':float(stats.shapiro(d).pvalue),
            'diagnostic_note':'Normality test has low power at n=10; repeats are not independent biological samples.'}

def main():
    kernels = load('kernel_experiments.json')
    print('kernel result keys:', list(kernels))
    inf = load('inference_reanalysis.json')
    gam = load('gamma_lambda_sensitivity.json')
    ft = load('second_trait.json')
    group = load('grouped_permutation.json')['ward']
    gamma = {k: dict(v['vs_linear_block']) for k,v in gam['fixed_gamma_ablation'].items()}
    trait = {}
    for name,x in ft['traits'].items():
        trait[name+'_full'] = dict(x['paired_vs_block_mean']['local_rbf_fixed_0.1'])
        trait[name+'_excluded'] = dict(x['subset_without_dormancy_panel']['paired_vs_block_mean']['local_rbf_fixed_0.1'])
    grouped = {k:dict(v) for k,v in group.items() if isinstance(v,dict) and 'p_two_sided_abs' in v}
    result = {
        'purpose':'Consistent multiplicity accounting and numeric audit, not new outcome-driven model selection.',
        'gamma_13':annotate(gamma,'p_nadeau_bengio'),
        'flowering_6':annotate(trait,'p_nadeau_bengio'),
        'grouped_6':annotate(grouped,'p_two_sided_abs'),
        'original_primary_family':inf['primary_comparisons'],
        'input_sha256':{n:hashlib.sha256((OUT/n).read_bytes()).hexdigest() for n in
                         ['kernel_experiments.json','inference_reanalysis.json','gamma_lambda_sensitivity.json','second_trait.json','grouped_permutation.json']}
    }
    models=kernels.get('models',kernels)
    if 'block' in models and 'snp' in models and 'r_per_repeat' in models['block']:
        rb=np.asarray(models['block']['r_per_repeat'])
        rs=np.asarray(models['snp']['r_per_repeat'])
        result['paired_assumption_check']=nb(rb-rs)
        assert abs(result['paired_assumption_check']['p_corrected']-inf['primary_comparisons']['block_vs_snp']['p_nadeau_bengio'])<1e-8
    (OUT/'statistical_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('gamma=0.1 adjusted:',result['gamma_13']['gamma_0.1']['p_holm'])
    print('FT adjusted:',{k:round(v['p_holm'],6) for k,v in result['flowering_6'].items()})
    print('group min adjusted:', min(v['p_holm'] for v in result['grouped_6'].values()))

if __name__=='__main__':
    main()
