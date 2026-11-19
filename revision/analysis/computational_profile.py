"""Isolated-process CPU accounting for fixed-parameter representation fits."""
import kernel_benchmark as core
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import psutil
from numba import njit
from threadpoolctl import threadpool_limits


def standardize(x,n):
    mean=x[:n].mean(0);sd=x[:n].std(0)
    keep=sd>1e-10
    return (x[:,keep]-mean[keep])/sd[keep]


def regional_means(x,group):
    _,inverse,counts=np.unique(group,return_inverse=True,return_counts=True)
    b=np.empty((len(x),len(counts)))
    for row in range(len(x)):
        b[row]=np.bincount(inverse,weights=x[row],minlength=len(counts))/counts
    return b


@njit(cache=False)
def streaming_means(raw,indexes,alleles,inverse,counts):
    result=np.zeros((raw.shape[0],len(counts)))
    for row in range(raw.shape[0]):
        for j in range(len(indexes)):
            if raw[row,indexes[j]]==alleles[indexes[j]]:
                result[row,inverse[j]]+=1.0
        for window in range(len(counts)):
            result[row,window]/=counts[window]
    return result


def one_model(name,x,group,n,ytrain,raw=None,indexes=None,alleles=None):
    begin=time.perf_counter()
    if name=='regional_streaming':
        _,inverse,counts=np.unique(group,return_inverse=True,return_counts=True)
        feature=streaming_means(raw,indexes,alleles,inverse,counts)
    else:
        feature=x if name=='snp' else regional_means(x,group)
    z=standardize(feature,n)
    feature_bytes=z.nbytes
    if name=='local':
        raw=np.zeros((len(x),len(x)))
        for start in range(0,z.shape[1],256):
            part=z[:,start:start+256]
            d=(part[:,None,:]-part[None,:,:])**2
            raw+=np.expm1(-.05*d).sum(2)
        raw/=z.shape[1]
    else:
        raw=z@z.T/z.shape[1]
    kt,kc=core.centered_pair(raw[:n,:n],raw[n:,:n])
    representation_time=time.perf_counter()-begin
    begin=time.perf_counter();pred=core.eig_predictions(kt,kc,ytrain,[1e-12])[:,0]
    fit_predict_time=time.perf_counter()-begin
    assert np.isfinite(pred).all()
    return {'representation_seconds':representation_time,'fit_predict_seconds':fit_predict_time,
        'feature_matrix_bytes':feature_bytes,'feature_count':z.shape[1],
        'training_kernel_bytes':kt.nbytes,'cross_kernel_bytes':kc.nbytes}


def worker(name):
    codes,y,meta,ids,groups=core.load_dormancy()
    state=json.loads((core.RESULTS/'strict_mixed_precision_state.json').read_text(encoding='utf-8'))
    first=state['folds'][0];tr=np.asarray(first['train_indices']);te=np.asarray(first['test_indices'])
    begin=time.perf_counter()
    counts=np.stack([(codes[tr]==c).sum(0) for c in range(4)])
    classes=(counts>0).sum(0)
    minor=np.where(counts>0,counts,np.inf).argmin(0)
    maf=1-counts.max(0)/len(tr)
    keep=np.where((classes==2)&(maf>.05))[0]
    raw=codes[np.r_[tr,te]]
    x=None if name=='regional_streaming' else (raw[:,keep]==minor[keep]).astype(float)
    group=groups[25][keep]
    common=time.perf_counter()-begin
    one_model(name,x,group,len(tr),y[tr],raw,keep,minor);gc.collect()
    trials=[]
    for repeat in range(3):
        trials.append(one_model(name,x,group,len(tr),y[tr],raw,keep,minor));gc.collect()
    memory=psutil.Process(os.getpid()).memory_info()
    has_peak=hasattr(memory,'peak_wset')
    report={'model':name,'cpu_threads':2,'train_n':len(tr),'test_n':len(te),
        'marker_n':len(keep),'fixed_window':25,'fixed_local_gamma':.05 if name=='local' else None,
        'fixed_lambda':1e-12,'shared_marker_preprocessing_seconds':common,
        'median_representation_seconds':float(np.median([v['representation_seconds'] for v in trials])),
        'median_fit_predict_seconds':float(np.median([v['fit_predict_seconds'] for v in trials])),
        'feature_matrix_bytes':trials[0]['feature_matrix_bytes'],
        'feature_count':trials[0]['feature_count'],
        'training_kernel_bytes':trials[0]['training_kernel_bytes'],
        'cross_kernel_bytes':trials[0]['cross_kernel_bytes'],
        'isolated_process_memory_bytes':int(memory.peak_wset if has_peak else memory.rss),
        'memory_metric':'Windows peak working set' if has_peak else 'RSS at end; not a peak',
        'scope':'Independent fresh process per model; same raw inputs and fixed split, three warm repeats. Common marker preparation is separate. No hyperparameter-search or GPU-training runtime is included. Concurrent system workloads may affect wall time.',
        'trials':trials}
    core.save_json(core.RESULTS/('computational_profile_'+name+'.json'),report)
    print(json.dumps(report),flush=True)


def equivalence_test():
    codes,y,meta,ids,groups=core.load_dormancy()
    from sklearn.model_selection import KFold
    tr,te=next(KFold(5,shuffle=True,random_state=0).split(y))
    counts=np.stack([(codes[tr]==c).sum(0) for c in range(4)])
    minor=np.where(counts>0,counts,np.inf).argmin(0)
    keep=np.where(((counts>0).sum(0)==2)&((1-counts.max(0)/len(tr))>.05))[0]
    raw=codes[np.r_[tr,te]]
    group=groups[25][keep]
    _,inverse,number=np.unique(group,return_inverse=True,return_counts=True)
    x=(raw[:,keep]==minor[keep]).astype(float)
    a=regional_means(x,group)
    b=streaming_means(raw,keep,minor,inverse,number)
    assert np.array_equal(a,b)
    predictions=[]
    for features in [a,b]:
        z=standardize(features,len(tr));k=z@z.T/z.shape[1]
        kt,kc=core.centered_pair(k[:len(tr),:len(tr)],k[len(tr):,:len(tr)])
        predictions.append(core.eig_predictions(kt,kc,y[tr],[1e-12])[:,0])
    error=float(np.max(abs(predictions[0]-predictions[1])))
    assert error<1e-10
    core.save_json(core.RESULTS/'streaming_aggregation_equivalence.json',{
        'passed':True,'regional_features_exactly_equal':True,'prediction_max_error_days':error,
        'raw_full_indicator_matrix_materialized_in_streaming_worker':False,
        'note':'Implementation equivalence only; no new kernel, fitted parameters, or primary scores are selected.'})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',choices=['snp','regional','regional_streaming','local'])
    args=parser.parse_args()
    with threadpool_limits(2):
        if args.model:worker(args.model);return
        equivalence_test()
        results={}
        for name in ['snp','regional','regional_streaming','local']:
            run=subprocess.run([sys.executable,'-s','-B',str(Path(__file__).resolve()),'--model',name],
                capture_output=True,text=True,encoding='utf-8')
            if run.returncode:raise RuntimeError(run.stdout+run.stderr)
            results[name]=json.loads((core.RESULTS/('computational_profile_'+name+'.json')).read_text(encoding='utf-8'))
        core.save_json(core.RESULTS/'computational_profile.json',{'models':results,
            'warning':'These fixed-parameter CPU fit measurements are not whole nested-CV/GPU-pretraining timings or universal speedup claims.'})
        print({n:{k:v[k] for k in ['median_representation_seconds','median_fit_predict_seconds','isolated_process_memory_bytes']} for n,v in results.items()},flush=True)


if __name__=='__main__':main()
