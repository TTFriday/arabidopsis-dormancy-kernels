"""Parallelize independent BayesC outer folds without changing their computation.

The sequential writer must be stopped before starting this coordinator. Each
worker owns a distinct (repeat, fold); only the coordinator writes aggregates.
Existing completed inner fits are reused with identical seeds/settings.
"""
import os
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
os.environ['NUMBA_NUM_THREADS']='2'
import strong_baselines as sb
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import time
import numpy as np
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits
import numba
import psutil

OUT=sb.OUTPUT
ARGS=SimpleNamespace(trees=300,inner_burn=1000,inner_draws=2000,outer_burn=2000,outer_draws=4000,extensions=1)
ENGINE=Y=IDS=None


def initialize_worker():
    global ENGINE,Y,IDS
    sys.stdout=(OUT/'parallel_workers'/f'worker_{os.getpid()}.log').open('a',encoding='utf-8',buffering=1)
    sys.stderr=sys.stdout
    sb.torch.set_num_threads(2);numba.set_num_threads(2)
    codes,Y,metadata,IDS,_=sb.kb.load_dormancy()
    ENGINE=sb.Preprocessor(codes,metadata)


def one_fold(job):
    rep,fold=job
    with threadpool_limits(2):
        train,test=list(KFold(5,shuffle=True,random_state=rep).split(Y))[fold]
        foldpath=OUT/'folds'/f'bayesc_r{rep:02d}_f{fold:02d}.json'
        if foldpath.exists() and foldpath.with_suffix('.npz').exists():return job
        began=time.perf_counter()
        pooled=np.full((len(train),len(sb.PI_GRID)),np.nan)
        inner_records=[]
        inner_seed=1000+rep*5+fold
        base=int(sb.kb.PROTOCOL['random_seed']+rep*100000+fold*10000)
        for inner,(it,iv) in enumerate(KFold(5,shuffle=True,random_state=inner_seed).split(train)):
            a,v=train[it],train[iv]
            split=sb.split_name(rep,fold,inner)
            taskpath=OUT/'tasks'/('bayesc_'+split+'.npz');taskmeta=taskpath.with_suffix('.json')
            if taskpath.exists() and taskmeta.exists():
                with np.load(taskpath,allow_pickle=False) as src:pooled[iv]=src['bayesc']
                inner_records.append(json.loads(taskmeta.read_text(encoding='utf-8')))
                continue
            zt,zq,info,audit=ENGINE.prepare(a,v,OUT/'preprocessing'/(split+'.npz'))
            pr=info['pruned_local'];preds=[];diagnostics=[]
            for candidate,pi in enumerate(sb.PI_GRID):
                prediction,diagnostic=sb.run_bayesc(zt[pr],zq[pr],Y[a],pi,
                    base+inner*100+candidate*7,ARGS.inner_burn,ARGS.inner_draws,ARGS.extensions,
                    OUT/'tasks'/('bayesc_'+split+f'_pi{candidate}.npz'))
                preds.append(prediction);diagnostics.append(diagnostic)
            pred=np.column_stack(preds)
            assert np.isfinite(pred).all();pooled[iv]=pred
            item={'inner_fold':inner,'train_indices':a.tolist(),'validation_indices':v.tolist(),
                  'seed':base+inner*100,'preprocessing':audit,'bayesc':diagnostics}
            sb.atomic_npz(taskpath,train_indices=a,validation_indices=v,bayesc=pred)
            sb.kb.save_json(taskmeta,item);inner_records.append(item)
            del zt,zq
        assert np.isfinite(pooled).all()
        losses=np.mean((pooled-Y[train,None])**2,axis=0);choice=int(losses.argmin())
        split=sb.split_name(rep,fold)
        zt,zq,info,audit=ENGINE.prepare(train,test,OUT/'preprocessing'/(split+'.npz'))
        pr=info['pruned_local']
        prediction,diagnostic=sb.run_bayesc(zt[pr],zq[pr],Y[train],sb.PI_GRID[choice],
            base+9000,ARGS.outer_burn,ARGS.outer_draws,ARGS.extensions,
            OUT/'tasks'/('bayesc_'+split+'.npz'))
        record={'repeat':rep,'fold':fold,'train_indices':train.tolist(),'test_indices':test.tolist(),
            'train_ids':IDS[train].tolist(),'test_ids':IDS[test].tolist(),'inner_seed':inner_seed,
            'outer_model_seed':base+9000,'preprocessing':audit,'inner_fits':inner_records,
            'selection':{'bayesc':{'candidate_index':choice,'parameters':sb.candidate_parameters('bayesc')[choice],
                'pooled_inner_mse':float(losses[choice]),'all_candidate_parameters':sb.candidate_parameters('bayesc'),
                'all_candidate_pooled_mse':losses.tolist()}},
            'outer_bayesc':diagnostic,'test_metrics':{'bayesc':sb.kb.metrics(Y[test],prediction)},
            'seconds':time.perf_counter()-began,'scheduling':'independent outer-fold worker; unchanged seeds and numerical settings'}
        sb.atomic_npz(foldpath.with_suffix('.npz'),train_indices=train,test_indices=test,bayesc=prediction,inner_oof_bayesc=pooled)
        sb.kb.save_json(foldpath,record)
        return job


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args()
    runtime=json.loads((OUT/'runtime.json').read_text(encoding='utf-8'))
    old_pid=runtime.get('pid')
    if old_pid and psutil.pid_exists(old_pid):
        p=psutil.Process(old_pid)
        if any('strong_baselines.py' in c for c in p.cmdline()):
            raise RuntimeError('Sequential writer still active; refusing overlapping writers.')
    _,y,_,ids,_=sb.kb.load_dormancy()
    jobs=[(r,f) for r in range(10) for f in range(5)
          if not ((OUT/'folds'/f'bayesc_r{r:02d}_f{f:02d}.json').exists() and
                  (OUT/'folds'/f'bayesc_r{r:02d}_f{f:02d}.npz').exists())]
    sb.kb.save_json(OUT/'parallel_runtime.json',{'status':'running','coordinator_pid':os.getpid(),
        'workers':args.workers,'jobs_remaining_at_start':len(jobs),'unchanged_numerical_settings':vars(ARGS),
        'policy':'Disjoint fold writers, one aggregate writer, same deterministic seeds and exact cached preprocessing.'})
    (OUT/'parallel_workers').mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize_worker) as pool:
        futures={pool.submit(one_fold,job):job for job in jobs}
        for future in as_completed(futures):
            rep,fold=future.result()
            count=sb.update_aggregate(OUT,'bayesc',y,ids)
            print(f'BayesC parallel: completed {count}/50; repeat={rep+1}, fold={fold+1}',flush=True)
    count=sb.update_aggregate(OUT,'bayesc',y,ids)
    assert count==50
    sb.kb.save_json(OUT/'parallel_runtime.json',{'status':'complete','coordinator_pid':os.getpid(),
        'workers':args.workers,'completed_outer_folds':count,'unchanged_numerical_settings':vars(ARGS)})
    print('All50 BayesC outer folds complete. Convergence failures remain explicitly reported.',flush=True)


if __name__=='__main__':main()
