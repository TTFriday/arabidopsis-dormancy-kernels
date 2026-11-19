"""Training-isolated flowering-time checks, without fixed dormancy bandwidths."""
import kernel_benchmark as core
from pathlib import Path
import contextlib
import hashlib
import json
import time
import numpy as np
import pandas as pd

TRAITS={'FT10':'5_FT10','FT16':'6_FT16','FT22':'7_FT22'}
RAW=core.ROOT/'data'/'raw'/'atpolydb'


def main():
    with (RAW/'call_method_75'/'call_method_75_TAIR9.csv').open('rb') as f:
        f.readline();header=f.readline().decode().strip().split(',')
    header_ids=np.asarray([int(v) for v in header[2:]])
    mapping={int(e):i for i,e in enumerate(header_ids)}
    full=np.load(core.DATA/'genotypes_all_1307.npy',mmap_mode='r')
    metadata=pd.read_csv(core.DATA/'snp_metadata.csv')
    groups=core.window_groups(metadata)
    pheno=pd.read_csv(RAW/'phenotype_published_raw.tsv',sep='\t',encoding='latin-1')
    dormancy=set(pd.read_csv(core.DATA/'dormancy_phenotype.csv').ecotype_id.astype(int))
    summary={'policy':'Additional related traits are exploratory, not independent confirmation. Every inner and outer training split rebuilds filters, indicators, scales and normalized kernels; window, bandwidth and ridge are selected inside each cohort independently.',
             'temperature_celsius':{'FT10':10,'FT16':16,'FT22':22},
             'cohorts':{},'status':'running','protocol':core.PROTOCOL}
    trait_sets={}
    for trait,column in TRAITS.items():
        values=pd.to_numeric(pheno[column],errors='coerce')
        frame=pheno.loc[values.notna(),['ecotype id']].copy()
        frame['phenotype']=values[values.notna()].to_numpy()
        frame=frame[frame['ecotype id'].isin(mapping)]
        ids=frame['ecotype id'].to_numpy(int)
        assert len(ids)==len(set(ids))
        trait_sets[trait]=set(ids)
        for excluded in [False,True]:
            selected=frame[~frame['ecotype id'].isin(dormancy)] if excluded else frame
            ids=selected['ecotype id'].to_numpy(int)
            cols=np.asarray([mapping[int(e)] for e in ids])
            assert np.array_equal(header_ids[cols],ids)
            y=selected.phenotype.to_numpy(float)
            assert np.isfinite(y).all()
            codes=np.asarray(full[:,cols]).T.copy()
            tag='flowering_'+trait+('_excluded' if excluded else '_full')
            overlap=len(set(ids)&dormancy)
            if excluded:assert overlap==0
            print(f'{tag}: beginning complete 10x5 nested CV, n={len(ids)}, dormancy-overlap={overlap}',flush=True)
            engine=core.GenomicKernels(codes,groups,local_dtype=core.torch.float32)
            start=time.time()
            with (core.RESULTS/(tag+'_run.log')).open('a',encoding='utf-8') as log:
                with contextlib.redirect_stdout(log):
                    predictions,metrics=core.nested_cv(engine,y,'strict',repeats=10,tag=tag)
            np.savez_compressed(core.RESULTS/(tag+'_identity.npz'),ids=ids,raw_genotype_columns=cols,y=y)
            summary['cohorts'][tag]={'n':len(ids),'dormancy_overlap':overlap,
                'ids_sha256':hashlib.sha256(ids.tobytes()).hexdigest(),
                'elapsed_this_run_seconds':time.time()-start,'models':metrics,
                'predictions':str(core.RESULTS/(tag+'_predictions.npz')),
                'parameter_state':str(core.RESULTS/(tag+'_state.json'))}
            core.save_json(core.RESULTS/'flowering_validation.json',summary)
            print(f'{tag}: finished '+str({k:v['pearson_mean'] for k,v in metrics.items() if k in ['snp','block_selected','local_selected']}),flush=True)
    summary['shared_full_panel_ids']=len(set.intersection(*trait_sets.values()))
    summary['status']='complete'
    summary['completed_cohorts']=len(summary['cohorts'])
    core.save_json(core.RESULTS/'flowering_validation.json',summary)
    print('All six full/excluded flowering-time cohorts completed and identity checked.',flush=True)


if __name__=='__main__':main()
