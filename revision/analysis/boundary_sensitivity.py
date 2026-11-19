"""Preserve primary results; separately test ridge plateaus and the linear limit."""
import kernel_benchmark as core
import contextlib
from pathlib import Path
import json
import numpy as np
from sklearn.model_selection import KFold


class WithLinearLimit(core.GenomicKernels):
    def build(self,train,query,scales=None,local=True,global_rbf=False):
        out=super().build(train,query,scales,local,global_rbf)
        for key,pair in list(out.items()):
            if key[0]=='block':
                out[('local',key[1],0.)]=pair
        return out


def main():
    codes,y,metadata,ids,groups=core.load_dormancy()
    engine=core.GenomicKernels(codes,groups,local_dtype=core.torch.float32)
    penalties=np.array([1e-12,1e-10,1e-8,1e-6,1e-4,.001,.01,.1,1.,10.,100.])
    predictions={name:np.zeros((len(penalties),10,len(y))) for name in ['snp','block25']}
    for rep in range(10):
        for tr,te in KFold(5,shuffle=True,random_state=rep).split(y):
            kernels=engine.build(tr,te,[25],local=False)
            for name,key in [('snp',('snp',0,0.)),('block25',('block',25,0.))]:
                p=core.eig_predictions(*kernels[key],y[tr],penalties)
                predictions[name][:,rep,te]=p.T
    reports={name:{f'{lam:g}':core.summarize(y,p[i]) for i,lam in enumerate(penalties)} for name,p in predictions.items()}
    core.save_json(core.RESULTS/'fixed_ridge_sensitivity.json',{
        'role':'Sensitivity only: training-isolated preprocessing and fixed25/reference windows; no selection of a headline ridge setting from outer performance.',
        'penalties':penalties.tolist(),'models':reports,
        'maximum_prediction_difference_1e12_to_1e4_days':{name:float(np.max(abs(p[0]-p[4]))) for name,p in predictions.items()}})
    np.savez_compressed(core.RESULTS/'fixed_ridge_sensitivity.npz',y=y,ids=ids,lambdas=penalties,**predictions)
    print('Fixed-ridge plateau evaluated on all50 training-isolated folds.',flush=True)
    engine=WithLinearLimit(codes,groups,local_dtype=core.torch.float32)
    with (core.RESULTS/'linear_limit_sensitivity.log').open('a',encoding='utf-8') as log:
        with contextlib.redirect_stdout(log):
            saved,metrics=core.nested_cv(engine,y,'strict',repeats=10,tag='linear_limit_sensitivity')
    state=json.loads((core.RESULTS/'linear_limit_sensitivity_state.json').read_text(encoding='utf-8'))
    selected=[f['selection']['local_selected']['gamma'] for f in state['folds']]
    core.save_json(core.RESULTS/'boundary_sensitivity_summary.json',{
        'role':'Revision boundary sensitivity, not a replacement for the primary frozen protocol.',
        'linear_limit_definition':'gamma=0 is defined by the normalized regional linear limit, not by a zero raw Gaussian kernel.',
        'completed_outer_folds':len(state['folds']),
        'selected_linear_limit_outer_folds':selected.count(0.),'local_metrics':metrics['local_selected'],
        'primary_outputs_modified':False})
    print('Linear-limit selection sensitivity complete; chosen in',selected.count(0.),'of50 folds.',flush=True)


if __name__=='__main__':main()
