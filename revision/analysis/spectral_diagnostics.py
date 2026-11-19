"""Kernel geometry and small-penalty diagnostics on a fixed training split."""
import kernel_benchmark as core
from pathlib import Path
import json
import time
import numpy as np
from scipy import linalg


def spectrum(kt, penalty):
    q=np.maximum(linalg.eigvalsh(kt),0)
    nz=q[q>max(1.,q.max())*1e-10]
    return {'eigenvalues_ascending':q.tolist(),
        'positive_rank':int(len(nz)),
        'positive_condition':float(nz.max()/nz.min()) if len(nz) else None,
        'participation_rank':float(q.sum()**2/(q@q)),
        'ridge_effective_dimension':float(np.sum(q/(q+penalty))),
        'positive_mode_shrinkage_min':float(np.min(nz/(nz+penalty))) if len(nz) else None}


def main():
    codes,y,meta,ids,groups=core.load_dormancy()
    state=json.loads((core.RESULTS/'strict_mixed_precision_state.json').read_text(encoding='utf-8'))
    first=state['folds'][0]
    tr=np.asarray(first['train_indices']);te=np.asarray(first['test_indices'])
    engine=core.GenomicKernels(codes,groups,local_dtype=core.torch.float32)
    start=time.perf_counter();kernels=engine.build(tr,te,[25],True,True)
    elapsed=time.perf_counter()-start
    names={('snp',0,0.):'SNP',('block',25,0.):'Regional',('local',25,.001):'Local, γ=0.001',('local',25,.2):'Local, γ=0.2'}
    details={}
    for key,name in names.items():
        kt,kc=kernels[key]
        train_pred=core.eig_predictions(kt,kt,y[tr],[1e-12])[:,0]
        detail=spectrum(kt,1e-12)
        detail['training_rmse_at_lambda_1e12']=float(np.sqrt(np.mean((train_pred-y[tr])**2)))
        detail['train_n']=len(tr)
        detail['test_metrics_at_fixed_lambda']=core.metrics(y[te],core.eig_predictions(kt,kc,y[tr],[1e-12])[:,0])
        details[name]=detail
    linear=kernels[('block',25,0.)][0]
    angles=[]
    for gamma in core.GAMMAS:
        local=kernels[('local',25,gamma)][0]
        cosine=float(np.sum(linear*local)/(np.linalg.norm(linear)*np.linalg.norm(local)))
        angles.append({'gamma':gamma,'angular_deviation':max(0.,1-cosine),
            'relative_frobenius_distance':float(np.linalg.norm(linear-local)/np.linalg.norm(linear))})
    marker_count=engine.last_audit['retained_markers']
    bcount=engine.last_audit['window_25_variable_features']
    report={'scope':'Fixed first outer training split, no outcome-based choice of this diagnostic split. Not an additional accuracy benchmark.',
        'train_ids':ids[tr].tolist(),'test_ids':ids[te].tolist(),
        'device':str(core.DEVICE),'gpu_name':core.torch.cuda.get_device_name(0) if core.DEVICE.type=='cuda' else None,
        'spectrum':details,'local_geometry':angles,
        'computational_accounting':{
            'markers':marker_count,'regional_features':bcount,
            'snp_feature_matrix_bytes_float64':int(len(codes)*marker_count*8),
            'regional_feature_matrix_bytes_float64':int(len(codes)*bcount*8),
            'both_full_kernel_matrix_bytes_float64':int(len(codes)**2*8),
            'feature_storage_reduction_factor':float(marker_count/bcount),
            'joint_feature_and_kernel_construction_seconds':elapsed,
            'note':'Feature compression does not reduce the n-by-n kernel dimension. Joint runtime is not a separate model-specific speedup and may be affected by concurrent GPU workloads.'},
        'interpretation':'At tiny positive lambda and full positive rank n_train-1, training interpolation has essentially no explicit positive-mode shrinkage. Kernel geometry still determines the minimum-RKHS-norm interpolant and its cross-sample extension; spectrum is not evidence of substantial explicit ridge shrinkage.'}
    core.save_json(core.RESULTS/'spectral_diagnostics.json',report)
    print({name:{k:v for k,v in d.items() if k in ['positive_rank','participation_rank','training_rmse_at_lambda_1e12']} for name,d in details.items()})


if __name__=='__main__':
    main()
