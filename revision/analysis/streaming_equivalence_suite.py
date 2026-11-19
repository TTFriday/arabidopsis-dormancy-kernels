"""Check streaming regional means on every completed primary outer split."""
import computational_profile as profile
import kernel_benchmark as core
import json
import numpy as np


def main():
    codes,y,metadata,ids,groups=core.load_dormancy()
    state=json.loads((core.RESULTS/'strict_mixed_precision_state.json').read_text(encoding='utf-8'))
    checks=[]
    for record in state['folds']:
        tr=np.asarray(record['train_indices']);te=np.asarray(record['test_indices'])
        counts=np.stack([(codes[tr]==c).sum(0) for c in range(4)])
        minor=np.where(counts>0,counts,np.inf).argmin(0)
        keep=np.where(((counts>0).sum(0)==2)&((1-counts.max(0)/len(tr))>.05))[0]
        raw=codes[np.r_[tr,te]]
        x=(raw[:,keep]==minor[keep]).astype(float)
        scales={25,record['selection']['block_selected']['window_scale'],record['selection']['local_selected']['window_scale']}
        for scale in sorted(scales):
            group=groups[scale][keep]
            _,inverse,number=np.unique(group,return_inverse=True,return_counts=True)
            dense=profile.regional_means(x,group)
            streamed=profile.streaming_means(raw,keep,minor,inverse,number)
            assert np.array_equal(dense,streamed)
            checks.append({'repeat':record['repeat'],'fold':record['fold'],'window_scale':scale,
                'shape':list(dense.shape),'max_feature_error':float(np.max(abs(dense-streamed))),
                'training_marker_count':len(keep)})
    core.save_json(core.RESULTS/'streaming_equivalence_all_folds.json',{
        'passed':True,'outer_folds':len(state['folds']),'window_checks':len(checks),
        'scope':'Every outer fold and all distinct windows selected for the regional/local primary models, plus fixed25. Identical features imply identical downstream deterministic scaling and kernels; no additional accuracy claim.',
        'checks':checks})
    print('Streaming mean equivalence verified:',len(state['folds']),'outerfolds,',len(checks),'window checks; all exact.',flush=True)


if __name__=='__main__':main()
