"""Ten-PC regression with PCA fitted only within each outer training fold."""
import kernel_benchmark as core
import numpy as np
from scipy import linalg
from sklearn.model_selection import KFold


def main():
    codes,y,metadata,ids,groups=core.load_dormancy()
    engine=core.GenomicKernels(codes,groups,local_dtype=core.torch.float32)
    predictions=np.zeros((10,len(y)))
    records=[]
    for rep in range(10):
        for fold,(tr,te) in enumerate(KFold(5,shuffle=True,random_state=rep).split(y)):
            kt,kc=engine.build(tr,te,scales=[],local=False)[('snp',0,0.)]
            q,u=linalg.eigh(kt)
            keep=np.argsort(q)[-10:]
            assert np.all(q[keep]>1e-10)
            # PCA in feature space and this Gram representation are equivalent.
            # Only the first ten TRAINING principal components enter OLS.
            predictions[rep,te]=y[tr].mean()+kc@u[:,keep]@((u[:,keep].T@(y[tr]-y[tr].mean()))/q[keep])
            records.append({'repeat':rep,'fold':fold,'n_pcs':10,'train_ids':ids[tr].tolist(),
                'test_ids':ids[te].tolist(),'preprocessing':dict(engine.last_audit),
                'explained_training_kernel_trace_fraction':float(q[keep].sum()/q[q>0].sum()),
                'test_metrics':core.metrics(y[te],predictions[rep,te])})
    core.save_json(core.RESULTS/'pc_regression_metrics.json',{
        'status':'complete','outer_folds':len(records),'n_pcs_fixed':10,
        'method':'Ordinary least squares on ten PCs of training-standardized retained SNPs; eigenspace fitted independently in every outer training fold. No test genotypes/phenotypes fit PCA. Number10 retains the submitted fixed-PC comparator, not outer-score selection.',
        'model':core.summarize(y,predictions),'folds':records})
    np.savez_compressed(core.RESULTS/'pc_regression_predictions.npz',y=y,ids=ids,predictions=predictions)
    print('Training-only10-PC regression complete:',core.summarize(y,predictions)['pearson_mean'],flush=True)


if __name__=='__main__':main()
