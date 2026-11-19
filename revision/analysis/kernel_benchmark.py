"""Training-only genomic kernel benchmark with auditable nested CV.

The CPU eigensolver uses centered float64 kernels and an explicit unpenalized
intercept. Optional CUDA accelerates feature products and local similarities,
not hyperparameter selection. Original project files are never written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
import scipy
from scipy import linalg, stats
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits
# Use the PyTorch installation selected by the active environment.
import torch
_cuda_available = torch.cuda.is_available()

HERE = Path(__file__).resolve().parent
REVISION = HERE.parent
ROOT = REVISION.parent
DATA = ROOT / 'data' / 'processed'
RESULTS = REVISION / 'results'
PROTOCOL = json.loads((HERE / 'protocol.json').read_text(encoding='utf-8'))
LAMBDAS = np.asarray(PROTOCOL['lambda_grid'], dtype=float)
GAMMAS = PROTOCOL['gamma_grid']
SCALES = PROTOCOL['window_scales']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32 = False


def save_json(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def metrics(y, pred):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    varied = np.std(pred) > 1e-12 and np.std(y) > 1e-12
    denom = np.sum((y-y.mean())**2)
    return {
        'pearson': float(stats.pearsonr(y, pred)[0]) if varied else None,
        'spearman': float(stats.spearmanr(y, pred)[0]) if varied else None,
        'mae': float(np.mean(abs(y-pred))),
        'rmse': float(np.sqrt(np.mean((y-pred)**2))),
        'predictive_r2': float(1-np.sum((y-pred)**2)/denom) if denom > 0 else None,
        'calibration_slope': float(np.dot(pred-pred.mean(), y-y.mean()) / np.sum((pred-pred.mean())**2)) if varied else None,
        'bias': float(np.mean(pred-y)),
    }


def summarize(y, predictions):
    values = [metrics(y, p) for p in predictions]
    out = {'per_repeat': values, 'averaged_prediction': metrics(y, predictions.mean(0))}
    for name in values[0]:
        v = [row[name] for row in values if row[name] is not None]
        out[name+'_mean'] = float(np.mean(v)) if v else None
        out[name+'_sd'] = float(np.std(v, ddof=1)) if len(v)>1 else None
    return out


def window_groups(metadata):
    """Partition reference coordinates independently of genotypes/outcomes."""
    ranges = np.load(DATA / 'block_snp_map.npz')
    base = np.empty(len(metadata), dtype=np.int32)
    chrom = metadata.chrom.to_numpy()
    groups = {s: np.empty(len(metadata), dtype=np.int32) for s in SCALES}
    for s in (5, 10, 25):
        k = 0
        for a, b in zip(ranges['snp_start'], ranges['snp_end']):
            step = int(b-a) if s == 25 else s
            for start in range(int(a), int(b), step):
                groups[s][start:min(int(b), start+step)] = k
                k += 1
    base[:] = groups[25]
    for scale, merge in ((50, 2), (100, 4)):
        k = 0
        for c in np.unique(chrom):
            bi = np.unique(base[chrom==c])
            for start in range(0, len(bi), merge):
                selected = bi[start:start+merge]
                for old in selected:
                    groups[scale][base==old] = k
                k += 1
    return groups


def load_dormancy():
    pheno = pd.read_csv(DATA / 'dormancy_phenotype.csv')
    metadata = pd.read_csv(DATA / 'snp_metadata.csv')
    codes = np.load(DATA / 'genotypes_83.npy').T.copy()
    cols = np.load(DATA / 'dormancy_cols_in_1307.npy')
    full = np.load(DATA / 'genotypes_all_1307.npy', mmap_mode='r')
    assert np.array_equal(codes.T, full[:, cols])
    assert len(pheno.ecotype_id.unique()) == len(pheno) == codes.shape[0]
    assert np.isfinite(pheno.dsds50).all()
    return codes, pheno.dsds50.to_numpy(float), metadata, pheno.ecotype_id.to_numpy(), window_groups(metadata)


def centered_pair(train, cross):
    train = np.asarray(train, float)
    cross = np.asarray(cross, float)
    mu = train.mean(0)
    total = float(mu.mean())
    kt = train-mu[:, None]-mu[None, :]+total
    scale = float(np.trace(kt)/len(kt))
    if scale <= 1e-14:
        scale = 1.0
    return (kt+kt.T)/(2*scale), (cross-cross.mean(1)[:, None]-mu[None, :]+total)/scale


def eig_predictions(kt, cross, y, lambdas=LAMBDAS):
    """All penalty predictions; no inverse subtraction at tiny penalties."""
    y = np.asarray(y, float)
    h = np.eye(len(y))-np.ones((len(y), len(y)))/len(y)
    q, u = linalg.eigh(h @ kt @ h, check_finite=False)
    threshold = max(1.0, float(np.max(abs(q))))*1e-13
    q = np.where(q>threshold, q, 0.0)
    coeff = u.T @ (y-y.mean())
    coeff[q==0] = 0.0
    weights = coeff[:, None]/(q[:, None]+np.asarray(lambdas)[None, :])
    return y.mean()+cross @ u @ weights


class GenomicKernels:
    def __init__(self, codes, groups, fixed_markers=None, fixed_minor=None, legacy_features=None, local_dtype=torch.float64):
        self.codes = torch.as_tensor(np.asarray(codes).copy(), dtype=torch.uint8, device=DEVICE)
        self.groups = {int(s): torch.as_tensor(g.astype(np.int64), device=DEVICE) for s,g in groups.items()}
        self.fixed_markers = fixed_markers
        self.fixed_minor = fixed_minor
        self.legacy_features = legacy_features
        self.local_dtype = local_dtype
        self.last_audit = {}

    def indicators(self, train, query):
        anchor = self.codes[torch.as_tensor(train, dtype=torch.long, device=DEVICE)]
        if self.fixed_markers is None:
            counts = torch.stack([(anchor==c).sum(0) for c in range(4)])
            called = counts.sum(0)
            nonzero = (counts>0).sum(0)
            masked = counts.to(torch.float64).masked_fill(counts==0, float('inf'))
            minor = masked.argmin(0)
            maf = counts.max(0).values.to(torch.float64)/called.clamp(min=1)
            keep = (nonzero==2)&((1-maf)>0.05)
            indexes = torch.where(keep)[0]
        else:
            indexes = torch.as_tensor(self.fixed_markers, device=DEVICE)
            minor = torch.as_tensor(self.fixed_minor, device=DEVICE)
        rows = np.r_[train, query]
        allrows = self.codes[torch.as_tensor(rows, dtype=torch.long, device=DEVICE)]
        allele = minor[indexes]
        x = (allrows[:, indexes]==allele[None, :]).to(torch.float64)
        n = len(train)
        self.last_audit = {'retained_markers': int(len(indexes)), 'training_n': n,
                           'marker_index_sha256': hashlib.sha256(indexes.cpu().numpy().tobytes()).hexdigest(),
                           'minor_code_sha256': hashlib.sha256(allele.cpu().numpy().tobytes()).hexdigest()}
        return x, indexes, n

    @staticmethod
    def standardized(x, n):
        mean = x[:n].mean(0)
        sd = x[:n].std(0, unbiased=False)
        valid = sd>1e-10
        return (x[:, valid]-mean[valid])/sd[valid], valid

    @staticmethod
    def pair(raw, n):
        a = raw[:n, :n].cpu().numpy()
        c = raw[n:, :n].cpu().numpy()
        return centered_pair(a, c)

    def build(self, train, query, scales=None, local=True, global_rbf=False):
        scales = SCALES if scales is None else list(scales)
        x, indexes, n = self.indicators(train, query)
        out = {}
        z, _ = self.standardized(x, n)
        raw = z @ z.T / z.shape[1]
        out[('snp', 0, 0.0)] = self.pair(raw, n)
        if global_rbf:
            d = torch.clamp(raw.diag()[:, None]+raw.diag()[None, :]-2*raw, min=0)
            for gamma in GAMMAS:
                out[('snp_rkhs', 0, gamma)] = self.pair(torch.exp(-gamma*d), n)
        for scale in scales:
            gid = self.groups[scale][indexes]
            unique, inverse, counts = torch.unique(gid, sorted=True, return_inverse=True, return_counts=True)
            b = torch.zeros((len(x), len(unique)), dtype=torch.float64, device=DEVICE)
            b.scatter_add_(1, inverse[None, :].expand(len(x), -1), x)
            b /= counts[None, :]
            if self.legacy_features is not None:
                assert scale==25
                b = torch.as_tensor(self.legacy_features[np.r_[train, query]], dtype=torch.float64, device=DEVICE)
            bz, valid = self.standardized(b, n)
            rawb = bz @ bz.T / bz.shape[1]
            out[('block', scale, 0.0)] = self.pair(rawb, n)
            if local:
                accum = torch.zeros((len(GAMMAS), len(x), len(x)), dtype=torch.float64, device=DEVICE)
                gamma_tensor = torch.as_tensor(GAMMAS, dtype=self.local_dtype, device=DEVICE)
                for start in range(0, bz.shape[1], 512):
                    part = bz[:, start:start+512].to(self.local_dtype)
                    d = (part[:, None, :]-part[None, :, :])**2
                    # Vectorize bandwidths; expm1 avoids small-gamma cancellation.
                    values = torch.expm1(-gamma_tensor[:, None, None, None]*d[None, :, :, :])
                    accum += values.sum(3, dtype=torch.float64)
                accum /= bz.shape[1]
                for i,gamma in enumerate(GAMMAS):
                    out[('local', scale, gamma)] = self.pair(accum[i], n)
            self.last_audit['window_'+str(scale)+'_variable_features'] = int(bz.shape[1])
        return out


def candidate_families(key, mode):
    kind, scale, gamma = key
    if kind=='snp':
        return ['snp']
    if kind=='snp_rkhs':
        return ['snp_rkhs']
    if mode=='fair':
        return ['block_matched_25'] if kind=='block' else ['local_matched_25']
    if kind=='block':
        return ['block_selected', 'block_fixed_25'] if scale==25 else ['block_selected']
    if kind=='local':
        return ['local_selected']
    return []


def nested_cv(engine, y, mode, repeats=10, scales=None, local=True, global_rbf=False, tag=None):
    tag = tag or mode
    n = len(y)
    path = RESULTS/(tag+'_state.json')
    checkpoint = RESULTS/(tag+'_predictions.npz')
    state = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {
        'mode':mode, 'repeats':repeats, 'folds':[], 'device':str(DEVICE), 'protocol':PROTOCOL,
        'started':time.strftime('%Y-%m-%d %H:%M:%S')}
    assert state['repeats']==repeats
    saved = {k:v for k,v in dict(np.load(checkpoint)).items() if k != 'y'} if checkpoint.exists() else {}
    done = {(f['repeat'], f['fold']) for f in state['folds']}
    started = time.time()
    for rep in range(repeats):
        for fold,(train,test) in enumerate(KFold(5, shuffle=True, random_state=rep).split(y)):
            if (rep,fold) in done:
                continue
            selection = {}
            losses = {}
            inner = KFold(5, shuffle=True, random_state=1000+rep*5+fold)
            for it,iv in inner.split(train):
                a,v = train[it],train[iv]
                kernels = engine.build(a,v,scales,local,global_rbf)
                for key,(kt,kv) in kernels.items():
                    pred = eig_predictions(kt,kv,y[a])
                    loss = ((pred-y[v,None])**2).sum(0)
                    for family in candidate_families(key,mode):
                        choice_key = (family,key)
                        losses[choice_key] = losses.get(choice_key,np.zeros(len(LAMBDAS)))+loss
            families = sorted({k[0] for k in losses})
            for family in families:
                candidates = [(float(v.min()/len(train)), k[1], int(v.argmin()))
                              for k,v in losses.items() if k[0]==family]
                selection[family] = min(candidates,key=lambda a:(a[0],a[1],a[2]))
            outer = engine.build(train,test,scales,local,global_rbf)
            audit = dict(engine.last_audit)
            record = {'repeat':rep,'fold':fold,'train_indices':train.tolist(),
                      'test_indices':test.tolist(),'preprocessing':audit,'selection':{}}
            for family,(loss,key,ilam) in selection.items():
                kt,kv = outer[key]
                pred = eig_predictions(kt,kv,y[train],[LAMBDAS[ilam]])[:,0]
                if family not in saved:
                    saved[family] = np.full((repeats,n),np.nan)
                saved[family][rep,test] = pred
                q = np.maximum(linalg.eigvalsh(kt),0)
                nz = q[q>max(1.,q.max())*1e-10]
                record['selection'][family] = {'kind':key[0],'window_scale':key[1],
                    'gamma':key[2],'lambda':float(LAMBDAS[ilam]),'inner_mse':loss,
                    'test_metrics':metrics(y[test],pred),'spectrum':{
                        'largest_eigenvalue':float(q.max()),'positive_rank':int(len(nz)),
                        'condition_positive':float(nz.max()/nz.min()) if len(nz) else None,
                        'effective_dimension':float(np.sum(q/(q+LAMBDAS[ilam]))),
                        'participation_rank':float(q.sum()**2/np.sum(q*q))}}
            for name,value in [('train_mean',y[train].mean()),('train_median',np.median(y[train]))]:
                if name not in saved:
                    saved[name] = np.full((repeats,n),np.nan)
                saved[name][rep,test] = value
            state['folds'].append(record)
            state['elapsed_current_run_seconds'] = time.time()-started
            with checkpoint.with_suffix('.tmp').open('wb') as f:
                np.savez_compressed(f,y=y,**saved)
            checkpoint.with_suffix('.tmp').replace(checkpoint)
            save_json(path,state)
            print(f'{tag} repeat={rep+1}/{repeats} fold={fold+1}/5 '+
                  ' '.join(f'{k}: {v[1]},lambda={LAMBDAS[v[2]]:g}' for k,v in selection.items()),flush=True)
    assert all(np.isfinite(p).all() for p in saved.values())
    report = {name:summarize(y,p) for name,p in saved.items()}
    save_json(RESULTS/(tag+'_metrics.json'),report)
    return saved,report


def selftest():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(20,5)); t=rng.normal(size=(4,5)); y=rng.normal(size=20)
    raw=x@x.T/5; cross=t@x.T/5
    kt,kc=centered_pair(raw,cross)
    alpha_b=np.linalg.solve(np.block([[kt+0.3*np.eye(20),np.ones((20,1))],
                                     [np.ones((1,20)),np.zeros((1,1))]]),np.r_[y,0.])
    direct=kc@alpha_b[:-1]+alpha_b[-1]
    stable=eig_predictions(kt,kc,y,[0.3])[:,0]
    error=float(np.max(abs(direct-stable)))
    assert error<1e-10,error
    assert abs(kt.mean())<1e-12
    assert abs(np.trace(kt)/len(kt)-1)<1e-12
    save_json(RESULTS/'package_kernel_unit_tests.json',{'passed':True,'augmented_solver_prediction_max_error':error})


def fair_ablation(codes,y,metadata,groups):
    """Fixed, matched input ablation; no claim of strict inductive evaluation."""
    kept=np.where((metadata.maf_83.to_numpy()>0.05)&(metadata.minor_code.to_numpy()<4))[0]
    minor=metadata.minor_code.to_numpy()
    x=(codes[:,kept]==minor[kept][None,:]).astype(float)
    ids=groups[25][kept]
    _,inv,counts=np.unique(ids,return_inverse=True,return_counts=True)
    b=np.zeros((len(y),len(counts)))
    for i in range(len(y)):
        b[i]=np.bincount(inv,weights=x[i],minlength=len(counts))/counts
    legacy=np.asarray(np.load(DATA/'block_features_83.npy')[:,:,0],float)
    counts_all=metadata[['n_A_all','n_C_all','n_G_all','n_T_all']].to_numpy()
    present=np.where(counts_all>0,counts_all,np.inf)
    minor_reference=present.argmin(1)
    arrays={'snp':x,'block_original':legacy,'block_matched':b}
    # Factorial input audit: change marker filtering and allele convention
    # separately while keeping reference-coordinate windows fixed.
    for name,markers,alleles in [
            ('block_filtered_reference_coding',kept,minor_reference),
            ('block_all_training_coding',np.arange(codes.shape[1]),minor)]:
        gid=groups[25][markers]
        _,inverse,number=np.unique(gid,return_inverse=True,return_counts=True)
        means=np.zeros((len(y),len(number)))
        for row in range(len(y)):
            indicator=(codes[row,markers]==alleles[markers]).astype(float)
            means[row]=np.bincount(inverse,weights=indicator,minlength=len(number))/number
        arrays[name]=means
    raws={}
    for name,a in arrays.items():
        sd=np.where(a.std(0)>1e-10,a.std(0),1.)
        z=(a-a.mean(0))/sd
        raws[name]=z@z.T/a.shape[1]
    preds={name:np.zeros((10,len(y))) for name in arrays}
    # Legacy tuning convention: full-panel features and kernel slices.
    legacy_lam=np.logspace(-8,2,21)
    for rep in range(10):
        for tr,te in KFold(5,shuffle=True,random_state=rep).split(y):
            for name,raw in raws.items():
                costs=np.zeros(len(legacy_lam))
                for it,iv in KFold(5,shuffle=True,random_state=rep*100).split(tr):
                    a,v=tr[it],tr[iv]
                    # With explicit intercept, centering changes neither prediction nor scale;
                    # keep original trace units for this reproduction/ablation only.
                    k=raw[np.ix_(a,a)]; c=raw[np.ix_(v,a)]
                    h=np.eye(len(a))-np.ones((len(a),len(a)))/len(a)
                    kt=h@k@h; kc=c-c.mean(1)[:,None]-k.mean(0)[None,:]+k.mean()
                    p=eig_predictions(kt,kc,y[a],legacy_lam)
                    costs+=((p-y[v,None])**2).sum(0)
                ilam=costs.argmin()
                k=raw[np.ix_(tr,tr)]; c=raw[np.ix_(te,tr)]
                h=np.eye(len(tr))-np.ones((len(tr),len(tr)))/len(tr)
                preds[name][rep,te]=eig_predictions(h@k@h,c-c.mean(1)[:,None]-k.mean(0)[None,:]+k.mean(),y[tr],[legacy_lam[ilam]])[:,0]
    np.savez_compressed(RESULTS/'fair_input_predictions.npz',y=y,**preds)
    report={name:summarize(y,p) for name,p in preds.items()}
    report['protocol']='Legacy panel-level scaling and lambda grid preserved solely to isolate input matching; new primary evaluation is strictly training-only.'
    report['retained_markers']=len(kept)
    save_json(RESULTS/'fair_input_metrics.json',report)
    print('fair input ablation:',{k:v['pearson_mean'] for k,v in report.items() if isinstance(v,dict)},flush=True)


def fair_local_ablation(codes,y,metadata,groups):
    kept=np.where((metadata.maf_83.to_numpy()>0.05)&(metadata.minor_code.to_numpy()<4))[0]
    minor=metadata.minor_code.to_numpy()
    group=groups[25][kept]
    _,inverse,counts=np.unique(group,return_inverse=True,return_counts=True)
    matched=np.zeros((len(y),len(counts)))
    for row in range(len(y)):
        x=(codes[row,kept]==minor[kept]).astype(float)
        matched[row]=np.bincount(inverse,weights=x,minlength=len(counts))/counts
    original=np.asarray(np.load(DATA/'block_features_83.npy')[:,:,0],float)
    gammas=[.05,.1,.2,.5,1.,2.,5.,10.]
    penalties=np.logspace(-8,2,21)
    predictions={}
    selections={}
    for name,features in [('local_original',original),('local_matched',matched)]:
        z=(features-features.mean(0))/np.where(features.std(0)>1e-10,features.std(0),1.)
        tensor=torch.as_tensor(z,dtype=torch.float64,device=DEVICE)
        raw=torch.zeros((len(gammas),len(y),len(y)),dtype=torch.float64,device=DEVICE)
        gg=torch.as_tensor(gammas,dtype=torch.float64,device=DEVICE)
        for start in range(0,z.shape[1],512):
            part=tensor[:,start:start+512]
            d=(part[:,None,:]-part[None,:,:])**2
            raw+=torch.expm1(-gg[:,None,None,None]*d[None,:,:,:]).sum(3)
        raw/=z.shape[1]
        candidates=[]
        for i,gamma in enumerate(gammas):
            matrix=raw[i].cpu().numpy()
            normalized,_=centered_pair(matrix,np.empty((0,len(y))))
            candidates.append(normalized)
        pred=np.zeros((10,len(y)));chosen=[]
        for rep in range(10):
            for fold,(tr,te) in enumerate(KFold(5,shuffle=True,random_state=rep).split(y)):
                costs=np.zeros((len(gammas),len(penalties)))
                for it,iv in KFold(5,shuffle=True,random_state=rep*100).split(tr):
                    a,v=tr[it],tr[iv]
                    for i,matrix in enumerate(candidates):
                        kt=matrix[np.ix_(a,a)];kv=matrix[np.ix_(v,a)]
                        h=np.eye(len(a))-np.ones((len(a),len(a)))/len(a)
                        cross=kv-kv.mean(1)[:,None]-kt.mean(0)[None,:]+kt.mean()
                        p=eig_predictions(h@kt@h,cross,y[a],penalties)
                        costs[i]+=((p-y[v,None])**2).sum(0)
                ig,il=np.unravel_index(costs.argmin(),costs.shape)
                kt=candidates[ig][np.ix_(tr,tr)];kv=candidates[ig][np.ix_(te,tr)]
                h=np.eye(len(tr))-np.ones((len(tr),len(tr)))/len(tr)
                cross=kv-kv.mean(1)[:,None]-kt.mean(0)[None,:]+kt.mean()
                pred[rep,te]=eig_predictions(h@kt@h,cross,y[tr],[penalties[il]])[:,0]
                chosen.append({'repeat':rep,'fold':fold,'gamma':gammas[ig],'lambda':float(penalties[il])})
        predictions[name]=pred;selections[name]=chosen
    report={'protocol':'Legacy fixed-window, full-panel feature scaling and 8-bandwidth/21-penalty grids retained; matched local inputs exactly equal the SNP baseline indicators. This isolates input matching, not strict inductive evaluation.',
            'models':{name:summarize(y,pred) for name,pred in predictions.items()},'selected_parameters':selections}
    np.savez_compressed(RESULTS/'fair_local_predictions.npz',y=y,**predictions)
    save_json(RESULTS/'fair_local_metrics.json',report)
    print('fair local ablation:',{k:v['pearson_mean'] for k,v in report['models'].items()},flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['tests','fair','fair-local','strict','smoke'],default='tests')
    args=parser.parse_args()
    with threadpool_limits(2):
        selftest()
        if args.mode=='tests':
            print('kernel unit tests passed',flush=True); return
        codes,y,meta,ids,groups=load_dormancy()
        if args.mode=='fair':
            fair_ablation(codes,y,meta,groups); return
        if args.mode=='fair-local':
            fair_local_ablation(codes,y,meta,groups); return
        engine=GenomicKernels(codes,groups,local_dtype=torch.float32)
        np.save(REVISION/'data'/'dormancy_ids.npy',ids)
        if args.mode=='smoke':
            tr,te=next(KFold(5,shuffle=True,random_state=0).split(y))
            start=time.time(); kernels=engine.build(tr,te,scales=[25],local=True)
            print('smoke kernel count',len(kernels),'seconds',time.time()-start,flush=True)
            print(engine.last_audit,flush=True)
            for key,(kt,kc) in kernels.items():
                assert np.isfinite(kt).all() and np.isfinite(kc).all()
                assert abs(np.trace(kt)/len(kt)-1)<1e-8
                assert abs(kt.mean())<1e-8
            return
        nested_cv(engine,y,'strict',repeats=10,global_rbf=True,tag='strict_mixed_precision')

if __name__=='__main__':
    main()
