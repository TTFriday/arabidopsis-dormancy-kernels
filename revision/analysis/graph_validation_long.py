"""Independent-reference masked graph validation; original project is read-only.

Only this new script, results/graph_validation.json and results/graph_reference/
are written. Fixed-coordinate windows are the sole legacy preprocessing reused.
Run with the system scientific stack: python -s -B <this-file> --mode run.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
HERE = Path(__file__).resolve().parent
REVISION = HERE.parent
ROOT = REVISION.parent
DATA = ROOT / 'data' / 'processed'
RESULTS = REVISION / 'results'
OUT = RESULTS / 'graph_reference_long'
REPORT = RESULTS / 'graph_validation_long.json'
# Import the compatible NumPy/SciPy stack BEFORE the observed user CUDA torch.
_spec = importlib.util.spec_from_file_location('graph_kernel_core', HERE / 'kernel_benchmark.py')
core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(core)
np, pd, torch = core.np, core.pd, core.torch
from scipy import linalg
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits
import torch.nn as nn
import torch.nn.functional as F

DEVICE = core.DEVICE
WIDTHS = (16, 64)
SEEDS = (20261024, 7, 42)
SPLIT_SEED = 20261024
MASK_SEED = 20261025
CONFIG = {'widths': list(WIDTHS), 'seeds': list(SEEDS), 'in_dim': 4,
          'heads': 4, 'n_layers': 2, 'edge_dim': 6, 'dropout': 0.1,
          'mask_ratio': 0.15, 'epochs': 100, 'batch_size': 32,
          'microbatch_size': 16, 'lr': 0.003, 'weight_decay': 0.0001,
          'gradient_clip_norm': 5.0, 'scheduler': 'cosine, T_max=100',
          'reference_split_seed': SPLIT_SEED, 'validation_mask_seed': MASK_SEED,
          'reference_validation_fraction': 0.1,
          'outer_repeats': 10, 'outer_folds': 5, 'inner_folds': 5,
          'lambda_grid': core.LAMBDAS.tolist(),
          'selection': 'Pooled inner-validation MSE; tie uses lambda grid order.',
          'policy': 'Locked widths/seeds; no downstream-best encoder selection.'}
OLD_ENCODER = ROOT / 'scripts' / 'reference_encoder.py'


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def clean(obj):
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, np.generic):
        return clean(obj.item())
    if isinstance(obj, Path):
        return obj.as_posix()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def save_json(path, obj):
    path = Path(path)
    assert path == REPORT or OUT in path.parents
    tmp = path.with_suffix(path.suffix + '.tmp')
    text = json.dumps(clean(obj), indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    tmp.write_text(text, encoding='utf-8')
    try:
        tmp.replace(path)
    except PermissionError as exc:
        # Windows live readers can deny DELETE/rename sharing while permitting
        # ordinary writes. Publish ONLY our authorized new result file in place;
        # never change permissions or abandon completed training checkpoints.
        if os.name != 'nt' or not path.exists() or getattr(exc, 'winerror', None) not in (5, 32):
            raise
        path.write_text(text, encoding='utf-8')
        tmp.unlink(missing_ok=True)


def save_npz(path, **arrays):
    path = Path(path)
    assert OUT in path.parents
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(path)


def save_torch(path, obj):
    path = Path(path)
    assert OUT in path.parents
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp)
    tmp.replace(path)


def read_torch(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def report_base():
    if REPORT.exists():
        result = json.loads(REPORT.read_text(encoding='utf-8'))
        assert result['locked_experiment'] == CONFIG, 'Experiment differs from locked report'
        return result
    return {'status': 'not_started', 'new_training_filename': Path(__file__).resolve(),
            'locked_experiment': CONFIG, 'original_files_modified': False,
            'legacy_disclosure': 'Old PCs, allele conventions, LD edges and graph embeddings used the full 1307 panel and remain legacy/transductive. None is reused here. Only outcome-independent coordinate windows and encoder class definitions are reused.',
            'encoder_source': OLD_ENCODER, 'encoder_source_sha256': sha_file(OLD_ENCODER),
            'core_source': HERE / 'kernel_benchmark.py',
            'core_source_sha256_at_start': sha_file(HERE / 'kernel_benchmark.py'),
            'runtime': {'python': sys.executable, 'numpy': np.__version__,
                        'torch': torch.__version__, 'device': str(DEVICE)},
            'runs': {}, 'limitations': [
                'Reference validation selects the best masked-reconstruction epoch; its reported minimum is not an unbiased third-test estimate.',
                'Reconstruction improvement does not imply downstream improvement.',
                'Ten CV repeats describe split variability, not ten independent biological replications.',
                'Reference preprocessing is fitted only on reference train, then frozen; held-out genotypes never enter that fit.',
                'CUDA scatter reductions may be nondeterministic; seeds and RNG states are saved, but bitwise device reproducibility is not promised.']}


def encoder_class():
    # Execute only the two existing class definitions, not legacy I/O/imports.
    tree = ast.parse(OLD_ENCODER.read_text(encoding='utf-8'))
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and
               n.name in {'GraphTransformerLayer', 'MaskedGraphEncoder'}]
    assert len(classes) == 2
    namespace = {'torch': torch, 'nn': nn, 'F': F}
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(OLD_ENCODER), 'exec'), namespace)
    return namespace['MaskedGraphEncoder']


Encoder = encoder_class()


def identities():
    header = ROOT / 'data' / 'raw' / 'atpolydb' / 'call_method_75' / 'call_method_75_TAIR9.csv'
    with header.open('r', encoding='utf-8', newline='') as f:
        rows = csv.reader(f)
        first, second = next(rows), next(rows)
    ids = np.array([int(v) for v in second[2:]], dtype=np.int64)
    panel = pd.read_csv(DATA / 'accessions_all.csv')
    pheno_ids = pd.read_csv(DATA / 'dormancy_phenotype.csv', usecols=['ecotype_id']).ecotype_id.to_numpy(np.int64)
    dormcols = np.load(DATA / 'dormancy_cols_in_1307.npy')
    assert len(ids) == len(set(ids)) == len(panel) == 1307
    assert len(first[2:]) == len(ids)
    assert set(ids) == set(panel.ecotype_id)
    assert np.array_equal(ids[dormcols], pheno_ids)
    assert len(pheno_ids) == len(set(pheno_ids)) == 83
    rawcol = {int(e): i for i, e in enumerate(ids)}
    # Audit rows preserve ALL names; combine exact and normalized candidates.
    # The sorted metadata row is used only to recover an ID, NEVER a raw column.
    audit_path = RESULTS / 'dryad_matching_identity_audit.csv'
    audit = pd.read_csv(audit_path)
    exact, normalized = defaultdict(set), defaultdict(set)
    norm = lambda v: str(v).strip().lower().replace(' ', '')
    for r in panel.itertuples():
        exact[str(r.nativename)].add(int(r.ecotype_id))
        normalized[norm(r.nativename)].add(int(r.ecotype_id))
    dryad_ids, ambiguous, candidate_rows = set(), [], []
    for r in audit.itertuples():
        candidates = sorted(exact[str(r.nativename)] | normalized[norm(r.nativename)])
        dryad_ids.update(candidates)
        legacy = json.loads(r.candidate_ecotype_ids)
        assert set(legacy).issubset(candidates)
        if int(r.panel_col) >= 0:
            assert int(panel.iloc[int(r.panel_col)].ecotype_id) in candidates
        if len(candidates) > 1:
            ambiguous.append({'name': r.nativename, 'candidate_ecotype_ids': candidates,
                              'candidate_raw_columns': [rawcol[e] for e in candidates]})
        if candidates:
            candidate_rows.append({'name': r.nativename, 'candidate_ecotype_ids': candidates})
    assert len(audit) == 523 and int((audit.panel_col >= 0).sum()) == 206
    ext_path = RESULTS / 'external_validation_published29_matching.csv'
    ext = pd.read_csv(ext_path, usecols=['canonical_ecotype_id', 'rawcol'])
    ext_ids = ext.canonical_ecotype_id.to_numpy(np.int64)
    assert len(ext_ids) == len(set(ext_ids)) == 29
    assert not set(ext_ids) & set(pheno_ids)
    assert np.array_equal(ids[ext.rawcol.to_numpy(int)], ext_ids)
    assert set(dryad_ids) | set(ext_ids) <= set(ids)
    excluded = set(pheno_ids) | set(ext_ids) | dryad_ids
    reference_cols = np.array([i for i, e in enumerate(ids) if e not in excluded], dtype=np.int64)
    reference_ids = ids[reference_cols]
    perm = np.random.default_rng(SPLIT_SEED).permutation(len(reference_cols))
    nv = int(round(0.1 * len(reference_cols)))
    val_cols = np.sort(reference_cols[perm[:nv]])
    train_cols = np.sort(reference_cols[perm[nv:]])
    external_ids = np.array(sorted((dryad_ids | set(ext_ids)) - set(pheno_ids)), dtype=np.int64)
    external_cols = np.array([rawcol[int(e)] for e in external_ids], dtype=np.int64)
    checks = {'reference_disjoint_dormancy83': not bool(set(reference_ids) & set(pheno_ids)),
              'reference_disjoint_published29': not bool(set(reference_ids) & set(ext_ids)),
              'reference_disjoint_ALL_dryad_candidates': not bool(set(reference_ids) & dryad_ids),
              'train_validation_disjoint': not bool(set(train_cols) & set(val_cols)),
              'train_validation_partition_reference': set(train_cols) | set(val_cols) == set(reference_cols),
              'raw_header_dormancy_alignment': True, 'raw_header_external29_alignment': True,
              'all_ambiguous_candidates_excluded': all(not set(a['candidate_ecotype_ids']) & set(reference_ids) for a in ambiguous)}
    assert all(checks.values())
    info = {'raw_header_file': header, 'raw_header_id_sha256': digest(ids),
            'metadata_raw_column_order_mismatches': int(np.sum(panel.ecotype_id.to_numpy() != ids)),
            'identity_rule': 'Second raw CSV header ecotype ID -> raw column; metadata positions are never genotype indices.',
            'dryad_audit_source': audit_path, 'dryad_audit_sha256': sha_file(audit_path),
            'external29_identity_source': ext_path,
            'reference_ids': reference_ids, 'reference_ids_sha256': digest(reference_ids),
            'reference_train_ids': ids[train_cols], 'reference_train_ids_sha256': digest(ids[train_cols]),
            'reference_validation_ids': ids[val_cols], 'reference_validation_ids_sha256': digest(ids[val_cols]),
            'n_reference': len(reference_ids), 'nreftrain': len(train_cols), 'nval': len(val_cols),
            'excluded_union_ids': sorted(excluded), 'excluded_union_n': len(excluded),
            'excluded_dormancy_ids': pheno_ids, 'excluded_external29_ids': ext_ids,
            'excluded_dryad_ALL_candidate_ids': sorted(dryad_ids),
            'dryad_candidate_id_n': len(dryad_ids), 'dryad_legacy_matched_rows': 206,
            'ambiguous_names_and_ALL_excluded_candidates': ambiguous,
            'dryad_name_candidate_records': candidate_rows,
            'exclude_overlap_assertions': checks,
            'external_applied_ids': external_ids,
            'external_note': 'All non-dormancy name candidates are transformed for provenance, including ambiguous ones. No ambiguous phenotype matching or external outcome evaluation is performed here.'}
    save_npz(OUT / 'identity_columns.npz', header_ids=ids, reference_columns=reference_cols,
             train_columns=train_cols, validation_columns=val_cols,
             dormancy_columns=dormcols, external_columns=external_cols,
             dormancy_ids=pheno_ids, external_ids=external_ids)
    return ids, train_cols, val_cols, dormcols, external_cols, info


def fit_block(codes):
    """codes: reference TRAIN accessions by SNP; deterministic loading sign."""
    counts = np.stack([(codes == a).sum(0) for a in range(4)], axis=1)
    minor = np.argmin(np.where(counts > 0, counts, np.inf), axis=1).astype(np.uint8)
    minor[(counts > 0).sum(1) < 2] = 255
    x = (codes == minor[None, :]).astype(np.float64)
    # Missing code 255 must not count as a monomorphic pseudo-minor allele.
    x[:, minor == 255] = 0
    mean = x.mean(0)
    z = x - mean
    q, u = linalg.eigh(z.T @ z / len(z), check_finite=False)
    loadings = np.zeros((len(minor), 3), dtype=np.float64)
    for k, j in enumerate(np.argsort(-q, kind='stable')[:3]):
        if q[j] <= 1e-12:
            continue
        v = u[:, j].copy()
        if v[np.argmax(abs(v))] < 0:
            v *= -1
        loadings[:, k] = v
    return minor, mean, loadings, counts


def apply_block(codes, minor, mean, loadings):
    x = (codes == minor[None, :]).astype(np.float64)
    x[:, minor == 255] = 0
    return np.column_stack((x.mean(1), (x - mean) @ loadings)).astype(np.float32)


def reference_edges(train_feats, blockmap):
    mean = np.asarray(train_feats[:, :, 0], dtype=np.float64).T
    sd = mean.std(1, keepdims=True)
    z = (mean - mean.mean(1, keepdims=True)) / np.where(sd > 1e-12, sd, 1)
    chrom, starts, ends = blockmap['chrom'], blockmap['start'], blockmap['end']
    edges = {}
    for c in np.unique(chrom):
        local = np.where(chrom == c)[0]
        for i, j in zip(local[:-1], local[1:]):
            edges[(int(i), int(j))] = (float(starts[j] - ends[i]) / 1e6, 1.0, 0)
        for k, i in enumerate(local):
            cand = local[k + 1:k + 51]
            cand = cand[(starts[cand] - ends[i]) <= 1000000]
            if not len(cand):
                continue
            r = z[cand] @ z[i] / len(train_feats)
            order = np.argsort(-abs(r), kind='stable')[:3]
            for j in order:
                if abs(r[j]) >= 0.5:
                    key = (int(i), int(cand[j]))
                    if key not in edges:  # Coordinate adjacency has precedence.
                        edges[key] = (float(starts[cand[j]] - ends[i]) / 1e6, float(abs(r[j])), 1)
    edge_index = np.array(list(edges), dtype=np.int64).T
    attr = np.array(list(edges.values()), dtype=np.float32)
    n = train_feats.shape[1]
    src, dst = edge_index
    loop = np.arange(n, dtype=np.int64)
    virt = np.full(n, n, dtype=np.int64)
    src_all = np.r_[src, dst, loop, virt, loop]
    dst_all = np.r_[dst, src, loop, loop, virt]
    attr_all = np.concatenate([attr, attr, np.tile([0., 1., 2.], (n, 1)),
                               np.tile([0., 1., 3.], (n, 1)), np.tile([0., 1., 3.], (n, 1))]).astype(np.float32)
    cont = attr_all[:, :2].copy()
    cont[:, 0] = np.clip(cont[:, 0], 0, 1)
    efeat = np.column_stack((cont, np.eye(4, dtype=np.float32)[attr_all[:, 2].astype(int)]))
    assert efeat.shape == (len(src_all), 6)
    return {'edge_index': edge_index, 'edge_attr': attr, 'src': src_all, 'dst': dst_all, 'efeat': efeat}


def prepare(report):
    manifest_path = OUT / 'reference_manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        assert manifest['locked_experiment'] == CONFIG
        assert manifest['block_map_sha256'] == sha_file(DATA / 'block_snp_map.npz')
        for filename in manifest['required_artifacts']:
            assert (OUT / filename).is_file(), filename
        report['reference'] = manifest
        return manifest
    ids, train_cols, val_cols, dorm_cols, external_cols, identity = identities()
    codes = np.load(DATA / 'genotypes_all_1307.npy', mmap_mode='r')
    assert codes.shape == (214051, len(ids))
    assert np.array_equal(codes[:, dorm_cols], np.load(DATA / 'genotypes_83.npy'))
    blockmap = dict(np.load(DATA / 'block_snp_map.npz'))
    a, b = blockmap['snp_start'], blockmap['snp_end']
    assert a[0] == 0 and b[-1] == len(codes) and np.array_equal(a[1:], b[:-1])
    assert np.all((b - a > 0) & (b - a <= 25))
    nblocks = len(a)
    feats = np.zeros((len(ids), nblocks, 4), np.float32)
    minor_all = np.zeros(len(codes), np.uint8)
    means_all = np.zeros(len(codes), np.float64)
    loadings_all = np.zeros((len(codes), 3), np.float64)
    counts_all = np.zeros((len(codes), 4), np.int32)
    started = time.time()
    for k, (lo, hi) in enumerate(zip(a, b)):
        block_codes = np.asarray(codes[lo:hi]).T
        minor, mean, loading, counts = fit_block(block_codes[train_cols])
        feats[:, k] = apply_block(block_codes, minor, mean, loading)
        minor_all[lo:hi], means_all[lo:hi] = minor, mean
        loadings_all[lo:hi], counts_all[lo:hi] = loading, counts
        if (k + 1) % 1000 == 0:
            print(f'reference PCA {k + 1}/{nblocks}, elapsed={time.time() - started:.1f}s', flush=True)
    # Every fitted scale is reference TRAIN-only, even though reference validation is unlabeled.
    flat_train = feats[train_cols].reshape(-1, 4).astype(np.float64)
    mu, sd = flat_train.mean(0), flat_train.std(0)
    sd[sd < 1e-6] = 1
    edges = reference_edges(feats[train_cols], blockmap)
    baseline_block = np.clip((feats[train_cols].astype(np.float64) - mu) / sd, -6, 6).mean(0).astype(np.float32)
    for name, cols in [('train', train_cols), ('validation', val_cols),
                       ('dormancy83', dorm_cols), ('external_candidates', external_cols)]:
        scaled = np.clip((feats[cols].astype(np.float64) - mu) / sd, -6, 6).astype(np.float32)
        assert np.isfinite(scaled).all()
        np.save(OUT / (name + '_features.npy'), scaled)
    np.save(OUT / 'training_block_channel_mean.npy', baseline_block)
    val_mask = np.random.default_rng(MASK_SEED).random((len(val_cols), nblocks)) < CONFIG['mask_ratio']
    assert (val_mask.sum(1) >= 8).all()
    np.save(OUT / 'validation_fixed_mask.npy', val_mask)
    save_npz(OUT / 'reference_transform.npz', minor_codes=minor_all, allele_means=means_all,
             pca_loadings=loadings_all, allele_counts_train=counts_all,
             feature_mu=mu, feature_sd=sd, **blockmap)
    save_npz(OUT / 'reference_graph.npz', **edges)
    required = ['identity_columns.npz', 'reference_transform.npz', 'reference_graph.npz',
                'train_features.npy', 'validation_features.npy', 'dormancy83_features.npy',
                'external_candidates_features.npy', 'training_block_channel_mean.npy',
                'validation_fixed_mask.npy']
    manifest = {**identity, 'locked_experiment': CONFIG,
                'fit_population': 'REFERENCE TRAIN ONLY: allele counts, minor alleles, allele means, 3 SNP-loading PCs, LD, flat channel mean/std. Reference validation only applies frozen transforms.',
                'block_map_source': DATA / 'block_snp_map.npz',
                'block_map_sha256': sha_file(DATA / 'block_snp_map.npz'),
                'genotype_source': DATA / 'genotypes_all_1307.npy',
                'n_blocks': nblocks, 'n_snps': len(codes),
                'monomorphic_or_uncalled_train_snps': int((minor_all == 255).sum()),
                'minor_rule': 'Least frequent positive-count A/C/G/T allele in reference train; monomorphic/all-uncalled SNPs are constant zero indicators. Missing or unseen non-minor alleles map to zero.',
                'pca_rule': '25x25-or-smaller covariance eigendecomposition, top3 descending; largest-absolute SNP loading positive. Eigenvalues <=1e-12 give zero PC. Apply (minor indicator - frozen allele mean) @ frozen loading.',
                'feature_flat_train_mu': mu, 'feature_flat_train_sd': sd,
                'clip': [-6, 6], 'validation_mask_sha256': digest(val_mask),
                'graph_rule': 'Coordinate adjacency + same-chromosome next50 candidate blocks, within1Mb, top3 absolute reference-train mean-dosage correlations >=0.5. Bidirectional/self-loop/virtual edges; legacy six-feature encoding.',
                'undirected_edges': edges['edge_index'].shape[1],
                'adjacency_edges': int((edges['edge_attr'][:, 2] == 0).sum()),
                'ld_edges': int((edges['edge_attr'][:, 2] == 1).sum()),
                'directed_edges_with_loops_virtual': len(edges['src']),
                'edge_index_sha256': digest(edges['edge_index']),
                'edge_attr_sha256': digest(edges['edge_attr']),
                'reference_transform_sha256': sha_file(OUT / 'reference_transform.npz'),
                'required_artifacts': required, 'preparation_seconds': time.time() - started}
    save_json(manifest_path, manifest)
    report['reference'] = manifest
    report['status'] = 'reference_prepared'
    save_json(REPORT, report)
    print(f"reference ready: train={len(train_cols)}, val={len(val_cols)}, excluded={identity['excluded_union_n']}, blocks={nblocks}, edges={len(edges['src'])}", flush=True)
    return manifest


def graph_device():
    graph = np.load(OUT / 'reference_graph.npz')
    return (torch.as_tensor(graph['src'], dtype=torch.long, device=DEVICE),
            torch.as_tensor(graph['dst'], dtype=torch.long, device=DEVICE),
            torch.as_tensor(graph['efeat'], dtype=torch.float32, device=DEVICE))


def new_model(width, seed):
    torch.manual_seed(seed)
    if DEVICE.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    return Encoder(in_dim=4, d=width, heads=4, n_layers=2, edge_dim=6, dropout=0.1).to(DEVICE)


def reconstruction_mse(model, x, fixed_mask, graph):
    model.eval()
    sse = np.zeros(4, dtype=np.float64)
    n = 0
    with torch.no_grad():
        for start in range(0, len(x), CONFIG['microbatch_size']):
            xb = torch.as_tensor(np.asarray(x[start:start + CONFIG['microbatch_size']]).copy(), device=DEVICE)
            mask = torch.as_tensor(np.asarray(fixed_mask[start:start + len(xb)]).copy(), device=DEVICE)
            pred = model(xb, mask, *graph)
            err = (pred[mask] - xb[mask]).to(torch.float64)
            sse += (err * err).sum(0).cpu().numpy()
            n += int(mask.sum().item())
    assert n > 0
    return {'overall_mse': float(sse.sum() / (n * 4)), 'channel_mse': (sse / n).tolist(),
            'channels': ['block_mean_minor_indicator', 'PC1', 'PC2', 'PC3'], 'masked_blocks': n}


def mean_baselines(train, val, mask):
    block_mean = np.load(OUT / 'training_block_channel_mean.npy')
    channel_mean = np.asarray(train, np.float64).mean((0, 1))
    n = int(mask.sum())
    out = {}
    for name, baseline in [('per_block_channel_training_mean', block_mean),
                           ('global_channel_training_mean', channel_mean)]:
        sse = np.zeros(4)
        for start in range(0, len(val), 16):
            x = np.asarray(val[start:start + 16], np.float64)
            m = mask[start:start + len(x)]
            err = (x - baseline)[m]
            sse += (err * err).sum(0)
        out[name] = {'overall_mse': float(sse.sum() / (n * 4)),
                     'channel_mse': (sse / n).tolist(), 'masked_blocks': n}
    return out


def train_epoch(model, x, optimizer, generator, graph):
    model.train()
    order = torch.randperm(len(x), generator=generator).numpy()
    sse, masked = 0., 0
    batch_size, micro = CONFIG['batch_size'], CONFIG['microbatch_size']
    for start in range(0, len(order), batch_size):
        ix = order[start:start + batch_size]
        mask_cpu = torch.rand((len(ix), x.shape[1]), generator=generator) < CONFIG['mask_ratio']
        assert torch.all(mask_cpu.sum(1) >= 8)
        count = int(mask_cpu.sum())
        optimizer.zero_grad(set_to_none=True)
        for offset in range(0, len(ix), micro):
            indexes = ix[offset:offset + micro]
            xb = torch.as_tensor(np.asarray(x[indexes]).copy(), device=DEVICE)
            mask = mask_cpu[offset:offset + len(indexes)].to(DEVICE)
            pred = model(xb, mask, *graph)
            loss_sum = F.mse_loss(pred[mask], xb[mask], reduction='sum')
            # Exact logical-batch normalization, not an unweighted microbatch mean.
            (loss_sum / (count * 4)).backward()
            sse += float(loss_sum.detach())
        nn.utils.clip_grad_norm_(model.parameters(), CONFIG['gradient_clip_norm'])
        optimizer.step()
        masked += count
    return sse / (masked * 4)


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def improvements(trained, baseline):
    return {'fraction_improvement': float(1 - trained['overall_mse'] / baseline['overall_mse']),
            'channel_fraction_improvement': [float(1 - a / b) if b > 0 else None
                                               for a, b in zip(trained['channel_mse'], baseline['channel_mse'])],
            'definition': '(baseline_MSE - trained_MSE) / baseline_MSE on the SAME validation accessions and fixed mask.'}


def train_one(width, seed, report, train, val, valmask, graph, baselines):
    tag = f'width{width}_seed{seed}'
    folder = OUT / tag
    folder.mkdir(exist_ok=True)
    summary_path = folder / 'training_summary.json'
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        assert summary['epochs_completed'] == CONFIG['epochs']
        report['runs'].setdefault(tag, {})['training'] = summary
        return summary
    model = new_model(width, seed)
    init_path, best_path, last_path = folder / 'untrained.pt', folder / 'trained_best.pt', folder / 'training_resume.pt'
    base_config = {**CONFIG, 'width': width, 'seed': seed}
    if not init_path.exists():
        save_torch(init_path, {'state_dict': cpu_state(model), 'config': base_config,
                              'reference_ids_sha256': report['reference']['reference_ids_sha256'],
                              'role': 'Same initialization from which this paired trained model starts.'})
    initial = read_torch(init_path)
    assert initial['config'] == base_config
    model.load_state_dict(initial['state_dict'])
    untrained = reconstruction_mse(model, val, valmask, graph)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])
    gen = torch.Generator().manual_seed(seed)
    logs, best, best_epoch, start_epoch = [], float('inf'), None, 1
    previous_elapsed = 0.
    if last_path.exists():
        resume = read_torch(last_path)
        assert resume['config'] == base_config
        model.load_state_dict(resume['state_dict'])
        optimizer.load_state_dict(resume['optimizer'])
        scheduler.load_state_dict(resume['scheduler'])
        gen.set_state(resume['generator_state'])
        torch.set_rng_state(resume['torch_rng_state'])
        if DEVICE.type == 'cuda':
            torch.cuda.set_rng_state_all(resume['cuda_rng_states'])
        logs, best, best_epoch = resume['logs'], resume['best_val'], resume['best_epoch']
        start_epoch = resume['epoch'] + 1
        previous_elapsed = logs[-1]['elapsed_s']
    if DEVICE.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    for epoch in range(start_epoch, CONFIG['epochs'] + 1):
        train_loss = train_epoch(model, train, optimizer, gen, graph)
        val_loss = reconstruction_mse(model, val, valmask, graph)
        scheduler.step()
        elapsed = previous_elapsed + time.time() - started
        row = {'epoch': epoch, 'train_masked_mse': train_loss,
               'validation_masked_mse': val_loss['overall_mse'],
               'validation_channel_mse': val_loss['channel_mse'],
               'lr_after_scheduler': optimizer.param_groups[0]['lr'], 'elapsed_s': elapsed}
        logs.append(row)
        if val_loss['overall_mse'] < best:
            best, best_epoch = val_loss['overall_mse'], epoch
            save_torch(best_path, {'state_dict': cpu_state(model), 'config': base_config,
                                  'epoch': epoch, 'validation_masked_mse': val_loss,
                                  'reference_ids_sha256': report['reference']['reference_ids_sha256'],
                                  'transform_filename': str(OUT / 'reference_transform.npz'),
                                  'graph_filename': str(OUT / 'reference_graph.npz')})
        save_torch(last_path, {'state_dict': cpu_state(model), 'config': base_config,
                              'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                              'generator_state': gen.get_state(), 'torch_rng_state': torch.get_rng_state(),
                              'cuda_rng_states': torch.cuda.get_rng_state_all() if DEVICE.type == 'cuda' else [],
                              'epoch': epoch, 'best_val': best, 'best_epoch': best_epoch, 'logs': logs})
        save_json(folder / 'training_log.json', logs)
        report['runs'].setdefault(tag, {})['progress'] = {'phase': 'reference_pretraining',
            'epoch_completed': epoch, 'epochs_total': CONFIG['epochs'], 'best_validation_masked_mse': best,
            'last_validation_masked_mse': val_loss['overall_mse'], 'elapsed_s': elapsed,
            'untrained_validation': untrained, 'training_mean_baselines': baselines}
        save_json(REPORT, report)
        print(f'{tag} epoch={epoch}/{CONFIG["epochs"]} train={train_loss:.6f} val={val_loss["overall_mse"]:.6f} elapsed={elapsed:.1f}s', flush=True)
    model.load_state_dict(read_torch(best_path)['state_dict'])
    trained = reconstruction_mse(model, val, valmask, graph)
    assert abs(trained['overall_mse'] - best) < 1e-5
    compare = {name: improvements(trained, value) for name, value in baselines.items()}
    compare['same_initialization_untrained'] = improvements(trained, untrained)
    summary = {'width': width, 'seed': seed, 'epochs_completed': len(logs),
               'best_validation_epoch': best_epoch, 'best_validation_masked_mse': best,
               'trained_validation': trained, 'untrained_validation': untrained,
               'training_mean_baselines': baselines, 'relative_improvement': compare,
               'validation_mask_sha256': report['reference']['validation_mask_sha256'],
               'same_validation_accessions_and_mask_for_all_conditions': True,
               'untrained_checkpoint': init_path, 'trained_checkpoint': best_path,
               'last_epoch_resume_checkpoint': last_path, 'training_log': folder / 'training_log.json',
               'runtime_seconds': logs[-1]['elapsed_s'],
               'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated() if DEVICE.type == 'cuda' else None,
               'gain_vs_training_block_mean': compare['per_block_channel_training_mean']['fraction_improvement'] > 0,
               'gain_vs_untrained': compare['same_initialization_untrained']['fraction_improvement'] > 0,
               'interpretation': 'A positive fraction is a reduction in actually measured validation masked MSE. Standardized variance 1 is NOT substituted for a validation baseline; zero/negative gains are retained.'}
    save_json(summary_path, summary)
    report['runs'][tag]['training'] = summary
    report['runs'][tag]['progress']['phase'] = 'pretraining_complete'
    save_json(REPORT, report)
    del model, optimizer, scheduler
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    return summary


def encode_one(width, seed, graph):
    folder = OUT / f'width{width}_seed{seed}'
    for role, checkpoint in [('untrained', 'untrained.pt'), ('trained', 'trained_best.pt')]:
        target = folder / f'{role}_embeddings83.npy'
        if target.exists():
            continue
        model = new_model(width, seed)
        model.load_state_dict(read_torch(folder / checkpoint)['state_dict'])
        model.eval()
        x = np.load(OUT / 'dormancy83_features.npy', mmap_mode='r')
        # No masking, gradients, refitting or downstream labels at encoding.
        nodes = np.lib.format.open_memmap(target.with_suffix('.tmp.npy'), mode='w+',
                                          dtype=np.float32, shape=(len(x), x.shape[1], width))
        with torch.no_grad():
            for start in range(0, len(x), CONFIG['microbatch_size']):
                xb = torch.as_tensor(np.asarray(x[start:start + CONFIG['microbatch_size']]).copy(), device=DEVICE)
                mask = torch.zeros(xb.shape[:2], dtype=torch.bool, device=DEVICE)
                _, h = model(xb, mask, *graph, return_nodes=True)
                nodes[start:start + len(xb)] = h.cpu().numpy()
        nodes.flush()
        del nodes, model
        target.with_suffix('.tmp.npy').replace(target)
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
        print(f'encoded frozen {folder.name} {role}: {(len(x), x.shape[1], width)}', flush=True)


class FrozenBlockKernel:
    """Cache only immutable raw products; each fold chooses its own block scales."""
    def __init__(self, embeddings):
        n, nb, d = embeddings.shape
        self.n = n
        # Float64 avoids cancellation when training traces are small.
        self.grams = torch.empty((nb, n, n), dtype=torch.float64, device=DEVICE)
        for start in range(0, nb, 128):
            h = torch.as_tensor(np.asarray(embeddings[:, start:start + 128]).transpose(1, 0, 2).copy(),
                                dtype=torch.float64, device=DEVICE)
            self.grams[start:start + len(h)] = h @ h.transpose(1, 2)
        self.flat = self.grams.reshape(nb, -1)
        self.diag = self.grams.diagonal(dim1=1, dim2=2)
        self.last_audit = {}

    def build(self, train, query):
        ix = torch.as_tensor(train, dtype=torch.long, device=DEVICE)
        gtt = self.grams.index_select(1, ix).index_select(2, ix)
        scales = self.diag.index_select(1, ix).mean(1) - gtt.mean((1, 2))
        valid = scales > 1e-12
        assert bool(valid.any())
        weights = torch.where(valid, 1 / scales.clamp(min=1e-12), 0.) / valid.sum()
        # Centering commutes with this weighted average. Scales ONLY use train.
        raw = (weights @ self.flat).reshape(self.n, self.n).cpu().numpy()
        kt, kc = core.centered_pair(raw[np.ix_(train, train)], raw[np.ix_(query, train)])
        assert abs(np.trace(kt) / len(train) - 1) < 1e-8
        assert abs(kt.mean()) < 1e-8
        self.last_audit = {'training_n': len(train), 'training_variable_blocks': int(valid.sum().item()),
                           'training_block_trace_sha256': digest(scales.cpu().numpy()),
                           'center_and_trace_fit': 'Each block train kernel centered and divided by trace/n using this fold TRAIN only, then equally averaged; cross kernel uses frozen train means/scales. Aggregate roundoff re-normalization also train-only.',
                           'precomputed_products': 'Raw frozen-encoder block dot products, float64; no panel standardization or label-dependent fitting.'}
        return kt, kc


def kernel_unit_tests(report):
    rng = np.random.default_rng(123)
    ref = rng.integers(0, 4, (40, 9), dtype=np.uint8)
    query = rng.integers(0, 4, (5, 9), dtype=np.uint8)
    m, mean, loading, counts = fit_block(ref)
    full = np.r_[ref, query]
    first = fit_block(full[:40])
    full[40:] = 255
    second = fit_block(full[:40])
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    assert np.allclose(apply_block(ref, m, mean, loading)[:, 1:].mean(0), 0, atol=1e-6)
    mono = np.full((40, 3), 255, np.uint8)
    mm, mu, ll, _ = fit_block(mono)
    assert not apply_block(mono, mm, mu, ll).any()
    h = rng.normal(size=(19, 11, 4)).astype(np.float32)
    tr, te = np.arange(14), np.arange(14, 19)
    engine = FrozenBlockKernel(h)
    kt, kc = engine.build(tr, te)
    direct_train, direct_cross = [], []
    for b in range(h.shape[1]):
        z = h[:, b].astype(np.float64) - h[tr, b].astype(np.float64).mean(0)
        scale = (z[tr] ** 2).sum() / len(tr)
        direct_train.append(z[tr] @ z[tr].T / scale)
        direct_cross.append(z[te] @ z[tr].T / scale)
    error = max(float(np.max(abs(kt - np.mean(direct_train, 0)))),
                float(np.max(abs(kc - np.mean(direct_cross, 0)))))
    assert error < 1e-10, error
    h2 = h.copy(); h2[te] *= 20
    other = FrozenBlockKernel(h2)
    kt2, _ = other.build(tr, te)
    assert np.max(abs(kt - kt2)) < 1e-12
    y = rng.normal(size=len(tr))
    augmented = np.block([[kt + .3 * np.eye(len(tr)), np.ones((len(tr), 1))],
                          [np.ones((1, len(tr))), np.zeros((1, 1))]])
    sol = np.linalg.solve(augmented, np.r_[y, 0])
    stable = core.eig_predictions(kt, kc, y, [.3])[:, 0]
    pred_error = float(np.max(abs(stable - (kc @ sol[:-1] + sol[-1]))))
    assert pred_error < 1e-10
    result = {'passed': True, 'reference_fit_unchanged_when_excluded_genotypes_mutated': True,
              'uncalled_monomorphic_indicators_zero': True,
              'strict_block_kernel_vs_direct_feature_fit_max_error': error,
              'heldout_genotype_perturbation_does_not_change_training_kernel': True,
              'core_centered_eigensolver_vs_augmented_intercept_max_prediction_error': pred_error}
    save_json(OUT / 'unit_tests.json', result)
    report['unit_tests'] = result
    save_json(REPORT, report)
    del engine, other
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()


def smoke(report):
    graph = graph_device()
    train = np.load(OUT / 'train_features.npy', mmap_mode='r')[:32]
    val = np.load(OUT / 'validation_features.npy', mmap_mode='r')[:8]
    mask = np.load(OUT / 'validation_fixed_mask.npy')[:8]
    records = []
    for width in WIDTHS:
        model = new_model(width, SEEDS[0])
        if DEVICE.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
        gen = torch.Generator().manual_seed(SEEDS[0])
        start = time.time()
        loss = train_epoch(model, train, optimizer, gen, graph)
        v = reconstruction_mse(model, val, mask, graph)
        assert np.isfinite(loss) and np.isfinite(v['overall_mse'])
        records.append({'width': width, 'logical_batch': 32, 'microbatch': CONFIG['microbatch_size'],
                        'one_batch_train_mse': loss, 'validation': v,
                        'seconds': time.time() - start,
                        'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated() if DEVICE.type == 'cuda' else None})
        print(f'smoke width={width}, seconds={records[-1]["seconds"]:.2f}, peak={records[-1]["peak_cuda_allocated_bytes"]}', flush=True)
        del model, optimizer
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
    result = {'passed': True, 'not_an_experiment_result': True, 'records': records}
    save_json(OUT / 'smoke_tests.json', result)
    report['smoke_tests'] = result
    report['status'] = 'smoke_passed'
    save_json(REPORT, report)


def nested_one(width, seed, role, y, ids, report):
    tag = f'width{width}_seed{seed}'
    folder = OUT / tag
    state_path = folder / f'{role}_nested_state.json'
    pred_path = folder / f'{role}_nested_predictions.npz'
    metrics_path = folder / f'{role}_nested_metrics.json'
    if metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding='utf-8'))
    embeddings = np.load(folder / f'{role}_embeddings83.npy', mmap_mode='r')
    engine = FrozenBlockKernel(embeddings)
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding='utf-8'))
        assert state['locked_experiment'] == CONFIG and state['ids_sha256'] == digest(ids)
        saved = np.load(pred_path)
        pred, mean_pred = saved['predictions'].copy(), saved['train_mean'].copy()
    else:
        state = {'role': role, 'width': width, 'seed': seed, 'locked_experiment': CONFIG,
                 'ids_sha256': digest(ids), 'folds': [],
                 'label_policy': 'Phenotypes used only for training-fold ridge fits/inner MSE; encoder and raw features frozen on independent reference beforehand.',
                 'strict_normalization': 'Per-block centering and trace normalization re-estimated separately in every inner/outer training split, then equally averaged. NO legacy all-83 graph_block_kernel scale.'}
        pred, mean_pred = np.full((10, len(y)), np.nan), np.full((10, len(y)), np.nan)
    done = {(f['repeat'], f['fold']) for f in state['folds']}
    started = time.time()
    for rep in range(10):
        for fold, (train, test) in enumerate(KFold(5, shuffle=True, random_state=rep).split(y)):
            if (rep, fold) in done:
                continue
            costs = np.zeros(len(core.LAMBDAS))
            inner_audits = []
            for it, iv in KFold(5, shuffle=True, random_state=1000 + rep * 5 + fold).split(train):
                a, v = train[it], train[iv]
                kt, kc = engine.build(a, v)
                p = core.eig_predictions(kt, kc, y[a])
                costs += ((p - y[v, None]) ** 2).sum(0)
                inner_audits.append({'train_indices': a, 'validation_indices': v, **engine.last_audit})
            ilam = int(costs.argmin())
            kt, kc = engine.build(train, test)
            p = core.eig_predictions(kt, kc, y[train], [core.LAMBDAS[ilam]])[:, 0]
            assert np.isfinite(p).all()
            pred[rep, test], mean_pred[rep, test] = p, y[train].mean()
            state['folds'].append({'repeat': rep, 'fold': fold,
                'train_indices': train, 'test_indices': test, 'train_ids': ids[train], 'test_ids': ids[test],
                'lambda': float(core.LAMBDAS[ilam]), 'inner_mse': float(costs[ilam] / len(train)),
                'all_lambda_inner_mse': costs / len(train),
                'test_metrics': core.metrics(y[test], p), 'outer_preprocessing': engine.last_audit,
                'inner_preprocessing': inner_audits})
            state['elapsed_current_run_seconds'] = time.time() - started
            save_npz(pred_path, y=y, ids=ids, predictions=pred, train_mean=mean_pred)
            save_json(state_path, state)
            report['runs'][tag]['progress'] = {'phase': f'{role}_strict_nested_cv',
                'outer_folds_completed': len(state['folds']), 'outer_folds_total': 50,
                'elapsed_current_run_seconds': time.time() - started}
            save_json(REPORT, report)
            print(f'{tag} {role} nested repeat={rep + 1}/10 fold={fold + 1}/5 lambda={core.LAMBDAS[ilam]:g}', flush=True)
    assert np.isfinite(pred).all() and np.isfinite(mean_pred).all()
    metrics = {'model': core.summarize(y, pred), 'train_mean': core.summarize(y, mean_pred),
               'prediction_filename': pred_path, 'fold_audit_filename': state_path,
               'embeddings_filename': folder / f'{role}_embeddings83.npy',
               'outer_folds_completed': len(state['folds'])}
    save_json(metrics_path, metrics)
    del engine
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    return metrics


def paired_downstream(width, seed):
    folder = OUT / f'width{width}_seed{seed}'
    trained = np.load(folder / 'trained_nested_predictions.npz')
    untrained = np.load(folder / 'untrained_nested_predictions.npz')
    y = trained['y']
    assert np.array_equal(y, untrained['y']) and np.array_equal(trained['ids'], untrained['ids'])
    a, b = trained['predictions'], untrained['predictions']
    ma = [core.metrics(y, p) for p in a]
    mb = [core.metrics(y, p) for p in b]
    delta = {}
    for key in ma[0]:
        values = [x[key] - z[key] for x, z in zip(ma, mb) if x[key] is not None and z[key] is not None]
        delta[key] = {'per_repeat_trained_minus_untrained': values,
                      'mean': float(np.mean(values)) if values else None,
                      'sd': float(np.std(values, ddof=1)) if len(values) > 1 else None}
    fraction = 1 - np.mean((a - y) ** 2, axis=1) / np.mean((b - y) ** 2, axis=1)
    return {'metrics_delta': delta, 'paired_per_repeat_MSE_fraction_improvement': fraction,
            'paired_MSE_fraction_improvement_mean': float(fraction.mean()),
            'gain_by_mean_repeat_MSE': bool(fraction.mean() > 0),
            'interpretation': 'Identical fixed reference, initialization pair and inner/outer fold IDs; each model selects only ridge lambda by its inner folds. Negative/zero gains are reported; no best width/seed is chosen.'}


def summarize_locked_runs(report):
    """Describe every locked condition; never select the best width or seed."""
    grouped, nongains = {}, []
    for width in WIDTHS:
        runs = [report['runs'][f'width{width}_seed{seed}'] for seed in SEEDS]
        reconstruction = [r['training']['relative_improvement']['per_block_channel_training_mean']['fraction_improvement'] for r in runs]
        downstream = [r['downstream']['paired_trained_vs_untrained']['paired_MSE_fraction_improvement_mean'] for r in runs]
        pearson_delta = [r['downstream']['paired_trained_vs_untrained']['metrics_delta']['pearson']['mean'] for r in runs]
        grouped[str(width)] = {
            'seeds': list(SEEDS), 'validation_MSE_fraction_gain_vs_block_training_mean_per_seed': reconstruction,
            'validation_MSE_fraction_gain_vs_block_training_mean_seed_average': float(np.mean(reconstruction)),
            'downstream_paired_repeat_MSE_fraction_gain_per_seed': downstream,
            'downstream_paired_repeat_MSE_fraction_gain_seed_average': float(np.mean(downstream)),
            'downstream_pearson_trained_minus_untrained_per_seed': pearson_delta,
            'trained_mean_repeat_pearson_seed_average': float(np.mean([r['downstream']['trained']['model']['pearson_mean'] for r in runs])),
            'untrained_mean_repeat_pearson_seed_average': float(np.mean([r['downstream']['untrained']['model']['pearson_mean'] for r in runs])),
            'trained_mean_repeat_RMSE_seed_average': float(np.mean([r['downstream']['trained']['model']['rmse_mean'] for r in runs])),
            'untrained_mean_repeat_RMSE_seed_average': float(np.mean([r['downstream']['untrained']['model']['rmse_mean'] for r in runs])),
            'interpretation': 'Descriptive average over all three locked seeds, not selection and not an independent-replication significance test.'}
        for seed, run, mse_gain, correlation_gain in zip(SEEDS, runs, downstream, pearson_delta):
            if mse_gain <= 0:
                nongains.append({'width': width, 'seed': seed, 'metric': 'paired repeat downstream MSE fraction gain', 'value': mse_gain})
            if correlation_gain is not None and correlation_gain <= 0:
                nongains.append({'width': width, 'seed': seed, 'metric': 'downstream mean-repeat Pearson trained minus untrained', 'value': correlation_gain})
            tr = run['training']
            for channel, fraction in zip(tr['trained_validation']['channels'], tr['relative_improvement']['per_block_channel_training_mean']['channel_fraction_improvement']):
                if fraction is not None and fraction <= 0:
                    nongains.append({'width': width, 'seed': seed, 'metric': 'validation channel fraction gain vs block training mean', 'channel': channel, 'value': fraction})
    report['all_locked_width_seed_summary'] = grouped
    report['zero_or_negative_gain_conditions'] = nongains
    report['overall_interpretation'] = 'Masked reconstruction improves against measured validation training-mean baselines, but downstream gains are small and metric/seed dependent; do not claim robust pretrained-encoder superiority. All locked conditions, including negative results, are retained.'
    report['supplementary_audits'] = {name: OUT / name for name in
        ('reference_projection_audit.json', 'graph_edge_audit.json', 'publishing_selftest.json', 'completion_audit.json') if (OUT / name).exists()}
    report['background_work_remaining'] = []


def run_full(report):
    train = np.load(OUT / 'train_features.npy', mmap_mode='r')
    val = np.load(OUT / 'validation_features.npy', mmap_mode='r')
    valmask = np.load(OUT / 'validation_fixed_mask.npy')
    baselines = mean_baselines(train, val, valmask)
    report['validation_training_mean_baselines'] = baselines
    report['status'] = 'reference_pretraining_running'
    save_json(REPORT, report)
    graph = graph_device()
    # All six prespecified pretrainings complete before downstream phenotypes are read.
    for width in WIDTHS:
        for seed in SEEDS:
            train_one(width, seed, report, train, val, valmask, graph, baselines)
            encode_one(width, seed, graph)
    del graph, train, val
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    report['status'] = 'all_six_pretrainings_complete_strict_nested_cv_running'
    save_json(REPORT, report)
    pheno = pd.read_csv(DATA / 'dormancy_phenotype.csv', usecols=['ecotype_id', 'dsds50'])
    ids, y = pheno.ecotype_id.to_numpy(np.int64), pheno.dsds50.to_numpy(float)
    columns = np.load(OUT / 'identity_columns.npz')
    assert np.array_equal(ids, columns['dormancy_ids']) and np.isfinite(y).all()
    for width in WIDTHS:
        for seed in SEEDS:
            tag = f'width{width}_seed{seed}'
            for role in ('trained', 'untrained'):
                metrics = nested_one(width, seed, role, y, ids, report)
                report['runs'][tag].setdefault('downstream', {})[role] = metrics
                save_json(REPORT, report)
            report['runs'][tag]['downstream']['paired_trained_vs_untrained'] = paired_downstream(width, seed)
            report['runs'][tag]['progress'] = {'phase': 'complete', 'paired_outer_folds': 50}
            save_json(REPORT, report)
    summarize_locked_runs(report)
    report['status'] = 'complete'
    report['completed_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
    report['completed_prespecified_pretrainings'] = 6
    report['completed_strict_nested_cv_models'] = 12
    report['completed_outer_fold_evaluations'] = 600
    report['encoder_readonly_source_unchanged'] = sha_file(OLD_ENCODER) == report['encoder_source_sha256']
    assert report['encoder_readonly_source_unchanged']
    save_json(REPORT, report)
    print('COMPLETE: six reference pretrainings, twelve strict 10x5 nested-CV models; no width/seed selected.', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['prepare', 'tests', 'smoke', 'run'], default='tests')
    parser.add_argument('--cuda-memory-fraction', type=float, default=0.45)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if DEVICE.type == 'cuda':
        assert 0 < args.cuda_memory_fraction <= 0.6
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction)
        torch.backends.cuda.matmul.allow_tf32 = False
    report = report_base()
    if 'failure' in report:
        report.setdefault('recovered_attempts', []).append(report.pop('failure'))
    report['new_training_file_sha256_at_latest_launch'] = sha_file(Path(__file__).resolve())
    report['runtime']['pipeline_process_id'] = os.getpid()
    report['runtime']['sympy_version'] = sys.modules['sympy'].__version__ if 'sympy' in sys.modules else None
    report['runtime']['sympy_file'] = sys.modules['sympy'].__file__ if 'sympy' in sys.modules else None
    report['runtime']['cuda_memory_fraction_limit'] = args.cuda_memory_fraction if DEVICE.type == 'cuda' else None
    lock = OUT / 'pipeline.lock'
    # Do not allow two writers. An interrupted run may be resumed after verifying its PID.
    fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode('ascii'))
    os.close(fd)
    try:
        with threadpool_limits(2):
            if args.mode == 'tests':
                kernel_unit_tests(report)
            else:
                prepare(report)
                kernel_unit_tests(report)
                if args.mode == 'smoke':
                    smoke(report)
                elif args.mode == 'run':
                    if not (OUT / 'smoke_tests.json').exists():
                        smoke(report)
                    run_full(report)
    except Exception as exc:
        report['status'] = 'failed'
        report['failure'] = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
        save_json(REPORT, report)
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
