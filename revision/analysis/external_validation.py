"""Canonical-ID external validation; only external_validation_* outputs are written.

Run: python -s -B external_validation.py --stage all --device cpu
Revalidate completed artifacts without reselection: --stage verify-existing
Resume refuses changed code/protocol; verify-existing preserves all choices.
The legacy identity repair preserves P2's original kernel units and lambda.
The inductive experiment selects three model families by pooled training-only
five-fold MSE, then refits on all 83 training accessions and freezes prediction.
It is NOT the core repeated nested-CV experiment. External outcomes are never
passed to selection or kernel fitting. All bootstrap intervals condition on the
frozen training set and model; they do not cover training or ancestry uncertainty.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import platform
import site
import sys
import time
import types
from collections import defaultdict
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from scipy import linalg, stats
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

# Execute one captured copy of the read-only API, not a file that can change
# midway through a concurrent core run. __file__ keeps its paths.
_API_PATH = Path(__file__).resolve().parent / 'kernel_benchmark.py'
_API_BYTES = _API_PATH.read_bytes()
core = types.ModuleType('external_validation_kernel_api')
core.__file__ = str(_API_PATH)
sys.modules[core.__name__] = core
exec(compile(_API_BYTES, str(_API_PATH), 'exec'), core.__dict__)

HERE = Path(__file__).resolve().parent
REVISION = HERE.parent
ROOT = REVISION.parent
DATA = ROOT / 'data' / 'processed'
OUT = REVISION / 'results'
HEADER = ROOT / 'data/raw/atpolydb/call_method_75/call_method_75_TAIR9.csv'
XLS = ROOT / 'data/external/arabidopsis_germ_phen_chilling_DATA.xls'
TSV = ROOT / 'data/raw/atpolydb/phenotype_published_raw.tsv'
SEED = int(core.PROTOCOL['random_seed'])
INNER_SEED = 1000
FAMILIES = ['snp', 'block_selected', 'local_selected']
MODEL_ORDER = ['train_mean', 'train_median', 'legacy_snp', 'legacy_block',
               'inductive_snp', 'inductive_block', 'inductive_local']
METRIC_NAMES = ['mae_days', 'rmse_days', 'predictive_r2', 'calibration_slope',
                'pearson_r', 'spearman_rho', 'bias_days']


def output(name):
    return OUT / ('external_validation_' + name)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def digest_array(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def clean_json(x):
    if x is pd.NA:
        return None
    if isinstance(x, dict):
        return {str(k): clean_json(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [clean_json(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean_json(x.tolist())
    if isinstance(x, np.generic):
        return clean_json(x.item())
    if isinstance(x, float) and not np.isfinite(x):
        return None
    if isinstance(x, Path):
        return x.as_posix()
    return x


def save_json(name, value):
    path = output(name)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(clean_json(value), indent=2, ensure_ascii=False,
                              allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def save_npz(name, **values):
    path = output(name)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **values)
    tmp.replace(path)


def norm_name(value):
    # Preserve the original alias convention; no speculative fuzzy matching.
    return str(value).strip().lower().replace(' ', '')


def load_xlrd():
    try:
        import xlrd
        return xlrd
    except ModuleNotFoundError:
        # -s remains active. Load only this existing pure-Python XLS reader,
        # not the whole user site (which would also change the torch build).
        path = Path(site.getusersitepackages()) / 'xlrd' / '__init__.py'
        if not path.is_file():
            raise RuntimeError('xlrd is required to verify the original Dryad XLS')
        spec = importlib.util.spec_from_file_location(
            'xlrd', path, submodule_search_locations=[str(path.parent)])
        module = importlib.util.module_from_spec(spec)
        sys.modules['xlrd'] = module
        spec.loader.exec_module(module)
        return module


def identity_tables(pheno, ids):
    panel = pd.read_csv(DATA / 'accessions_all.csv')
    with HEADER.open('r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        first, second = next(reader), next(reader)
    header_ids = np.asarray([int(v) for v in second[2:]], dtype=np.int64)
    cols83 = np.load(DATA / 'dormancy_cols_in_1307.npy')
    assert len(header_ids) == len(panel) == 1307
    assert len(first[2:]) == len(header_ids)
    assert len(np.unique(header_ids)) == len(header_ids)
    assert panel.ecotype_id.is_unique
    assert set(header_ids) == set(panel.ecotype_id)
    assert len(ids) == len(np.unique(ids)) == 83
    assert len(cols83) == len(np.unique(cols83)) == 83
    assert np.array_equal(header_ids[cols83], ids)
    assert np.array_equal(pheno.ecotype_id.to_numpy(), ids)
    rawcol = {int(e): int(c) for c, e in enumerate(header_ids)}
    train_ids = set(map(int, ids))
    panel_by_id = panel.set_index('ecotype_id')
    exact, normalized = defaultdict(set), defaultdict(set)
    for row in panel.itertuples():
        exact[str(row.nativename)].add(int(row.ecotype_id))
        normalized[norm_name(row.nativename)].add(int(row.ecotype_id))

    xlrd = load_xlrd()
    sheet = xlrd.open_workbook(str(XLS)).sheet_by_index(0)
    dryad_rows, invalid = [], []
    for r in range(1, sheet.nrows):
        name, value = str(sheet.cell_value(r, 2)).strip(), sheet.cell_value(r, 21)
        try:
            y = float(value)
            if not np.isfinite(y):
                raise ValueError('nonfinite DSDS50')
        except (ValueError, TypeError):
            invalid.append({'xls_row_1based': r + 1, 'nativename': name,
                            'value': str(value), 'reason': 'invalid_DSDS50'})
            continue
        exact_ids = sorted(exact.get(name, set()))
        norm_ids = sorted(normalized.get(norm_name(name), set()))
        # Conservative: a unique exact spelling cannot silently resolve a
        # conflicting normalized alias. Preserve all candidates in the report.
        candidates = sorted(set(exact_ids) | set(norm_ids))
        eid = candidates[0] if len(candidates) == 1 else None
        reason = ('no_unique_name_match' if not candidates else
                  'ambiguous_name_multiple_ecotype_ids' if len(candidates) > 1 else '')
        rule = 'exact' if exact_ids else ('normalized' if norm_ids else 'unmatched')
        dryad_rows.append({
            'cohort': 'dryad', 'source_row_1based': r + 1, 'nativename': name,
            'normalized_name': norm_name(name), 'dsds50_external': y,
            'match_rule': rule, 'exact_candidate_ids': json.dumps(exact_ids),
            'normalized_candidate_ids': json.dumps(norm_ids),
            'candidate_ids': json.dumps(candidates),
            'candidate_rawcols': json.dumps([rawcol[e] for e in candidates]),
            'candidate_training_flags': json.dumps([e in train_ids for e in candidates]),
            'canonical_ecotype_id': eid, 'rawcol': rawcol.get(eid),
            'canonical_name': str(panel_by_id.loc[eid].nativename) if eid is not None else '',
            'overlap_train83': eid in train_ids if eid is not None else None,
            'excluded_reason': reason, 'included': eid is not None})
    dryad = pd.DataFrame(dryad_rows)
    assert len(dryad) == 523, f'Expected 523 valid XLS rows, got {len(dryad)}'

    raw = pd.read_csv(TSV, sep='\t', encoding='latin-1', low_memory=False)
    ext = raw[raw['279_DSDS50'].notna() & raw['8_Seed Dormancy'].isna()].copy()
    assert len(ext) == 29
    ext_rows = []
    for index, row in ext.iterrows():
        eid = int(row['ecotype id'])
        y = float(row['279_DSDS50'])
        assert np.isfinite(y)
        ext_rows.append({
            'cohort': 'published29', 'source_row_1based': int(index) + 2,
            'nativename': str(row['nativename']), 'dsds50_external': y,
            'match_rule': 'canonical_ecotype_id_from_raw_TSV',
            'canonical_ecotype_id': eid, 'rawcol': rawcol.get(eid),
            'canonical_name': str(panel_by_id.loc[eid].nativename) if eid in rawcol else '',
            'overlap_train83': eid in train_ids,
            'excluded_reason': '' if eid in rawcol else 'ecotype_id_missing_from_raw_header',
            'included': eid in rawcol})
    external29 = pd.DataFrame(ext_rows)
    assert not external29.overlap_train83.any(), '29 must be independent of phenotype training'
    assert external29.included.all()
    duplicate_records = {}
    for label, frame in [('dryad', dryad), ('published29', external29)]:
        for col in ['canonical_ecotype_id', 'rawcol']:
            frame[col] = frame[col].astype('Int64')
        duplicate = frame.included & frame.canonical_ecotype_id.duplicated(keep=False)
        duplicate_records[label] = frame.loc[duplicate].to_dict('records')
        # No selecting one of repeated outcomes. Exclude all repeated IDs.
        frame.loc[duplicate, 'included'] = False
        frame.loc[duplicate, 'excluded_reason'] = 'duplicate_canonical_id_within_cohort'
        included = frame[frame.included]
        assert included.canonical_ecotype_id.is_unique and included.rawcol.is_unique
        assert np.array_equal(header_ids[included.rawcol.to_numpy(dtype=int)],
                              included.canonical_ecotype_id.to_numpy(dtype=int))
        assert np.array_equal(included.overlap_train83.to_numpy(dtype=bool),
                              included.canonical_ecotype_id.isin(train_ids).to_numpy())
        frame.to_csv(output(label + '_matching.csv'), index=False)
    assert int(external29.included.sum()) == 29

    included = dryad[dryad.included]
    overlap = included[included.overlap_train83.astype(bool)].copy()
    train_y = dict(zip(map(int, ids), pheno.dsds50.to_numpy(float)))
    overlap['dsds50_training_measurement'] = [train_y[int(e)] for e in overlap.canonical_ecotype_id]
    overlap.to_csv(output('overlap_measurements.csv'), index=False)
    intersect = sorted(set(included.canonical_ecotype_id.astype(int)) &
                       set(external29.canonical_ecotype_id.astype(int)))
    summary = {
        'identity_policy': 'Canonical ecotype ID -> second raw genotype header -> zero-based raw column; NEVER sorted metadata row position.',
        'name_policy': 'Exact and legacy-normalized names must have a single candidate ID in their union. All ambiguous names and all within-cohort repeated canonical IDs are excluded, without examining outcome values.',
        'train_n': 83, 'training_raw_header_alignment_passed': True,
        'raw_header_unique_ids': len(header_ids), 'raw_header_ids_sha256': digest_array(header_ids),
        'metadata_position_raw_position_mismatches': int(np.sum(panel.ecotype_id.to_numpy() != header_ids)),
        'dryad_valid_rows': len(dryad), 'dryad_invalid_rows': invalid,
        'dryad_included': len(included),
        'dryad_overlap_measurement_only': len(overlap),
        'dryad_independent': int((~included.overlap_train83.astype(bool)).sum()),
        'dryad_exclusion_counts': dryad.loc[~dryad.included, 'excluded_reason'].value_counts().to_dict(),
        'ambiguous_names': dryad[dryad.excluded_reason.eq('ambiguous_name_multiple_ecotype_ids')].to_dict('records'),
        'within_cohort_duplicate_ids': duplicate_records,
        'published29_rule': '279_DSDS50.notna() & 8_Seed Dormancy.isna(); raw TSV row order retained',
        'published29_n': len(external29), 'published29_training_overlap': 0,
        'dryad_published29_intersection_ids': intersect,
        'cohorts_not_pooled': 'Cross-cohort shared IDs are disclosed; do not treat the two evaluations as independent replications or pool their measurements.',
        'xlrd_path': str(xlrd.__file__),
        'phenotype_description': 'Dryad and published279 are distinct measurements/protocols from training trait 8; no outcome normalization or recalibration.',
        'old_P2_predictions': 'Invalid identity alignment (205/206 mismatches in the prior audit); no old external metric or prediction is reused.'}
    save_json('identity.json', summary)
    print('identity:', {k: summary[k] for k in ['dryad_valid_rows', 'dryad_included',
          'dryad_overlap_measurement_only', 'dryad_independent', 'published29_n']}, flush=True)
    return dryad, external29, overlap, cols83, summary


def metric_arrays(y, pred):
    """Vectorized statistics along the last axis, also used for bootstraps."""
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    yc = y - y.mean(axis=-1, keepdims=True)
    pc = pred - pred.mean(axis=-1, keepdims=True)
    yy = np.sum(yc * yc, axis=-1)
    pp = np.sum(pc * pc, axis=-1)
    dot = np.sum(yc * pc, axis=-1)
    sse = np.sum((pred - y) ** 2, axis=-1)
    varied = (yy > 1e-20) & (pp > 1e-20)
    yr = stats.rankdata(y, axis=-1)
    pr = stats.rankdata(pred, axis=-1)
    yr -= yr.mean(axis=-1, keepdims=True)
    pr -= pr.mean(axis=-1, keepdims=True)
    rrden = np.sqrt(np.sum(yr ** 2, axis=-1) * np.sum(pr ** 2, axis=-1))
    with np.errstate(divide='ignore', invalid='ignore'):
        values = {
            'mae_days': np.mean(abs(pred - y), axis=-1),
            'rmse_days': np.sqrt(np.mean((pred - y) ** 2, axis=-1)),
            'predictive_r2': np.where(yy > 1e-20, 1 - sse / yy, np.nan),
            'calibration_slope': np.where(varied, dot / pp, np.nan),
            'pearson_r': np.where(varied, dot / np.sqrt(yy * pp), np.nan),
            'spearman_rho': np.where(rrden > 0, np.sum(yr * pr, axis=-1) / rrden, np.nan),
            'bias_days': np.mean(pred - y, axis=-1)}
    return values


def bootstrap_report(y, predictions, draws, seed):
    """Identical paired accession draws for every model and every metric."""
    y = np.asarray(y, float)
    assert len(y) >= 2 and np.isfinite(y).all()
    sample = np.random.default_rng(seed).integers(0, len(y), size=(draws, len(y)))
    result = {'n': len(y), 'bootstrap': {
        'draws': draws, 'seed': seed, 'interval': 'two-sided 95% percentile',
        'unit': 'canonical accession within this cohort',
        'paired_across_models_and_metrics': True,
        'scope': 'Conditional on frozen training data and model; no training-refit, ancestry-cluster, or between-protocol uncertainty included.'}, 'models': {}}
    boots = {}
    for name, pred in predictions.items():
        pred = np.asarray(pred, float)
        assert pred.shape == y.shape and np.isfinite(pred).all()
        point = metric_arrays(y, pred)
        values = metric_arrays(y[sample], pred[sample])
        boots[name] = values
        result['models'][name] = {}
        for metric in METRIC_NAMES:
            valid = values[metric][np.isfinite(values[metric])]
            result['models'][name][metric] = {
                'estimate': float(point[metric]),
                'ci95': np.quantile(valid, [0.025, 0.975]).tolist() if len(valid) else [None, None],
                'valid_bootstrap_draws': len(valid)}
    result['prespecified_paired_differences'] = {}
    for a, b in [('inductive_block', 'inductive_snp'), ('inductive_local', 'inductive_block')]:
        if a not in predictions or b not in predictions:
            continue
        result['prespecified_paired_differences'][a + '_minus_' + b] = {}
        for metric in ['mae_days', 'rmse_days', 'predictive_r2', 'pearson_r', 'spearman_rho']:
            delta = boots[a][metric] - boots[b][metric]
            valid = delta[np.isfinite(delta)]
            point = (result['models'][a][metric]['estimate'] - result['models'][b][metric]['estimate'])
            result['prespecified_paired_differences'][a + '_minus_' + b][metric] = {
                'estimate': point,
                'ci95': np.quantile(valid, [0.025, 0.975]).tolist() if len(valid) else [None, None]}
    return clean_json(result)


def query_identity(dryad, external29, ids):
    joined = pd.concat([dryad[dryad.included], external29[external29.included]], ignore_index=True)
    query_ids = list(dict.fromkeys(map(int, joined.canonical_ecotype_id)))
    id_to_col = dict(zip(joined.canonical_ecotype_id.astype(int), joined.rawcol.astype(int)))
    # All queries, including overlaps, are duplicated after the training rows.
    # Only the first 83 rows can contribute to any fitting operation.
    full = np.load(DATA / 'genotypes_all_1307.npy', mmap_mode='r')
    query_codes = np.asarray(full[:, [id_to_col[e] for e in query_ids]]).T.copy()
    assert query_codes.shape == (len(query_ids), full.shape[0])
    return query_ids, query_codes


def legacy_predictions(codes, query_codes, y, cols83, query_ids, dryad, external29):
    meta = pd.read_csv(DATA / 'snp_metadata.csv')
    keep = np.flatnonzero((meta.maf_83.to_numpy() > 0.05) & (meta.minor_code.to_numpy() < 4))
    minor = meta.minor_code.to_numpy()[keep]
    xt = (codes[:, keep] == minor[None, :]).astype(float)
    xq = (query_codes[:, keep] == minor[None, :]).astype(float)
    full = np.load(DATA / 'block_features.npy', mmap_mode='r')
    bt = np.asarray(full[cols83, :, 0], dtype=float)
    assert np.array_equal(bt, np.asarray(np.load(DATA / 'block_features_83.npy')[:, :, 0], float))
    by_id = pd.concat([dryad[dryad.included], external29[external29.included]])
    rawcol = dict(zip(by_id.canonical_ecotype_id.astype(int), by_id.rawcol.astype(int)))
    bq = np.asarray(full[[rawcol[e] for e in query_ids], :, 0], dtype=float)
    predictions, report = {}, {}
    for name, train, query in [('legacy_snp', xt, xq), ('legacy_block', bt, bq)]:
        mean, sd = train.mean(0), train.std(0)
        sd = np.where(sd < 1e-8, 1., sd)
        zt, zq = (train - mean) / sd, (query - mean) / sd
        rawt, rawc = zt @ zt.T / zt.shape[1], zq @ zt.T / zt.shape[1]
        # P2 used feature-standardized K=zz.T/p, without extra trace scaling.
        mu, total = rawt.mean(0), float(rawt.mean())
        kt = rawt - mu[:, None] - mu[None, :] + total
        kc = rawc - rawc.mean(1)[:, None] - mu[None, :] + total
        predictions[name] = core.eig_predictions(kt, kc, y, [1e-8])[:, 0]
        report[name] = {
            'label': 'legacy transductive representation' if name == 'legacy_block' else 'legacy fixed 83-accession SNP filter/coding',
            'purpose': 'identity-correction audit only; not an external-selected model',
            'lambda': 1e-8, 'feature_count': train.shape[1],
            'kernel': 'training-standardized linear K=zz.T/p; original P2 units, NO extra trace normalization',
            'intercept': 'explicit unpenalized intercept via centered float64 eig_predictions; roundoff null modes discarded',
            'training_only_mean_std': True, 'canonical_rawcol_queries': True,
            'raw_trace_per_training_sample': float(np.trace(rawt) / len(y))}
    save_json('legacy_models.json', report)
    print('legacy canonical-ID projection completed', flush=True)
    return predictions


def inner_selection(codes, y, groups, ids, provenance):
    """There is intentionally no external-phenotype or external-genotype argument."""
    signature = {
        'training_ids_sha256': digest_array(ids), 'training_y_sha256': digest_array(y),
        'training_codes_sha256': digest_array(codes),
        'kernel_api_sha256': provenance['kernel_api_sha256'],
        'script_sha256': provenance['script_sha256'],
        'protocol_sha256': provenance['protocol_sha256'],
        'inner_seed': INNER_SEED, 'families': FAMILIES,
        'device': str(core.DEVICE)}
    state_path, checkpoint = output('selection_state.json'), output('selection_checkpoint.npz')
    if state_path.exists() and checkpoint.exists():
        state = json.loads(state_path.read_text(encoding='utf-8'))
        assert state['signature'] == signature, 'Refusing to resume selection with altered inputs/protocol/code'
        with np.load(checkpoint) as saved:
            losses, oof, baseline = saved['losses'], saved['oof'], saved['baseline']
        keys = [tuple(key) for key in state['candidate_keys']]
    else:
        state = {'signature': signature, 'status': 'training_inner5_in_progress', 'folds': [],
                 'selection_loss': 'pooled inner-validation MSE on training83 only',
                 'external_outcomes_used': False,
                 'not_core_nested_cv': True}
        keys, losses, oof = None, None, None
        baseline = np.full((len(y), 2), np.nan)
    done = {f['fold'] for f in state['folds']}
    engine = core.GenomicKernels(codes, groups)
    start = time.time()
    for fold, (train, validation) in enumerate(KFold(5, shuffle=True, random_state=INNER_SEED).split(y)):
        if fold in done:
            continue
        before = time.time()
        print(f'training-only inner fold {fold + 1}/5 building all five scales on {core.DEVICE}', flush=True)
        kernels = engine.build(train, validation, scales=core.SCALES, local=True, global_rbf=False)
        current = sorted(kernels)
        if keys is None:
            keys = current
            losses = np.zeros((len(keys), len(core.LAMBDAS)))
            oof = np.full((len(y), len(keys), len(core.LAMBDAS)), np.nan)
            state['candidate_keys'] = keys
        assert current == keys
        fold_losses = np.empty_like(losses)
        for i, key in enumerate(keys):
            kt, kc = kernels[key]
            pred = core.eig_predictions(kt, kc, y[train])
            assert np.isfinite(pred).all()
            fold_losses[i] = ((pred - y[validation, None]) ** 2).sum(0)
            oof[validation, i] = pred
        losses += fold_losses
        baseline[validation, 0] = y[train].mean()
        baseline[validation, 1] = np.median(y[train])
        state['folds'].append({
            'fold': fold, 'train_indices': train, 'validation_indices': validation,
            'train_ecotype_ids': ids[train], 'validation_ecotype_ids': ids[validation],
            'preprocessing': dict(engine.last_audit),
            'candidate_squared_error_sums': fold_losses,
            'elapsed_seconds': time.time() - before})
        state['elapsed_current_run_seconds'] = time.time() - start
        save_npz('selection_checkpoint.npz', losses=losses, oof=oof, baseline=baseline)
        save_json('selection_state.json', state)
        print(f'training-only inner fold {fold + 1}/5 completed in {time.time() - before:.1f}s', flush=True)
        del kernels
    assert len(state['folds']) == 5 and np.isfinite(oof).all() and np.isfinite(baseline).all()
    curves = []
    choices = {}
    for family in FAMILIES:
        candidates = []
        for i, key in enumerate(keys):
            if family not in core.candidate_families(key, 'strict'):
                continue
            for j, lam in enumerate(core.LAMBDAS):
                mse = float(losses[i, j] / len(y))
                curves.append({'family': family, 'kind': key[0], 'window_scale': key[1],
                               'gamma': key[2], 'lambda': float(lam), 'pooled_inner_mse': mse,
                               'pooled_inner_rmse': float(np.sqrt(mse)), 'validation_n': len(y)})
                candidates.append((mse, key, j, i))
        mse, key, j, i = min(candidates, key=lambda row: (row[0], row[1], row[2]))
        choices[family] = {'kind': key[0], 'window_scale': key[1], 'gamma': key[2],
                           'lambda': float(core.LAMBDAS[j]), 'inner_mse': mse,
                           'candidate_index': i, 'lambda_index': j}
    pd.DataFrame(curves).to_csv(output('selection_curves.csv'), index=False)
    fold_curves = []
    for fold in state['folds']:
        for i, key in enumerate(keys):
            for j, lam in enumerate(core.LAMBDAS):
                fold_curves.append({'fold': fold['fold'], 'kind': key[0], 'window_scale': key[1],
                                    'gamma': key[2], 'lambda': float(lam),
                                    'validation_n': len(fold['validation_indices']),
                                    'validation_sse': float(fold['candidate_squared_error_sums'][i][j])})
    pd.DataFrame(fold_curves).to_csv(output('selection_fold_curves.csv'), index=False)
    selection = {
        'status': 'completed_training_only_inner5', 'selection': choices,
        'selection_basis': 'pooled MSE across five validation folds of the original 83 only',
        'inner_folds': 5, 'shuffle': True, 'inner_seed': INNER_SEED,
        'tie_break': 'MSE, then candidate tuple(kind,scale,gamma), then lambda-grid index',
        'protocol': core.PROTOCOL, 'signature': signature,
        'training_mean_days': float(y.mean()), 'training_median_days': float(np.median(y)),
        'no_external_outcomes_or_genotypes_in_selection': True,
        'candidate_count': len(keys),
        'selection_oof_note': 'These OOF predictions are model-selection curves, not unbiased performance of the selected model and not completed core outer CV.'}
    state['status'] = selection['status']
    save_json('selection_state.json', state)
    # This file is committed to disk before any final external prediction.
    save_json('selection.json', selection)
    print('frozen training-only selections:', choices, flush=True)
    return selection


def selected_pairs(engine, train, query, selected):
    """API-equivalent final fit, computing only selected train/cross kernels.

    No query-query kernel and no unselected scale/gamma are constructed.
    Every fitted statistic is saved from the training rows only.
    """
    torch = core.torch
    x, indexes, n = engine.indicators(train, query)
    keys = sorted(set(selected))
    pairs, arrays, diagnostics = {}, {}, {}
    counts = torch.stack([(engine.codes[torch.as_tensor(train, dtype=torch.long, device=core.DEVICE)] == c).sum(0)
                          for c in range(4)])
    masked = counts.to(torch.float64).masked_fill(counts == 0, float('inf'))
    minor = masked.argmin(0)[indexes]
    arrays['retained_marker_indices'] = indexes.cpu().numpy()
    arrays['minor_allele_codes'] = minor.cpu().numpy()

    def standardize(a, label):
        mean = a[:n].mean(0)
        sd = a[:n].std(0, unbiased=False)
        valid = sd > 1e-10
        assert valid.any(), 'No variable training features'
        arrays[label + '_mean'] = mean.cpu().numpy()
        arrays[label + '_sd'] = sd.cpu().numpy()
        arrays[label + '_valid'] = valid.cpu().numpy()
        return (a[:, valid] - mean[valid]) / sd[valid]

    def store(key, a, c):
        rawt, rawc = a.cpu().numpy(), c.cpu().numpy()
        kt, kc = core.centered_pair(rawt, rawc)
        assert np.isfinite(kt).all() and np.isfinite(kc).all()
        assert abs(kt.mean()) < 1e-10
        assert abs(np.trace(kt) / n - 1.) < 1e-9
        pairs[key] = kt, kc
        label = f'{key[0]}_{key[1]}_{key[2]}'
        arrays[label + '_raw_train_kernel_mean'] = rawt.mean(0)
        scale = float(np.trace(rawt - rawt.mean(0)[:, None] - rawt.mean(0)[None, :] + rawt.mean()) / n)
        diagnostics[label] = {'centered_train_trace_scale': scale,
                              'raw_train_grand_mean': float(rawt.mean())}

    if any(k[0] == 'snp' for k in keys):
        z = standardize(x, 'snp')
        a, c = z[:n], z[n:]
        store(('snp', 0, 0.0), a @ a.T / z.shape[1], c @ a.T / z.shape[1])
        del z, a, c
    scales = sorted({k[1] for k in keys if k[0] in ['block', 'local']})
    for scale in scales:
        gid = engine.groups[scale][indexes]
        unique, inverse, sizes = torch.unique(gid, sorted=True, return_inverse=True, return_counts=True)
        b = torch.zeros((len(x), len(unique)), dtype=torch.float64, device=core.DEVICE)
        b.scatter_add_(1, inverse[None, :].expand(len(x), -1), x)
        b /= sizes[None, :]
        label = 'window_' + str(scale)
        arrays[label + '_group_ids'] = unique.cpu().numpy()
        arrays[label + '_retained_marker_counts'] = sizes.cpu().numpy()
        bz = standardize(b, label)
        engine.last_audit[label + '_variable_features'] = bz.shape[1]
        a, c = bz[:n], bz[n:]
        if ('block', scale, 0.0) in keys:
            store(('block', scale, 0.0), a @ a.T / bz.shape[1], c @ a.T / bz.shape[1])
        for key in [k for k in keys if k[0] == 'local' and k[1] == scale]:
            gamma = key[2]
            rawt = torch.zeros((n, n), dtype=torch.float64, device=core.DEVICE)
            rawc = torch.zeros((len(query), n), dtype=torch.float64, device=core.DEVICE)
            for start in range(0, bz.shape[1], 256):
                at, cq = a[:, start:start + 256], c[:, start:start + 256]
                rawt += torch.exp(-gamma * (at[:, None, :] - at[None, :, :]) ** 2).sum(2)
                rawc += torch.exp(-gamma * (cq[:, None, :] - at[None, :, :]) ** 2).sum(2)
            store(key, rawt / bz.shape[1], rawc / bz.shape[1])
        del b, bz, a, c
    assert set(pairs) == set(keys)
    return pairs, arrays, {'preprocessing': dict(engine.last_audit), 'kernels': diagnostics,
                           'only_selected_candidates': keys, 'query_query_kernel_computed': False}


def selftests():
    rng = np.random.default_rng(SEED)
    codes = rng.integers(0, 2, size=(19, 60), dtype=np.uint8)
    groups = {s: (np.arange(60) // s).astype(np.int32) for s in core.SCALES}
    train, query = np.arange(14), np.arange(14, 19)
    keys = [('snp', 0, 0.0), ('block', 25, 0.0), ('local', 25, 0.2)]
    engine = core.GenomicKernels(codes, groups)
    reference = engine.build(train, query, scales=[25], local=True, global_rbf=False)
    pairs, arrays, audit = selected_pairs(engine, train, query, keys)
    errors = {}
    for key in keys:
        errors[str(key)] = max(float(np.max(abs(pairs[key][i] - reference[key][i]))) for i in [0, 1])
        assert errors[str(key)] < 1e-10
    changed = codes.copy()
    changed[query] = rng.integers(0, 4, size=(len(query), 60), dtype=np.uint8)
    other, other_arrays, _ = selected_pairs(core.GenomicKernels(changed, groups), train, query, keys)
    train_error = max(float(np.max(abs(pairs[k][0] - other[k][0]))) for k in keys)
    assert train_error < 1e-12
    for label in arrays:
        assert np.array_equal(arrays[label], other_arrays[label]), label
    small, _, _ = selected_pairs(engine, train, query[:2], keys)
    batch_error = max(float(np.max(abs(pairs[k][1][:2] - small[k][1]))) for k in keys)
    assert batch_error < 1e-10
    y = rng.normal(size=len(train)) * 10 + 30
    kt, kc = pairs[keys[0]]
    direct = linalg.solve(np.block([[kt + .3 * np.eye(len(y)), np.ones((len(y), 1))],
                                   [np.ones((1, len(y))), np.zeros((1, 1))]]), np.r_[y, 0.])
    solver_error = float(np.max(abs(core.eig_predictions(kt, kc, y, [.3])[:, 0] -
                                    (kc @ direct[:-1] + direct[-1]))))
    assert solver_error < 1e-9
    manual = metric_arrays(np.arange(5.), np.array([1., 0., 2., 4., 3.]))
    assert abs(manual['pearson_r'] - stats.pearsonr(np.arange(5.), [1., 0., 2., 4., 3.])[0]) < 1e-12
    assert abs(manual['spearman_rho'] - stats.spearmanr(np.arange(5.), [1., 0., 2., 4., 3.])[0]) < 1e-12
    return {'passed': True, 'selected_pair_vs_core_build_max_errors': errors,
            'query_mutation_train_kernel_max_error': train_error,
            'query_mutation_all_fitted_preprocessing_identical': True,
            'query_batch_size_cross_kernel_max_error': batch_error,
            'explicit_intercept_augmented_solver_max_error_days': solver_error,
            'metric_vs_scipy_passed': True, 'final_fit_optimization': audit}


def attach_predictions(frame, query_ids, predictions, y):
    included = frame[frame.included].copy()
    lookup = {e: i for i, e in enumerate(query_ids)}
    positions = np.asarray([lookup[int(e)] for e in included.canonical_ecotype_id], dtype=int)
    included['train_mean'] = float(y.mean())
    included['train_median'] = float(np.median(y))
    for name in MODEL_ORDER:
        if name in predictions:
            included[name] = predictions[name][positions]
    return included


def evaluate(dryad, external29, overlap, query_ids, predictions, y, draws, stage):
    dp = attach_predictions(dryad, query_ids, predictions, y)
    ep = attach_predictions(external29, query_ids, predictions, y)
    dp.to_csv(output('dryad_' + stage + '_predictions.csv'), index=False)
    ep.to_csv(output('published29_' + stage + '_predictions.csv'), index=False)
    datasets = {
        'dryad_independent': dp[~dp.overlap_train83.astype(bool)],
        'published29_independent': ep,
        'dryad_overlap_prediction_measurement_only': dp[dp.overlap_train83.astype(bool)]}
    report = {'stage': stage, 'model_order_prespecified_not_sorted_by_external_results': MODEL_ORDER,
              'training_mean_days': float(y.mean()), 'training_median_days': float(np.median(y)),
              'cohort_results': {},
              'metric_definitions': {'predictive_r2': '1 - sum((observed-predicted)^2)/sum((observed-observed_cohort_mean)^2)',
                                     'calibration_slope': 'OLS observed on predicted with intercept; undefined for constant predictions',
                                     'bias_days': 'mean(predicted-observed)'},
              'no_external_y_selection_or_recalibration': True,
              'overlap_policy': 'Overlaps are measurement agreement only, never independent external performance; the prior same38 mean is not used as an independent-set result or baseline.'}
    for offset, (label, sub) in enumerate(datasets.items()):
        models = {name: sub[name].to_numpy(float) for name in MODEL_ORDER if name in sub.columns}
        report['cohort_results'][label] = bootstrap_report(sub.dsds50_external.to_numpy(float), models, draws, SEED + offset)
    local = overlap.dsds50_training_measurement.to_numpy(float)
    measurement = bootstrap_report(overlap.dsds50_external.to_numpy(float),
                                   {'training_measurement': local}, draws, SEED + 2)
    measurement['interpretation'] = 'Paired measurements of the same canonical accessions; NOT independent predictive validation.'
    measurement['training_measurement_mean_days'] = float(local.mean())
    measurement['dryad_measurement_mean_days'] = float(overlap.dsds50_external.mean())
    report['overlap_measurement_agreement'] = measurement
    save_json(stage + '_metrics.json', report)
    for label in ['dryad_independent', 'published29_independent']:
        models = report['cohort_results'][label]['models']
        print(label, {name: {m: models[name][m]['estimate'] for m in ['pearson_r', 'mae_days', 'rmse_days', 'predictive_r2']}
                      for name in models}, flush=True)
    return report


def verify_existing(draws):
    """Close out already computed results after concurrent on-disk API drift.

    Frozen selection/curves/predictions are NOT changed. New API equivalence,
    actual 83-training preprocessing, every saved prediction, and common CIs
    must pass. A changed protocol or data remains a hard error.
    """
    start = time.time()
    status = json.loads(output('status.json').read_text(encoding='utf-8'))
    selection = json.loads(output('selection.json').read_text(encoding='utf-8'))
    state = json.loads(output('selection_state.json').read_text(encoding='utf-8'))
    assert selection['status'] == 'completed_training_only_inner5'
    assert state['status'] == selection['status'] and len(state['folds']) == 5
    assert selection['signature'] == state['signature']
    original = status['provenance']
    assert sha256(HERE / 'protocol.json') == original['protocol_sha256']
    for path, checksum in original['input_hashes'].items():
        assert sha256(ROOT / path if not Path(path).is_absolute() else path) == checksum, 'Original data changed: ' + path
    names = ['selection.json', 'selection_curves.csv', 'selection_fold_curves.csv',
             'selection_checkpoint.npz', 'frozen_preprocessing.npz', 'frozen_fit.json',
             'dryad_final_predictions.csv', 'published29_final_predictions.csv', 'final_metrics.json']
    before = {name: sha256(output(name)) for name in names}
    tests = selftests()
    save_json('revalidation_selftests.json', tests)
    save_json('kernel_api_snapshot.json', {
        'path': _API_PATH, 'sha256': hashlib.sha256(_API_BYTES).hexdigest(),
        'source': _API_BYTES.decode('utf-8'), 'purpose': 'Exact API source captured for revalidation; first numerical run recorded its API SHA but did not archive source.'})
    codes, y, meta, ids, groups = core.load_dormancy()
    for label, value in [('training_ids', ids), ('training_y', y), ('training_codes', codes)]:
        assert digest_array(value) == selection['signature'][label + '_sha256']
    fold_checks = []
    validation_seen = []
    for fold in state['folds']:
        tr, va = np.asarray(fold['train_indices'], int), np.asarray(fold['validation_indices'], int)
        assert not set(tr) & set(va)
        assert sorted(list(tr) + list(va)) == list(range(83))
        validation_seen.extend(va.tolist())
        counts = np.stack([(codes[tr] == c).sum(0) for c in range(4)])
        keep = np.flatnonzero(((counts > 0).sum(0) == 2) &
                             ((1 - counts.max(0) / np.maximum(counts.sum(0), 1)) > .05)).astype(np.int64)
        minor = np.where(counts > 0, counts, np.inf).argmin(0)[keep].astype(np.int64)
        audit = fold['preprocessing']
        assert len(keep) == audit['retained_markers']
        assert digest_array(keep) == audit['marker_index_sha256']
        assert digest_array(minor) == audit['minor_code_sha256']
        fold_checks.append({'fold': fold['fold'], 'training_n': len(tr),
                            'validation_n': len(va), 'retained_markers': len(keep),
                            'original_elapsed_seconds': fold['elapsed_seconds']})
    assert sorted(validation_seen) == list(range(83))
    curves = pd.read_csv(output('selection_curves.csv'))
    folds = pd.read_csv(output('selection_fold_curves.csv'))
    assert len(curves) == 561 and len(folds) == 2805
    assert set(curves.window_scale) == {0, 5, 10, 25, 50, 100}
    with np.load(output('selection_checkpoint.npz')) as saved:
        assert np.isfinite(saved['oof']).all() and np.isfinite(saved['baseline']).all()
        np.testing.assert_allclose(saved['losses'], np.sum(
            [np.asarray(f['candidate_squared_error_sums']) for f in state['folds']], axis=0), rtol=1e-13)
    # Independently recover the exact train-only minima, without any external y.
    for family in FAMILIES:
        sub = curves[curves.family.eq(family)].sort_values(
            ['pooled_inner_mse', 'kind', 'window_scale', 'gamma', 'lambda'], kind='stable')
        best, frozen = sub.iloc[0], selection['selection'][family]
        assert best.kind == frozen['kind']
        assert int(best.window_scale) == frozen['window_scale']
        assert float(best.gamma) == frozen['gamma']
        assert float(best['lambda']) == frozen['lambda']
        assert abs(float(best.pooled_inner_mse) - frozen['inner_mse']) < 1e-9
    dryad = pd.read_csv(output('dryad_matching.csv'))
    ext = pd.read_csv(output('published29_matching.csv'))
    overlap = pd.read_csv(output('overlap_measurements.csv'))
    assert len(dryad) == 523 and int(dryad.included.sum()) == 198
    assert len(ext) == int(ext.included.sum()) == 29 and not ext.overlap_train83.any()
    for frame in [dryad[dryad.included], ext]:
        assert frame.canonical_ecotype_id.is_unique and frame.rawcol.is_unique
        assert np.array_equal(frame.overlap_train83.astype(bool), frame.canonical_ecotype_id.isin(ids))
    query_ids, query_codes = query_identity(dryad, ext, ids)
    selected = [(r['kind'], r['window_scale'], r['gamma']) for r in selection['selection'].values()]
    engine = core.GenomicKernels(np.concatenate([codes, query_codes]), groups)
    tr, qu = np.arange(83), np.arange(83, 83 + len(query_ids))
    pairs, arrays, audit = selected_pairs(engine, tr, qu, selected)
    with np.load(output('frozen_preprocessing.npz')) as saved:
        assert set(saved.files) == set(arrays)
        for name in arrays:
            np.testing.assert_allclose(arrays[name], saved[name], rtol=1e-11, atol=1e-12)
    # The updated expm1 implementation differs by a constant raw kernel only;
    # validate the centered/trace-normalized kernels on actual training83.
    reference = engine.build(tr, qu[:3], scales=sorted({k[1] for k in selected if k[1]}),
                             local=True, global_rbf=False)
    api_errors = {}
    for key in selected:
        error = max(float(np.max(abs(pairs[key][0] - reference[key][0]))),
                    float(np.max(abs(pairs[key][1][:3] - reference[key][1]))))
        assert error < 1e-10, (key, error)
        api_errors[str(key)] = error
    predicted = {}
    for family, name in [('snp', 'inductive_snp'), ('block_selected', 'inductive_block'),
                         ('local_selected', 'inductive_local')]:
        r = selection['selection'][family]
        predicted[name] = core.eig_predictions(*pairs[(r['kind'], r['window_scale'], r['gamma'])],
                                               y, [r['lambda']])[:, 0]
    lookup = {eid: i for i, eid in enumerate(query_ids)}
    dp = pd.read_csv(output('dryad_final_predictions.csv'))
    ep = pd.read_csv(output('published29_final_predictions.csv'))
    prediction_errors = {}
    for label, frame in [('dryad', dp), ('published29', ep)]:
        assert len(frame) == (198 if label == 'dryad' else 29)
        assert frame.canonical_ecotype_id.is_unique
        positions = [lookup[int(e)] for e in frame.canonical_ecotype_id]
        np.testing.assert_allclose(frame.train_mean, y.mean(), rtol=0, atol=1e-12)
        np.testing.assert_allclose(frame.train_median, np.median(y), rtol=0, atol=1e-12)
        for name in predicted:
            error = float(np.max(abs(frame[name].to_numpy(float) - predicted[name][positions])))
            assert error < 1e-7, (label, name, error)
            prediction_errors[label + '_' + name] = error
    report = json.loads(output('final_metrics.json').read_text(encoding='utf-8'))
    assert len(overlap) == 36
    expected_overlap_y = dict(zip(map(int, ids), y))
    np.testing.assert_array_equal(overlap.dsds50_training_measurement,
                                  [expected_overlap_y[int(e)] for e in overlap.canonical_ecotype_id])
    metric_checks = 0
    for offset, (label, sub) in enumerate([
            ('dryad_independent', dp[~dp.overlap_train83]), ('published29_independent', ep),
            ('dryad_overlap_prediction_measurement_only', dp[dp.overlap_train83])]):
        target = sub.dsds50_external.to_numpy(float)
        models = {name: sub[name].to_numpy(float) for name in MODEL_ORDER}
        previous = report['cohort_results'][label]
        assert previous['bootstrap']['draws'] == draws
        reproduced = bootstrap_report(target, models, draws, SEED + offset)
        for name in models:
            for metric in METRIC_NAMES:
                a, b = reproduced['models'][name][metric], previous['models'][name][metric]
                if a['estimate'] is None:
                    assert b['estimate'] is None and b['ci95'] == [None, None]
                else:
                    np.testing.assert_allclose(a['estimate'], b['estimate'], rtol=1e-10, atol=1e-8)
                    np.testing.assert_allclose(a['ci95'], b['ci95'], rtol=1e-10, atol=1e-8)
                assert a['valid_bootstrap_draws'] == b['valid_bootstrap_draws']
                metric_checks += 1
    current_sha = hashlib.sha256(_API_BYTES).hexdigest()
    source_audit = {
        'original_consumed_api_sha256': original['kernel_api_sha256'],
        'revalidation_consumed_api_sha256': current_sha,
        'current_on_disk_api_sha256': sha256(_API_PATH),
        'source_changed_since_original_import': current_sha != original['kernel_api_sha256'],
        'explanation': 'The original process had a fixed imported API in memory and completed all numerical outputs, then its final on-disk hash assertion detected concurrent source changes. No core/protocol file was edited or restored by this script. Saved training choices and predictions have not been replaced.',
        'protocol_unchanged_verified': True,
        'actual_training83_updated_api_equivalence_max_errors': api_errors}
    verification = {'status': 'passed', 'source_audit': source_audit,
                    'original_training_inner5_checks': fold_checks,
                    'train_only_selection_minima_and_all_curves_verified': True,
                    'frozen_training_preprocessing_reproduced': True,
                    'all_actual_external_inductive_prediction_max_errors_days': prediction_errors,
                    'point_metric_and_common_95CI_checks': metric_checks,
                    'original_frozen_artifact_sha256': before,
                    'elapsed_revalidation_seconds': time.time() - start,
                    'selection_or_prediction_files_changed': False,
                    'new_script_sha256': sha256(__file__), 'selftests': tests}
    assert before == {name: sha256(output(name)) for name in names}
    save_json('revalidation.json', verification)
    status['inductive_status'] = 'completed_training_inner5_and_frozen_external_prediction_verified'
    status['original_run_completion_guard'] = 'Post-computation on-disk API hash assertion failed; numerical outputs subsequently independently verified without changing them.'
    status['source_audit'] = source_audit
    status['final_metrics_path'] = output('final_metrics.json')
    status['revalidation_path'] = output('revalidation.json')
    status['verified_completed'] = time.strftime('%Y-%m-%d %H:%M:%S')
    status['elapsed_original_numerical_seconds'] = output('final_metrics.json').stat().st_mtime - time.mktime(
        time.strptime(original['started'], '%Y-%m-%d %H:%M:%S'))
    status['elapsed_revalidation_seconds'] = time.time() - start
    save_json('status.json', status)
    print('COMPLETED AND VERIFIED: original training-inner5 + frozen external predictions; '
          'source drift disclosed, saved choices/predictions unchanged, not core outer CV', flush=True)
    print('actual external prediction max errors:', prediction_errors, flush=True)
    print('point metric/95CI checks:', metric_checks, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['all', 'identity-legacy', 'inductive', 'tests', 'verify-existing'], default='all')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--bootstrap', type=int, default=2000)
    args = parser.parse_args()
    assert OUT.is_dir() and HERE.is_dir()
    assert args.threads > 0 and args.bootstrap >= 100
    if args.device == 'cuda' and not core.torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable in this -s interpreter; use CPU or an explicitly provisioned interpreter')
    core.DEVICE = core.torch.device(args.device)
    core.torch.set_num_threads(args.threads)
    if args.stage == 'verify-existing':
        with threadpool_limits(args.threads):
            verify_existing(args.bootstrap)
        return
    save_json('kernel_api_snapshot.json', {
        'path': _API_PATH, 'sha256': hashlib.sha256(_API_BYTES).hexdigest(),
        'source': _API_BYTES.decode('utf-8'), 'purpose': 'Exact read-only API source captured before execution.'})
    provenance = {
        'started': time.strftime('%Y-%m-%d %H:%M:%S'), 'python_executable': sys.executable,
        'python_version': sys.version, 'platform': platform.platform(),
        'no_user_site_flag': sys.flags.no_user_site, 'no_bytecode_flag': sys.dont_write_bytecode,
        'device': str(core.DEVICE), 'threads': args.threads, 'torch_version': core.torch.__version__,
        'numpy_version': np.__version__, 'pandas_version': pd.__version__,
        'script_path': Path(__file__).resolve(), 'script_sha256': sha256(__file__),
        'kernel_api_path': HERE / 'kernel_benchmark.py', 'kernel_api_sha256': hashlib.sha256(_API_BYTES).hexdigest(),
        'protocol_sha256': sha256(HERE / 'protocol.json'),
        'input_hashes': {p.as_posix(): sha256(p) for p in [
            XLS, TSV, DATA / 'accessions_all.csv', DATA / 'dormancy_phenotype.csv',
            DATA / 'dormancy_cols_in_1307.npy', DATA / 'snp_metadata.csv', DATA / 'block_snp_map.npz']},
        'write_scope': 'Only this script and results/external_validation_*; core/protocol and original inputs are read-only.',
        'core_nested_cv_status': 'Not executed or claimed complete by this external-validation script.'}
    start = time.time()
    with threadpool_limits(args.threads):
        tests = selftests()
        save_json('selftests.json', tests)
        print('external validation selftests passed', flush=True)
        if args.stage == 'tests':
            return
        codes, y, meta, ids, groups = core.load_dormancy()
        pheno = pd.read_csv(DATA / 'dormancy_phenotype.csv')
        dryad, external29, overlap, cols83, identity = identity_tables(pheno, ids)
        query_ids, query_codes = query_identity(dryad, external29, ids)
        predictions = legacy_predictions(codes, query_codes, y, cols83, query_ids, dryad, external29)
        evaluate(dryad, external29, overlap, query_ids, predictions, y, args.bootstrap, 'legacy')
        status = {'provenance': provenance, 'identity': identity,
                  'identity_legacy_status': 'completed', 'inductive_status': 'not_started',
                  'selftests': tests, 'core_nested_cv_status': provenance['core_nested_cv_status']}
        save_json('status.json', status)
        if args.stage == 'identity-legacy':
            status['elapsed_seconds'] = time.time() - start
            save_json('status.json', status)
            return
        status['inductive_status'] = 'training_only_inner5_in_progress'
        save_json('status.json', status)
        selection = inner_selection(codes, y, groups, ids, provenance)
        status['inductive_status'] = 'selection_completed_final_projection_in_progress'
        status['frozen_selection'] = selection['selection']
        save_json('status.json', status)
        engine = core.GenomicKernels(np.concatenate([codes, query_codes]), groups)
        selected = [(row['kind'], row['window_scale'], row['gamma'])
                    for row in selection['selection'].values()]
        train, query = np.arange(len(y)), np.arange(len(y), len(y) + len(query_ids))
        pairs, fitted_arrays, fit_audit = selected_pairs(engine, train, query, selected)
        for family, name in [('snp', 'inductive_snp'), ('block_selected', 'inductive_block'),
                             ('local_selected', 'inductive_local')]:
            row = selection['selection'][family]
            key = row['kind'], row['window_scale'], row['gamma']
            kt, kc = pairs[key]
            predictions[name] = core.eig_predictions(kt, kc, y, [row['lambda']])[:, 0]
        save_npz('frozen_preprocessing.npz', **fitted_arrays)
        fit_audit['training_ids'] = ids
        fit_audit['query_ids'] = query_ids
        fit_audit['frozen_selection'] = selection['selection']
        fit_audit['label'] = 'inductive representation fitted using training83 only'
        fit_audit['external_outcomes_used_for_fit'] = False
        save_json('frozen_fit.json', fit_audit)
        report = evaluate(dryad, external29, overlap, query_ids, predictions, y, args.bootstrap, 'final')
        status['inductive_status'] = 'completed_training_inner5_and_frozen_external_prediction'
        status['final_metrics_path'] = output('final_metrics.json')
        status['elapsed_seconds'] = time.time() - start
        assert sha256(HERE / 'protocol.json') == provenance['protocol_sha256']
        observed_api_sha = sha256(HERE / 'kernel_benchmark.py')
        status['source_audit'] = {
            'consumed_snapshot_api_sha256': provenance['kernel_api_sha256'],
            'final_on_disk_api_sha256': observed_api_sha,
            'concurrent_on_disk_api_change': observed_api_sha != provenance['kernel_api_sha256'],
            'protocol_unchanged_verified': True,
            'note': 'All numerical functions used one captured API source; on-disk concurrent edits were not reloaded. This script never edits core/protocol.'}
        save_json('status.json', status)
        print(f'COMPLETED external validation on {core.DEVICE} in {time.time() - start:.1f}s; not core outer CV', flush=True)


if __name__ == '__main__':
    main()
