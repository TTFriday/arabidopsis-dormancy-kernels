"""Equal-weight regional SNP-RKHS; not a reproduction of MKLMM/MultiBLUP.

Within each reference-coordinate window w, retain the complete vector of SNPs
standardized using the current training split, and compute
  R_w(i,j) = exp[-gamma * sum_{s in w}(z_is-z_js)^2 / p_w].
Average RAW R_w equally over nonempty retained-marker windows, then use the
core's training-only HKH and trace(K)/n normalization and intercept solver.
No window-mean dosage, learned region weights, global RBF, or block candidates
are used. Akdemir & Jannink (2015), DOI 10.1534/genetics.114.173658, is a
regional/local-interaction precedent, not a claim of exact method replication.
All writes are confined to this module's regional_snp_rkhs_* result files.
"""
from __future__ import annotations

# Import core FIRST: it loads the compatible scientific stack before CUDA torch.
import kernel_benchmark as core

import argparse
import copy
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

np = core.np
torch = core.torch
linalg = core.linalg
TAG = 'regional_snp_rkhs'
RESULTS = core.RESULTS


def digest_array(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def method_definition():
    return {
        'name': 'equal-weight regional SNP-RKHS',
        'family_aliases': {'local_selected': 'regional_snp_vector_rkhs_selected',
                           'snp': 'training_aligned_snp_linear_companion'},
        'standardization': 'Reuse core.indicators and core.standardized: training-only biallelic MAF>0.05, training minor-code convention, population SD, same standardized retained SNPs as SNP linear. Query genotypes never estimate preprocessing.',
        'window_kernel_formula': 'R_w(i,j;gamma) = exp(-gamma * sum_{s in retained(w)} (z_is-z_js)^2 / p_w), p_w=number of training-retained, nonconstant SNPs in w',
        'raw_aggregation_formula': 'R(i,j;scale,gamma) = (1/B) * sum_{w:p_w>0} R_w(i,j;gamma)',
        'normalization_formula': 'K_train = H R_train H / [trace(H R_train H)/n]; K_query uses only training column means, training grand mean and training trace scale',
        'normalization_order': 'Average RAW RBF kernels across windows BEFORE a single training-only centering/trace normalization; do not normalize individual windows.',
        'window_weighting': 'Equal weight per nonempty training-retained window, irrespective of SNP count. Windows with no retained SNPs are excluded; windows are NOT filtered by mean-dosage variability.',
        'window_boundaries': 'Reuse core outcome-independent reference coordinates: subdivide base at-most-25-SNP/50kb windows for 5/10, retain for 25, merge two/four adjacent same-chromosome base windows for 50/100. Scale is not a guaranteed physical width or post-MAF retained count.',
        'difference_from_core_local': 'Core local kernel acts on standardized window-mean dosage; this module uses full standardized within-window SNP vectors and genuine nonlinear RBFs for all nine gamma values.',
        'selection': 'Joint scale x gamma x lambda selection by pooled 5-inner-fold validation MSE inside each of 10 x 5 outer splits, with identical core seeds and penalty grid.',
        'reference': {'authors': 'Akdemir & Jannink', 'year': 2015,
                      'doi': '10.1534/genetics.114.173658',
                      'role': 'Regional GRM/local-interaction precedent only.'},
        'scope_limits': [
            'A transparent equal-weight regional SNP-RKHS baseline, not exact MKLMM, MultiBLUP, or Akdemir & Jannink reproduction.',
            'No estimated region weights, sparsity, variance-component learning, or separate chromosome kernels; no cross-window interactions are explicitly modeled.',
            'Within-window SNP information is preserved, but predictive performance alone does not establish biological epistasis or causal interactions.',
            'Repeat variability is split variability, not ten independent biological replications; no independent significance claim is made.',
            'This task evaluates the existing dormancy training cohort only, not external cohorts; all completed results are retained irrespective of direction.',
            'Float32 distances/RBF exponentials with float64 accumulation can produce roundoff-level deviations from exact PSD; the unchanged core solver thresholds eigenvalues. Float64 oracle and real-data tests quantify this.'
        ],
    }


def regional_protocol():
    protocol = copy.deepcopy(core.PROTOCOL)
    protocol['window_rule'] = method_definition()['window_boundaries']
    protocol['strict_preprocessing'] = method_definition()['standardization']
    protocol['primary_comparisons'] = ['equal-weight regional SNP-RKHS versus identical SNP linear companion']
    protocol['additional_comparisons'] = 'Descriptive comparison with saved core window-mean local Gaussian, without rerunning or selecting that baseline.'
    protocol['kernel_definition'] = method_definition()
    return protocol


class RegionalSNPRKHS(core.GenomicKernels):
    """Only SNP-linear and ('local',scale,gamma) vector-RBF candidates."""

    def __init__(self, codes, groups, window_batch=512, local_dtype=torch.float32):
        super().__init__(codes, groups, local_dtype=local_dtype)
        if window_batch < 1:
            raise ValueError('window_batch must be positive')
        self.window_batch = int(window_batch)

    @torch.no_grad()
    def regional_raw(self, z, gid):
        """Pack complete SNP vectors; allocate gamma x batch x n x n, not p."""
        unique, inverse, counts = torch.unique(gid, sorted=True, return_inverse=True, return_counts=True)
        if len(unique) == 0:
            raise ValueError('No nonempty retained-marker windows')
        order = torch.argsort(inverse, stable=True)
        sorted_inverse = inverse[order]
        starts = counts.cumsum(0) - counts
        offsets = torch.arange(len(gid), device=core.DEVICE) - starts[sorted_inverse]
        max_count = int(counts.max().item())
        slots = sorted_inverse * max_count + offsets
        rows, windows = len(z), len(unique)
        packed = torch.zeros((rows, windows * max_count), dtype=self.local_dtype, device=core.DEVICE)
        packed.scatter_(1, slots[None, :].expand(rows, -1), z[:, order].to(self.local_dtype))
        packed = packed.view(rows, windows, max_count).permute(1, 0, 2)
        accum = torch.zeros((len(core.GAMMAS), rows, rows), dtype=torch.float64, device=core.DEVICE)
        gamma = torch.as_tensor(core.GAMMAS, dtype=self.local_dtype, device=core.DEVICE)
        for start in range(0, windows, self.window_batch):
            part = packed[start:start + self.window_batch].contiguous()
            gram = torch.bmm(part, part.transpose(1, 2))
            diagonal = gram.diagonal(dim1=1, dim2=2)
            distance = diagonal[:, :, None] + diagonal[:, None, :] - 2 * gram
            distance.clamp_(min=0)
            distance.diagonal(dim1=1, dim2=2).zero_()
            distance /= counts[start:start + len(part)].to(self.local_dtype)[:, None, None]
            values = -gamma[:, None, None, None] * distance[None, :, :, :]
            values.exp_()  # Actual RAW RBF, including its unit diagonal.
            accum += values.sum(1, dtype=torch.float64)
        accum /= windows
        audit = {
            'nonempty_retained_windows': windows,
            'retained_snp_count_min': int(counts.min().item()),
            'retained_snp_count_max': max_count,
            'retained_snp_count_sum': int(counts.sum().item()),
            'window_id_sha256': digest_array(unique.cpu().numpy()),
            'window_retained_count_sha256': digest_array(counts.cpu().numpy()),
        }
        return accum, audit

    @torch.no_grad()
    def build(self, train, query, scales=None, local=True, global_rbf=False):
        if global_rbf:
            raise ValueError('This baseline does not offer global RBF candidates')
        scales = core.SCALES if scales is None else list(scales)
        x, indexes, n = self.indicators(train, query)
        z, valid = self.standardized(x, n)
        indexes = indexes[valid]
        if z.shape[1] == 0:
            raise ValueError('No training-standardized retained SNPs')
        # Identical float64 SNP product and normalization to the core baseline.
        raw = z @ z.T / z.shape[1]
        out = {('snp', 0, 0.0): self.pair(raw, n)}
        self.last_audit.update({
            'standardized_retained_markers': int(z.shape[1]),
            'standardized_marker_index_sha256': digest_array(indexes.cpu().numpy()),
            'kernel_family': 'equal-weight regional SNP-RKHS (full SNP vectors)',
            'rbf_dtype': str(self.local_dtype), 'accumulation_dtype': 'torch.float64',
            'window_batch': self.window_batch, 'query_estimates_statistics': False,
        })
        if local:
            for scale in scales:
                raws, audit = self.regional_raw(z, self.groups[scale][indexes])
                for i, gamma in enumerate(core.GAMMAS):
                    out[('local', scale, gamma)] = self.pair(raws[i], n)
                self.last_audit['window_' + str(scale)] = audit
        return out


def oracle_raw(z, gid):
    groups = [np.flatnonzero(gid == g) for g in np.unique(gid)]
    out = np.zeros((len(core.GAMMAS), len(z), len(z)))
    for columns in groups:
        distance = ((z[:, None, columns] - z[None, :, columns]) ** 2).sum(2) / len(columns)
        for i, gamma in enumerate(core.GAMMAS):
            out[i] += np.exp(-gamma * distance)
    return out / len(groups)


def check_kernels(kernels, y, psd_tolerance):
    minimum_eigenvalue, solver_error = 0., 0.
    for key, (kt, cross) in kernels.items():
        assert np.isfinite(kt).all() and np.isfinite(cross).all(), key
        assert np.max(abs(kt - kt.T)) < 1e-12, key
        assert abs(kt.mean()) < 1e-12, key
        assert abs(np.trace(kt) / len(kt) - 1) < 1e-10, key
        smallest = float(linalg.eigvalsh(kt)[0])
        minimum_eigenvalue = min(minimum_eigenvalue, smallest)
        assert smallest > -psd_tolerance, (key, smallest)
        predictions = core.eig_predictions(kt, cross, y)
        assert predictions.shape == (len(cross), len(core.LAMBDAS))
        assert np.isfinite(predictions).all(), key
        # Direct KKT is reliable at these penalties; tiny penalties are tested
        # for finiteness, not compared with a numerically unstable inverse.
        for penalty in (0.001, 0.1, 1., 10.):
            system = np.block([[kt + penalty * np.eye(len(kt)), np.ones((len(kt), 1))],
                               [np.ones((1, len(kt))), np.zeros((1, 1))]])
            coef = linalg.solve(system, np.r_[y, 0.], assume_a='sym')
            direct = cross @ coef[:-1] + coef[-1]
            stable = core.eig_predictions(kt, cross, y, [penalty])[:, 0]
            solver_error = max(solver_error, float(np.max(abs(direct - stable))))
    return minimum_eigenvalue, solver_error


def selftests(codes, y, groups, window_batch):
    started = time.time()
    rng = np.random.default_rng(20261118)
    toy = rng.integers(0, 2, size=(24, 29), dtype=np.uint8)
    toy[:, 0] = 0                       # Constant excluded.
    toy[:18, 1] = 0; toy[18:, 1] = 1   # Query-only polymorphism excluded.
    toy[:18, 2] = np.arange(18) % 3     # Triallelic excluded.
    toy[0, 3] = 4                      # Missing code follows core convention.
    train, query = np.arange(18), np.arange(18, 24)
    toy_groups = {s: ((np.arange(29) * 7) % 29 // min(s, 29)).astype(np.int64)
                  for s in core.SCALES}
    mutant = toy.copy(); mutant[query] = 3

    # FIRST: test the existing core's genotype isolation, including its local
    # mean-dose kernels, without calling core.selftest (which writes audit).
    baseline = core.GenomicKernels(toy, toy_groups, local_dtype=torch.float32)
    altered = core.GenomicKernels(mutant, toy_groups, local_dtype=torch.float32)
    original = baseline.build(train, query, global_rbf=True)
    original_audit = copy.deepcopy(baseline.last_audit)
    changed = altered.build(train, query, global_rbf=True)
    assert baseline.last_audit == altered.last_audit
    core_isolation = max(float(np.max(abs(original[k][0] - changed[k][0]))) for k in original)
    assert core_isolation < 1e-12, core_isolation
    x, indexes, n = baseline.indicators(train, query)
    z, valid = baseline.standardized(x, n)
    changed_x, changed_indexes, _ = altered.indicators(train, query)
    changed_z, changed_valid = altered.standardized(changed_x, n)
    assert torch.equal(indexes, changed_indexes) and torch.equal(valid, changed_valid)
    assert torch.equal(z[:n], changed_z[:n])
    counts = np.stack([(toy[train] == c).sum(0) for c in range(4)])
    expected_keep = ((counts > 0).sum(0) == 2) & ((1 - counts.max(0) / counts.sum(0).clip(1)) > 0.05)
    assert np.array_equal(indexes.cpu().numpy(), np.flatnonzero(expected_keep))
    assert not expected_keep[:3].any()
    print('core genotype-isolation test passed', flush=True)

    precise = RegionalSNPRKHS(toy, toy_groups, window_batch=window_batch, local_dtype=torch.float64)
    mixed = RegionalSNPRKHS(toy, toy_groups, window_batch=window_batch)
    altered_regional = RegionalSNPRKHS(mutant, toy_groups, window_batch=window_batch)
    full64, full32 = precise.build(train, query), mixed.build(train, query)
    mutated32 = altered_regional.build(train, query)
    assert mixed.last_audit == altered_regional.last_audit
    regional_isolation = max(float(np.max(abs(full32[k][0] - mutated32[k][0]))) for k in full32)
    assert regional_isolation < 1e-12, regional_isolation
    oracle_error64 = oracle_error32 = raw_error64 = batch_error = 0.
    standardized = z.cpu().numpy()
    retained = indexes[valid].cpu().numpy()
    for scale in core.SCALES:
        gid = toy_groups[scale][retained]
        oracle = oracle_raw(standardized, gid)
        raw64, _ = precise.regional_raw(z, precise.groups[scale][indexes[valid]])
        raw_error64 = max(raw_error64, float(np.max(abs(raw64.cpu().numpy() - oracle))))
        one_at_a_time = RegionalSNPRKHS(toy, toy_groups, window_batch=1, local_dtype=torch.float64)
        raw1, _ = one_at_a_time.regional_raw(z, precise.groups[scale][indexes[valid]])
        batch_error = max(batch_error, float((raw64 - raw1).abs().max().item()))
        for i, gamma in enumerate(core.GAMMAS):
            pair = core.centered_pair(oracle[i, :n, :n], oracle[i, n:, :n])
            key = ('local', scale, gamma)
            oracle_error64 = max(oracle_error64, *(float(np.max(abs(a - b))) for a, b in zip(full64[key], pair)))
            oracle_error32 = max(oracle_error32, *(float(np.max(abs(a - b))) for a, b in zip(full32[key], pair)))
    assert raw_error64 < 1e-13 and oracle_error64 < 1e-9 and batch_error < 1e-13
    assert oracle_error32 < 1e-4, oracle_error32
    toy_y = rng.normal(size=len(train))
    eig64, solver64 = check_kernels(full64, toy_y, 1e-9)
    eig32, _ = check_kernels(full32, toy_y, 1e-4)
    assert solver64 < 1e-8, solver64
    # One-window Gaussian limit, using the full vector and the 1/p factor.
    vector = rng.normal(size=(23, 7))
    vector_tensor = torch.as_tensor(vector, dtype=torch.float64, device=core.DEVICE)
    one_gid = torch.zeros(7, dtype=torch.long, device=core.DEVICE)
    single, _ = precise.regional_raw(vector_tensor, one_gid)
    one_window_error = float(np.max(abs(single.cpu().numpy() - oracle_raw(vector, np.zeros(7)))))
    assert one_window_error < 1e-13, one_window_error
    same_mean = torch.tensor([[1., -1.], [-1., 1.], [1., -1.]], dtype=torch.float64, device=core.DEVICE)
    nonlinear, _ = precise.regional_raw(same_mean, torch.zeros(2, dtype=torch.long, device=core.DEVICE))
    gamma_one = list(core.GAMMAS).index(1)
    vector_similarity = float(nonlinear[gamma_one, 0, 1].item())
    assert abs(vector_similarity - np.exp(-4)) < 1e-14
    assert nonlinear[gamma_one, 0, 2] == 1

    # Actual cohort: same outer split as core, full scales/gammas; no phenotype
    # enters preprocessing or construction. Mutate only held-out genotypes.
    real_train, real_query = next(core.KFold(5, shuffle=True, random_state=0).split(y))
    real = RegionalSNPRKHS(codes, groups, window_batch=window_batch)
    real_kernels = real.build(real_train, real_query)
    real_audit = copy.deepcopy(real.last_audit)
    core_real = core.GenomicKernels(codes, groups)
    core_snp = core_real.build(real_train, real_query, scales=[], local=False)
    snp_alignment = max(float(np.max(abs(a - b))) for a, b in zip(real_kernels[('snp', 0, 0.0)], core_snp[('snp', 0, 0.0)]))
    assert snp_alignment == 0, snp_alignment
    real_mutant = codes.copy(); real_mutant[real_query] = 3
    mutated_real = RegionalSNPRKHS(real_mutant, groups, window_batch=window_batch)
    mutated_kernels = mutated_real.build(real_train, real_query)
    assert real_audit == mutated_real.last_audit
    real_isolation = max(float(np.max(abs(real_kernels[k][0] - mutated_kernels[k][0]))) for k in real_kernels)
    assert real_isolation < 1e-12, real_isolation
    real64 = RegionalSNPRKHS(codes, groups, window_batch=window_batch, local_dtype=torch.float64)
    real_kernels64 = real64.build(real_train, real_query)
    real_precision = max(float(np.max(abs(a - b))) for k in real_kernels for a, b in zip(real_kernels[k], real_kernels64[k]))
    assert real_precision < 1e-5, real_precision
    real_min_eig, real_solver = check_kernels(real_kernels, y[real_train], 1e-8)
    assert real_solver < 1e-6, real_solver
    report = {
        'passed': True, 'device': str(core.DEVICE), 'tested_candidates_per_build': len(real_kernels),
        'core_genotype_isolation_training_kernel_max_error': core_isolation,
        'core_training_maf_minor_code_standardization_matches_direct_counts': True,
        'core_query_mutation_fitted_preprocessing_identical': True,
        'toy_core_preprocessing': original_audit,
        'regional_query_mutation_training_kernel_max_error': regional_isolation,
        'float64_raw_rbf_oracle_max_error': raw_error64,
        'float64_normalized_oracle_max_error': oracle_error64,
        'float32_normalized_oracle_max_error': oracle_error32,
        'float64_batch_partition_max_error': batch_error,
        'one_window_full_vector_gaussian_limit_raw_max_error': one_window_error,
        'same_window_mean_different_vector_similarity_gamma1': vector_similarity,
        'same_window_mean_core_scalar_rbf_similarity': 1.,
        'toy_float64_minimum_centered_eigenvalue': eig64,
        'toy_float32_minimum_centered_eigenvalue': eig32,
        'toy_float64_augmented_solver_max_prediction_error': solver64,
        'real_snp_kernel_core_alignment_max_error': snp_alignment,
        'real_query_mutation_training_kernel_max_error': real_isolation,
        'real_float32_vs_float64_normalized_kernel_max_error': real_precision,
        'real_float32_minimum_centered_eigenvalue': real_min_eig,
        'real_augmented_solver_max_prediction_error_days': real_solver,
        'full_11_lambda_grid_including_1e12_negative_power_finite': True,
        'direct_solver_comparison_lambdas': [0.001, 0.1, 1., 10.],
        'real_outer_split_preprocessing': real_audit,
        'elapsed_seconds': time.time() - started,
        'method_definition': method_definition(),
    }
    core.save_json(RESULTS / (TAG + '_selftests.json'), report)
    print('regional SNP-RKHS tests passed: ' + json.dumps({k: report[k] for k in (
        'float64_raw_rbf_oracle_max_error', 'one_window_full_vector_gaussian_limit_raw_max_error',
        'real_float32_vs_float64_normalized_kernel_max_error', 'real_augmented_solver_max_prediction_error_days')}), flush=True)
    return report


def protected_snapshot():
    paths = [Path(core.__file__), core.HERE / 'protocol.json']
    paths += [RESULTS / name for name in ('strict_state.json', 'strict_predictions.npz',
              'strict_mixed_precision_state.json', 'strict_mixed_precision_predictions.npz',
              'strict_mixed_precision_metrics.json')]
    return {str(p.resolve()): digest_file(p) for p in paths if p.exists()}


def run_cv(codes, y, ids, groups, tests, window_batch):
    protected = protected_snapshot()
    provenance = {
        'implementation_sha256': digest_file(__file__), 'core_sha256': digest_file(core.__file__),
        'codes_sha256': digest_array(codes), 'phenotype_sha256': digest_array(y),
        'accession_ids_sha256': digest_array(ids),
        'group_sha256': {str(s): digest_array(groups[s]) for s in core.SCALES},
        'python_executable': sys.executable, 'python': sys.version, 'numpy': np.__version__,
        'torch': torch.__version__, 'device': str(core.DEVICE),
        'gpu': torch.cuda.get_device_name(core.DEVICE) if core.DEVICE.type == 'cuda' else None,
        'rbf_dtype': 'torch.float32', 'accumulation_dtype': 'torch.float64',
        'window_batch': window_batch, 'tf32_enabled': torch.backends.cuda.matmul.allow_tf32,
        'protected_files_sha256': protected,
    }
    path = RESULTS / (TAG + '_state.json')
    if path.exists():
        state = json.loads(path.read_text(encoding='utf-8'))
        for name in ('implementation_sha256', 'core_sha256', 'codes_sha256', 'phenotype_sha256', 'accession_ids_sha256', 'group_sha256', 'rbf_dtype', 'window_batch'):
            if state.get('provenance', {}).get(name) != provenance[name]:
                raise RuntimeError('Refusing incompatible regional checkpoint: ' + name)
    else:
        state = {'mode': 'strict', 'repeats': 10, 'folds': [], 'device': str(core.DEVICE),
                 'protocol': regional_protocol(), 'method_definition': method_definition(),
                 'provenance': provenance, 'started': time.strftime('%Y-%m-%d %H:%M:%S')}
        core.save_json(path, state)
    engine = RegionalSNPRKHS(codes, groups, window_batch=window_batch)
    with core.threadpool_limits(2):
        predictions, report = core.nested_cv(engine, y, mode='strict', repeats=10,
                                            scales=core.SCALES, local=True, global_rbf=False, tag=TAG)
    assert set(predictions) == {'snp', 'local_selected', 'train_mean', 'train_median'}
    state = json.loads(path.read_text(encoding='utf-8'))
    assert len(state['folds']) == 50
    assert {(f['repeat'], f['fold']) for f in state['folds']} == {(r, f) for r in range(10) for f in range(5)}
    assert all(set(f['selection']) == {'snp', 'local_selected'} for f in state['folds'])
    # Validate against SAVED primary strict SNP predictions, not rounded r only.
    strict_report = json.loads((RESULTS / 'strict_mixed_precision_metrics.json').read_text(encoding='utf-8'))
    with np.load(RESULTS / 'strict_mixed_precision_predictions.npz') as old:
        assert np.array_equal(y, old['y'])
        prediction_error = float(np.max(abs(predictions['snp'] - old['snp'])))
    r_error = abs(report['snp']['pearson_mean'] - strict_report['snp']['pearson_mean'])
    assert prediction_error < 1e-7 and r_error < 1e-10, (prediction_error, r_error)
    assert protected_snapshot() == protected, 'A protected core/strict file changed during this run'
    named = {'regional_snp_vector_rkhs_selected': predictions['local_selected'],
             'snp_linear_companion': predictions['snp'], 'train_mean': predictions['train_mean'],
             'train_median': predictions['train_median']}
    with (RESULTS / (TAG + '_named_predictions.npz')).open('wb') as stream:
        np.savez_compressed(stream, y=y, accession_ids=ids, **named)
    parameters = []
    for fold in state['folds']:
        parameters.append({
            'repeat': fold['repeat'], 'fold': fold['fold'],
            'train_indices': fold['train_indices'], 'test_indices': fold['test_indices'],
            'regional_snp_vector_rkhs_selected': dict(fold['selection']['local_selected'], kind='regional_snp_vector_rkhs'),
            'snp_linear_companion': fold['selection']['snp'],
        })
    histogram = Counter((p['regional_snp_vector_rkhs_selected']['window_scale'],
                         p['regional_snp_vector_rkhs_selected']['gamma'],
                         p['regional_snp_vector_rkhs_selected']['lambda']) for p in parameters)
    local = report['local_selected']
    comparisons = {}
    for label, base in [('snp_linear_companion', report['snp']),
                        ('saved_core_window_mean_local_selected', strict_report['local_selected'])]:
        comparisons[label] = {m + '_repeat_mean_difference': local[m + '_mean'] - base[m + '_mean']
                              for m in ('pearson', 'spearman', 'mae', 'rmse', 'predictive_r2')}
    summary = {
        'method_definition': method_definition(), 'protocol': regional_protocol(),
        'provenance': provenance, 'selftests': tests,
        'n_accessions': len(y), 'outer_predictions_shape': [10, len(y)], 'completed_outer_folds': 50,
        'regional_candidates_per_inner_split': len(core.SCALES) * len(core.GAMMAS) * len(core.LAMBDAS),
        'models': {name: core.summarize(y, pred) for name, pred in named.items()},
        'alignment': {'reference': 'strict_mixed_precision',
                      'snp_pearson_mean': report['snp']['pearson_mean'],
                      'reference_snp_pearson_mean': strict_report['snp']['pearson_mean'],
                      'snp_pearson_mean_absolute_difference': r_error,
                      'all_snp_oof_predictions_max_absolute_difference_days': prediction_error,
                      'protected_core_and_strict_files_unchanged': True},
        'descriptive_comparisons': comparisons,
        'selected_parameters': parameters,
        'joint_parameter_counts': [{'window_scale': s, 'gamma': g, 'lambda': lam, 'outer_fold_count': count}
                                   for (s, g, lam), count in sorted(histogram.items())],
        'files': {name: str((RESULTS / (TAG + suffix)).resolve()) for name, suffix in (
            ('state_with_all_fold_indices_preprocessing_and_parameters', '_state.json'),
            ('original_core_api_predictions', '_predictions.npz'),
            ('explicitly_named_predictions_with_accession_ids', '_named_predictions.npz'),
            ('metrics', '_metrics.json'), ('selftests', '_selftests.json'))},
        'finished': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    # Preserve the core API keys, but unambiguously define local_selected here.
    core.save_json(RESULTS / (TAG + '_metrics.json'), dict(report, family_definitions=method_definition()['family_aliases'], method_definition=method_definition()))
    core.save_json(RESULTS / (TAG + '_summary.json'), summary)
    print('FINAL ' + json.dumps({'regional_pearson_mean': local['pearson_mean'],
          'regional_pearson_sd': local['pearson_sd'], 'regional_rmse_mean': local['rmse_mean'],
          'snp_pearson_mean': report['snp']['pearson_mean'], 'snp_prediction_alignment_max_error': prediction_error,
          'summary': str((RESULTS / (TAG + '_summary.json')).resolve())}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['tests', 'cv', 'all'], default='all')
    parser.add_argument('--window-batch', type=int, default=512)
    args = parser.parse_args()
    assert list(core.SCALES) == [5, 10, 25, 50, 100]
    assert len(core.GAMMAS) == 9 and len(core.LAMBDAS) == 11
    codes, y, _, ids, groups = core.load_dormancy()
    with core.threadpool_limits(2):
        if args.mode in ('tests', 'all'):
            tests = selftests(codes, y, groups, args.window_batch)
        else:
            tests = json.loads((RESULTS / (TAG + '_selftests.json')).read_text(encoding='utf-8'))
            assert tests['passed'] and tests['real_outer_split_preprocessing']['window_batch'] == args.window_batch
        if args.mode in ('cv', 'all'):
            run_cv(codes, y, ids, groups, tests, args.window_batch)


if __name__ == '__main__':
    main()
