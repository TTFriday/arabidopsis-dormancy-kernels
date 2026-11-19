"""CPU imports, numerical checks, identities and archive-array completeness."""
import argparse
import ast
import importlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
NAMES = ['kernel_benchmark', 'strong_baselines', 'parallel_bayesc', 'regional_snp_rkhs', 'robustness_checks', 'numerical_audit', 'external_validation', 'rice_validation', 'flowering_validation', 'graph_validation', 'graph_validation_long', 'boundary_sensitivity', 'pc_regression', 'spectral_diagnostics', 'computational_profile', 'streaming_equivalence_suite']

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        parser.error('Use a new check output directory')
    out.mkdir(parents=True)
    modules = {name: importlib.import_module(name) for name in NAMES}
    core = modules['kernel_benchmark']
    for m in modules.values():
        if hasattr(m, 'DEVICE'):
            m.DEVICE = torch.device('cpu')
        for attr in ('core', 'kb'):
            if hasattr(m, attr):
                getattr(m, attr).DEVICE = torch.device('cpu')
    core.DEVICE = torch.device('cpu')
    report = {'imports_passed': list(modules), 'device': 'cpu', 'numpy': np.__version__, 'pandas': pd.__version__, 'torch': torch.__version__}
    writes = {}
    original_save_json = core.save_json
    core.save_json = lambda path, obj: writes.update({str(path.name): obj})
    core.selftest()
    report['core'] = list(writes.values())[-1]
    numerical = modules['numerical_audit']
    report['rank1'] = numerical.numerical_selftests()
    report['external'] = modules['external_validation'].selftests()
    for name in ('graph_validation', 'graph_validation_long'):
        m = modules[name]
        m.save_json = lambda path, obj: None
        r = {}
        m.kernel_unit_tests(r)
        report[name] = r['unit_tests']
        assert m.OLD_ENCODER == ROOT / 'scripts/reference_encoder.py'
    modules['strong_baselines'].selftest(out)
    report['strong_baselines'] = list(writes.values())[-1]
    # Use the included frozen raw-missingness dataset without relying on the
    # historical manifest's machine-specific file paths or changed code hashes.
    rice = modules['rice_validation']
    def frozen_rice():
        with np.load(ROOT / 'revision/data/rice_dataset.npz', allow_pickle=False) as f:
            return f['dosage'].copy(), f['y'].copy(), f['ids'].copy(), {s: f['groups_' + str(s)].copy() for s in rice.SCALES}, {'dataset_sha256': rice.sha256(ROOT / 'revision/data/rice_dataset.npz')}
    rice.load_dataset = frozen_rice
    report['rice'] = rice.selftests('cpu')
    codes, y, meta, ids, groups = core.load_dormancy()
    report['robustness'] = modules['robustness_checks'].selftests(core.GenomicKernels(codes, groups, local_dtype=torch.float32), codes, y, ids)
    report['regional_rkhs'] = modules['regional_snp_rkhs'].selftests(codes, y, groups, 512)
    modules['computational_profile'].equivalence_test()
    report['streaming'] = writes['streaming_aggregation_equivalence.json']
    # Full, not sampled, copied genotype-subset consistency.
    header = ROOT / 'data/raw/atpolydb/call_method_75/call_method_75_TAIR9.csv'
    with header.open('rb') as f:
        first, second = f.readline(), f.readline()
    rawids = np.array([int(x) for x in second.decode().rstrip().split(',')[2:]])
    cols = np.load(core.DATA / 'dormancy_cols_in_1307.npy')
    assert len(rawids) == len(set(rawids)) == 1307
    assert np.array_equal(rawids[cols], ids)
    assert set(rawids) == set(pd.read_csv(core.DATA / 'accessions_all.csv').ecotype_id)
    full = np.load(core.DATA / 'genotypes_all_1307.npy', mmap_mode='r')
    subset = np.load(core.DATA / 'genotypes_83.npy', mmap_mode='r')
    for start in range(0, len(full), 8192):
        assert np.array_equal(full[start:start+8192][:, cols], subset[start:start+8192])
    bf = np.load(core.DATA / 'block_features.npy', mmap_mode='r')
    b83 = np.load(core.DATA / 'block_features_83.npy', mmap_mode='r')
    assert np.array_equal(bf[cols], b83)
    numeric_out = out / 'numerical'
    numeric_out.mkdir()
    numerical.OUT = numeric_out
    report['identity_audit'] = numerical.identity_audit()['column_order_audit']
    audit = numerical.numerical_audit()
    report['numerical_rebuild'] = {'old_function_max_error': audit['original_function_reproduction']['max_abs_mean_r_difference'], 'selected_blocks': list(audit['selected_blocks'])}
    graph_counts = {}
    for folder in ('graph_reference', 'graph_reference_long'):
        files = list((ROOT / 'revision/results' / folder).glob('width*/*_nested_predictions.npz'))
        assert len(files) == 12
        for path in files:
            with np.load(path, allow_pickle=False) as f:
                assert np.array_equal(f['ids'], ids) and np.array_equal(f['y'], y)
                assert f['predictions'].shape == (10, 83) and np.isfinite(f['predictions']).all()
                assert f['train_mean'].shape == (10, 83) and np.isfinite(f['train_mean']).all()
        graph_counts[folder] = len(files)
    for path in (ROOT / 'revision/results').glob('flowering_*_predictions.npz'):
        identity = np.load(path.with_name(path.name.replace('_predictions', '_identity')))
        assert np.array_equal(rawids[identity['raw_genotype_columns']], identity['ids'])
        if '_excluded_' in path.name:
            assert not set(identity['ids']) & set(ids)
        with np.load(path, allow_pickle=False) as f:
            assert np.array_equal(f['y'], identity['y'])
            assert all(v.shape == (10, len(identity['ids'])) and np.isfinite(v).all() for k, v in dict(f).items() if k != 'y')
    assert len(list((ROOT / 'revision/results').glob('flowering_*_predictions.npz'))) == 6
    strong = ROOT / 'revision/results/strong_baselines_20261118'
    strong_counts = {}
    for phase in ('fast', 'bayesc'):
        state = json.loads((strong / (phase + '_state.json')).read_text(encoding='utf-8'))
        assert state['status'] == 'complete' and len(state['folds']) == 50
        assert {(f['repeat'], f['fold']) for f in state['folds']} == {(r, f) for r in range(10) for f in range(5)}
        with np.load(strong / (phase + '_oof_predictions.npz'), allow_pickle=False) as f:
            assert np.array_equal(f['ids'], ids) and np.array_equal(f['y'], y)
            assert all(a.shape == (10, 83) and np.isfinite(a).all() for k, a in dict(f).items() if k not in ('ids', 'y'))
        strong_counts[phase] = 50
    report['strong_final_oof_folds'] = strong_counts
    # Reparse every raw CSV row with the original allele translation; compare
    # ALL panel columns and SNP coordinates without writing any model arrays.
    translation = bytes.maketrans(b'ACGTacgt', b'\x00\x01\x02\x03\x00\x01\x02\x03')
    from importlib import util
    for filename in ('01_build_genotype_matrix.py', '05_build_haplotype_graph.py'):
        spec = util.spec_from_file_location('legacy_' + filename[:2], ROOT / 'scripts' / filename)
        legacy = util.module_from_spec(spec)
        spec.loader.exec_module(legacy)
        if filename.startswith('01'):
            assert legacy.TRANS == translation
    coordinates = meta[['chrom', 'pos']].to_numpy(np.int64)
    with header.open('rb') as f:
        f.readline(); f.readline()
        row_count = 0
        for i, line in enumerate(f):
            chrom, pos, letters = line.rstrip(b'\n').split(b',', 2)
            raw = np.frombuffer(letters.replace(b',', b'').translate(translation), np.uint8)
            raw = np.where(raw > 3, np.uint8(4), raw)
            assert np.array_equal(raw, full[i]), i
            assert (int(chrom), int(pos)) == tuple(coordinates[i]), i
            row_count += 1
    assert row_count == len(full) == 214051
    report['copied_ids_and_full_arrays'] = {'passed': True, 'raw_panel_n': 1307, 'training_n': 83, 'all_snp_rows': len(full), 'raw_csv_every_call_and_coordinate_reparsed': True, 'all_block_channels': True, 'graph_oof_counts': graph_counts, 'flowering_cohorts': 6}
    # Actual OFFLINE rice reconstruction from the included source archives,
    # not merely an import or a frozen-dataset lookup. No recorded result changes.
    import shutil
    rice_input = out / 'rice_reconstruction/data'
    shutil.copytree(ROOT / 'revision/data', rice_input)
    rice_result = out / 'rice_reconstruction/results'
    rice_result.mkdir()
    rice.DATA, rice.RESULTS = rice_input, rice_result
    rice.DATASET_PATH = rice_input / 'rice_dataset.npz'
    rice.AUDIT_PATH = rice_result / 'rice_dataset_audit.json'
    rice.MANIFEST_PATH = rice_result / 'rice_manifest.json'
    core.save_json = original_save_json
    manifest = rice.prepare()
    assert rice.prepare()['dataset_sha256'] == manifest['dataset_sha256']
    with np.load(rice.DATASET_PATH, allow_pickle=False) as actual, np.load(ROOT / 'revision/data/rice_dataset.npz', allow_pickle=False) as expected:
        assert set(actual.files) == set(expected.files)
        for key in actual.files:
            assert actual[key].dtype == expected[key].dtype and actual[key].shape == expected[key].shape
            assert actual[key].tobytes() == expected[key].tobytes(), key
    report['rice_offline_raw_reconstruction'] = {'passed': True, 'all_array_bytes_equal': True, 'final_n': manifest['final_n'], 'snp_count': manifest['snp_count'], 'frozen_manifest_resume_passed': True}
    (out / 'checks.json').write_text(json.dumps(report, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + '\n', encoding='utf-8')
    print('PACKAGE CHECKS PASSED: all CPU imports, unit checks, full copied IDs/arrays and final OOF coverage', flush=True)

if __name__ == '__main__':
    main()
