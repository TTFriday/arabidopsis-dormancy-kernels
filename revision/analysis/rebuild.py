"""Run an included analysis in a new, independent reconstruction workspace."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args, options = parser.parse_known_args()
    if options[:1] == ['--']:
        options = options[1:]
    script = ROOT / 'revision/analysis' / (args.analysis + '.py')
    if script.parent != ROOT / 'revision/analysis' or not script.is_file() or args.analysis in ('rebuild', 'package_checks'):
        parser.error('Choose an included scientific analysis name')
    out = args.output.resolve()
    if out.exists() or ROOT == out or out in ROOT.parents:
        parser.error('Output must be a new directory, not an existing input location')
    out.mkdir(parents=True)
    for rel in ('data', 'scripts', 'results/legacy', 'revision/data', 'revision/analysis'):
        shutil.copytree(ROOT / rel, out / rel)
    result = out / 'revision/results'
    result.mkdir()
    copied = []
    if args.analysis in ('regional_snp_rkhs', 'spectral_diagnostics', 'computational_profile', 'streaming_equivalence_suite', 'robustness_checks'):
        for suffix in ('metrics.json', 'state.json', 'predictions.npz'):
            name = 'strict_mixed_precision_' + suffix
            shutil.copy2(ROOT / 'revision/results' / name, result / name)
            copied.append(name)
    if args.analysis in ('graph_validation', 'graph_validation_long'):
        for name in ('dryad_matching_identity_audit.csv', 'external_validation_published29_matching.csv'):
            shutil.copy2(ROOT / 'revision/results' / name, result / name)
            copied.append(name)
    (result / 'reconstruction_inputs.json').write_text(json.dumps({'frozen_comparator_or_identity_inputs': copied, 'historical_checkpoints_reused': False}, indent=2) + '\n', encoding='utf-8')
    run = subprocess.run([sys.executable, '-s', '-B', str(out / 'revision/analysis' / script.name), *options], cwd=out)
    raise SystemExit(run.returncode)

if __name__ == '__main__':
    main()
