"""Read-only rank-one intercept and Dryad identity audits.

Run with python -s -B numerical_audit.py. Only this revision's results directory
is written. Original code is never imported or executed at module level.
"""
from __future__ import annotations

import ast
import hashlib
import json
import platform
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import linalg, stats
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

SCRIPT = Path(__file__).resolve()
REVISION = SCRIPT.parents[1]
ROOT = SCRIPT.parents[2]
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "legacy" / "tables"
EXPERIMENTS = ROOT / "results" / "legacy" / "experiments"
OUT = REVISION / "results"
LAMBDA = 1e-8
N_REPEATS = 10
N_FOLDS = 5
HEADER = ROOT / "data/raw/atpolydb/call_method_75/call_method_75_TAIR9.csv"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(type(value).__name__)


def write_json(filename, result):
    (OUT / filename).write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False,
                   default=json_default) + "\n", encoding="utf-8"
    )


def difference(a, b):
    delta = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return {"n_predictions": int(delta.size),
            "max_abs_prediction_error_days": float(np.max(np.abs(delta))),
            "mean_abs_prediction_error_days": float(np.mean(np.abs(delta))),
            "rms_prediction_error_days": float(np.sqrt(np.mean(delta ** 2)))}


def metrics(y, pred):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    if len(y) < 2 or np.ptp(y) == 0 or np.ptp(pred) == 0:
        pearson = spearman = None
    else:
        pearson = float(stats.pearsonr(y, pred)[0])
        spearman = float(stats.spearmanr(y, pred).statistic)
    return {"n": int(len(y)), "pearson_r": pearson, "spearman_rho": spearman,
            "mae_days": float(np.mean(np.abs(pred - y))) if len(y) else None,
            "rmse_days": float(np.sqrt(np.mean((pred - y) ** 2))) if len(y) else None}


def repeat_metrics(y, predictions):
    rows = [metrics(y, pred) for pred in predictions]
    result = {"per_repeat": rows, "sd_definition": "population, ddof=0"}
    for key in ("pearson_r", "spearman_rho", "mae_days", "rmse_days"):
        values = np.array([row[key] for row in rows], dtype=float)
        result[key + "_mean"] = float(values.mean())
        result[key + "_sd"] = float(values.std())
    return result


def rank1_predict(ztr, ytr, zte, lam=LAMBDA):
    """Unpenalized intercept, same original K=zz.T scale, no fold rescaling."""
    ztr = np.asarray(ztr, dtype=np.float64)
    zte = np.asarray(zte, dtype=np.float64)
    mean_z = ztr.mean(axis=0)
    mean_y = ytr.mean()
    ztrc = ztr - mean_z
    beta = ztrc.T @ (ytr - mean_y) / ((ztrc ** 2).sum(axis=0) + lam)
    return mean_y + (zte - mean_z) * beta, beta, mean_y - mean_z * beta


def stable_generic_krr(Ktr, ytr, Kcross, lam=LAMBDA):
    """Kernel-only centered eigensolve in the space orthogonal to the intercept.

    In these exact rank-one kernels, roundoff-sized null modes must not be
    inverted and then multiplied by Kcross: that would amplify rounding by
    1/lambda. Predictions use only the resolved kernel eigenspace. No C^-1 y
    minus b*C^-1 1, large nullspace alpha, or fold trace normalization is used.
    The spectral cutoff and retained rank are returned rather than hidden.
    """
    n = len(ytr)
    Q = linalg.helmert(n, full=False).T
    reduced = Q.T @ Ktr @ Q
    reduced = (reduced + reduced.T) * 0.5
    eigenvalues, eigenvectors = linalg.eigh(reduced, check_finite=True)
    # Projection of a constant kernel can leave roundoff with tiny positive
    # AND negative eigenvalues. Bound that error on the input kernel scale,
    # not on the near-zero projected spectrum alone.
    radius = max(float(np.max(np.abs(eigenvalues))), float(linalg.norm(Ktr, "fro")))
    cutoff = 64 * np.finfo(np.float64).eps * n * radius
    if eigenvalues[0] < -cutoff:
        raise AssertionError("Centered kernel has a negative non-roundoff eigenvalue")
    keep = eigenvalues > cutoff
    V = Q @ eigenvectors[:, keep]
    coefficients = (V.T @ (ytr - ytr.mean())) / (eigenvalues[keep] + lam)
    mean_train = Ktr.mean(axis=0)
    Kcross_c = (Kcross - Kcross.mean(axis=1, keepdims=True)
                - mean_train[None, :] + Ktr.mean())
    predictions = ytr.mean() + (Kcross_c @ V) @ coefficients
    intercept = float(ytr.mean() - (mean_train @ V) @ coefficients)
    diagnostics = {"retained_rank": int(keep.sum()),
                   "spectral_cutoff": cutoff,
                   "min_eigenvalue": float(eigenvalues[0]),
                   "max_eigenvalue": float(eigenvalues[-1]),
                   "discarded_max_abs_eigenvalue": float(
                       np.max(np.abs(eigenvalues[~keep]))) if (~keep).any() else 0.0,
                   "raw_kernel_trace_per_train_sample": float(np.trace(Ktr) / n)}
    return predictions, intercept, diagnostics


def numerical_selftests():
    rng = np.random.default_rng(20261118)
    y = rng.normal(size=17) * 20 + 40
    cases = [
        ("nonzero_train_mean", rng.normal(size=17) + 3, rng.normal(size=6) + 2, LAMBDA),
        ("constant_train", np.ones(17), np.arange(6, dtype=float), LAMBDA),
        ("zero_feature", np.zeros(17), np.zeros(6), LAMBDA),
        ("finite_regularization", rng.normal(size=17), rng.normal(size=6), 0.3),
        ("nonunit_kernel_scale", rng.normal(size=17) * 4, rng.normal(size=6) * 4, LAMBDA),
    ]
    results = []
    for label, ztr, zte, lam in cases:
        fixed, _, intercept = rank1_predict(ztr, y, zte, lam)
        generic, generic_intercept, diag = stable_generic_krr(
            np.outer(ztr, ztr), y, np.outer(zte, ztr), lam)
        error = difference(fixed, generic)
        assert error["max_abs_prediction_error_days"] < 1e-9, (label, error)
        assert abs(intercept - generic_intercept) < 1e-9
        results.append({"case": label, "lambda": lam, **error, **diag})
    ztr, zte = rng.normal(size=17), rng.normal(size=6)
    p0, _, _ = rank1_predict(ztr, y, zte)
    p1, _, _ = rank1_predict(ztr + 5, y, zte + 5)
    error = difference(p0, p1)
    assert error["max_abs_prediction_error_days"] < 1e-9
    results.append({"case": "feature_offset_invariance", **error})
    return {"passed": True, "assertion_tolerance_days": 1e-9, "cases": results}


def column_pearson(pred, y):
    pc = pred - pred.mean(axis=0)
    yc = y - y.mean()
    return (pc.T @ yc) / np.sqrt((pc ** 2).sum(axis=0) * (yc ** 2).sum())


def original_function_scores(Z, y):
    """Execute only the original function AST, without importing its module."""
    path = ROOT / "scripts/legacy_rank1.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == "single_kernel_cv_r")
    isolated = ast.Module(body=[node], type_ignores=[])
    scope = {"np": np, "stats": stats, "KFold": KFold}
    exec(compile(isolated, str(path), "exec"), scope)
    return scope["single_kernel_cv_r"](Z, y, LAMBDA, N_REPEATS, N_FOLDS)


def numerical_audit():
    start = time.perf_counter()
    selftest_results = numerical_selftests()
    pheno = pd.read_csv(PROCESSED / "dormancy_phenotype.csv")
    y = pheno.dsds50.to_numpy(dtype=np.float64)
    bm = np.asarray(np.load(PROCESSED / "block_features_83.npy", mmap_mode="r")[:, :, 0],
                    dtype=np.float64)
    blocks = pd.read_csv(PROCESSED / "haplotype_blocks.csv")
    archived = json.loads((EXPERIMENTS / "results/dog1_mechanism.json").read_text(
        encoding="utf-8"))
    dog1 = int(archived["block_of_dog1"])
    old_best = int(archived["single_block_sufficiency"]["argmax_block"])
    assert bm.shape == (83, 8589)
    assert np.array_equal(blocks.block_id.to_numpy(), np.arange(8589))
    sd = bm.std(axis=0)
    Z = (bm - bm.mean(axis=0)) / np.where(sd < 1e-8, 1.0, sd)
    all_stats = {kind: {metric: np.empty((N_REPEATS, Z.shape[1]))
                       for metric in ("r", "mae", "rmse")}
                 for kind in ("old", "corrected")}
    focus = sorted({dog1, old_best})
    focus_predictions = {kind: np.empty((N_REPEATS, len(y), len(focus)))
                         for kind in ("old", "corrected")}
    folds = []
    for rep in range(N_REPEATS):
        old_pred = np.zeros_like(Z)
        new_pred = np.zeros_like(Z)
        rep_folds = list(KFold(n_splits=N_FOLDS, shuffle=True, random_state=rep).split(Z))
        folds.append(rep_folds)
        for tr, te in rep_folds:
            b = y[tr].mean()
            num = Z[tr].T @ (y[tr] - b)
            den = (Z[tr] ** 2).sum(axis=0) + LAMBDA
            old_pred[te] = b + Z[te] * (num / den)[None, :]
            new_pred[te], _, _ = rank1_predict(Z[tr], y[tr], Z[te])
        for kind, pred in (("old", old_pred), ("corrected", new_pred)):
            all_stats[kind]["r"][rep] = column_pearson(pred, y)
            all_stats[kind]["mae"][rep] = np.mean(np.abs(pred - y[:, None]), axis=0)
            all_stats[kind]["rmse"][rep] = np.sqrt(np.mean((pred - y[:, None]) ** 2, axis=0))
            focus_predictions[kind][rep] = pred[:, focus]
    scores = {kind: all_stats[kind]["r"].mean(axis=0) for kind in all_stats}
    assert all(np.isfinite(s).all() for s in scores.values())
    original_scores = original_function_scores(Z, y)
    max_original_diff = float(np.max(np.abs(scores["old"] - original_scores)))
    assert max_original_diff < 1e-12
    assert int(np.argmax(scores["old"])) == old_best
    new_best = int(np.argmax(scores["corrected"]))
    ranks = {kind: stats.rankdata(-s, method="min").astype(int) for kind, s in scores.items()}
    screen = blocks.copy()
    for kind in all_stats:
        screen[kind + "_rank"] = ranks[kind]
        for metric, values in all_stats[kind].items():
            screen[kind + "_" + metric + "_mean"] = values.mean(axis=0)
            screen[kind + "_" + metric + "_sd"] = values.std(axis=0)
    screen["delta_r_corrected_minus_old"] = scores["corrected"] - scores["old"]
    screen["delta_rank_corrected_minus_old"] = ranks["corrected"] - ranks["old"]
    screen.to_csv(OUT / "rank1_block_screen.csv", index=False)
    selected, prediction_rows = {}, []
    for block in sorted({dog1, old_best, new_best}):
        z = Z[:, block]
        old = np.empty((N_REPEATS, len(y)))
        fixed = np.empty_like(old)
        generic = np.empty_like(old)
        fold_records = []
        for rep, rep_folds in enumerate(folds):
            for fold, (tr, te) in enumerate(rep_folds):
                b = y[tr].mean()
                old[rep, te] = b + z[te] * (z[tr] @ (y[tr] - b)) / (
                    (z[tr] ** 2).sum() + LAMBDA)
                fixed[rep, te], beta, intercept = rank1_predict(z[tr], y[tr], z[te])
                generic[rep, te], generic_intercept, diag = stable_generic_krr(
                    np.outer(z[tr], z[tr]), y[tr], np.outer(z[te], z[tr]))
                assert diag["retained_rank"] == int(np.ptp(z[tr]) > 0)
                err = difference(fixed[rep, te], generic[rep, te])
                assert err["max_abs_prediction_error_days"] < 1e-9
                fold_records.append({"repeat": rep, "fold": fold,
                                     "n_train": len(tr), "n_test": len(te),
                                     "training_y_mean": float(b),
                                     "training_z_mean": float(z[tr].mean()),
                                     "corrected_beta": float(beta),
                                     "corrected_raw_kernel_intercept": float(intercept),
                                     "generic_raw_kernel_intercept": generic_intercept,
                                     "abs_intercept_difference": float(abs(
                                         intercept - generic_intercept)),
                                     "old_vs_stable_generic": difference(old[rep, te], generic[rep, te]),
                                     "corrected_vs_stable_generic": err,
                                     "generic_diagnostics": diag})
                for i in te:
                    prediction_rows.append({"block_id": block, "repeat": rep, "fold": fold,
                                            "phenotype_row": int(i),
                                            "ecotype_id": int(pheno.iloc[i].ecotype_id),
                                            "nativename": pheno.iloc[i].nativename,
                                            "observed_days": y[i], "old_prediction": old[rep, i],
                                            "corrected_prediction": fixed[rep, i],
                                            "stable_generic_prediction": generic[rep, i]})
        if block in focus:
            j = focus.index(block)
            assert np.max(np.abs(fixed - focus_predictions["corrected"][:, :, j])) < 1e-9
            assert np.max(np.abs(old - focus_predictions["old"][:, :, j])) < 1e-9
        roles = [role for role, value in (("DOG1", dog1), ("original_bestblock", old_best),
                                         ("corrected_bestblock", new_best)) if value == block]
        selected[str(block)] = {
            "roles": roles, "old_metrics": repeat_metrics(y, old),
            "corrected_metrics": repeat_metrics(y, fixed),
            "stable_generic_metrics": repeat_metrics(y, generic),
            "old_vs_stable_generic": difference(old, generic),
            "corrected_vs_stable_generic": difference(fixed, generic),
            "old_vs_corrected": difference(old, fixed),
            "old_rank": int(ranks["old"][block]),
            "corrected_rank": int(ranks["corrected"][block]),
            "old_percentile_strictly_lower": float((scores["old"] < scores["old"][block]).mean() * 100),
            "corrected_percentile_strictly_lower": float((scores["corrected"] < scores["corrected"][block]).mean() * 100),
            "abs_mean_r_error_old_vs_generic": float(abs(
                stats_mean_r(old, y) - stats_mean_r(generic, y))),
            "abs_mean_r_error_corrected_vs_generic": float(abs(
                stats_mean_r(fixed, y) - stats_mean_r(generic, y))),
            "folds": fold_records,
        }
    pd.DataFrame(prediction_rows).sort_values(["block_id", "repeat", "phenotype_row"]).to_csv(
        OUT / "rank1_prediction_comparison.csv", index=False)
    return {
        "protocol": {"n_accessions": len(y), "n_blocks": Z.shape[1], "repeats": N_REPEATS,
                     "folds": N_FOLDS, "shuffle": True, "random_state": "rep (0..9)",
                     "lambda": LAMBDA, "dtype": "float64", "kernel": "K = z z.T, p=1",
                     "standardization": "original full-83 mean and population std retained; no fold rescaling",
                     "intercept": "unpenalized; center z on each training fold for closed form",
                     "score": "mean of 10 out-of-fold Pearson r values",
                     "rank": "descending Pearson mean, competition rank (min); percentile uses strict <",
                     "generic": "kernel-only centered eigensolve; discard roundoff null modes at 64*eps*n*max(input kernel Frobenius norm, centered spectral radius)",
                     "generic_scope": "all 50 folds of DOG1, original bestblock, and corrected bestblock",
                     "interpretation": "Numerical-only repair, not a new model-selection or preprocessing protocol."},
        "root_cause": {"file": (ROOT / "scripts/legacy_rank1.py").as_posix(),
                       "lines": [91, 110],
                       "description": "Full-sample zero mean does not imply fold-training zero mean; ytr.mean is not the raw-feature intercept unless ztr.mean*beta=0."},
        "selftests": selftest_results,
        "original_function_reproduction": {"method": "isolated original function AST, all 8589 columns",
                                           "max_abs_mean_r_difference": max_original_diff,
                                           "archived_report": archived["single_block_sufficiency"],
                                           "abs_dog1_r_difference_from_archive": float(abs(
                                               scores["old"][dog1] - archived["single_block_sufficiency"]["dog1_block_r"])),
                                           "abs_best_r_difference_from_archive": float(abs(
                                               scores["old"][old_best] - archived["single_block_sufficiency"]["genome_max"]))},
        "screen_summary": {kind: {"genome_mean_r": float(s.mean()), "genome_sd_r": float(s.std()),
                                  "genome_max_r": float(s.max()), "bestblock": int(np.argmax(s)),
                                  "top10": [{"block_id": int(i), "rank": int(ranks[kind][i]),
                                             "pearson_mean": float(s[i])}
                                            for i in np.argsort(-s, kind="stable")[:10]]}
                           for kind, s in scores.items()},
        "screen_changes": {"n_blocks_with_changed_rank": int((ranks["old"] != ranks["corrected"]).sum()),
                           "max_abs_r_change": float(np.max(np.abs(scores["corrected"] - scores["old"]))),
                           "mean_abs_r_change": float(np.mean(np.abs(scores["corrected"] - scores["old"])))},
        "selected_blocks": selected,
        "precision_note": "Only measured absolute errors are reported. No unmeasured machine-precision equivalence is asserted.",
        "outputs": [(OUT / name).as_posix() for name in (
            "numerical_audit.json", "rank1_block_screen.csv", "rank1_prediction_comparison.csv")],
        "elapsed_seconds": time.perf_counter() - start,
    }


def stats_mean_r(predictions, y):
    return float(np.mean([stats.pearsonr(pred, y)[0] for pred in predictions]))


def norm_name(value):
    return str(value).strip().lower().replace(" ", "")


def records(frame):
    return json.loads(frame.to_json(orient="records", force_ascii=False))


def identity_audit():
    panel = pd.read_csv(PROCESSED / "accessions_all.csv")
    pheno = pd.read_csv(PROCESSED / "dormancy_phenotype.csv")
    train_cols = np.load(PROCESSED / "dormancy_cols_in_1307.npy")
    with HEADER.open("rb") as handle:
        first = handle.readline()
        second = handle.readline()
    header_ids = np.array([int(value) for value in second.decode().rstrip().split(",")[2:]])
    assert len(panel) == len(header_ids) == 1307
    assert len(train_cols) == len(pheno) == 83 and len(set(train_cols)) == 83
    assert np.all((train_cols >= 0) & (train_cols < len(panel)))
    assert len(set(header_ids)) == len(header_ids)
    assert panel.ecotype_id.is_unique
    assert set(header_ids) == set(panel.ecotype_id)
    header_col_by_id = {int(eid): col for col, eid in enumerate(header_ids)}
    used_train_ids = set(map(int, header_ids[train_cols]))
    phenotype_ids = set(map(int, pheno.ecotype_id))
    local_names = set(pheno.nativename.astype(str))
    train_column_set = set(map(int, train_cols))
    panel_by_id = panel.set_index("ecotype_id")
    exact, normalized = defaultdict(list), defaultdict(list)
    for col, name in enumerate(panel.nativename.astype(str)):
        exact[name].append(col)
        normalized[norm_name(name)].append(col)

    def candidates(cols):
        return [{"metadata_row": int(col),
                 "genotype_col": int(header_col_by_id[int(panel.iloc[col].ecotype_id)]),
                 "ecotype_id": int(panel.iloc[col].ecotype_id),
                 "nativename": str(panel.iloc[col].nativename),
                 "in_phenotype_train83": int(panel.iloc[col].ecotype_id) in phenotype_ids}
                for col in cols]

    def annotate(frame):
        rows = []
        for source_row, row in frame.iterrows():
            item = row.to_dict()
            col = int(row.panel_col)
            valid = 0 <= col < len(panel)
            exact_cols = exact.get(str(row.nativename), [])
            norm_cols = normalized.get(norm_name(row.nativename), [])
            candidate_cols = exact_cols or norm_cols
            candidate_ids = [int(panel.iloc[c].ecotype_id) for c in candidate_cols]
            # Reproduce the original exact-last / normalized-first selection,
            # while explicitly preserving all candidates for ambiguity reporting.
            expected_col = exact_cols[-1] if exact_cols else (norm_cols[0] if norm_cols else -1)
            expected_rule = "exact" if exact_cols else ("normalised" if norm_cols else "unmatched")
            metadata_id = int(panel.iloc[col].ecotype_id) if valid else None
            projected_id = int(header_ids[col]) if valid else None
            item.update({
                "source_csv_line": int(source_row + 2),
                "normalized_name": norm_name(row.nativename),
                "valid_panel_col": valid,
                "metadata_ecotype_id": metadata_id,
                "metadata_panel_name": str(panel.iloc[col].nativename) if valid else None,
                "genotype_col_by_matched_ecotype_id": header_col_by_id[metadata_id] if valid else None,
                "projected_genotype_ecotype_id": projected_id,
                "projected_genotype_name": str(panel_by_id.loc[projected_id].nativename) if valid else None,
                "metadata_matches_projected_genotype": metadata_id == projected_id if valid else None,
                "in_train83_by_matched_ecotype_id": valid and metadata_id in phenotype_ids,
                "in_train83_by_raw_column": valid and col in train_column_set,
                "in_train83_by_projected_ecotype_id": valid and projected_id in used_train_ids,
                "original_flag_reproduced_by_exact_name": bool(row.in_train83) == (str(row.nativename) in local_names),
                "frozen_match_reproduced": col == expected_col and str(row.match_rule) == expected_rule,
                "exact_candidate_metadata_rows": json.dumps(exact_cols),
                "normalized_candidate_metadata_rows": json.dumps(norm_cols),
                "candidate_ecotype_ids": json.dumps(candidate_ids),
                "identity_ambiguous": len(candidate_cols) > 1,
                "candidate_overlap_lower": bool(candidate_ids) and all(eid in phenotype_ids for eid in candidate_ids),
                "candidate_overlap_upper": any(eid in phenotype_ids for eid in candidate_ids),
                "candidate_train_membership_ambiguous": bool(candidate_ids) and
                    len({eid in phenotype_ids for eid in candidate_ids}) > 1,
            })
            rows.append(item)
        return pd.DataFrame(rows)

    matching = annotate(pd.read_csv(TABLES / "P2_dryad_matching.csv"))
    predictions = annotate(pd.read_csv(TABLES / "P2_dryad_predictions.csv"))
    assert (matching.in_train83_by_raw_column == matching.in_train83_by_projected_ecotype_id).all()
    assert (predictions.in_train83_by_raw_column == predictions.in_train83_by_projected_ecotype_id).all()
    matching.to_csv(OUT / "dryad_matching_identity_audit.csv", index=False)
    predictions.to_csv(OUT / "dryad_predictions_identity_audit.csv", index=False)

    def summarize(frame):
        matched = frame[frame.valid_panel_col]
        return {
            "total_rows": len(frame), "matched_rows": len(matched),
            "unmatched_rows": int((frame.panel_col == -1).sum()),
            "invalid_other_columns": int((~frame.valid_panel_col & frame.panel_col.ne(-1)).sum()),
            "original_name_overlap": int(matched.in_train83.sum()),
            "matched_metadata_ecotype_id_overlap": int(matched.in_train83_by_matched_ecotype_id.sum()),
            "matched_metadata_ecotype_id_nonoverlap": int((~matched.in_train83_by_matched_ecotype_id).sum()),
            "projected_raw_column_overlap": int(matched.in_train83_by_raw_column.sum()),
            "projected_genotype_ecotype_id_overlap": int(matched.in_train83_by_projected_ecotype_id.sum()),
            "projected_genotype_nonoverlap": int((~matched.in_train83_by_projected_ecotype_id).sum()),
            "original_vs_matched_ecotype_flag_changes": int((matched.in_train83 != matched.in_train83_by_matched_ecotype_id).sum()),
            "original_vs_projected_genotype_flag_changes": int((matched.in_train83 != matched.in_train83_by_projected_ecotype_id).sum()),
            "projected_overlaps_originally_flagged_independent": int((~matched.in_train83 & matched.in_train83_by_projected_ecotype_id).sum()),
            "original_overlap_but_projected_nonoverlap": int((matched.in_train83 & ~matched.in_train83_by_projected_ecotype_id).sum()),
            "metadata_vs_projected_genotype_identity_mismatches": int((~matched.metadata_matches_projected_genotype.astype(bool)).sum()),
            "ambiguous_matched_names": int(matched.identity_ambiguous.sum()),
            "ambiguous_training_membership_names": int(matched.candidate_train_membership_ambiguous.sum()),
            "name_candidate_overlap_lower_bound": int(matched.candidate_overlap_lower.sum()),
            "name_candidate_overlap_upper_bound": int(matched.candidate_overlap_upper.sum()),
            "original_exact_flag_reproduction_failures": int((~frame.original_flag_reproduced_by_exact_name).sum()),
            "original_frozen_matching_reproduction_failures": int((~frame.frozen_match_reproduced).sum()),
        }

    key = ["nativename", "panel_col", "dsds50_dryad"]
    comparison = predictions.merge(matching[key + ["in_train83"]], on=key, how="outer",
                                   suffixes=("_prediction", "_matching"), indicator=True,
                                   validate="one_to_one")
    matched_comparison = comparison[comparison.panel_col.ge(0)]
    alignment = pd.DataFrame({
        "phenotype_row": np.arange(83), "genotype_col": train_cols,
        "phenotype_ecotype_id": pheno.ecotype_id,
        "phenotype_name": pheno.nativename,
        "raw_header_ecotype_id": header_ids[train_cols],
        "sorted_metadata_ecotype_id_at_same_position": panel.iloc[train_cols].ecotype_id.to_numpy(),
        "sorted_metadata_name_at_same_position": panel.iloc[train_cols].nativename.to_numpy(),
    })
    alignment.to_csv(OUT / "training_identity_column_order.csv", index=False)
    panel_order = pd.DataFrame({"position": np.arange(len(panel)),
                               "metadata_ecotype_id": panel.ecotype_id,
                               "metadata_name": panel.nativename,
                               "raw_genotype_ecotype_id": header_ids,
                               "raw_genotype_name": [str(panel_by_id.loc[int(e)].nativename) for e in header_ids]})
    panel_order.to_csv(OUT / "panel_identity_column_order.csv", index=False)
    genotype83 = np.load(PROCESSED / "genotypes_83.npy", mmap_mode="r")
    genotype_all = np.load(PROCESSED / "genotypes_all_1307.npy", mmap_mode="r")
    sampled_rows = np.unique(np.linspace(0, genotype_all.shape[0] - 1, 1000, dtype=int))
    sampled_genotype_match = np.array_equal(genotype83[sampled_rows], genotype_all[sampled_rows][:, train_cols])
    full_block_subset = np.asarray(np.load(PROCESSED / "block_features.npy", mmap_mode="r")[train_cols, :, 0], float)
    saved_block_subset = np.asarray(np.load(PROCESSED / "block_features_83.npy", mmap_mode="r")[:, :, 0], float)
    block_subset_error = float(np.max(np.abs(full_block_subset - saved_block_subset)))
    collisions = [{"normalized_name": name, "candidates": candidates(cols)}
                  for name, cols in normalized.items() if len(cols) > 1]
    ambiguous = [{"dryad_name": row.nativename, "frozen_metadata_row": int(row.panel_col),
                  "frozen_ecotype_id": int(row.metadata_ecotype_id),
                  "candidates": candidates(exact.get(str(row.nativename), []) or
                                            normalized.get(norm_name(row.nativename), [])),
                  "train_membership_ambiguous": bool(row.candidate_train_membership_ambiguous)}
                 for row in matching[matching.valid_panel_col & matching.identity_ambiguous].itertuples()]
    duplicate_dryad = matching[matching.normalized_name.duplicated(keep=False)]
    intersection_columns = ["nativename", "panel_col", "metadata_ecotype_id",
                            "genotype_col_by_matched_ecotype_id", "projected_genotype_ecotype_id",
                            "projected_genotype_name", "in_train83", "identity_ambiguous"]
    descriptive = {}
    for label, mask in (
            ("original_exact_name_overlap", predictions.in_train83),
            ("original_exact_name_nonoverlap", ~predictions.in_train83),
            ("matched_metadata_id_overlap", predictions.in_train83_by_matched_ecotype_id),
            ("matched_metadata_id_nonoverlap", ~predictions.in_train83_by_matched_ecotype_id),
            ("projected_genotype_id_overlap", predictions.in_train83_by_projected_ecotype_id),
            ("projected_genotype_id_nonoverlap", ~predictions.in_train83_by_projected_ecotype_id)):
        sub = predictions[mask & predictions.valid_panel_col]
        descriptive[label] = {kind: metrics(sub.dsds50_dryad, sub["pred_" + kind])
                              for kind in ("block", "snp")}
    return {
        "scope": "Identity audit of frozen P2 CSVs, no phenotype/model refitting and no original changes.",
        "identity_keys": {
            "matched_metadata_identity": "accessions_all.csv.iloc[panel_col].ecotype_id (frozen chosen name match)",
            "actual_projected_genotype_identity": "raw genotype header ecotype ID at panel_col, since P2 indexes genotype/block arrays directly",
            "phenotype_training_identity": "dormancy_phenotype.csv.ecotype_id; checked against raw_header[dormancy_cols_in_1307.npy]",
            "warning": "Metadata row positions and raw genotype column positions use different orders; direct panel_col membership describes projected genotypes, not nominal Dryad names."},
        "matching_table": summarize(matching), "prediction_table": summarize(predictions),
        "table_consistency": {
            "all_matched_rows_have_one_prediction": bool((matched_comparison._merge == "both").all()),
            "matched_prediction_rows": len(predictions),
            "overlap_flags_agree_between_csvs": bool((matched_comparison.in_train83_prediction == matched_comparison.in_train83_matching).all()),
            "duplicate_matched_metadata_columns": int(matching[matching.valid_panel_col].panel_col.duplicated(keep=False).sum())},
        "column_order_audit": {
            "raw_header_file": HEADER.as_posix(),
            "raw_header_two_lines_sha256": hashlib.sha256(first + second).hexdigest(),
            "header_ecotype_ids": header_ids.tolist(),
            "training_genotype_columns": train_cols.tolist(),
            "raw_header_train_ids_equal_phenotype_ids_in_order": bool(np.array_equal(header_ids[train_cols], pheno.ecotype_id)),
            "raw_header_train_id_set_equals_phenotype_id_set": used_train_ids == phenotype_ids,
            "metadata_row_order_mismatch_count_out_of_1307": int((header_ids != panel.ecotype_id.to_numpy()).sum()),
            "sorted_metadata_rows_at_train_columns_mismatch_count_out_of_83": int((panel.iloc[train_cols].ecotype_id.to_numpy() != pheno.ecotype_id.to_numpy()).sum()),
            "genotypes_83_vs_raw_column_subset_sampled_snp_rows": len(sampled_rows),
            "genotypes_83_vs_raw_column_subset_sample_matches": sampled_genotype_match,
            "block_features_83_vs_full_raw_column_subset_max_abs_error": block_subset_error,
            "interpretation": "Training IDs align with the raw header. The 82/83 apparent training mismatch arises ONLY from indexing sorted metadata with raw columns, not from misordered training phenotypes.",
            "source_evidence": [
                {"file": (ROOT / "scripts/01_build_genotype_matrix.py").as_posix(), "lines": [54, 88],
                 "fact": "Genotype columns retain the raw header order; training genotype columns are reordered to phenotype IDs."},
                {"file": (ROOT / "scripts/01_build_genotype_matrix.py").as_posix(), "lines": [123, 126],
                 "fact": "accessions_all.csv is independently sorted by ecotype_id."},
                {"file": (ROOT / "scripts/05_build_haplotype_graph.py").as_posix(), "lines": [176, 183],
                 "fact": "dormancy_cols_in_1307.npy indexes the raw genotype header."},
                {"file": (ROOT / "scripts/legacy_projection.py").as_posix(), "lines": [96, 112],
                 "fact": "Name matching chooses metadata row positions; overlap flag uses exact names."},
                {"file": (ROOT / "scripts/legacy_projection.py").as_posix(), "lines": [142, 150],
                 "fact": "Metadata row positions are used directly as genotype/block columns without identity remapping."}]},
        "actual_intersections": {
            "frozen_matched_metadata_ecotype_ids_with_phenotype_training": records(matching[matching.in_train83_by_matched_ecotype_id][intersection_columns]),
            "actual_projected_genotype_ids_with_training_genotypes": records(matching[matching.in_train83_by_projected_ecotype_id][intersection_columns]),
            "original_name_flags_changed_by_matched_ecotype_id": records(matching[matching.valid_panel_col & (matching.in_train83 != matching.in_train83_by_matched_ecotype_id)][intersection_columns]),
            "original_name_flags_changed_by_projected_genotype_id": records(matching[matching.valid_panel_col & (matching.in_train83 != matching.in_train83_by_projected_ecotype_id)][intersection_columns])},
        "ambiguities": {"panel_normalized_name_collision_groups": len(collisions),
                        "panel_collision_details": collisions, "matched_dryad_ambiguous_details": ambiguous,
                        "dryad_normalized_duplicate_rows": records(duplicate_dryad[["nativename", "dsds50_dryad", "panel_col"]]),
                        "resolution": "The CSVs have no independent Dryad ecotype ID. Do not resolve same-name distinct ecotypes by arbitrary first/last choice; retain candidate lists and overlap bounds."},
        "frozen_prediction_descriptive_metrics_only": descriptive,
        "prediction_validity_warning": "205/206 frozen projections use a genotype identity different from the matched metadata identity. Regrouped metrics are forensic/descriptive only, not validated external accession performance. Correcting identity mapping and rerunning external projection is outside this audit.",
        "leakage_semantics": "Participation of external genotypes in SSL/preprocessing is transductive and is not itself phenotype leakage. Actual training-genotype reuse and incorrect accession identity invalidate independence/identity claims but do not prove Dryad phenotypes entered training. No such phenotype leakage is inferred here.",
        "outputs": [(OUT / name).as_posix() for name in (
            "dryad_identity_audit.json", "dryad_matching_identity_audit.csv",
            "dryad_predictions_identity_audit.csv", "training_identity_column_order.csv",
            "panel_identity_column_order.csv")],
    }


def main():
    if not OUT.is_dir():
        raise FileNotFoundError("Authorized results directory must already exist: " + str(OUT))
    protected = [
        ROOT / "scripts/legacy_rank1.py",
        EXPERIMENTS / "results/dog1_mechanism.json",
        ROOT / "scripts/P0_2_kernel_corrections.py", ROOT / "scripts/legacy_projection.py",
        ROOT / "scripts/01_build_genotype_matrix.py", ROOT / "scripts/05_build_haplotype_graph.py",
        PROCESSED / "accessions_all.csv", PROCESSED / "dormancy_cols_in_1307.npy",
        PROCESSED / "dormancy_phenotype.csv", PROCESSED / "block_features_83.npy",
        PROCESSED / "haplotype_blocks.csv", PROCESSED / "genotypes_83.npy",
        TABLES / "P2_dryad_matching.csv", TABLES / "P2_dryad_predictions.csv",
    ]
    before = {path.as_posix(): sha256(path) for path in protected}
    start = time.perf_counter()
    with threadpool_limits(limits=1):
        numerical = numerical_audit()
        identity = identity_audit()
    after = {path.as_posix(): sha256(path) for path in protected}
    if before != after:
        raise RuntimeError("A protected input changed during the audit; refusing final JSON reports")
    environment = {"python_executable": sys.executable, "python": sys.version,
                   "platform": platform.platform(), "numpy": np.__version__,
                   "scipy": scipy.__version__, "pandas": pd.__version__,
                   "sklearn": sklearn.__version__, "no_user_site": bool(sys.flags.no_user_site),
                   "dont_write_bytecode": bool(sys.dont_write_bytecode), "blas_thread_limit": 1}
    provenance = {"created_utc": datetime.now(timezone.utc).isoformat(),
                  "script": SCRIPT.as_posix(), "script_sha256": sha256(SCRIPT),
                  "environment": environment, "protected_source_sha256": before,
                  "protected_inputs_unchanged_after_audit": True,
                  "hash_scope_note": "The large raw genotype CSV is checked by its two header lines; full-panel arrays are read-only mmap inputs. Listed protected files receive full SHA256 before/after.",
                  "run_command": 'python -s -B "' + SCRIPT.as_posix() + '"'}
    numerical["provenance"] = provenance
    identity["provenance"] = provenance
    write_json("numerical_audit.json", numerical)
    write_json("dryad_identity_audit.json", identity)
    for block, result in numerical["selected_blocks"].items():
        print("block", block, result["roles"], "r old/fixed/generic",
              result["old_metrics"]["pearson_r_mean"], result["corrected_metrics"]["pearson_r_mean"],
              result["stable_generic_metrics"]["pearson_r_mean"],
              "rank old/fixed", result["old_rank"], result["corrected_rank"],
              "max prediction errors old/fixed vs generic",
              result["old_vs_stable_generic"]["max_abs_prediction_error_days"],
              result["corrected_vs_stable_generic"]["max_abs_prediction_error_days"])
    print("screen changes", numerical["screen_changes"])
    print("Dryad identity counts", json.dumps(identity["prediction_table"]))
    print("Original protected inputs unchanged:", before == after)
    print("Completed seconds:", time.perf_counter() - start)
    print("Reports:", (OUT / "numerical_audit.json").as_posix(),
          (OUT / "dryad_identity_audit.json").as_posix())


if __name__ == "__main__":
    main()
