"""Frozen-model validation on the legacy 29-accession external cohort.

Read-only inputs; exclusive-create outputs under results/sensitivity/.
No new data, outer-CV rerun, external tuning, calibration or sample removal.
The primary whole-panel-83 feature convention is retained, not fully fold-wise
inner-CV scaling. The shared linear feature kernel returns standardized
Z Z.T / p without an additional trace normalization, so computational trace
normalization requires lambda / trace_scale for linear models to retain the
published lambda units and exactly the same model."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

# Do not create imported-script __pycache__ outside the permitted directory.
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import stats
from sklearn.model_selection import KFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "results" / "sensitivity"
PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw" / "atpolydb"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import kernel_experiments as primary  # noqa: E402
from kernel_utils import (  # noqa: E402
    apply_kernel_centering_test,
    center_scale_kernel,
    feature_kernel_centered_scaled,
    solve_kernel_ridge_with_intercept,
)

LAMBDAS = np.logspace(-8, 2, 21)
GAMMAS = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0])
BOOTSTRAPS = 10000
BOOTSTRAP_SEED = 20261109
MODEL_NAMES = ("snp", "block", "local_gaussian", "mix_0.5_exploratory")
OUTPUT_NAMES = ("external29_primary_frozen.json", "external29_primary.npz",
                "external29_primary.json")


def maxdiff(a, b):
    assert a.shape == b.shape, (a.shape, b.shape)
    return float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))))


def array_hash(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(json.dumps(a.shape).encode())
    h.update(memoryview(a).cast("B"))
    return h.hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json_new(path, obj):
    assert path.parent.resolve() == OUT.resolve()
    with path.open("x", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def select_training_only(candidates, y_train83, rows=None):
    """Joint gamma/lambda selection by pooled 5-fold OOF MSE, strict < ties.

    Only the already-built wholepanel83 primary kernels and TRAINING labels
    enter this function. rows is used solely for one published outer-CV point.
    No foldwise feature restandardization, recentering or retracing is added.
    """
    assert y_train83.shape == (83,) and np.isfinite(y_train83).all()
    rows = np.arange(83) if rows is None else np.asarray(rows)
    y = y_train83[rows]
    splits = list(KFold(5, shuffle=True, random_state=0).split(y))
    curve, best = [], None
    for gamma, K83 in candidates:
        assert K83.shape == (83, 83)
        K = K83[np.ix_(rows, rows)]
        for lam in LAMBDAS:
            pred = np.empty(len(rows))
            for tr, va in splits:
                _, _, pred[va] = solve_kernel_ridge_with_intercept(
                    K[np.ix_(tr, tr)], y[tr], lam, K[np.ix_(va, tr)]
                )
            mse = float(np.mean((pred - y) ** 2))
            entry = {"gamma": None if gamma is None else float(gamma),
                     "lambda": float(lam), "train83_cv_mse": mse}
            curve.append(entry)
            if best is None or mse < best["train83_cv_mse"]:
                best = dict(entry)
                best_pred = pred.copy()
    return best, curve, best_pred


def train_kernels(data):
    """Reuse the exact feature dtypes and primary normalization, not prose."""
    candidates, parts, audit = {}, {}, {}
    for name, key in (("snp", "X_snp"), ("block", "block_mean")):
        X = data[key]
        K, Z, mu, sd = feature_kernel_centered_scaled(X)
        Kct, row, total, scale = center_scale_kernel(K.astype(np.float64))
        # The shared feature kernel has no additional trace normalization; preserve its K.
        candidates[name] = [(None, K)]
        parts[name] = {"K_primary": K, "K_computational": Kct, "Z": Z,
                       "mu": mu, "sd": sd, "row": row, "total": float(total),
                       "scale": float(scale)}
        manual_mu, manual_sd = X.mean(0), X.std(0, ddof=0)
        manual_sd = np.where(manual_sd < 1e-8, 1.0, manual_sd)
        manual_Z = (X - manual_mu) / manual_sd
        ref = manual_Z @ manual_Z.T / X.shape[1]
        assert maxdiff(K, ref) == 0.0
        audit[name] = {
            "shape": list(X.shape), "feature_dtype": str(X.dtype),
            "constant_columns_train83": int((X.std(0, ddof=0) < 1e-8).sum()),
            "primary_K_vs_independent_formula_max_abs_diff": maxdiff(K, ref),
            "primary_trace_per_83": float(np.trace(K) / 83),
            "computational_center_trace_scale": float(scale),
            "center_trace_K_vs_primary_K_max_abs_diff": maxdiff(Kct, K),
            "reversed_center_trace_vs_primary_K_max_abs_diff": maxdiff(
                Kct * scale + row[:, None] + row[None, :] - total, K),
            "computational_lambda_rule": "selected_primary_lambda / trace_scale",
        }
    local, gaussian_parts, local_audit = [], {}, {}
    for g in GAMMAS:
        # Same per-block standardization loop and H K H / (trace / 83) as primary.
        Kraw = primary.perblock_rbf(data["block_mean"], float(g))
        K = primary.center_trace(Kraw)
        _, row, total, scale = center_scale_kernel(Kraw)
        Kcheck = apply_kernel_centering_test(Kraw, row, total, scale)
        assert maxdiff(K, Kcheck) < 1e-11
        local.append((float(g), K))
        gaussian_parts[float(g)] = {"row": row, "total": float(total),
                                    "scale": float(scale), "K_raw": Kraw}
        local_audit[f"{g:g}"] = {
            "primary_K_vs_shared_center_formula_max_abs_diff": maxdiff(K, Kcheck),
            "trace_scale_from_train83": float(scale),
            "primary_trace_per_83": float(np.trace(K) / 83),
        }
    candidates["local_gaussian"] = local
    Kmix = 0.5 * parts["snp"]["K_primary"] + 0.5 * parts["block"]["K_primary"]
    Kct, row, total, scale = center_scale_kernel(Kmix.astype(np.float64))
    candidates["mix_0.5_exploratory"] = [(None, Kmix)]
    parts["mix_0.5_exploratory"] = {
        "K_primary": Kmix, "K_computational": Kct, "row": row,
        "total": float(total), "scale": float(scale)}
    audit["local_gaussian"] = local_audit
    audit["source_semantics"] = (
        "kernel_utils.feature_kernel_centered_scaled: column mean/std ddof=0, "
        "epsilon=1e-8, Z@Z.T/p; no explicit H or trace step. All linear columns "
        "are variable here, so trace/83 is approximately 1 (float32 SNP). "
        "Gaussian uses primary.center_trace. Linear inference applies H/trace "
        "and lambda/scale, preserving primary regularization units, not silently "
        "changing the model. No feature stats include the 29 accessions."
    )
    return candidates, parts, gaussian_parts, audit


def published_training_checks(candidates, y):
    """One rep0/outer-fold0 parameter point, NOT 50 outer-fold reruns."""
    tr, te = next(KFold(5, shuffle=True, random_state=0).split(y))
    result, store = {}, {}
    with np.load(OUT / "kernel_experiments_preds.npz") as saved:
        np.testing.assert_array_equal(saved["y"], y)
        for name, saved_name in (("snp", "snp_preds"), ("block", "block_preds"),
                                 ("local_gaussian", "block_rbf_perblock_preds")):
            choice, _, _ = select_training_only(candidates[name], y, rows=tr)
            K = dict(candidates[name])[choice["gamma"]]
            _, _, pred = solve_kernel_ridge_with_intercept(
                K[np.ix_(tr, tr)], y[tr], choice["lambda"], K[np.ix_(te, tr)]
            )
            diff = maxdiff(pred, saved[saved_name][0, te])
            # Numerical tolerance covers BLAS/float32 multiplication differences.
            assert diff < 2e-3, (name, diff)
            result[name] = {"rep": 0, "outer_fold": 0, "n_outer_train": len(tr),
                            "n_outer_test": len(te), "joint_selected": choice,
                            "saved_predictions_max_abs_diff": diff,
                            "reference": str(OUT / "kernel_experiments_preds.npz")}
            store[name + "_published_point_prediction"] = pred
    store["published_point_train_rows"] = tr
    store["published_point_test_rows"] = te
    return result, store


def assert_exact(name, a, b):
    assert a.shape == b.shape, (name, a.shape, b.shape)
    diff = maxdiff(a, b)
    assert diff == 0.0, (name, diff)
    return diff


def verify_order_and_load_external(data):
    """Raw-header column mapping, raw-order 29 selection, cache reconstruction.

    NPY caches contain no IDs: reconstruct genotypes directly from the full
    encoded panel mapped by raw header; reconstruct block channel 0 BOTH by
    panel slicing and by original block SNP ranges/panel minor coding.
    """
    pheno = pd.read_csv(PROCESSED / "dormancy_phenotype.csv")
    train_ids = pheno["ecotype_id"].to_numpy(np.int64)
    assert_exact("train labels", pheno["dsds50"].to_numpy(float), data["y"])
    raw = pd.read_csv(RAW / "phenotype_published_raw.tsv", sep="\t", encoding="latin1")
    # Retain source row order; never sort or filter for metrics.
    ext = raw[raw["279_DSDS50"].notna() & ~raw["8_Seed Dormancy"].notna()].copy()
    ext_ids = ext["ecotype id"].to_numpy(np.int64)
    assert len(ext_ids) == 29 and len(train_ids) == 83
    assert len(np.unique(ext_ids)) == 29 and len(np.unique(train_ids)) == 83
    assert not set(ext_ids) & set(train_ids)
    raw_train = raw[raw["8_Seed Dormancy"].notna()].sort_values("ecotype id")
    np.testing.assert_array_equal(raw_train["ecotype id"].to_numpy(np.int64), train_ids)
    np.testing.assert_array_equal(raw_train["8_Seed Dormancy"].to_numpy(float), data["y"])
    with (RAW / "call_method_75" / "call_method_75_TAIR9.csv").open("rb") as f:
        f.readline()
        header = f.readline().decode().strip().split(",")
    header_ids = np.array([int(x) for x in header[2:]], dtype=np.int64)
    assert len(header_ids) == 1307 and len(np.unique(header_ids)) == 1307
    mapping = {eid: col for col, eid in enumerate(header_ids)}
    trcols = np.array([mapping[eid] for eid in train_ids])
    tecols = np.array([mapping[eid] for eid in ext_ids])
    np.testing.assert_array_equal(header_ids[trcols], train_ids)
    np.testing.assert_array_equal(header_ids[tecols], ext_ids)
    np.testing.assert_array_equal(np.load(PROCESSED / "dormancy_cols_in_1307.npy"), trcols)
    all_codes = np.load(PROCESSED / "genotypes_all_1307.npy", mmap_mode="r")
    assert all_codes.shape == (214051, 1307)
    reconstructed = np.asarray(all_codes[:, tecols])
    ext_cache = np.load(PROCESSED / "genotypes_external29.npy")
    genotype_diff = assert_exact("external genotype raw-header reconstruction", reconstructed, ext_cache)
    train_diff = assert_exact("train genotype raw-header mapping", np.asarray(all_codes[:, trcols]), data["codes"])
    snp = pd.read_csv(PROCESSED / "snp_metadata.csv")
    keep = np.flatnonzero((snp["maf_83"].to_numpy() > 0.05) & (snp["minor_code"].to_numpy() < 4))
    minor = snp["minor_code"].to_numpy()[keep]
    Xte = (ext_cache[keep, :] == minor[:, None]).astype(np.float32).T
    assert_exact("Xte keep/minor reconstruction", Xte,
                 (reconstructed[keep, :] == minor[:, None]).astype(np.float32).T)
    assert_exact("Xtrain keep/minor reproduction", data["X_snp"],
                 (data["codes"][keep, :] == minor[:, None]).astype(np.float32).T)
    block_cache = np.load(PROCESSED / "block_features_external29.npy", mmap_mode="r")
    assert block_cache.shape == (29, 8589, 4)
    Bte_cache = np.asarray(block_cache[:, :, 0], dtype=np.float64)
    assert Bte_cache.shape == (29, 8589) and data["block_mean"].shape == (83, 8589)
    all_blocks = np.load(PROCESSED / "block_features.npy", mmap_mode="r")
    assert all_blocks.shape == (1307, 8589, 4)
    Bte = np.asarray(all_blocks[tecols, :, 0], dtype=np.float64)
    block_diff = assert_exact("external block raw-header slicing", Bte, Bte_cache)
    block_train_diff = assert_exact("train block raw-header slicing", data["block_mean"],
                                    np.asarray(all_blocks[trcols, :, 0], dtype=np.float64))
    panel_minor = primary.panel_minor_codes()
    with np.load(PROCESSED / "block_snp_map.npz") as ranges:
        starts, ends = ranges["snp_start"], ranges["snp_end"]
        assert len(starts) == 8589 and starts[0] == 0 and ends[-1] == len(snp)
        np.testing.assert_array_equal(starts[1:], ends[:-1])
        B_rebuilt = np.stack([
            (reconstructed[s:e] == panel_minor[s:e, None]).astype(np.float32).mean(0)
            for s, e in zip(starts, ends)
        ], axis=1).astype(np.float64)
    block_rebuilt_diff = assert_exact("block independently rebuilt from genotype", B_rebuilt, Bte_cache)
    yte = ext["279_DSDS50"].to_numpy(float)
    assert np.isfinite(Xte).all() and np.isfinite(Bte).all() and np.isfinite(yte).all()
    # Legacy IDs are an additional independent order/phenotype reference.
    with np.load(ROOT / "results" / "tables" / "external_validation_predictions.npz") as old:
        np.testing.assert_array_equal(old["ecotype_id"], ext_ids)
        np.testing.assert_array_equal(old["observed"], yte)
    audit = {
        "rule": "279_DSDS50.notna() & ~8_Seed Dormancy.notna(); raw TSV row order",
        "n_train": 83, "n_external": 29, "train_external_id_overlap": 0,
        "cache_has_id_metadata": False,
        "fallback": "raw second-header ecotype ID -> full-panel column; arrays rebuilt and compared exactly",
        "train_ids": train_ids.tolist(), "external_ids": ext_ids.tolist(),
        "external_source_tsv_row_indices": ext.index.tolist(),
        "train_raw_header_columns": trcols.tolist(), "external_raw_header_columns": tecols.tolist(),
        "external_genotype_reconstruction_max_abs_diff": genotype_diff,
        "train_genotype_mapping_max_abs_diff": train_diff,
        "external_block_panel_slice_max_abs_diff": block_diff,
        "train_block_panel_slice_max_abs_diff": block_train_diff,
        "external_block_independent_genotype_rebuild_max_abs_diff": block_rebuilt_diff,
        "legacy_ids_and_y_exact_match": True,
        "snp_keep_rule": "maf_83 > 0.05 and minor_code < 4 (training83 definitions)",
        "n_retained_snps": len(keep), "block_mean_external_shape": list(Bte.shape),
        "block_allele_convention": "original panel1307 minor coding; no recoding/scaling on external phenotypes",
    }
    return train_ids, ext_ids, yte, Xte, Bte, keep, minor, audit


def gaussian_cross(Bte, mu, sd, Ztrain, gamma):
    # Average of 1-D kernels, NOT exp(-gamma * average squared distance).
    Kcross = np.zeros((len(Bte), len(Ztrain)), dtype=np.float64)
    for b in range(Bte.shape[1]):
        zte = (Bte[:, b] - mu[b]) / sd[b]
        Kcross += np.exp(-gamma * (zte[:, None] - Ztrain[None, :, b]) ** 2)
    return Kcross / Bte.shape[1]


def project_frozen(data, candidates, parts, gaussian_parts, choices, Xte, Bte):
    predictions, audit, store, crosses = {}, {}, {}, {}
    for name, Xtest in (("snp", Xte), ("block", Bte)):
        info = parts[name]
        Ztest = (Xtest - info["mu"]) / info["sd"]
        crosses[name] = Ztest @ info["Z"].T / info["Z"].shape[1]
        store[name + "_feature_mean_train83"] = info["mu"]
        store[name + "_feature_sd_train83"] = info["sd"]
    crosses["mix_0.5_exploratory"] = 0.5 * crosses["snp"] + 0.5 * crosses["block"]
    for name in MODEL_NAMES:
        choice = choices[name]
        Kprimary = dict(candidates[name])[choice["gamma"]]
        if name == "local_gaussian":
            gamma = choice["gamma"]
            info = gaussian_parts[gamma]
            raw_cross = gaussian_cross(Bte, parts["block"]["mu"], parts["block"]["sd"],
                                       parts["block"]["Z"], gamma)
            Kcomp, effective_lambda = Kprimary, choice["lambda"]
        else:
            info = parts[name]
            raw_cross = crosses[name]
            Kcomp = info["K_computational"]
            effective_lambda = choice["lambda"] / info["scale"]
        Kcross = apply_kernel_centering_test(raw_cross, info["row"], info["total"], info["scale"])
        alpha, intercept, pred = solve_kernel_ridge_with_intercept(Kcomp, data["y"], effective_lambda, Kcross)
        predictions[name] = pred
        # Direct original-primary projection is mathematically equivalent.
        reference_cross = Kcross if name == "local_gaussian" else raw_cross
        _, _, direct = solve_kernel_ridge_with_intercept(Kprimary, data["y"], choice["lambda"], reference_cross)
        err = maxdiff(pred, direct)
        assert err < 2e-3, (name, err)
        assert np.isfinite(pred).all()
        audit[name] = {"primary_lambda": choice["lambda"], "computational_lambda": float(effective_lambda),
                       "intercept": float(intercept), "sum_alpha_abs": float(abs(alpha.sum())),
                       "trace_conversion_prediction_max_abs_diff": err,
                       "cross_rowmean_max_abs": float(abs(Kcross.mean(1)).max())}
        store[name + "_prediction"] = pred
        store[name + "_alpha"] = alpha
        store[name + "_intercept"] = np.array(intercept)
        store[name + "_K_train83_primary"] = Kprimary
        store[name + "_K_train83_center_trace"] = Kcomp
        store[name + "_K_test29_train83_center_trace"] = Kcross
        store[name + "_kernel_row_mean_train83"] = info["row"]
        store[name + "_kernel_total_mean_train83"] = np.array(info["total"])
        store[name + "_kernel_trace_scale_train83"] = np.array(info["scale"])
        store[name + "_selected_lambda"] = np.array(choice["lambda"])
        store[name + "_computational_lambda"] = np.array(effective_lambda)
        store[name + "_selected_gamma"] = np.array(np.nan if choice["gamma"] is None else choice["gamma"])
    return predictions, audit, store


def unitchecks():
    """Training-only synthetic checks; no real external labels are loaded."""
    rng = np.random.default_rng(20261108)
    X, T, y = rng.normal(size=(24, 7)), rng.normal(size=(5, 7)), rng.normal(size=24)
    R, cross = X @ X.T / 7, T @ X.T / 7
    K, row, total, scale = center_scale_kernel(R)
    C = apply_kernel_centering_test(cross, row, total, scale)
    H = np.eye(24) - np.ones((24, 24)) / 24
    Z, Zte = X - X.mean(0), T - X.mean(0)
    diffs = {
        "train_HKH": maxdiff(K, H @ R @ H / scale),
        "cross_feature_formula": maxdiff(C, Zte @ Z.T / 7 / scale),
        "cross_train_identity": maxdiff(apply_kernel_centering_test(R, row, total, scale), K),
        "reverse_cross_transpose_formula": maxdiff(C.T, (cross.T - cross.mean(1)[None, :] - row[:, None] + total) / scale),
        "reverse_train_centering": maxdiff(K * scale + row[:, None] + row[None, :] - total, R),
    }
    tr, va = next(KFold(5, shuffle=True, random_state=0).split(y))
    A, Q, lam = K[np.ix_(tr, tr)], K[np.ix_(va, tr)], 0.3
    alpha, b, pred = solve_kernel_ridge_with_intercept(A, y[tr], lam, Q)
    system = np.block([[A + lam * np.eye(len(tr)), np.ones((len(tr), 1))],
                       [np.ones((1, len(tr))), np.zeros((1, 1))]])
    exact = np.linalg.solve(system, np.r_[y[tr], 0.0])
    diffs["explicit_intercept_vs_augmented_system_prediction"] = maxdiff(pred, Q @ exact[:-1] + exact[-1])
    diffs["explicit_intercept_sum_alpha"] = float(abs(alpha.sum()))
    old = Q @ np.linalg.solve(A + lam * np.eye(len(tr)), y[tr] - y[tr].mean()) + y[tr].mean()
    old_err = maxdiff(pred, old)
    assert old_err > 1e-5  # Ensures this test detects mean-only intercept misuse.
    _, Zp, mu, sd, Cp, Ztp = feature_kernel_centered_scaled(X, T)
    one = feature_kernel_centered_scaled(X, T[:1])[4]
    diffs["test_batch_invariance_no_test_statistics"] = maxdiff(Cp[:1], one)
    diffs["standardization_ddof0"] = maxdiff(sd, X.std(0, ddof=0))
    for name, err in diffs.items():
        assert err < 1e-10, (name, err)
    return {"passed": True, "max_abs_differences": diffs,
            "mean_only_intercept_detected_prediction_difference": old_err}


def metrics_and_intervals(predictions, y):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, len(y), size=(BOOTSTRAPS, len(y)))
    results, store = {}, {"bootstrap_indices": idx}
    for name, pred in predictions.items():
        pearson, spearman = stats.pearsonr(pred, y), stats.spearmanr(pred, y)
        r, rho = float(pearson.statistic), float(spearman.statistic)
        z = np.arctanh(np.clip(r, -1 + 1e-15, 1 - 1e-15))
        fisher = np.tanh(z + np.array([-1, 1]) * stats.norm.ppf(0.975) / np.sqrt(len(y) - 3))
        pb, yb = pred[idx], y[idx]
        pc, yc = pb - pb.mean(1, keepdims=True), yb - yb.mean(1, keepdims=True)
        den = np.sqrt((pc ** 2).sum(1) * (yc ** 2).sum(1))
        boot_r = np.divide((pc * yc).sum(1), den, out=np.full(BOOTSTRAPS, np.nan), where=den > 0)
        pr, yr = stats.rankdata(pb, axis=1), stats.rankdata(yb, axis=1)
        pr, yr = pr - pr.mean(1, keepdims=True), yr - yr.mean(1, keepdims=True)
        denom_ranks = np.sqrt((pr ** 2).sum(1) * (yr ** 2).sum(1))
        boot_rho = np.divide((pr * yr).sum(1), denom_ranks, out=np.full(BOOTSTRAPS, np.nan), where=denom_ranks > 0)
        boot_mae, boot_rmse = abs(pb - yb).mean(1), np.sqrt(((pb - yb) ** 2).mean(1))
        boots = {"pearson": boot_r, "spearman": boot_rho, "mae": boot_mae, "rmse": boot_rmse}
        ci = {metric: np.nanquantile(vals, [0.025, 0.975]).tolist() for metric, vals in boots.items()}
        for metric, vals in boots.items():
            store[name + "_bootstrap_" + metric] = vals
        results[name] = {
            "n": len(y), "pearson": r, "pearson_p_two_sided": float(pearson.pvalue),
            "spearman": rho, "spearman_p_two_sided": float(spearman.pvalue),
            "mae": float(abs(pred - y).mean()), "rmse": float(np.sqrt(((pred - y) ** 2).mean())),
            "pearson_fisher_95ci": fisher.tolist(), "paired_bootstrap_percentile_95ci": ci,
            "bootstrap_valid_correlation_draws": int(np.isfinite(boot_r).sum()),
            "positive_pearson_signal_nominal_p_lt_0.05": bool(r > 0 and pearson.pvalue < 0.05),
            "positive_spearman_signal_nominal_p_lt_0.05": bool(rho > 0 and spearman.pvalue < 0.05),
            "pearson_bootstrap_ci_excludes_zero_positive": bool(ci["pearson"][0] > 0),
            "spearman_bootstrap_ci_excludes_zero_positive": bool(ci["spearman"][0] > 0),
        }
    # State nominal AND multiplicity-adjusted signal, not an externally selected winner.
    tests = [(name, metric, results[name][metric + "_p_two_sided"])
             for name in MODEL_NAMES for metric in ("pearson", "spearman")]
    tests.sort(key=lambda entry: entry[2])
    adjusted = 0.0
    for i, (name, metric, p) in enumerate(tests):
        adjusted = min(1.0, max(adjusted, (len(tests) - i) * p))
        results[name][metric + "_p_holm_8_tests"] = adjusted
        results[name][metric + "_significant_holm_0.05"] = bool(adjusted < 0.05)
    return results, store


def legacy_comparison(ids, y):
    table = ROOT / "results" / "tables"
    summary = json.loads((table / "external_validation.json").read_text(encoding="utf-8"))
    models = {}
    with np.load(table / "external_validation_predictions.npz") as saved:
        np.testing.assert_array_equal(saved["ecotype_id"], ids)
        np.testing.assert_array_equal(saved["observed"], y)
        for name, key in (("SNP-GBLUP", "SNP_GBLUP"), ("block-kernel", "block_kernel"),
                          ("dual-SNP-block", "dual_SNP_block")):
            p = saved[key]
            actual = {"pearson": float(stats.pearsonr(p, y).statistic),
                      "spearman": float(stats.spearmanr(p, y).statistic),
                      "mae": float(abs(p - y).mean()), "rmse": float(np.sqrt(((p - y) ** 2).mean()))}
            for metric in ("pearson", "spearman", "mae"):
                assert abs(actual[metric] - summary["models"][name][metric]) < 1e-8
            models[name] = {**actual, "legacy_lambda": summary["models"][name]["lambda"]}
    return {"classification": "LEGACY, NOT same-primary validation",
            "differences": ["old lambda grid 1e-4..3", "mean-only intercept",
                            "unstandardized SNP", "no local Gaussian model"],
            "ids_and_observations_exact_match": True, "models": models,
            "not_used_for_any_selection": True, "source": str(table / "external_validation.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unitchecks-only", action="store_true")
    args = parser.parse_args()
    checks = unitchecks()
    print("Training-only unitchecks:", json.dumps(checks), flush=True)
    if args.unitchecks_only:
        return
    assert HERE.name == "sensitivity"
    assert OUT.is_dir() and primary.OUT.resolve() == OUT.resolve()
    for name in OUTPUT_NAMES:
        assert not (OUT / name).exists(), f"Refusing to overwrite: {OUT / name}"
    np.testing.assert_array_equal(LAMBDAS, primary.LAMBDAS)
    np.testing.assert_array_equal(GAMMAS, primary.GAMMAS)
    t0 = time.time()
    data = primary.load_data()
    assert data["y"].shape == (83,)
    candidates, parts, gaussian_parts, normalization = train_kernels(data)
    selected, curves, store = {}, {}, {}
    for name in MODEL_NAMES:
        selected[name], curves[name], oof = select_training_only(candidates[name], data["y"])
        store[name + "_train83_selection_oof"] = oof
        print("Train83 selected", name, selected[name], flush=True)
    published_checks, published_store = published_training_checks(candidates, data["y"])
    store.update(published_store)
    print("Published parameter-point reproduction:", json.dumps(published_checks), flush=True)
    frozen = {
        "external_y_not_loaded_when_frozen": True,
        "selected_hyperparameters": selected,
        "selection": "train83 only; pooled OOF MSE; KFold(5, shuffle=True, random_state=0); strict <",
        "lambda_grid": LAMBDAS.tolist(), "local_gamma_grid": GAMMAS.tolist(),
        "feature_convention": "wholepanel83 mean/std(ddof=0) and kernel construction BEFORE inner slicing; NOT fully foldwise",
        "mix_weight_snp": 0.5, "mix_weight_block": 0.5,
        "train_y_sha256": array_hash(data["y"]),
        "training_K_sha256": {name: array_hash(dict(candidates[name])[selected[name]["gamma"]]) for name in MODEL_NAMES},
        "bootstrap_draws": BOOTSTRAPS, "bootstrap_seed": BOOTSTRAP_SEED,
    }
    freeze_path = OUT / OUTPUT_NAMES[0]
    save_json_new(freeze_path, frozen)
    frozen_digest = file_hash(freeze_path)
    selected_snapshot = json.dumps(selected, sort_keys=True)
    # Only now load the external table, labels and genotypes; settings are frozen.
    train_ids, ext_ids, yte, Xte, Bte, keep, minor, ordering = verify_order_and_load_external(data)
    pred, projection_audit, projection_store = project_frozen(data, candidates, parts, gaussian_parts, selected, Xte, Bte)
    store.update(projection_store)
    evaluated, bootstrap_store = metrics_and_intervals(pred, yte)
    store.update(bootstrap_store)
    legacy = legacy_comparison(ext_ids, yte)
    assert json.dumps(selected, sort_keys=True) == selected_snapshot
    assert file_hash(freeze_path) == frozen_digest
    assert len(yte) == 29 and all(len(p) == 29 for p in pred.values())
    consumed = [PROCESSED / name for name in (
        "dormancy_phenotype.csv", "genotypes_83.npy", "snp_metadata.csv",
        "block_features_83.npy", "genotypes_external29.npy", "block_features_external29.npy",
        "genotypes_all_1307.npy", "block_features.npy", "block_snp_map.npz",
        "dormancy_cols_in_1307.npy", "graph_embeddings_83.npy", "sample_embeddings_1307.npy")]
    consumed += [RAW / "phenotype_published_raw.tsv",
                 RAW / "call_method_75" / "call_method_75_TAIR9.csv",
                 HERE / "kernel_experiments.py", Path(__file__).resolve(),
                 ROOT / "scripts" / "kernel_utils.py",
                 OUT / "kernel_experiments_preds.npz",
                 ROOT / "results" / "tables" / "external_validation.json",
                 ROOT / "results" / "tables" / "external_validation_predictions.npz"]
    hashes = {str(path): {"sha256": file_hash(path), "bytes": path.stat().st_size} for path in consumed}
    arrays = {"train_y": data["y"], "external_y": yte, "train_ids": train_ids,
              "external_ids": ext_ids, "X_snp_train83": data["X_snp"], "X_snp_external29": Xte,
              "block_mean_train83": data["block_mean"], "block_mean_external29": Bte,
              "snp_keep": keep, "snp_minor_train83": minor}
    store.update({"y_train83": data["y"], "y_external29": yte, "train_ecotype_ids": train_ids,
                  "external_ecotype_ids": ext_ids, "snp_keep_indices": keep, "snp_minor_train83": minor,
                  "lambda_grid": LAMBDAS, "local_gamma_grid": GAMMAS,
                  "selected_hyperparameters_json": np.array(selected_snapshot),
                  "no_external_tuning": np.array(True), "frozen_json_sha256": np.array(frozen_digest)})
    npz_path = OUT / OUTPUT_NAMES[1]
    with npz_path.open("xb") as f:
        np.savez_compressed(f, **store)
    output = {
        "purpose": "Correct legacy29 protocol mismatch using published primary kernels and explicit intercept",
        "n_train": 83, "n_external": 29, "external_trait": "279_DSDS50",
        "interpretation": "Same legacy cohort, different phenotyping experiment; not a new fully independent discovery. MAE/RMSE in raw days are cross-protocol, uncalibrated.",
        "model_roles": {"snp": "primary SNP baseline", "block": "primary linear block",
                        "local_gaussian": "primary local perblock Gaussian", "mix_0.5_exploratory": "fixed exploratory comparator, NOT externally optimized"},
        "frozen_protocol": frozen, "train83_cv_candidate_scores": curves,
        "no_external_tuning_assertions": {
            "selection_functions_receive_only_train83_y_and_training_kernels": True,
            "model_settings_written_before_external_table_and_y_loaded": True,
            "frozen_file_hash_unchanged_after_external_evaluation": True,
            "hyperparameters_unchanged_after_external_evaluation": True,
            "all_29_kept_no_outcome_based_exclusions": True,
            "no_lambda_grid_gamma_grid_weight_changes": True,
            "no_outer50_rerun": True,
        },
        "ordering_validation": ordering, "normalization_audit": normalization,
        "published_training_parameter_point_checks": published_checks,
        "projection_checks": projection_audit, "training_only_unitchecks": checks,
        "models": evaluated, "legacy_comparison": legacy,
        "train_mean_external_baseline": {"prediction": float(data["y"].mean()),
            "mae": float(abs(data["y"].mean() - yte).mean()),
            "rmse": float(np.sqrt(((data["y"].mean() - yte) ** 2).mean()))},
        "inference": {"fisher": "Pearson 95%, normal Fisher z with SE=1/sqrt(29-3)",
            "bootstrap": "10000 paired accession draws, common indices across all models, percentile 95%; conditional on frozen fit, not training uncertainty",
            "spearman_p": "SciPy two-sided asymptotic p; small-n uncertainty also reported by paired bootstrap",
            "multiplicity": "Nominal p and Holm across all four models x two correlation tests (8); no winner selection"},
        "observations_and_predictions": {
            "train_ecotype_ids": train_ids.tolist(), "y_train83": data["y"].tolist(),
            "external_ecotype_ids": ext_ids.tolist(), "y_external29": yte.tolist(),
            "predictions": {name: values.tolist() for name, values in pred.items()},
        },
        "input_file_hashes": hashes, "input_array_sha256": {name: array_hash(a) for name, a in arrays.items()},
        "artifacts": {"script": str(Path(__file__).resolve()), "frozen_json": str(freeze_path),
                      "frozen_json_sha256": frozen_digest, "npz": str(npz_path), "npz_sha256": file_hash(npz_path)},
        "reproducibility": {"python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
            "execution": "python -s -B; OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=OMP_NUM_THREADS=1"},
        "elapsed_seconds": time.time() - t0,
    }
    save_json_new(OUT / OUTPUT_NAMES[2], output)
    for name, res in evaluated.items():
        print(f"External29 {name}: r={res['pearson']:.6f}, rho={res['spearman']:.6f}, "
              f"MAE={res['mae']:.4f}, RMSE={res['rmse']:.4f}, "
              f"p_r={res['pearson_p_two_sided']:.6g}, p_rho={res['spearman_p_two_sided']:.6g}", flush=True)
    print("All ID/cache reconstruction checks exact; all 29 retained.", flush=True)
    print(f"Done {output['elapsed_seconds']:.1f}s -> {OUT / OUTPUT_NAMES[2]}", flush=True)


if __name__ == "__main__":
    main()
