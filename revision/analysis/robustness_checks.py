"""Bounded, training-refitted sensitivity checks; only this module's outputs.

Run with python -s -B: kernel_benchmark must import the compatible scientific
stack before the user CUDA torch. All main-analysis files remain read-only.
"""
from __future__ import annotations

import kernel_benchmark as core  # Must precede numpy, sklearn and torch imports.

import argparse
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
from scipy import linalg, stats
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

VERSION = 1
SEED = 20261118
K_VALUES = (6, 8, 10)
FAMILIES = ("block_fixed_25", "block_selected", "local_selected", "snp", "snp_rkhs")
BASELINES = ("train_mean", "train_median")
COMPARISONS = (("regional_minus_snp", "block_selected", "snp"),
               ("local_minus_regional", "local_selected", "block_selected"))
OWN_NAMES = {
    "grouped_strict.json", "grouped_strict.npz",
    "training_refit_robustness.json", "training_refit_robustness.npz",
    "extreme_removed_identity.json", "robustness_selftests.json",
    "conditional_uncertainty.json", "conditional_uncertainty.npz",
    "robustness_run_status.json",
}


def own_path(name):
    if name not in OWN_NAMES:
        raise ValueError(f"Output is not owned by robustness_checks: {name}")
    path = core.RESULTS / name
    if not path.parent.is_dir():
        raise FileNotFoundError(path.parent)
    return path


def json_save(name, value):
    core.save_json(own_path(name), value)


def npz_save(name, **arrays):
    path = own_path(name)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    tmp.replace(path)


def digest_array(array):
    array = np.ascontiguousarray(array)
    if array.dtype.kind == "O":
        return hashlib.sha256(json.dumps(array.tolist(), sort_keys=True).encode()).hexdigest()
    return hashlib.sha256(array.tobytes()).hexdigest()


def provenance(codes, y, ids, groups):
    return {
        "version": VERSION, "n": len(y), "marker_n": int(codes.shape[1]),
        "ids_sha256": digest_array(ids), "outcomes_sha256": digest_array(y),
        "codes_sha256": digest_array(codes),
        "windows_sha256": {str(k): digest_array(v) for k, v in groups.items()},
        "core_sha256": hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
        "protocol": core.PROTOCOL, "device": str(core.DEVICE),
        "local_dtype": "float32", "local_accumulation": "float64",
        "software": {"numpy": np.__version__, "torch": core.torch.__version__,
                     "scipy": core.scipy.__version__},
    }


def signature(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def load_checkpoint(json_name, npz_name, config, initial, arrays):
    jp, ap = own_path(json_name), own_path(npz_name)
    if jp.exists() != ap.exists():
        raise RuntimeError(f"Incomplete checkpoint pair: {jp} and {ap}; no automatic overwrite")
    if not jp.exists():
        initial["config"] = config
        initial["config_sha256"] = signature(config)
        return initial, arrays
    state = json.loads(jp.read_text(encoding="utf-8"))
    if state.get("config_sha256") != signature(config):
        raise RuntimeError(f"Checkpoint configuration changed: {jp}")
    with np.load(ap, allow_pickle=False) as saved:
        restored = {name: saved[name].copy() for name in saved.files}
    if set(restored) != set(arrays):
        raise RuntimeError(f"Checkpoint array schema changed: {ap}")
    for name, template in arrays.items():
        if restored[name].shape != template.shape:
            raise RuntimeError(f"Checkpoint array shape changed: {name}")
    return state, restored


def checkpoint(json_name, npz_name, state, arrays):
    npz_save(npz_name, **arrays)
    json_save(json_name, state)


def scored(y, prediction):
    return {"n": int(len(y)), **core.metrics(y, prediction)}


def group_partitions(n):
    """Old outcome-independent grouping only; never a model representation."""
    path = core.DATA / "kinship_83.npy"
    kinship = np.load(path, allow_pickle=False)
    if kinship.shape != (n, n) or not np.isfinite(kinship).all():
        raise ValueError("Invalid kinship83 grouping input")
    if not np.allclose(kinship, kinship.T, atol=1e-10, rtol=1e-10):
        raise ValueError("Kinship83 must be symmetric in the original accession order")
    if kinship.max() <= 0:
        raise ValueError("Kinship83 maximum must be positive")
    distance = 1 - kinship / kinship.max()
    np.fill_diagonal(distance, 0)
    condensed = distance[np.triu_indices_from(distance, k=1)]
    tree = linkage(condensed, method="average")
    labels = np.asarray([fcluster(tree, t=k, criterion="maxclust") for k in K_VALUES])
    if any(len(np.unique(row)) != k for k, row in zip(K_VALUES, labels)):
        raise ValueError("Requested 6/8/10 maxclust partitions were not obtained")
    return labels, {
        "kinship_path": str(path.resolve()), "kinship_sha256": digest_array(kinship),
        "distance": "1 - kinship / kinship.max(); diagonal zero; upper-triangle condensed",
        "linkage": "average", "cut": "fcluster(maxclust)",
        "role": "Full-panel, outcome-independent holdout DESIGN only; not a training kernel or feature matrix",
        "ordering": "Original kinship83 order, matched to the audited load_dormancy accession order",
    }


def split_record(train, test, ids, labels=None, positions=None):
    if np.intersect1d(ids[train], ids[test]).size:
        raise AssertionError("Canonical accession leakage between training and validation/test")
    record = {
        "train_indices": train.tolist(), "test_indices": test.tolist(),
        "train_ids": ids[train].tolist(), "test_ids": ids[test].tolist(),
        "train_n": len(train), "test_n": len(test),
    }
    if labels is not None:
        if np.intersect1d(labels[train], labels[test]).size:
            raise AssertionError("Ancestry group leakage")
        record["train_groups"] = labels[train].tolist()
        record["test_groups"] = labels[test].tolist()
    if positions is not None:
        record["training_sample_positions"] = positions[0].tolist()
        record["validation_sample_positions"] = positions[1].tolist()
    return record


def select_and_predict(engine, y, ids, train, test, inner_labels, ancestry_labels=None):
    """Full primary grid, with all transformations rebuilt inside every split."""
    if np.intersect1d(ids[train], ids[test]).size:
        raise AssertionError("Fixed test IDs entered the training sample")
    unique_groups = np.unique(inner_labels)
    inner_n = min(5, len(unique_groups))
    if inner_n < 2:
        raise ValueError("At least two unique training groups are required for inner CV")
    losses, inner_records = {}, []
    for fold, (it, iv) in enumerate(GroupKFold(inner_n).split(train, groups=inner_labels)):
        a, v = train[it], train[iv]
        if np.intersect1d(inner_labels[it], inner_labels[iv]).size:
            raise AssertionError("Inner grouping leakage")
        record = split_record(a, v, ids, ancestry_labels, (it, iv))
        record["fold"] = fold
        kernels = engine.build(a, v, scales=core.SCALES, local=True, global_rbf=True)
        record["preprocessing"] = dict(engine.last_audit)
        inner_records.append(record)
        for key, (kt, cross) in kernels.items():
            pred = core.eig_predictions(kt, cross, y[a], core.LAMBDAS)
            if not np.isfinite(pred).all():
                raise FloatingPointError(f"Nonfinite inner predictions: {key}")
            squared = np.square(pred - y[v, None]).sum(axis=0)
            for family in core.candidate_families(key, "strict"):
                choice = (family, key)
                losses[choice] = losses.get(choice, np.zeros(len(core.LAMBDAS))) + squared
    selection = {}
    for family in FAMILIES:
        candidates = [(float(cost.min() / len(train)), key, int(cost.argmin()))
                      for (f, key), cost in losses.items() if f == family]
        if not candidates:
            raise ValueError(f"Missing main-grid family: {family}")
        selection[family] = min(candidates, key=lambda item: (item[0], item[1], item[2]))
    outer = engine.build(train, test, scales=core.SCALES, local=True, global_rbf=True)
    audit = dict(engine.last_audit)
    predictions, params = {}, {}
    for family, (loss, key, penalty) in selection.items():
        kt, cross = outer[key]
        pred = core.eig_predictions(kt, cross, y[train], [core.LAMBDAS[penalty]])[:, 0]
        if not np.isfinite(pred).all():
            raise FloatingPointError(f"Nonfinite outer predictions: {family}")
        predictions[family] = pred
        params[family] = {
            "kind": key[0], "window_scale": int(key[1]), "gamma": float(key[2]),
            "lambda": float(core.LAMBDAS[penalty]), "inner_mse": loss,
        }
    predictions["train_mean"] = np.full(len(test), y[train].mean())
    predictions["train_median"] = np.full(len(test), np.median(y[train]))
    # Outer outcomes are read only after parameters and predictions are fixed.
    return predictions, {
        "inner_n_splits": inner_n, "inner_folds": inner_records,
        "selection": params, "preprocessing": audit,
        "test_metrics": {name: scored(y[test], pred) for name, pred in predictions.items()},
    }


def paired_differences(y, predictions):
    result = {}
    for name, candidate, reference in COMPARISONS:
        a, b = core.metrics(y, predictions[candidate]), core.metrics(y, predictions[reference])
        result[name] = {
            "candidate": candidate, "reference": reference,
            "delta_pearson": a["pearson"] - b["pearson"] if a["pearson"] is not None and b["pearson"] is not None else None,
            "delta_mae_days": a["mae"] - b["mae"],
        }
    return result


def grouped_holdouts(engine, codes, y, ids, windows, prov):
    labels, grouping = group_partitions(len(y))
    config = {"provenance": prov, "grouping": grouping, "k": list(K_VALUES),
              "outer": "GroupKFold(5)", "inner": "GroupKFold(min(5, available training groups))",
              "full_primary_grid": True, "selection": "pooled inner-validation MSE",
              "baselines": list(BASELINES)}
    arrays = {"y": y, "ids": ids, "k_values": np.asarray(K_VALUES), "group_labels": labels,
              "outer_test_fold": np.full(labels.shape, -1, dtype=np.int16)}
    arrays.update({name: np.full(labels.shape, np.nan) for name in FAMILIES + BASELINES})
    state, arrays = load_checkpoint("grouped_strict.json", "grouped_strict.npz", config,
                                   {"status": "running", "folds": [], "partitions": {}}, arrays)
    done = {(row["k"], row["fold"]) for row in state["folds"]}
    for partition, (k, groups) in enumerate(zip(K_VALUES, labels)):
        for fold, (train, test) in enumerate(GroupKFold(5).split(np.arange(len(y)), groups=groups)):
            if (k, fold) in done:
                continue
            record = split_record(train, test, ids, groups)
            pred, fitted = select_and_predict(engine, y, ids, train, test, groups[train], groups)
            record.update(fitted)
            record.update({"k": k, "fold": fold})
            for name, p in pred.items():
                arrays[name][partition, test] = p
            arrays["outer_test_fold"][partition, test] = fold
            state["folds"].append(record)
            checkpoint("grouped_strict.json", "grouped_strict.npz", state, arrays)
            print(f"grouped_strict k={k} fold={fold + 1}/5 completed n={len(test)}", flush=True)
        rows = [row for row in state["folds"] if row["k"] == k]
        pooled = {name: scored(y, arrays[name][partition]) for name in FAMILIES + BASELINES}
        signs = {}
        for name in FAMILIES + BASELINES:
            r = [row["test_metrics"][name]["pearson"] for row in rows]
            signs[name] = {"positive_folds": sum(v is not None and v > 0 for v in r),
                           "negative_folds": sum(v is not None and v < 0 for v in r),
                           "zero_folds": sum(v is not None and v == 0 for v in r),
                           "undefined_folds": sum(v is None for v in r)}
        state["partitions"][str(k)] = {
            "group_sizes": {str(g): int(np.sum(groups == g)) for g in np.unique(groups)},
            "pooled_oof": pooled, "fold_pearson_signs": signs,
            "paired_pooled_differences": paired_differences(y, {name: arrays[name][partition] for name in FAMILIES}),
        }
    if not all(np.isfinite(arrays[name]).all() for name in FAMILIES + BASELINES):
        raise AssertionError("Incomplete grouped OOF predictions")
    state["status"] = "completed"
    state["interpretation"] = (
        "Whole-group transfer on three prespecified genetic partitions. Report fold sizes and fold-wise "
        "metrics alongside pooled OOF metrics. Negative pooled correlation does not imply that all "
        "folds reverse direction; between-fold location shifts can affect pooled correlation."
    )
    checkpoint("grouped_strict.json", "grouped_strict.npz", state, arrays)
    return state


def extreme_sensitivity(codes, y, ids, windows, prov):
    maximum = float(y.max())
    removed = np.flatnonzero(y == maximum)
    if maximum != 189 or len(removed) != 1:
        raise ValueError("Prespecified sensitivity requires the single local maximum DSDS50=189")
    keep = np.flatnonzero(np.arange(len(y)) != removed[0])
    config = {"provenance": prov, "removed_original_index": int(removed[0]),
              "removed_ecotype_id": ids[removed[0]].item(), "removed_dsds50_days": maximum,
              "retained_original_indices": keep.tolist(), "retained_ids": ids[keep].tolist(),
              "retained_y": y[keep].tolist(), "core_tag": "extreme_removed",
              "mode": "strict", "repeats": 10, "outer_folds": 5, "inner_folds": 5,
              "full_primary_grid": True, "global_rbf": True,
              "exclusion_rule": "Only the single prespecified maximum; no exclusion search; main 83 retained"}
    path = own_path("extreme_removed_identity.json")
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing["config_sha256"] != signature(config):
            raise RuntimeError("Extreme-removed checkpoint configuration changed")
    elif (core.RESULTS / "extreme_removed_state.json").exists() or (core.RESULTS / "extreme_removed_predictions.npz").exists():
        raise RuntimeError("Existing extreme_removed outputs lack this runner's identity manifest")
    manifest = {"status": "running", "config": config, "config_sha256": signature(config)}
    json_save("extreme_removed_identity.json", manifest)
    print(f"extreme_removed: remove ecotype={ids[removed[0]]}, DSDS50={maximum:g}; 10x5, FULL main grid", flush=True)
    engine = core.GenomicKernels(codes[keep], windows, local_dtype=core.torch.float32)
    saved, report = core.nested_cv(engine, y[keep], "strict", repeats=10, scales=core.SCALES,
                                 local=True, global_rbf=True, tag="extreme_removed")
    if not all(np.isfinite(p).all() and p.shape == (10, len(keep)) for p in saved.values()):
        raise AssertionError("Incomplete extreme sensitivity")
    manifest["status"] = "completed"
    manifest["paired_averaged_oof_differences"] = paired_differences(y[keep], {name: p.mean(0) for name, p in saved.items()})
    manifest["outputs"] = [str((core.RESULTS / f"extreme_removed_{suffix}").resolve())
                           for suffix in ("metrics.json", "state.json", "predictions.npz")]
    json_save("extreme_removed_identity.json", manifest)
    return report


def training_refits(engine, codes, y, ids, windows, prov, resamples=30):
    if resamples not in (30, 50):
        raise ValueError("This bounded design uses 30 or 50 training refits")
    permutation = np.random.default_rng(SEED).permutation(len(y))
    test, pool = permutation[:17], permutation[17:]
    if len(pool) != 66 or np.intersect1d(ids[test], ids[pool]).size:
        raise AssertionError("Expected a disjoint outcome-independent 17/66 partition")
    rng = np.random.default_rng(SEED + 1)
    samples = rng.choice(pool, size=(resamples, len(pool)), replace=True)
    config = {
        "provenance": prov, "holdout_rule": "default_rng(20261118).permutation(83), first 17",
        "holdout_seed": SEED, "training_bootstrap_seed": SEED + 1, "resamples": resamples,
        "test_indices": test.tolist(), "test_ids": ids[test].tolist(),
        "training_pool_indices": pool.tolist(), "training_pool_ids": ids[pool].tolist(),
        "inner": "GroupKFold(min(5, unique sampled canonical IDs)); duplicate copies stay together",
        "sampling": "66 training rows with replacement per refit; test IDs NEVER eligible",
        "full_primary_grid": True, "selection": "pooled inner-validation MSE, counted by resampled row frequency",
        "point_prediction": "Full original 66-row training pool, same canonical-ID grouped inner selection",
    }
    arrays = {"y_test": y[test], "test_ids": ids[test], "test_indices": test,
              "training_pool_ids": ids[pool], "training_pool_indices": pool,
              "sampled_indices": samples, "sampled_canonical_ids": ids[samples],
              "sampled_counts_original_order": np.asarray([np.bincount(s, minlength=len(y)) for s in samples])}
    arrays.update({name: np.full((resamples, len(test)), np.nan) for name in FAMILIES + BASELINES})
    arrays.update({"point_" + name: np.full(len(test), np.nan) for name in FAMILIES + BASELINES})
    state, arrays = load_checkpoint("training_refit_robustness.json", "training_refit_robustness.npz", config,
                                   {"status": "running", "refits": [], "point_fit": None}, arrays)
    if not np.array_equal(arrays["sampled_indices"], samples):
        raise AssertionError("Saved bootstrap training schedule changed")
    if state["point_fit"] is None:
        predictions, record = select_and_predict(engine, y, ids, pool, test, ids[pool])
        state["point_fit"] = record
        for name, pred in predictions.items():
            arrays["point_" + name] = pred
        checkpoint("training_refit_robustness.json", "training_refit_robustness.npz", state, arrays)
        print("training_refit point fit completed: 66 train / fixed 17 test", flush=True)
    done = {row["refit"] for row in state["refits"]}
    for refit, train in enumerate(samples):
        if refit in done:
            continue
        predictions, fitted = select_and_predict(engine, y, ids, train, test, ids[train])
        record = {
            "refit": refit, "sampled_indices": train.tolist(),
            "sampled_canonical_ids": ids[train].tolist(), "unique_training_ids": int(len(np.unique(ids[train]))),
            "test_ids": ids[test].tolist(), **fitted,
        }
        for name, pred in predictions.items():
            arrays[name][refit] = pred
        state["refits"].append(record)
        checkpoint("training_refit_robustness.json", "training_refit_robustness.npz", state, arrays)
        print(f"training_refit {refit + 1}/{resamples} completed unique_training_ids={record['unique_training_ids']}", flush=True)
    state["training_variation"] = {}
    for name in FAMILIES + BASELINES:
        p = arrays[name]
        if not np.isfinite(p).all():
            raise AssertionError("Incomplete training-refitted predictions")
        state["training_variation"][name] = {
            "fixed_test_metrics": core.summarize(y[test], p),
            "point_metrics": scored(y[test], arrays["point_" + name]),
            "point_prediction": arrays["point_" + name].tolist(),
            "test_ids": ids[test].tolist(), "prediction_mean": p.mean(0).tolist(),
            "prediction_sd_days": p.std(0, ddof=1).tolist(),
            "prediction_percentiles_2_5_50_97_5": np.percentile(p, [2.5, 50, 97.5], axis=0).tolist(),
            "mean_abs_change_from_point_days": float(np.mean(np.abs(p - arrays["point_" + name]))),
        }
    state["paired_differences_per_refit"] = [paired_differences(y[test], {name: arrays[name][i] for name in FAMILIES})
                                             for i in range(resamples)]
    state["point_paired_differences"] = paired_differences(y[test], {name: arrays["point_" + name] for name in FAMILIES})
    state["status"] = "completed"
    state["interpretation"] = (
        "Training-set perturbation / parameter-selection variation on one fixed, label-independent "
        "17-accession holdout. Every refit recomputes training-frequency allele filtering, scaling and "
        "all nested parameter choices. Prediction percentile bands are descriptive training variation, "
        "NOT a full generalization confidence interval or independent biological replication. The "
        "17 test outcomes are never used for selection, exclusion, sample design or choosing a refit."
    )
    checkpoint("training_refit_robustness.json", "training_refit_robustness.npz", state, arrays)
    return state


def weighted_pair_metrics(y, prediction, weights):
    """Frozen-prediction paired bootstrap, including unequal-size cluster draws."""
    w = np.asarray(weights, float)
    total = w.sum(1)
    my, mp = (w @ y) / total, (w @ prediction) / total
    vy = np.maximum((w @ np.square(y)) / total - my * my, 0)
    vp = np.maximum((w @ np.square(prediction)) / total - mp * mp, 0)
    covariance = (w @ (y * prediction)) / total - my * mp
    correlation = np.full(len(total), np.nan)
    valid = (vy > 1e-12) & (vp > 1e-12)
    correlation[valid] = np.clip(covariance[valid] / np.sqrt(vy[valid] * vp[valid]), -1, 1)
    mae = (w @ np.abs(y - prediction)) / total
    return correlation, mae


def conditional_paired_bootstrap(y, predictions, weights):
    """Conditional on already fitted models/averaged OOF; never refits training."""
    values = {name: weighted_pair_metrics(y, p, weights) for name, p in predictions.items()}
    report, saved = {}, {}
    observed = paired_differences(y, predictions)
    for name, candidate, reference in COMPARISONS:
        dr = values[candidate][0] - values[reference][0]
        dm = values[candidate][1] - values[reference][1]
        saved[name + "_delta_r"] = dr
        saved[name + "_delta_mae"] = dm
        report[name] = {**observed[name], "draws": len(weights)}
        for key, array in (("delta_pearson", dr), ("delta_mae_days", dm)):
            valid = array[np.isfinite(array)]
            report[name][key + "_conditional_percentile95"] = np.percentile(valid, [2.5, 97.5]).tolist() if len(valid) else None
            report[name][key + "_valid_draws"] = len(valid)
    return report, saved


def nadeau_bengio_approximation(y, saved, state):
    """Appendix-only corrected repeated-CV sensitivity, not biological n=10."""
    result = {
        "role": "APPENDIX approximate sensitivity ONLY; not primary model-superiority inference",
        "independent_accessions": len(y), "repeated_splits": len(state["folds"]),
        "biological_replicates": "Neither 10 repeats nor 50 overlapping folds are independent biological samples",
        "formula": "SE = sqrt((1/m + mean(n_test/n_train)) * sample_variance(fold paired differences)); t df=m-1",
        "limitations": "Overlapping training folds, correlated accessions and nonlinear correlation metrics violate simplifying assumptions; intervals are approximate",
        "comparisons": {},
    }
    ratios = [len(row["test_indices"]) / len(row["train_indices"]) for row in state["folds"]]
    for name, candidate, reference in COMPARISONS:
        differences = {"delta_pearson": [], "delta_mae_days": []}
        for row in state["folds"]:
            repeat, test = row["repeat"], np.asarray(row["test_indices"])
            a = core.metrics(y[test], saved[candidate][repeat, test])
            b = core.metrics(y[test], saved[reference][repeat, test])
            if a["pearson"] is not None and b["pearson"] is not None:
                differences["delta_pearson"].append(a["pearson"] - b["pearson"])
            differences["delta_mae_days"].append(a["mae"] - b["mae"])
        result["comparisons"][name] = {}
        for metric, values in differences.items():
            array = np.asarray(values)
            m = len(array)
            se = float(np.sqrt((1 / m + np.mean(ratios)) * array.var(ddof=1))) if m > 1 else None
            half = float(stats.t.ppf(0.975, m - 1) * se) if se is not None else None
            result["comparisons"][name][metric] = {
                "mean_fold_difference": float(array.mean()) if m else None,
                "overlapping_split_count_not_biological_n": m,
                "mean_test_train_ratio": float(np.mean(ratios)), "corrected_se_approx": se,
                "approximate_interval95": [float(array.mean() - half), float(array.mean() + half)] if half is not None else None,
            }
    return result


def conditional_uncertainty(y, ids, draws=5000):
    tag = "strict_mixed_precision"
    paths = {suffix: core.RESULTS / f"{tag}_{suffix}"
             for suffix in ("metrics.json", "state.json", "predictions.npz")}
    # One availability check and one read per source. No polling or waiting.
    if not all(path.is_file() for path in paths.values()):
        report = {"status": "deferred", "reason": "Main strict metrics/state/predictions not all present; checked once, no polling"}
        json_save("conditional_uncertainty.json", report)
        print("conditional uncertainty deferred: main outputs unavailable at single check", flush=True)
        return report
    source_bytes = {name: path.read_bytes() for name, path in paths.items()}
    main_metrics = json.loads(source_bytes["metrics.json"])
    state = json.loads(source_bytes["state.json"])
    with np.load(io.BytesIO(source_bytes["predictions.npz"]), allow_pickle=False) as archive:
        saved = {name: archive[name].copy() for name in archive.files}
    if not np.array_equal(saved.pop("y"), y):
        raise AssertionError("Main OOF outcomes/order do not match current cohort")
    expected = {(rep, fold) for rep in range(10) for fold in range(5)}
    if {(row["repeat"], row["fold"]) for row in state["folds"]} != expected or len(state["folds"]) != 50:
        raise AssertionError("Main strict analysis does not have all 10x5 folds")
    names = {c for _, a, b in COMPARISONS for c in (a, b)}
    for name in names:
        if saved[name].shape != (10, len(y)) or not np.isfinite(saved[name]).all() or name not in main_metrics:
            raise AssertionError("Main averaged OOF source is incomplete")
    predictions = {name: saved[name].mean(0) for name in sorted(names)}
    rng = np.random.default_rng(SEED + 2)
    weights = rng.multinomial(len(y), np.full(len(y), 1 / len(y)), size=draws)
    accession, differences = conditional_paired_bootstrap(y, predictions, weights)
    arrays = {"y": y, "ids": ids, "accession_weights": weights.astype(np.int16)}
    arrays.update({"averaged_oof_" + name: pred for name, pred in predictions.items()})
    arrays.update({"accession_" + name: value for name, value in differences.items()})
    labels, grouping = group_partitions(len(y))
    arrays["k_values"], arrays["group_labels"] = np.asarray(K_VALUES), labels
    cluster_report = {}
    for k, groups in zip(K_VALUES, labels):
        unique, inverse = np.unique(groups, return_inverse=True)
        cluster_counts = rng.multinomial(k, np.full(k, 1 / k), size=draws)
        cluster_weights = cluster_counts[:, inverse]
        result, delta = conditional_paired_bootstrap(y, predictions, cluster_weights)
        cluster_report[str(k)] = {"clusters": len(unique), "unit": "whole ancestry cluster, preserving all rows and their sizes", "comparisons": result}
        arrays[f"cluster_k{k}_draw_counts"] = cluster_counts.astype(np.int16)
        arrays.update({f"cluster_k{k}_" + name: value for name, value in delta.items()})
    report = {
        "status": "completed", "source_tag": tag,
        "source_paths": {name: str(path.resolve()) for name, path in paths.items()},
        "source_hashes": {name: hashlib.sha256(data).hexdigest() for name, data in source_bytes.items()},
        "draws": draws, "seed": SEED + 2, "n_accessions": len(y),
        "averaging": "One accession prediction obtained by averaging its 10 repeated OOF predictions",
        "delta_direction": "candidate - reference; positive delta-r and negative delta-MAE favor candidate",
        "accession_pairs_bootstrap": accession, "ancestry_cluster_pairs_bootstrap": cluster_report,
        "grouping": grouping,
        "interpretation": "CONDITIONAL uncertainty with frozen fitted predictions and no training refitting; cannot replace training_refit_robustness (C), or establish full generalization coverage / independent biological replication",
        "nadeau_bengio_appendix_approximation": nadeau_bengio_approximation(y, saved, state),
    }
    npz_save("conditional_uncertainty.npz", **arrays)
    json_save("conditional_uncertainty.json", report)
    print("conditional uncertainty completed: averaged OOF, 5000 paired accession/cluster draws, no refitting" if draws == 5000 else f"conditional uncertainty completed: {draws} paired draws", flush=True)
    return report


def selftests(engine, codes, y, ids):
    """Actual duplicate-row preprocessing and eigensolver, not just a mock."""
    train = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 2, 4, 4, 4, 6, 7, 7, 7, 7, 1])
    query = np.asarray([8, 9, 10, 11])
    kernels = engine.build(train, query, scales=core.SCALES, local=True, global_rbf=True)
    counts = np.stack([(codes[train] == c).sum(0) for c in range(4)])
    called = counts.sum(0)
    keep = ((counts > 0).sum(0) == 2) & ((1 - counts.max(0) / np.maximum(called, 1)) > 0.05)
    if int(keep.sum()) != engine.last_audit["retained_markers"]:
        raise AssertionError("Duplicate training-frequency marker filtering is incorrect")
    if digest_array(np.flatnonzero(keep).astype(np.int64)) != engine.last_audit["marker_index_sha256"]:
        raise AssertionError("Training-frequency retained marker identities do not match")
    errors = {}
    for key, (kt, cross) in kernels.items():
        pred = core.eig_predictions(kt, cross, y[train], core.LAMBDAS)
        if not np.isfinite(pred).all() or pred.shape != (len(query), len(core.LAMBDAS)):
            raise AssertionError("Duplicate-row predictions not finite across full lambda grid")
        if not np.allclose(kt, kt.T, atol=1e-10) or abs(kt.mean()) > 1e-8 or abs(np.trace(kt) / len(train) - 1) > 1e-8:
            raise AssertionError("Duplicate-row centered/scaled kernel invariant failed")
        maximum_error = 0.0
        for penalty in (0.001, 0.1, 1.0, 10.0):
            system = np.block([[kt + penalty * np.eye(len(train)), np.ones((len(train), 1))],
                               [np.ones((1, len(train))), np.zeros((1, 1))]])
            coefficients = linalg.solve(system, np.r_[y[train], 0.0], assume_a="sym")
            direct = cross @ coefficients[:-1] + coefficients[-1]
            stable = core.eig_predictions(kt, cross, y[train], [penalty])[:, 0]
            maximum_error = max(maximum_error, float(np.max(np.abs(direct - stable))))
        if maximum_error > 1e-6:
            raise AssertionError(f"Duplicate-row solver mismatch: {key}, {maximum_error}")
        errors[str(key)] = maximum_error
    fold_count = 0
    for it, iv in GroupKFold(5).split(train, groups=ids[train]):
        split_record(train[it], train[iv], ids, positions=(it, iv))
        fold_count += 1
    # Check weighted cluster implementation against explicit repeated pairs.
    weights = np.asarray([[1, 2, 0, 3], [2, 0, 2, 1]])
    yy, pp = np.asarray([1., 2., 6., 10.]), np.asarray([2., 1., 5., 9.])
    rr, mm = weighted_pair_metrics(yy, pp, weights)
    for row, w in enumerate(weights):
        expanded = np.repeat(np.arange(4), w)
        m = core.metrics(yy[expanded], pp[expanded])
        if abs(rr[row] - m["pearson"]) > 1e-12 or abs(mm[row] - m["mae"]) > 1e-12:
            raise AssertionError("Paired cluster bootstrap weighting differs from expanded pairs")
    report = {
        "passed": True, "real_genotype_duplicate_training_rows": train.tolist(),
        "query_indices_disjoint": query.tolist(), "tested_kernel_count": len(kernels),
        "full_scale_gamma_lambda_grid_finite": True,
        "duplicated_training_frequency_marker_filter_matches_direct_counts": True,
        "canonical_id_inner_group_disjoint_folds": fold_count,
        "augmented_intercept_solver_comparison_penalties": [0.001, 0.1, 1, 10],
        "maximum_prediction_difference_days": max(errors.values()), "per_kernel_difference_days": errors,
        "tiny_penalties": "All 11 including 1e-12 tested finite; ill-conditioned direct KKT comparisons restricted to >=0.001",
        "weighted_pairs_equals_expanded_rows": True,
    }
    json_save("robustness_selftests.json", report)
    print(f"robustness selftests passed: {len(kernels)} real duplicate-row kernels, max solver error={max(errors.values()):.3g} days", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("all", "tests", "grouped", "extreme", "training", "conditional"), default="all")
    parser.add_argument("--resamples", type=int, choices=(30, 50), default=30)
    parser.add_argument("--conditional-draws", type=int, default=5000)
    args = parser.parse_args()
    if args.conditional_draws < 100:
        parser.error("At least 100 conditional paired-bootstrap draws are required")
    started = time.time()
    with threadpool_limits(2):
        codes, y, metadata, ids, windows = core.load_dormancy()
        if len(y) != 83 or len(np.unique(ids)) != 83:
            raise AssertionError("Expected 83 unique canonical accession IDs")
        engine = core.GenomicKernels(codes, windows, local_dtype=core.torch.float32)
        prov = provenance(codes, y, ids, windows)
        modes = ("tests", "conditional", "grouped", "extreme", "training") if args.mode == "all" else (args.mode,)
        status = {"mode": args.mode, "status": "running", "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                  "completed": [], "pending": list(modes), "resamples": args.resamples, "errors": []}
        json_save("robustness_run_status.json", status)
        try:
            for mode in modes:
                status["current"] = mode
                json_save("robustness_run_status.json", status)
                if mode == "tests":
                    result = selftests(engine, codes, y, ids)
                elif mode == "grouped":
                    result = grouped_holdouts(engine, codes, y, ids, windows, prov)
                elif mode == "extreme":
                    result = extreme_sensitivity(codes, y, ids, windows, prov)
                elif mode == "training":
                    result = training_refits(engine, codes, y, ids, windows, prov, args.resamples)
                else:
                    result = conditional_uncertainty(y, ids, args.conditional_draws)
                status["completed"].append({"mode": mode, "status": result.get("status", "completed")})
                status["pending"].remove(mode)
                status["elapsed_seconds"] = time.time() - started
                json_save("robustness_run_status.json", status)
            status["status"] = "completed"
            status["current"] = None
            json_save("robustness_run_status.json", status)
            print(f"robustness {args.mode} completed in {time.time() - started:.1f}s", flush=True)
        except Exception as exc:
            status["status"] = "failed"
            status["errors"].append({"type": type(exc).__name__, "message": str(exc)})
            status["elapsed_seconds"] = time.time() - started
            json_save("robustness_run_status.json", status)
            raise


if __name__ == "__main__":
    main()
