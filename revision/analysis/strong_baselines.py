"""Strict training-refitted baselines for genomic prediction.

Only this new script and its own results subtree are written. No old QC output,
phenotype-selected locus, or full-panel allele/LD estimate is reused. Run with
python -s -B; --mode tests and --mode benchmark precede --mode run.
"""
from __future__ import annotations

import os
import sys

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "2"
sys.dont_write_bytecode = True

import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numba
import numpy as np
import pandas as pd
import scipy
from numba import njit
from scipy import linalg, stats
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits, threadpool_info
import torch
import kernel_benchmark as kb

kb.DEVICE = torch.device("cpu")
torch.set_num_threads(2)
numba.set_num_threads(2)
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUTPUT = HERE.parent / "results" / "strong_baselines_20261118"
WINDOW_BP = 100_000
R2_THRESHOLD = 0.2
PI_GRID = (0.05, 0.2, 0.5)  # Inclusion, NOT the exclusion-probability convention.
DOG1_CHROM, DOG1_START, DOG1_END = 5, 18_589_418, 18_591_506
DOG1_FLANK = 50_000
DOG1_MIX = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
RF_GRID = tuple((leaf, feat) for leaf in (1, 3, 5) for feat in ("sqrt", 0.1))
FAST_FAMILIES = ("ld_pruned_snp", "ld_score_inverse_linear", "snp_plus_dog1", "random_forest")


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def atomic_npz(path, **arrays):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    tmp.replace(path)


@njit(cache=False, inline="always")
def popcount(value):
    value = value - ((value >> np.uint64(1)) & np.uint64(0x5555555555555555))
    value = (value & np.uint64(0x3333333333333333)) + ((value >> np.uint64(2)) & np.uint64(0x3333333333333333))
    value = (value + (value >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return int((value * np.uint64(0x0101010101010101)) >> np.uint64(56))


@njit(cache=False)
def allele_selection(codes, train):
    p = codes.shape[0]
    keep = np.zeros(p, dtype=np.bool_)
    minor = np.zeros(p, dtype=np.uint8)
    count = np.zeros(p, dtype=np.int64)
    for j in range(p):
        counts = np.zeros(4, dtype=np.int64)
        for row in train:
            counts[codes[j, row]] += 1
        classes = 0
        smallest = len(train) + 1
        allele = 0
        for c in range(4):
            if counts[c] > 0:
                classes += 1
                if counts[c] < smallest:
                    smallest, allele = counts[c], c
        if classes == 2 and smallest / len(train) > 0.05:
            keep[j] = True
            minor[j], count[j] = allele, smallest
    return keep, minor, count


@njit(cache=False)
def pack_binary(x):
    p, n = x.shape
    out = np.zeros((p, (n + 63) // 64), dtype=np.uint64)
    for j in range(p):
        for i in range(n):
            if x[j, i] != 0:
                out[j, i // 64] |= np.uint64(1) << np.uint64(i % 64)
    return out


@njit(cache=False, inline="always")
def binary_r2(bits, counts, n, j, k):
    both = 0
    for word in range(bits.shape[1]):
        both += popcount(bits[j, word] & bits[k, word])
    a, b = float(counts[j]), float(counts[k])
    numerator = float(n) * both - a * b
    return numerator * numerator / (a * (n - a) * b * (n - b))


@njit(cache=False)
def strict_ld(bits, counts, n, chrom, pos, window_bp, threshold):
    """Coordinate-ordered greedy pruning and exact local all-pair LD scores.

    The retained pair constraint is r^2 <= threshold, within window_bp on the
    same chromosome. Score_j = 1 + sum_{k != j, distance <= window_bp} r_jk^2,
    on ALL training-QC markers, not just pruned markers. Packed carrier counts
    give exactly the Pearson correlation squared of binary allele indicators.
    """
    p = len(counts)
    selected = np.zeros(p, dtype=np.bool_)
    score = np.ones(p, dtype=np.float64)
    left = 0
    pairs = 0
    for j in range(p):
        while left < j and (chrom[left] != chrom[j] or pos[j] - pos[left] > window_bp):
            left += 1
        accept = True
        for k in range(left, j):
            r2 = binary_r2(bits, counts, n, j, k)
            score[j] += r2
            score[k] += r2
            pairs += 1
            if selected[k] and r2 > threshold:
                accept = False
        selected[j] = accept
    return selected, score, pairs


class Preprocessor:
    def __init__(self, codes, metadata):
        assert codes.min() >= 0 and codes.max() <= 3
        self.codes = np.ascontiguousarray(codes.T, dtype=np.uint8)
        self.chrom = metadata.chrom.to_numpy(dtype=np.int64)
        self.pos = metadata.pos.to_numpy(dtype=np.int64)
        self.order = np.lexsort((np.arange(len(metadata)), self.pos, self.chrom))

    def prepare(self, train, query, cache=None):
        train, query = np.asarray(train, np.int64), np.asarray(query, np.int64)
        assert len(np.intersect1d(train, query)) == 0
        started = time.perf_counter()
        cached = cache is not None and cache.exists()
        if cached:
            with np.load(cache, allow_pickle=False) as src:
                info = {k: src[k].copy() for k in src.files}
            assert np.array_equal(info["train"], train) and np.array_equal(info["query"], query)
        else:
            keep, minor, count = allele_selection(self.codes, train)
            indexes = self.order[keep[self.order]]
            assert len(indexes) > 0
            minor, count = minor[indexes], count[indexes]
            mean = count.astype(float) / len(train)
            sd = np.sqrt(mean * (1 - mean))
            xt = (self.codes[np.ix_(indexes, train)] == minor[:, None]).astype(np.float64)
            bits = pack_binary(xt)
            selected, score, pairs = strict_ld(bits, count, len(train), self.chrom[indexes],
                                               self.pos[indexes], WINDOW_BP, R2_THRESHOLD)
            info = {"train": train, "query": query, "indexes": indexes, "minor": minor,
                    "mean": mean, "sd": sd, "pruned_local": np.flatnonzero(selected),
                    "ld_score": score, "ld_pairs": np.asarray(pairs),
                    "compute_seconds": np.asarray(time.perf_counter() - started)}
            if cache is not None:
                atomic_npz(cache, **info)
        indexes, minor = info["indexes"], info["minor"]
        xt = (self.codes[np.ix_(indexes, train)] == minor[:, None]).astype(np.float64)
        xq = (self.codes[np.ix_(indexes, query)] == minor[:, None]).astype(np.float64)
        zt = np.ascontiguousarray((xt - info["mean"][:, None]) / info["sd"][:, None])
        zq = np.ascontiguousarray((xq - info["mean"][:, None]) / info["sd"][:, None])
        audit = {"ld_route": "strict_per_training_split", "reference_panel_used": False,
                 "training_n": len(train), "query_n": len(query), "training_qc_markers": len(indexes),
                 "pruned_markers": len(info["pruned_local"]), "local_ld_pairs": int(info["ld_pairs"]),
                 "ld_compute_seconds": float(info["compute_seconds"]), "cache_reused": cached,
                 "marker_index_sha256": digest(indexes), "minor_code_sha256": digest(minor),
                 "train_scale_sha256": digest(np.r_[info["mean"], info["sd"]]),
                 "pruned_index_sha256": digest(indexes[info["pruned_local"]]),
                 "training_maf_min": float(info["mean"].min()), "training_maf_max": float(info["mean"].max()),
                 "ld_score_min": float(info["ld_score"].min()), "ld_score_max": float(info["ld_score"].max())}
        return zt, zq, info, audit

    def kernels(self, zt, zq, info):
        pr = info["pruned_local"]
        p = len(info["indexes"])
        full = kb.centered_pair(zt.T @ zt / p, zq.T @ zt / p)
        pruned = kb.centered_pair(zt[pr].T @ zt[pr] / len(pr), zq[pr].T @ zt[pr] / len(pr))
        weights = 1.0 / info["ld_score"]
        ld = kb.centered_pair(zt.T @ (weights[:, None] * zt) / weights.sum(),
                              zq.T @ (weights[:, None] * zt) / weights.sum())
        idx = info["indexes"]
        dog = ((self.chrom[idx] == DOG1_CHROM) & (self.pos[idx] >= DOG1_START - DOG1_FLANK)
               & (self.pos[idx] <= DOG1_END + DOG1_FLANK))
        if np.any(dog):
            dogpair = kb.centered_pair(zt[dog].T @ zt[dog] / dog.sum(), zq[dog].T @ zt[dog] / dog.sum())
        else:
            dogpair = (np.zeros_like(full[0]), np.zeros_like(full[1]))
        out = {"ld_pruned_snp": [pruned], "ld_score_inverse_linear": [ld],
               "snp_plus_dog1": [((1 - w) * full[0] + w * dogpair[0],
                                  (1 - w) * full[1] + w * dogpair[1]) for w in DOG1_MIX]}
        return out, int(dog.sum())


@njit(cache=False)
def gibbs_chain(z, zquery, y, pi, burn, draws, seed, update_variances=True):
    """BayesC: point mass + shared normal slab; conjugate variance Gibbs.

    Here pi is Pr(nonzero). Standardized y = mu + Zb + e. Priors:
    mu ~ flat; b_j ~ (1-pi)delta_0 + pi N(0,tau2);
    sigma_e2 ~ IG(3, 2*(1-h2)); tau2 ~ IG(3, 2*h2/(p*pi)), h2=.5.
    IG(a,b) has density proportional to s^(-a-1) exp(-b/s).
    Conditional v_j = (Z_j'Z_j/sigma_e2 + 1/tau2)^(-1),
    m_j = v_j Z_j'r_j/sigma_e2; log BF = .5[log(v_j/tau2)+m_j^2/v_j].
    Inclusion odds = pi/(1-pi)*BF, then N(m_j,v_j) or exactly zero.
    """
    np.random.seed(seed)
    p, n = z.shape
    nq = zquery.shape[1]
    h2 = 0.5
    ae, ab = 3.0, 3.0
    be, bb = 2.0 * (1 - h2), 2.0 * h2 / (p * pi)
    sigma_e, tau = 1 - h2, h2 / (p * pi)
    if update_variances:
        sigma_e = be / np.random.gamma(ae, 1.0)
        tau = bb / np.random.gamma(ab, 1.0)
    scale_accepts = 0
    max_residual_drift = 0.0
    mu = np.random.normal(0, math.sqrt(sigma_e / n))
    b = np.zeros(p)
    residual = y - mu
    norm = np.zeros(p)
    for j in range(p):
        for i in range(n):
            norm[j] += z[j, i] * z[j, i]
    order = np.arange(p)
    bmean = np.zeros(p)
    pip = np.zeros(p)
    mumean = 0.0
    traces = np.zeros((draws, 3 + nq))
    logodds = math.log(pi / (1 - pi))
    for step in range(burn + draws):
        for i in range(n):
            residual[i] += mu
        mu = np.mean(residual) + np.random.normal(0, math.sqrt(sigma_e / n))
        for i in range(n):
            residual[i] -= mu
        np.random.shuffle(order)
        active, squared = 0, 0.0
        for jj in range(p):
            j = order[jj]
            dot = 0.0
            for i in range(n):
                residual[i] += z[j, i] * b[j]
                dot += z[j, i] * residual[i]
            variance = sigma_e / (norm[j] + sigma_e / tau)
            mean = variance * dot / sigma_e
            odds = logodds + 0.5 * (math.log(variance / tau) + mean * mean / variance)
            if odds >= 0:
                prob = 1 / (1 + math.exp(-odds))
            else:
                expodds = math.exp(odds)
                prob = expodds / (1 + expodds)
            if np.random.random() < prob:
                b[j] = mean + np.random.normal(0, math.sqrt(variance))
                active += 1
                squared += b[j] * b[j]
            else:
                b[j] = 0.0
            for i in range(n):
                residual[i] -= z[j, i] * b[j]
        rss = np.dot(residual, residual)
        if update_variances:
            sigma_e = (be + 0.5 * rss) / np.random.gamma(ae + 0.5 * n, 1.0)
            tau = (bb + 0.5 * squared) / np.random.gamma(ab + 0.5 * active, 1.0)
            # Exact joint scale MH move within Gibbs, needed for high-p slab
            # variance mixing. log(c) is symmetric N(0,.15^2); b'=c*b,
            # tau'=c^2*tau, indicators unchanged. Jacobian c^(active+2)
            # cancels the normal-slab scale terms. This preserves the SAME
            # BayesC posterior, rather than tempering or changing its prior.
            logc = np.random.normal(0, 0.15)
            c = math.exp(logc)
            rss_new = 0.0
            for i in range(n):
                genetic = y[i] - mu - residual[i]
                proposed = residual[i] + (1 - c) * genetic
                rss_new += proposed * proposed
            logratio = -(rss_new - rss) / (2 * sigma_e) - 2 * ab * logc + bb / tau * (1 - math.exp(-2 * logc))
            if math.log(max(np.random.random(), 1e-300)) < logratio:
                for i in range(n):
                    residual[i] += (1 - c) * (y[i] - mu - residual[i])
                for j in range(p):
                    b[j] *= c
                tau *= c * c
                scale_accepts += 1
        if (step + 1) % 50 == 0:
            for i in range(n):
                exact_residual = y[i] - mu
                for j in range(p):
                    exact_residual -= z[j, i] * b[j]
                max_residual_drift = max(max_residual_drift, abs(exact_residual - residual[i]))
                residual[i] = exact_residual
        if step >= burn:
            t = step - burn
            for j in range(p):
                bmean[j] += b[j]
                pip[j] += 1.0 if b[j] != 0 else 0.0
            mumean += mu
            traces[t, 0], traces[t, 1], traces[t, 2] = sigma_e, tau, active
            for q in range(nq):
                pred = mu
                for j in range(p):
                    pred += zquery[j, q] * b[j]
                traces[t, 3 + q] = pred
    return bmean / draws, pip / draws, mumean / draws, traces, np.array([scale_accepts / (burn + draws), max_residual_drift])


def basic_rhat(x):
    n = x.shape[1]
    w = np.mean(np.var(x, axis=1, ddof=1))
    if w < 1e-20:
        return 1.0 if np.var(x) < 1e-20 else float("inf")
    between = n * np.var(x.mean(1), ddof=1)
    return float(np.sqrt(((n - 1) / n * w + between / n) / w))


def chain_diagnostics(traces):
    """Rank-normalized/folded split Rhat; FFT/Geyer approximate bulk ESS.

    ESS uses rank-normalized split chains, between-chain variance and the
    initial-positive, monotone paired autocorrelation sequence. Diagnostics
    cover residual/slab variances, active counts and EVERY query prediction;
    they cannot certify mixing of all individual marker effects.
    """
    half = traces.shape[1] // 2
    split = np.concatenate((traces[:, :half], traces[:, -half:]), axis=0)
    names = ["sigma_e2", "sigma_slab2", "active_count"] + ["prediction_" + str(i) for i in range(traces.shape[2] - 3)]
    records = {}
    for j, name in enumerate(names):
        x = split[:, :, j]
        def ranknormal(values):
            ranks = stats.rankdata(values.ravel()).reshape(values.shape)
            return stats.norm.ppf((ranks - 0.375) / (values.size + 0.25))
        rank = ranknormal(x)
        folded = ranknormal(abs(x - np.median(x)))
        rhat = max(basic_rhat(rank), basic_rhat(folded))
        centered = rank - rank.mean(axis=1, keepdims=True)
        fft = np.fft.rfft(centered, n=2 * half, axis=1)
        autocov = np.fft.irfft(fft * fft.conj(), n=2 * half, axis=1)[:, :half].real / half
        w = np.mean(np.var(rank, axis=1, ddof=1))
        v = (half - 1) / half * w + np.var(rank.mean(1), ddof=1)
        if v < 1e-20:
            ess = float(rank.size)
        else:
            rho = 1 - (w - autocov.mean(0)) / v
            rho[0] = 1.0
            pairs = []
            for k in range(0, half - 1, 2):
                pair = float(rho[k] + rho[k + 1])
                if pair < 0:
                    break
                pairs.append(min(pair, pairs[-1]) if pairs else pair)
            tau = max(1.0, -1 + 2 * sum(pairs))
            ess = min(float(rank.size), float(rank.size / tau))
        records[name] = {"rank_folded_split_rhat": rhat if np.isfinite(rhat) else None,
                         "approx_bulk_ess": ess}
    rhats = [v["rank_folded_split_rhat"] for v in records.values()]
    maximum = max(r for r in rhats if r is not None) if all(r is not None for r in rhats) else None
    minimum = min(v["approx_bulk_ess"] for v in records.values())
    return {"parameters": records, "maximum_rhat": maximum, "minimum_bulk_ess": minimum,
            "passes_prespecified_thresholds": maximum is not None and maximum <= 1.1 and minimum >= 100,
            "thresholds": {"rhat_max": 1.1, "approx_bulk_ess_min": 100},
            "limitation": "Prediction/global-parameter diagnostics, not proof of convergence of every SNP effect; no tail ESS."}


def run_bayesc(zt, zq, y, pi, seed, burn, draws, extensions, path):
    diagnostic_path = path.with_suffix(".json")
    if path.exists() and diagnostic_path.exists():
        with np.load(path, allow_pickle=False) as src:
            pred = src["prediction"].copy()
        return pred, json.loads(diagnostic_path.read_text(encoding="utf-8"))
    scale = float(np.std(y))
    assert scale > 1e-12
    standardized_y = (y - y.mean()) / scale
    attempts = []
    start = time.perf_counter()
    seeds = [int(seed), int(seed + 104729)]
    for attempt in range(extensions + 1):
        nburn, ndraw = burn * 2 ** attempt, draws * 2 ** attempt
        chains = [gibbs_chain(zt, zq, standardized_y, pi, nburn, ndraw, chainseed) for chainseed in seeds]
        traces = np.stack([chain[3] for chain in chains])
        diagnostic = chain_diagnostics(traces)
        attempts.append({"burn": nburn, "draws_per_chain": ndraw, "diagnostics": diagnostic})
        if diagnostic["passes_prespecified_thresholds"]:
            break
    means = np.stack([chain[0] for chain in chains])
    pips = np.stack([chain[1] for chain in chains])
    intercepts = np.asarray([chain[2] for chain in chains])
    prediction = y.mean() + scale * (intercepts.mean() + means.mean(0) @ zq)
    prediction_error = float(np.max(abs(prediction - (y.mean() + scale * traces[:, :, 3:].mean((0, 1))))))
    assert np.isfinite(prediction).all() and prediction_error < 1e-8
    mixing = np.stack([chain[4] for chain in chains])
    assert mixing[:, 1].max() < 1e-8
    record = {"posterior_mean_vs_trace_prediction_max_error": prediction_error,
              "scale_mh_acceptance_by_chain": mixing[:, 0].tolist(),
              "maximum_residual_recompute_drift_by_chain": mixing[:, 1].tolist(),
              "pi_inclusion": pi, "pi_exclusion": 1 - pi, "chain_seeds": seeds,
              "response_mean_train": float(y.mean()), "response_sd_train": scale,
              "marker_count": len(zt), "chains": 2, "thin": 1, "attempts": attempts,
              "diagnostics_pass": diagnostic["passes_prespecified_thresholds"],
              "restart_extension_rule": "Restart same independent seeds, double warmup/draws, max configured extensions; no outcome-based convergence rule.",
              "seconds": time.perf_counter() - start}
    atomic_npz(path, prediction=prediction, beta_posterior_mean_by_chain=means,
               posterior_inclusion_probability_by_chain=pips, intercept_posterior_mean_by_chain=intercepts,
               traces_standardized_response=traces, chain_seeds=np.asarray(seeds), scale_mh_and_residual_checks_by_chain=mixing)
    kb.save_json(diagnostic_path, record)
    print(f"BayesC {path.stem}: pi={pi:g} seconds={record['seconds']:.1f} "
          f"Rhat={diagnostic['maximum_rhat']} ESS={diagnostic['minimum_bulk_ess']:.1f} "
          f"diagnostic_pass={record['diagnostics_pass']}", flush=True)
    return prediction, record


def specification(args, codes, y, ids, metadata):
    return {"schema": 1, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "core_sha256": hashlib.sha256((HERE / "kernel_benchmark.py").read_bytes()).hexdigest(),
            "protocol": kb.PROTOCOL, "device": "cpu", "threads": 2,
            "inputs": {"codes_sha256": digest(codes), "y_sha256": digest(y), "ids_sha256": digest(ids),
                       "coordinates_sha256": digest(metadata[["chrom", "pos"]].to_numpy())},
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                         "sklearn": sklearn.__version__, "numba": numba.__version__, "torch": torch.__version__},
            "preprocessing": "Each inner/outer TRAIN only: original codes 0..3; exactly two occurring allele classes; min nonzero allele count / n > .05; lowest code breaks a count tie; indicator 0/1, not diploid dosage. Mean=f, sd=sqrt(f(1-f)) estimated only on train. Novel query allele codes map to 0 under frozen train indicator.",
            "ld": {"route": "strict_per_training_split", "independent_reference_used": False,
                   "window_bp": WINDOW_BP, "r2_threshold": R2_THRESHOLD,
                   "pruning": "Coordinate-ordered greedy; retain j iff no previously retained same-chromosome marker <=100kb has Pearson r^2 > .2. No stride/subsampling or old 9245-marker list.",
                   "old_implementation_error": "scripts/02_qc_population_structure.py:68 compares max(abs(r)) < r2_thr, so .2 is |r|, i.e. r^2 < .04, not the intended .2.",
                   "score": "ell_j = 1 + sum_{k != j, same chromosome, |pos_j-pos_k|<=100000} r_jk^2 on all train-QC markers; self term 1.",
                   "weighted_kernel": "K = sum_j (1/ell_j) z_j z_j' / sum_j (1/ell_j), z=(indicator-f)/sqrt(f(1-f)); then train-only centered trace normalization.",
                   "maf_weights": "No extra MAF exponent (exponent 0 for standardized features). On unstandardized centered indicators the weight is 1/[f_j(1-f_j)*ell_j]. MAF cutoff, minor coding, variance and LD scores are fold-train estimates, all arrays saved.",
                   "limitations": "Local inverse-LD-score linear kernel, NOT exact LDAK. No LDAK iterative redundancy weights, recombination model, genome-wide correction or claimed LDAK heritability parameterization; sample LD is noisy at n~53-67."},
            "bayesc": {"model": "BayesC fixed-pi point-mass + common normal slab; pi defined as inclusion probability.",
                       "pi_grid": list(PI_GRID), "selection": "pooled nested inner MSE of two-chain posterior mean predictions",
                       "equations": "y* = mu + sum_j z_j b_j + e; e~N(0,sigma_e2 I); b_j~(1-pi)delta_0+pi*N(0,tau2); mu~flat; sigma_e2~IG(3,1); tau2~IG(3,1/(p*pi)); prior h2=.5, y* train standardized. v=(z'z/sigma_e2+1/tau2)^-1; m=v*z'r/sigma_e2; logBF=.5*(log(v/tau2)+m^2/v). Bernoulli inclusion odds=pi/(1-pi)*BF. sigma_e2|.~IG(3+n/2,1+RSS/2); tau2|.~IG(3+k/2,1/(p*pi)+sum_nonzero b^2/2).",
                       "inner_burn": args.inner_burn, "inner_draws": args.inner_draws,
                       "outer_burn": args.outer_burn, "outer_draws": args.outer_draws,
                       "chains": 2, "thin": 1, "maximum_diagnostic_extensions": args.extensions,
                       "mixing_step": "Coordinate Gibbs plus exact joint scale Metropolis-within-Gibbs: log c~N(0,.15^2), b'=c*b, tau2'=c^2*tau2, same indicators. log acceptance = -(RSS'-RSS)/(2*sigma_e2) -2*a_b*log c + b_b/tau2*(1-c^-2); Jacobian c^(k+2). Same posterior, no tempering. Independently seeded prior-variance initialization, residual recomputation every 50 sweeps; acceptance and numerical residual drift saved.",
                       "diagnostics": "rank-normalized/folded split Rhat<=1.1, approximate Geyer bulk ESS>=100, all query predictions and 3 global parameters; failures retained/flagged, never silently declared converged.",
                       "limitations": "Shared slab variance and fixed .5 prior h2; pi is tuned but variance prior is not. Finite MCMC, no diagnostic for every SNP effect; correlated-marker mixing can fail. Not BayesianRidge and not BayesCpi with a sampled pi."},
            "random_forest": {"trees": args.trees, "grid": [{"min_samples_leaf": leaf, "max_features": feat} for leaf, feat in RF_GRID],
                              "other": "bootstrap=True, squared_error, unbounded depth, n_jobs=2; same split-pruned SNPs; no supervised SNP screening."},
            "dog1": {"gene": "AT5G45830", "chromosome": DOG1_CHROM, "gene_start": DOG1_START,
                     "gene_end": DOG1_END, "flank_bp": DOG1_FLANK, "mix_grid": list(DOG1_MIX),
                     "model": "(1-w) K_all_train_QC_SNP + w K_DOG1; both centered/trace-normalized on train. w and lambda pooled-inner tuned. Empty DOG1 gives zero regional kernel.",
                     "selection": "Fixed TAIR10 annotation gene plus 50kb flanks, not full-83-y GWAS or a selected association peak."},
            "cv": {"outer_repeats": 10, "outer_folds": 5, "outer_seed": "rep", "inner_folds": 5,
                   "inner_seed": "1000+rep*5+fold", "selection": "pooled squared error / number of outer training accessions", "seed_base": kb.PROTOCOL["random_seed"]},
            "metrics_policy": "Only new strict OOF predictions. Separate smoke subtree; no old reported metrics imported. Repeat SD is split variability, not independent replications."}


def initialize(out, spec):
    out.mkdir(parents=True, exist_ok=True)
    for name in ("preprocessing", "tasks", "folds"):
        (out / name).mkdir(exist_ok=True)
    path = out / "manifest.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != spec:
            raise RuntimeError("Checkpoint specification differs. Refuse to mix runs; use a new authorized --output subtree.")
    else:
        kb.save_json(path, spec)


def split_name(rep, fold, inner=None):
    return f"r{rep:02d}_f{fold:02d}_" + ("outer" if inner is None else f"inner{inner:02d}")


def candidate_parameters(family):
    if family in ("ld_pruned_snp", "ld_score_inverse_linear"):
        return [{"lambda": float(lam)} for lam in kb.LAMBDAS]
    if family == "snp_plus_dog1":
        return [{"dog1_weight": w, "lambda": float(lam)} for w in DOG1_MIX for lam in kb.LAMBDAS]
    if family == "random_forest":
        return [{"min_samples_leaf": leaf, "max_features": feat} for leaf, feat in RF_GRID]
    return [{"pi_inclusion": pi} for pi in PI_GRID]


def fast_predictions(engine, zt, zq, info, y, args, seed):
    kernels, ndog = engine.kernels(zt, zq, info)
    predictions = {}
    for family, pairs in kernels.items():
        predictions[family] = np.concatenate([kb.eig_predictions(kt, kq, y) for kt, kq in pairs], axis=1)
    pr = info["pruned_local"]
    a, q = np.ascontiguousarray(zt[pr].T, dtype=np.float32), np.ascontiguousarray(zq[pr].T, dtype=np.float32)
    predictions["random_forest"] = np.column_stack([
        RandomForestRegressor(n_estimators=args.trees, min_samples_leaf=leaf, max_features=feat,
                              n_jobs=2, random_state=seed, criterion="squared_error").fit(a, y).predict(q)
        for leaf, feat in RF_GRID])
    return predictions, ndog


def selected_fast_predictions(engine, zt, zq, info, y, args, seed, choices):
    kernels, ndog = engine.kernels(zt, zq, info)
    predictions = {}
    for family, pairs in kernels.items():
        selected = choices[family]
        which, ilam = divmod(selected, len(kb.LAMBDAS))
        kt, kq = pairs[which]
        predictions[family] = kb.eig_predictions(kt, kq, y, [kb.LAMBDAS[ilam]])[:, 0]
    params = candidate_parameters("random_forest")[choices["random_forest"]]
    pr = info["pruned_local"]
    model = RandomForestRegressor(n_estimators=args.trees, n_jobs=2, random_state=seed, **params)
    model.fit(np.ascontiguousarray(zt[pr].T, dtype=np.float32), y)
    predictions["random_forest"] = model.predict(np.ascontiguousarray(zq[pr].T, dtype=np.float32))
    return predictions, ndog


def update_aggregate(out, phase, y, ids):
    families = FAST_FAMILIES if phase == "fast" else ("bayesc",)
    predictions = {family: np.full((10, len(y)), np.nan) for family in families}
    records = []
    for path in sorted((out / "folds").glob(phase + "_r*_f*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as src:
            rep, test = record["repeat"], src["test_indices"]
            for family in families:
                assert np.isnan(predictions[family][rep, test]).all()
                predictions[family][rep, test] = src[family]
        records.append(record)
    atomic_npz(out / (phase + "_oof_predictions.npz"), y=y, ids=ids, **predictions)
    completed = [r for r in range(10) if all(np.isfinite(predictions[f][r]).all() for f in families)]
    state = {"phase": phase, "completed_outer_folds": len(records), "total_outer_folds": 50,
             "completed_repeats": completed, "status": "complete" if len(records) == 50 else "partial",
             "folds": records}
    kb.save_json(out / (phase + "_state.json"), state)
    report = {"status": state["status"], "completed_outer_folds": len(records), "completed_repeats": completed,
              "warning": "Only fully completed repeats summarized; unfinished OOF cells remain NaN in npz. No historical metrics mixed."}
    if completed:
        report["models"] = {f: kb.summarize(y, p[completed]) for f, p in predictions.items()}
    if phase == "bayesc":
        report["outer_diagnostic_failures"] = sum(not r["outer_bayesc"]["diagnostics_pass"] for r in records)
        report["inner_diagnostic_failures"] = sum(not c["diagnostics_pass"] for r in records for item in r["inner_fits"] for c in item["bayesc"])
    kb.save_json(out / (phase + "_metrics.json"), report)
    return len(records)


def run_phase(engine, y, ids, args, out, phase, limit=None):
    families = FAST_FAMILIES if phase == "fast" else ("bayesc",)
    finished_now = 0
    for rep in range(10):
        for fold, (train, test) in enumerate(KFold(5, shuffle=True, random_state=rep).split(y)):
            name = f"{phase}_r{rep:02d}_f{fold:02d}"
            foldpath = out / "folds" / (name + ".json")
            if foldpath.exists() and foldpath.with_suffix(".npz").exists():
                continue
            began = time.perf_counter()
            pooled = {f: np.full((len(train), len(candidate_parameters(f))), np.nan) for f in families}
            inner_records = []
            inner_seed = 1000 + rep * 5 + fold
            base = int(kb.PROTOCOL["random_seed"] + rep * 100000 + fold * 10000)
            for inner, (it, iv) in enumerate(KFold(5, shuffle=True, random_state=inner_seed).split(train)):
                a, v = train[it], train[iv]
                split = split_name(rep, fold, inner)
                taskpath = out / "tasks" / (phase + "_" + split + ".npz")
                taskmeta = taskpath.with_suffix(".json")
                if taskpath.exists() and taskmeta.exists():
                    with np.load(taskpath, allow_pickle=False) as src:
                        for family in families:
                            pooled[family][iv] = src[family]
                    inner_records.append(json.loads(taskmeta.read_text(encoding="utf-8")))
                    continue
                zt, zq, info, audit = engine.prepare(a, v, out / "preprocessing" / (split + ".npz"))
                item = {"inner_fold": inner, "train_indices": a.tolist(), "validation_indices": v.tolist(),
                        "seed": base + inner * 100, "preprocessing": audit}
                if phase == "fast":
                    pred, ndog = fast_predictions(engine, zt, zq, info, y[a], args, base + inner * 100)
                    item["dog1_training_qc_markers"] = ndog
                else:
                    pr = info["pruned_local"]
                    preds, diagnostics = [], []
                    for candidate, pi in enumerate(PI_GRID):
                        pi_path = out / "tasks" / ("bayesc_" + split + f"_pi{candidate}.npz")
                        prediction, diagnostic = run_bayesc(zt[pr], zq[pr], y[a], pi,
                            base + inner * 100 + candidate * 7, args.inner_burn, args.inner_draws, args.extensions, pi_path)
                        preds.append(prediction)
                        diagnostics.append(diagnostic)
                    pred = {"bayesc": np.column_stack(preds)}
                    item["bayesc"] = diagnostics
                for family in families:
                    assert np.isfinite(pred[family]).all()
                    pooled[family][iv] = pred[family]
                atomic_npz(taskpath, train_indices=a, validation_indices=v, **pred)
                kb.save_json(taskmeta, item)
                inner_records.append(item)
                del zt, zq
                print(f"{phase} rep={rep + 1}/10 fold={fold + 1}/5 inner={inner + 1}/5 saved", flush=True)
            assert all(np.isfinite(p).all() for p in pooled.values())
            losses = {f: np.mean((p - y[train, None]) ** 2, axis=0) for f, p in pooled.items()}
            choices = {f: int(loss.argmin()) for f, loss in losses.items()}
            split = split_name(rep, fold)
            zt, zq, info, audit = engine.prepare(train, test, out / "preprocessing" / (split + ".npz"))
            record = {"repeat": rep, "fold": fold, "train_indices": train.tolist(), "test_indices": test.tolist(),
                      "train_ids": ids[train].tolist(), "test_ids": ids[test].tolist(), "inner_seed": inner_seed,
                      "outer_model_seed": base + 9000, "preprocessing": audit, "inner_fits": inner_records,
                      "selection": {f: {"candidate_index": choices[f], "parameters": candidate_parameters(f)[choices[f]],
                                        "pooled_inner_mse": float(losses[f][choices[f]]),
                                        "all_candidate_parameters": candidate_parameters(f),
                                        "all_candidate_pooled_mse": losses[f].tolist()} for f in families}}
            if phase == "fast":
                pred, ndog = selected_fast_predictions(engine, zt, zq, info, y[train], args, base + 9000, choices)
                record["dog1_training_qc_markers"] = ndog
            else:
                pr = info["pruned_local"]
                prediction, diagnostic = run_bayesc(zt[pr], zq[pr], y[train], PI_GRID[choices["bayesc"]],
                    base + 9000, args.outer_burn, args.outer_draws, args.extensions,
                    out / "tasks" / ("bayesc_" + split + ".npz"))
                pred = {"bayesc": prediction}
                record["outer_bayesc"] = diagnostic
            record["test_metrics"] = {f: kb.metrics(y[test], pred[f]) for f in families}
            record["seconds"] = time.perf_counter() - began
            arrays = {"train_indices": train, "test_indices": test, **pred}
            arrays.update({"inner_oof_" + f: p for f, p in pooled.items()})
            atomic_npz(foldpath.with_suffix(".npz"), **arrays)
            kb.save_json(foldpath, record)
            del zt, zq
            count = update_aggregate(out, phase, y, ids)
            print(f"{phase}: completed {count}/50 outer folds; current seconds={record['seconds']:.1f}; "
                  + str({f: record['selection'][f]['parameters'] for f in families}), flush=True)
            finished_now += 1
            if limit is not None and finished_now >= limit:
                return
    update_aggregate(out, phase, y, ids)


def selftest(out):
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(123)
    x = (rng.random((50, 83)) < rng.uniform(.1, .5, (50, 1))).astype(float)
    bits, counts = pack_binary(x), x.sum(1).astype(np.int64)
    assert all(popcount(np.uint64(v)) == int(v).bit_count() for v in (0, 1, 2 ** 64 - 1, 2 ** 63 + 19))
    corr2 = np.corrcoef(x) ** 2
    error = max(abs(binary_r2(bits, counts, 83, j, k) - corr2[j, k]) for j in range(50) for k in range(50))
    assert error < 1e-12
    chrom, pos = np.ones(50, np.int64), np.arange(50, dtype=np.int64) * 5000
    selected, score, pairs = strict_ld(bits, counts, 83, chrom, pos, WINDOW_BP, R2_THRESHOLD)
    dense_score = np.ones(50)
    dense_keep = []
    for j in range(50):
        for k in range(j):
            if pos[j] - pos[k] <= WINDOW_BP:
                dense_score[j] += corr2[j, k]
                dense_score[k] += corr2[j, k]
        if not any(pos[j] - pos[k] <= WINDOW_BP and corr2[j, k] > .2 for k in dense_keep):
            dense_keep.append(j)
    assert np.array_equal(np.flatnonzero(selected), dense_keep)
    assert np.max(abs(score - dense_score)) < 1e-11
    # r=.3 gives r^2=.09: retained by corrected .2 rule, rejected by old |r|<.2.
    special = np.zeros((2, 100))
    special[0, :50] = 1
    special[1, :32] = 1
    special[1, 50:68] = 1
    r2 = binary_r2(pack_binary(special), np.array([50, 50]), 100, 0, 1)
    sk, _, _ = strict_ld(pack_binary(special), np.array([50, 50]), 100,
                         np.ones(2, np.int64), np.array([0, 10]), 100000, .2)
    assert r2 < .2 and np.sqrt(r2) > .2 and sk.all()
    # Allele 0 is a real allele. Tie selects smallest occurring code; tri-allelic rejected.
    raw = np.array([[0, 0, 2, 2, 3, 3], [1, 2, 3, 1, 0, 0], [0, 0, 0, 0, 1, 1]], np.uint8)
    keep, minor, count = allele_selection(raw, np.arange(4, dtype=np.int64))
    assert keep.tolist() == [True, False, False] and minor[0] == 0 and count[0] == 2
    meta = pd.DataFrame({"chrom": np.ones(50, int), "pos": pos})
    codes = (x.T.astype(np.uint8) * 2)
    tr, qu = np.arange(60), np.arange(60, 83)
    e1 = Preprocessor(codes, meta)
    a1, _, i1, _ = e1.prepare(tr, qu)
    mutated = codes.copy()
    mutated[qu] = rng.integers(0, 4, size=(len(qu), 50))
    a2, _, i2, _ = Preprocessor(mutated, meta).prepare(tr, qu)
    for key in ("indexes", "minor", "mean", "sd", "pruned_local", "ld_score"):
        assert np.array_equal(i1[key], i2[key]), key
    assert np.array_equal(a1, a2)
    kernels, _ = e1.kernels(*e1.prepare(tr, qu)[:3])
    for family, candidates in kernels.items():
        for kt, kq in candidates:
            assert np.isfinite(kt).all() and np.isfinite(kq).all()
            assert abs(kt.mean()) < 1e-10
            assert linalg.eigvalsh(kt).min() > -1e-9
    # One-marker exact posterior with fixed variances; intercept is orthogonal.
    z = np.array([[-1., -1., 1., 1.]])
    y = np.array([-.3, -.7, .4, .6])
    pi = .2
    bm, pip, mu, trace, checks = gibbs_chain(z, z, y, pi, 1000, 20000, 7701, False)
    sigma_e, tau = .5, .5 / pi
    v = sigma_e / (4 + sigma_e / tau)
    m = v * float(z @ y) / sigma_e
    logbf = .5 * (np.log(v / tau) + m * m / v)
    probability = float(1 / (1 + np.exp(-(np.log(pi / (1 - pi)) + logbf))))
    assert abs(pip[0] - probability) < .02
    assert abs(bm[0] - probability * m) < .02
    assert np.any(pip < 1) and np.any(pip > 0)
    duplicate = gibbs_chain(z, z, y, pi, 1000, 20000, 7701, False)
    assert np.array_equal(trace, duplicate[3])
    # Verify joint scale-MH acceptance against FULL density + Jacobian.
    active_b = np.array([.2, -.4, .1])
    tau0, errorvar, ab, bb, c = .7, .8, 3., .3, 1.17
    rss0, rss1 = 2.1, 2.3
    def logdensity(b, tau, rss):
        return -rss / (2 * errorvar) - len(b) / 2 * np.log(tau) - np.dot(b, b) / (2 * tau) - (ab + 1) * np.log(tau) - bb / tau
    direct_ratio = logdensity(c * active_b, c * c * tau0, rss1) - logdensity(active_b, tau0, rss0) + (len(active_b) + 2) * np.log(c)
    compact_ratio = -(rss1 - rss0) / (2 * errorvar) - 2 * ab * np.log(c) + bb / tau0 * (1 - c ** -2)
    assert abs(direct_ratio - compact_ratio) < 1e-12
    hierarchical = gibbs_chain(z, z, y, pi, 1000, 2000, 8821, True)
    assert 0 < hierarchical[4][0] < 1 and hierarchical[4][1] < 1e-10
    assert np.max(abs(hierarchical[2] + hierarchical[0] @ z - hierarchical[3][:, 3:].mean(0))) < 1e-10
    diag_rng = np.random.default_rng(91)
    iid = diag_rng.normal(size=(2, 2000, 5))
    diagnostic = chain_diagnostics(iid)
    assert diagnostic["passes_prespecified_thresholds"]
    iid[1, :, 0] += 4
    assert not chain_diagnostics(iid)["passes_prespecified_thresholds"]
    test_y = np.array([10., 11., 12., 13., 14.])
    pred = np.column_stack([test_y + 1, test_y + 2])
    assert np.mean((pred - test_y[:, None]) ** 2, axis=0).tolist() == [1., 4.]
    report = {"passed": True, "binary_r2_max_error_vs_numpy": error,
              "ld_score_max_error_vs_dense": float(np.max(abs(score - dense_score))),
              "pruning_matches_dense": True, "r_vs_r2_regression_test_r2": r2,
              "query_genotype_mutation_leaves_training_filter_scales_ld_unchanged": True,
              "allele_zero_and_triallelic_tests": True, "kernel_centering_psd_tests": True,
              "bayesc_exact_one_marker_inclusion_probability": probability,
              "bayesc_gibbs_inclusion_probability": float(pip[0]),
              "bayesc_exact_posterior_mean": probability * m, "bayesc_gibbs_posterior_mean": float(bm[0]),
              "bayesc_seed_reproducible": True, "scale_mh_log_density_jacobian_test": True,
              "hierarchical_gibbs_residual_and_posterior_prediction_checks": True,
              "diagnostics_detect_shifted_chain": True,
              "pooled_mse_test": True, "device": str(kb.DEVICE)}
    kb.save_json(out / "unit_tests.json", report)
    print("All numerical/isolation tests passed: " + json.dumps(report), flush=True)


def benchmark(out, codes, y, metadata, ids):
    engine = Preprocessor(codes, metadata)
    train, test = next(KFold(5, shuffle=True, random_state=0).split(y))
    innertrain, innerval = next(KFold(5, shuffle=True, random_state=1000).split(train))
    reports = []
    for name, tr, qu in (("outer", train, test), ("inner", train[innertrain], train[innerval])):
        started = time.perf_counter()
        zt, zq, info, audit = engine.prepare(tr, qu)
        kernels, ndog = engine.kernels(zt, zq, info)
        for pairs in kernels.values():
            for kt, kq in pairs:
                assert np.isfinite(kb.eig_predictions(kt, kq, y[tr])).all()
        audit.update({"split": name, "total_preprocess_plus_kernel_seconds": time.perf_counter() - started,
                      "dog1_markers": ndog, "training_ids": ids[tr].tolist()})
        reports.append(audit)
        print("LD feasibility " + json.dumps(audit), flush=True)
    report = {"decision": "strict_per_training_split", "evidence": reports,
              "reference_panel_used": False, "evaluation_genotypes_used_for_LD": "Only the current training split; no outer/inner held-out, external29, Dryad, or 1307 reference panel used.",
              "estimated_300_split_ld_seconds": 50 * (reports[0]["ld_compute_seconds"] + 5 * reports[1]["ld_compute_seconds"]),
              "threadpools": threadpool_info()}
    kb.save_json(out / "ld_feasibility.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("tests", "benchmark", "smoke", "run"), default="tests")
    parser.add_argument("--phase", choices=("all", "fast", "bayesc"), default="all")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--inner-burn", type=int, default=1000)
    parser.add_argument("--inner-draws", type=int, default=2000)
    parser.add_argument("--outer-burn", type=int, default=2000)
    parser.add_argument("--outer-draws", type=int, default=4000)
    parser.add_argument("--extensions", type=int, default=1)
    args = parser.parse_args()
    out = args.output.resolve()
    permitted = (HERE.parent / "results").resolve()
    if permitted not in out.parents or out == permitted:
        raise ValueError("Outputs must be a new own subtree of revision/results, never original inputs or another analysis output.")
    if min(args.trees, args.inner_burn, args.inner_draws, args.outer_burn, args.outer_draws) < 1 or args.extensions < 0:
        raise ValueError("Positive sampling/tree counts required.")
    assert kb.PROTOCOL["outer_cv"]["repeats"] == 10 and kb.PROTOCOL["outer_cv"]["folds"] == 5
    assert kb.PROTOCOL["inner_cv"]["folds"] == 5
    with threadpool_limits(2):
        selftest(out)
        if args.mode == "tests":
            return
        codes, y, metadata, ids, _ = kb.load_dormancy()
        if args.mode == "benchmark":
            benchmark(out, codes, y, metadata, ids)
            return
        if args.mode == "smoke":
            out = out / "smoke_validated_mixing"
            args.trees, args.inner_burn, args.inner_draws = 30, 100, 200
            args.outer_burn, args.outer_draws, args.extensions = 100, 200, 0
        elif args.trees != 300 or args.inner_burn < 1000 or args.inner_draws < 2000 or args.outer_burn < 2000 or args.outer_draws < 4000:
            raise ValueError("Reduced settings allowed only in separate smoke mode, not production nested CV.")
        spec = specification(args, codes, y, ids, metadata)
        initialize(out, spec)
        engine = Preprocessor(codes, metadata)
        runtime = {"pid": os.getpid(), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "command": sys.argv, "mode": args.mode, "requested_phase": args.phase,
                   "status": "running", "device": "cpu", "threads": 2}
        kb.save_json(out / "runtime.json", runtime)
        try:
            for phase in (("fast", "bayesc") if args.phase == "all" else (args.phase,)):
                runtime["current_phase"] = phase
                kb.save_json(out / "runtime.json", runtime)
                run_phase(engine, y, ids, args, out, phase, limit=1 if args.mode == "smoke" else None)
        except BaseException as error:
            runtime.update(status="failed_or_interrupted", error=repr(error), finished=time.strftime("%Y-%m-%d %H:%M:%S"))
            kb.save_json(out / "runtime.json", runtime)
            raise
        runtime.update(status="smoke_complete" if args.mode == "smoke" else "requested_phase_complete",
                       finished=time.strftime("%Y-%m-%d %H:%M:%S"))
        kb.save_json(out / "runtime.json", runtime)
        print("Requested run finished; consult phase_state.json for completion, not smoke metrics.", flush=True)


if __name__ == "__main__":
    main()
