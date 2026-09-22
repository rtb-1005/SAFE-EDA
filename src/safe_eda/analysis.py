"""Recompute participant-level experimental tables from fixed metric exports."""


from __future__ import annotations


from pathlib import Path


from zipfile import ZipFile


import numpy as np


import pandas as pd


from scipy import stats


BOOT = 20_000


BOOT_SEED = 20260829


TOST_BOUND = 0.05


CLASSES = {0: "baseline", 1: "stress", 2: "amusement"}


WESAD_SCRATCH = ("B1_TRUE_WESAD_20260816", "safe_scratch")


WESAD_TRANSFER = ("U2_SAFE_EDA_EDABE_TO_DUAL_TARGET_20260819", "safe_edabe_transfer")


WEARABLE_RUN = "D1_WEARABLE_LOSO_ALLARMS_20260821_recovery_01"


NORM_RUN_WEARABLE = "E2_NORMALIZATION_LADDER_N1A_20260823"


NORM_RUN_WESAD_Z = "E2B_WESAD_NORMALIZATION_PER_SUBJECT_Z_20260825"


NORM_RUN_WESAD_D = "E2_NORMALIZATION_LADDER_N1B_20260823"


HOP_RUN_WESAD = "E3_HOP_LADDER_N6A_20260825_triage_01"


HOP_RUN_WEARABLE = "E3_HOP_LADDER_N6B_HOP1_COST_TRIAGE_20260825"


PERM_RUN = "E1_WESAD_LABEL_PERMUTATION_20260823"


EPOCH_RUN = "N2_EPOCH_CURVES_20260825_triage_01"


HOP_WINDOWS = {"wesad": {1: 113778, 4: 28430, 60: 1896, 240: 479},
               "wearable": {4: 148798, 60: 9955, 240: 2513}}


def subject_scores(df, run, arm, metric="macro_f1", **filt):
    """Fold-level metric averaged over seeds, indexed by held-out subject."""
    q = (df.run_id == run) & (df.arm == arm) & (df.metric == metric)
    for k, v in filt.items():
        q &= (df[k] == v)
    sub = df[q]
    if sub.empty:
        return pd.Series(dtype=float)
    return sub.groupby("held_out_subject")["value"].mean()


def paired(a: pd.Series, b: pd.Series) -> dict:
    """Subject-paired comparison of a minus b."""
    idx = sorted(set(a.index) & set(b.index))
    x, y = a.loc[idx].to_numpy(float), b.loc[idx].to_numpy(float)
    d = x - y
    n = len(idx)
    rng = np.random.default_rng(BOOT_SEED)
    boot = rng.choice(d, size=(BOOT, n), replace=True).mean(axis=1)
    lo95, hi95 = np.percentile(boot, [2.5, 97.5])
    lo90, hi90 = np.percentile(boot, [5.0, 95.0])
    sd = d.std(ddof=1)
    # TOST against the fixed absolute bound
    se = sd / np.sqrt(n)
    if se > 0:
        p_lo = stats.t.sf((d.mean() + TOST_BOUND) / se, n - 1)
        p_hi = stats.t.cdf((d.mean() - TOST_BOUND) / se, n - 1)
        tost_p = max(p_lo, p_hi)
    else:
        tost_p = np.nan
    return {
        "n_subjects": n,
        "mean_a": x.mean(), "mean_b": y.mean(), "delta": d.mean(),
        "wins": int((d > 0).sum()),
        "wilcoxon_p": stats.wilcoxon(x, y).pvalue if n >= 6 else np.nan,
        "paired_t_p": stats.ttest_rel(x, y).pvalue,
        "cohens_dz": d.mean() / sd if sd > 0 else np.nan,
        "ci95_low": lo95, "ci95_high": hi95,
        "ci90_low": lo90, "ci90_high": hi90,
        "tost_bound": TOST_BOUND, "tost_p": tost_p,
        "tost_verdict": ("superior" if lo95 > 0 else
                         "equivalent_within_0.05" if (lo90 > -TOST_BOUND and hi90 < TOST_BOUND) else
                         "inconclusive"),
    }


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(running, 1.0)
    return adj


def t01_protocol_ladder(m):
    mf = m[(m.metric == "macro_f1") & (m.task == "three_class")]
    rows = []
    spec = [("wesad", "wesad_random_window", "random_window", "safe_scratch"),
            ("wesad", "wesad_subject_holdout", "fixed_subject_holdout", "safe_scratch"),
            ("wesad", "wesad_loso15", "loso", "safe_scratch_B1"),
            ("wearable", "wearable_random_window", "random_window", "safe_scratch"),
            ("wearable", "wearable_subject_holdout", "fixed_subject_holdout", "safe_scratch"),
            ("wearable", "wearable_loso", "loso", "safe_scratch")]
    for ds, proto, label, scratch_arm in spec:
        s = mf[(mf.dataset == ds) & (mf.protocol == proto) & (mf.arm == scratch_arm)]["value"]
        t = mf[(mf.dataset == ds) & (mf.protocol == proto) & (mf.arm == "safe_edabe_transfer")]["value"]
        rows.append({"dataset": ds, "evaluation_split": label,
                     "scratch_macro_f1": round(s.mean(), 4) if len(s) else np.nan,
                     "transfer_macro_f1": round(t.mean(), 4) if len(t) else np.nan,
                     "n_scratch": len(s), "n_transfer": len(t)})
    return pd.DataFrame(rows)


def t04_edabe(per_subject, counts):
    """EDABE source task vs the two reference predictions shipped with the benchmark.

    `post` (post-processed) is the stronger of the two and is used as the primary
    comparator; `pred` (raw pipeline output) is reported alongside it.
    """
    rows = []
    for split, label in [("official_test", "official test"), ("validation", "validation")]:
        ours = per_subject[per_subject.split == split].groupby("record")["artifact_f1_at_0p5"].mean()
        for baseline in ["post", "pred"]:
            b = counts[(counts.split == split) & (counts.threshold == 0.5) & (counts.baseline == baseline)]
            f1 = b.assign(f1=lambda d: 2 * d.tp / (2 * d.tp + d.fp + d.fn)).groupby("record")["f1"].mean()
            idx = sorted(set(ours.index) & set(f1.index))
            x, y = ours.loc[idx].to_numpy(), f1.loc[idx].to_numpy()
            d = x - y
            rng = np.random.default_rng(BOOT_SEED)
            boot = rng.choice(d, size=(BOOT, len(d)), replace=True).mean(axis=1)
            rows.append({"split": label,
                         "reference": {"post": "post-processed (primary)",
                                       "pred": "raw pipeline output"}[baseline],
                         "n_records": len(idx),
                         "safe_eda_artifact_f1": round(x.mean(), 4),
                         "benchmark_reference_f1": round(y.mean(), 4),
                         "delta": round(d.mean(), 4), "wins": int((d > 0).sum()),
                         "wilcoxon_p": round(stats.wilcoxon(x, y).pvalue, 5),
                         "cohens_dz": round(d.mean() / d.std(ddof=1), 3),
                         "ci95_low": round(np.percentile(boot, 2.5), 4),
                         "ci95_high": round(np.percentile(boot, 97.5), 4)})
    return pd.DataFrame(rows)


def t05_normalization(m, s):
    rows = []
    spec = [("wesad", "global_train", *WESAD_SCRATCH, *WESAD_TRANSFER, {}),
            ("wesad", "per_subject_z", NORM_RUN_WESAD_Z, "safe_scratch", NORM_RUN_WESAD_Z, "safe_edabe_transfer", {}),
            ("wesad", "differenced", NORM_RUN_WESAD_D, "safe_scratch", NORM_RUN_WESAD_D, "safe_edabe_transfer", {}),
            ("wearable", "global_train", WEARABLE_RUN, "safe_scratch", WEARABLE_RUN, "safe_edabe_transfer", {})]
    for nz in ["per_subject_z", "per_subject_robust", "baseline_referenced", "differenced"]:
        spec.append(("wearable", nz, NORM_RUN_WEARABLE, "safe_scratch",
                     NORM_RUN_WEARABLE, "safe_edabe_transfer", {"normalization": nz}))
    for ds, nz, r_s, a_s, r_t, a_t, filt in spec:
        sc = subject_scores(s, r_s, a_s, **filt)
        tr = subject_scores(s, r_t, a_t, **filt)
        r = paired(tr, sc)
        rows.append({"dataset": ds, "normalization": nz,
                     "scratch_macro_f1": round(r["mean_b"], 4),
                     "transfer_macro_f1": round(r["mean_a"], 4),
                     "delta": round(r["delta"], 4),
                     "wins": f'{r["wins"]}/{r["n_subjects"]}',
                     "wilcoxon_p": round(r["wilcoxon_p"], 5),
                     "cohens_dz": round(r["cohens_dz"], 3),
                     "ci95": f'[{r["ci95_low"]:+.4f}, {r["ci95_high"]:+.4f}]'})
    out = pd.DataFrame(rows)
    out["wilcoxon_p_holm"] = holm(out.wilcoxon_p.to_numpy()).round(5)
    return out


def t06_hop(s):
    rows = []
    for ds, run, hops in [("wesad", HOP_RUN_WESAD, [1, 4, 60, 240]),
                          ("wearable", HOP_RUN_WEARABLE, [4, 60, 240])]:
        for hop in hops:
            if ds == "wesad" and hop == 1:
                sc, tr = subject_scores(s, *WESAD_SCRATCH), subject_scores(s, *WESAD_TRANSFER)
            elif ds == "wearable" and hop == 60:
                sc = subject_scores(s, WEARABLE_RUN, "safe_scratch")
                tr = subject_scores(s, WEARABLE_RUN, "safe_edabe_transfer")
            else:
                sc = subject_scores(s, run, "safe_scratch", hop_samples=hop)
                tr = subject_scores(s, run, "safe_edabe_transfer", hop_samples=hop)
            r = paired(tr, sc)
            rows.append({"dataset": ds, "hop_samples": hop,
                         "window_overlap_pct": round((1 - hop / 240) * 100, 1),
                         "train_windows_per_fold": HOP_WINDOWS[ds][hop],
                         "scratch_macro_f1": round(r["mean_b"], 4),
                         "transfer_macro_f1": round(r["mean_a"], 4),
                         "delta": round(r["delta"], 4),
                         "wins": f'{r["wins"]}/{r["n_subjects"]}',
                         "wilcoxon_p": float(f'{r["wilcoxon_p"]:.3g}'),
                         "cohens_dz": round(r["cohens_dz"], 3)})
    return pd.DataFrame(rows)


def t07_stability(t05, t06):
    rows = []
    for ds in ["wesad", "wearable"]:
        cells = pd.concat([
            t05[t05.dataset == ds][["scratch_macro_f1", "transfer_macro_f1"]],
            t06[(t06.dataset == ds) & (t06.hop_samples != (1 if ds == "wesad" else 60))]
                [["scratch_macro_f1", "transfer_macro_f1"]]])
        sc, tr = cells.scratch_macro_f1.to_numpy(), cells.transfer_macro_f1.to_numpy()
        rows.append({"dataset": ds, "n_configurations": len(cells),
                     "scratch_min": round(sc.min(), 4), "scratch_max": round(sc.max(), 4),
                     "scratch_sd": round(sc.std(ddof=1), 4),
                     "transfer_min": round(tr.min(), 4), "transfer_max": round(tr.max(), 4),
                     "transfer_sd": round(tr.std(ddof=1), 4),
                     "sd_ratio_scratch_over_transfer": round(sc.std(ddof=1) / tr.std(ddof=1), 2),
                     "transfer_superior_cells": int((tr > sc).sum())})
    df = pd.DataFrame(rows)
    pos, n = int(df.transfer_superior_cells.sum()), int(df.n_configurations.sum())
    df["sign_test_p_one_sided_pooled"] = round(stats.binomtest(pos, n, 0.5, alternative="greater").pvalue, 5)
    return df


def t08_permutation(s):
    tr = subject_scores(s, *WESAD_TRANSFER)
    sh = subject_scores(s, PERM_RUN, "shuffled_labels")
    sc = subject_scores(s, *WESAD_SCRATCH)
    total = tr.mean() - sc.mean()
    rows = []
    for label, a, b in [("transfer - scratch (total gain)", tr, sc),
                        ("shuffled - scratch (pretraining itself)", sh, sc),
                        ("transfer - shuffled (artifact label semantics)", tr, sh)]:
        r = paired(a, b)
        rows.append({"contrast": label, "delta": round(r["delta"], 4),
                     "wins": f'{r["wins"]}/{r["n_subjects"]}',
                     "wilcoxon_p": round(r["wilcoxon_p"], 5),
                     "cohens_dz": round(r["cohens_dz"], 3),
                     "share_of_total_gain": round(r["delta"] / total * 100, 1)})
    return pd.DataFrame(rows)


def t10_class_level(dis):
    ov = dis[dis.stratum_type == "true_class"].copy()
    ov["net_change_pp"] = ((ov.treatment_fixed - ov.treatment_broke) / ov.n_pairs * 100).round(2)
    ov["class_name"] = ov.stratum.astype(int).map(CLASSES)
    keep = ["dataset", "normalization", "class_name", "treatment_fixed",
            "treatment_broke", "n_pairs", "net_change_pp"]
    return ov[keep].sort_values(["dataset", "normalization", "class_name"]).reset_index(drop=True)


def t12_headline(s):
    rows = []
    for metric in ["macro_f1", "accuracy", "balanced_accuracy", "cohen_kappa",
                   "f1_class0", "f1_class1", "f1_class2"]:
        v = subject_scores(s, NORM_RUN_WESAD_Z, "safe_edabe_transfer", metric=metric)
        rng = np.random.default_rng(BOOT_SEED)
        boot = rng.choice(v.to_numpy(), size=(BOOT, len(v)), replace=True).mean(axis=1)
        name = CLASSES.get(int(metric[-1]), metric) if metric.startswith("f1_class") else metric
        rows.append({"metric": ("f1_" + name) if metric.startswith("f1_class") else name,
                     "value": round(v.mean(), 4),
                     "subject_sd": round(v.std(ddof=1), 4),
                     "ci95_low": round(np.percentile(boot, 2.5), 4),
                     "ci95_high": round(np.percentile(boot, 97.5), 4),
                     "n_subjects": len(v)})
    return pd.DataFrame(rows)


def t13_epochs(ec):
    e = ec[ec.run_id == EPOCH_RUN]
    g = (e.groupby(["dataset", "arm", "epoch"])
           .agg(heldout_macro_f1=("heldout_macro_f1", "mean"),
                train_loss=("train_loss", "mean"), n_fits=("fold", "size"))
           .round(4).reset_index())
    return g


def t14_method_ladder(m):
    w = m[(m.dataset == "wesad") & (m.protocol == "wesad_loso15") &
          (m.task == "three_class") & (m.metric == "macro_f1")]
    g = (w.groupby("arm")["value"].agg(macro_f1="mean", sd=lambda x: x.std(ddof=1), n="size")
           .sort_values("macro_f1").round(4).reset_index())
    return g


def t15_equivalence(t05):
    keep = t05[["dataset", "normalization", "delta", "ci95", "wilcoxon_p"]].copy()
    return keep


def t16_transition(err):
    w = err[err.dataset == "wesad"]
    rows = []
    for radius in [0, 60, 240]:
        for arm in ["safe_scratch", "safe_edabe_transfer"]:
            base = w[w.arm == arm].groupby("subject")["correct"].mean()
            cut = w[(w.arm == arm) & (w.distance_to_label_change > radius)].groupby("subject")["correct"].mean()
            idx = sorted(set(base.index) & set(cut.index))
            d = cut.loc[idx] - base.loc[idx]
            kept = (w.distance_to_label_change > radius).mean()
            rows.append({"exclusion_radius_samples": radius, "arm": arm,
                         "fraction_windows_kept": round(kept, 4),
                         "accuracy_after_exclusion": round(cut.loc[idx].mean(), 4),
                         "delta_vs_no_exclusion": round(d.mean(), 4),
                         "wilcoxon_p": (round(stats.wilcoxon(cut.loc[idx], base.loc[idx]).pvalue, 4)
                                        if radius else np.nan)})
    return pd.DataFrame(rows)


def _wesad_cell(s, e4, norm, hop):
    """Subject-level scores for one (normalization, hop) cell, both arms."""
    if norm == "global_train":
        if hop == 1:
            return (subject_scores(s, *WESAD_SCRATCH),
                    subject_scores(s, *WESAD_TRANSFER))
        return (subject_scores(s, HOP_RUN_WESAD, "safe_scratch", hop_samples=hop),
                subject_scores(s, HOP_RUN_WESAD, "safe_edabe_transfer", hop_samples=hop))
    if hop == 1:
        return (subject_scores(s, NORM_RUN_WESAD_Z, "safe_scratch"),
                subject_scores(s, NORM_RUN_WESAD_Z, "safe_edabe_transfer"))
    q = lambda arm: e4[(e4.arm == arm) & (e4.metric == "macro_f1") & (e4.hop_samples == hop)] \
                      .groupby("held_out_subject")["value"].mean()
    return q("safe_scratch"), q("safe_edabe_transfer")


def t17_norm_hop_cross(s, e4, loss_e4):
    rows = []
    lg = s[s.metric == "train_loss_final"]
    for norm in ["global_train", "per_subject_z"]:
        for hop in [1, 4, 60, 240]:
            sc, tr = _wesad_cell(s, e4, norm, hop)
            r = paired(tr, sc)
            if norm == "global_train":
                sel = lg[(lg.run_id == HOP_RUN_WESAD) & (lg.hop_samples == hop)]
            else:
                sel = loss_e4[loss_e4.hop_samples == hop] if hop != 1 else None
            ls = (sel.groupby("arm")["value"].mean().to_dict() if sel is not None and len(sel) else {})
            rows.append({
                "normalization": norm, "hop_samples": hop,
                "window_overlap_pct": round((1 - hop / 240) * 100, 1),
                "train_windows_per_fold": HOP_WINDOWS["wesad"][hop],
                "scratch_macro_f1": round(r["mean_b"], 4),
                "transfer_macro_f1": round(r["mean_a"], 4),
                "delta": round(r["delta"], 4),
                "wins": f'{r["wins"]}/{r["n_subjects"]}',
                "wilcoxon_p": float(f'{r["wilcoxon_p"]:.3g}'),
                "cohens_dz": round(r["cohens_dz"], 3),
                "ci95": f'[{r["ci95_low"]:+.4f}, {r["ci95_high"]:+.4f}]',
                "scratch_train_loss_final": round(ls.get("safe_scratch", np.nan), 3),
                "transfer_train_loss_final": round(ls.get("safe_edabe_transfer", np.nan), 3),
            })
    out = pd.DataFrame(rows)
    out["wilcoxon_p_holm"] = holm(out.wilcoxon_p.to_numpy()).round(5)
    return out


def t18_interaction(s, e4):
    """Does the measured pretraining gain depend on the normalization? Subject-paired."""
    rows = []
    for hop in [1, 4, 60, 240]:
        sc_g, tr_g = _wesad_cell(s, e4, "global_train", hop)
        sc_z, tr_z = _wesad_cell(s, e4, "per_subject_z", hop)
        dg, dz = (tr_g - sc_g), (tr_z - sc_z)
        idx = sorted(set(dg.index) & set(dz.index))
        x, y = dg.loc[idx].to_numpy(), dz.loc[idx].to_numpy()
        d = x - y
        rng = np.random.default_rng(BOOT_SEED)
        boot = rng.choice(d, size=(BOOT, len(d)), replace=True).mean(axis=1)
        rows.append({"hop_samples": hop,
                     "gain_under_global_z": round(x.mean(), 4),
                     "gain_under_per_subject_z": round(y.mean(), 4),
                     "difference": round(d.mean(), 4),
                     "wilcoxon_p": float(f"{stats.wilcoxon(x, y).pvalue:.3g}"),
                     "ci95_low": round(np.percentile(boot, 2.5), 4),
                     "ci95_high": round(np.percentile(boot, 97.5), 4),
                     "n_subjects": len(idx)})
    return pd.DataFrame(rows)


def t19_stability_by_normalization(t17):
    """Cross-hop stability of each arm, computed separately within each normalization."""
    rows = []
    for norm, g in t17.groupby("normalization", sort=False):
        sc, tr = g.scratch_macro_f1.to_numpy(), g.transfer_macro_f1.to_numpy()
        rows.append({"normalization": norm, "n_hops": len(g),
                     "scratch_range": round(sc.max() - sc.min(), 4),
                     "scratch_sd": round(sc.std(ddof=1), 4),
                     "transfer_range": round(tr.max() - tr.min(), 4),
                     "transfer_sd": round(tr.std(ddof=1), 4),
                     "sd_ratio_scratch_over_transfer": round(sc.std(ddof=1) / tr.std(ddof=1), 2)})
    return pd.DataFrame(rows)


def t20_normalization_value_by_hop(s, e4):
    """What per-subject normalization is worth to each arm, at each hop."""
    rows = []
    for hop in [1, 4, 60, 240]:
        sc_g, tr_g = _wesad_cell(s, e4, "global_train", hop)
        sc_z, tr_z = _wesad_cell(s, e4, "per_subject_z", hop)
        for arm, a, b in [("safe_scratch", sc_z, sc_g), ("safe_edabe_transfer", tr_z, tr_g)]:
            r = paired(a, b)
            rows.append({"hop_samples": hop, "arm": arm,
                         "global_z": round(r["mean_b"], 4),
                         "per_subject_z": round(r["mean_a"], 4),
                         "delta": round(r["delta"], 4),
                         "wins": f'{r["wins"]}/{r["n_subjects"]}',
                         "wilcoxon_p": float(f'{r["wilcoxon_p"]:.3g}')})
    return pd.DataFrame(rows)


def t11_calibration(scores):
    rows = []
    for norm in ["per_subject_z", "differenced"]:
        per = {}
        for row in scores[scores.normalization == norm].itertuples(index=False):
            per.setdefault((row.arm, row.subject), []).append((row.ece, row.brier))
        arms = {}
        for arm in ["safe_scratch", "safe_edabe_transfer"]:
            subs = sorted({s for (a, s) in per if a == arm})
            arms[arm] = (
                pd.Series({s: np.mean([v[0] for v in per[(arm, s)]]) for s in subs}),
                pd.Series({s: np.mean([v[1] for v in per[(arm, s)]]) for s in subs}))
        for metric, i in [("ECE", 0), ("Brier", 1)]:
            sc, tr = arms["safe_scratch"][i], arms["safe_edabe_transfer"][i]
            r = paired(tr, sc)
            rows.append({"dataset": "wesad", "normalization": norm, "metric": metric,
                         "scratch": round(r["mean_b"], 4), "transfer": round(r["mean_a"], 4),
                         "delta": round(r["delta"], 4),
                         "transfer_better_in": f'{r["n_subjects"] - r["wins"]}/{r["n_subjects"]}',
                         "wilcoxon_p": round(r["wilcoxon_p"], 5)})
    return pd.DataFrame(rows)


def t21_participant_grid(s, e4):
    rows = []
    for normalization in ["global_train", "per_subject_z"]:
        for hop in [1, 4, 60, 240]:
            scratch, transfer = _wesad_cell(s, e4, normalization, hop)
            for arm, scores in [("safe_edabe_transfer", transfer), ("safe_scratch", scratch)]:
                for participant, score in scores.items():
                    rows.append(dict(normalization=normalization, hop_samples=hop, arm=arm,
                                     held_out_subject=participant, macro_f1=score))
    return pd.DataFrame(rows).sort_values(["normalization", "hop_samples", "arm", "held_out_subject"])


def literature_tables(coding: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build the supplementary study-code matrix and reporting counts."""
    columns = {
        "paper_id": "study",
        "window_length_reported": "window_length",
        "window_hop_reported": "window_hop",
        "explicit_overlap_reported": "explicit_overlap",
        "split_type_reported": "split_type",
        "loso_described": "loso",
        "metric_aggregation_reported": "metric_aggregation",
        "normalization_code": "normalization",
    }
    study_codes = coding[list(columns)].rename(columns=columns)
    total = len(study_codes)
    specs = [
        ("Window length", "window_length", True),
        ("Window hop", "window_hop", True),
        ("Explicit overlap", "explicit_overlap", True),
        ("Split type", "split_type", True),
        ("LOSO described", "loso", True),
        ("Metric aggregation", "metric_aggregation", False),
        ("Normalization statistics source", "normalization", True),
        ("Evaluated-user statistics used", None, False),
    ]
    rows = []
    z = stats.norm.ppf(0.975)
    for label, column, include_ci in specs:
        if column is None:
            count = int((coding["evaluated_user_stats"] == "Y").sum())
        elif column == "normalization":
            count = int((study_codes[column] != "--").sum())
        else:
            count = int((study_codes[column] == "1").sum())
        proportion = count / total
        denominator = 1 + z ** 2 / total
        center = (proportion + z ** 2 / (2 * total)) / denominator
        half_width = z * np.sqrt(
            proportion * (1 - proportion) / total + z ** 2 / (4 * total ** 2)
        ) / denominator
        interval = f"[{round(100 * (center - half_width))},{round(100 * (center + half_width))}]"
        rows.append({"field": label, "studies": f"{count}/{total}",
                     "percent": round(100 * proportion),
                     "ci95_percent": interval if include_ci else "--"})
    return {
        "S01_literature_study_codes": study_codes,
        "S02_literature_reporting_counts": pd.DataFrame(rows),
    }


def build_tables(inputs: Path, output: Path, literature: Path | None = None) -> list[Path]:
    """Read fixed inputs and write the experimental and literature tables."""
    with ZipFile(inputs) as archive:
        def read(name, **kwargs):
            with archive.open(name) as stream:
                return pd.read_csv(stream, **kwargs)
        m = read("metrics.csv", low_memory=False)
        s = read("supplemental_metrics.csv", low_memory=False)
        ec = read("epoch_curves.csv", low_memory=False)
        eps = read("artifact_scores.csv")
        ebc = read("artifact_reference_counts.csv")
        dis = read("disagreement_counts.csv")
        err = read("transition_errors.csv", low_memory=False)
        e4 = read("grid_metrics.csv", low_memory=False)
        calibration = read("calibration_scores.csv", float_precision="round_trip")
    loss_e4 = e4[e4.metric == "train_loss_final"]
    t05 = t05_normalization(m, s)
    t06 = t06_hop(s)
    t17 = t17_norm_hop_cross(s, e4, loss_e4)
    tables = {
        "T01_protocol_ladder": t01_protocol_ladder(m),
        "T04_edabe_source_task": t04_edabe(eps, ebc),
        "T05_normalization_ladder": t05,
        "T06_hop_ladder": t06,
        "T07_cross_configuration_stability": t07_stability(t05, t06),
        "T08_label_permutation": t08_permutation(s),
        "T10_class_level_net_change": t10_class_level(dis),
        "T11_calibration_subject_level": t11_calibration(calibration),
        "T12_headline_result": t12_headline(s),
        "T13_epoch_curves": t13_epochs(ec),
        "T14_method_ladder": t14_method_ladder(m),
        "T15_equivalence": t15_equivalence(t05),
        "T16_transition_window_axis": t16_transition(err),
        "T17_normalization_hop_cross_grid": t17,
        "T18_interaction_test": t18_interaction(s, e4),
        "T19_stability_by_normalization": t19_stability_by_normalization(t17),
        "T20_normalization_value_by_hop": t20_normalization_value_by_hop(s, e4),
        "T21_subject_level_grid": t21_participant_grid(s, e4),
    }
    literature = literature or inputs.parent / "literature" / "study_coding.csv"
    tables.update(literature_tables(pd.read_csv(literature, keep_default_na=False)))
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, frame in tables.items():
        path = output / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths.append(path)
    return paths
