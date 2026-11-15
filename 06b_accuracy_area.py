# -*- coding: utf-8 -*-
"""
06b - Accuracy and error-adjusted areas (Olofsson et al., 2014) + Cohen's kappa
===============================================================================
Reads the blind photo-interpretation (output/validation/interpretation_sheet.csv,
filled by the interpreter) and the sample design (sample.csv: map class and
stratum size of every sample). Only the ACCURACY rows are used; calibration rows
are reported separately (they were used to tune the thresholds).

Reference labels: retained, tilled, burned, not_cereal, not_interpretable.
 - not_interpretable rows are dropped (reported);
 - not_cereal is kept as a fourth reference class ("other"): a commission error
   of every map class, so the map area of each class is corrected for it.

Estimators (stratified sampling; Olofsson et al. 2014, and Stehman 2014 when the
map assessed - the current classes.parquet - differs from the map used to draw
the strata, as after the threshold calibration):
   W_i   = N_i / N                          map share of class i (stratum size)
   p_ij  = W_i * n_ij / n_i.                error matrix in area proportions
   OA    = sum_k p_kk
   UA_i  = p_ii / p_i.      PA_j = p_jj / p._j
   p._j  = estimated area proportion of reference class j, with standard error
           SE(p._j) = sqrt( sum_i W_i^2 * (n_ij/n_i.)(1 - n_ij/n_i.) / (n_i. - 1) )
   95% CI = 1.96 SE; areas in ha = proportion x area of the harvested cereal
   point-years (ISTAT durum area of each year, as in script 07); the pooled
   area is the sum over all years (ha x years), the annual areas are in ha.
Annual areas: the yearly map shares are corrected with the pooled conditional
proportions n_ij / n_i. (the sample is too small for a separate matrix per year).

Second interpreter: on the first run the script writes
   output/validation/interpretation_sheet_2.csv
with 30 accuracy rows per map class and EMPTY labels, for the second
interpreter (never overwritten). When it has been filled, Cohen's kappa
between the two interpreters is computed on the rows labelled by both.

Outputs (output/validation/):
   accuracy_matrix.csv     error matrix in area proportions (+ counts)
   accuracy_summary.csv    OA, UA, PA with 95% CI
   area_estimates.csv      pooled and annual error-adjusted shares and areas
   kappa.txt               inter-interpreter agreement (when available)
"""
import numpy as np
import pandas as pd

import config as C

MAP = ["retained", "tilled", "burned"]
REF = MAP + ["not_cereal"]
VAL = C.OUT_DIR / "validation"
Z = 1.96


def read_sheet(f):
    s = pd.read_csv(f, sep=";", encoding="utf-8-sig", dtype=str)
    s["label"] = s["label"].fillna("").str.strip().str.lower().str.replace(" ", "_")
    return s


def load():
    smp = pd.read_csv(VAL / "sample.csv")
    # map being assessed = CURRENT classification (after calibration); the
    # sample strata stay the map classes at sampling time (column cls)
    cur = pd.read_parquet(C.OUT_DIR / "classes.parquet")[["point_id", "year", "cls"]]
    smp = smp.merge(cur.rename(columns={"cls": "map"}), on=["point_id", "year"], how="left")
    smp["map"] = smp["map"].fillna("excluded")
    sh = read_sheet(VAL / "interpretation_sheet.csv")
    d = smp[["sample_id", "cls", "map", "role", "year", "stratum_size"]].merge(
        sh[["sample_id", "label"]], on="sample_id", how="left")
    # calibration labels interpreted separately (used only to tune thresholds)
    fc = VAL / "calibration_labels.csv"
    if fc.exists():
        cl = read_sheet(fc).set_index("sample_id")["label"]
        empty = (d["role"] == "calibration") & (d["label"].fillna("") == "")
        d.loc[empty, "label"] = d.loc[empty, "sample_id"].map(cl).fillna("")
    bad = d[~d["label"].isin(REF + ["not_interpretable", ""])]
    if len(bad):
        print("Unknown labels (fix them in the sheet):")
        print(bad[["sample_id", "label"]].to_string(index=False))
    return smp, d


def second_sheet(smp):
    f = VAL / "interpretation_sheet_2.csv"
    if f.exists():
        return
    acc = smp[smp["role"] == "accuracy"]
    pick = (acc.groupby("cls", group_keys=False)
               .apply(lambda g: g.sample(min(30, len(g)), random_state=11)))
    sh = read_sheet(VAL / "interpretation_sheet.csv")
    out = sh[sh["sample_id"].isin(pick["sample_id"])].copy()
    for c in ["label", "confidence_1to3", "date_of_change", "source_used", "interpreter", "notes"]:
        out[c] = ""
    out = out.sample(frac=1, random_state=12)
    out.to_csv(f, sep=";", index=False, encoding="utf-8-sig")
    print(f"Written {f.name}: {len(out)} rows for the second interpreter (labels empty)")


def _strat(acc, y, x=None):
    """Stratified estimate of mean(y) (or of the ratio sum y / sum x) and its
    variance; strata = map class at sampling time (column 'cls'), weights N_h/N."""
    g = acc.assign(_y=y.astype(float), _x=(x if x is not None else pd.Series(1.0, index=acc.index)).astype(float))
    N = g.groupby("cls")["stratum_size"].first()
    W = N / N.sum()
    st = g.groupby("cls").agg(n=("_y", "size"), ym=("_y", "mean"), xm=("_x", "mean"),
                              vy=("_y", "var"), vx=("_x", "var"))
    cxy = g.groupby("cls")[["_y", "_x"]].apply(lambda d: d["_y"].cov(d["_x"]) if len(d) > 1 else 0.0)
    st = st.join(W.rename("W")).fillna(0.0)
    Y, X = (st.W * st.ym).sum(), (st.W * st.xm).sum()
    if x is None:
        return Y, (st.W ** 2 * st.vy / st.n).sum()
    if X == 0:
        return np.nan, np.nan
    R = Y / X
    v = (st.W ** 2 * (st.vy + R ** 2 * st.vx - 2 * R * cxy.reindex(st.index).fillna(0)) / st.n).sum() / X ** 2
    return R, v


def olofsson(acc):
    """Stehman (2014) estimators: valid when the strata (map classes at sampling
    time, column 'cls') differ from the map being assessed (column 'map'), as
    after the threshold calibration; equal to Olofsson et al. (2014) otherwise.
    acc: accuracy rows with cls, map, label, stratum_size."""
    maps = MAP + (["excluded"] if (acc["map"] == "excluded").any() else [])
    p = pd.DataFrame(0.0, index=maps, columns=REF)
    for k in maps:
        for j in REF:
            p.at[k, j] = _strat(acc, (acc["map"] == k) & (acc["label"] == j))[0]
    pj, se_pj = {}, {}
    for j in REF:
        e, v = _strat(acc, acc["label"] == j)
        pj[j], se_pj[j] = e, np.sqrt(v)
    pj, se_pj = pd.Series(pj), pd.Series(se_pj)
    oa, v_oa = _strat(acc, acc["map"] == acc["label"])
    rows = {}
    for k in MAP:
        ua, v_ua = _strat(acc, (acc["map"] == k) & (acc["label"] == k), acc["map"] == k)
        pa, v_pa = _strat(acc, (acc["map"] == k) & (acc["label"] == k), acc["label"] == k)
        rows[k] = [ua, Z * np.sqrt(v_ua), pa, Z * np.sqrt(v_pa), int((acc["map"] == k).sum())]
    summ = pd.DataFrame(rows, index=["users_acc", "users_ci95", "producers_acc", "producers_ci95",
                                     "n_accuracy"]).T
    summ.loc["overall"] = [oa, Z * np.sqrt(v_oa), np.nan, np.nan, len(acc)]
    n = pd.crosstab(acc["map"], acc["label"]).reindex(index=maps, columns=REF, fill_value=0)
    # conditional proportions P(reference j | map k), for the annual correction
    q = p.div(p.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    return n, q, p, pj, se_pj, summ


def istat_area():
    if C.ISTAT_YIELD_FILE.exists():
        ist = pd.read_csv(C.ISTAT_YIELD_FILE, sep=None, engine="python").set_index("year")
        if "area_ha" in ist:
            return ist["area_ha"].astype(float)
    return None


def areas(q, pj, se_pj):
    cl = pd.read_parquet(C.OUT_DIR / "classes.parquet")
    cl = cl[cl["cls"].isin(MAP)]
    A = istat_area()
    q = q.reindex(MAP)
    rows = []
    for yr, g in cl.groupby("year"):
        w = g["cls"].value_counts(normalize=True).reindex(MAP, fill_value=0)
        adj = (q.mul(w, axis=0)).sum(axis=0)       # corrected reference shares
        a = A.get(yr, np.nan) if A is not None else np.nan
        for k in REF:
            rows.append(dict(year=yr, cls=k, map_share=w.get(k, 0.0), adj_share=adj[k],
                             area_ha=adj[k] * a))
    tot = A.loc[sorted(cl["year"].unique())].sum() if A is not None else np.nan
    for k in REF:
        rows.append(dict(year="pooled", cls=k, map_share=np.nan, adj_share=pj[k],
                         adj_share_ci95=Z * se_pj[k], area_ha=pj[k] * tot,
                         area_ci95_ha=Z * se_pj[k] * tot))
    return pd.DataFrame(rows)


def kappa():
    f2 = VAL / "interpretation_sheet_2.csv"
    if not f2.exists():
        return None
    a = read_sheet(VAL / "interpretation_sheet.csv")[["sample_id", "label"]]
    b = read_sheet(f2)[["sample_id", "label"]]
    m = a.merge(b, on="sample_id", suffixes=("_1", "_2"))
    m = m[(m["label_1"] != "") & (m["label_2"] != "")]
    m = m[~m[["label_1", "label_2"]].isin(["not_interpretable"]).any(axis=1)]
    if len(m) < 10:
        return None
    cats = sorted(set(m["label_1"]) | set(m["label_2"]))
    t = pd.crosstab(m["label_1"], m["label_2"]).reindex(index=cats, columns=cats, fill_value=0)
    n = t.values.sum()
    po = np.trace(t.values) / n
    pe = (t.sum(axis=1).values * t.sum(axis=0).values).sum() / n ** 2
    k = (po - pe) / (1 - pe)
    se = np.sqrt(po * (1 - po) / (n * (1 - pe) ** 2))
    txt = (f"Cohen's kappa = {k:.3f} (95% CI {k - Z * se:.3f} to {k + Z * se:.3f}); "
           f"observed agreement {po:.1%}; n = {n}\n\n{t.to_string()}\n")
    (VAL / "kappa.txt").write_text(txt, encoding="utf-8")
    return txt


def main():
    smp, d = load()
    second_sheet(smp)

    cal = d[(d["role"] == "calibration") & (d["label"] != "")]
    if len(cal):
        print(f"\nCalibration rows labelled: {len(cal)}/90 - agreement with the current map: "
              f"{(cal['map'] == cal['label']).mean():.1%}")
        print(pd.crosstab(cal["map"], cal["label"]).to_string())

    acc = d[d["role"] == "accuracy"]
    done = acc[acc["label"] != ""]
    print(f"\nAccuracy rows labelled: {len(done)}/{len(acc)}")
    ni = done[done["label"] != "not_interpretable"].groupby("cls").size()
    print(f"not interpretable: {(done['label'] == 'not_interpretable').sum()}")
    if len(done) < len(acc) or (ni.reindex(MAP).fillna(0) < 2).any():
        print("Accuracy sample not complete: estimates are provisional")
    use = done[done["label"].isin(REF)]
    if (use.groupby("cls").size().reindex(MAP).fillna(0) < 2).any():
        print("Too few labelled rows per class for the estimators: stop here.")
        return

    n, q, p, pj, se_pj, summ = olofsson(use)
    mat = p.copy()
    mat["map_total"] = p.sum(axis=1)
    mat.loc["reference_total"] = list(pj) + [1.0]
    mat.round(4).to_csv(VAL / "accuracy_matrix.csv")
    n.to_csv(VAL / "accuracy_counts.csv")
    summ.round(4).to_csv(VAL / "accuracy_summary.csv")
    ar = areas(q, pj, se_pj)
    ar.round(4).to_csv(VAL / "area_estimates.csv", index=False)

    print("\nError matrix (area proportions; rows = map, columns = reference):")
    print(mat.round(3).to_string())
    print("\nAccuracies (95% CI):")
    print((summ * [100, 100, 100, 100, 1]).round(1).to_string())
    print("\nError-adjusted shares, pooled 2018-2025 (95% CI):")
    pool = ar[ar["year"] == "pooled"].set_index("cls")
    for k in REF:
        print(f"  {k:<11} {pool.at[k, 'adj_share']:.1%} +/- {pool.at[k, 'adj_share_ci95']:.1%}"
              f"   ~{pool.at[k, 'area_ha']:,.0f} ha summed over the years")
    k = kappa()
    print("\n" + (k if k else "Kappa: second sheet not filled yet (interpretation_sheet_2.csv)"))


if __name__ == "__main__":
    main()
