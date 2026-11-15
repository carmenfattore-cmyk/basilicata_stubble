# -*- coding: utf-8 -*-
"""
05 - Rule-based classification of post-harvest trajectories
===========================================================
For each harvested winter-cereal point-year (script 03: is_cereal and
enough_obs) the post-harvest window is scanned in time order and the FIRST
management event is recorded:

  burned     abrupt NBR drop (>= STEP_BURN from the previous observation) that
             the landscape does not share (NBR change since the baseline lower
             than the tile-date median of all cereal points by >= ANOM_BURN),
             dark NIR (B08 <= B08_CHAR), and anomaly still <= -ANOM_BURN/2 at the
             next observation. The tile-date median removes the autumn decline
             of NBR due to sun angle, rain and widespread tillage. Dark-area
             and shadow observations (SCL 2/3) are allowed here, because fresh
             char is often labelled that way.
  tilled     NDTI < NDTI_SOIL and NDTI drop >= DNDTI_TILL from baseline,
             persisting for TILL_PERSIST clear observations (BSI is stored in the
             observation table for diagnostics but not used in the rule).
  retained   no burn, no tillage and no partial decrease until the window ends.
  minimum    NDTI decreased by at least DNDTI_MIN for TILL_PERSIST consecutive
             observations (shallow harrowing or uncertain) without reaching the
             tillage condition.
A burn followed by tillage is recorded as burned. With MERGE_TILLAGE (default)
'minimum' is merged into 'tilled' in the column 'cls'; the four-class result is
kept in 'cls_detail' for the supplementary material. Observations with NDVI above
NDVI_REGROWTH (weeds, volunteers, new crop) are not used.

Each burn is then checked against VIIRS (script 04): confirmed when a detection
within VIIRS_RADIUS_M falls between the last observation before the drop and
the first observation of the drop.

A sensitivity analysis multiplies each threshold by SENS_FACTORS, one at a time,
and reports the share of each class (Table 3 of the paper).

Outputs:
  output/classes.parquet       one row per classified point-year
  output/class_shares.csv      share of each class per year (unweighted sample)
  output/sensitivity.csv       class shares for each threshold perturbation
"""
import numpy as np
import pandas as pd

import config as C

CLASSES = ["retained", "tilled", "burned"] if C.MERGE_TILLAGE else ["retained", "minimum", "tilled", "burned"]


def default_params():
    return dict(STEP_BURN=C.STEP_BURN, ANOM_BURN=C.ANOM_BURN, B08_CHAR=C.B08_CHAR,
                B11_CHAR=C.B11_CHAR, NDTI_CHAR=C.NDTI_CHAR,
                NDTI_SOIL=C.NDTI_SOIL, DNDTI_TILL=C.DNDTI_TILL, DNDTI_MIN=C.DNDTI_MIN,
                NDVI_REGROWTH=C.NDVI_REGROWTH)


def _persist(flags, i, n):
    """True if flags[i:i+n] are all True (fewer than n obs left -> False)."""
    return i + n <= len(flags) and bool(flags[i:i + n].all())


def classify_one(w, p):
    """w: window observations of one point-year, sorted by date.
    Returns dict with class, event date, and diagnostics."""
    w = w[(w["NDVI"] <= p["NDVI_REGROWTH"]) | w["NDVI"].isna()]
    clear = w[w["clear"]]
    out = dict(cls="unclassified", event_date=pd.NaT, prev_date=pd.NaT,
               nbr0=np.nan, ndti0=np.nan, dnbr=np.nan, anom=np.nan, dndti=np.nan, n_used=len(w))
    if len(clear) < C.N_BASELINE_OBS + 1:
        return out
    base = clear.iloc[:C.N_BASELINE_OBS]
    nbr0, ndti0 = base["NBR"].median(), base["NDTI"].median()
    out.update(nbr0=nbr0, ndti0=ndti0)
    t0 = base["date"].iloc[-1]

    # ---- burn: all observations after the baseline (clear + dark) ----
    a = w[w["date"] > t0].reset_index(drop=True)
    an = a["anom"].to_numpy()
    char = (((a["B11"] <= p["B11_CHAR"]) & (a["NDTI"] <= p["NDTI_CHAR"]))
            | (a["B12"] >= C.B12_FLAME))
    burn = ((a["step"] <= -p["STEP_BURN"]) & (a["anom"] <= -p["ANOM_BURN"])
            & (a["B08"] <= p["B08_CHAR"]) & char).to_numpy()
    i_burn = next((i for i in range(len(a) - 1)
                   if burn[i] and an[i + 1] <= -p["ANOM_BURN"] / 2), None)

    # ---- tillage: clear observations after the baseline ----
    c = clear[clear["date"] > t0].reset_index(drop=True)
    dndti = ndti0 - c["NDTI"]
    till = ((c["NDTI"] < p["NDTI_SOIL"]) & (dndti >= p["DNDTI_TILL"])).to_numpy()
    i_till = next((i for i in range(len(c)) if till[i] and _persist(till, i, C.TILL_PERSIST)), None)

    # char also lowers NDTI: a tillage onset that already shows the burn
    # signature (NBR and B11 drop) is a burn, even if the NBR drop does not
    # persist (cloud gaps, partial burns, ash removed by wind)
    if i_till is not None:
        on = a.index[a["date"] == c.at[i_till, "date"]]
        if len(on) and burn[on[0]] and (i_burn is None or on[0] < i_burn):
            i_burn = int(on[0])
    d_burn = a.at[i_burn, "date"] if i_burn is not None else pd.NaT
    d_till = c.at[i_till, "date"] if i_till is not None else pd.NaT

    if i_burn is not None and (pd.isna(d_till) or d_burn <= d_till):
        prev = w[w["date"] < d_burn]["date"]
        out.update(cls="burned", event_date=d_burn,
                   prev_date=prev.iloc[-1] if len(prev) else pd.NaT,
                   dnbr=nbr0 - a.at[i_burn, "NBR"], anom=a.at[i_burn, "anom"])
    elif i_till is not None:
        prev = clear[clear["date"] < d_till]["date"]
        out.update(cls="tilled", event_date=d_till,
                   prev_date=prev.iloc[-1] if len(prev) else pd.NaT,
                   dndti=dndti.iloc[i_till])
    else:
        # partial decrease must also persist, so that single noisy obs do not count
        part = (dndti >= p["DNDTI_MIN"]).to_numpy()
        i_min = next((i for i in range(len(c)) if part[i] and _persist(part, i, C.TILL_PERSIST)), None)
        mx = dndti.rolling(C.TILL_PERSIST).min().max() if len(dndti) else 0.0
        if i_min is not None:
            prev = clear[clear["date"] < c.at[i_min, "date"]]["date"]
            out.update(cls="minimum", event_date=c.at[i_min, "date"],
                       prev_date=prev.iloc[-1] if len(prev) else pd.NaT, dndti=mx)
        else:
            out.update(cls="retained", dndti=mx)
    return out


def classify_all(obs_by_py, p):
    rows = []
    for (pid, yr), w in obs_by_py:
        r = classify_one(w, p)
        r.update(point_id=pid, year=yr)
        rows.append(r)
    df = pd.DataFrame(rows)
    df["cls_detail"] = df["cls"]                 # four-class detail kept for the supplement
    if C.MERGE_TILLAGE:
        df.loc[df["cls"] == "minimum", "cls"] = "tilled"
    return df


def load(years):
    py = pd.read_parquet(C.OUT_DIR / "point_years.parquet")
    py = py[py["year"].isin(years) & py["is_cereal"] & py["enough_obs"]]
    keep = set(zip(py["point_id"], py["year"]))
    parts = []
    for y in years:
        f = C.OUT_DIR / f"obs_indices_{y}.parquet"
        if f.exists():
            o = pd.read_parquet(f)
            o = o[o["in_window"]]
            o["year"] = y
            parts.append(o)
    obs = pd.concat(parts, ignore_index=True)
    obs = obs[[k in keep for k in zip(obs["point_id"], obs["year"])]]
    # irrigated summer crops (green in July-August) are not harvested winter cereal
    summer = obs[obs["clear"] & (obs["date"].dt.month.isin([7, 8]))]
    smax = summer.groupby(["point_id", "year"])["NDVI"].max()
    drop = set(smax[smax > C.SUMMER_NDVI_MAX].index)
    if drop:
        print(f"{len(drop):,} point-years excluded as summer crops (NDVI > {C.SUMMER_NDVI_MAX} in Jul-Aug)")
        obs = obs[[k not in drop for k in zip(obs["point_id"], obs["year"])]]
    obs = obs[(obs["NDVI"] <= C.NDVI_REGROWTH) | obs["NDVI"].isna()]
    obs = obs.sort_values(["point_id", "year", "date"]).reset_index(drop=True)
    return add_anomalies(obs), py


def add_anomalies(obs):
    """NBR step from the previous observation and NBR anomaly with respect to
    the median change of all cereal points of the same tile and date."""
    g = obs.groupby(["point_id", "year"], sort=False)
    clear = obs[obs["clear"]]
    base = (clear.groupby(["point_id", "year"]).head(C.N_BASELINE_OBS)
                 .groupby(["point_id", "year"])["NBR"].median().rename("nbr_base"))
    obs = obs.join(base, on=["point_id", "year"])
    obs["rel"] = obs["NBR"] - obs["nbr_base"]
    med = obs[obs["clear"]].groupby(["tile", "date"])["rel"].median().rename("rel_med")
    obs = obs.join(med, on=["tile", "date"])
    obs["anom"] = obs["rel"] - obs["rel_med"].fillna(0)
    obs["step"] = g["NBR"].diff()
    return obs


def viirs_confirm(cl):
    f = C.OUT_DIR / "viirs_near_points.parquet"
    cl["viirs"] = False
    if not f.exists():
        print("viirs_near_points.parquet not found: run 04 first (burns left unconfirmed)")
        return cl
    v = pd.read_parquet(f)
    b = cl[cl["cls"] == "burned"][["point_id", "year", "prev_date", "event_date"]]
    m = b.reset_index().merge(v[["point_id", "year", "date"]], on=["point_id", "year"])
    lo = m["prev_date"].fillna(m["event_date"] - pd.Timedelta(days=10))
    hit = m[(m["date"] >= lo) & (m["date"] <= m["event_date"])]["index"].unique()
    cl.loc[hit, "viirs"] = True
    return cl


def shares(cl):
    s = cl[cl["cls"] != "unclassified"]
    return (s.groupby("year")["cls"].value_counts(normalize=True)
             .unstack(fill_value=0).reindex(columns=CLASSES, fill_value=0))


def main(years=None, sensitivity=True):
    years = years or C.YEARS
    obs, py = load(years)
    groups = list(obs.groupby(["point_id", "year"], sort=False))
    print(f"{len(groups):,} point-years to classify ({', '.join(map(str, years))})")

    p = default_params()
    cl = classify_all(groups, p)
    cl = viirs_confirm(cl)
    cl.to_parquet(C.OUT_DIR / "classes.parquet", index=False)

    sh = shares(cl)
    sh.to_csv(C.OUT_DIR / "class_shares.csv")
    print("\nClass shares (sample, unweighted):")
    print((sh * 100).round(1).to_string())
    nb = (cl["cls"] == "burned").sum()
    print(f"\nBurns: {nb:,}; confirmed by VIIRS: {cl['viirs'].sum():,} "
          f"({cl['viirs'].sum() / max(nb, 1):.0%})")
    print(f"Unclassified (too few usable obs): {(cl['cls'] == 'unclassified').mean():.1%}")

    if sensitivity:
        base = shares(cl).mean()
        rows = [dict(param="baseline", factor=1.0, **base.to_dict())]
        for k in p:
            for fct in C.SENS_FACTORS:
                if fct == 1.0:
                    continue
                q = dict(p)
                q[k] = p[k] * fct
                s = shares(classify_all(groups, q)).mean()
                rows.append(dict(param=k, factor=fct, **s.to_dict()))
            print(f"  sensitivity done: {k}")
        sens = pd.DataFrame(rows)
        sens.to_csv(C.OUT_DIR / "sensitivity.csv", index=False)
        print("\nSensitivity (mean class share over years, %):")
        print((sens.set_index(["param", "factor"]) * 100).round(1).to_string())


if __name__ == "__main__":
    # First test on the year already downloaded, without sensitivity:
    main([2023], sensitivity=False)
