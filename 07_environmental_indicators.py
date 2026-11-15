# -*- coding: utf-8 -*-
"""
07 - Environmental indicators of residue management (no field data)
====================================================================
A. Burning: dry matter burned, emissions (CO2, CO, CH4, N2O, PM2.5, BC) and
   losses of carbon and nitrogen, with the IPCC approach

        E = A x M x Cf x EF

   A  burned area (ha/year): sample share of 'burned' x area of the harvested
      cereal stratum (to be replaced by the Olofsson estimate of script 06b);
   M  residue available (t DM/ha) = grain yield x straw/grain ratio x dry matter
      x share of straw not baled;
   Cf combustion factor; EF emission factors (g/kg DM).
   Every coefficient is drawn from its range (uniform) N_MONTE_CARLO times;
   results are reported as median and 5-95% interval.

B. Soil exposure: for every classified point-year
   - bare_start = date of tillage or burning (none for 'retained');
   - bare_end   = first later observation with NDVI >= NDVI_COVER (regrowth or
                  new crop), otherwise EXPOSURE_END (censored: the series stops
                  on 15 November);
   - rainfall during the bare period from ERA5-Land (07a): total mm, number of
     days >= HEAVY_RAIN_MM, and share of the season's rainfall (Jul-15 Nov)
     that fell on bare soil;

Outputs (output/):
   env_burning_mc.csv          Monte Carlo summary per year (and total)
   env_exposure_points.parquet per point-year exposure
   env_exposure_summary.csv    exposure by class and year
"""
import numpy as np
import pandas as pd
import geopandas as gpd

import config as C

rng = np.random.default_rng(2026)


# ----------------------------------------------------------------------------
# A. Burning
# ----------------------------------------------------------------------------
def u(rangepair, n):
    lo, hi = rangepair
    return rng.uniform(lo, hi, n)


def yields(years):
    if C.ISTAT_YIELD_FILE.exists():
        y = pd.read_csv(C.ISTAT_YIELD_FILE, sep=None, engine="python")
        y = y.set_index("year")["yield_t_ha"]
        print(f"ISTAT yields read from {C.ISTAT_YIELD_FILE.name}")
        return {yr: (float(y.get(yr, np.nan)),) * 2 if yr in y.index else C.YIELD_RANGE for yr in years}
    print("ISTAT yield file not found: using YIELD_RANGE for every year")
    return {yr: C.YIELD_RANGE for yr in years}


def stratum_area_ha():
    """Area of the cereal stratum, from the 01 log (19.9% of 9,986 km2) unless
    a better estimate is written to output/stratum_area_ha.txt."""
    f = C.OUT_DIR / "stratum_area_ha.txt"
    if f.exists():
        return float(f.read_text().strip())
    return 0.199 * 998_600.0


def adjusted_burned_share(yr):
    """Error-adjusted burned share of year yr and its standard error, from
    output/validation/area_estimates.csv (06b). The pooled relative standard
    error is applied to every year."""
    f = C.OUT_DIR / "validation" / "area_estimates.csv"
    if not f.exists():
        return None
    a = pd.read_csv(f)
    a["year"] = a["year"].astype(str)
    b = a[a["cls"] == "burned"].set_index("year")
    if str(yr) not in b.index or "pooled" not in b.index:
        return None
    rel = (b.at["pooled", "adj_share_ci95"] / 1.96) / b.at["pooled", "adj_share"]
    m = float(b.at[str(yr), "adj_share"])
    return m, m * rel


def burning(cl):
    n = C.N_MONTE_CARLO
    years = sorted(cl["year"].unique())
    yl = yields(years)
    py = pd.read_parquet(C.OUT_DIR / "point_years.parquet")
    rows = []
    for yr in years:
        sub = cl[(cl["year"] == yr) & (cl["cls"] != "unclassified")]
        allp = py[py["year"] == yr]
        # share of the stratum that was harvested winter cereal that year
        p_cereal = allp["is_cereal"].mean()
        p_burn = (sub["cls"] == "burned").mean()
        if adjusted_burned_share(yr) is not None:
            p_burn = adjusted_burned_share(yr)[0]
        # binomial sampling uncertainty of the two shares
        pc = rng.beta(allp["is_cereal"].sum() + 1, (~allp["is_cereal"]).sum() + 1, n)
        pb = rng.beta((sub["cls"] == "burned").sum() + 1, (sub["cls"] != "burned").sum() + 1, n)
        adj = adjusted_burned_share(yr)
        if adj is not None:                       # error-adjusted share (script 06b)
            m, se = adj
            pb = np.clip(rng.normal(m, se, n), 0, 1)
        if C.ISTAT_YIELD_FILE.exists():
            ist = pd.read_csv(C.ISTAT_YIELD_FILE, sep=None, engine="python").set_index("year")
            harvested = float(ist.at[yr, "area_ha"]) if yr in ist.index and "area_ha" in ist else None
        else:
            harvested = None
        # burned area = burned share x official durum wheat area (ISTAT) when available;
        # otherwise stratum area x cereal share (the stratum over-estimates the cereal area)
        A = (harvested * pb) if harvested else stratum_area_ha() * pc * pb      # ha
        M = (u(yl[yr], n) * u(C.STRAW_GRAIN_RATIO, n) * u(C.DRY_MATTER, n)
             * u(C.RESIDUE_LEFT, n))                                        # t DM / ha
        burned_dm = A * M * u(C.COMBUSTION_FACTOR, n)                       # t DM
        out = {"year": yr, "share_cereal": p_cereal, "share_burned": p_burn,
               "area_burned_ha": A, "dm_burned_t": burned_dm,
               "C_lost_t": burned_dm * u(C.C_CONTENT, n),
               "N_lost_t": burned_dm * u(C.N_CONTENT, n)}
        for g, rngpair in C.EMISSION_FACTORS.items():
            out[f"{g}_t"] = burned_dm * u(rngpair, n) / 1000.0              # g/kg = kg/t -> t
        rows.append(out)

    summ = []
    keys = [k for k in rows[0] if k not in ("year", "share_cereal", "share_burned")]
    for r in rows:
        s = {"year": r["year"], "share_cereal": round(r["share_cereal"], 3),
             "share_burned": round(r["share_burned"], 3)}
        for k in keys:
            s[f"{k}_median"] = np.median(r[k])
            s[f"{k}_p05"], s[f"{k}_p95"] = np.percentile(r[k], [5, 95])
        summ.append(s)
    tot = {"year": "total"}
    for k in keys:
        v = np.sum([r[k] for r in rows], axis=0)
        tot[f"{k}_median"] = np.median(v)
        tot[f"{k}_p05"], tot[f"{k}_p95"] = np.percentile(v, [5, 95])
    summ.append(tot)
    df = pd.DataFrame(summ)
    df.to_csv(C.OUT_DIR / "env_burning_mc.csv", index=False)
    show = ["year", "area_burned_ha_median", "dm_burned_t_median", "CO2_t_median",
            "PM2.5_t_median", "C_lost_t_median", "N_lost_t_median"]
    print("\nBurning (median of Monte Carlo; 5-95% in the CSV):")
    print(df[show].round(0).to_string(index=False))
    return df


# ----------------------------------------------------------------------------
# B. Soil exposure
# ----------------------------------------------------------------------------
def load_rain(pts):
    if not C.ERA5_FILE.exists():
        print(f"{C.ERA5_FILE.name} not found: run 07a first (rainfall part skipped)")
        return None
    import xarray as xr
    ds = xr.open_dataset(C.ERA5_FILE)
    tp = ds["tp_mm"]
    lat = xr.DataArray(pts["lat"].values, dims="p")
    lon = xr.DataArray(pts["lon"].values, dims="p")
    s = tp.sel(lat=lat, lon=lon, method="nearest").to_pandas()   # time x points
    s.columns = pts["point_id"].values
    s.index = pd.to_datetime(s.index).normalize()
    return s.clip(lower=0)


def exposure(cl):
    pts = gpd.read_file(C.SAMPLE_FILE)[["point_id", "lon", "lat"]]
    rain = load_rain(pts)
    rows = []
    for yr, cly in cl[cl["cls"] != "unclassified"].groupby("year"):
        obs = pd.read_parquet(C.OUT_DIR / f"obs_indices_{yr}.parquet",
                              columns=["point_id", "date", "NDVI", "clear", "in_window"])
        obs = obs[obs["in_window"] & obs["clear"]].sort_values(["point_id", "date"])
        end_season = pd.Timestamp(f"{yr}-{C.EXPOSURE_END}")
        season_start = pd.Timestamp(f"{yr}-07-01")
        g = dict(tuple(obs.groupby("point_id")))
        for r in cly.itertuples():
            start = r.event_date if r.cls in ("tilled", "burned") else pd.NaT
            end, censored = pd.NaT, False
            if pd.notna(start):
                o = g.get(r.point_id)
                later = o[(o["date"] > start) & (o["NDVI"] >= C.NDVI_COVER)] if o is not None else []
                if len(later):
                    end = later["date"].iloc[0]
                else:
                    end, censored = end_season, True
            row = {"point_id": r.point_id, "year": yr, "cls": r.cls, "bare_start": start,
                   "bare_end": end, "censored": censored,
                   "bare_days": (end - start).days if pd.notna(start) else 0}
            if rain is not None and r.point_id in rain.columns:
                pr = rain[r.point_id]
                s0 = min(season_start, start) if pd.notna(start) else season_start
                season = pr[(pr.index >= s0) & (pr.index <= end_season)]
                row["rain_season_mm"] = season.sum()
                if pd.notna(start):
                    bare = pr[(pr.index >= start) & (pr.index < end)]
                    row["rain_bare_mm"] = bare.sum()
                    row["heavy_days_bare"] = int((bare >= C.HEAVY_RAIN_MM).sum())
                else:
                    row["rain_bare_mm"], row["heavy_days_bare"] = 0.0, 0
                row["share_rain_on_bare"] = (row["rain_bare_mm"] / row["rain_season_mm"]
                                             if row["rain_season_mm"] > 0 else np.nan)
            rows.append(row)
    ex = pd.DataFrame(rows)
    ex.to_parquet(C.OUT_DIR / "env_exposure_points.parquet", index=False)

    agg = {"bare_days": "median", "censored": "mean"}
    for c in ["rain_bare_mm", "heavy_days_bare", "share_rain_on_bare"]:
        if c in ex.columns:
            agg[c] = "median" if c != "share_rain_on_bare" else "mean"
    s = ex.groupby(["year", "cls"]).agg(agg).round(2)
    s.to_csv(C.OUT_DIR / "env_exposure_summary.csv")
    print("\nSoil exposure by class (median days bare; share censored at 15 Nov):")
    print(s.to_string())
    return ex


def main():
    cl = pd.read_parquet(C.OUT_DIR / "classes.parquet")
    burning(cl)
    exposure(cl)


if __name__ == "__main__":
    main()
