# -*- coding: utf-8 -*-
"""
03 - Spectral indices, winter-cereal check and post-harvest window
===================================================================
For each year:
  * computes NDVI, NDTI, NBR and BSI for every observation of script 02;
  * checks, point by point, that the season was a harvested winter cereal:
      - spring NDVI peak (PEAK_START..PEAK_END) >= NDVI_PEAK_MIN,
        from at least MIN_OBS_PEAK clear observations;
      - senescence (first clear NDVI < NDVI_SENESCENCE after the peak)
        before SENESCENCE_BY;
  * sets the post-harvest window:
      start = max(senescence + HARVEST_LAG_DAYS, WINDOW_START_MIN), end = WINDOW_END;
  * counts clear observations in the window (point-years with fewer than
    MIN_OBS_WINDOW are flagged, not dropped).

Observations with SCL 2 (dark area) or 3 (cloud shadow) are kept with a flag,
because char on burned stubble is often labelled that way; script 05 decides
how to use them. Phenology uses clear observations only (SCL_CLEAR).

Outputs:
  output/obs_indices_<year>.parquet   one row per observation, with in_window flag
  output/point_years.parquet          one row per point-year (all years)
"""
import numpy as np
import pandas as pd

import config as C


def add_indices(df):
    # Very dark pixels can give slightly negative reflectance after the -1000
    # offset (baseline >= 04.00); clip at 0 so that indices stay within [-1, 1]
    for b in ("B02", "B04", "B08", "B11", "B12"):
        df[b] = df[b].clip(lower=0)
    b2, b4, b8, b11, b12 = (df[b] for b in ("B02", "B04", "B08", "B11", "B12"))
    with np.errstate(divide="ignore", invalid="ignore"):
        df["NDVI"] = (b8 - b4) / (b8 + b4)
        df["NDTI"] = (b11 - b12) / (b11 + b12)
        df["NBR"] = (b8 - b12) / (b8 + b12)
        df["BSI"] = ((b11 + b4) - (b8 + b2)) / ((b11 + b4) + (b8 + b2))
    df["clear"] = df["scl"].isin(C.SCL_CLEAR)
    df["dark"] = df["scl"].isin([2, 3])
    return df


def phenology(g, year):
    """g: observations of one point in one year (sorted by date)."""
    d = lambda mmdd: pd.Timestamp(f"{year}-{mmdd}")
    clear = g[g["clear"] & g["NDVI"].notna()]
    spring = clear[(clear["date"] >= d(C.PEAK_START)) & (clear["date"] <= d(C.PEAK_END))]
    out = {"n_peak": len(spring), "peak_ndvi": np.nan, "peak_date": pd.NaT,
           "sen_date": pd.NaT, "is_cereal": False,
           "win_start": pd.NaT, "win_end": d(C.WINDOW_END),
           "n_win_clear": 0, "n_win_dark": 0, "enough_obs": False}
    if len(spring) == 0:
        return out
    i = spring["NDVI"].idxmax()
    out["peak_ndvi"], out["peak_date"] = spring.at[i, "NDVI"], spring.at[i, "date"]
    after = clear[(clear["date"] > out["peak_date"]) & (clear["NDVI"] < C.NDVI_SENESCENCE)]
    if len(after):
        out["sen_date"] = after["date"].iloc[0]
    out["is_cereal"] = bool(out["n_peak"] >= C.MIN_OBS_PEAK
                            and out["peak_ndvi"] >= C.NDVI_PEAK_MIN
                            and pd.notna(out["sen_date"])
                            and out["sen_date"] <= d(C.SENESCENCE_BY))
    if pd.notna(out["sen_date"]):
        start = max(out["sen_date"] + pd.Timedelta(days=C.HARVEST_LAG_DAYS), d(C.WINDOW_START_MIN))
        out["win_start"] = start
        w = g[(g["date"] >= start) & (g["date"] <= out["win_end"])]
        out["n_win_clear"] = int(w["clear"].sum())
        out["n_win_dark"] = int(w["dark"].sum())
        out["enough_obs"] = out["n_win_clear"] >= C.MIN_OBS_WINDOW
    return out


def run_year(year):
    f = C.OUT_DIR / f"s2_points_{year}.parquet"
    if not f.exists():
        print(f"{year}: {f.name} not found, skipped")
        return None
    obs = add_indices(pd.read_parquet(f))
    obs["date"] = pd.to_datetime(obs["date"])
    obs = obs.sort_values(["point_id", "date"])

    rows = []
    for pid, g in obs.groupby("point_id", sort=False):
        r = phenology(g, year)
        r.update(point_id=pid, year=year)
        rows.append(r)
    py = pd.DataFrame(rows)

    obs = obs.merge(py[["point_id", "win_start", "win_end"]], on="point_id", how="left")
    obs["in_window"] = (obs["date"] >= obs["win_start"]) & (obs["date"] <= obs["win_end"])
    obs.drop(columns=["win_start", "win_end"]).to_parquet(
        C.OUT_DIR / f"obs_indices_{year}.parquet", index=False)

    n, nc = len(py), int(py["is_cereal"].sum())
    ne = int((py["is_cereal"] & py["enough_obs"]).sum())
    print(f"{year}: {n:,} points observed | winter cereal {nc:,} ({nc / max(n, 1):.0%}) | "
          f"cereal with >= {C.MIN_OBS_WINDOW} clear obs in window {ne:,} | "
          f"median clear obs in window {py.loc[py.is_cereal, 'n_win_clear'].median():.0f}")
    return py


def main(years=None):
    out = [run_year(y) for y in (years or C.YEARS)]
    out = [p for p in out if p is not None]
    if out:
        allpy = pd.concat(out, ignore_index=True)
        cols = ["point_id", "year"] + [c for c in allpy.columns if c not in ("point_id", "year")]
        allpy[cols].to_parquet(C.OUT_DIR / "point_years.parquet", index=False)
        print(f"\nSaved {len(allpy):,} point-years to point_years.parquet")


if __name__ == "__main__":
    main()
