# -*- coding: utf-8 -*-
"""
04 - VIIRS active-fire detections near the sample points
========================================================
Reads all NASA FIRMS files (VIIRS 375 m: S-NPP, NOAA-20 = jpss1, NOAA-21 =
jpss2) found in data/firms/ or data/ - plain CSVs or the zip files of the
FIRMS yearly summaries (also 'all_countries'), read in chunks without
unzipping - keeps detections inside the bounding box of the sample points, in the
months of the post-harvest window, with confidence in VIIRS_MIN_CONF, and
finds every detection within VIIRS_RADIUS_M of each sample point.

The match with the NBR drop of each point-year (the date interval) is made in
script 05, so this script runs only once, independently of the thresholds.

Output: output/viirs_near_points.parquet with columns
point_id, year, date, satellite, confidence, frp, dist_m
"""
import zipfile

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree

import config as C


def _read_one(src, name, bbox):
    """Read one FIRMS CSV (plain or inside a zip) in chunks, keeping only rows in bbox."""
    keep = []
    for ch in pd.read_csv(src, chunksize=500_000, low_memory=False):
        ch.columns = [c.lower() for c in ch.columns]
        if "country_id" in ch.columns and "Italy" not in name:
            ch = ch[ch["country_id"].astype(str).isin(["ITA", "Italy"])]
        ch = ch[ch["longitude"].between(bbox[0], bbox[2]) & ch["latitude"].between(bbox[1], bbox[3])]
        if "instrument" in ch.columns:
            ch = ch[ch["instrument"].astype(str).str.upper().str.contains("VIIRS")]
        keep.append(ch)
    df = pd.concat(keep, ignore_index=True) if keep else pd.DataFrame()
    print(f"  {name}: {len(df):,} VIIRS detections in the study area")
    return df


def load_firms(bbox):
    """All FIRMS files in data/firms/ and data/: .csv, or .zip containing CSVs
    (country files such as viirs-snpp_2023_Italy.csv, or the 'all_countries' zips)."""
    files = sorted(set(C.FIRMS_DIR.glob("*.csv")) | set(C.FIRMS_DIR.glob("*.zip"))
                   | set(C.DATA_DIR.glob("viirs*.csv")) | set(C.DATA_DIR.glob("viirs*.zip")))
    if not files:
        raise FileNotFoundError(f"No FIRMS file in {C.FIRMS_DIR} or {C.DATA_DIR}")
    parts = []
    for f in files:
        if f.suffix.lower() == ".zip":
            with zipfile.ZipFile(f) as z:
                for n in z.namelist():
                    if n.lower().endswith(".csv") and ("italy" in n.lower() or "all_countries" in f.name
                                                       or "/" not in n):
                        if "all_countries" in f.name and "italy" not in n.lower() and len(z.namelist()) > 5:
                            continue        # per-country files inside: read only Italy
                        with z.open(n) as fh:
                            parts.append(_read_one(fh, f"{f.name}/{n}", bbox))
        else:
            parts.append(_read_one(f, f.name, bbox))
    fire = pd.concat([p for p in parts if len(p)], ignore_index=True)
    fire["date"] = pd.to_datetime(fire["acq_date"])
    fire["confidence"] = fire["confidence"].astype(str).str.lower().str[0]
    fire = fire.drop_duplicates(["latitude", "longitude", "date", "acq_time", "satellite"])
    return fire


def main():
    pts = gpd.read_file(C.SAMPLE_FILE)
    b = pts.to_crs("EPSG:4326").total_bounds
    pad = 0.02
    fire = load_firms([b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad])
    fire = fire[fire["date"].dt.year.isin(C.YEARS)
                & fire["date"].dt.month.between(6, 11)
                & fire["confidence"].isin(C.VIIRS_MIN_CONF)].copy()
    print(f"VIIRS detections in the area, Jun-Nov {C.YEARS[0]}-{C.YEARS[-1]}: {len(fire):,}")

    fg = gpd.GeoDataFrame(fire, geometry=gpd.points_from_xy(fire.longitude, fire.latitude),
                          crs="EPSG:4326").to_crs(C.WORK_CRS)
    fx, fy = fg.geometry.x.values, fg.geometry.y.values
    tree = cKDTree(np.c_[fx, fy])
    hits = tree.query_ball_point(np.c_[pts["x"], pts["y"]], r=C.VIIRS_RADIUS_M)

    rows = []
    for pid, x, y, idx in zip(pts["point_id"], pts["x"], pts["y"], hits):
        for j in idx:
            r = fg.iloc[j]
            rows.append(dict(point_id=pid, year=r["date"].year, date=r["date"],
                             satellite=r.get("satellite"), confidence=r["confidence"],
                             frp=r.get("frp"), dist_m=float(np.hypot(fx[j] - x, fy[j] - y))))
    out = pd.DataFrame(rows, columns=["point_id", "year", "date", "satellite",
                                      "confidence", "frp", "dist_m"])
    out.to_parquet(C.OUT_DIR / "viirs_near_points.parquet", index=False)
    print(f"Saved {len(out):,} point-detection pairs "
          f"({out['point_id'].nunique():,} points with at least one detection)")
    if len(out):
        print(out.groupby("year")["point_id"].nunique().rename("points with VIIRS").to_string())


if __name__ == "__main__":
    main()
