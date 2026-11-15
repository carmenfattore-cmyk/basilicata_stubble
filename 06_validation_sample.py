# -*- coding: utf-8 -*-
"""
06 - Stratified validation sample and image chips for photo-interpretation
==========================================================================
1. Draws a stratified random sample of classified point-years (strata = map
   classes retained / tilled / burned):
     - a CALIBRATION subset (CAL_PER_CLASS per class), used only to tune the
       thresholds of script 05;
     - an ACCURACY sample (VAL_PER_CLASS per class), never used for tuning,
       for the error matrix and the Olofsson area estimates (script 06b).
2. For every sampled point-year writes a PNG panel with CHIP_DATES Sentinel-2
   dates (1 km x 1 km): true colour (B4-B3-B2) and SWIR false colour
   (B12-B8-B4, char looks black, tilled soil brown/pink, stubble light), plus
   the NBR and NDTI trajectory of the point. The map class is NOT shown, so
   that the interpretation is blind.
3. Writes the interpretation sheet (CSV, open it in Excel) and a KML to look
   at the same points in the Google Earth Pro historical imagery.

Interpretation labels to type in the column 'label':
   retained | tilled | burned | not_cereal | not_interpretable
   (tilled = any visible soil disturbance, ploughing or shallow harrowing)

Outputs in output/validation/:
   sample.csv                 the sample (with map class, for script 06b)
   interpretation_sheet.csv   blind sheet to fill (sample_id, point, year, coords)
   points.kml                 for Google Earth Pro
   chips/<sample_id>.png      one panel per point-year
Re-running skips chips that already exist (resumable).
"""
import time

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds
from pyproj import Transformer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pystac_client
import planetary_computer

import config as C

CHIP_VERSION = "dates-v2"   # chips of older versions without an event are redrawn


def _old_chip(png):
    try:
        from PIL import Image
        with Image.open(png) as im:
            return im.info.get("Description") != CHIP_VERSION
    except Exception:
        return True

CLASSES = ["retained", "tilled", "burned"] if C.MERGE_TILLAGE else ["retained", "minimum", "tilled", "burned"]
GDAL_ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
                GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="3", GDAL_HTTP_TIMEOUT="60")


# ----------------------------------------------------------------------------
# 1. Sample
# ----------------------------------------------------------------------------
def draw_sample():
    cl = pd.read_parquet(C.OUT_DIR / "classes.parquet")
    cl = cl[cl["cls"].isin(CLASSES)]
    if C.VAL_YEARS:
        cl = cl[cl["year"].isin(C.VAL_YEARS)]
    rng = np.random.default_rng(C.VAL_SEED)
    rows = []
    for k in CLASSES:
        sub = cl[cl["cls"] == k]
        n = min(len(sub), C.CAL_PER_CLASS + C.VAL_PER_CLASS)
        pick = sub.iloc[rng.permutation(len(sub))[:n]].copy()
        pick["role"] = ["calibration"] * min(C.CAL_PER_CLASS, n) + \
                       ["accuracy"] * (n - min(C.CAL_PER_CLASS, n))
        pick["stratum_size"] = len(sub)
        rows.append(pick)
    s = pd.concat(rows).sample(frac=1, random_state=C.VAL_SEED).reset_index(drop=True)
    s.insert(0, "sample_id", [f"S{i + 1:04d}" for i in range(len(s))])
    pts = gpd.read_file(C.SAMPLE_FILE)[["point_id", "lon", "lat", "x", "y", "crop_class"]]
    return s.merge(pts, on="point_id", how="left")


def write_sheet_and_kml(s):
    C.VAL_DIR.mkdir(parents=True, exist_ok=True)
    s.to_csv(C.VAL_DIR / "sample.csv", index=False)
    sheet = s[["sample_id", "role", "point_id", "year", "lat", "lon"]].copy()
    sheet["chip"] = "chips/" + sheet["sample_id"] + ".png"
    for c in ["label", "confidence_1to3", "date_of_change", "source_used", "interpreter", "notes"]:
        sheet[c] = ""
    f = C.VAL_DIR / "interpretation_sheet.csv"
    if f.exists():
        print("interpretation_sheet.csv already exists: NOT overwritten (your labels are safe)")
    else:
        sheet.to_csv(f, index=False, sep=";", encoding="utf-8-sig")   # opens directly in Excel (IT)
    with open(C.VAL_DIR / "points.kml", "w", encoding="utf-8") as k:
        k.write('<?xml version="1.0" encoding="UTF-8"?>\n<kml xmlns="http://www.opengis.net/kml/2.2"><Document>\n')
        for r in sheet.itertuples():
            k.write(f'<Placemark><name>{r.sample_id} ({r.year})</name>'
                    f'<Point><coordinates>{r.lon:.6f},{r.lat:.6f},0</coordinates></Point></Placemark>\n')
        k.write("</Document></kml>\n")


# ----------------------------------------------------------------------------
# 2. Chips
# ----------------------------------------------------------------------------
def pick_dates(obs, row):
    """Candidate clear dates of the window, in order of priority: the first one
    (stubble), the two before and the two after the event, the last one, then
    the others evenly spread. make_chip keeps the first CHIP_DATES of them whose
    scene is not too cloudy."""
    o = obs[obs["clear"]].sort_values("date")
    if o.empty:
        return []
    d = list(o["date"])
    pri = [d[0]]
    if pd.notna(row["event_date"]):
        ev = pd.Timestamp(row["event_date"])
        before = [x for x in d if x < ev]
        after = [x for x in d if x >= ev]
        pri += before[-2:][::-1] + after[:2]
    pri.append(d[-1])
    # then dates spread evenly IN TIME over the whole window (not by index:
    # July has many more clear scenes than autumn), first coarse, then finer
    ts = pd.to_datetime(pd.Series(d))
    for k in (C.CHIP_DATES, 3 * C.CHIP_DATES):
        for t in pd.date_range(ts.iloc[0], ts.iloc[-1], periods=k):
            pri.append(d[int((ts - t).abs().argmin())])
    seen, out = set(), []
    for x in pri:
        if x not in seen:
            seen.add(x); out.append(x)
    return out


def find_items(cat, lon, lat, dates, tile, max_cloud=30):
    """First CHIP_DATES dates (in priority order) whose scene cloud cover is
    <= max_cloud; returned in time order."""
    out = []
    for d in dates:
        if len(out) >= C.CHIP_DATES:
            break
        day = pd.Timestamp(d).strftime("%Y-%m-%d")
        items = list(cat.search(collections=[C.S2_COLLECTION], intersects={"type": "Point", "coordinates": [lon, lat]},
                                datetime=f"{day}/{day}").items())
        items = [i for i in items if i.properties.get("s2:mgrs_tile") == tile] or items
        items = [i for i in items if i.properties.get("eo:cloud_cover", 0) <= max_cloud]
        if items:
            out.append(items[0])
    return sorted(out, key=lambda i: i.datetime)


def read_rgb(item, x, y, bands):
    signed = planetary_computer.sign(item)
    arrs = []
    with rasterio.Env(**GDAL_ENV):
        for b in bands:
            with rasterio.open(signed.assets[b].href) as src:
                t = Transformer.from_crs(C.WORK_CRS, src.crs, always_xy=True)
                cx, cy = t.transform(x, y)
                h = C.CHIP_HALF_M
                win = from_bounds(cx - h, cy - h, cx + h, cy + h, src.transform)
                a = src.read(1, window=win, out_shape=(100, 100), boundless=True, fill_value=0).astype(float)
                arrs.append(a)
    base = float(item.properties.get("s2:processing_baseline", "0") or 0)
    off = 1000 if base >= 4.0 else 0
    rgb = np.clip((np.dstack(arrs) - off) / 10000.0, 0, None)
    return rgb


def stretch(rgb, lo=0.0, hi=0.35):
    return np.clip((rgb - lo) / (hi - lo), 0, 1)


def make_chip(row, obs, cat, out_png):
    ob = obs[(obs["point_id"] == row["point_id"])]
    dates = pick_dates(ob[ob["in_window"]], row)
    items = find_items(cat, row["lon"], row["lat"], dates, ob["tile"].iloc[0])
    n = len(items)
    fig = plt.figure(figsize=(2.1 * max(n, 1), 6.4))
    gs = fig.add_gridspec(3, max(n, 1), height_ratios=[1, 1, 0.9])
    for j, it in enumerate(items):
        tc = stretch(read_rgb(it, row["x"], row["y"], ["B04", "B03", "B02"]), 0, 0.30)
        sw = stretch(read_rgb(it, row["x"], row["y"], ["B12", "B08", "B04"]), 0, 0.45)
        for r, img in enumerate([tc, sw]):
            ax = fig.add_subplot(gs[r, j])
            ax.imshow(img)
            ax.plot(49.5, 49.5, marker="s", ms=6, mfc="none", mec="yellow", mew=1.2)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(pd.Timestamp(it.datetime).strftime("%d %b"), fontsize=9)
    axt = fig.add_subplot(gs[2, :])
    w = ob[ob["clear"]].sort_values("date")
    axt.plot(w["date"], w["NBR"], "-o", ms=2, lw=1, label="NBR")
    axt.plot(w["date"], w["NDTI"], "-o", ms=2, lw=1, label="NDTI")
    for it in items:
        axt.axvline(pd.Timestamp(it.datetime).tz_localize(None), color="0.7", lw=0.6)
    axt.legend(fontsize=8, loc="lower left"); axt.tick_params(labelsize=8)
    fig.suptitle(f"{row['sample_id']}  -  point {row['point_id']}  -  {row['year']}  "
                 f"({row['lat']:.5f}, {row['lon']:.5f})   top: true colour, middle: SWIR B12-B8-B4",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110, metadata={"Description": CHIP_VERSION})
    plt.close(fig)


def main(make_chips=True, limit=None):
    f = C.VAL_DIR / "sample.csv"
    if f.exists():
        # the sample is drawn ONCE: after the threshold calibration the map changes,
        # but the sample (and the blind interpretation) must stay the same
        s = pd.read_csv(f)
        print("sample.csv already exists: the same sample is kept (not redrawn)")
    else:
        s = draw_sample()
        write_sheet_and_kml(s)
    print(f"Sample: {len(s)} point-years "
          f"({(s.role == 'calibration').sum()} calibration, {(s.role == 'accuracy').sum()} accuracy)")
    print(s.groupby(["cls", "role"]).size().unstack().to_string())
    if not make_chips:
        return
    chips = C.VAL_DIR / "chips"
    chips.mkdir(exist_ok=True)
    cat = pystac_client.Client.open(C.STAC_URL)
    todo = s if limit is None else s.head(limit)
    obs_cache = {}
    t0, done, failed = time.time(), 0, 0
    for _, row in todo.iterrows():
        png = chips / f"{row['sample_id']}.png"
        if png.exists():
            # retained points (no event) drawn with the old date choice are redrawn
            if pd.notna(row["event_date"]) or not _old_chip(png):
                continue
        y = int(row["year"])
        if y not in obs_cache:
            obs_cache[y] = pd.read_parquet(C.OUT_DIR / f"obs_indices_{y}.parquet")
        try:
            make_chip(row, obs_cache[y], cat, png)
            done += 1
        except Exception as e:
            failed += 1
            print(f"  {row['sample_id']}: failed ({e})")
        if done and done % 10 == 0:
            print(f"  {done} chips, {failed} failed, {(time.time() - t0) / done:.0f} s per chip")
    print(f"Chips done: {done}, failed: {failed} (run again to retry the failed ones)")


if __name__ == "__main__":
    # All 390 chips (about 30-40 s each; resumable). For a quick test: main(limit=5)
    main()
