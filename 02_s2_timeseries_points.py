# -*- coding: utf-8 -*-
"""
02 - Sentinel-2 L2A time series at the sample points (local, no GEE)
=====================================================================
Reads Sentinel-2 L2A surface reflectance from the Microsoft Planetary Computer
STAC catalogue at the points of script 01, for each year in config.YEARS and
the season SEASON_START..SEASON_END.

For each scene:
  1. the scene classification layer (SCL, 20 m) is sampled at the points;
  2. only points whose SCL is in SCL_STORE are kept (clouds, cirrus, snow,
     saturated and no-data pixels are dropped);
  3. the bands are sampled at those points only. 10 m bands are read from
     their first overview (20 m), so all bands share the 20 m grid of B11/B12;
  4. reflectance = (DN - offset) / 10000, with offset = 1000 for processing
     baseline >= 04.00 (scenes from 25 Jan 2022) and 0 before.

Each point is assigned to ONE MGRS tile (the one in which it lies farthest from
the tile edge), so overlapping tiles are not read twice for the same point.

The script is RESUMABLE: one parquet per scene is written to
output/s2_points/<year>/, and scenes already done are skipped. If the
connection drops, just run it again.

Output: output/s2_points_<year>.parquet with columns
point_id, date, tile, item_id, scl, B02, B04, B08, B11, B12
"""
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from pyproj import Transformer
import pystac_client
import planetary_computer

import config as C

GDAL_ENV = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
    GDAL_HTTP_MULTIRANGE="YES",
    GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
    GDAL_HTTP_MAX_RETRY="5",
    GDAL_HTTP_RETRY_DELAY="3",
    GDAL_HTTP_TIMEOUT="60",          # no read can hang forever (e.g. after sleep/hibernation)
    GDAL_HTTP_CONNECTTIMEOUT="30",
    GDAL_CACHEMAX=512,
    VSI_CACHE="TRUE",
)
BANDS = C.BANDS_10M + C.BANDS_20M
COLUMNS = ["point_id", "date", "tile", "item_id", "scl"] + BANDS


# ----------------------------------------------------------------------------
# Catalogue
# ----------------------------------------------------------------------------
def open_catalog():
    return pystac_client.Client.open(C.STAC_URL)


def search_items(cat, bbox, start, end):
    search = cat.search(
        collections=[C.S2_COLLECTION], bbox=bbox, datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": C.MAX_SCENE_CLOUD}})
    items = list(search.items())
    # Keep one item per (platform, sensing time, tile): the most recent processing
    best = {}
    for it in items:
        key = (it.properties.get("platform"), it.datetime,
               it.properties.get("s2:mgrs_tile"))
        if key not in best or it.id > best[key].id:
            best[key] = it
    return sorted(best.values(), key=lambda i: i.datetime)


def tile_extent(item):
    """Full tile extent (xmin, ymin, xmax, ymax) and EPSG from the SCL asset."""
    a = item.assets["SCL"].extra_fields
    if "proj:transform" in a and "proj:shape" in a:
        t, shape = a["proj:transform"], a["proj:shape"]
        xmin, ymax = t[2], t[5]
        xmax, ymin = xmin + t[0] * shape[1], ymax + t[4] * shape[0]
        epsg = item.properties.get("proj:epsg") or a.get("proj:epsg")
        code = item.properties.get("proj:code") or a.get("proj:code")
        if epsg is None and code:
            epsg = int(str(code).split(":")[-1])
        if epsg is not None:
            return (xmin, ymin, xmax, ymax), int(epsg)
    # Fallback: read the header of the SCL file itself
    signed = planetary_computer.sign(item)
    with rasterio.Env(**GDAL_ENV), rasterio.open(signed.assets["SCL"].href) as src:
        b = src.bounds
        return (b.left, b.bottom, b.right, b.top), int(src.crs.to_epsg())


def assign_tiles(points, items):
    """Assign each point to the tile in which it lies farthest from the edge."""
    tiles = {}
    for it in items:
        tid = it.properties["s2:mgrs_tile"]
        if tid not in tiles:
            tiles[tid] = tile_extent(it)
    best_tile = np.full(len(points), "", dtype=object)
    best_dist = np.full(len(points), -np.inf)
    for tid, ((xmin, ymin, xmax, ymax), epsg) in tiles.items():
        t = Transformer.from_crs(C.WORK_CRS, f"EPSG:{epsg}", always_xy=True)
        x, y = t.transform(points["x"].values, points["y"].values)
        d = np.minimum.reduce([np.asarray(x) - xmin, xmax - np.asarray(x),
                               np.asarray(y) - ymin, ymax - np.asarray(y)])
        better = d > best_dist
        best_tile[better], best_dist[better] = tid, d[better]
    best_tile[best_dist <= 0] = ""          # outside every tile
    return pd.Series(best_tile, index=points.index), tiles


# ----------------------------------------------------------------------------
# One scene
# ----------------------------------------------------------------------------
def sample_asset(href, xs, ys, overview=False):
    kwargs = {"OVERVIEW_LEVEL": 0} if overview else {}
    with rasterio.open(href, **kwargs) as src:
        return np.fromiter((v[0] for v in src.sample(zip(xs, ys))),
                           dtype=np.int32, count=len(xs))


def process_item(item, pts, out_path, retries=3):
    """pts: DataFrame with point_id, tx, ty (coordinates in the tile CRS)."""
    for attempt in range(1, retries + 1):
        try:
            signed = planetary_computer.sign(item)
            with rasterio.Env(**GDAL_ENV):
                scl = sample_asset(signed.assets["SCL"].href, pts["tx"].values, pts["ty"].values)
                keep = np.isin(scl, C.SCL_STORE)
                if not keep.any():
                    pd.DataFrame(columns=COLUMNS).to_parquet(out_path)
                    return 0
                sub = pts[keep]
                xs, ys = sub["tx"].values, sub["ty"].values
                base = float(item.properties.get("s2:processing_baseline", "0") or 0)
                offset = 1000 if base >= 4.0 else 0
                data = {"point_id": sub["point_id"].values,
                        "date": item.datetime.strftime("%Y-%m-%d"),
                        "tile": item.properties["s2:mgrs_tile"],
                        "item_id": item.id,
                        "scl": scl[keep].astype(np.int8)}
                for b in BANDS:
                    dn = sample_asset(signed.assets[b].href, xs, ys,
                                      overview=b in C.BANDS_10M).astype(np.float32)
                    refl = (dn - offset) / 10000.0
                    refl[dn == 0] = np.nan
                    data[b] = refl
            df = pd.DataFrame(data)[COLUMNS]
            df.to_parquet(out_path)
            return len(df)
        except Exception as e:                       # network errors, expired tokens
            if attempt == retries:
                print(f"  FAILED {item.id}: {e}")
                return -1
            time.sleep(5 * attempt)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(years=None):
    years = years or C.YEARS
    pts = gpd.read_file(C.SAMPLE_FILE)
    ll = pts.to_crs("EPSG:4326").total_bounds
    bbox = [float(v) for v in ll]
    cat = open_catalog()

    # Tile assignment from one month of scenes (any year gives the same tiles)
    probe = search_items(cat, bbox, f"{years[-1]}-07-01", f"{years[-1]}-07-31")
    pts["tile"], tiles = assign_tiles(pts, probe)
    print("Points per tile:\n" + pts["tile"].value_counts().to_string())
    pts[["point_id", "tile"]].to_csv(C.OUT_DIR / "point_tile_assignment.csv", index=False)

    # Point coordinates in each tile CRS, sorted north to south (better block caching)
    pts_by_tile = {}
    for tid, (_, epsg) in tiles.items():
        sub = pts[pts["tile"] == tid].copy()
        if sub.empty:
            continue
        t = Transformer.from_crs(C.WORK_CRS, f"EPSG:{epsg}", always_xy=True)
        sub["tx"], sub["ty"] = t.transform(sub["x"].values, sub["y"].values)
        pts_by_tile[tid] = sub.sort_values("ty", ascending=False)[["point_id", "tx", "ty"]]

    for year in years:
        ydir = C.S2_DIR / str(year)
        ydir.mkdir(exist_ok=True)
        items = [it for it in search_items(cat, bbox, f"{year}-{C.SEASON_START}",
                                           f"{year}-{C.SEASON_END}")
                 if it.properties["s2:mgrs_tile"] in pts_by_tile]
        todo = [it for it in items if not (ydir / f"{it.id}.parquet").exists()]
        print(f"\n{year}: {len(items)} scenes, {len(todo)} to process")

        t0, done, failed = time.time(), 0, 0
        with ThreadPoolExecutor(max_workers=C.N_WORKERS) as ex:
            futs = {ex.submit(process_item, it, pts_by_tile[it.properties["s2:mgrs_tile"]],
                              ydir / f"{it.id}.parquet"): it for it in todo}
            for f in as_completed(futs):
                done += 1
                failed += f.result() < 0
                if done % 10 == 0 or done == len(todo):
                    rate = (time.time() - t0) / done
                    print(f"  {done}/{len(todo)} scenes, {failed} failed, "
                          f"~{rate * (len(todo) - done) / 60:.0f} min left")

        parts = [pd.read_parquet(p) for p in sorted(ydir.glob("*.parquet"))]
        parts = [p for p in parts if not p.empty]
        if parts:
            ts = pd.concat(parts, ignore_index=True)
            ts["date"] = pd.to_datetime(ts["date"])
            ts = ts.sort_values(["point_id", "date"])
            ts.to_parquet(C.OUT_DIR / f"s2_points_{year}.parquet", index=False)
            print(f"{year}: {len(ts):,} observations, "
                  f"{ts['point_id'].nunique():,} points -> s2_points_{year}.parquet")
        if failed:
            print(f"{year}: {failed} scenes failed - run the script again to retry them.")


if __name__ == "__main__":
    # All years in config.YEARS (2018-2025); 2023 is already done and is skipped.
    main()
