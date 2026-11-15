# -*- coding: utf-8 -*-
"""
01 - Cereal stratum and random sample of points
================================================
Draws random points inside Basilicata that fall in the cereal classes of
EUCROPMAP (optionally also in CORINE non-irrigated arable land), at least
EDGE_EROSION_PX pixels away from any non-cereal pixel and at least
MIN_SPACING_M metres from each other.

The full 10 m raster is never loaded: candidate points are drawn at random and
checked against the raster, so the script runs on an ordinary PC.

Output: output/sample_points.gpkg (EPSG:32633) with point_id, crop_class,
x, y (UTM 33N) and lon, lat (WGS84).
"""
import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.windows import Window
from pyproj import Transformer
import shapely

import config as C


def load_region():
    reg = gpd.read_file(C.REGION_FILE)
    reg = reg[reg[C.REGION_FIELD].astype(int) == C.REGION_CODE]
    if reg.empty:
        raise ValueError("Region not found: check REGION_FIELD / REGION_CODE in config.py")
    return reg.to_crs(C.WORK_CRS).dissolve().geometry.iloc[0]


class CropMap:
    """Reads EUCROPMAP values at points across one or more GeoTIFF tiles."""

    def __init__(self, files):
        if not files:
            raise FileNotFoundError("No EUCROPMAP file found: check EUCROPMAP_FILES in config.py")
        self.srcs = [rasterio.open(f) for f in files]
        self.crs = self.srcs[0].crs
        self.to_map = Transformer.from_crs(C.WORK_CRS, self.crs, always_xy=True)
        self.cereal = np.array(C.CEREAL_CLASSES)

    def _which(self, mx, my):
        """Index of the tile containing each point (-1 if none)."""
        idx = np.full(mx.shape, -1, dtype=int)
        for i, s in enumerate(self.srcs):
            b = s.bounds
            inside = (mx >= b.left) & (mx < b.right) & (my > b.bottom) & (my <= b.top)
            idx[(idx < 0) & inside] = i
        return idx

    def values(self, x, y):
        """Class value at each point (0 where outside every tile)."""
        mx, my = self.to_map.transform(x, y)
        mx, my = np.asarray(mx), np.asarray(my)
        out = np.zeros(mx.shape, dtype=np.int32)
        which = self._which(mx, my)
        for i, s in enumerate(self.srcs):
            sel = np.where(which == i)[0]
            if sel.size:
                vals = np.fromiter((v[0] for v in s.sample(zip(mx[sel], my[sel]))),
                                   dtype=np.int32, count=sel.size)
                out[sel] = vals
        return out, mx, my, which

    def is_core(self, mx, my, which, k):
        """True if all pixels within k px of the point are cereal."""
        ok = np.zeros(mx.shape, dtype=bool)
        size = 2 * k + 1
        for j in range(mx.size):
            if which[j] < 0:
                continue
            s = self.srcs[which[j]]
            row, col = s.index(mx[j], my[j])
            win = Window(col - k, row - k, size, size)
            arr = s.read(1, window=win, boundless=True, fill_value=0)
            ok[j] = np.isin(arr, self.cereal).all()
        return ok


_CLC = {}


def _load_clc_vector(region):
    """Arable CORINE polygons (vector: .gdb/.gpkg/.shp) clipped to the region.
    Returns None if the file does not cover the whole region."""
    clc = gpd.read_file(C.CLC_FILE, mask=gpd.GeoSeries([region], crs=C.WORK_CRS))
    clc = clc.to_crs(C.WORK_CRS)
    cover = clc.union_all().intersection(region).area / region.area
    if cover < 0.99:
        print(f"WARNING: CORINE file covers only {cover:.0%} of the region -> CORINE filter "
              f"NOT used (it would bias the sample). Use the full CLC 2018 raster or vector.")
        return None
    ar = clc[clc[C.CLC_CODE_FIELD].astype(str).isin(C.CLC_ARABLE_CODES)]
    print(f"CORINE vector loaded: {len(ar):,} arable polygons, coverage {cover:.0%}")
    return ar.union_all()


def clc_filter(x, y, region=None):
    """True where CORINE is arable, or everywhere if CORINE is missing/unusable."""
    f = C.CLC_FILE
    if f is None or not f.exists():
        return np.ones(np.size(x), dtype=bool)
    if f.suffix.lower() in (".gdb", ".gpkg", ".shp"):
        if "geom" not in _CLC:
            _CLC["geom"] = _load_clc_vector(region)
        if _CLC["geom"] is None:
            return np.ones(np.size(x), dtype=bool)
        return shapely.contains_xy(_CLC["geom"], x, y)
    with rasterio.open(f) as s:
        t = Transformer.from_crs(C.WORK_CRS, s.crs, always_xy=True)
        cx, cy = t.transform(x, y)
        vals = np.fromiter((v[0] for v in s.sample(zip(cx, cy))), dtype=np.int32, count=np.size(x))
    return np.isin(vals, C.CLC_ARABLE_GRIDCODES)


def thin_by_spacing(x, y, d):
    """Greedy thinning in the given (random) order: keep a point only if no kept
    point lies within d metres. Uses a grid of cell size d for neighbour lookup."""
    keep, grid = [], {}
    for i in range(x.size):
        gx, gy = int(x[i] // d), int(y[i] // d)
        clash = False
        for ix in (gx - 1, gx, gx + 1):
            for iy in (gy - 1, gy, gy + 1):
                for j in grid.get((ix, iy), ()):
                    if (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2 < d * d:
                        clash = True
                        break
                if clash:
                    break
            if clash:
                break
        if not clash:
            keep.append(i)
            grid.setdefault((gx, gy), []).append(i)
    return np.array(keep, dtype=int)


def main():
    rng = np.random.default_rng(C.RANDOM_SEED)
    region = load_region()
    xmin, ymin, xmax, ymax = region.bounds
    cmap = CropMap(C.EUCROPMAP_FILES)
    if C.CLC_FILE is None or not C.CLC_FILE.exists():
        print('Arable-land filter file not found: sampling on EUCROPMAP cereal classes only.')

    acc_x, acc_y, acc_cls = np.empty(0), np.empty(0), np.empty(0, dtype=np.int32)
    n_drawn = n_in_region = n_cereal = n_core = 0
    batch = 20000
    while True:
        x = rng.uniform(xmin, xmax, batch)
        y = rng.uniform(ymin, ymax, batch)
        inside = shapely.contains_xy(region, x, y)
        x, y = x[inside], y[inside]
        n_drawn += batch
        n_in_region += x.size

        cls, mx, my, which = cmap.values(x, y)
        cer = np.isin(cls, C.CEREAL_CLASSES) & clc_filter(x, y, region)
        x, y, cls, mx, my, which = x[cer], y[cer], cls[cer], mx[cer], my[cer], which[cer]
        n_cereal += x.size

        core = cmap.is_core(mx, my, which, C.EDGE_EROSION_PX)
        x, y, cls = x[core], y[core], cls[core]
        n_core += x.size

        acc_x = np.concatenate([acc_x, x])
        acc_y = np.concatenate([acc_y, y])
        acc_cls = np.concatenate([acc_cls, cls])
        kept = thin_by_spacing(acc_x, acc_y, C.MIN_SPACING_M)
        print(f"candidates in region {n_in_region:,} | cereal {n_cereal:,} | "
              f"core {n_core:,} | spaced {kept.size:,}")
        if kept.size >= C.N_POINTS:
            break
        if n_drawn > 400 * C.N_POINTS:
            print("WARNING: stopping early, cereal stratum smaller than expected.")
            break

    kept = kept[: C.N_POINTS]
    gdf = gpd.GeoDataFrame(
        {"point_id": np.arange(1, kept.size + 1),
         "crop_class": acc_cls[kept],
         "x": acc_x[kept], "y": acc_y[kept]},
        geometry=gpd.points_from_xy(acc_x[kept], acc_y[kept]), crs=C.WORK_CRS)
    ll = gdf.to_crs("EPSG:4326")
    gdf["lon"], gdf["lat"] = ll.geometry.x.values, ll.geometry.y.values
    gdf.to_file(C.SAMPLE_FILE, driver="GPKG")

    # Share of cereal land in the region, from the random draw (for the paper)
    share = n_cereal / max(n_in_region, 1)
    use_clc = C.CLC_FILE is not None and C.CLC_FILE.exists() and _CLC.get('geom', 1) is not None
    area_km2 = region.area / 1e6
    print(f"\nSaved {len(gdf):,} points to {C.SAMPLE_FILE}")
    print(f"Region area {area_km2:,.0f} km2; cereal share {share:.1%} "
          f"(~{share * area_km2:,.0f} km2 in EUCROPMAP{' x arable filter' if use_clc else ''})")
    print(pd.Series(gdf["crop_class"]).value_counts().rename("points per class").to_string())


if __name__ == "__main__":
    main()
