# -*- coding: utf-8 -*-
"""
07a - Daily precipitation from ERA5-Land for Basilicata (Copernicus CDS)
=======================================================================
Requires a free Copernicus Climate Data Store account and the file
C:\\Users\\<you>\\.cdsapirc with your key (https://cds.climate.copernicus.eu ->
your profile -> API key). Install once in the 'stubble' environment:
    conda install -c conda-forge cdsapi xarray netcdf4

ERA5-Land total_precipitation is accumulated from 00 UTC: the value at 00 UTC
of day d+1 is the total of day d. Only the 00 UTC field is therefore downloaded
(small files), for June-December of each year.

Output: data/era5land_tp_daily_basilicata.nc  (variable tp_mm, dims time, lat, lon)
"""
import numpy as np
import cdsapi
import xarray as xr

import config as C

TMP = C.DATA_DIR / "era5land_tmp"


def download_year(c, year):
    out = TMP / f"tp_{year}.nc"
    if out.exists():
        return out
    c.retrieve("reanalysis-era5-land", {
        "variable": "total_precipitation",
        "year": str(year),
        "month": [f"{m:02d}" for m in range(6, 13)],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": "00:00",
        "area": C.ERA5_AREA,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }, str(out))
    return out


def main():
    TMP.mkdir(exist_ok=True)
    c = cdsapi.Client()
    files = []
    for y in C.YEARS:
        print(f"ERA5-Land {y} ...")
        files.append(download_year(c, y))
    # open one by one (no dask needed); drop the per-time 'expver' and 'number' coords
    parts = []
    for f in files:
        d = xr.open_dataset(f)
        d = d.drop_vars([v for v in ("expver", "number") if v in d.variables])
        parts.append(d["tp"].load())
        d.close()
    tname = "valid_time" if "valid_time" in parts[0].dims else "time"
    tp = xr.concat(parts, dim=tname).sortby(tname) * 1000.0     # m -> mm
    # value at 00 UTC of day d+1 = total of day d
    tp = tp.assign_coords({tname: tp[tname] - np.timedelta64(1, "D")})
    tp = tp.rename({tname: "time"}).rename("tp_mm")
    tp = tp.rename({"latitude": "lat", "longitude": "lon"})
    tp.to_netcdf(C.ERA5_FILE)
    print(f"Saved {C.ERA5_FILE}")


if __name__ == "__main__":
    main()
