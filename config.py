# -*- coding: utf-8 -*-
"""
Configuration for the Basilicata stubble / residue-management project.
All scripts import this file. Edit the paths in the INPUT section before running.

Layout (flat, as in the RSASE repository):
    config.py
    01_cereal_stratum_and_sample.py
    02_s2_timeseries_points.py
    03_indices_and_phenology.py
    data/     <- input files downloaded by hand
    output/   <- everything the scripts write
"""
from pathlib import Path

# ----------------------------------------------------------------------------
# FOLDERS
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "output"
S2_DIR = OUT_DIR / "s2_points"          # one sub-folder per year, one parquet per scene
for _d in (DATA_DIR, OUT_DIR, S2_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# INPUT FILES (download by hand, then set the names here)
# ----------------------------------------------------------------------------
# Regional boundary: Basilicata extracted from ISTAT Limiti01012026 (Reg01012026_WGS84, EPSG:32632)
REGION_FILE = DATA_DIR / "Basilicata_ISTAT_2026.gpkg"
REGION_FIELD = "COD_REG"     # attribute holding the region code
REGION_CODE = 17             # Basilicata

# EUCROPMAP 2022 v1 (JRC, 10 m), EU27-HR stratum file (EU_CropMap_22_v1_stratum_EU27-HR.tif)
EUCROPMAP_FILES = sorted(DATA_DIR.glob("EU_CropMap_22*.tif"))
# Winter cereal classes of the 2022 legend (checked on the GEE catalogue page
# JRC/D5/EUCROPMAP/V1): 211 common wheat, 212 durum wheat, 213 barley, 214 rye,
# 215 oats, 218 triticale, 219 other cereals. Maize (216) and rice (217) excluded.
CEREAL_CLASSES = [211, 212, 213, 214, 215, 218, 219]

# Optional arable-land filter applied on top of EUCROPMAP.
# Used: ISPRA "Carta nazionale di copertura del suolo 2023" (10 m, whole Italy,
# integrates CLC 2018, CLC+ Backbone and the ISPRA soil-consumption map).
# Class 2211 = "Erbaceo periodico" (annual herbaceous = arable land).
# The file is read in place, at the sample points only (no need to copy it).
# A full CORINE 2018 raster (.tif, grid code 12) or vector (.gdb/.gpkg/.shp,
# field Code_18 = "211") also works; a vector that does not cover the whole
# region is ignored. Set CLC_FILE = None to sample on EUCROPMAP only.
CLC_FILE = DATA_DIR / "Copertura_del_suolo_2023.tif"   # ISPRA national land cover 2023 (10 m)
CLC_ARABLE_GRIDCODES = [2211]      # raster values accepted (ISPRA 2211; CLC raster: 12)
CLC_CODE_FIELD = "Code_18"         # vector only
CLC_ARABLE_CODES = ["211"]         # vector only

# ----------------------------------------------------------------------------
# SAMPLING (script 01)
# ----------------------------------------------------------------------------
WORK_CRS = "EPSG:32633"      # UTM 33N, same CRS as the Sentinel-2 tiles over Basilicata
N_POINTS = 8000              # extra points cost almost nothing in script 02 (see notes)
MIN_SPACING_M = 200          # minimum distance between points
EDGE_EROSION_PX = 3          # 3 px x 10 m = 30 m away from any non-cereal pixel
RANDOM_SEED = 2026
SAMPLE_FILE = OUT_DIR / "sample_points.gpkg"

# ----------------------------------------------------------------------------
# SENTINEL-2 (script 02)
# ----------------------------------------------------------------------------
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
S2_COLLECTION = "sentinel-2-l2a"
YEARS = list(range(2018, 2026))   # 2017 discarded (only S2A for most of the season)
# Season read for each year: phenology (Feb-Jun) + post-harvest window (Jun-Nov)
SEASON_START = "02-01"
SEASON_END = "11-15"
MAX_SCENE_CLOUD = 80         # eo:cloud_cover filter on whole scenes (%); pixels are masked later
# Bands read at the points. 10 m bands are read from the first overview (20 m),
# so that all bands share the 20 m grid of B11/B12 and less data is downloaded.
BANDS_10M = ["B02", "B04", "B08"]
BANDS_20M = ["B11", "B12"]
# SCL classes for which reflectance is read and stored.
# 2 = dark area pixels and 3 = cloud shadows are KEPT on purpose: fresh char on
# burned stubble is often labelled 2 or 3 by Sen2Cor. They are filtered in 03.
SCL_STORE = [2, 3, 4, 5, 6, 7]
N_WORKERS = 6                # parallel scenes; back to 4 if scenes start to fail

# ----------------------------------------------------------------------------
# PHENOLOGY AND WINDOW (script 03) - first values, to tune on the calibration subset
# ----------------------------------------------------------------------------
SCL_CLEAR = [4, 5]           # observations used for phenology
PEAK_START, PEAK_END = "02-01", "05-15"
NDVI_PEAK_MIN = 0.55         # minimum spring NDVI peak of a winter cereal
NDVI_SENESCENCE = 0.30       # NDVI below which the crop is considered senescent
SENESCENCE_BY = "07-15"      # senescence must occur before this date
HARVEST_LAG_DAYS = 10        # post-harvest window starts this many days after senescence
WINDOW_START_MIN = "06-10"   # and never before this date
WINDOW_END = "11-15"
MIN_OBS_PEAK = 3             # clear observations needed in the peak period
MIN_OBS_WINDOW = 5           # clear observations needed in the post-harvest window

# ----------------------------------------------------------------------------
# VIIRS ACTIVE FIRES (script 04)
# ----------------------------------------------------------------------------
# NASA FIRMS "country yearly" CSV files for Italy, one per sensor and year
# (VIIRS S-NPP, NOAA-20, NOAA-21), saved in data/firms/. Any file name works:
# the script reads every .csv in the folder.
FIRMS_DIR = DATA_DIR / "firms"
VIIRS_RADIUS_M = 500          # a detection within this distance counts for a point
VIIRS_MIN_CONF = ["n", "h"]   # VIIRS confidence classes kept (nominal, high); "l" = low

# ----------------------------------------------------------------------------
# TRAJECTORY CLASSIFICATION (script 05) - first values, tuned on the
# calibration subset and varied in the sensitivity analysis
# ----------------------------------------------------------------------------
N_BASELINE_OBS = 2            # first clear obs of the window = post-harvest stubble baseline
NDVI_REGROWTH = 0.30          # obs with NDVI above this (weeds, volunteers, new crop) are not used
# Burn: an ABRUPT drop of NBR that the surrounding landscape does not share.
#  - step:    NBR falls by >= STEP_BURN from the previous usable observation
#  - anomaly: the NBR change since the post-harvest baseline is lower than the
#             median change of all cereal points of the same tile and date by
#             >= ANOM_BURN (removes the autumn decline due to sun angle, rain, tillage)
#  - char:    NIR (B08) <= B08_CHAR (char is dark in the NIR)
#  - persistence: at the next observation the anomaly is still <= -ANOM_BURN / 2
# First values from the 2023 run, against VIIRS-confirmed burns (median step
# -0.33, median anomaly -0.27, median B08 0.12).
STEP_BURN = 0.15
ANOM_BURN = 0.15
B08_CHAR = 0.20
# Calibration (90 photo-interpreted point-years, Nov 2026): half of the burns of
# the first rule were tilled soil darkened after the first rains. Char also has
# a low SWIR1 reflectance and no cellulose feature (NDTI near zero): the event
# observation must also have B11 <= B11_CHAR and NDTI <= NDTI_CHAR, unless the
# fire is burning at the acquisition (B12 >= B12_FLAME).
B11_CHAR = 0.22
NDTI_CHAR = 0.04
B12_FLAME = 0.50
# Summer crops: a point-year with NDVI above SUMMER_NDVI_MAX in July-August is
# not a harvested winter cereal (irrigated second crop) and is excluded.
SUMMER_NDVI_MAX = 0.65
# Tillage: NDTI falls below NDTI_SOIL and by at least DNDTI_TILL from the baseline,
# BSI rises, and the change persists for TILL_PERSIST obs
NDTI_SOIL = 0.08
DNDTI_TILL = 0.06
TILL_PERSIST = 2
# Partial NDTI decrease (shallow harrowing / uncertain): DNDTI_MIN <= drop < DNDTI_TILL
DNDTI_MIN = 0.03
# Final classes: 'minimum' (partial NDTI decrease) is merged into 'tilled'
# ("soil disturbed"), because the two are the hardest to separate and to validate.
MERGE_TILLAGE = True
# Sensitivity analysis: each threshold is multiplied by these factors, one at a time
SENS_FACTORS = [0.75, 0.875, 1.0, 1.125, 1.25]

# ----------------------------------------------------------------------------
# VALIDATION SAMPLE AND CHIPS (script 06)
# ----------------------------------------------------------------------------
VAL_PER_CLASS = 100           # accuracy sample: point-years per map class (equal allocation)
CAL_PER_CLASS = 30            # calibration subset per class, drawn first, NOT used for accuracy
VAL_YEARS = None              # None = all years available in classes.parquet
VAL_SEED = 7
CHIP_HALF_M = 500             # chip = 1 km x 1 km around the point
CHIP_DATES = 6                # Sentinel-2 dates shown per point-year
VAL_DIR = OUT_DIR / "validation"

# ----------------------------------------------------------------------------
# ENVIRONMENTAL INDICATORS (scripts 07a, 07)
# All coefficients are literature values with a range; script 07 propagates the
# ranges with a Monte Carlo. Values marked [check] must be verified against the
# cited source before submission.
# ----------------------------------------------------------------------------
ERA5_FILE = DATA_DIR / "era5land_tp_daily_basilicata.nc"   # written by 07a
ERA5_AREA = [41.20, 15.30, 39.85, 16.90]                     # N, W, S, E
# Durum wheat grain yield (t/ha). Put the ISTAT table in data/istat_yield.csv with
# columns year;yield_t_ha (regional) to use real values; otherwise this range is used.
ISTAT_YIELD_FILE = DATA_DIR / "istat_yield.csv"
YIELD_RANGE = (2.0, 3.5)            # t/ha, Basilicata durum wheat [check ISTAT]
STRAW_GRAIN_RATIO = (0.9, 1.3)      # straw/grain, dry matter basis [check IPCC 2006 / 2019]
DRY_MATTER = (0.85, 0.90)           # dry matter fraction of residue [check]
RESIDUE_LEFT = (0.3, 0.7)           # share of straw NOT baled before burning (scenario)
COMBUSTION_FACTOR = (0.80, 0.90)    # IPCC 2006 Table 2.6, wheat residues 0.90 [check]
# Emission factors, g per kg of dry matter burned (low, high) [check IPCC 2006 Table 2.5; Andreae 2019]
EMISSION_FACTORS = {"CO2": (1400, 1550), "CO": (75, 95), "CH4": (2.5, 6.0),
                    "N2O": (0.05, 0.10), "PM2.5": (6.0, 9.0), "BC": (0.3, 0.6)}
C_CONTENT = (0.43, 0.47)            # C fraction of residue dry matter [check]
N_CONTENT = (0.004, 0.007)          # N fraction of residue dry matter [check]
N_MONTE_CARLO = 5000
# Soil exposure
NDVI_COVER = 0.30                   # bare period ends when NDVI exceeds this (regrowth or new crop)
EXPOSURE_END = "11-15"              # last observed date of the season (censoring)
HEAVY_RAIN_MM = 20                  # daily rainfall considered erosive
