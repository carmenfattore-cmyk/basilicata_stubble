# Post-harvest residue management in Basilicata (2018–2025)

Code and data for:

Fattore C., Fiorentino C., D'Antonio P. *Post-harvest residue management in Mediterranean durum wheat systems: satellite mapping of tillage, stubble burning and soil exposure.* Submitted to Soil & Tillage Research.

Training-free classification of Sentinel-2 L2A time series at 8,000 points of the cereal stratum of Basilicata (southern Italy) into residue **retained**, **tilled** or **burned**, with design-based accuracy and area estimation (Olofsson et al. 2014; Stehman 2014) and environmental indicators (emissions, C and N losses, bare-soil exposure to rainfall).

## Workflow (Python 3.11; paths in `config.py`)

| Script | Step |
|---|---|
| `01_cereal_stratum_and_sample.py` | cereal stratum (EUCROPMAP 2022 × ISPRA land cover 2023) and 8,000 random points |
| `02_s2_timeseries_points.py` | Sentinel-2 L2A reflectance at the points (Microsoft Planetary Computer STAC) |
| `03_indices_and_phenology.py` | NDVI, NDTI, NBR; winter-cereal phenology; post-harvest window |
| `04_viirs_points.py` | VIIRS 375 m active fires (NASA FIRMS) near the points |
| `05_classify_trajectories.py` | rule-based classification and sensitivity analysis |
| `06_validation_sample.py` | stratified validation sample and image panels |
| `06b_accuracy_area.py` | accuracy and error-adjusted areas |
| `07a_download_era5land.py`, `07_environmental_indicators.py` | ERA5-Land rainfall; Monte Carlo emissions; bare-soil exposure |
| `08_figures.py` | figures of the paper |

Main packages: numpy, pandas, geopandas, rasterio, pystac-client, planetary-computer, xarray, cdsapi, matplotlib.

## Data in this repository

- `output/sample_points.gpkg` – sample points (EPSG:4326)
- `output/point_years.parquet` – phenology and cereal flag per point-year
- `output/classes.parquet` – class, event date and diagnostics per classified point-year
- `output/class_shares.csv`, `output/sensitivity.csv` – map shares and threshold sensitivity
- `output/validation/` – validation sample, reference labels, error matrix, accuracies, error-adjusted areas. In `interpretation_sheet.csv`, `interpreter` = AUTO marks labels from the automated pre-labelling of the image panels, CF marks labels verified visually by the first author (Section 2.6 of the paper); `calibration_labels.csv` holds the labels of the calibration subset used to set the thresholds.
- `output/env_*` – emissions (Monte Carlo) and bare-soil exposure
- `data/istat_yield.csv` – ISTAT durum wheat area, yield and production

Raw Sentinel-2 time series (≈ several GB) are not included; they are reproduced by script 02. Input data are open: Copernicus Sentinel-2, EUCROPMAP 2022 (JRC), ISPRA land cover 2023, NASA FIRMS VIIRS, ERA5-Land (Copernicus CDS), ISTAT.

Contains modified Copernicus Sentinel data (2018–2025).

## Licence

Code: MIT. Data: CC BY 4.0.
