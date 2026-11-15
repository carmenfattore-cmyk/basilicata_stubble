# -*- coding: utf-8 -*-
"""08 - Figures and tables of the paper (Fig. 1, 3, 4, 5; Tables 3-4 as CSV)."""
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C

OUT = C.BASE_DIR / "manuscript" / "figs"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
COL = {"retained": "#4a8c3f", "tilled": "#a6662b", "burned": "#2b2b2b"}
VAL = C.OUT_DIR / "validation"

reg = gpd.read_file(C.REGION_FILE).to_crs(32633)
pts = gpd.read_file(C.SAMPLE_FILE).to_crs(32633)
cl = pd.read_parquet(C.OUT_DIR / "classes.parquet")
cl = cl[cl["cls"].isin(COL)]

# ---- Fig. 1: study area
fig, ax = plt.subplots(figsize=(5.5, 5.5))
reg.boundary.plot(ax=ax, color="k", lw=0.6)
pts.plot(ax=ax, markersize=0.6, color="#c89b2c")
ax.set_axis_off()
ax.set_title("Basilicata: 8,000 sample points in the cereal stratum", fontsize=9)
x0, y0 = reg.total_bounds[0], reg.total_bounds[1]
ax.plot([x0 + 5e3, x0 + 25e3], [y0 + 5e3] * 2, "k-", lw=2)
ax.text(x0 + 15e3, y0 + 7e3, "20 km", ha="center", fontsize=8)
fig.savefig(OUT / "Figure_1.png", dpi=300, bbox_inches="tight"); plt.close(fig)

# ---- Fig. 3: error-adjusted shares per year
a = pd.read_csv(VAL / "area_estimates.csv")
pool = a[a["year"] == "pooled"].set_index("cls")
y = a[a["year"] != "pooled"].copy()
y["year"] = y["year"].astype(int)
fig, ax = plt.subplots(figsize=(6.5, 3.4))
for k in ["retained", "tilled", "burned"]:
    s = y[y["cls"] == k].sort_values("year")
    rel = pool.at[k, "adj_share_ci95"] / pool.at[k, "adj_share"]
    ax.errorbar(s["year"], s["adj_share"] * 100, yerr=s["adj_share"] * rel * 100,
                color=COL[k], marker="o", ms=4, capsize=2, lw=1.2, label=k.capitalize())
ax.axvline(2022.5, color="grey", ls="--", lw=0.8)
ax.text(2022.6, 62, "CAP 2023-2027", fontsize=7, color="grey")
ax.set_ylabel("Share of harvested cereal area (%)")
ax.set_ylim(0, 75); ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))
ax.spines[["top", "right"]].set_visible(False)
fig.savefig(OUT / "Figure_3.png", dpi=300, bbox_inches="tight"); plt.close(fig)

# ---- Fig. 4: frequency of burning and retention on a 5 km grid (map classes)
g = pts[["point_id", "geometry"]].merge(cl, on="point_id")
g["gx"] = (g.geometry.x // 5000).astype(int); g["gy"] = (g.geometry.y // 5000).astype(int)
fr = g.groupby(["gx", "gy"]).agg(n=("cls", "size"),
                                 burned=("cls", lambda s: (s == "burned").mean()),
                                 retained=("cls", lambda s: (s == "retained").mean())).reset_index()
fr = fr[fr["n"] >= 20]
fig, axs = plt.subplots(1, 2, figsize=(8, 4.2))
for ax, k, cmap in zip(axs, ["burned", "retained"], ["Reds", "Greens"]):
    reg.boundary.plot(ax=ax, color="k", lw=0.5)
    sc = ax.scatter(fr["gx"] * 5000 + 2500, fr["gy"] * 5000 + 2500, c=fr[k] * 100, cmap=cmap,
                    marker="s", s=22, vmin=0, vmax=np.nanpercentile(fr[k] * 100, 98))
    plt.colorbar(sc, ax=ax, shrink=0.6, label=f"{k.capitalize()} point-years (%)")
    ax.set_axis_off()
fig.savefig(OUT / "Figure_4.png", dpi=300, bbox_inches="tight"); plt.close(fig)

# ---- Fig. 5: emissions per year + rainfall on bare soil
b = pd.read_csv(C.OUT_DIR / "env_burning_mc.csv")
b = b[b["year"] != "total"].copy(); b["year"] = b["year"].astype(int)
ex = pd.read_parquet(C.OUT_DIR / "env_exposure_points.parquet")
ex = ex[ex["cls"].isin(["tilled", "burned"])]
fig, axs = plt.subplots(1, 2, figsize=(8, 3.3))
ax = axs[0]
ax.bar(b["year"], b["CO2_t_median"] / 1000, color="#555555",
       yerr=[(b["CO2_t_median"] - b["CO2_t_p05"]) / 1000, (b["CO2_t_p95"] - b["CO2_t_median"]) / 1000],
       capsize=2)
ax.set_ylabel("CO$_2$ from stubble burning (kt)"); ax.set_title("a", loc="left", fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
ax = axs[1]
data = [ex.loc[ex["cls"] == k, "share_rain_on_bare"].dropna() * 100 for k in ["tilled", "burned"]]
bp = ax.boxplot(data, tick_labels=["Tilled", "Burned"], patch_artist=True, showfliers=False, widths=0.5)
for p, k in zip(bp["boxes"], ["tilled", "burned"]):
    p.set_facecolor(COL[k]); p.set_alpha(0.6)
ax.set_ylabel("Jul-15 Nov rainfall on bare soil (%)"); ax.set_title("b", loc="left", fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / "Figure_5.png", dpi=300, bbox_inches="tight"); plt.close(fig)
print("figures written to", OUT)
