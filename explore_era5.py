"""
Quick exploration and plotting of downloaded ERA5 NetCDF data.
"""
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

DATA_PATH = Path(__file__).parent / "test_data.nc"
PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# ── 1. Load & inspect ────────────────────────────────────────────────────────
print("Loading dataset...")
ds = xr.open_dataset(DATA_PATH)

print("\n=== Dataset overview ===")
print(ds)
print("\n=== Variables ===")
for var in ds.data_vars:
    da = ds[var]
    print(f"  {var}: shape={da.shape}, min={float(da.min()):.2f}, max={float(da.max()):.2f}, units={da.attrs.get('units', 'N/A')}")

print(f"\n=== Coordinates ===")
print(f"  time : {ds.time.values[0]} → {ds.time.values[-1]}  ({ds.time.size} steps)")
print(f"  lat  : {float(ds.latitude.min()):.2f} → {float(ds.latitude.max()):.2f}  ({ds.latitude.size} points)")
print(f"  lon  : {float(ds.longitude.min()):.2f} → {float(ds.longitude.max()):.2f}  ({ds.longitude.size} points)")

# ── 2. Helper for a quick map ─────────────────────────────────────────────────
def plot_map(da, title, filename, cmap="RdBu_r", units=""):
    fig, ax = plt.subplots(
        figsize=(14, 7),
        subplot_kw={"projection": ccrs.Robinson()},
    )
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":")

    im = ax.pcolormesh(
        da.longitude, da.latitude, da.values,
        transform=ccrs.PlateCarree(),
        cmap=cmap, shading="auto",
    )
    plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.04, shrink=0.8, label=units)
    ax.set_title(title, fontsize=13)
    fig.savefig(PLOTS_DIR / filename, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: plots/{filename}")


# ── 3. Snapshot plots (first timestep) ───────────────────────────────────────
t0 = ds.time.values[0]
t0_str = str(t0)[:16]

print(f"\n=== Plotting snapshots at {t0_str} ===")

plot_map(
    ds["2m_temperature"].isel(time=0) - 273.15,
    f"2m Temperature [°C] — {t0_str}",
    "t2m_snapshot.png", cmap="RdBu_r", units="°C",
)

plot_map(
    ds["mean_sea_level_pressure"].isel(time=0) / 100,
    f"Mean Sea Level Pressure [hPa] — {t0_str}",
    "mslp_snapshot.png", cmap="viridis", units="hPa",
)

# Wind speed from components
u = ds["10m_u_component_of_wind"].isel(time=0)
v = ds["10m_v_component_of_wind"].isel(time=0)
wspd = np.sqrt(u**2 + v**2)
wspd.attrs["long_name"] = "10m Wind Speed"
plot_map(wspd, f"10m Wind Speed [m/s] — {t0_str}", "wind_speed_snapshot.png", cmap="plasma", units="m/s")

# ── 4. Time series: global-mean 2m temperature ───────────────────────────────
print("\n=== Plotting global-mean time series ===")
t2m_global_mean = (ds["2m_temperature"] - 273.15).mean(dim=["latitude", "longitude"])

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(ds.time.values, t2m_global_mean.values, marker="o", markersize=4)
ax.set_xlabel("Time")
ax.set_ylabel("Temperature [°C]")
ax.set_title("Global-mean 2m Temperature over time")
ax.grid(True, alpha=0.3)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(PLOTS_DIR / "t2m_global_mean_timeseries.png", dpi=120)
plt.close(fig)
print("  Saved: plots/t2m_global_mean_timeseries.png")

print("\nAll done. Plots saved to ./plots/")
