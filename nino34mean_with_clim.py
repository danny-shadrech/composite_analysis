import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# ==============================================================================
# ── Helpers ───────────────────────────────────────────────────────────────────
# ==============================================================================
def open_ds(path):
    return xr.open_mfdataset(path, combine='by_coords') if any(c in path for c in '*?[') else xr.open_dataset(path)

def get_coords(ds):
    t  = next((n for n in ['time', 'valid_time', 'times', 'date'] if n in ds.coords), None)
    lo = next((n for n in ['longitude', 'lon', 'longitudes'] if n in ds.coords), None)
    la = next((n for n in ['latitude', 'lat', 'latitudes'] if n in ds.coords), None)
    if not all([t, lo, la]): raise KeyError("Cannot resolve coordinates.")
    return t, lo, la

def event_slice(series, year):
    return series.loc[f"{year}-06": f"{year+1}-09"]

def line_plot(ax, x, vals, color, title, ylabel, fill_zero=False, thresholds=False):
    ax.plot(x, vals, color=color, lw=2, marker='o', ms=4.5, mfc='white', zorder=3)
    ax.fill_between(x, vals, 0 if fill_zero else min(vals), alpha=0.18, color=color)
    if thresholds:
        for val, col, lbl in [(0.5, '#d73027', '+0.5 °C'), (-0.5, '#4575b4', '−0.5 °C')]:
            ax.axhline(val, color=col, lw=1, ls='--', label=lbl)
            ax.fill_between(x, vals, 0, where=(vals >= 0.5) if val > 0 else (vals <= -0.5),
                            color=col, alpha=0.25, zorder=2)
        ax.axhline(0, color='black', lw=0.8)
        ax.legend(fontsize=8, loc='upper right', framealpha=0.6)
    ax.set(xticks=x, xlim=(-0.5, 15.5), ylabel=ylabel, title=title)
    ax.set_xticklabels(['Jun','Jul','Aug','Sep','Oct','Nov','Dec','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep'], fontsize=8)
    ax.grid(axis='y', ls=':', lw=0.5, alpha=0.7)


def add_29_isotherm(ax, lons, lats, data, proj=ccrs.PlateCarree()):
    """Draw the 29 °C isotherm contour with inline label."""
    try:
        cs = ax.contour(lons, lats, data, levels=[29],
                        colors='limegreen', linewidths=0.5, linestyles='-',
                        transform=proj, zorder=5)
        ax.clabel(cs, fmt='29°C', fontsize=7, inline=True, inline_spacing=4)
    except Exception:
        pass  # silently skip if contour cannot be drawn (e.g. 29°C not in range)


# ==============================================================================
# ── Core computations ─────────────────────────────────────────────────────────
# ==============================================================================
def compute_enso_events(nc_path):
    ds = open_ds(nc_path)
    t, lo, la = get_coords(ds)

    lon_min, lon_max = (-170, -120) if ds[lo].max() <= 180 else (190, 240)
    lat_sl = slice(5, -5) if ds[la].values[0] > ds[la].values[-1] else slice(-5, 5)

    box        = ds['sst'].sel({la: lat_sl, lo: slice(lon_min, lon_max)}).mean(dim=[la, lo])
    clim       = box.groupby(f'{t}.month').mean(t)
    oni_series = (box.groupby(f'{t}.month') - clim).rolling({t: 3}, center=True).mean(t).to_series()
    nino34_raw = box.to_series()

    el_nino, la_nina = [], []
    for yr in np.unique(ds[t].dt.year.values)[:-1]:
        try:
            v = oni_series.loc[f"{yr}-12"]
            v = v.values[0] if hasattr(v, 'values') else v
            (el_nino if v >= 0.5 else la_nina if v <= -0.5 else []).append(yr)
        except KeyError:
            continue

    return el_nino, la_nina, oni_series, nino34_raw


# ==============================================================================
# ── Per-event plot (seasonal SST + ONI panels) ────────────────────────────────
# ==============================================================================
def plot_and_save_enso(nc_path, year, event_type, oni_series, nino34_raw, output_dir="ENSO_plots"):
    os.makedirs(output_dir, exist_ok=True)
    ds = open_ds(nc_path)
    t, lo, la = get_coords(ds)

    lat_sl = slice(40, -40) if ds[la].values[0] > ds[la].values[-1] else slice(-40, 40)
    sst    = ds['sst'].sel({t: slice(f'{year}-06', f'{year+1}-09'), lo: slice(120, 290), la: lat_sl})
    sst    = sst - 273.15 if sst.mean() > 200 else sst
    seas   = sst.groupby(f'{t}.season').mean(t)

    # Pre-extract lon/lat arrays for contour drawing
    lons_2d = sst[lo].values
    lats_2d = sst[la].values

    oni_vals = event_slice(oni_series, year)
    oni_vals = oni_vals.values if len(oni_vals) == 16 else np.full(16, np.nan)
    sst_vals = event_slice(nino34_raw, year)
    sst_vals = sst_vals.values.copy() if len(sst_vals) == 16 else np.full(16, np.nan)
    if np.nanmean(sst_vals) > 200: sst_vals -= 273.15

    fig = plt.figure(figsize=(15, 13))
    gs  = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 0.85], hspace=0.55, wspace=0.25)
    x   = np.arange(16)

    season_titles = {'JJA': f'JJA ({year})', 'SON': f'SON ({year})',
                     'DJF': f'DJF ({year}/{year+1})', 'MAM': f'MAM ({year+1})'}
    im = None

    for i, ((row, col), season) in enumerate(zip([(0,0),(0,1),(1,0),(1,1)], ['JJA','SON','DJF','MAM'])):
        ax = fig.add_subplot(gs[row, col], projection=ccrs.PlateCarree(180))
        try:
            data_season = seas.sel(season=season)
            im = data_season.plot(ax=ax, transform=ccrs.PlateCarree(),
                                  cmap='RdYlBu_r', vmin=5, vmax=30, add_colorbar=False)

            # ── 29 °C isotherm ───────────────────────────────────────────────
            add_29_isotherm(ax, lons_2d, lats_2d, data_season.values)
            # ─────────────────────────────────────────────────────────────────

            ax.coastlines(lw=0.6)
            ax.set_extent([120, 290, -40, 40], crs=ccrs.PlateCarree())
            ax.set_title(season_titles[season], fontsize=10, weight='bold')

            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                              linewidth=0.5, color='gray', alpha=0.5, linestyle=':')
            gl.top_labels    = False
            gl.right_labels  = False
            gl.bottom_labels = True if row == 1 else False
            gl.left_labels   = True if col == 0 else False
            gl.xformatter    = LONGITUDE_FORMATTER
            gl.yformatter    = LATITUDE_FORMATTER
            gl.xlabel_style  = {'size': 8, 'color': 'dimgray'}
            gl.ylabel_style  = {'size': 8, 'color': 'dimgray'}
            gl.xlocator      = plt.FixedLocator([120, 150, 180, -150, -120, -90])
            gl.ylocator      = plt.FixedLocator([-30, -15, 0, 15, 30])

        except KeyError:
            ax.set_visible(False)

    if im is not None:
        fig.colorbar(im, cax=fig.add_axes([0.15, 0.32, 0.70, 0.015]),
                     orientation='horizontal', label='SST (°C)', shrink=0.8)

    line_plot(fig.add_subplot(gs[2, 0]), x, oni_vals, '#333333',
              f'Niño 3.4 Index  {year}–{year+1}', 'SSTA (°C)', fill_zero=True, thresholds=True)
    line_plot(fig.add_subplot(gs[2, 1]), x, sst_vals, '#e05c00',
              f'Monthly Mean SST  {year}–{year+1}', 'SST (°C)')

    colour = '#c0392b' if 'Nino' in event_type else '#2471a3'
    fig.suptitle(f'Pacific SST Seasonal Evolution & Niño 3.4: {year}–{year+1}  ({event_type})',
                 fontsize=13, weight='bold', color=colour, y=0.99)

    save_path = os.path.join(output_dir, f"{event_type.replace(' ','_')}_{year}_{year+1}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.3)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ==============================================================================
# ── NEW: Climatology figure (12-panel + 29 °C isotherm) ──────────────────────
# ==============================================================================
def plot_climatology(nc_path, output_dir="ENSO_plots"):
    os.makedirs(output_dir, exist_ok=True)
    ds = open_ds(nc_path)
    t, lo, la = get_coords(ds)

    lat_sl   = slice(40, -40) if ds[la].values[0] > ds[la].values[-1] else slice(-40, 40)
    sst_full = ds['sst'].sel({lo: slice(120, 290), la: lat_sl})
    sst_full = sst_full - 273.15 if sst_full.mean() > 200 else sst_full

    # 1981–2010 baseline climatology, one map per calendar month
    climo = (sst_full.sel({t: slice('1981', '2010')})
                     .groupby(f'{t}.month').mean(t))

    lons_2d = climo[lo].values
    lats_2d = climo[la].values

    MONTH_LABELS = ['Jan','Feb','Mar','Apr','May','Jun',
                    'Jul','Aug','Sep','Oct','Nov','Dec']

    #vmin = float(climo.min())
    #vmax = float(climo.max())

    fig, axes = plt.subplots(
        4, 3, figsize=(18, 13),
        subplot_kw={'projection': ccrs.PlateCarree(180)},
        gridspec_kw={'hspace': 0.35, 'wspace': 0.08}
    )
    fig.patch.set_facecolor('#f7f7f7')

    im = None
    for m_idx, ax in enumerate(axes.flatten()):
        data = climo.isel(month=m_idx)

        im = ax.pcolormesh(
            lons_2d, lats_2d, data.values,
            cmap='RdYlBu_r', vmin=5, vmax=30,
            transform=ccrs.PlateCarree(), shading='auto', zorder=1
        )

        # ── 29 °C isotherm ───────────────────────────────────────────────────
        add_29_isotherm(ax, lons_2d, lats_2d, data.values)
        # ─────────────────────────────────────────────────────────────────────

        ax.add_feature(cfeature.LAND,      facecolor='#cccccc', zorder=2)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6,        zorder=3)
        ax.set_extent([120, 290, -40, 40], crs=ccrs.PlateCarree())
        ax.set_title(MONTH_LABELS[m_idx], fontsize=10, weight='bold')

        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=0.4, color='gray', alpha=0.4, linestyle=':')
        gl.top_labels    = False
        gl.right_labels  = False
        gl.bottom_labels = True if m_idx >= 9 else False   # bottom row only
        gl.left_labels   = True if m_idx % 3 == 0 else False  # left column only
        gl.xformatter    = LONGITUDE_FORMATTER
        gl.yformatter    = LATITUDE_FORMATTER
        gl.xlabel_style  = {'size': 7, 'color': 'dimgray'}
        gl.ylabel_style  = {'size': 7, 'color': 'dimgray'}
        gl.xlocator      = plt.FixedLocator([120, 150, 180, -150, -120, -90])
        gl.ylocator      = plt.FixedLocator([-30, -15, 0, 15, 30])

    # Shared colourbar
    cbar = fig.colorbar(im, ax=axes, orientation='horizontal',
                        fraction=0.025, pad=0.05, aspect=55, shrink=0.75)
    cbar.set_label('SST (°C)', fontsize=11)

    fig.suptitle('Pacific SST Climatology  (1981–2010)  |  29 °C Isotherm shown',
                 fontsize=13, weight='bold', y=1.01)

    save_path = os.path.join(output_dir, 'climatology_monthly_29C_isotherm.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.3,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ==============================================================================
# ── Run ───────────────────────────────────────────────────────────────────────
# ==============================================================================
file_path      = '/Users/dsha0113/Documents/PhD_Work/Data_Analysis/Netcdf_data/MonthlymeanSST.nc'
save_directory = '/Users/dsha0113/Documents/PhD_Work/Data_Analysis/Output_Plots'

el_nino_list, la_nina_list, oni_series, nino34_raw = compute_enso_events(file_path)
print(f"El Niño years: {el_nino_list}\nLa Niña years: {la_nina_list}")

# Per-event seasonal plots (all with 29 °C isotherm)
for year, etype in [(y, "El Nino") for y in el_nino_list] + [(y, "La Nina") for y in la_nina_list]:
    plot_and_save_enso(file_path, year, etype, oni_series, nino34_raw, save_directory)

# Climatology figure (12-panel, with 29 °C isotherm)
plot_climatology(file_path, save_directory)

print("\n✓  Done.")