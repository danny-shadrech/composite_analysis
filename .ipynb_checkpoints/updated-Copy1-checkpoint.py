import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
 
def compute_enso_events(nc_path):
    if any(ch in nc_path for ch in ['*', '?', '[']):
        ds = xr.open_mfdataset(nc_path, combine='by_coords')
    else:
        ds = xr.open_dataset(nc_path)

    time_coord = next((n for n in ['time', 'valid_time', 'times', 'date'] if n in ds.coords), None)
    lon_coord  = next((n for n in ['longitude', 'lon', 'longitudes'] if n in ds.coords), None)
    lat_coord  = next((n for n in ['latitude', 'lat', 'latitudes'] if n in ds.coords), None)

    if not all([time_coord, lon_coord, lat_coord]):
        raise KeyError("Could not dynamically resolve spatial or temporal coordinates.")

    lon_min, lon_max = 190, 240
    if ds[lon_coord].max() <= 180:
        lon_min, lon_max = -170, -120

    nino34_box  = ds['sst'].sel({
        lat_coord: slice(5, -5) if ds[lat_coord].values[0] > ds[lat_coord].values[-1] else slice(-5, 5),
        lon_coord: slice(lon_min, lon_max)
    })
    nino34_mean = nino34_box.mean(dim=[lat_coord, lon_coord])
    climatology = nino34_mean.groupby(f'{time_coord}.month').mean(time_coord)
    anomalies   = nino34_mean.groupby(f'{time_coord}.month') - climatology
    oni_index   = anomalies.rolling({time_coord: 3}, center=True).mean(time_coord)
    oni_series  = oni_index.to_series()

    el_nino_years, la_nina_years = [], []
    unique_years = np.unique(ds[time_coord].dt.year.values)[:-1]

    for yr in unique_years:
        dec_time = f"{yr}-12"
        try:
            dec_val = oni_series.loc[dec_time].values[0] if hasattr(oni_series.loc[dec_time], 'values') else oni_series.loc[dec_time]
            if dec_val >= 0.5:
                el_nino_years.append(yr)
            elif dec_val <= -0.5:
                la_nina_years.append(yr)
        except KeyError:
            continue

    return el_nino_years, la_nina_years, oni_series


def build_composite_oni(oni_series, years, n_months=16):
    """
    Extracts ONI lifecycle Jun(yr0) → Sep(yr+1) = 16 months.
    Returns mean and ±1 std across all events.
    """
    monthly_traces = []
    for yr in years:
        try:
            trace = oni_series.loc[f'{yr}-06': f'{yr+1}-09']
            if len(trace) == n_months:
                monthly_traces.append(trace.values)
            else:
                print(f"  Skipping ONI trace for {yr}: expected {n_months} months, got {len(trace)}")
        except KeyError:
            continue

    if not monthly_traces:
        raise ValueError("No valid ONI lifecycle traces found.")

    traces_arr = np.array(monthly_traces)
    return np.nanmean(traces_arr, axis=0), np.nanstd(traces_arr, axis=0)


def build_composite(nc_path, years):
    """
    Builds seasonal composite for Year 0 seasons (JJA, SON, DJF, MAM)
    using Jun(yr) → May(yr+1) window.
    """
    if any(ch in nc_path for ch in ['*', '?', '[']):
        ds = xr.open_mfdataset(nc_path, combine='by_coords')
    else:
        ds = xr.open_dataset(nc_path)

    time_coord = next((n for n in ['time', 'valid_time', 'times', 'date'] if n in ds.coords), None)
    lon_coord  = next((n for n in ['longitude', 'lon', 'longitudes'] if n in ds.coords), None)
    lat_coord  = next((n for n in ['latitude', 'lat', 'latitudes'] if n in ds.coords), None)

    lat_slice = slice(40, -40) if ds[lat_coord].values[0] > ds[lat_coord].values[-1] else slice(-40, 40)
    seasonal_composites = []

    for yr in years:
        selection = {
            time_coord: slice(f'{yr}-06', f'{yr+1}-05'),
            lon_coord:  slice(120, 290),
            lat_coord:  lat_slice,
        }
        sst_yr = ds['sst'].sel(selection)
        if sst_yr.sizes[time_coord] == 0:
            print(f"  Skipping {yr}: no data in window.")
            continue
        if float(sst_yr.mean()) > 200:
            sst_yr = sst_yr - 273.15
        seasonal = sst_yr.groupby(f'{time_coord}.season').mean(time_coord)
        seasonal_composites.append(seasonal)

    if not seasonal_composites:
        raise ValueError("No valid event years found.")

    return xr.concat(seasonal_composites, dim='event').mean(dim='event')


def build_composite_yr1(nc_path, years):
    """
    Builds seasonal composite for Year+1 seasons (JJA, SON only)
    using Jun(yr+1) → Sep(yr+1) window.
    """
    if any(ch in nc_path for ch in ['*', '?', '[']):
        ds = xr.open_mfdataset(nc_path, combine='by_coords')
    else:
        ds = xr.open_dataset(nc_path)

    time_coord = next((n for n in ['time', 'valid_time', 'times', 'date'] if n in ds.coords), None)
    lon_coord  = next((n for n in ['longitude', 'lon', 'longitudes'] if n in ds.coords), None)
    lat_coord  = next((n for n in ['latitude', 'lat', 'latitudes'] if n in ds.coords), None)

    lat_slice = slice(40, -40) if ds[lat_coord].values[0] > ds[lat_coord].values[-1] else slice(-40, 40)
    seasonal_composites = []

    for yr in years:
        selection = {
            time_coord: slice(f'{yr+1}-06', f'{yr+1}-09'),   # Jun–Sep of yr+1 only
            lon_coord:  slice(120, 290),
            lat_coord:  lat_slice,
        }
        sst_yr = ds['sst'].sel(selection)
        if sst_yr.sizes[time_coord] == 0:
            print(f"  Skipping yr+1 {yr+1}: no data in window.")
            continue
        if float(sst_yr.mean()) > 200:
            sst_yr = sst_yr - 273.15
        seasonal = sst_yr.groupby(f'{time_coord}.season').mean(time_coord)
        seasonal_composites.append(seasonal)

    if not seasonal_composites:
        raise ValueError("No valid Year+1 composites found.")

    return xr.concat(seasonal_composites, dim='event').mean(dim='event')


def plot_combined_composite(el_nino_composite,    la_nina_composite,
                            el_nino_composite_y1, la_nina_composite_y1,
                            el_nino_years,        la_nina_years,
                            oni_series,           output_dir="ENSO_plots"):
    """
    Layout (GridSpec  5 rows × 4 cols):
      Row 0  : El Niño  Year 0  maps  — JJA(0)  SON(0)  DJF   MAM
      Row 1  : La Niña  Year 0  maps  — JJA(0)  SON(0)  DJF   MAM
      Row 2  : El Niño  Year+1  maps  — JJA(+1) SON(+1) [blank] [blank]
      Row 3  : La Niña  Year+1  maps  — JJA(+1) SON(+1) [blank] [blank]
      Row 4  : Composite Niño 3.4 index lifecycle (16 months)
    """
    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(26, 20))
    gs  = gridspec.GridSpec(
        5, 4,
        figure=fig,
        height_ratios=[1, 1, 1, 1, 0.85],
        hspace=0.28,
        wspace=0.06
    )

    # ── Build all map axes ─────────────────────────────────────────────────────
    # Rows 0-1: Year 0 (4 seasons each)
    map_axes_y0 = []
    for row in range(2):
        row_axes = [fig.add_subplot(gs[row, col], projection=ccrs.PlateCarree(180)) for col in range(4)]
        map_axes_y0.append(row_axes)

    # Rows 2-3: Year+1 (2 seasons — JJA, SON — cols 0 and 1 only)
    map_axes_y1 = []
    for row in range(2, 4):
        row_axes = [fig.add_subplot(gs[row, col], projection=ccrs.PlateCarree(180)) for col in range(2)]
        map_axes_y1.append(row_axes)

    # Row 4: ONI index panel spanning all columns
    ax_index = fig.add_subplot(gs[4, :])

    # ── Helper: draw one SST panel ─────────────────────────────────────────────
    def draw_panel(ax, sst_season, title, show_title=True):
        im = sst_season.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap='RdYlBu_r',
            vmin=5, vmax=30,
            add_colorbar=False
        )
        lons = sst_season.coords[
            next(n for n in ['longitude', 'lon', 'longitudes'] if n in sst_season.coords)
        ].values
        lats = sst_season.coords[
            next(n for n in ['latitude', 'lat', 'latitudes'] if n in sst_season.coords)
        ].values
        ax.contour(
            lons, lats, sst_season.values,
            levels=[29], colors='black',
            linewidths=1.2, linestyles='--',
            transform=ccrs.PlateCarree()
        )
        ax.coastlines(linewidth=0.5)
        ax.set_extent([120, 290, -40, 40], crs=ccrs.PlateCarree())
        if show_title:
            ax.set_title(title, fontsize=11, weight='bold')
        else:
            ax.set_title('')
        return im

    im = None   # will hold last mappable for colorbar

    # ── Year 0 rows (El Niño row 0, La Niña row 1) ────────────────────────────
    y0_seasons   = ['JJA', 'SON', 'DJF', 'MAM']
    y0_titles    = ['JJA (Yr 0)', 'SON (Yr 0)', 'DJF (Yr 0/1)', 'MAM (Yr +1)']
    y0_composites = [el_nino_composite, la_nina_composite]
    y0_labels     = [
        f'El Niño  Yr 0\n(n={len(el_nino_years)})',
        f'La Niña  Yr 0\n(n={len(la_nina_years)})'
    ]

    for row, (composite, label) in enumerate(zip(y0_composites, y0_labels)):
        for col, (season, title) in enumerate(zip(y0_seasons, y0_titles)):
            show = (row == 0)           # column titles only on first row
            im = draw_panel(map_axes_y0[row][col], composite.sel(season=season), title, show)
        map_axes_y0[row][0].text(
            -0.10, 0.5, label,
            transform=map_axes_y0[row][0].transAxes,
            fontsize=10, weight='bold', va='center', ha='right', rotation=90
        )

    # ── Year+1 rows (El Niño row 2, La Niña row 3) ────────────────────────────
    y1_seasons    = ['JJA', 'SON']
    y1_titles     = ['JJA (Yr +1)', 'SON (Yr +1)']
    y1_composites = [el_nino_composite_y1, la_nina_composite_y1]
    y1_labels     = [
        f'El Niño  Yr +1\n(n={len(el_nino_years)})',
        f'La Niña  Yr +1\n(n={len(la_nina_years)})'
    ]

    for row, (composite, label) in enumerate(zip(y1_composites, y1_labels)):
        for col, (season, title) in enumerate(zip(y1_seasons, y1_titles)):
            show = (row == 0)
            im = draw_panel(map_axes_y1[row][col], composite.sel(season=season), title, show)
        map_axes_y1[row][0].text(
            -0.10, 0.5, label,
            transform=map_axes_y1[row][0].transAxes,
            fontsize=10, weight='bold', va='center', ha='right', rotation=90
        )

    # ── Shared colorbar (right side, all map rows) ─────────────────────────────
    all_map_axes = (
        [ax for row in map_axes_y0 for ax in row] +
        [ax for row in map_axes_y1 for ax in row]
    )
    cbar = fig.colorbar(
        im, ax=all_map_axes,
        orientation='horizontal', shrink=0.6, pad=0.02, cmap='RdYlBu_r',shading = 'gourond',
        label='SST (°C)', extend='both'
    )
    cbar.ax.axhline(29, color='black', linewidth=1.2, linestyle='--')
    cbar.ax.text(2.6, 29, '29°C', va='center', fontsize=8, color='black')

    # ── Composite Niño 3.4 index panel (16 months) ────────────────────────────
    month_labels = [
        'Jun','Jul','Aug',        # JJA  yr0   0-2
        'Sep','Oct','Nov',        # SON  yr0   3-5
        'Dec','Jan','Feb',        # DJF        6-8
        'Mar','Apr','May',        # MAM  yr1   9-11
        'Jun','Jul','Aug',        # JJA  yr1   12-14
        'Sep'                     # SON  yr1   15
    ]
    x = np.arange(16)

    en_mean, en_std = build_composite_oni(oni_series, el_nino_years)
    ln_mean, ln_std = build_composite_oni(oni_series, la_nina_years)

    ax_index.plot(x, en_mean, color='red',  linewidth=2.0,
                  label=f'El Niño composite (n={len(el_nino_years)})', zorder=3)
    ax_index.fill_between(x, en_mean - en_std, en_mean + en_std,
                          color='red',  alpha=0.20, zorder=2)
    ax_index.plot(x, ln_mean, color='blue', linewidth=2.0,
                  label=f'La Niña composite (n={len(la_nina_years)})', zorder=3)
    ax_index.fill_between(x, ln_mean - ln_std, ln_mean + ln_std,
                          color='blue', alpha=0.20, zorder=2)

    ax_index.axhline( 0.5, color='red',  linewidth=0.8, linestyle='--', alpha=0.6)
    ax_index.axhline(-0.5, color='blue', linewidth=0.8, linestyle='--', alpha=0.6)
    ax_index.axhline( 0.0, color='grey', linewidth=0.7)

    season_bands = [
        ('JJA(0)',   0,  3,  '#fffde7'),
        ('SON(0)',   3,  6,  '#fff3e0'),
        ('DJF',      6,  9,  '#e3f2fd'),
        ('MAM',      9,  12, '#e8f5e9'),
        ('JJA(+1)', 12, 15,  '#fffde7'),
        ('SON(+1)', 15, 16,  '#fff3e0'),
    ]
    for s_label, x0, x1, color in season_bands:
        ax_index.axvspan(x0, x1 - 0.02, alpha=0.30, color=color, zorder=1)
        ax_index.text((x0 + x1) / 2, -2.6, s_label,
                      ha='center', va='top', fontsize=9, color='dimgray', style='italic')

    ax_index.axvline(11.5, color='dimgray', linewidth=1.0, linestyle=':', zorder=4)
    ax_index.text(5.5,  2.7, 'Year 0',  ha='center', fontsize=10, color='dimgray', weight='bold')
    ax_index.text(13.5, 2.7, 'Year +1', ha='center', fontsize=10, color='dimgray', weight='bold')

    ax_index.set_xticks(x)
    ax_index.set_xticklabels(month_labels, fontsize=9)
    ax_index.set_xlim(-0.5, 15.5)
    ax_index.set_ylabel('SST Anomaly (°C)', fontsize=10)
    ax_index.set_xlabel('Month of ENSO Lifecycle', fontsize=10)
    ax_index.set_title('Composite Niño 3.4 Index Lifecycle  (Jun Year 0 → Sep Year +1)',
                       fontsize=11, weight='bold')
    ax_index.legend(loc='upper right', fontsize=9, framealpha=0.8)
    ax_index.grid(True, linestyle=':', linewidth=0.5, alpha=0.7)

    fig.suptitle(
        'Pacific SST Seasonal Composite: El Niño vs La Niña\n'
        '(Year 0: JJA → SON → DJF → MAM  |  Year +1: JJA → SON)  |  Dashed contour = 29°C',
        fontsize=13, weight='bold', y=1.005
    )

    save_path = os.path.join(output_dir, "Composite_ElNino_vs_LaNina_29C_ONI.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_and_save_enso(nc_path, year, event_type, output_dir="ENSO_plots"):
    os.makedirs(output_dir, exist_ok=True)

    if any(ch in nc_path for ch in ['*', '?', '[']):
        ds = xr.open_mfdataset(nc_path, combine='by_coords')
    else:
        ds = xr.open_dataset(nc_path)

    time_coord = next((n for n in ['time', 'valid_time', 'times', 'date'] if n in ds.coords), None)
    lon_coord  = next((n for n in ['longitude', 'lon', 'longitudes'] if n in ds.coords), None)
    lat_coord  = next((n for n in ['latitude', 'lat', 'latitudes'] if n in ds.coords), None)

    selection = {time_coord: slice(f'{year}-06', f'{year+1}-09')}
    if lon_coord:
        selection[lon_coord] = slice(120, 290)
    if lat_coord:
        lat_slice = slice(40, -40) if ds[lat_coord].values[0] > ds[lat_coord].values[-1] else slice(-40, 40)
        selection[lat_coord] = lat_slice

    sst_data = ds['sst'].sel(selection)
    if sst_data.mean() > 200:
        event_sst = sst_data.groupby(f'{time_coord}.season').mean(time_coord) - 273.15
    else:
        event_sst = sst_data.groupby(f'{time_coord}.season').mean(time_coord)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree(180)})
    for ax, season in zip(axes.flatten(), ['JJA', 'SON', 'DJF', 'MAM']):
        im = event_sst.sel(season=season).plot(
            ax=ax, transform=ccrs.PlateCarree(),
            cmap='RdYlBu_r', vmin=5, vmax=30, add_colorbar=False
        )
        ax.coastlines(linewidth=0.5)
        ax.set_extent([120, 290, -40, 40], crs=ccrs.PlateCarree())
        display_year = year + 1 if season == 'MAM' else year
        ax.set_title(f'{season} ({display_year})', weight='bold')

    fig.colorbar(im, ax=axes.ravel().tolist(), orientation='horizontal',
                 shrink=0.6, pad=0.08, label='SST (°C)')
    plt.suptitle(f'Pacific SST Seasonal Evolution: {year}-{year+1} ({event_type})', fontsize=14, weight='bold')
    filename = f"{event_type.replace(' ', '_')}_{year}_{year+1}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {filename}")


# ── EXECUTION ──────────────────────────────────────────────────────────────────

file_path      = '/Users/dsha0113/Documents/PhD_Work/Data_Analysis/Netcdf_data/MonthlymeanSST.nc'
save_directory = '/Users/dsha0113/Documents/PhD_Work/Data_Analysis/SST_ENSO_codes/ENSO_seasonal_cycle_plots'

print("Detecting ENSO events...")
el_nino_list, la_nina_list, oni_series = compute_enso_events(file_path)
print(f"El Niño years : {el_nino_list}")
print(f"La Niña years : {la_nina_list}")

for year in el_nino_list:
    plot_and_save_enso(file_path, year, event_type="El Nino", output_dir=save_directory)
for year in la_nina_list:
    plot_and_save_enso(file_path, year, event_type="La Nina", output_dir=save_directory)

print("\nBuilding Year 0 composites...")
el_nino_composite    = build_composite(file_path, el_nino_list)
la_nina_composite    = build_composite(file_path, la_nina_list)

print("Building Year+1 composites...")
el_nino_composite_y1 = build_composite_yr1(file_path, el_nino_list)
la_nina_composite_y1 = build_composite_yr1(file_path, la_nina_list)

plot_combined_composite(
    el_nino_composite,    la_nina_composite,
    el_nino_composite_y1, la_nina_composite_y1,
    el_nino_years=el_nino_list,
    la_nina_years=la_nina_list,
    oni_series=oni_series,
    output_dir=save_directory
)

print("\nAll plots saved successfully.")