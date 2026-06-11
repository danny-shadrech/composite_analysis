import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

# 1. Load data and setup shifting coordinates for DJF
ds = xr.open_dataset('/Users/dsha0113/Documents/PhD_Work/Data_Analysis/Netcdf_data/skj_Btot_hist_1980_2010.nc', decode_times=False)
time = pd.date_range('1980-01-15', periods=ds.sizes['time'], freq='MS') + pd.Timedelta(days=14)
ds = ds.assign_coords(time=time)
ds.coords['season_year'] = ds.time.dt.year + (ds.time.dt.month // 12)
ds.coords['s'] = ds.time.dt.season
w = ds.time.dt.days_in_month

# 2. Compute Climatology and Seasonal Time Series
data = ds['skj_Btot']
clim = {s: (data * w).where(ds.s == s).groupby('season_year').sum('time') / w.where(ds.s == s).groupby('season_year').sum('time') for s in ['DJF', 'MAM', 'JJA', 'SON']}
ts = {s: clim[s] for s in clim}  # Reference time series
clim_mean = {s: clim[s].mean('season_year') for s in clim}

# 3. Define Lifecycle Sequence (Season, Year Offset)
stages = [('JJA', 0), ('SON', 0), ('DJF', 1), ('MAM', 1), ('JJA', 1), ('SON', 1)]
offset_label = {0: 'Early Development', 1: 'Intensification', 2: 'Peak Phase', 3: 'Early Decay', 4: ' Mid Decay', 5: 'Late Decay'}


lanina_years = [1985,1988, 1995, 1998, 1999, 2000, 2007, 2008, 2010, 2010]
elnino_years = [1982, 1986, 1987, 1991, 1994, 1997, 2002, 2004, 2006, 2009]

# 4. Compute Composites and Plot immediately
fig1, axes = plt.subplots(3, 2, figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree(180)})
# Add a main figure title and reserve space at top
fig1.suptitle('Composite: Skip-Jack tuna total biomass anomaly mean distribution during La Niña life cycle', fontsize=14, fontweight='semibold')
# Increase vertical spacing between subplots
fig1.subplots_adjust(top=0.90, hspace=0.35)
for ax, (idx,(s, offset)) in zip(axes.flatten(), enumerate(stages)):
    
    # Select target years and subtract climatology
    t_years = [y + offset for y in lanina_years] # Personally add 'El nino or La Nina selected years here:
    anom = ts[s].sel(season_year=ts[s].season_year.isin(t_years)).mean('season_year') - clim_mean[s]
    
    # Smooth the anomaly field for better plotting
    anom = anom.rolling(lat=8, lon=8, center=True, min_periods=1).mean()
    
    # Plot directly to the axis
    lon = anom['lon']
    lat = anom['lat']
    mesh = ax.pcolormesh(lon, lat, anom, transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=-0.05, vmax=0.05, shading='auto')
    ax.coastlines()
    ax.set_extent([120, 290, -30, 30], crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), linewidth=0.5, color='gray', alpha=0.7, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8}
    ax.set_title(f"{s} ({offset_label[idx]})")

fig1.colorbar(mesh, ax=axes.ravel().tolist(), orientation='horizontal', shrink=0.5, pad=0.05, label='Anomaly (mt/km²)')



# 5. Compute Composites and Plot immediately
fig2, axes = plt.subplots(3, 2, figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree(180)})
# Add a main figure title and reserve space at top
fig2.suptitle('Composite: Skip-Jack tuna total biomass anomaly mean distribution during El Niño life cycle', fontsize=14, fontweight='semibold')
# Increase vertical spacing between subplots
fig2.subplots_adjust(top=0.90, hspace=0.35)
for ax, (idx,(s, offset)) in zip(axes.flatten(), enumerate(stages)):
    
    # Select target years and subtract climatology
    t_years = [y + offset for y in elnino_years] # Personally add 'El nino or La Nina selected years here:
    anom = ts[s].sel(season_year=ts[s].season_year.isin(t_years)).mean('season_year') - clim_mean[s]
    
    # Smooth the anomaly field for better plotting
    anom = anom.rolling(lat=8, lon=8, center=True, min_periods=1).mean()
    
    # Plot directly to the axis
    lon = anom['lon']
    lat = anom['lat']
    mesh = ax.pcolormesh(lon, lat, anom, transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=-0.05, vmax=0.05, shading='auto')
    ax.coastlines()
    ax.set_extent([120, 290, -30, 30], crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), linewidth=0.5, color='gray', alpha=0.7, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8}
    ax.set_title(f"{s} ({offset_label[idx]})")

fig2.colorbar(mesh, ax=axes.ravel().tolist(), orientation='horizontal', shrink=0.5, pad=0.05, label='Anomaly (mt/km²)')
plt.show()