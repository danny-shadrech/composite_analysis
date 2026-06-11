import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from numpy import s_
import xarray as xr

# 1. Load data and setup shifting coordinates for DJF
ds = xr.open_dataset('/Users/dsha0113/Documents/PhD_Work/Data_Analysis/Netcdf_data/MonthlymeanSST.nc')
ds.coords['season_year'] = ds.valid_time.dt.year + (ds.valid_time.dt.month // 12)
ds.coords['s'] = ds.valid_time.dt.season
ds.coords['longitude'] = ((ds.longitude + 180) % 360)-180 # shift to -180 to 180 for easier plotting
ds = ds.sortby('longitude')  # Ensure longitudes are in ascending order after shifting
w = ds.valid_time.dt.days_in_month

# 2. Compute Climatology and Seasonal Time Series
clim = {s: (ds.sst * w).where(ds.s == s).groupby('season_year').sum('valid_time') / w.where(ds.s == s).groupby('season_year').sum('valid_time') for s in ['DJF', 'MAM', 'JJA', 'SON']}
ts = {s: clim[s] for s in clim}  # Reference time series
clim_mean = {s: clim[s].mean('season_year') for s in clim}

# 3. Define Lifecycle Sequence (Season, Year Offset)
stages = [('JJA', 0), ('SON', 0), ('DJF', 1), ('MAM', 1), ('JJA', 1), ('SON',1)]
offset_label = {0:'Early Development - previous yr', 1: 'Intensification - previous yr', 2: 'Peak Phase- pre/next yr', 3: 'Early Decay - next yr', 4: 'Mid Decay - next yr', 5: 'Late Decay - next yr'}

elnino_years = [1982, 1986, 1987, 1991, 1994, 1997, 2002, 2004, 2006, 2009]
lanina_years = [1985,1988, 1995, 1999, 2000, 2007, 2010]

# 4. Compute Composites and Plot immediately
fig1, axes1 = plt.subplots(3, 2, figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree(180)})
# Add a main figure title and reserve space at top
fig1.suptitle('El Niño Composite Lifecycle', fontsize=14, fontweight='semibold')
# Increase vertical spacing between subplots
fig1.subplots_adjust(top=0.90, hspace=0.35)

for ax1, (idx, (s, offset)) in zip(axes1.flatten(), enumerate(stages)):
    # Select target years and subtract climatology
    t_years = [y + offset for y in elnino_years]
    anom = ts[s].sel(season_year=ts[s].season_year.isin(t_years)).mean('season_year') - clim_mean[s]
    
    # Plot directly to the axis
    lon, lat = anom.coords[anom.dims[1]], anom.coords[anom.dims[0]]
    mesh = ax1.pcolormesh(lon, lat, anom, transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=-1.5, vmax=1.5)
    ax1.coastlines()
    ax1.set_extent([120, 290, -30, 30], crs=ccrs.PlateCarree())
    gl = ax1.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), linewidth=0.5, color='gray', alpha=0.7, linestyle='--', xlocs=range(120, 291, 20), ylocs=range(-30, 31, 10))
    gl.top_labels = False
    gl.right_labels = False
    gl.bottom_labels = True
    gl.left_labels = True
    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8}
    def lon_formatter(x, pos):
        x = x % 360
        if x == 0:
            return '0°'
        if x <= 180:
            return f"{int(x)}°E"
        return f"{int(360 - x)}°W"
    gl.xformatter = mticker.FuncFormatter(lon_formatter)
    gl.yformatter = cticker.LatitudeFormatter()
    ax1.set_title(f"{s} ({offset_label[idx]})")

fig1.colorbar(mesh, ax=axes1.ravel().tolist(), orientation='horizontal', shrink=0.5, pad=0.05, label='SST Anomaly (°C)')

  
# 5. Compute Composites and Plot immediately
fig2, axes2 = plt.subplots(3, 2, figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree(180)})
# Add a main figure title and reserve space at top
fig2.suptitle('La Niña Composite Lifecycle', fontsize=14, fontweight='semibold')
# Increase vertical spacing between subplots
fig2.subplots_adjust(top=0.90, hspace=0.35)

for ax2, (idx, (s, offset)) in zip(axes2.flatten(), enumerate(stages)):
    # Select target years and subtract climatology
    t_years = [y + offset for y in lanina_years]
    anom = ts[s].sel(season_year=ts[s].season_year.isin(t_years)).mean('season_year') - clim_mean[s]
    
    # Plot directly to the axis
    lon, lat = anom.coords[anom.dims[1]], anom.coords[anom.dims[0]]
    mesh = ax2.pcolormesh(lon, lat, anom, transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=-1.5, vmax=1.5)
    ax2.coastlines()
    ax2.set_extent([120, 290, -30, 30], crs=ccrs.PlateCarree())
    gl = ax2.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), linewidth=0.5, color='gray', alpha=0.7, linestyle='--', xlocs=range(120, 291, 20), ylocs=range(-30, 31, 10))
    gl.top_labels = False
    gl.right_labels = False
    gl.bottom_labels = True
    gl.left_labels = True
    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8}
    def lon_formatter(x, pos):
        x = x % 360
        if x == 0:
            return '0°'
        if x <= 180:
            return f"{int(x)}°E"
        return f"{int(360 - x)}°W"
    gl.xformatter = mticker.FuncFormatter(lon_formatter)
    gl.yformatter = cticker.LatitudeFormatter()
    ax2.set_title(f"{s} ({offset_label[idx]})")

fig2.colorbar(mesh, ax=axes2.ravel().tolist(), orientation='horizontal', shrink=0.5, pad=0.05, label='SST Anomaly (°C)')


# 6. Compute Composites and Plot immediately
fig3, axes3 = plt.subplots(3, 2, figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree(180)})
# Add a main figure title and reserve space at top
fig3.suptitle('Mean SST for El Niño Composite Lifecycle', fontsize=14, fontweight='semibold')
# Increase vertical spacing between subplots
fig3 .subplots_adjust(top=0.90, hspace=0.35)

for ax3, (idx, (s, offset)) in zip(axes3.flatten(), enumerate(stages)):
    # Select target years and plot the mean state (not anomaly)
    t_years = [y + offset for y in elnino_years]
    s_mean = ts[s].sel(season_year=ts[s].season_year.isin(t_years)).mean('season_year')
    
    # Plot directly to the axis
    lon, lat = s_mean.coords[s_mean.dims[1]], s_mean.coords[s_mean.dims[0]]
    mesh = ax3.pcolormesh(lon, lat, s_mean, transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=5, vmax=320)
    ax3.coastlines()
    ax3.set_extent([120, 290, -30, 30], crs=ccrs.PlateCarree())
    gl = ax3.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), linewidth=0.5, color='gray', alpha=0.7, linestyle='--', xlocs=range(120, 291, 20), ylocs=range(-30, 31, 10))
    gl.top_labels = False
    gl.right_labels = False
    gl.bottom_labels = True
    gl.left_labels = True
    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8}
    def lon_formatter(x, pos):
        x = x % 360
        if x == 0:
            return '0°'
        if x <= 180:
            return f"{int(x)}°E"
        return f"{int(360 - x)}°W"
    gl.xformatter = mticker.FuncFormatter(lon_formatter)
    gl.yformatter = cticker.LatitudeFormatter()
    ax3.set_title(f"{s} ({offset_label[idx]})")

fig3.colorbar(mesh, ax=axes3.ravel().tolist(), orientation='horizontal', shrink=0.5, pad=0.05, label='SST Anomaly (°C)')

plt.show()
