"""
Functions for looking at observed correlations between reanalysis products and the observed energy data.

Author: Ben Hutchins
Date: February 2023
"""

# Import local modules
import sys
import os
import glob

# Import third-party modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import (LongitudeFormatter, LatitudeFormatter, LatitudeLocator)
import iris
import xarray as xr
from tqdm import tqdm
from scipy.stats import pearsonr, linregress, t
from scipy import signal
from datetime import datetime
import geopandas as gpd
import regionmask

sys.path.append("/home/users/benhutch/energy-met-corr/")
# Import local modules
import dictionaries_em as dicts

# Import external modules
sys.path.append("/home/users/benhutch/skill-maps/python/")
import paper1_plots_functions as p1p_funcs
import nao_alt_lag_functions as nal_funcs
import functions as fnc


# Define a function to form the dataframe for the offshore wind farm data
def extract_offshore_eez_to_df(
    filepath: str,
    countries_list: list = [
        "France",
        "Italy",
        "Portugal",
        "Estonia",
        "Latvia",
        "Lithuania",
        "Croatia",
        "Romania",
        "Slovenia",
        "Greece",
        "Montenegro",
        "Albania",
        "Bulgaria",
        "Spain",
        "Norway",
        "United Kingdom",
        "Ireland",
        "Finland",
        "Sweden",
        "Belgium",
        "Netherlands",
        "Germany",
        "Denmark",
        "Poland",
    ],
    rolling_window: int = 8,
    centre: bool = True,
    annual_offset: int = 3,
    months: list = [10, 11, 12, 1, 2, 3],
    start_date: str = "1950-01-01",
    time_unit: str = "h",
) -> pd.DataFrame:
    """
    Extracts the offshore wind farm data from the given file and returns it as a dataframe.

    Args:
        filepath: str
            The path to the file containing the offshore wind farm data.
        rolling_window: int
            The number of hours to use for the rolling window average.
        centre: bool
            Whether to centre the rolling window average.
        annual_offset: int
            The number of months to offset the annual average by.
        months: list
            The months to include in the annual average.
        start_date: str
            The start date for the data.
        time_unit: str
            The time unit for the data.
    Returns:
        df: pd.DataFrame
            The dataframe containing the offshore wind farm data.
    """
    # Find files
    files = glob.glob(filepath)

    # Assert that the file exists
    assert len(files) > 0, f"No files found at {filepath}"

    # Assert that there is only one file
    assert len(files) == 1, f"Multiple files found at {filepath}"

    # Load the data
    ds = xr.open_dataset(files[0])

    # Extract the values
    nuts_keys = ds.NUTS_keys.values

    # Turn the data into a dataframe
    df = ds.to_dataframe()

    # Create columns for each of the indexed NUTS regions
    # Pivot the DataFrame
    df = df.reset_index().pivot(
        index="time_in_hours_from_first_jan_1950",
        columns="NUTS",
        values="timeseries_data",
    )

    # Assuming country_dict is a dictionary that maps NUTS keys to country names
    df.columns = [
        f"{dicts.country_dict[nuts_keys[i]]}_{col}" for i, col in enumerate(df.columns)
    ]

    # Convert 'time_in_hours_from_first_jan_1950' column to datetime
    df.index = pd.to_datetime(df.index, unit=time_unit, origin=start_date)

    # Collapse the dataframes into monthly averages
    df = df.resample("M").mean()

    # Select only the months of interest
    df = df[df.index.month.isin(months)]

    # Shift the data by the annual offset
    df.index = df.index - pd.DateOffset(months=annual_offset)

    # TODO: Fix hard coded here
    # Throw away the first 3 months of data and last 3 months of data
    df = df.iloc[3:-3]

    # Calculate the annual average
    df = df.resample("A").mean()

    # Take the rolling average
    df = df.rolling(window=rolling_window, center=centre).mean()

    # Throw away the NaN values
    df = df.dropna()

    # Return the dataframe
    return df


# Write a function to calculate the stats
def calc_nao_spatial_corr(
    season: str,
    forecast_range: str,
    start_year: int,
    end_year: int,
    corr_var: str = "tos",
    corr_var_obs_file: str = dicts.regrid_file,
    nao_obs_var: str = "msl",
    nao_obs_file: str = dicts.regrid_file,
    nao_n_grid: dict = dicts.iceland_grid_corrected,
    nao_s_grid: dict = dicts.azores_grid_corrected,
    sig_threshold: float = 0.05,
    level: int = 0,
):
    """
    Calculates the spatial correlations between the NAO index (winter default)
    and the variable to correlate for the observations.

    Args:
    -----

    season: str
        The season to calculate the correlation for.

    forecast_range: str
        The forecast range to calculate the correlation for.

    start_year: int
        The start year to calculate the correlation for.

    end_year: int
        The end year to calculate the correlation for.

    corr_var: str
        The variable to correlate with the NAO index.

    corr_var_obs_file: str
        The file containing the observations of the variable to correlate.

    nao_obs_var: str
        The variable to use for the NAO index.

    nao_obs_file: str
        The file containing the observations of the NAO index.

    nao_n_grid: dict
        The dictionary containing the grid information for the northern node
        of the winter NAO index.

    nao_s_grid: dict
        The dictionary containing the grid information for the southern node
        of the winter NAO index.

    sig_threshold: float
        The significance threshold for the correlation.

    level: int
        The pressure level at which to extract the observed variable.
        in Pa, so 850 hPa = 85000 Pa.

    Returns:
    --------

    stats_dict: dict
        The dictionary containing the correlation statistics.
    """

    # Set up the mdi
    mdi = -9999.0

    # Form the dictionary
    stats_dict = {
        "nao": [],
        "corr_var_ts": [],
        "corr_var": corr_var,
        "corr_nao_var": [],
        "corr_nao_var_pval": [],
        "init_years": [],
        "valid_years": [],
        "lats": [],
        "lons": [],
        "season": season,
        "forecast_range": forecast_range,
        "start_year": start_year,
        "end_year": end_year,
        "sig_threshold": sig_threshold,
    }

    # Set up the init years
    stats_dict["init_years"] = np.arange(start_year, end_year + 1)

    # Assert that the season is a winter season
    assert season in ["DJF", "ONDJFM", "DJFM"], "The season must be a winter season."

    # Assert that the forecast range is a valid forecast range
    assert "-" in forecast_range, "The forecast range must be a valid forecast range."

    # Set up the lons and lats for the south grid
    s_lon1, s_lon2 = nao_s_grid["lon1"], nao_s_grid["lon2"]
    s_lat1, s_lat2 = nao_s_grid["lat1"], nao_s_grid["lat2"]

    # and for the north grid
    n_lon1, n_lon2 = nao_n_grid["lon1"], nao_n_grid["lon2"]
    n_lat1, n_lat2 = nao_n_grid["lat1"], nao_n_grid["lat2"]

    # First check that the file exists for psl
    assert os.path.exists(
        corr_var_obs_file
    ), "The file for the variable to correlate does not exist."

    # Check that the file exists for the NAO index
    assert os.path.exists(nao_obs_file), "The file for the NAO index does not exist."

    # Load the observations for psl
    psl = fnc.load_obs(variable=nao_obs_var, regrid_obs_path=nao_obs_file)

    # Load the observations for the matching var
    corr_var_field = fnc.load_obs(variable=corr_var, regrid_obs_path=corr_var_obs_file)

    # print the dims of the corr
    print("corr_var_field.dims: ", corr_var_field)

    # If level is not 0
    if level != 0:
        # Extract the level
        corr_var_field = corr_var_field.extract(iris.Constraint(air_pressure
                                                                =level))

    # print the dimensions of the corr_var_field
    print("corr_var_field.dims: ", corr_var_field)

    # extract the months
    months = dicts.season_month_map[season]

    # Set up an iris constraint for the start and end years
    start_date = datetime(int(start_year), months[0], 1)
    end_date = datetime(int(end_year), months[-1], 31)

    # Form the constraint
    time_constraint = iris.Constraint(
        time=lambda cell: start_date <= cell.point <= end_date
    )

    # Apply the constraint
    psl = psl.extract(time_constraint)

    # Apply the constraint
    corr_var_field = corr_var_field.extract(time_constraint)

    # Set up the constrain for months
    month_constraint = iris.Constraint(time=lambda cell: cell.point.month in months)

    # Apply the constraint
    psl = psl.extract(month_constraint)

    # Apply the constraint
    corr_var_field = corr_var_field.extract(month_constraint)

    # Calculate the climatology by collapsing the time dimension
    psl_clim = psl.collapsed("time", iris.analysis.MEAN)

    # Calculate the climatology by collapsing the time dimension
    corr_var_clim = corr_var_field.collapsed("time", iris.analysis.MEAN)

    # Calculate the anomalies
    psl_anom = psl - psl_clim

    # Calculate the anomalies
    corr_var_anom = corr_var_field - corr_var_clim

    # Calculate the annual mean anoms
    psl_anom = fnc.calculate_annual_mean_anomalies(
        obs_anomalies=psl_anom, season=season
    )

    # Calculate the annual mean anoms
    corr_var_anom = fnc.calculate_annual_mean_anomalies(
        obs_anomalies=corr_var_anom, season=season
    )

    # # Print psl anom at the first time step
    # print("psl anom at the first time step: ", psl_anom.isel(time=0).values)

    # # print corr_var anom at the first time step
    # print("corr_var anom at the first time step: ", corr_var_anom.isel(time=0).values)

    # Select the forecast range
    psl_anom = fnc.select_forecast_range(
        obs_anomalies_annual=psl_anom, forecast_range=forecast_range
    )

    # Select the forecast range
    corr_var_anom = fnc.select_forecast_range(
        obs_anomalies_annual=corr_var_anom, forecast_range=forecast_range
    )

    # Print the length of the time axis for psl_anom
    print("len(psl_anom.time): ", len(psl_anom.time))

    # Print the length of the time axis for corr_var_anom
    print("len(corr_var_anom.time): ", len(corr_var_anom.time))

    # Years 2-9, gives an 8 year running mean
    # Which means that the first 4 years (1960, 1961, 1962, 1963) are not valid
    # And the last 4 years (2011, 2012, 2013, 2014) are not valid
    # extract the digits from the forecast range
    digits = [int(x) for x in forecast_range.split("-")]
    # Find the absolute difference between the digits
    diff = abs(digits[0] - digits[1])

    # Find the number of invalid years after centred running mean on each end
    n_invalid_years = (diff + 1) / 2

    # Print the number of invalid years
    print("n_invalid_years: ", n_invalid_years)

    # Subset corr_var_anom to remove the invalid years
    corr_var_anom = corr_var_anom.isel(
        time=slice(int(n_invalid_years), -int(n_invalid_years))
    )

    # # Loop over the years in psl_anom
    # for year in psl_anom.time.dt.year.values:
    #     # Extract the data for the year
    #     psl_anom_year = psl_anom.sel(time=f"{year}")

    #     # If there are any NaNs, log it
    #     if np.isnan(psl_anom_year).any():
    #         print("There are NaNs in the psl_anom_year for year: ", year)
    #         # if all values are NaN, then continue
    #         if np.all(np.isnan(psl_anom_year)):
    #             print("All values are NaN for year: ", year)
    #             print("Removing the year: ", year)
    #             # Remove the year from the psl_anom
    #             psl_anom = psl_anom.sel(time=psl_anom.time.dt.year != year)

    # # Loop over the first 10 years and last 10 years in psl_anom
    # for year in corr_var_anom.time.dt.year.values[:10]:
    #     # Extract the data for the year
    #     corr_var_anom_year = corr_var_anom.sel(time=f"{year}")

    #     # If there are any NaNs, log it
    #     if np.isnan(corr_var_anom_year).any():
    #         print("There are NaNs in the corr_var_anom_year for year: ", year)
    #         # if all values are NaN, then continue
    #         if np.all(np.isnan(corr_var_anom_year)):
    #             print("All values are NaN for year: ", year)
    #             print("Removing the year: ", year)
    #             # Remove the year from the psl_anom
    #             corr_var_anom = corr_var_anom.sel(time=corr_var_anom.time.dt.year != year)

    # # Loop over the last 10 years in psl_anom
    # for year in corr_var_anom.time.dt.year.values[-10:]:
    #     # Extract the data for the year
    #     corr_var_anom_year = corr_var_anom.sel(time=f"{year}")

    #     # If there are any NaNs, log it
    #     if np.isnan(corr_var_anom_year).any():
    #         print("There are NaNs in the corr_var_anom_year for year: ", year)
    #         # if all values are NaN, then continue
    #         if np.all(np.isnan(corr_var_anom_year)):
    #             print("All values are NaN for year: ", year)
    #             print("Removing the year: ", year)
    #             # Remove the year from the psl_anom
    #             corr_var_anom = corr_var_anom.sel(time=corr_var_anom.time.dt.year != year)

    # print the type of psl_anom
    print("type of psl_anom: ", type(psl_anom))

    # print the type of corr_var_anom
    print("type of corr_var_anom: ", type(corr_var_anom))

    # Extract the years for psl anom
    # years_psl = psl_anom.time.dt.year.values
    years_corr_var = corr_var_anom.time.dt.year.values

    # # Set the time axis for psl_anom to the years
    # psl_anom = psl_anom.assign_coords(time=years_psl)

    # Set the time axis for corr_var_anom to the years
    corr_var_anom = corr_var_anom.assign_coords(time=years_corr_var)

    # Lat goes from 90 to -90
    # Lon goes from 0 to 360

    # # If s_lat1 is smaller than s_lat2, then we need to switch them
    # if s_lat1 < s_lat2:
    #     s_lat1, s_lat2 = s_lat2, s_lat1

    # # If n_lat1 is smaller than n_lat2, then we need to switch them
    # if n_lat1 < n_lat2:
    #     n_lat1, n_lat2 = n_lat2, n_lat1

    # # Asert that the lons are within the range of 0 to 360
    # assert 0 <= s_lon1 <= 360, "The southern lon is not within the range of 0 to 360."

    # # Asert that the lons are within the range of 0 to 360
    # assert 0 <= s_lon2 <= 360, "The southern lon is not within the range of 0 to 360."

    # # Asert that the lons are within the range of 0 to 360
    # assert 0 <= n_lon1 <= 360, "The northern lon is not within the range of 0 to 360."

    # # Asert that the lons are within the range of 0 to 360
    # assert 0 <= n_lon2 <= 360, "The northern lon is not within the range of 0 to 360."

    # Constraint the psl_anom to the south grid
    psl_anom_s = psl_anom.sel(
        lon=slice(s_lon1, s_lon2), lat=slice(s_lat1, s_lat2)
    ).mean(dim=["lat", "lon"])

    # Constraint the psl_anom to the north grid
    psl_anom_n = psl_anom.sel(
        lon=slice(n_lon1, n_lon2), lat=slice(n_lat1, n_lat2)
    ).mean(dim=["lat", "lon"])

    # Calculate the nao index azores - iceland
    nao_index = psl_anom_s - psl_anom_n

    # Loop over the first 10 years and last 10 years in nao_index
    # for year in nao_index.time.dt.year.values:
    #     # Extract the data for the year
    #     nao_index_year = nao_index.sel(time=f"{year}")

    #     # If there are any NaNs, log it
    #     if np.isnan(nao_index_year).any():
    #         print("There are NaNs in the nao_index_year for year: ", year)
    #         # if all values are NaN, then continue
    #         if np.all(np.isnan(nao_index_year)):
    #             print("All values are NaN for year: ", year)
    #             print("Removing the year: ", year)
    #             # Remove the year from the nao_index
    #             nao_index = nao_index.sel(time=nao_index.time.dt.year != year)

    # Subset the nao_index to remove the invalid years
    nao_index = nao_index.isel(time=slice(int(n_invalid_years), -int(n_invalid_years)))

    # Extract the years for nao_index
    years_nao = nao_index.time.dt.year.values

    # Extract the years for corr_var_anom
    years_corr_var = corr_var_anom.time.values

    # Assert that the years are the same
    assert np.array_equal(
        years_nao, years_corr_var
    ), "The years for the NAO index and the variable to correlate are not the same."

    # Set the valid years
    stats_dict["valid_years"] = years_nao

    # extract tyhe lats and lons
    lats = corr_var_anom.lat.values

    # extract the lons
    lons = corr_var_anom.lon.values

    # Store the lats and lons in the dictionary
    stats_dict["lats"] = lats
    stats_dict["lons"] = lons

    # Extract the values for the NAO index
    nao_index_values = nao_index.values

    # Extract the values for the variable to correlate
    corr_var_anom_values = corr_var_anom.values

    # Store the nao index values in the dictionary
    stats_dict["nao"] = nao_index_values

    # Store the variable to correlate values in the dictionary
    stats_dict["corr_var_ts"] = corr_var_anom_values

    # # Create an empty array with the correct shape for the correlation
    # corr_nao_var = np.empty((len(lats), len(lons)))

    # # Create an empty array with the correct shape for the p-value
    # corr_nao_var_pval = np.empty((len(lats), len(lons)))

    # # Loop over the lats
    # for i in tqdm(range(len(lats)), desc="Calculating spatial correlation"):
    #     # Loop over the lons
    #     for j in range(len(lons)):
    #         # Extract the values for the variable to correlate
    #         corr_var_anom_values = corr_var_anom.values[:, i, j]

    #         # Calculate the correlation
    #         corr, pval = pearsonr(nao_index_values, corr_var_anom_values)

    #         # Store the correlation in the array
    #         corr_nao_var[i, j] = corr

    #         # Store the p-value in the array
    #         corr_nao_var_pval[i, j] = pval

    # # Store the correlation in the dictionary
    # stats_dict["corr_nao_var"] = corr_nao_var

    # # Store the p-value in the dictionary
    # stats_dict["corr_nao_var_pval"] = corr_nao_var_pval

    # return none
    return stats_dict


# define a simple function for plotting the correlation
def plot_corr(
    corr_array: np.ndarray,
    pval_array: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    variable: str,
    sig_threshold: float = 0.05,
    plot_gridbox: list = None,
    nao: np.ndarray = None,
    corr_var_ts: np.ndarray = None,
):
    """
    Plots the correlation and p-values for the spatial correlation.

    Args:
    -----

    corr_array: np.ndarray
        The array containing the correlation values.

    pval_array: np.ndarray
        The array containing the p-values.

    lats: np.ndarray
        The array containing the latitudes.

    lons: np.ndarray
        The array containing the longitudes.

    variable: str
        The variable to use for the plot title.

    sig_threshold: float
        The significance threshold for the correlation.

    plot_gridbox: list
        List of gridboxes to plot on the plot.

    nao: np.ndarray
        The array containing the NAO index values.

    corr_var_ts: np.ndarray
        The array containing the variable to correlate values.

    Returns:
    --------

    None
    """

    # Plot these values
    # Set up a single subplot
    fig = plt.figure(figsize=(10, 5))

    # Set up the projection
    proj = ccrs.PlateCarree(central_longitude=0)

    # Focus on the euro-atlantic region
    lat1_grid, lat2_grid = 30, 80
    lon1_grid, lon2_grid = -60, 40

    lat1_idx_grid = np.argmin(np.abs(lats - lat1_grid))
    lat2_idx_grid = np.argmin(np.abs(lats - lat2_grid))

    lon1_idx_grid = np.argmin(np.abs(lons - lon1_grid))
    lon2_idx_grid = np.argmin(np.abs(lons - lon2_grid))

    # # Print the indices
    # print("lon1_idx_grid: ", lon1_idx_grid)
    # print("lon2_idx_grid: ", lon2_idx_grid)
    # print("lat1_idx_grid: ", lat1_idx_grid)
    # print("lat2_idx_grid: ", lat2_idx_grid)

    # # # If lat1_idx_grid is greater than lat2_idx_grid, then switch them
    # # if lat1_idx_grid > lat2_idx_grid:
    # #     lat1_idx_grid, lat2_idx_grid = lat2_idx_grid, lat1_idx_grid

    # # Print the indices
    # print("lon1_idx_grid: ", lon1_idx_grid)
    # print("lon2_idx_grid: ", lon2_idx_grid)
    # print("lat1_idx_grid: ", lat1_idx_grid)
    # print("lat2_idx_grid: ", lat2_idx_grid)

    # Constrain the lats and lons to the grid
    lats = lats[lat1_idx_grid:lat2_idx_grid]
    lons = lons[lon1_idx_grid:lon2_idx_grid]

    # Constrain the corr_array to the grid
    corr_array = corr_array[lat1_idx_grid:lat2_idx_grid, lon1_idx_grid:lon2_idx_grid]

    # Constrain the pval_array to the grid
    pval_array = pval_array[lat1_idx_grid:lat2_idx_grid, lon1_idx_grid:lon2_idx_grid]

    # # print the shape of the pval_array
    # print("pval_array.shape: ", pval_array.shape)

    # # Print the values of the pval_array
    # print("pval_array: ", pval_array)

    # If nao and corr_var_ts are not None
    if nao is not None and corr_var_ts is not None:
        # Constraint the corr_var_ts array to the grid
        corr_var_ts = corr_var_ts[
            :, lat1_idx_grid:lat2_idx_grid, lon1_idx_grid:lon2_idx_grid
        ]

    # Set up the contour levels
    clevs = np.arange(-1.0, 1.1, 0.1)

    # Set up the axis
    ax = plt.axes(projection=proj)

    # Include coastlines
    ax.coastlines()

    # # Shift lon back to -180 to 180
    # lons = lons - 180

    # Set up the contour plot
    cf = ax.contourf(lons, lats, corr_array, clevs, transform=proj, cmap="RdBu_r")

    # if any of the p values are greater or less than the significance threshold
    sig_threshold = 0.05
    pval_array[(pval_array > sig_threshold) & (pval_array < 1 - sig_threshold)] = np.nan

    # Plot the p-values
    ax.contourf(lons, lats, pval_array, hatches=["...."], alpha=0.0, transform=proj)

    # Set up the colorbar
    cbar = plt.colorbar(cf, ax=ax, orientation="horizontal", pad=0.05, shrink=0.8)

    # If the plot_gridbox is not None
    if plot_gridbox is not None:
        # Assert that it is a list
        assert isinstance(
            plot_gridbox, list
        ), "The plot_gridbox must be a list of gridboxes."

        # Assert that it is not empty
        assert len(plot_gridbox) > 0, "The plot_gridbox list is empty."

        # Loop over the gridboxes
        for gridbox in plot_gridbox:
            # Extract the lons and lats
            lon1, lon2 = gridbox["lon1"], gridbox["lon2"]
            lat1, lat2 = gridbox["lat1"], gridbox["lat2"]

            # Find the indices for the lons and lats
            lon1_idx = np.argmin(np.abs(lons - lon1))
            lon2_idx = np.argmin(np.abs(lons - lon2))

            lat1_idx = np.argmin(np.abs(lats - lat1))
            lat2_idx = np.argmin(np.abs(lats - lat2))

            # Add the gridbox to the plot
            ax.plot(
                [lon1, lon2, lon2, lon1, lon1],
                [lat1, lat1, lat2, lat2, lat1],
                color="green",
                linewidth=2,
                transform=proj,
            )

            # Constrain the corr_var_ts array to the gridbox
            corr_var_ts_gridbox = corr_var_ts[
                :, lat1_idx:lat2_idx, lon1_idx:lon2_idx
            ].mean(axis=(1, 2))

            # Print the len of the time series
            print("len(corr_var_ts_gridbox): ", len(corr_var_ts_gridbox))
            print("len(nao): ", len(nao))

            # Calculate the correlation
            corr, pval = pearsonr(nao, corr_var_ts_gridbox)

            # Print the p-value
            print("pval: ", pval)

            # Include the correlation on the plot
            ax.text(
                lon2,
                lat2,
                f"r = {corr:.2f}",
                fontsize=10,
                color="white",
                transform=proj,
                bbox=dict(facecolor="green", alpha=0.5, edgecolor="black"),
            )
    else:
        print("No gridboxes to plot.")

        # Add a title
        ax.set_title(f"Correlation (obs NAO, obs {variable})")

    # Set up the colorbar label
    cbar.set_label("correlation coefficient")

    # Render the plot
    plt.show()

    # Return none
    return None

# PLot corr subplots function
def plot_corr_subplots(
    corr_array_1: np.ndarray,
    pval_array_1: np.ndarray,
    corr_array_2: np.ndarray,
    pval_array_2: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    variable_1: str,
    variable_2: str,
    sig_threshold: float = 0.05,
    plot_gridbox: list = None,
    nao: np.ndarray = None,
    corr_var_ts_1: np.ndarray = None,
    corr_var_ts_2: np.ndarray = None,
    lat_bounds: list = [30, 80],
    lon_bounds: list = [-60, 40],
    figsize_x_px: int = 90,
    figsize_y_px: int = 45,
    save_dpi: int = 600,
    plot_dir: str = "/gws/nopw/j04/canari/users/benhutch/plots",
    fig_labels: list = ["b", "c"],
    fontsize: int = 12,
    w_space: float = 0.1,
):
    """
    Plots the correlation and p-values for the spatial correlation.
    Produces 1 x 2 subplots for the spatial correlations of the NAO.

    Args:
    -----

    corr_array_1: np.ndarray
        The array containing the correlation values for the first variable.

    pval_array_1: np.ndarray
        The array containing the p-values for the first variable.

    corr_array_2: np.ndarray
        The array containing the correlation values for the second variable.

    pval_array_2: np.ndarray
        The array containing the p-values for the second variable.

    lats: np.ndarray
        The array containing the latitudes.

    lons: np.ndarray
        The array containing the longitudes.

    variable_1: str
        The variable to use for the plot title.

    variable_2: str
        The variable to use for the plot title.

    sig_threshold: float
        The significance threshold for the correlation.

    plot_gridbox: list
        List of gridboxes to plot on the plot.

    nao: np.ndarray
        The array containing the NAO index values.

    corr_var_ts_1: np.ndarray
        The array containing the variable to correlate values for the first variable.

    corr_var_ts_2: np.ndarray
        The array containing the variable to correlate values for the second variable.

    lat_bounds: list
        The bounds for the latitude.

    lon_bounds: list
        The bounds for the longitude.

    figsize_x_px: int
        The x dimension for the figure size in px.

    figsize_y_px: int
        The y dimension for the figure size in px.

    save_dpi: int
        The dpi to save the figure at.
        Default is 600.

    plot_dir: str
        The directory to save the plot in.
        Default is /gws/nopw/j04/canari/users/benhutch/plots

    fig_labels: list
        The list of labels for the subplots.

    fontsize: int
        The fontsize for the text on the plot.

    w_space: float
        The width space between the subplots.

    Returns:
    --------

    None
    """

    # Set up the projection
    proj = ccrs.PlateCarree(central_longitude=0)

    # print the dpi
    print("plt.rcParams['figure.dpi']: ", plt.rcParams["figure.dpi"])

    px = 1 / plt.rcParams["figure.dpi"]

    # print the px
    print("px: ", px)
    
    # Calculate the figure size
    print("figsize_x_px * px: ", figsize_x_px * px)
    print("figsize_y_px * px: ", figsize_y_px * px)

    # Print total size
    print("Total size: ", (figsize_x_px * px) * (figsize_y_px * px))

    # # assert that (figsize_x_px * px) * (figsize_y_px * px) <= 1800
    # assert (figsize_x_px * px) * (figsize_y_px * px) >= 1800, "The figure size is too large."

    # Calculate the figure size in inches
    figsize_x_in = figsize_x_px * px

    # Calculate the figure size in inches
    figsize_y_in = figsize_y_px * px

    # Set up the wspace

    # Plot these values
    # Set up a single subplot
    fig, axs = plt.subplots(nrows=1, ncols=2,
                            figsize=(figsize_x_in, figsize_y_in),
                            subplot_kw={"projection": proj})

    # Adjust the space between the subplots
    plt.subplots_adjust(wspace=w_space)
    
    # Focus on the euro-atlantic region
    lat1_grid, lat2_grid = lat_bounds[0], lat_bounds[1]
    lon1_grid, lon2_grid = lon_bounds[0], lon_bounds[1]

    lat1_idx_grid = np.argmin(np.abs(lats - lat1_grid))
    lat2_idx_grid = np.argmin(np.abs(lats - lat2_grid))

    lon1_idx_grid = np.argmin(np.abs(lons - lon1_grid))
    lon2_idx_grid = np.argmin(np.abs(lons - lon2_grid))

    # # Print the indices
    # print("lon1_idx_grid: ", lon1_idx_grid)
    # print("lon2_idx_grid: ", lon2_idx_grid)
    # print("lat1_idx_grid: ", lat1_idx_grid)
    # print("lat2_idx_grid: ", lat2_idx_grid)

    # # # If lat1_idx_grid is greater than lat2_idx_grid, then switch them
    # # if lat1_idx_grid > lat2_idx_grid:
    # #     lat1_idx_grid, lat2_idx_grid = lat2_idx_grid, lat1_idx_grid

    # # Print the indices
    # print("lon1_idx_grid: ", lon1_idx_grid)
    # print("lon2_idx_grid: ", lon2_idx_grid)
    # print("lat1_idx_grid: ", lat1_idx_grid)
    # print("lat2_idx_grid: ", lat2_idx_grid)

    # Constrain the lats and lons to the grid
    lats = lats[lat1_idx_grid:lat2_idx_grid]
    lons = lons[lon1_idx_grid:lon2_idx_grid]

    # Constrain the corr_array to the grid
    corr_array_1 = corr_array_1[lat1_idx_grid:lat2_idx_grid, lon1_idx_grid:lon2_idx_grid]

    # # print the shape of the pval_array_1
    # print("pval_array_1.shape: ", pval_array_1.shape)

    # # Print the values of the pval_array_1
    # print("pval_array_1: ", pval_array_1)

    # Constrain the pval_array to the grid
    pval_array_1 = pval_array_1[lat1_idx_grid:lat2_idx_grid, lon1_idx_grid:lon2_idx_grid]

    # # Print the values of the pval_array_1
    # print("pval_array_1: ", pval_array_1)

    # # print the shape of the pval_array_2
    # print("pval_array_1.shape: ", pval_array_1.shape)

    # Constrain the corr_array to the grid
    corr_array_2 = corr_array_2[lat1_idx_grid:lat2_idx_grid, lon1_idx_grid:lon2_idx_grid]

    # # print the shape of the pval_array_2
    # print("pval_array_2.shape: ", pval_array_2.shape)

    # # Print the values of the pval_array_2
    # print("pval_array_2: ", pval_array_2)

    # Constrain the pval_array to the grid
    pval_array_2 = pval_array_2[lat1_idx_grid:lat2_idx_grid, lon1_idx_grid:lon2_idx_grid]

    # If nao and corr_var_ts_1 and corr_var_ts_2 are not None
    if nao is not None and corr_var_ts_1 is not None and corr_var_ts_2 is not None:
        # Constraint the corr_var_ts array to the grid
        corr_var_ts_1 = corr_var_ts_1[
            :, lat1_idx_grid:lat2_idx_grid, lon1_idx_grid:lon2_idx_grid
        ]

        # Constraint the corr_var_ts array to the grid
        corr_var_ts_2 = corr_var_ts_2[
            :, lat1_idx_grid:lat2_idx_grid, lon1_idx_grid:lon2_idx_grid
        ]

    # Set up the contour levels
    clevs = np.arange(-1.0, 1.1, 0.1)

    # Include coastlines
    axs[0].coastlines()
    axs[1].coastlines()

    # Include the gridlines as dashed lines
    gl0 = axs[0].gridlines(linestyle="--", alpha=0.5, draw_labels=True)
    gl1 = axs[1].gridlines(linestyle="--", alpha=0.5, draw_labels=True)

    # Set the labels for the gridlines
    gl0.top_labels = False
    gl0.right_labels = False
    gl1.top_labels = False
    gl1.left_labels = False
    gl1.right_labels = False

    # plot the first contour plot on the first subplot
    cf1 = axs[0].contourf(lons, lats, corr_array_1, clevs, transform=proj, cmap="RdBu_r")

    # plot the second contour plot on the second subplot
    cf2 = axs[1].contourf(lons, lats, corr_array_2, clevs, transform=proj, cmap="RdBu_r")

    # if any of the p values are greater or less than the significance threshold
    # Set where the p-values are greater or less than
    # the significance threshold to nan
    pval_array_1[(pval_array_1 > sig_threshold) & (pval_array_1 < 1 - sig_threshold)] = np.nan

    # Assert that not all of the values are nan
    assert not np.all(np.isnan(pval_array_1)), "All values in the pval_array_1 are nan."

    # assert that not all of the values are nan
    assert not np.all(np.isnan(pval_array_1)), "All values in the pval_array_1 are nan."

    # print the pval_array_2
    print("pval_array_2: ", pval_array_2)

    # same for the other pval_array
    pval_array_2[(pval_array_2 > sig_threshold) & (pval_array_2 < 1 - sig_threshold)] = np.nan

    # assert that not all of the values are nan
    assert not np.all(np.isnan(pval_array_2)), "All values in the pval_array_2 are nan."

    # assert that not all of the values are nan
    assert not np.all(np.isnan(pval_array_2)), "All values in the pval_array_2 are nan."

    # How can I invert the p_val_arrays here?
    # so that where the values are NaN, these are replaced with ones
    # and where the values are not NaN, these are replaced with NaNs
    # pval_array_1 = np.where(np.isnan(pval_array_1), 1, np.nan)
    # pval_array_2 = np.where(np.isnan(pval_array_2), 1, np.nan)

    # Plot the p-values
    axs[0].contourf(lons, lats, pval_array_1, hatches=["...."], alpha=0.0, transform=proj)

    # Plot the p-values
    axs[1].contourf(lons, lats, pval_array_2, hatches=["...."], alpha=0.0, transform=proj)

    # Set up the colorbar
    # To be used for both subplots
    cbar = plt.colorbar(cf1, ax=axs, orientation="horizontal", pad=0.05, shrink=0.6)

    # If the plot_gridbox is not None
    if plot_gridbox is not None:
        # Assert that it is a list
        assert isinstance(
            plot_gridbox, list
        ), "The plot_gridbox must be a list of gridboxes."
        # Loop over the gridboxes
        for i, gridbox in enumerate(plot_gridbox):
            # Extract the lons and lats
            lon1, lon2 = gridbox["lon1"], gridbox["lon2"]
            lat1, lat2 = gridbox["lat1"], gridbox["lat2"]

            # Find the indices for the lons and lats
            lon1_idx = np.argmin(np.abs(lons - lon1))
            lon2_idx = np.argmin(np.abs(lons - lon2))

            lat1_idx = np.argmin(np.abs(lats - lat1))
            lat2_idx = np.argmin(np.abs(lats - lat2))

            # Add the gridbox to the plot
            axs[0].plot(
                [lon1, lon2, lon2, lon1, lon1],
                [lat1, lat1, lat2, lat2, lat1],
                color="green",
                linewidth=2,
                transform=proj,
            )

            # Add the gridbox to the secon plot
            axs[1].plot(
                [lon1, lon2, lon2, lon1, lon1],
                [lat1, lat1, lat2, lat2, lat1],
                color="green",
                linewidth=2,
                transform=proj,
            )

            # Constrain the corr_var_ts array to the gridbox
            corr_var_ts_gridbox_1 = corr_var_ts_1[
                :, lat1_idx:lat2_idx, lon1_idx:lon2_idx
            ].mean(axis=(1, 2))

            # Constrain the corr_var_ts array to the gridbox
            corr_var_ts_gridbox_2 = corr_var_ts_2[
                :, lat1_idx:lat2_idx, lon1_idx:lon2_idx
            ].mean(axis=(1, 2))

            # Calculate the correlation
            corr_1, pval_1 = pearsonr(nao, corr_var_ts_gridbox_1)

            # Calculate the correlation
            corr_2, pval_2 = pearsonr(nao, corr_var_ts_gridbox_2)

            # Include the correlation on the plot
            axs[0].text(
                0.05,
                0.05,
                (
                    f"ACC = {corr_1:.2f} "
                    f"(P = {pval_1:.2f})"
                ),
                transform=axs[0].transAxes,
                fontsize=fontsize,
                verticalalignment="bottom",
                horizontalalignment="left",
                bbox=dict(facecolor="white", alpha=0.5),
            )

            # Include the figure label
            axs[0].text(
                0.95,
                0.05,
                fig_labels[0],
                transform=axs[0].transAxes,
                fontsize=fontsize,
                verticalalignment="bottom",
                horizontalalignment="right",
                bbox=dict(facecolor="white", alpha=0.5),
            )

            # Include the correlation on the plot
            axs[1].text(
                0.05,
                0.05,
                (
                    f"ACC = {corr_2:.2f} "
                    f"(P = {pval_2:.2f})"
                ),
                transform=axs[1].transAxes,
                fontsize=fontsize,
                verticalalignment="bottom",
                horizontalalignment="left",
                bbox=dict(facecolor="white", alpha=0.5),
            )

            # Include the variable name in the top left of the plot
            axs[0].text(
                0.05,
                0.95,
                variable_1,
                transform=axs[0].transAxes,
                fontsize=fontsize,
                verticalalignment="top",
                horizontalalignment="left",
                bbox=dict(facecolor="white", alpha=0.5),
            )

            # Include the variable name in the top left of the plot
            axs[1].text(
                0.05,
                0.95,
                variable_2,
                transform=axs[1].transAxes,
                fontsize=fontsize,
                verticalalignment="top",
                horizontalalignment="left",
                bbox=dict(facecolor="white", alpha=0.5),
            )

            # Include the figure label
            axs[1].text(
                0.95,
                0.05,
                fig_labels[1],
                transform=axs[1].transAxes,
                fontsize=fontsize,
                verticalalignment="bottom",
                horizontalalignment="right",
                bbox=dict(facecolor="white", alpha=0.5),
            )
    else:
        print("No gridboxes to plot.")

    # Set up the colorbar label
    cbar.set_label("Pearson correlation with ONDJFM NAO")

    # # Specify a tight layout
    # plt.tight_layout()

    # Set up the current time
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set up the fname
    fname = f"{variable_1}_{variable_2}_nao_corr_plot_{current_time}.pdf"

    # Save the plot
    plt.savefig(os.path.join(plot_dir, fname), dpi=save_dpi)

    # Render the plot
    plt.show()

    # Return none
    return None

# Define a function to process the data for plotting scatter plots
def process_data_for_scatter(
    season: str,
    forecast_range: str,
    start_year: int,
    end_year: int,
    predictor_var: str,
    predictor_var_dict: dict,
    predictand_var: str,
    predictand_var_file: str,
    region: dict,
    region_name: str,
    quantiles: list = [0.75, 0.95],
):
    """
    Function which processes the data for the scatter plots.

    Args:
    -----

    season: str
        The season to calculate the correlation for.
        E.g. ONDJFM, DJFM, DJF

    forecast_range: str
        The forecast range to calculate the correlation for.
        E.g. 2-5, 2-9

    start_year: int
        The start year to calculate the correlation for.
        E.g. 1960

    end_year: int
        The end year to calculate the correlation for.
        E.g. 2014

    predictor_var: str
        The variable to use as the predictor.
        E.g. "si10"

    predictor_var_dict: dict
        The dictionary containing the grid information for the predictor variable.
        E.g. {
            "lag": 0,
            "alt_lag": False,
            "region": "global",
        }

    predictand_var: str
        The variable to use as the predictand.
        E.g. "pr"

    predictand_var_file: str
        The file containing the predictand variable.
        Could be the observed or model data.

    region: dict
        The dictionary containing the region information.
        E.g. {"lon1": 332, "lon2": 340, "lat1": 40, "lat2": 36}
        Could also be a shapefile.

    region_name: str
        The name of the region to use for the scatter plot.
        E.g. "europe"

    quantiles: list
        The quantiles to calculate for the scatter plot.
        E.g. [0.75, 0.95]

    Returns:
    --------

    scatter_dict: dict
        The dictionary containing the scatter plot data.
    """

    # Set up the mdi
    mdi = -9999.0

    # Set up the scatter dictionary
    scatter_dict = {
        "predictor_var": predictor_var,
        "predictand_var": predictand_var,
        "season": season,
        "forecast_range": forecast_range,
        "start_year": start_year,
        "end_year": end_year,
        "quantiles": quantiles,
        "region": region,
        "predictor_var_ts": [],
        "predictand_var_ts": [],
        "predictor_var_mean": mdi,
        "predictand_var_mean": mdi,
        "rval": mdi,
        "pval": mdi,
        "slope": mdi,
        "intercept": mdi,
        "std_err": mdi,
        f"first_quantile_{quantiles[0]}": mdi,
        f"second_quantile_{quantiles[1]}": mdi,
        "init_years": [],
        "valid_years": [],
        "nens": mdi,
        "ts_corr": mdi,
        "ts_pval": mdi,
        "ts_rpc": mdi,
        "ts_rps": mdi,
        "lag": mdi,
        "gridbox": region,
        "gridbox_name": region_name,
        "method": mdi,
    }

    # Set up the init years
    scatter_dict["init_years"] = np.arange(start_year, end_year + 1)

    # Assert that the season is a winter season
    assert season in ["DJF", "ONDJFM", "DJFM"], "The season must be a winter season."

    # # Assert that the file exists for the predictor variable
    # assert os.path.exists(predictor_var_file), "The file for the predictor variable does not exist."

    # # Assert that the file exists for the predictand variable
    # assert os.path.exists(predictand_var_file), "The file for the predictand variable does not exist."

    # Assert that predictor_var_dict is a dictionary
    assert isinstance(
        predictor_var_dict, dict
    ), "The predictor_var_dict must be a dictionary."

    # Assert that predictor_var_dict contains keys for lag, alt_lag, and region
    assert (
        "lag" in predictor_var_dict.keys()
    ), "The predictor_var_dict must contain a key for lag."

    # Assert that predictor_var_dict contains keys for lag, alt_lag, and region
    assert (
        "alt_lag" in predictor_var_dict.keys()
    ), "The predictor_var_dict must contain a key for alt_lag."

    # Assert that predictor_var_dict contains keys for lag, alt_lag, and region
    assert (
        "region" in predictor_var_dict.keys()
    ), "The predictor_var_dict must contain a key for region."

    # If the region is a dictionary
    if isinstance(region, dict):
        print("The region is a dictionary.")
        print("Extracting the lats and lons from the region dictionary.")
        # Extract the lats and lons from the region dictionary
        lon1, lon2 = region["lon1"], region["lon2"]
        lat1, lat2 = region["lat1"], region["lat2"]
    else:
        print("The region is not a dictionary.")
        AssertionError("The region must be a dictionary. Not a shapefile.")

    # If the predictor var is nao
    if predictor_var == "nao":
        print("The predictor variable is the NAO index.")
        print("Extracting the NAO index from the predictor variable file.")

        # Load the psl data for processing the NAO stats
        psl_data = nal_funcs.load_data(
            season=season,
            forecast_range=forecast_range,
            start_year=start_year,
            end_year=end_year,
            lag=predictor_var_dict["lag"],
            method="alt_lag",
            region=predictor_var_dict["region"],
            variable="psl",
        )

        # Use the function to calculate the NAO stats
        nao_stats = nal_funcs.calc_nao_stats(
            data=psl_data,
            season=season,
            forecast_range=forecast_range,
            start_year=start_year,
            end_year=end_year,
            lag=predictor_var_dict["lag"],
            alt_lag=True,
        )

        # Extract the data for the predictor variable
        predictor_var_data = nal_funcs.load_data(
            season=season,
            forecast_range=forecast_range,
            start_year=start_year,
            end_year=end_year,
            lag=predictor_var_dict["lag"],
            method=predictor_var_dict["method"],
            region=predictor_var_dict["region"],
            variable=predictand_var,
        )

        # Load the data for the predictor variable
        rm_dict = p1p_funcs.load_ts_data(
            data=predictor_var_data,
            season=season,
            forecast_range=forecast_range,
            start_year=start_year,
            end_year=end_year,
            lag=predictor_var_dict["lag"],
            gridbox=region,
            gridbox_name=region_name,
            variable=predictand_var,
            alt_lag=predictor_var_dict["alt_lag"],
            region=predictor_var_dict["region"],
        )

        # Append to the scatter dictionary
        scatter_dict["predictor_var_ts"] = nao_stats["model_nao_mean"]
        scatter_dict["predictand_var_ts"] = rm_dict["obs_ts"]

        # append the init years
        scatter_dict["init_years"] = rm_dict["init_years"]

        # append the valid years
        scatter_dict["valid_years"] = rm_dict["valid_years"]

        # append the nens
        scatter_dict["nens"] = nao_stats["nens"]

        # append the ts_corr
        scatter_dict["ts_corr"] = nao_stats["corr1"]

        # append the ts_pval
        scatter_dict["ts_pval"] = nao_stats["p1"]

        # append the ts_rpc
        scatter_dict["ts_rpc"] = nao_stats["rpc1"]

        # append the ts_rps
        scatter_dict["ts_rps"] = nao_stats["rps1"]

        # append the lag
        scatter_dict["lag"] = rm_dict["lag"]

        # append the gridbox
        scatter_dict["gridbox"] = rm_dict["gridbox"]

        # Append the method
        scatter_dict["method"] = "alt_lag"

        # If the predictand variable is 'pr'
        if predictand_var == "pr":
            # Convert obs to mm day-1
            scatter_dict["predictand_var_ts"] = scatter_dict["predictand_var_ts"] * 1000

        # Divide the predictor variable by 100
        scatter_dict["predictor_var_ts"] = scatter_dict["predictor_var_ts"] / 100

        # # Standardize predictor_var_ts
        # scatter_dict["predictor_var_ts"] = (
        #     scatter_dict["predictor_var_ts"] - np.mean(scatter_dict["predictor_var_ts"])
        # ) / np.std(scatter_dict["predictor_var_ts"])

        # # Standardize predictand_var_ts
        # scatter_dict["predictand_var_ts"] = (
        #     scatter_dict["predictand_var_ts"]
        #     - np.mean(scatter_dict["predictand_var_ts"])
        # ) / np.std(scatter_dict["predictand_var_ts"])

        # Perform a linear regression
        # and calculate the quantiles
        slope, intercept, r_value, p_value, std_err = linregress(
            scatter_dict["predictor_var_ts"], scatter_dict["predictand_var_ts"]
        )

        # Store the linear regression values in the dictionary
        scatter_dict["rval"] = r_value
        scatter_dict["pval"] = p_value
        scatter_dict["slope"] = slope
        scatter_dict["intercept"] = intercept
        scatter_dict["std_err"] = std_err

        # Define a lamda function for the quantiles
        tinv = lambda p, df: abs(t.ppf(p / 2, df))

        # Calculate the degrees of freedom
        df = len(scatter_dict["predictor_var_ts"]) - 2

        # Calculate the first quantile
        q1 = tinv(1 - quantiles[0], df) * scatter_dict["std_err"]

        # Calculate the second quantile
        q2 = tinv(1 - quantiles[1], df) * scatter_dict["std_err"]

        # Store the quantiles in the dictionary
        scatter_dict[f"first_quantile_{quantiles[0]}"] = q1

        # Store the quantiles in the dictionary
        scatter_dict[f"second_quantile_{quantiles[1]}"] = q2

        # Calculate the mean of the predictor variable
        scatter_dict["predictor_var_mean"] = np.mean(scatter_dict["predictor_var_ts"])

        # Calculate the mean of the predictand variable
        scatter_dict["predictand_var_mean"] = np.mean(scatter_dict["predictand_var_ts"])
    else:
        print("The predictor variable is not the NAO index.")
        print("Extracting the predictor variable from the predictor variable file.")

        # Extract the data for the predictor variable
        predictor_var_data = nal_funcs.load_data(
            season=season,
            forecast_range=forecast_range,
            start_year=start_year,
            end_year=end_year,
            lag=predictor_var_dict["lag"],
            method=predictor_var_dict["method"],
            region=predictor_var_dict["region"],
            variable=predictor_var,
        )

        # Load the data for the predictor variable
        rm_dict = p1p_funcs.load_ts_data(
            data=predictor_var_data,
            season=season,
            forecast_range=forecast_range,
            start_year=start_year,
            end_year=end_year,
            lag=predictor_var_dict["lag"],
            gridbox=region,
            gridbox_name=region_name,
            variable=predictor_var,
            alt_lag=predictor_var_dict["alt_lag"],
            region=predictor_var_dict["region"],
        )

        # Append to the scatter dictionary
        scatter_dict["predictor_var_ts"] = rm_dict["fcst_ts_mean"]
        scatter_dict["predictand_var_ts"] = rm_dict["obs_ts"]

        # append the init years
        scatter_dict["init_years"] = rm_dict["init_years"]

        # append the valid years
        scatter_dict["valid_years"] = rm_dict["valid_years"]

        # append the nens
        scatter_dict["nens"] = rm_dict["nens"]

        # append the ts_corr
        scatter_dict["ts_corr"] = rm_dict["corr"]

        # append the ts_pval
        scatter_dict["ts_pval"] = rm_dict["p"]

        # append the ts_rpc
        scatter_dict["ts_rpc"] = rm_dict["rpc"]

        # append the ts_rps
        scatter_dict["ts_rps"] = rm_dict["rps"]

        # append the lag
        scatter_dict["lag"] = rm_dict["lag"]

        # append the gridbox
        scatter_dict["gridbox"] = rm_dict["gridbox"]

        # Append the method
        scatter_dict["method"] = rm_dict["alt_lag"]

        # if the predictor variable is 'pr'
        if predictor_var == "pr":
            # ERA5 is in units of m day-1
            # Model is in units of kg m-2 s-1
            # Convert the model units to m day-1
            scatter_dict["predictor_var_ts"] = scatter_dict["predictor_var_ts"] * 86400

            # ERA5 is in units of m day-1
            # Convert to mm day-1
            scatter_dict["predictand_var_ts"] = scatter_dict["predictand_var_ts"] * 1000

        # Ussing the time series
        # perform a linear regression
        # and calculate the quantiles
        # for the scatter plot
        # Calculate the linear regression
        slope, intercept, r_value, p_value, std_err = linregress(
            scatter_dict["predictor_var_ts"], scatter_dict["predictand_var_ts"]
        )

        # Store the linear regression values in the dictionary
        scatter_dict["rval"] = r_value
        scatter_dict["pval"] = p_value
        scatter_dict["slope"] = slope
        scatter_dict["intercept"] = intercept
        scatter_dict["std_err"] = std_err

        # Define a lamda function for the quantiles
        tinv = lambda p, df: abs(t.ppf(p / 2, df))

        # Calculate the degrees of freedom
        df = len(scatter_dict["predictor_var_ts"]) - 2

        # Calculate the first quantile
        q1 = tinv(1 - quantiles[0], df) * scatter_dict["std_err"]

        # Calculate the second quantile
        q2 = tinv(1 - quantiles[1], df) * scatter_dict["std_err"]

        # Store the quantiles in the dictionary
        scatter_dict[f"first_quantile_{quantiles[0]}"] = q1

        # Store the quantiles in the dictionary
        scatter_dict[f"second_quantile_{quantiles[1]}"] = q2

        # Calculate the mean of the predictor variable
        scatter_dict["predictor_var_mean"] = np.mean(scatter_dict["predictor_var_ts"])

        # Calculate the mean of the predictand variable
        scatter_dict["predictand_var_mean"] = np.mean(scatter_dict["predictand_var_ts"])

    # Return the dictionary
    return scatter_dict


# Define a function to plot the scatter plot
def plot_scatter(
    scatter_dict: dict,
):
    """
    Function which plots the scatter plot.

    Args:
    -----

    scatter_dict: dict
        The dictionary containing the scatter plot data.

    Returns:
    --------

    None
    """

    # Set up the mdi
    mdi = -9999.0

    # Set up the figure
    fig = plt.figure(figsize=(8, 8))

    # Set up the axis
    ax = fig.add_subplot(111)

    # Find the minimum and maximum of the current x values
    x_min = np.min(scatter_dict["predictor_var_ts"])
    x_max = np.max(scatter_dict["predictor_var_ts"])

    # Extend the range by a certain amount (e.g., 10% of the range)
    x_range = x_max - x_min
    x_min -= 0.1 * x_range
    x_max += 0.1 * x_range

    # Create a new array of x values using this extended range
    x_values_extended = np.linspace(x_min, x_max, 100)

    # print the stanard error from the scatter dictionary
    print("Standard error: ", scatter_dict["std_err"])

    # Plot the regression line using the new array of x values
    ax.plot(
        x_values_extended,
        scatter_dict["slope"] * x_values_extended + scatter_dict["intercept"],
        "k",
    )

    # Print the first quantile
    print(
        "First quantile: ",
        scatter_dict[f"first_quantile_{scatter_dict['quantiles'][0]}"],
    )

    # Print the second quantile
    print(
        "Second quantile: ",
        scatter_dict[f"second_quantile_{scatter_dict['quantiles'][1]}"],
    )

    # Fill between the 95th quantile
    ax.fill_between(
        x_values_extended,
        scatter_dict["slope"] * x_values_extended
        + scatter_dict["intercept"]
        - scatter_dict[f"second_quantile_{scatter_dict['quantiles'][1]}"],
        scatter_dict["slope"] * x_values_extended
        + scatter_dict["intercept"]
        + scatter_dict[f"second_quantile_{scatter_dict['quantiles'][1]}"],
        color="0.8",
        alpha=0.5,
    )

    # Fill between the 75th quantile
    ax.fill_between(
        x_values_extended,
        scatter_dict["slope"] * x_values_extended
        + scatter_dict["intercept"]
        - scatter_dict[f"first_quantile_{scatter_dict['quantiles'][0]}"],
        scatter_dict["slope"] * x_values_extended
        + scatter_dict["intercept"]
        + scatter_dict[f"first_quantile_{scatter_dict['quantiles'][0]}"],
        color="0.6",
        alpha=0.5,
    )

    # Plot the scatter plot
    ax.scatter(scatter_dict["predictor_var_ts"], scatter_dict["predictand_var_ts"])

    # Plot the mean of the predictor variable as a vertical line
    ax.axvline(
        scatter_dict["predictor_var_mean"],
        color="k",
        linestyle="--",
        alpha=0.5,
        linewidth=0.5,
    )

    # Plot the mean of the predictand variable as a horizontal line
    ax.axhline(
        scatter_dict["predictand_var_mean"],
        color="k",
        linestyle="--",
        alpha=0.5,
        linewidth=0.5,
    )

    # Plot the r value in a text box in the top left corner
    ax.text(
        0.05,
        0.95,
        f"r = {scatter_dict['rval']:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=dict(facecolor="white", alpha=0.5),
        fontsize=12,
    )

    # Set up the x-axis label
    ax.set_xlabel(f"Hindcast {scatter_dict['predictor_var']} anomalies (hPa)")

    # Set up the y-axis label
    ax.set_ylabel(f"Observed {scatter_dict['predictand_var']} anomalies (m/s)")

    # Set up the title
    ax.set_title(
        f"Scatter plot for {scatter_dict['season']} {scatter_dict['forecast_range']} {scatter_dict['start_year']} - {scatter_dict['end_year']} {scatter_dict['gridbox_name']} gridbox"
    )

    # limit the x-axis
    ax.set_xlim(x_min, x_max)

    # Include axis ticks on the inside and outside
    ax.tick_params(axis="both", direction="in", top=True, right=True)

    # Set up the legend
    ax.legend()

    # # Set up the grid
    # ax.grid()

    # Show the plot
    plt.show()

    # Return none
    return None


# Write a function to correlate the NAO with wind power CFs/demand/irradiance
# Within a specific region as defined by the UREAD file being extracted
def correlate_nao_uread(
    filename: str,
    shp_file: str = None,
    shp_file_dir: str = None,
    forecast_range: str = "2-9",
    months: list = [10, 11, 12, 1, 2, 3],
    annual_offset: int = 3,
    start_date: str = "1950-01-01",
    time_unit: str = "h",
    centre: bool = True,
    directory: str = dicts.clearheads_dir,
    obs_var: str = "msl",
    obs_var_data_path: str = dicts.regrid_file,
    start_year: str = "1960",
    end_year: str = "2023",
    nao_n_grid: dict = dicts.iceland_grid_corrected,
    nao_s_grid: dict = dicts.azores_grid_corrected,
    avg_grid: dict = None,
    use_model_data: bool = False,
    model_config: dict = None,
    df_dir: str = "/gws/nopw/j04/canari/users/benhutch/nao_stats_df/",
    model_arr_dir: str = "/gws/nopw/j04/canari/users/benhutch/alternate-lag-processed-data/",
    level: int = 0,
) -> pd.DataFrame:
    """
    Function which correlates the observed NAO (from ERA5) with demand,
    wind power CFs, irradiance, or other variables, from the UREAD datasets and
    returns the correlation values.

    Args:

    filename: str
        The filename to use for extracting the UREAD data.

    shp_file: str
        The shapefile to use for extracting the UREAD data.

    shp_file_dir: str
        The directory to use for extracting the UREAD data.

    forecast_range: str
        The forecast range to use for extracting the UREAD data.
        Default is "2-9", 8-year running mean.

    months: list
        The months to use for extracting the UREAD data.
        Default is the winter months, October to March.

    annual_offset: int
        The annual offset to use for extracting the UREAD data.
        Default is 3, for the winter months.

    start_date: str
        The start date to use for extracting the UREAD data.
        Default is "1950-01-01".

    time_unit: str
        The time unit to use for extracting the UREAD data.
        Default is "h".

    centre: bool
        Whether to use the centre of the window for the rolling average.

    directory: str
        The directory to use for extracting the UREAD data.

    obs_var: str
        The observed variable to use for calculating the NAO index.
        Default is "msl".

    obs_var_data_path: str
        The path to the observed variable data.

    start_year: int
        The start year to use for extracting the UREAD data.

    end_year: int
        The end year to use for extracting the UREAD data.

    nao_n_grid: dict
        The dictionary containing the grid information for the northern NAO grid.

    nao_s_grid: dict
        The dictionary containing the grid information for the southern NAO grid.

    avg_grid: dict
        The dictionary containing the grid information for the average grid.

    use_model_data
        Whether to use model data or not.

    model_config
        The set up of the model used

    df_dir
        The directory in which the dataframes are stored for the model data

    model_arr_dir
        The directory in which the arrays containing the
        processed model data are stored

    level
        The level to use for the model data

    Returns:

    df: pd.DataFrame
        The dataframe containing the correlation values.
    """

    # Find the files
    files = glob.glob(f"{directory}/{filename}")

    # If there are no files, raise an error
    if len(files) == 0:
        raise FileNotFoundError("No files found.")

    # If there are multiple files, raise an error
    if len(files) > 1:
        raise ValueError("Multiple files found.")

    # Load the data
    data = xr.open_dataset(files[0])

    # Assert that NUTS_keys can be extracted from the data
    assert "NUTS_keys" in data.variables, "NUTS_keys not found in the data."

    # Extract the NUTS keys
    NUTS_keys = data["NUTS_keys"].values

    # Print the nuts keys
    print("NUTS_keys for UREAD data: ", NUTS_keys)

    # Turn this data into a dataframe
    df = data.to_dataframe()

    # # Print the head of the dataframe
    # print("Head of the dataframe: ", df.head())

    # Pivot the dataframe
    df = df.reset_index().pivot(
        index="time_in_hours_from_first_jan_1950",
        columns="NUTS",
        values="timeseries_data",
    )

    # # Print the head of the dataframe again
    # print("Head of the dataframe: ", df.head())

    # Add the nuts keys to the columns
    df.columns = NUTS_keys

    # Convert 'time_in_hours_from_first_jan_1950' column to datetime
    df.index = pd.to_datetime(df.index, unit=time_unit, origin=start_date)

    # Collapse the dataframes into monthly averages
    df = df.resample("M").mean()

    # Select only the months of interest
    df = df[df.index.month.isin(months)]

    # Shift the data by the annual offset
    df.index = df.index - pd.DateOffset(months=annual_offset)

    # TODO: Fix hard coded here
    # Throw away the first 3 months of data and last 3 months of data
    df = df.iloc[3:-3]

    # Calculate the annual average
    df = df.resample("A").mean()

    # Calculate the rolling window
    ff_year = int(forecast_range.split("-")[1])
    lf_year = int(forecast_range.split("-")[0])

    # Calculate the rolling window
    rolling_window = (ff_year - lf_year) + 1  # e.g. (9-2) + 1 = 8

    # # Print the first 10 rows of the dataframe
    # print("First 10 rows of the dataframe: ", df.head(10))

    # Take the rolling average
    df = df.rolling(window=rolling_window, center=centre).mean()

    # Throw away the NaN values
    df = df.dropna()

    # load in the ERA5 data
    clim_var = xr.open_mfdataset(
        obs_var_data_path,
        combine="by_coords",
        parallel=True,
        chunks={"time": "auto", "latitude": "auto", "longitude": "auto"},
    )[
        obs_var
    ]  # for mean sea level pressure

    # If expver is a variable in the dataset
    if "expver" in clim_var.coords:
        # Combine the first two expver variables
        clim_var = clim_var.sel(expver=1).combine_first(clim_var.sel(expver=5))

    # if level is not 0
    if level != 0:
        # Extract the data for the level
        clim_var = clim_var.sel(plev=level)

    # Constrain obs to ONDJFM
    clim_var = clim_var.sel(time=clim_var.time.dt.month.isin(months))

    # Shift the time index back by 3 months
    clim_var_shifted = clim_var.shift(time=-annual_offset)

    # Take annual means
    clim_var_annual = clim_var_shifted.resample(time="Y").mean()

    # Throw away years 1959
    clim_var_annual = clim_var_annual.sel(time=slice(start_year, None))

    # Remove the climatology
    clim_var_anomaly = clim_var_annual - clim_var_annual.mean(dim="time")

    # If the obs var is "msl"
    if obs_var == "msl" and use_model_data is False:
        # Print that we are using msl and calculating the NAO index
        print("Using mean sea level pressure to calculate the NAO index.")

        # Extract the lat and lons of iceland
        lat1_n, lat2_n = nao_n_grid["lat1"], nao_n_grid["lat2"]
        lon1_n, lon2_n = nao_n_grid["lon1"], nao_n_grid["lon2"]

        # Extract the lat and lons of the azores
        lat1_s, lat2_s = nao_s_grid["lat1"], nao_s_grid["lat2"]
        lon1_s, lon2_s = nao_s_grid["lon1"], nao_s_grid["lon2"]

        # Calculate the msl mean for the icealndic region
        msl_mean_n = clim_var_anomaly.sel(
            lat=slice(lat1_n, lat2_n), lon=slice(lon1_n, lon2_n)
        ).mean(dim=["lat", "lon"])

        # Calculate the msl mean for the azores region
        msl_mean_s = clim_var_anomaly.sel(
            lat=slice(lat1_s, lat2_s), lon=slice(lon1_s, lon2_s)
        ).mean(dim=["lat", "lon"])

        # Calculate the NAO index (azores - iceland)
        nao_index = msl_mean_s - msl_mean_n

        # Extract the time values
        time_values = nao_index.time.values

        # Extract the values
        nao_values = nao_index.values

        # Create a dataframe for the NAO data
        nao_df = pd.DataFrame({"time": time_values, "NAO anomaly (Pa)": nao_values})

        # Take a central rolling average
        nao_df = (
            nao_df.set_index("time")
            .rolling(window=rolling_window, center=centre)
            .mean()
        )

        # Drop the NaN values
        nao_df = nao_df.dropna()

        # Print the head of nao_df
        print("NAO df head: ", nao_df.head())

        # Print the head of df
        print("df head: ", df.head())

        # Merge the dataframes, using the index of the first
        merged_df = df.join(nao_df, how="inner")

        # Drop the NaN values
        merged_df = merged_df.dropna()

        # Create a new dataframe for the correlations
        corr_df = pd.DataFrame(columns=["region", "correlation", "p-value"])

        # Loop over the columns
        for col in merged_df.columns[:-1]:
            # Calculate the correlation
            corr, pval = pearsonr(merged_df[col], merged_df["NAO anomaly (Pa)"])

            # Append to the dataframe
            corr_df_to_append = pd.DataFrame(
                {"region": [col], "correlation": [corr], "p-value": [pval]}
            )

            # Append to the dataframe
            corr_df = pd.concat([corr_df, corr_df_to_append], ignore_index=True)
    elif obs_var == "msl" and use_model_data is True:
        print("Extracting the stored NAO data from the model data.")

        # set up the file name using model config
        model_filename = f"""{model_config["variable"]}_{model_config["season"]}_{model_config["region"]}_{model_config["start_year"]}_{model_config["end_year"]}_{model_config["forecast_range"]}_{model_config['lag']}_{model_config['nao']}.csv"""

        # Set up the path to the file
        filepath = f"{df_dir}{model_filename}"

        # Print the filepath
        print("Filepath: ", filepath)

        # assert that the file exists
        assert os.path.exists(filepath), "The file does not exist."

        # print the filepath
        print("Filepath: ", filepath)

        # Load the dataframe
        df_model_nao = pd.read_csv(filepath)

        # process the observations
        # Extract the lat and lons of iceland
        lat1_n, lat2_n = nao_n_grid["lat1"], nao_n_grid["lat2"]
        lon1_n, lon2_n = nao_n_grid["lon1"], nao_n_grid["lon2"]

        # Extract the lat and lons of the azores
        lat1_s, lat2_s = nao_s_grid["lat1"], nao_s_grid["lat2"]
        lon1_s, lon2_s = nao_s_grid["lon1"], nao_s_grid["lon2"]

        # Calculate the msl mean for the icealndic region
        msl_mean_n = clim_var_anomaly.sel(
            lat=slice(lat1_n, lat2_n), lon=slice(lon1_n, lon2_n)
        ).mean(dim=["lat", "lon"])

        # Calculate the msl mean for the azores region
        msl_mean_s = clim_var_anomaly.sel(
            lat=slice(lat1_s, lat2_s), lon=slice(lon1_s, lon2_s)
        ).mean(dim=["lat", "lon"])

        # Calculate the NAO index (azores - iceland)
        nao_index = msl_mean_s - msl_mean_n

        # Extract the time values
        time_values = nao_index.time.values

        # Extract the values
        nao_values = nao_index.values

        # Create a dataframe for the NAO data
        nao_df = pd.DataFrame({"time": time_values, "NAO anomaly (Pa)": nao_values})

        # Take a central rolling average
        nao_df = (
            nao_df.set_index("time")
            .rolling(window=rolling_window, center=centre)
            .mean()
        )

        # Drop the NaN values
        nao_df = nao_df.dropna()

        # Set the year as the index
        nao_df.index = nao_df.index.year

        # Set the index for the loaded data as valid_time
        df_model_nao = df_model_nao.set_index("valid_time")

        # join the two dataframes
        merged_df = df_model_nao.join(nao_df)

        # Set the volumn with the name value to obs_nao_pd
        merged_df = merged_df.rename(columns={"value": "obs_nao_pd"})

        # Print the head of this df
        print("Head of merged_df: ", merged_df.head())

        # For rthe uread dataset, set the index to years
        df.index = df.index.year

        # print the head of the UREAD df
        print("Head of UREAD df: ", df.head())

        # merge with the CF data
        merged_df_full = df.join(merged_df, how="inner")

        # print the head of the merged
        print(merged_df_full.head())

        # Create a new dataframe for the correlations
        corr_df = pd.DataFrame(columns=["region", "correlation", "p-value"])

        # Loop over the columns
        for col in merged_df_full.columns[:-6]:
            # Calculate the correlation
            corr, pval = pearsonr(merged_df_full[col], merged_df_full["model_nao_mean"])

            # Append to the dataframe
            corr_df_to_append = pd.DataFrame(
                {"region": [col], "correlation": [corr], "p-value": [pval]}
            )

            # Append to the dataframe
            corr_df = pd.concat([corr_df, corr_df_to_append], ignore_index=True)

        return df, merged_df, merged_df_full, corr_df

    else:
        print("The observed variable is not mean sea level pressure.")
        print("calculating correlation skill for gridpoint variable")

        # If the filename contains the string "eez"
        if shp_file is not None and "eez" in shp_file and use_model_data is False:
            print("Averaging data for EEZ domains")

            # Assert that shp_file is not None
            assert shp_file is not None, "The shapefile is None."

            # Assert that shp_file_dir is not None
            assert shp_file_dir is not None, "The shapefile directory is None."

            # Assert that the shp_file_dir exists
            assert os.path.exists(
                shp_file_dir
            ), "The shapefile directory does not exist."

            # Load the shapefile
            shapefile = gpd.read_file(os.path.join(shp_file_dir, shp_file))

            # Assert that the shp_file exists
            assert os.path.exists(
                os.path.join(shp_file_dir, shp_file)
            ), "The shapefile does not exist."

            # Throw away all columns
            # Apart from "GEONAME", "ISO_SOV1", and "geometry"
            shapefile = shapefile[["GEONAME", "ISO_SOV1", "geometry"]]

            # Pass the NUTS keys through the filter
            iso_sov_values = [dicts.iso_mapping[key] for key in NUTS_keys]

            # Constrain the geo dataframe to only include these values
            shapefile = shapefile[shapefile["ISO_SOV1"].isin(iso_sov_values)]

            # Filter df to only include the rows where GEONAME includes: "Exclusive Economic Zone"
            shapefile = shapefile[
                shapefile["GEONAME"].str.contains("Exclusive Economic Zone")
            ]

            # Remove any rows from EEZ shapefile which contain "(*)" in the GEONAME column
            # To limit to only Exlusive economic zones
            shapefile = shapefile[~shapefile["GEONAME"].str.contains(r"\(.*\)")]

            # Print the shape of clim_var_anomaly
            print("Shape of clim_var_anomaly: ", clim_var_anomaly.shape)

            # Calculate the mask
            # CALCULATE MASK
            shapefile["numbers"] = range(len(shapefile))

            # test the function
            eez_mask_poly = regionmask.from_geopandas(
                shapefile,
                names="GEONAME",
                abbrevs="ISO_SOV1",
                numbers="numbers",
            )

            # Create a mask to apply to the gridded dataset
            clim_var_anomaly_subset = clim_var_anomaly.isel(time=0)

            # # Print the values of the subset
            # print(f"clim var anomaly subset values: {clim_var_anomaly_subset.values()}")

            # Create the eez mask
            eez_mask = eez_mask_poly.mask(
                clim_var_anomaly_subset["lon"],
                clim_var_anomaly_subset["lat"],
            )

            # Create a dataframe
            df_ts = pd.DataFrame({"time": clim_var_anomaly.time.values})

            # Extract the lat and lons for the mask
            lat = eez_mask.lat.values
            lon = eez_mask.lon.values

            # Set up the n_flags
            n_flags = len(eez_mask.attrs["flag_values"])

            # Loop over the regions
            for i in tqdm((range(n_flags))):
                # Add a new column to the dataframe
                df_ts[eez_mask.attrs["flag_meanings"].split(" ")[i]] = np.nan

                # # Print the region
                print(
                    f"Calculating correlation for region: {eez_mask.attrs['flag_meanings'].split(' ')[i]}"
                )

                # Extract the mask for the region
                sel_mask = eez_mask.where(eez_mask == i).values

                # Set up the lon indices
                id_lon = lon[np.where(~np.all(np.isnan(sel_mask), axis=0))]

                # Set up the lat indices
                id_lat = lat[np.where(~np.all(np.isnan(sel_mask), axis=1))]

                # If the length of id_lon is 0 and the length of id_lat is 0
                if len(id_lon) == 0 and len(id_lat) == 0:
                    print(
                        f"Region {eez_mask.attrs['flag_meanings'].split(' ')[i]} is empty."
                    )
                    print("Continuing to the next region.")
                    continue

                # Print the id_lat and id_lon
                print("id_lat[0], id_lat[-1]: ", id_lat[0], id_lat[-1])

                # Print the id_lat and id_lon
                print("id_lon[0], id_lon[-1]: ", id_lon[0], id_lon[-1])

                # Select the region from the anoms
                out_sel = (
                    clim_var_anomaly.sel(
                        lat=slice(id_lat[0], id_lat[-1]),
                        lon=slice(id_lon[0], id_lon[-1]),
                    )
                    .compute()
                    .where(eez_mask == i)
                )

                # Group this into a mean
                out_sel = out_sel.mean(dim=["lat", "lon"])

                # Add this to the dataframe
                df_ts[eez_mask.attrs["flag_meanings"].split(" ")[i]] = out_sel.values

            # Take the central rolling average
            df_ts = (
                df_ts.set_index("time")
                .rolling(window=rolling_window, center=centre)
                .mean()
            )

            # modify each of the column names to include '_si10'
            # at the end of the string
            df_ts.columns = [
                f"{col}_{obs_var}" for col in df_ts.columns if col != "time"
            ]

            # pRINT THE column names
            print("Column names df_ts: ", df_ts.columns)

            # print the shape of df_ts
            print("Shape of df_ts: ", df_ts.shape)

            # Print df_ts head
            print("df_ts head: ", df_ts.head())

            # Drop the first rolling_window/2 rows
            df_ts = df_ts.iloc[int(rolling_window / 2) :]

            # join the dataframes
            merged_df = df.join(df_ts, how="inner")

            # Print merged df
            print("Merged df before NaN removed: ", merged_df.head())

            # # Drop the NaN values
            # merged_df = merged_df.dropna()

            # Print merged df
            print("Merged df: after Nan removed ", merged_df.head())

            # Print the column names in merged df
            print("Column names in merged df: ", merged_df.columns)

            # Create a new dataframe for the correlations
            corr_df = pd.DataFrame(columns=["region", "correlation", "p-value"])

            # Find the length of the merged_df.columns which don't contain "Si10"
            n_cols = len(
                [
                    col
                    for col in merged_df.columns
                    if obs_var not in col and "time" not in col
                ]
            )

            # print ncols
            print("Number of columns: ", n_cols)

            # Loop over the columns
            for i in tqdm(range(n_cols)):
                # Extract the column
                col = merged_df.columns[i]

                # Convert col to iso bname
                col_iso = dicts.iso_mapping[col]

                # If merged_df[f"{col_iso}_{obs_var}"] doesn't exist
                # Then create this
                # and fill with NaN values
                if f"{col_iso}_{obs_var}" not in merged_df.columns:
                    merged_df[f"{col_iso}_{obs_var}"] = np.nan

                # Check whether the length of the column is 4
                assert (
                    len(merged_df[col]) >= 2
                ), f"The length of the column is less than 2 for {col}"

                # Same check for the other one
                assert (
                    len(merged_df[f"{col_iso}_{obs_var}"]) >= 2
                ), f"The length of the column is less than 2 for {col_iso}_{obs_var}"

                # If merged_df[f"{col_iso}_{obs_var}"] contains NaN values
                # THEN fill the corr and pval with NaN
                if merged_df[f"{col_iso}_{obs_var}"].isnull().values.any():
                    corr = np.nan
                    pval = np.nan

                    # Append to the dataframe
                    corr_df_to_append = pd.DataFrame(
                        {"region": [col], "correlation": [corr], "p-value": [pval]}
                    )

                    # Append to the dataframe
                    corr_df = pd.concat([corr_df, corr_df_to_append], ignore_index=True)

                    # continue to the next iteration
                    continue

                # Calculate corr between wind power (GW) and wind speed
                corr, pval = pearsonr(merged_df[col], merged_df[f"{col_iso}_{obs_var}"])

                # Append to the dataframe
                corr_df_to_append = pd.DataFrame(
                    {"region": [col], "correlation": [corr], "p-value": [pval]}
                )

                # Append to the dataframe
                corr_df = pd.concat([corr_df, corr_df_to_append], ignore_index=True)

            # Return the dataframe
            return merged_df, corr_df, shapefile
        elif shp_file is not None and "NUTS" in shp_file and use_model_data is False:
            print("Averaging data for NUTS regions")

            # Assert that the shape file is not one
            assert shp_file is not None, "The shapefile is None."

            # Assert that the shape file directory is not none
            assert shp_file_dir is not None, "The shapefile directory is None."

            # Assert that the shape file directory exists
            assert os.path.exists(
                shp_file_dir
            ), "The shapefile directory does not exist."

            # Assert that the shapefile exists
            assert os.path.exists(
                os.path.join(shp_file_dir, shp_file)
            ), "The shapefile does not exist."

            # Load the shapefile
            shapefile = gpd.read_file(os.path.join(shp_file_dir, shp_file))

            # restrict to level code 0
            shapefile = shapefile[shapefile["LEVL_CODE"] == 0]

            # TODO: Fix hardcoded here
            country_codes = list(dicts.countries_nuts_id.values())

            # Limit the dataframe to those country_codes
            shapefile = shapefile[shapefile.NUTS_ID.isin(country_codes)]

            # Keep only the NUTS_ID, NUTS_NAME, and geometry columns
            shapefile = shapefile[["NUTS_ID", "NUTS_NAME", "geometry"]]

            # Set up the numbers for the mask
            shapefile["numbers"] = range(len(shapefile))

            # Test the masking function
            nuts_mask_poly = regionmask.from_geopandas(
                shapefile,
                names="NUTS_NAME",
                abbrevs="NUTS_ID",
                numbers="numbers",
            )

            # Create a subset of the clim data
            clim_var_anomaly_subset = clim_var_anomaly.isel(time=0)

            # Print the values of clim_var_anomaly_subset
            # Print the values of the subset
            print(f"clim var anomaly subset values: {clim_var_anomaly_subset.values}")

            # Create the mask
            nuts_mask = nuts_mask_poly.mask(
                clim_var_anomaly_subset["lon"],
                clim_var_anomaly_subset["lat"],
            )

            # Create a dataframe
            df_ts = pd.DataFrame({"time": clim_var_anomaly.time.values})

            # Extract the lat and lons for the mask
            lat = nuts_mask.lat.values
            lon = nuts_mask.lon.values

            # Set up the n_flags
            n_flags = len(nuts_mask.attrs["flag_values"])

            # Loop over the regions
            for i in tqdm((range(n_flags))):
                # Add a new column to the dataframe
                df_ts[nuts_mask.attrs["flag_meanings"].split(" ")[i]] = np.nan

                # Print the region
                print(
                    f"Calculating correlation for region: {nuts_mask.attrs['flag_meanings'].split(' ')[i]}"
                )

                # Extract the mask for the region
                sel_mask = nuts_mask.where(nuts_mask == i).values

                # Set up the lon indices
                id_lon = lon[np.where(~np.all(np.isnan(sel_mask), axis=0))]

                # Set up the lat indices
                id_lat = lat[np.where(~np.all(np.isnan(sel_mask), axis=1))]

                # If the length of id_lon is 0 and the length of id_lat is 0
                if len(id_lon) == 0 and len(id_lat) == 0:
                    print(
                        f"Region {nuts_mask.attrs['flag_meanings'].split(' ')[i]} is empty."
                    )
                    print("Continuing to the next region.")
                    continue

                # Print the id_lat and id_lon
                print("id_lat[0], id_lat[-1]: ", id_lat[0], id_lat[-1])

                # Print the id_lat and id_lon
                print("id_lon[0], id_lon[-1]: ", id_lon[0], id_lon[-1])

                # Select the region from the anoms
                out_sel = (
                    clim_var_anomaly.sel(
                        lat=slice(id_lat[0], id_lat[-1]),
                        lon=slice(id_lon[0], id_lon[-1]),
                    )
                    .compute()
                    .where(nuts_mask == i)
                )

                # # print the values of out_sel
                # print(f"out sel values {out_sel.values}")

                # Group this into a mean
                out_sel = out_sel.mean(dim=["lat", "lon"])

                # # Print the values of out sel
                # # print the values of out_sel
                # print(f"out sel values after mean {out_sel.values}")

                # Add this to the dataframe
                df_ts[nuts_mask.attrs["flag_meanings"].split(" ")[i]] = out_sel.values

            # Take the central rolling average
            df_ts = (
                df_ts.set_index("time")
                .rolling(window=rolling_window, center=centre)
                .mean()
            )

            # modify each of the column names to include '_si10'
            # at the end of the string
            df_ts.columns = [
                f"{col}_{obs_var}" for col in df_ts.columns if col != "time"
            ]

            # # pRINT THE column names
            # print("Column names df_ts: ", df_ts.columns)

            # # print the shape of df_ts
            # print("Shape of df_ts: ", df_ts.shape)

            # # Print df_ts head
            # print("df_ts head: ", df_ts.head())

            # Drop the first rolling window/2 values
            df_ts = df_ts.iloc[int(rolling_window / 2) :]

            # join the dataframes
            merged_df = df.join(df_ts, how="inner")

            # Prin the head of the merged_df
            print(f"merged df head: {merged_df.head()}")

            # Create a new dataframe for the correlations
            corr_df = pd.DataFrame(columns=["region", "correlation", "p-value"])

            # Find the length of the merged_df.columns which don't contain "Si10"
            n_cols = len(
                [
                    col
                    for col in merged_df.columns
                    if obs_var not in col and "time" not in col
                ]
            )

            # Loop over the columns
            for i in tqdm(range(n_cols)):
                # Extract the column
                col = merged_df.columns[i]

                # If merged_df[f"{col_iso}_{obs_var}"] doesn't exist
                # Then create this
                # and fill with NaN values
                if f"{col}_{obs_var}" not in merged_df.columns:
                    merged_df[f"{col}_{obs_var}"] = np.nan

                # Check whether the length of the column is 4
                assert (
                    len(merged_df[col]) >= 2
                ), f"The length of the column is less than 2 for {col}"

                # Same check for the other one
                assert (
                    len(merged_df[f"{col}_{obs_var}"]) >= 2
                ), f"The length of the column is less than 2 for {col_iso}_{obs_var}"

                # If merged_df[f"{col_iso}_{obs_var}"] contains NaN values
                # THEN fill the corr and pval with NaN
                if merged_df[f"{col}_{obs_var}"].isnull().values.any():
                    corr = np.nan
                    pval = np.nan

                    # Append to the dataframe
                    corr_df_to_append = pd.DataFrame(
                        {"region": [col], "correlation": [corr], "p-value": [pval]}
                    )

                    # Append to the dataframe
                    corr_df = pd.concat([corr_df, corr_df_to_append], ignore_index=True)

                    # continue to the next iteration
                    continue

                # Calculate corr between wind power (GW) and wind speed
                corr, pval = pearsonr(merged_df[col], merged_df[f"{col}_{obs_var}"])

                # Append to the dataframe
                corr_df_to_append = pd.DataFrame(
                    {"region": [col], "correlation": [corr], "p-value": [pval]}
                )

                # Append to the dataframe
                corr_df = pd.concat([corr_df, corr_df_to_append], ignore_index=True)

            # Return the dataframe
            return merged_df, corr_df, shapefile
        elif shp_file is not None and "NUTS" in shp_file and use_model_data is True:
            print("Using model data averaged over NUTS regions")

            # Assert that the model array directory exists
            assert os.path.isdir(
                model_arr_dir
            ), f"{model_arr_dir} does not exist or is not a directory"

            # Set up the filename root for the data
            # assert that variable, season, region, start_year, end_year,
            # forecast range, lag, and method are in model_config
            assert all(
                key in model_config
                for key in [
                    "variable",
                    "season",
                    "region",
                    "start_year",
                    "end_year",
                    "forecast_range",
                    "lag",
                    "method",
                ]
            ), "One or more required keys are missing from model_config"

            # Form the root of the filename
            fnames_root = f"{model_config['variable']}_{model_config['season']}_{model_config['region']}_{model_config['start_year']}_{model_config['end_year']}_{model_config['forecast_range']}_{model_config['lag']}_*_{model_config['method']}.npy"

            # Form the path to the files
            matching_files = glob.glob(f"{model_arr_dir}{fnames_root}")

            # Print the matching files
            print(f"matching files {model_arr_dir}{fnames_root}")

            # prit the globbed files
            print(f"globbed files {matching_files}")

            # If the len of matching files is greater than 1
            if len(matching_files) > 1:
                print("More than one matching file found.")

                # Extract the datetimes
                datetimes = [file.split("_")[7] for file in matching_files]

                # Remove the .npy from the datetimes
                datetimes = [datetime.split(".")[0] for datetime in datetimes]

                # Convert the datasetimes to datetimes using pandas
                datetimes = [
                    pd.to_datetime(datetime, unit="s") for datetime in datetimes
                ]

                # Find the latest datetime
                latest_datetime = max(datetimes)

                # Find the index of the latest datetime
                latest_datetime_index = datetimes.index(latest_datetime)

                # Print that we are using the latest datetime file
                print(
                    "Using the latest datetime file:",
                    matching_files[latest_datetime_index],
                )

                # Load the file
                data = np.load(matching_files[latest_datetime_index])
            else:
                # Load the file
                data = np.load(matching_files[0])

            # Print the dimensions of the data
            # shape of the data: (51, 664, 72, 144)
            print(f"shape of the data: {data.shape}")

            # Replace the 1th axis with the 0th axis and vice versa
            data = np.swapaxes(data, 0, 1)

            # If there are multiple ensemble members
            # then take the ensemble mean
            if data.shape[0] > 1:
                data = np.mean(data, axis=0)
            else:
                data = data[0]

            # assert that the shapefile is not none
            assert shp_file is not None, "The shapefile is None."

            # assert that the shapefile directory is not none
            assert shp_file_dir is not None, "The shapefile directory is None."

            # assert that the shapefile directory exists
            assert os.path.exists(
                shp_file_dir
            ), "The shapefile directory does not exist."

            # assert that the shapefile exists
            assert os.path.exists(
                os.path.join(shp_file_dir, shp_file)
            ), "The shapefile does not exist."

            # Load the shapefile
            shapefile = gpd.read_file(os.path.join(shp_file_dir, shp_file))

            # restrict to level code 0
            shapefile = shapefile[shapefile["LEVL_CODE"] == 0]

            # Set up the country codes
            country_codes = list(dicts.countries_nuts_id.values())

            # Limit the dataframe to those country_codes
            shapefile = shapefile[shapefile.NUTS_ID.isin(country_codes)]

            # Keep only the NUTS_ID, NUTS_NAME, and geometry columns
            shapefile = shapefile[["NUTS_ID", "NUTS_NAME", "geometry"]]

            # Set up the numbers for the mask
            shapefile["numbers"] = range(len(shapefile))

            # Test the masking function
            nuts_mask_poly = regionmask.from_geopandas(
                shapefile,
                names="NUTS_NAME",
                abbrevs="NUTS_ID",
                numbers="numbers",
            )

            # Also process the climate data - these are the obs
            # Create a subset of the clim data
            clim_var_anomaly_subset = clim_var_anomaly.isel(time=0)

            # Create the mask for the obs
            nuts_mask_obs = nuts_mask_poly.mask(
                clim_var_anomaly_subset["lon"],
                clim_var_anomaly_subset["lat"],
            )

            # Create a dataframe
            df_ts_obs = pd.DataFrame({"time": clim_var_anomaly.time.values})

            # Set up the lats and lons as we would expect them to be
            lons = np.arange(-180, 180, 2.5)
            lats = np.arange(-90, 90, 2.5)

            # Using regionmask, set up a numpy mask for these regions
            nuts_mask = nuts_mask_poly.mask(lons, lats)

            # Set up the n_flags
            n_flags = len(nuts_mask.attrs["flag_values"])

            # And for the obs
            n_flags_obs = len(nuts_mask_obs.attrs["flag_values"])

            # Set up the valid years
            if model_config["forecast_range"] == "2-9":
                valid_years = np.arange(
                    model_config["start_year"] + 5, model_config["end_year"] + 5 + 1
                )
            elif model_config["forecast_range"] == "2-5":
                raise NotImplementedError(
                    "The forecast range 2-5 is not yet implemented."
                )
            else:
                raise ValueError("The forecast range is not recognised.")

            # Create a dataframe
            df_ts = pd.DataFrame({"time": valid_years})

            # Extracts the lats and lons
            lats = nuts_mask.lat.values
            lons = nuts_mask.lon.values

            # And for the nuts mask obs
            lats_obs = nuts_mask_obs.lat.values
            lons_obs = nuts_mask_obs.lon.values

            # Extract the nuts_mask values
            nuts_mask_values = nuts_mask.values

            # # print the lats and lons
            # print("lats: ", lats)
            # print("lons: ", lons)

            # # Print the nuts mask values
            # print(f"nuts mask values {nuts_mask_values}")

            # Loop over the regions
            for i in tqdm((range(n_flags))):
                # add a new column to the dataframe
                df_ts[nuts_mask.attrs["flag_meanings"].split(" ")[i]] = np.nan

                # Print the region we are calculating correlations for
                print(
                    f"Calculating correlation for region: {nuts_mask.attrs['flag_meanings'].split(' ')[i]}"
                )

                # Extract the mask for the region
                sel_mask = nuts_mask.where(nuts_mask == i).values

                # Set up the lon indices
                id_lon = lons[np.where(~np.all(np.isnan(sel_mask), axis=0))]

                # Set up the lat indices
                id_lat = lats[np.where(~np.all(np.isnan(sel_mask), axis=1))]

                # If the length of id_lon is 0 and the length of id_lat is 0
                if len(id_lon) == 0 and len(id_lat) == 0:
                    print(
                        f"Region {nuts_mask.attrs['flag_meanings'].split(' ')[i]} is empty."
                    )
                    print("Continuing to the next region.")
                    continue

                # # Print the id_lat and id_lon
                # print("id_lat[0], id_lat[-1]: ", id_lat[0], id_lat[-1])

                # # Print the id_lat and id_lon
                # print("id_lon[0], id_lon[-1]: ", id_lon[0], id_lon[-1])

                # # print the id_lat and id_lon
                # print("id_lat: ", id_lat)
                # print("id_lon: ", id_lon)

                # print("id_lat type: ", type(id_lat))
                # print("id_lon type: ", type(id_lon))

                # Find the index for the id_lat[0] and id_lat[-1]
                id_lat0_idx = np.where(lats == id_lat[0])[0][0]
                id_lat1_idx = np.where(lats == id_lat[-1])[0][0]

                # Find the index for the id_lon[0] and id_lon[-1]
                id_lon0_idx = np.where(lons == id_lon[0])[0][0]
                id_lon1_idx = np.where(lons == id_lon[-1])[0][0]

                # Select the region from the data
                data_region = data[
                    :, id_lat0_idx : id_lat1_idx + 1, id_lon0_idx : id_lon1_idx + 1
                ]

                # Create a mask for the region
                region_mask = sel_mask[
                    id_lat0_idx : id_lat1_idx + 1, id_lon0_idx : id_lon1_idx + 1
                ]

                # Create a boolean region mask
                region_mask_bool = region_mask == i

                # # Print the shape of the region mask bool
                # print(f"region mask bool shape {region_mask_bool.shape}")

                # # print the type of the region mask bool
                # print(f"region mask bool type {type(region_mask_bool)}")

                # # print the region mask bool
                # print(f"region mask bool: {region_mask_bool}")

                # # print the shape of the data region
                # print(f"data region shape {data_region.shape}")

                # # print the shape of the region_mask
                # print(f"region mask shape {region_mask.shape}")

                # # Print the region mask
                # print(f"region mask: {region_mask}")

                # Initialise out_sel with the same shape as data region
                out_sel = np.zeros([data_region.shape[0]])

                # Loop over the first axis
                for j in range(data_region.shape[0]):
                    # Apply the mask
                    masked_data = data_region[j][region_mask_bool]

                    # if the masked data has two dimensions
                    if len(masked_data.shape) == 2:
                        # take the mean over the 0th and 1st axis
                        masked_data = np.mean(masked_data, axis=(0, 1))
                    elif len(masked_data.shape) == 1:
                        # Take the mean over the 0th axis
                        masked_data = np.mean(masked_data, axis=0)
                    else:
                        # Raise an error
                        raise ValueError("The masked data has more than 2 dimensions.")

                    # Assign the masked data to out_sel
                    out_sel[j] = masked_data

                # print the shape of out_sel
                # print(f"out sel shape {out_sel.shape}")
                # print(f"out set values {out_sel}")

                # Add this to the dataframe
                df_ts[nuts_mask.attrs["flag_meanings"].split(" ")[i]] = out_sel

            # Loop over the region for the obs
            for i in tqdm((range(n_flags_obs))):
                # add a new column to the dataframe
                df_ts_obs[nuts_mask_obs.attrs["flag_meanings"].split(" ")[i]] = np.nan

                # Print the region we are calculating correlations for
                print(
                    f"Calculating correlation for region: {nuts_mask_obs.attrs['flag_meanings'].split(' ')[i]}"
                )

                # Extract the mask for the region
                sel_mask = nuts_mask_obs.where(nuts_mask_obs == i).values

                # Set up the lon indices
                id_lon = lons_obs[np.where(~np.all(np.isnan(sel_mask), axis=0))]

                # Set up the lat indices
                id_lat = lats_obs[np.where(~np.all(np.isnan(sel_mask), axis=1))]

                # If the length of id_lon is 0 and the length of id_lat is 0
                if len(id_lon) == 0 and len(id_lat) == 0:
                    print(
                        f"Region {nuts_mask_obs.attrs['flag_meanings'].split(' ')[i]} is empty."
                    )
                    print("Continuing to the next region.")
                    continue

                # # Print the id_lat and id_lon
                # print("id_lat[0], id_lat[-1]: ", id_lat[0], id_lat[-1])

                # # Print the id_lat and id_lon
                # print("id_lon[0], id_lon[-1]: ", id_lon[0], id_lon[-1])

                # # print the id_lat and id_lon
                # print("id_lat: ", id_lat)
                # print("id_lon: ", id_lon)

                # print("id_lat type: ", type(id_lat))
                # print("id_lon type: ", type(id_lon))

                # Select the region for the anoms
                out_sel = (
                    clim_var_anomaly.sel(
                        lat=slice(id_lat[0], id_lat[-1]),
                        lon=slice(id_lon[0], id_lon[-1]),
                    )
                    .compute()
                    .where(nuts_mask_obs == i)
                )

                # Group this into a mean
                out_sel = out_sel.mean(dim=["lat", "lon"])

                # Add this to the dataframe
                df_ts_obs[nuts_mask_obs.attrs["flag_meanings"].split(" ")[i]] = out_sel.values

            # Take the central rolling average
            df_ts_obs = (
                df_ts_obs.set_index("time")
                .rolling(window=rolling_window, center=centre)
                .mean()
            )

            # Set the index to year
            df_ts_obs.index = df_ts_obs.index.year

            # Set up the columns for df_ts_obs
            df_ts_obs.columns = [
                f"{col}_{obs_var}_obs" for col in df_ts_obs.columns if col != "time"
            ]

            # if the obs variable is in ["ssrd", "rsds"]
            if obs_var in ["ssrd", "rsds"]:
                # Divide by 86400 to convert from J/m^2 to W m/m^2
                df_ts_obs = df_ts_obs / 86400
            else:
                # Raise an error for other cases
                raise NotImplementedError(f"Handling for obs_var '{obs_var}' is not implemented yet.")

            # Drop the first rolling window over 2 values
            df_ts_obs = df_ts_obs.iloc[int(rolling_window / 2) :]

            # print the head of df_ts
            print("Head of df_ts: ", df_ts.head())

            # print the head of the df_ts_obs
            print("Head of df_ts_obs: ", df_ts_obs.head())

            # Set the index of df_ts to time
            df_ts = df_ts.set_index("time")

            # modify each of the column names to include '_si10'
            # at the end of the string
            df_ts.columns = [
                f"{col}_{obs_var}" for col in df_ts.columns if col != "time"
            ]

            try:
                # try joining the dataframes
                merged_df_ts = df_ts_obs.join(df_ts, how="inner")
            except Exception as e:
                # print the exception
                print(e)

            # Print the len of the merged_df_ts
            print("Length of merged_df_ts: ", len(merged_df_ts))
            print("Length of df_ts: ", len(df_ts))
            print("Length of df_ts_obs: ", len(df_ts_obs))

            # Assert that not all the values in merged_df_ts are NaN
            assert not merged_df_ts.isnull().values.all(), "All the values are NaN."

            # # Drop the first rolling window/2 values
            # df_ts = df_ts.iloc[int(rolling_window / 2) :]

            # set the df index to year
            df.index = df.index.year

            # print the head of the df
            print("Head of df: ", df.head())

            # # print the head of merged df
            # print("Merged df head: ", merged_df.head())

            # join the dataframes
            merged_df = df.join(df_ts, how="inner")

            # Create a new dataframe for the correlations
            corr_df = pd.DataFrame(columns=["region", "correlation", "p-value"])

            # Find the length of the merged_df.columns which don't contain "Si10"
            n_cols = len(
                [
                    col
                    for col in merged_df.columns
                    if obs_var not in col and "time" not in col
                ]
            )


            # Loop over the columns
            for i in tqdm(range(n_cols)):
                # Extract the column
                col = merged_df.columns[i]

                # If merged_df[f"{col_iso}_{obs_var}"] doesn't exist
                # Then create this
                # and fill with NaN values
                if f"{col}_{obs_var}" not in merged_df.columns:
                    merged_df[f"{col}_{obs_var}"] = np.nan

                # Check whether the length of the column is 4
                assert (
                    len(merged_df[col]) >= 2
                ), f"The length of the column is less than 2 for {col}"

                # Same check for the other one
                assert (
                    len(merged_df[f"{col}_{obs_var}"]) >= 2
                ), f"The length of the column is less than 2 for {col_iso}_{obs_var}"

                # If merged_df[f"{col_iso}_{obs_var}"] contains NaN values
                # THEN fill the corr and pval with NaN
                if merged_df[f"{col}_{obs_var}"].isnull().values.any():
                    corr = np.nan
                    pval = np.nan

                    # Append to the dataframe
                    corr_df_to_append = pd.DataFrame(
                        {"region": [col], "correlation": [corr], "p-value": [pval]}
                    )

                    # Append to the dataframe
                    corr_df = pd.concat([corr_df, corr_df_to_append], ignore_index=True)

                    # continue to the next iteration
                    continue

                # Calculate corr between wind power (GW) and wind speed
                corr, pval = pearsonr(merged_df[col], merged_df[f"{col}_{obs_var}"])

                # Append to the dataframe
                corr_df_to_append = pd.DataFrame(
                    {"region": [col], "correlation": [corr], "p-value": [pval]}
                )

                # Append to the dataframe
                corr_df = pd.concat([corr_df, corr_df_to_append], ignore_index=True)

            # Return the dataframes
            return merged_df, corr_df, shapefile, merged_df_ts
        elif shp_file is not None and "eez" in shp_file and use_model_data is True:
            print("Using model data averaged over EEZ regions")

            # Not yet implemented error
            raise NotImplementedError("This function is not yet implemented.")
        elif use_model_data is False:
            print("Averaging over specified gridbox")

            # assert that avg_grid is not none
            assert avg_grid is not None, "The average grid is None."

            # Extract the lat and lons from the avg_grid
            lon1, lon2 = avg_grid["lon1"], avg_grid["lon2"]
            lat1, lat2 = avg_grid["lat1"], avg_grid["lat2"]

            # Calculate the mean of the clim var anomalies for this region
            clim_var_mean = clim_var_anomaly.sel(
                lat=slice(lat1, lat2), lon=slice(lon1, lon2)
            ).mean(dim=["lat", "lon"])

            # Extract the time values
            time_values = clim_var_mean.time.values

            # Extract the values
            clim_var_values = clim_var_mean.values

            # Create a dataframe for this data
            clim_var_df = pd.DataFrame(
                {"time": time_values, f"{obs_var} anomaly mean": clim_var_values}
            )

            # Take the central rolling average
            clim_var_df = (
                clim_var_df.set_index("time")
                .rolling(window=rolling_window, center=centre)
                .mean()
            )

            # Drop the NaN values
            clim_var_df = clim_var_df.dropna()

            # Merge the dataframes
            merged_df = df.join(clim_var_df, how="inner")

            # Drop the NaN values
            merged_df = merged_df.dropna()

            # Create a new dataframe for the correlations
            corr_df = pd.DataFrame(columns=["region", "correlation", "p-value"])

            # Loop over the columns
            for col in merged_df.columns[:-1]:
                # Calculate the correlation
                corr, pval = pearsonr(
                    merged_df[col], merged_df[f"{obs_var} anomaly mean"]
                )

                # Append to the dataframe
                corr_df_to_append = pd.DataFrame(
                    {"region": [col], "correlation": [corr], "p-value": [pval]}
                )

                # Append to the dataframe
                corr_df = pd.concat([corr_df, corr_df_to_append], ignore_index=True)
        elif use_model_data is True:
            print(
                "Extracting the stored gridbox averaged variable data for the specified box"
            )

            # TODO: finish off this function here
            # Set up the filename for the data
            # in the format:
            model_filename = f"""{model_config["variable"]}_{model_config["region"]}_{model_config["season"]}_{model_config["forecast_range"]}_{model_config["start_year"]}_{model_config["end_year"]}_{model_config["lag"]}_{model_config["gridbox"]}_{model_config["method"]}.csv"""

            # Print the filename
            print("Model filename: ", model_filename)

            # Set up the filepath
            filepath = f"{df_dir}{model_filename}"

            # Asser that the filepath exists
            assert os.path.exists(filepath), f"The filepath: {filepath} does not exist."

            # Load the dataframe
            df_model = pd.read_csv(filepath)

            # Set the index for the loaded data as valid time
            df_model = df_model.set_index("valid_years")

            # Set the df index as the year
            df.index = df.index.year

            # # Print the head of the df_model dataframe
            # print("Head of df_model: ", df_model.head())

            # # Print the head of the UREAD data
            # print("Head of UREAD data: ", df.head())

            # Try to join the two datadrames
            try:
                merged_df = df.join(df_model, how="inner")
            except Exception as e:
                print("Error: ", e)

            # Create a new dataframe for the correlations
            corr_df = pd.DataFrame(columns=["region", "correlation", "p-value"])

            # Loop over the columns
            for col in merged_df.columns[:-6]:
                # Calculate the correlation
                corr, pval = pearsonr(merged_df[col], merged_df["fcst_ts_mean"])

                # Append to the dataframe
                corr_df_to_append = pd.DataFrame(
                    {"region": [col], "correlation": [corr], "p-value": [pval]}
                )

                # Append to the dataframe
                corr_df = pd.concat([corr_df, corr_df_to_append], ignore_index=True)

            # Return these dfs
            return df, df_model, merged_df, corr_df

        else:
            raise ValueError("The shapefile is not recognised.")

    # Return the dataframe
    return merged_df, corr_df


# Define a function to plot scatter for observed variables
# To quantify the relationship between the two
def plot_scatter_obs(
    index: np.ndarray,
    variable: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    index_name: str,
    variable_name: str,
    plot_gridbox: dict,
    title: str = None,
    show_eqn: bool = False,
    figsize_x: int = 8,
    figsize_y: int = 8,
) -> None:
    """
    For a 1D observed index (time,) and a 3D observed variable array (time, lat, lon), plot the scatter plot for this, with the index
    on the x-axis and the mean observed variable, averaged over the
    provided gridbox, on the y-axis.

    Args:
    -----

    index: np.ndarray
        The 1D observed index array.

    variable: np.ndarray
        The 3D observed variable array.

    lats: np.ndarray
        The latitude values for the variable array.

    lons: np.ndarray
        The longitude values for the variable array.

    index_name: str
        The name of the index.

    variable_name: str
        The name of the variable.

    plot_gridbox: dict
        The dictionary containing the gridbox information to plot.

    title: str
        The title for the plot.

    show_eqn: bool
        Whether to show the equation of the regression line on the plot.

    figsize_x: int
        The x size of the figure.

    figsize_y: int
        The y size of the figure.

    Returns:
    --------

    None
    """

    # Set up the figure
    fig, ax = plt.subplots(figsize=(figsize_x, figsize_y))

    # Assert that a gridbox is provided
    # containing keys: lat1, lat2, lon1, lon2
    assert (
        "lat1" in plot_gridbox
        and "lat2" in plot_gridbox
        and "lon1" in plot_gridbox
        and "lon2" in plot_gridbox
    ), "The gridbox is not correctly defined."

    # Find the indexes of the lats and lons
    lat1_idx = np.argmin(np.abs(lats - plot_gridbox["lat1"]))
    lat2_idx = np.argmin(np.abs(lats - plot_gridbox["lat2"]))
    lon1_idx = np.argmin(np.abs(lons - plot_gridbox["lon1"]))
    lon2_idx = np.argmin(np.abs(lons - plot_gridbox["lon2"]))

    # Constraint the variable array to the gridbox
    # and take the mean over lat and lon
    variable = variable[:, lat1_idx:lat2_idx, lon1_idx:lon2_idx].mean(axis=(1, 2))

    # If the string "NAO" is in the index name
    if index_name in ["NAO", "delta p"]:
        print("Converting NAO index from Pa to hPa")
        # Convert the index to hPa
        index = index / 100

    # if the string "pr" is in the variable name
    if "pr" in variable_name:
        print("Converting obs precip from m day-1 to mm day-1")
        # Convert the variable to mm/day
        variable = variable * 1000

    # Plot the scatter plot
    ax.scatter(index, variable, color="k")

    # Set up the x-axis label
    ax.set_xlabel(index_name)

    # Set up the y-axis label
    ax.set_ylabel(variable_name)

    # Set up the title
    if title is not None:
        ax.set_title(title)

    # Calculate the regression line
    slope, intercept, r_value, p_value, std_err = linregress(index, variable)

    # print the value of the slope and intercept
    print(f"Slope: {slope}, Intercept: {intercept}")

    # Show the values
    if intercept < 0:
        equation = f"y = {slope:.2f}x - {abs(intercept):.3f}"
    else:
        equation = f"y = {slope:.2f}x + {intercept:.3f}"

    # If show_eqn is True
    if show_eqn is True:
        # Plot the regression line
        ax.plot(
            index,
            slope * index + intercept,
            color="r",
        )

        ax.text(
            0.05,
            0.95,
            f"{equation}\nr = {r_value:.2f}, p = {p_value:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            bbox=dict(facecolor="white", alpha=0.5),
            fontsize=12,
        )

        # Set up a dataframe to store the values
        # of the linear regression
        df = pd.DataFrame(
            {
                "slope": [slope],
                "intercept": [intercept],
                "r_value": [r_value],
                "p_value": [p_value],
                "std_err": [std_err],
            }
        )

    else:
        ax.text(
            0.05,
            0.95,
            f"r = {r_value:.2f}, p = {p_value:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            bbox=dict(facecolor="white", alpha=0.5),
            fontsize=12,
        )

    # Show the plot
    plt.show()

    if show_eqn is True:
        return df

    return None


# Model NAO obs gridpoint var correlations
def calc_model_nao_gridbox_var_corr(
    nao_df: pd.DataFrame,
    gridbox: dict,
    obs_var: str,
    months: list = [10, 11, 12, 1, 2, 3],
    annual_offset: int = 3,
    obs_var_data_path: str = dicts.regrid_file,
    obs_start_year: str = "1960",
    obs_end_year: str = "2023",
    coeff_dir: str = "/home/users/benhutch/energy-met-corr/coeffs",
    coeff_fname: str = "nao_pr_scandi_slope.csv",
) -> pd.DataFrame:
    """
    Forms a dataframe containing the model NAO (or model delta P)
    and the calibrated model NAO, min and max, according to the coefficient
    values obtained from the linear regression between the variable and
    gridbox var values.

    Args:
    -----

    nao_df: pd.DataFrame
        The dataframe containing the model NAO values.

    gridbox: dict
        The dictionary containing the gridbox information.

    obs_var: str
        The observed variable to use for calculating the NAO index.

    months: list
        The months to use for the NAO index.

    annual_offset: int
        The annual offset to use for the NAO index.

    obs_var_data_path: str
        The path to the observed variable data.

    obs_start_year: int
        The start year for the observed data.

    obs_end_year: int
        The end year for the observed data.

    coeff_dir: str
        The directory containing the coefficients.

    coeff_fname: str
        The filename containing the coefficients.

    Returns:
    --------

    df: pd.DataFrame
        The dataframe containing the model NAO, min and max values.

    """

    # Assert that lat1, lat2, lon1, lon2 are in the gridbox keys
    assert (
        "lat1" in gridbox
        and "lat2" in gridbox
        and "lon1" in gridbox
        and "lon2" in gridbox
    ), "The gridbox is not correctly defined."

    # Load the coefficients
    # Assert that the file exusts
    assert (
        len(glob.glob(f"{coeff_dir}/{coeff_fname}")) == 1
    ), f"The file {coeff_dir}/{coeff_fname} does not exist."

    # Load the coefficients
    coeffs = pd.read_csv(f"{coeff_dir}/{coeff_fname}")

    # Extract the lat and lons from the gridbox
    lat1, lat2 = gridbox["lat1"], gridbox["lat2"]
    lon1, lon2 = gridbox["lon1"], gridbox["lon2"]

    # Load in the ERA5 data to validate against
    clim_var = xr.open_mfdataset(
        obs_var_data_path,
        combine="by_coords",
        parallel=True,
        chunks={"time": "auto", "latitude": "auto", "longitude": "auto"},
    )[obs_var]

    # If expver is a variable in the dataset
    if "expver" in clim_var.coords:
        # Combine the first two expver variables
        clim_var = clim_var.sel(expver=1).combine_first(clim_var.sel(expver=5))

    # Constrain obs to ONDJFM
    clim_var = clim_var.sel(time=clim_var.time.dt.month.isin(months))

    # Shift the time index back by 3 months
    clim_var_shifted = clim_var.shift(time=-annual_offset)

    # Take annual means
    clim_var_annual = clim_var_shifted.resample(time="Y").mean()

    # Throw away years 1959, 2021, 2022 and 2023
    clim_var_annual = clim_var_annual.sel(time=slice(obs_start_year, obs_end_year))

    # Remove the climatology
    clim_var_anomaly = clim_var_annual - clim_var_annual.mean(dim="time")

    # Calculate the mean of the clim var anomalies for this region
    clim_var_mean = clim_var_anomaly.sel(
        lat=slice(lat1, lat2), lon=slice(lon1, lon2)
    ).mean(dim=["lat", "lon"])

    # Extract the time values
    time_values = clim_var_mean.time.values

    # Extract the values
    clim_var_values = clim_var_mean.values

    # Create a dataframe for this data
    clim_var_df = pd.DataFrame(
        {"time": time_values, f"{obs_var} anomaly mean": clim_var_values}
    )

    # Take the central rolling average
    clim_var_df = clim_var_df.set_index("time").rolling(window=8, center=True).mean()

    # Drop the NaN values
    clim_var_df = clim_var_df.dropna()

    # Extract the time axis index to only include the years
    clim_var_df.index = clim_var_df.index.year

    # Join the two dataframes
    df = nao_df.join(clim_var_df, how="inner")

    # Assuming df is your DataFrame and "var228 anomaly mean" is the column you want to convert
    if "var228 anomaly mean" in df.columns:
        df["var228 anomaly mean"] = df["var228 anomaly mean"] * 1000

    # Convert NAO columns to hPa
    df["obs_nao"] = df["obs_nao"] / 100
    df["model_nao_mean"] = df["model_nao_mean"] / 100
    df["model_nao_members_min"] = df["model_nao_members_min"] / 100
    df["model_nao_members_max"] = df["model_nao_members_max"] / 100
    df["NAO anomaly (Pa)"] = df["NAO anomaly (Pa)"] / 100

    # Create new columns
    df["calibrated_model_nao_mean"] = np.nan
    df["calibrated_model_nao_members_min"] = np.nan
    df["calibrated_model_nao_members_max"] = np.nan

    # print the head of the coeffs df
    print(coeffs.head())

    # Extract the slope values
    slope = coeffs["slope"].values[0]
    intercept = coeffs["intercept"].values[0]

    # Calculate the calibrated model NAO values
    df["calibrated_model_nao_mean"] = df["model_nao_mean"] * slope + intercept

    df["calibrated_model_nao_members_min"] = (
        df["model_nao_members_min"] * slope + intercept
    )

    df["calibrated_model_nao_members_max"] = (
        df["model_nao_members_max"] * slope + intercept
    )

    return df


# Define a function to plot these correlations
def plot_calib_corr(
    df: pd.DataFrame,
    predictand_var: str,
    index_name: str,
    ylabel: str,
    figsize_x: int = 10,
    figsize_y: int = 6,
    zero_line: bool = True,
) -> None:
    """
    Plots the calibrated results for NAO variable correlations.

    Args:
    -----

    df: pd.DataFrame
        The dataframe containing the calibrated results.

    predictand_var: str
        The observed variable.

    index_name: str
        The name of the index.

    figsize_x: int
        The x size of the figure.

    figsize_y: int
        The y size of the figure.

    Returns:
    --------

    None

    """

    # Set up the figure
    fig, ax = plt.subplots(figsize=(figsize_x, figsize_y))

    # Plot the model NAO mean
    ax.plot(
        df.index, df["calibrated_model_nao_mean"], label=f"{index_name}", color="red"
    )

    # Plot the model NAO members min
    ax.fill_between(
        df.index,
        df["calibrated_model_nao_members_min"],
        df["calibrated_model_nao_members_max"],
        color="red",
        alpha=0.5,
    )

    # Plot the observed time series
    ax.plot(df.index, df[f"{predictand_var}"], label=f"{predictand_var}", color="k")

    # Set up the x-axis label
    ax.set_xlabel("Initialization year")

    # Set up the ylabel
    ax.set_ylabel(f"{ylabel}")

    # Calculate the correlation coefficients
    corr, p_val = pearsonr(
        df["calibrated_model_nao_mean"], df[f"{predictand_var}"]
    )

    # Include a textbox in the top left hand corner with the corr and p values
    plt.text(
        0.05,
        0.95,
        f"Corr: {round(corr, 2)}\n p-value: {round(p_val, 3)}",
        horizontalalignment="left",
        verticalalignment="top",
        transform=plt.gca().transAxes,
        bbox=dict(facecolor="white", alpha=0.5),
    )

    print(p_val)

    if zero_line:
        # Include a horixzontal black dashed line at y=0
        plt.axhline(0, color="black", linestyle="--")

    # Include a legend
    plt.legend(loc="upper right")

    # Show the plot
    plt.show()

    return None

# Define a function for plotting the time series
def plot_time_series(
    df: pd.DataFrame,
    predictor_col: str,
    predictand_col: str,
    ylabel: str,
    figsize_x: int = 10,
    figsize_y: int = 5,
    twin_axes: bool = True,
    ylabel_2: str = None,
    do_detrend: bool = False,
) -> None:
    """
    Plots the time series for the model NAO and the observed variable.

    Args:
    -----

    df: pd.DataFrame
        The dataframe containing the calibrated results.

    predictor_col: str
        The name of the variable being used to predict.

    predictand_col: str
        The name of the variable we are trying to predict.

    ylabel: str
        The label for the (first) y-axis

    figsize_x: int
        The x size of the figure.

    figsize_y: int
        The y size of the figure.

    twin_axes: bool
        Whether to plot on twin axes or not, default is True.

    y_label2: str
        In the case of using twin axes, the label for this second axis.

    do_detrend: bool
        True for detrended time series.

    Returns:
    --------

    None

    """

    # if do_detrend is true
    if do_detrend is True:
        # Detrend the time series
        df[predictor_col] = signal.detrend(df[predictor_col])
        df[predictand_col] = signal.detrend(df[predictand_col])

    # Set up the figure
    fig, ax = plt.subplots(figsize=(figsize_x, figsize_y))

    # Plot the model NAO mean
    ax.plot(df.index, df[f"{predictor_col}"], color="b")

    # Plot the y label
    ax.set_ylabel(ylabel, color="b")

    if twin_axes is True:
        # Create a twin axes
        ax2 = ax.twinx()

        # Plot the second variable
        ax2.plot(df.index, df[f"{predictand_col}"], color="r")

        # Set up the y label for the second axis
        ax2.set_ylabel(ylabel_2, color="r")

        # Set the colour of the ticks
        ax2.tick_params("y", colors="r")
    else:
        # Plot the observed time series
        ax.plot(df.index, df[f"{predictand_col}"], color="r")

    # Set up the x-axis label
    ax.set_xlabel("Initialization year")

    # Calculate the correlation coefficients
    corr, p_val = pearsonr(df[f"{predictor_col}"], df[f"{predictand_col}"])

    # Include a textbox in the top left hand corner with the corr and p values
    plt.text(
        0.05,
        0.95,
        f"Corr: {round(corr, 2)}\n p-value: {round(p_val, 3)}",
        horizontalalignment="left",
        verticalalignment="top",
        transform=plt.gca().transAxes,
        bbox=dict(facecolor="white", alpha=0.5),
    )

    # print(p_val)

    # Include a horixzontal black dashed line at y=0
    plt.axhline(0, color="black", linestyle="--")

    # Include a legend
    plt.legend(loc="upper right")

    # Show the plot
    plt.show()

    # return none
    return None

# Write a function to plot with mean
# state for a variable over Europe
def plot_winter_mean(
    obs_var: str,
    obs_var_data_path: str,
    months: list = [10, 11, 12, 1, 2, 3],
    season: str = "ONDJFM",
    start_year=1960,
    end_year=2023,
    gridbox_plot: dict = dicts.north_atlantic_grid_plot,
    figsize_x: int = 10,
    figsize_y: int = 8,
    cmap: str = "coolwarm",
    vmin: float = None,
    vmax: float = None,
):
    """
    Function which calculates the mean state for the observed variable over 
    Europe.
    
    Args:
    
    obs_var: str
        The observed variable to use for calculating the mean state.
        
    obs_var_data_path: str
        The path to the observed variable data.
        
    months: list
        The months to use for the mean state.
        
    season: str
        The season to use for the mean state.
        
    start_year: int
        The start year for the mean state.
        
    end_year: int
        The end year for the mean state.

    gridbox_plot: dict
        The dictionary containing the gridbox information to plot.

    figsize_x: int
        The x size of the figure.

    figsize_y: int
        The y size of the figure.

    cmap: str
        The colormap to use for the plot.

    vmin: float
        The minimum value for the colorbar.

    vmax: float
        The maximum value for the colorbar.
        
    Returns:

    None
    """

    # Load in the data
    ds = xr.open_mfdataset(
        obs_var_data_path,
        combine="by_coords",
        parallel=True,
        chunks={"time": "auto", "latitude": "auto", "longitude": "auto"},
    )[obs_var]

    # If expver is a variable in the dataset
    if "expver" in ds.coords:
        # Combine the first two expver variables
        ds = ds.sel(expver=1).combine_first(ds.sel(expver=5))

    # if the variable is ssrd or rsds
    if obs_var in ["ssrd", "rsds"]:
        # divide by 86400 to convert from J/m^2 to W m/m^2
        ds = ds / 86400

    # Constrain obs to ONDJFM
    ds = ds.sel(time=ds.time.dt.month.isin(months))

    # Constrain the years
    ds = ds.sel(time=slice(f"{start_year}", f"{end_year}"))

    # Calculate the mean over the season
    ds_mean = ds.mean(dim="time")

    # Set up the figure
    fig = plt.figure(figsize=(figsize_x, figsize_y))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    if vmin is None and vmax is None:
        # Set the vmin and vmax
        vmin, vmax = ds_mean.min().values, ds_mean.max().values

    if obs_var not in ["ssrd", "rsds", "var228", "pr", "si10", "sfcWind"]:
        # set up the units
        units = ds_mean.units
    elif obs_var in ["si10", "sfcWind"]:
        # Set up the units
        units = "m s$^{-1}$"
    elif obs_var in ["pr", "var228"]:
        # Set up the units
        units = "kg m$^{-2}$"
    else:
        # Set up the units
        units = "W m$^{-2}$"

    # Include borders
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle="-")

    # Plot the mean state
    ds_mean.plot(ax=ax,
                transform=ccrs.PlateCarree(),
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                cbar_kwargs={
                    "label": f"{obs_var} ({units})",
                    "orientation": "horizontal",
                    "shrink": 0.8,
                    "pad": 0.05,
                    "extend": "both",  # Extend the colorbar at both ends
                             },
    )
    # Include coastlines
    ax.coastlines()

    # Set the title
    ax.set_title(f"{obs_var} mean state over Europe")

    # Constrain to specific bounds
    ax.set_extent([gridbox_plot["lon1"], gridbox_plot["lon2"],
                   gridbox_plot["lat1"], gridbox_plot["lat2"]],
                  crs=ccrs.PlateCarree())
    
    # Show the plot
    plt.show()

    return None

# Define another function to plot the data from the CLEARHEADS work
# Either in the EEZ or NUTS domains
def plot_eu_clearheads(
    file: str,
    shp_file: str,
    shp_file_dir: str,
    label: str,
    months: list = [10, 11, 12, 1, 2, 3],
    clearheads_dir: str = dicts.clearheads_dir,
    figsize_x: int = 10,
    figsize_y: int = 8,
    trend_level: float = None,
    start_date: str = "1950-01-01",
    time_units: str = "h",
    start_year: int = 1960,
    end_year: int = 2023,
    values="detrended_data",
    cmap="coolwarm",
    gridbox_plot: dict = dicts.north_atlantic_grid_plot,
):
    """
    Plots the climatology over Europe for the given variable.
    
    Args:
    
    file: str
        The filename for the data.
        
    shp_file: str
        The shapefile to use for plotting.
        
    shp_file_dir: str
        The directory containing the shapefile.

    label: str
        The label for the plot.
        
    clearheads_dir: str
        The directory containing the CLEARHEADS data.
        
    figsize_x: int
        The x size of the figure.
        
    figsize_y: int
        The y size of the figure.
        
    trend_level: float
        The level of the trend to plot.

    start_date: str
        The start date for the time axis.

    time_units: str
        The time units for the time axis.

    start_year: int
        The start year for the data.

    end_year: int
        The end year for the data.
    
    months: list
        The months to plot.

    values: str
        The values to plot from the data.

    cmap
        The colormap to use for the plot.

    gridbox_plot: dict
        The dictionary containing the gridbox information to plot.
        
    Returns:
    
    None
    """

    # asssert that the file exists in the clearheads directory
    assert os.path.exists(os.path.join(clearheads_dir, file)), f"The file {file} does not exist."

    # assert that the shapefile exists
    assert os.path.exists(os.path.join(shp_file_dir, shp_file)), f"The shapefile {shp_file} does not exist."

    # Load teh data
    ds = xr.open_dataset(os.path.join(clearheads_dir, file))

    # extract the nuts keys
    NUTS_keys = ds.NUTS_keys.values

    # if trend_level is not None
    if trend_level is not None:
        # extract the trend levels
        trend_levels = ds.trend_levels.values

        # Find the index of the trend level
        idx = np.where(trend_levels == trend_level)[0][0]

        # Extract the data
        ds = ds.isel(trend=idx)

    # turn the data into a dataframe
    df = ds.to_dataframe()

    # pivot the dataframe
    df = df.reset_index().pivot(
        index="time_in_hours_from_first_jan_1950",
        columns="NUTS",
        values=values,
    )

    # Add the NUTS_keys to the columns
    df.columns = NUTS_keys

    # Convert 'time_in_hours_from_first_jan_1950' to datetime
    df.index = pd.to_datetime(df.index, unit=time_units, origin=start_date)

    # restrict the data to the start and end years
    df = df.loc[f"{start_year}":f"{end_year}"]

    # Collapse the dataframe into monthly averages
    df = df.resample("M").mean()

    # Select only the months of interest
    df = df[df.index.month.isin(months)]

    # Calculate the time average
    df = df.mean()

    if "NUTS" in shp_file:
        # Load the shapefile
        shapefile = gpd.read_file(os.path.join(shp_file_dir, shp_file))

        # Restrict to level code 0
        shapefile = shapefile[shapefile.LEVL_CODE == 0]

        # Extract the second element of the tuple
        countries_codes = list(dicts.countries_nuts_id.values())

        # Limit the gpd to the countries in the dictionary
        shapefile = shapefile[shapefile.NUTS_ID.isin(countries_codes)]

        # Keep only the NUTS_ID, NUTS_NAME, and geometry columns
        shapefile = shapefile[["NUTS_ID", "NUTS_NAME", "geometry"]]

        # Loop over the columns in the shapefile
        # and add the values to the shapefile
        for index, row in shapefile.iterrows():
            # Extract the NUTS code
            nuts_id = row["NUTS_ID"]

            try:
                # Find the index of the row in df
                # which matches the NUTS code
                idx = df.index.get_loc(nuts_id)
            except KeyError:
                print(f"The NUTS code {nuts_id} is not in the dataframe.")
                continue

            # If the index is not None
            if idx is not None:
                # Add the value to the shapefile
                shapefile.loc[index, "value"] = df.iloc[idx]
            else:
                print(f"The NUTS code {nuts_id} is not in the dataframe.")
                continue

    # Assert that shapefile has the value column
    assert "value" in shapefile.columns, "The value column is not in the shapefile."

    # Set up the figure
    fig = plt.figure(figsize=(figsize_x, figsize_y))

    # Set up the axes
    ax = plt.axes(projection=ccrs.PlateCarree())

    # plot the shapefile
    shapefile.plot(
        column="value",
        ax=ax,
        legend=True,
        cmap=cmap,
        legend_kwds={
        "label": label,
        "orientation": "horizontal",
        "shrink": 0.8,
        "pad": 0.01,
    },
    )

    # add the coastlines
    ax.coastlines()

    # constrain the plot to Europe
    ax.set_extent([gridbox_plot["lon1"], gridbox_plot["lon2"],
                   gridbox_plot["lat1"], gridbox_plot["lat2"]],
                  crs=ccrs.PlateCarree())

    return None

# Define a function for aggregating correlations for multiple different
# observed predictors
# e.g. for onshore wind we consider:
# NAO
# Delta P
# 10m wind speed
# 850U
def aggregate_obs_correlations(
    uread_fname: str,
    shp_fname: str,
    shp_fpath: str,
    obs_vars: list,
    obs_var_data_paths: list,
    obs_var_levels: list,
    uread_fpath: str = dicts.clearheads_dir,
    nao_n_grid: dict = dicts.iceland_grid_corrected,
    nao_s_grid: dict = dicts.azores_grid_corrected,
    delta_p_n_grid: dict = dicts.uk_n_box_corrected,
    delta_p_s_grid: dict = dicts.uk_s_box_corrected,
    save_df_dir: str = "/home/users/benhutch/energy-met-corr/df/",
    save_fname: str = "corr_df_combined.csv",
) -> pd.DataFrame:
    """
    Function which aggregates the correlations for multiple observed predictors using the correlate_nao_uread function and pandas.
    The output is saved to a csv file.
    
    Args:
    
    uread_fname: str
        The filename for the UREAD data.
        
    shp_fname: str
        The filename for the shapefile.
        
    shp_fpath: str
        The filepath for the shapefile.
        
    obs_vars: list
        The list of observed variables to use.
        
    obs_var_data_paths: list
        The list of observed variable data paths.
        
    obs_var_levels: list
        The list of observed variable levels.

    uread_fpath: str
        The filepath for the UREAD data.
        
    nao_n_grid: dict
        The dictionary containing the gridbox information for the NAO North.
        Default is dicts.iceland_grid_corrected.
        
    nao_s_grid: dict
        The dictionary containing the gridbox information for the NAO South.
        Default is dicts.azores_grid_corrected.
        
    delta_p_n_grid: dict
        The dictionary containing the gridbox information for the Delta P North.
        Default is dicts.uk_n_box_corrected.
        
    delta_p_s_grid: dict
        The dictionary containing the gridbox information for the Delta P South.
        Default is dicts.uk_s_box_corrected.
        
    save_df_dir: str
        The directory to save the dataframe to.
        
    Output:

    df: pd.DataFrame
        The dataframe containing the aggregated correlations.
    """

    # Assert that the shapefile exists
    assert os.path.exists(os.path.join(shp_fpath, shp_fname)), f"The shapefile {shp_fname} does not exist."

    # Assert that the uread file exists
    assert os.path.exists(os.path.join(uread_fpath, uread_fname)), f"The UREAD file {uread_fname} does not exist."

    # assert that obs_var_levels is a list containing ints
    assert all(isinstance(x, int) for x in obs_var_levels), "The obs_var_levels list must contain integers."

    # Calculate the correlations for the NAO
    _, corr_df_nao = correlate_nao_uread(
        filename=uread_fname,
    )

    # append _nao to the correlation and p-value columns
    # using rename
    corr_df_nao = corr_df_nao.rename(
        columns={"correlation": "correlation_nao", "p-value": "p-value_nao"}
    )

    # set the index to the region column
    corr_df_nao = corr_df_nao.set_index("region", inplace=True)

    # Calculate the correlations for the Delta P
    _, corr_df_delta_p = correlate_nao_uread(
        filename=uread_fname,
        nao_n_grid=delta_p_n_grid,
        nao_s_grid=delta_p_s_grid,
    )

    # append _delta_p to the correlation and p-value columns
    # using rename
    corr_df_delta_p = corr_df_delta_p.rename(
        columns={"correlation": "correlation_delta_p", "p-value": "p-value_delta_p"}
    )

    # set the index to the region column
    corr_df_delta_p = corr_df_delta_p.set_index("region", inplace=True)

    # Join the dataframes
    corr_df_combined = corr_df_nao.join(corr_df_delta_p, how="inner")

    # Loop over the observed variables, paths and levels using zip
    for var, path, level in zip(obs_vars, obs_var_data_paths, obs_var_levels):
        
        # if level is not zero
        if level != 0:
            # add the level to the var_name
            var_name = f"{var}_{level}"

            # Calculate the correlations
            _, corr_df, _ = correlate_nao_uread(
                filename=uread_fname,
                shp_file=shp_fname,
                shp_file_dir=shp_fpath,
                obs_var=var,
                obs_var_data_path=path,
                level=level,
            )

            # append _var_name to the correlation and p-value columns
            # using rename
            corr_df = corr_df.rename(
                columns={"correlation": f"correlation_{var_name}", "p-value": f"p-value_{var_name}"}
            )
        else:
            # Calculate the correlations
            _, corr_df, _ = correlate_nao_uread(
                filename=uread_fname,
                shp_file=shp_fname,
                shp_file_dir=shp_fpath,
                obs_var=var,
                obs_var_data_path=path,
            )

            # append _var_name to the correlation and p-value columns
            # using rename
            corr_df = corr_df.rename(
                columns={"correlation": f"correlation_{var}", "p-value": f"p-value_{var}"}
            )

        # join the dataframes
        corr_df_combined = corr_df_combined.join(corr_df, how="inner")

    # For the save_path
    save_path = os.path.join(save_df_dir, save_fname)

    # Save the dataframe
    corr_df_combined.to_csv(save_path)

    return corr_df_combined