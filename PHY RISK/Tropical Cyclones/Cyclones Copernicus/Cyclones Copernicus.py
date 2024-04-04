#%% Load packages
import xarray as xr
import pandas as pd
import numpy as np
from tqdm import tqdm

#%% Call function to climate data
def get_data(var, route):
    data = xr.open_dataset(route)
    df = data[var].to_dataframe()
    df = df.reset_index()
    df["lon"] = ((df["lon"]+180)%360)-180
    return df

#%% Exposure Data
def get_exposure():
    return pd.read_excel(r"C:\Users\alfredo.serrano.fig1\Desktop\BAC\Codigos\Physical Risk\Tropical Cyclones\Cyclones Copernicus - To work on\Collateral_Columns.xlsx")

#%% Call to damage functions
def get_impf(haz):
    fun = pd.read_excel(r"C:\Users\alfredo.serrano.fig1\Desktop\BAC\Codigos\Physical Risk\Fire\Damage_Curves_Assets.xlsx", sheet_name = "Impact_Function")
    fun = fun[fun["peril"]== haz]
    return (fun["intensity"],fun["mdr"])

#%% Parameter setting
start_date = 2023
end_date = 2100
radius = 0.8
var = "sfcWind"
event = "Tropical cyclone"
threshold = 0.9

#%% Call to copernicus dataframe
df = get_data(var,r"C:\Users\alfredo.serrano.fig1\Desktop\BAC\Codigos\Physical Risk\Tropical Cyclones\Cyclones Copernicus - To work on\sfcWind_day_MIROC6_ssp245_r1i1p1f1_gn_20230101-21001231_v20200323.nc")

#%% Call to exposure data
exp = get_exposure()

#%% Call to intensity and mdr values from damage curves
impf_x, impf_y = get_impf(event)

#%% Extract [year] column from copernicus data
df["year"] = df["time"].dt.year
df = df[(df["year"] >= start_date) & (df["year"] <= end_date)]

#%% Create blank dataframe
impacts = pd.DataFrame()

#%% For loop for calculations
for i in tqdm(range(exp.shape[0])):
    val, lat, lon = exp.loc[i]
    aux = df[(df["lat"] >= lat-1) & (df["lat"] <= lat+1) & (df["lon"] >= lon-1) & (df["lon"] <= lon+1)]
    aux = aux.groupby("time").mean().reset_index()
    aux[var + "_event"] = aux[var]
    thr = aux.groupby("year").quantile(threshold).reset_index()[["year",var]]
    thr.columns = ["year", "threshold"]
    aux = aux.merge(right = thr, how = "left", on = "year")

    #Nueva tabla, join
    aux.loc[aux[var+"_event"] < aux["threshold"], var+"_event"] = np.nan
    res = pd.DataFrame()
    res["Mean"] = aux.groupby("year").mean()[var]
    res["Mean_event"] = aux.groupby("year").mean()[var+"_event"]
    res["Exceeds_Theshold"] = aux.groupby("year").count()[var+"_event"]
    res["Impact"] = np.interp(res["Mean_event"],impf_x,impf_y)
    res["Impact"] = 1-(1-res["Impact"])**res["Exceeds_Theshold"]
    res["Impact"] = res["Impact"] * val
    res["Latitutde"] = lat
    res["Longitud"] = lon
    res["id"] = i
    impacts = pd.concat([impacts,res])

#%% Reset index
impacts = impacts.reset_index()

#%% Create excel to store output data
impacts.to_excel(r"C:\Users\alfredo.serrano.fig1\Desktop\BAC\Codigos\Physical Risk\Tropical Cyclones\Cyclones Copernicus - To work on\Results\Raw Output\Cyclones_4.5_Miroc_.xlsx")
