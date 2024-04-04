
#%% Import necessary libraries
import xarray as xr # - Used to open and process data from NetCDF files
import pandas as pd # - Used to read and manipulate data tables and perform data analysis
import numpy as np # - Used for numerical operations and mathematical operations
from tqdm import tqdm # - Used to generate a progress bar for loops in Python
import matplotlib.pyplot as plt # - used to create plots and visuals
import plotly.express as px
import seaborn as sns
from sklearn.linear_model import LinearRegression

#%%Function to read and process precipitation data from a NetCDF file
def get_data(var, route):
    data = xr.open_dataset(route)
    df = data[var].to_dataframe()
    df = df.reset_index()
    df["lon"] = ((df["lon"]+180)%360)-180
    return df

#%% Function to read exposure data from a CSV file
def get_exposure():
     # Read exposure data and select relevant columns
    # 'Value' represents the asset value, 'Latitud' and 'Longitud' are the geographical coordinates
        # - ! Find out what is this csv file reference
    return pd.read_excel(r"C:\Users\alfredo.serrano.fig1\Desktop\BAC\Codigos\Physical Risk\Fire\Collateral_Columns.xlsx")

#%% Function to read impact functions (damage curves) from an Excel file
def get_impf(haz):
    # Read damage curves for the specified hazard (e.g., flood)
    # 'haz' is the hazard type, 'intensity' is the environmental intensity measure, and 'mdr' is mean damage ratio
    fun = pd.read_excel(r"C:\Users\alfredo.serrano.fig1\Desktop\BAC\Codigos\Physical Risk\Fire\Damage_Curves_Assets.xlsx", sheet_name = "Impact_Function")
    fun = fun[fun["peril"]== haz]
    return (fun["intensity"],fun["mdr"])

#%% Setting parameters for the analysis
start_date = 2023
end_date = 2100
radius = 0.8
var_1 = "tasmax" #- Daily Maximum Near-Surface Air Temperature
var_2 = "pr" #- Precipitation
var = "FWI" #- Wildfire variable within damage curves
event = "Wildfire"
threshold = 0.9

#%% Dataframe creation for two variables
df = get_data(var_1,r"C:\Users\alfredo.serrano.fig1\Desktop\BAC\Codigos\Physical Risk\Fire\tasmax_Amon_MIROC6_ssp245_r1i1p1f1_gn_20230116-21001216_v20190627.nc")
df[var_2] = get_data(var_2,r"C:\Users\alfredo.serrano.fig1\Desktop\BAC\Codigos\Physical Risk\Fire\pr_Amon_MIROC6_ssp245_r1i1p1f1_gn_20230116-21001216_v20190627.nc")[var_2]

#%% Retrieve exposure data
exp = get_exposure()

#%% Retrieve impact function for the flood event
impf_x, impf_y = get_impf(event)

#%% Extract year out of the dataframe
df["year"] = df["time"].dt.year
df = df[(df["year"] >= start_date) & (df["year"] <= end_date)]

#%% Initialize DataFrame to store impact results
impacts = pd.DataFrame()

#%% Iterate over each exposure point to calculate impact
for i in tqdm(range(exp.shape[0])):
    val, lat, lon = exp.loc[i]
    aux = df[(df["lat"] >= lat-1) & (df["lat"] <= lat+1) & (df["lon"] >= lon-1) & (df["lon"] <= lon+1)]
    aux = aux.groupby("time").mean().reset_index()
    total = aux.groupby("year")[var_2].sum().reset_index()
    total.columns = ["year", "sum"]
    aux = aux.merge(right = total, how = "left", on = "year")
    aux[var_2] = aux[var_2]/aux["sum"]
    aux[var] = -68.80982 + 0.2968547 * aux[var_1] - 8.823971 * aux[var_2] - 6133.46 * aux["sum"]
    aux[var + "_event"] = aux[var]
    thr = aux.groupby("year").quantile(threshold).reset_index()[["year",var]]
    thr.columns = ["year", "threshold"]
    aux = aux.merge(right = thr, how = "left", on = "year")

    #New table, join
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

#%% Export results to a XLSX file
impacts.to_excel(r"C:\Users\alfredo.serrano.fig1\Desktop\BAC\Codigos\Physical Risk\Fire\Results\Raw Output\Fire_Temp_4.5_Miroc_.xlsx")
