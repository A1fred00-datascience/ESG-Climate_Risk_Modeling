#%%Import the necessary libraries and packages
import pandas as pd
import os
from scipy.optimize import minimize
from scipy.stats import norm
import numpy as np

# Define constants and parameters for the model
# PD_RATINGS: Probability of Default ratings for different categories
# START_YEAR, END_YEAR: Start and end years for the analysis
# DECAY_RATE: Rate at which financial impacts decay over time
# Other constants are used for various calculations in the modelPD_RATINGS = [0.03, 0.1, 0.14, 0.27, 0.35]

#PD_RATINGS = [0.03, 0.1, 0.14, 0.27, 0.35, 0.45, 0.55, 0.65, 0.75] #Ratings commented as we are using SAN master scale - 12/12 
#%%
START_YEAR = 2020
END_YEAR = 2050
DECAY_RATE = .246810041816523 #Decay rate. Source: Standard & Poor's
CONV_SCALAR = 1.0E-09
MAX_ALPHA = 100000
MIN_INCREMENT = 0.10
COST_FLOOR = 0.01
LOWER_BOUND = -100
UPPER_BOUND = 100
SCENARIO = 'Net Zero 2050'
SECTOR = 'Power' # Sectors are: 'O&G' &'Power' 

#%% Define constants for different financial and environmental variables
BASE = 'Current Policies'
CARBON_PRICE = "Price|Carbon"
TOTAL_EMISSIONS = "Emissions|CO2|Energy"
PRIMARY_ENERGY = "Primary Energy"
PRIMARY_ENERGY_GAS = "Primary Energy|Gas"
PRIMARY_ENERGY_OIL = "Primary Energy|Oil|w/o CCS"
OIL_PRICE = "Price|Primary Energy|Oil"
GAS_PRICE = "Price|Primary Energy|Gas"
PRIMARY_ENERGY_ELECTRICITY = "Final Energy|Electricity"
ELECTRICITY_PRICE = "Price|Final Energy|Residential and Commercial|Residential|Electricity"
LOW_CARBON_INVESTMENT = "Post-processed|Investment|Low Carbon"
EMISSIONS_ELECTRICITY = "Emissions|CO2|Energy|Supply|Electricity"
ELECTRIC_COAL = "Secondary Energy|Electricity|Coal"
ELECTRIC_COAL_PRICE = "Price|Secondary Energy|Solids|Coal"
ELECTRIC_GAS = "Secondary Energy|Electricity|Gas"
ELECTRIC_GAS_PRICE = "Price|Secondary Energy|Gases|Natural Gas"
ELECTRIC_OIL = "Secondary Energy|Electricity|Oil"
ELECTRIC_OIL_PRICE = "Price|Secondary Energy|Liquids|Oil"
ELECTRICITY_INVESTMENT = "Post-processed|Investment|Energy Supply|Electricity"

#MAX_TTC = len(PD_RATINGS) #Max tc commented as PD_Ratings is no longer used - 12/12

#%% Define master scale as a list of tuples (lower bound, upper bound, PD Value)
master_scale = [
    (1,1.5,0.45),
    (1.5,2,0.45),
    (2,2.5,0.297468),
    (2.5,3,0.167052),
    (3,3.5,0.093813),
    (3.5,4,0.052684),
    (4.0,4.5,0.029586),
    (4.5,5,0.016615),
    (5.0,5.5,0.009331),
    (5.5,6,0.00524),
    (6,6.5,0.002943),
    (6.5,7,0.001653),
    (7,7.5,0.000928),
    (7.5,8,0.000521),
    (8,8.5,0.0003),
    (8.5,9.3,0.0003)
    ]



#%%Convert list of tuples to a Dataframe
master_scale = pd.DataFrame(master_scale, columns = ['Lower Rating', 'Upper Rating', 'Regulatory PD'])

#%% Function to map TTC to PD using the master scale
def get_pd_from_ttc(ttc_rating, master_scale):
    print(f"Current TTC Rating: {ttc_rating}") #Debug print
    #Iterate through the master scale to find matching PD
    for index, row in master_scale.iterrows():
        lower_bound = row['Lower Rating']
        upper_bound = row['Upper Rating']
        #pd_value = row['Regulatory PD']
        if lower_bound <= ttc_rating < upper_bound:
            return row['Regulatory PD']
    return None # If no PD found
   
        
# Function to select representative clients from a portfolio
#%%Filters clients based on subsector and outstanding amounts
def client_selection(portfolio):

    subsectors = portfolio.subsector.unique()
    subsets = np.array([])
    for value in subsectors:
        
        subset = portfolio[portfolio.subsector == value].copy()
        subset_mean = subset.outstanding.mean()
        filtered_cases = subset[(subset.outstanding >= subset_mean * 0.70) & 
                                (subset.outstanding <= subset_mean * 1.30)] 
        # for i in filtered_cases:#
        #     if i.index in [45698862, 45735415, 45711491]:#
        #         filtered_cases.drop(i)#
        filtered_cases = filtered_cases.groupby('pd', group_keys=False).apply(lambda x: x.sample(n=1))

        filtered_cases.index = filtered_cases.id
        subset.index = subset.id

        subset.drop(filtered_cases.index, inplace=True)

        additional_samples_needed = 5 - len(filtered_cases)
        if additional_samples_needed > 0:
            additional_cases = subset.sample(n=min(len(subset),additional_samples_needed)) 
            filtered_cases = pd.concat([filtered_cases, additional_cases])
            
        subsets = np.append(subsets, np.array(filtered_cases.id))
    
    return subsets


#%% Adjusts the ratings based on emissions and EBITDA
def climate_shock(clients, master_scale):
    rating_2030 = []
    rating_2040 = []

    epsilon_2030 = (clients.emissions * clients.price_2030) / clients.ebitda
    epsilon_2040 = (clients.emissions * clients.price_2040) / clients.ebitda
    

    for i, row in clients.iterrows():
        current_ttc = row['ttc_rating']
        
        #Adjust ttc rating for 2030
        if epsilon_2030[i] >=1:
            ttc_2030 = max(current_ttc - 0.5, min(master_scale['Lower Rating']))
        else:
            ttc_2030 = current_ttc
        
        #Adjust ttc rating for 2040
        if epsilon_2040[i] >=1:
            ttc_2040 = max(current_ttc - 1, min(master_scale['Lower Rating']))
        else:
            ttc_2040 = current_ttc - 0.5

        # Convert PD back to TTC rating for calibration (adjust this logic as needed)
        ttc_2030 = min(max(ttc_2030, min(master_scale['Lower Rating'])), max(master_scale['Upper Rating']))
        ttc_2040 = min(max(ttc_2040, min(master_scale['Lower Rating'])), max(master_scale['Upper Rating']))

        rating_2030.append(ttc_2030)
        rating_2040.append(ttc_2040)

    df_2030 = clients[['id', 'subsector', 'ttc_rating']].copy()
    df_2030['ttc_calibration'] = rating_2030
    df_2030['year'] = 2030

    df_2040 = clients[['id', 'subsector', 'ttc_rating']].copy()
    df_2040['ttc_calibration'] = rating_2040
    df_2040['year'] = 2040
    
    return pd.concat([df_2030, df_2040]).reset_index(drop=True)


#%%
# clients_dummy = pd.DataFrame({
#     'id': [45673271,45736014,45674230],
#     'subsector': ['Production of electricity','Production of Electricity', 'Distribution of electricity'],
#     'outstanding': [24055741.10,22808098.15,0.00],
#     'ebitda': [44979000.00,1102000000.00,8049000000.00],
#     'ttc_rating': [3.60,4.90,7.30],
#     'pd': [0.001653,0.016615,0.001653],
#     'lgd': [0.4,0.4,0.4],
#     'emissions': [46516.80,89687.26,15143.57],
#     'price_2030': [145.5269, 145.5269, 145.5269],
#     'price_2040': [215.94504, 215.94504, 215.94504]})

#%%
#adjusted_ratings = climate_shock(clients_dummy, master_scale)

# Function to calculate Risk Factor Profiles (RFPs)
#%% Uses different financial and environmental variables to compute RFPs
def calc_RFPs(id_sector, scenario, ngfs):
    start_year_aux = START_YEAR - (START_YEAR%5)
    end_year_aux = END_YEAR + 5 - (END_YEAR%5)

    if id_sector == 'O&G':
        carbon_price = ngfs.loc[(ngfs.scenario == SCENARIO) & (ngfs.variable == CARBON_PRICE) & (ngfs.year >= start_year_aux) & (ngfs.year <= end_year_aux)].sort_values(by='year').value.values
        total_emissions = ngfs.loc[(ngfs.scenario == SCENARIO) & (ngfs.variable == TOTAL_EMISSIONS) & (ngfs.year >= start_year_aux) & (ngfs.year <= end_year_aux)].sort_values(by='year').value.values
        primary_energy = ngfs.loc[(ngfs.scenario == SCENARIO) & (ngfs.variable == PRIMARY_ENERGY) & (ngfs.year >= start_year_aux) & (ngfs.year <= end_year_aux)].sort_values(by='year').value.values
        energy_gas = ngfs.loc[(ngfs.scenario == SCENARIO) & (ngfs.variable == PRIMARY_ENERGY_GAS) & (ngfs.year >= start_year_aux) & (ngfs.year <= end_year_aux)].sort_values(by='year').value.values
        energy_oil = ngfs.loc[(ngfs.scenario == SCENARIO) & (ngfs.variable == PRIMARY_ENERGY_OIL) & (ngfs.year >= start_year_aux) & (ngfs.year <= end_year_aux)].sort_values(by='year').value.values
        oil_price = ngfs.loc[(ngfs.scenario == SCENARIO) & (ngfs.variable == OIL_PRICE) & (ngfs.year >= start_year_aux) & (ngfs.year <= end_year_aux)].sort_values(by='year').value.values
        gas_price = ngfs.loc[(ngfs.scenario == SCENARIO) & (ngfs.variable == GAS_PRICE) & (ngfs.year >= start_year_aux) & (ngfs.year <= end_year_aux)].sort_values(by='year').value.values
        energy_electricity = ngfs.loc[(ngfs.scenario == SCENARIO) & (ngfs.variable == PRIMARY_ENERGY_ELECTRICITY) & (ngfs.year >= start_year_aux) & (ngfs.year <= end_year_aux)].sort_values(by='year').value.values
        electricity_price = ngfs.loc[(ngfs.scenario == SCENARIO) & (ngfs.variable == ELECTRICITY_PRICE) & (ngfs.year >= start_year_aux) & (ngfs.year <= end_year_aux)].sort_values(by='year').value.values
        low_carbon = ngfs.loc[(ngfs.scenario == SCENARIO) & (ngfs.variable == LOW_CARBON_INVESTMENT) & (ngfs.year >= start_year_aux) & (ngfs.year <= end_year_aux)].sort_values(by='year').value.values

        revenues = (energy_oil*oil_price + energy_gas*gas_price)*1000000000
        direct_cost= (total_emissions/primary_energy)*(energy_gas+energy_oil)*carbon_price*1000000
        indirect_cost = energy_electricity*electricity_price*1000000000
        capital = low_carbon*1000000000

        revenues_select = []
        direct_select = []
        indirect_select = []
        capital_select = []

        for i in range(1, len(revenues)):
            revenues_select.extend([revenues[i-1]])
            direct_select.extend([direct_cost[i-1]])
            indirect_select.extend([indirect_cost[i-1]])
            capital_select.extend([capital[i-1]])
            for j in range(1,5):
                revenues_select.extend([revenues[i-1]+j*(revenues[i]-revenues[i-1])/5])
                direct_select.extend([direct_cost[i-1]+j*(direct_cost[i]-direct_cost[i-1])/5])
                indirect_select.extend([indirect_cost[i-1]+j*(indirect_cost[i]-indirect_cost[i-1])/5])
                capital_select.extend([capital[i-1]+j*(capital[i]-capital[i-1])/5])

        revenues_select.extend([revenues[-1]])
        direct_select.extend([direct_cost[-1]])
        indirect_select.extend([indirect_cost[-1]])
        capital_select.extend([capital[-1]])

        carbon_price = ngfs.loc[(ngfs.scenario == BASE) & (ngfs.variable == CARBON_PRICE) & (ngfs.year >= start_year_aux) & (ngfs.year <= end_year_aux)].sort_values(by='year').value.values
        total_emissions = ngfs.loc[(ngfs.scenario == BASE) & (ngfs.variable == TOTAL_EMISSIONS) & (ngfs.year >= start_year_aux) & (ngfs.year <= end_year_aux)].sort_values(by='year').value.values
        primary_energy = ngfs.loc[(ngfs.scenario == BASE) & (ngfs.variable == PRIMARY_ENERGY) & (ngfs.year >= start_year_aux) & (ngfs.year <= end_year_aux)].sort_values(by='year').value.values
        energy_gas = ngfs.loc[(ngfs.scenario == BASE) & (ngfs.variable == PRIMARY_ENERGY_GAS) & (ngfs.year >= start_year_aux) & (ngfs.year <= end_year_aux)].sort_values(by='year').value.values
        energy_oil = ngfs.loc[(ngfs.scenario == BASE) & (ngfs.variable == PRIMARY_ENERGY_OIL) & (ngfs.year >= start_year_aux) & (ngfs.year <= end_year_aux)].sort_values(by='year').value.values
        oil_price = ngfs.loc[(ngfs.scenario == BASE) & (ngfs.variable == OIL_PRICE) & (ngfs.year >= start_year_aux) & (ngfs.year <= end_year_aux)].sort_values(by='year').value.values
        gas_price = ngfs.loc[(ngfs.scenario == BASE) & (ngfs.variable == GAS_PRICE) & (ngfs.year >= start_year_aux) & (ngfs.year <= end_year_aux)].sort_values(by='year').value.values
        energy_electricity = ngfs.loc[(ngfs.scenario == BASE) & (ngfs.variable == PRIMARY_ENERGY_ELECTRICITY) & (ngfs.year >= start_year_aux) & (ngfs.year <= end_year_aux)].sort_values(by='year').value.values
        electricity_price = ngfs.loc[(ngfs.scenario == BASE) & (ngfs.variable == ELECTRICITY_PRICE) & (ngfs.year >= start_year_aux) & (ngfs.year <= end_year_aux)].sort_values(by='year').value.values
        low_carbon = ngfs.loc[(ngfs.scenario == BASE) & (ngfs.variable == LOW_CARBON_INVESTMENT) & (ngfs.year >= start_year_aux) & (ngfs.year <= end_year_aux)].sort_values(by='year').value.values

        revenues = (energy_oil*oil_price + energy_gas*gas_price)*1000000000
        direct_cost= (total_emissions/primary_energy)*(energy_gas+energy_oil)*carbon_price*1000000
        indirect_cost = energy_electricity*electricity_price*1000000000
        capital = low_carbon*1000000000

        revenues_base = []
        direct_base = []
        indirect_base = []
        capital_base = []

        for i in range(1, len(revenues)):
            revenues_base.extend([revenues[i-1]])
            direct_base.extend([direct_cost[i-1]])
            indirect_base.extend([indirect_cost[i-1]])
            capital_base.extend([capital[i-1]])
            for j in range(1,5):
                revenues_base.extend([revenues[i-1]+j*(revenues[i]-revenues[i-1])/5])
                direct_base.extend([direct_cost[i-1]+j*(direct_cost[i]-direct_cost[i-1])/5])
                indirect_base.extend([indirect_cost[i-1]+j*(indirect_cost[i]-indirect_cost[i-1])/5])
                capital_base.extend([capital[i-1]+j*(capital[i]-capital[i-1])/5])

        revenues_base.extend([revenues[-1]])
        direct_base.extend([direct_cost[-1]])
        indirect_base.extend([indirect_cost[-1]])
        capital_base.extend([capital[-1]])

    if id_sector == 'Power':
        carbon_price = ngfs.loc[(ngfs.scenario == SCENARIO) & (ngfs.variable == CARBON_PRICE) & (ngfs.year >= start_year_aux) & (ngfs.year <= end_year_aux)].sort_values(by='year').value.values
        electricity = ngfs.loc[(ngfs.scenario == SCENARIO) & (ngfs.variable == PRIMARY_ENERGY_ELECTRICITY) & (ngfs.year >= start_year_aux) & (ngfs.year <= end_year_aux)].sort_values(by='year').value.values
        electricity_price = ngfs.loc[(ngfs.scenario == SCENARIO) & (ngfs.variable == ELECTRICITY_PRICE) & (ngfs.year >= start_year_aux) & (ngfs.year <= end_year_aux)].sort_values(by='year').value.values
        total_emissions = ngfs.loc[(ngfs.scenario == SCENARIO) & (ngfs.variable == EMISSIONS_ELECTRICITY) & (ngfs.year >= start_year_aux) & (ngfs.year <= end_year_aux)].sort_values(by='year').value.values
        electric_coal = ngfs.loc[(ngfs.scenario == SCENARIO) & (ngfs.variable == ELECTRIC_COAL) & (ngfs.year >= start_year_aux) & (ngfs.year <= end_year_aux)].sort_values(by='year').value.values
        electric_coal_price = ngfs.loc[(ngfs.scenario == SCENARIO) & (ngfs.variable == ELECTRIC_COAL_PRICE) & (ngfs.year >= start_year_aux) & (ngfs.year <= end_year_aux)].sort_values(by='year').value.values
        electric_gas = ngfs.loc[(ngfs.scenario == SCENARIO) & (ngfs.variable == ELECTRIC_GAS) & (ngfs.year >= start_year_aux) & (ngfs.year <= end_year_aux)].sort_values(by='year').value.values
        electric_gas_price = ngfs.loc[(ngfs.scenario == SCENARIO) & (ngfs.variable == ELECTRIC_GAS_PRICE) & (ngfs.year >= start_year_aux) & (ngfs.year <= end_year_aux)].sort_values(by='year').value.values
        electric_oil = ngfs.loc[(ngfs.scenario == SCENARIO) & (ngfs.variable == ELECTRIC_OIL) & (ngfs.year >= start_year_aux) & (ngfs.year <= end_year_aux)].sort_values(by='year').value.values
        electric_oil_price = ngfs.loc[(ngfs.scenario == SCENARIO) & (ngfs.variable == ELECTRIC_OIL_PRICE) & (ngfs.year >= start_year_aux) & (ngfs.year <= end_year_aux)].sort_values(by='year').value.values
        investment = ngfs.loc[(ngfs.scenario == SCENARIO) & (ngfs.variable == ELECTRICITY_INVESTMENT) & (ngfs.year >= start_year_aux) & (ngfs.year <= end_year_aux)].sort_values(by='year').value.values

        revenues = electricity*electricity_price*1000000000
        direct_cost= total_emissions*carbon_price*1000000
        indirect_cost = (electric_coal*electric_coal_price+electric_oil_price*electric_oil+electric_gas*electric_gas_price)*1000000000
        capital = investment*1000000000

        revenues_select = []
        direct_select = []
        indirect_select = []
        capital_select = []

        for i in range(1, len(revenues)):
            revenues_select.extend([revenues[i-1]])
            direct_select.extend([direct_cost[i-1]])
            indirect_select.extend([indirect_cost[i-1]])
            capital_select.extend([capital[i-1]])
            for j in range(1,5):
                revenues_select.extend([revenues[i-1]+j*(revenues[i]-revenues[i-1])/5])
                direct_select.extend([direct_cost[i-1]+j*(direct_cost[i]-direct_cost[i-1])/5])
                indirect_select.extend([indirect_cost[i-1]+j*(indirect_cost[i]-indirect_cost[i-1])/5])
                capital_select.extend([capital[i-1]+j*(capital[i]-capital[i-1])/5])

        revenues_select.extend([revenues[-1]])
        direct_select.extend([direct_cost[-1]])
        indirect_select.extend([indirect_cost[-1]])
        capital_select.extend([capital[-1]])

        carbon_price = ngfs.loc[(ngfs.scenario == BASE) & (ngfs.variable == CARBON_PRICE) & (ngfs.year >= start_year_aux) & (ngfs.year <= end_year_aux)].sort_values(by='year').value.values
        electricity = ngfs.loc[(ngfs.scenario == BASE) & (ngfs.variable == PRIMARY_ENERGY_ELECTRICITY) & (ngfs.year >= start_year_aux) & (ngfs.year <= end_year_aux)].sort_values(by='year').value.values
        electricity_price = ngfs.loc[(ngfs.scenario == BASE) & (ngfs.variable == ELECTRICITY_PRICE) & (ngfs.year >= start_year_aux) & (ngfs.year <= end_year_aux)].sort_values(by='year').value.values
        total_emissions = ngfs.loc[(ngfs.scenario == BASE) & (ngfs.variable == EMISSIONS_ELECTRICITY) & (ngfs.year >= start_year_aux) & (ngfs.year <= end_year_aux)].sort_values(by='year').value.values
        electric_coal = ngfs.loc[(ngfs.scenario == BASE) & (ngfs.variable == ELECTRIC_COAL) & (ngfs.year >= start_year_aux) & (ngfs.year <= end_year_aux)].sort_values(by='year').value.values
        electric_coal_price = ngfs.loc[(ngfs.scenario == BASE) & (ngfs.variable == ELECTRIC_COAL_PRICE) & (ngfs.year >= start_year_aux) & (ngfs.year <= end_year_aux)].sort_values(by='year').value.values
        electric_gas = ngfs.loc[(ngfs.scenario == BASE) & (ngfs.variable == ELECTRIC_GAS) & (ngfs.year >= start_year_aux) & (ngfs.year <= end_year_aux)].sort_values(by='year').value.values
        electric_gas_price = ngfs.loc[(ngfs.scenario == BASE) & (ngfs.variable == ELECTRIC_GAS_PRICE) & (ngfs.year >= start_year_aux) & (ngfs.year <= end_year_aux)].sort_values(by='year').value.values
        electric_oil = ngfs.loc[(ngfs.scenario == BASE) & (ngfs.variable == ELECTRIC_OIL) & (ngfs.year >= start_year_aux) & (ngfs.year <= end_year_aux)].sort_values(by='year').value.values
        electric_oil_price = ngfs.loc[(ngfs.scenario == BASE) & (ngfs.variable == ELECTRIC_OIL_PRICE) & (ngfs.year >= start_year_aux) & (ngfs.year <= end_year_aux)].sort_values(by='year').value.values
        investment = ngfs.loc[(ngfs.scenario == BASE) & (ngfs.variable == ELECTRICITY_INVESTMENT) & (ngfs.year >= start_year_aux) & (ngfs.year <= end_year_aux)].sort_values(by='year').value.values

        revenues = electricity*electricity_price*1000000000
        direct_cost= total_emissions*carbon_price*1000000
        indirect_cost = (electric_coal*electric_coal_price+electric_oil_price*electric_oil+electric_gas*electric_gas_price)*1000000000
        capital = investment*1000000000

        revenues_base = []
        direct_base = []
        indirect_base = []
        capital_base = []

        for i in range(1, len(revenues)):
            revenues_base.extend([revenues[i-1]])
            direct_base.extend([direct_cost[i-1]])
            indirect_base.extend([indirect_cost[i-1]])
            capital_base.extend([capital[i-1]])
            for j in range(1,5):
                revenues_base.extend([revenues[i-1]+j*(revenues[i]-revenues[i-1])/5])
                direct_base.extend([direct_cost[i-1]+j*(direct_cost[i]-direct_cost[i-1])/5])
                indirect_base.extend([indirect_cost[i-1]+j*(indirect_cost[i]-indirect_cost[i-1])/5])
                capital_base.extend([capital[i-1]+j*(capital[i]-capital[i-1])/5])

        revenues_base.extend([revenues[-1]])
        direct_base.extend([direct_cost[-1]])
        indirect_base.extend([indirect_cost[-1]])
        capital_base.extend([capital[-1]])

    df_direct = np.array(direct_select)-np.array(direct_base)
    df_indirect = np.array(indirect_select)-np.array(indirect_base)
    df_capital = np.array(capital_select)-np.array(capital_base)

    for i in range(len(df_direct)):
        df_direct[i] = -1*np.abs(df_direct[i])
        df_indirect[i] = -1*np.abs(df_indirect[i])
        df_capital[i] = -1*np.abs(df_capital[i])

    cols =  ["Value"]

    aux_direct = ['direct_cost']
    aux_direct.extend(df_direct)
    aux_indirect = ['indirect_cost']
    aux_indirect.extend(df_indirect)
    aux_capital = ['capital_expenditure']
    aux_capital.extend(df_capital)
    aux_revenue = ['revenue']
    aux_revenue.extend(np.array(revenues_select)-np.array(revenues_base))

    cols.extend(range(start_year_aux,start_year_aux+len(revenues_base)))

    df_RFP = pd.DataFrame(columns = cols)
    df_RFP.loc[0] = aux_direct
    df_RFP.loc[1] = aux_indirect
    df_RFP.loc[2] = aux_capital
    df_RFP.loc[3] = aux_revenue

    df_RFP.index = df_RFP['Value']
    df_RFP = df_RFP.drop(labels = 'Value', axis = 1)

    return df_RFP.drop(labels = range(END_YEAR+1,start_year_aux+len(revenues_base)), axis = 1)


#%% Function to calculate decay of financial impacts over time
def calc_decay(RFPs, heatmap, sensitivities):
    
    decay = heatmap.copy()

    for value in sensitivities['Value']:
        sensibility = []
        for  i in decay[value]:
            sensibility.append(sensitivities.loc[value, i])
        
        decay[value] = sensibility

    decay = decay.reindex(columns=np.append(decay.columns, range(START_YEAR, END_YEAR+1)))

    for i in range(len(decay)):
        sens = decay.loc[i][1:5].values
        RFP = RFPs.copy()
        RFP = RFP.multiply(sens,axis = 0)
        RFP = RFP.multiply(CONV_SCALAR)
        for j in range(START_YEAR+1,END_YEAR+1):
            RFP[j] = RFP[j] + RFP[j-1]*(1-DECAY_RATE)
        decay.iloc[i,5:] = RFP.sum()
        
    decay_melted = decay.melt(id_vars=['subsector', 'direct_cost', 'indirect_cost', 'capital_expenditure', 'revenue'], var_name = 'year', value_name = 'decay_value')
    #decay_melted['year'] = df_decayC['year'].astype(int)
    
    return decay_melted


#%% Function to calculate the error in alpha parameter
def error_alpha(alpha, calibration_points, df_decayC, master_scale): #Function modified on 12/6 to handle new range of TTC ratings
    SSE = 0
    for i in range(len(calibration_points)):
        original_rating = calibration_points.loc[i, 'ttc_rating']
        year = calibration_points.loc[i,'year'] 
        new_rating = calibration_points.loc[i, 'ttc_calibration']
        subsec = calibration_points.loc[i, 'subsector']
        
        #Check if original_rating is within the valid range
        if 1 <= original_rating <= 9.3 and 1 <= new_rating <= 9.3:
            original_pd = get_pd_from_ttc(original_rating, master_scale)
            new_pd = get_pd_from_ttc(new_rating, master_scale)
            
            #Filter df_decayC for the current subsector and yar
            filtered_df = df_decayC[(df_decayC['subsector'] == subsec) & (df_decayC['year'] == year)]
            
            # Access the decay_value for the specific year and subsector
            if not filtered_df.empty:
                cvalue = filtered_df['decay_value'].values[0]
                mod_dist = norm.ppf(original_pd) - cvalue / alpha
                cal_dist = norm.ppf(new_pd)
                weight = 1 / (calibration_points["subsector"].value_counts()[subsec]*len(df_decayC))
                error = weight * (cal_dist - mod_dist) ** 2
                SSE += error
            else:
            #Handle case where no matching subsector and year are found
                print(f"No data found for Subsector: {subsec}, Year: {year}")
         
        else:
        #Debugging print statement
            print(f"Invalid rating: original {original_rating}, new {new_rating}") 
    return SSE



#%% Function to calculate error in sensitivities
def error_sensitivities(x, df_RFPs,df_Heatmaps,df_Calibration, alpha_opt, master_scale):

    df_Sens = pd.DataFrame({
        'Value':['direct_cost', 'indirect_cost', 'capital_expenditure', 'revenue'],
        'No impact':[0,0,0,0],
        'Low':[x[0], x[5], x[10],x[15]],
        'Mod low':[x[1], x[6], x[11],x[16]],
        'Moderate':[x[2], x[7], x[12],x[17]],
        'Mod high':[x[3], x[8], x[13],x[18]],
        'High':[x[4], x[9], x[14],x[19]]
    })

    df_Sens.index = df_Sens["Value"]
    calibs = calc_decay(df_RFPs,df_Heatmaps, df_Sens)

    SSE = 0
    for i in range(len(calibration_points)):
        original_rating = calibration_points.loc[i, 'ttc_rating']
        new_rating = df_Calibration.loc[i, 'ttc_calibration']
        subsec = df_Calibration.loc[i, 'subsector']
        year = df_Calibration.loc[i, 'year']
        
        #Check if original_rating is within the valid range
        if original_rating >= master_scale['Lower Rating'].min() and original_rating <= master_scale['Upper Rating'].max():
            original_pd = get_pd_from_ttc(original_rating, master_scale)
            new_pd = get_pd_from_ttc(new_rating, master_scale)
            
            #Filter df_decayC for the current subsector and yar
            filtered_df = calibs[(calibs['subsector'] == subsec) & (calibs['year'] == year)]
            
            if not filtered_df.empty:
                cvalue = filtered_df['decay_value'].values[0]
                mod_dist = norm.ppf(original_pd) - cvalue / alpha_opt
                cal_dist = norm.ppf(new_pd)
                weight = 1 / (df_Calibration['subsector'].value_counts()[subsec]*len(df_Heatmaps))
                error = weight * (cal_dist - mod_dist) ** 2
                SSE += error
        else:
            print(f"No data found for Subsector: {subsec}, Year: {year}")
            continue
            
    else:
        print(f"Invalid original rating: {original_rating} at index [i]")
            #Handle the invalid rating case, e.g, continue, break, or set a default value    
    return SSE


# Main script execution starts here
#%% Read portfolio data and NGFS (Network for Greening the Financial System) data
portfolio = pd.read_excel('CIB_Portfolio_Pre.xlsx', sheet_name ='P&E') #Change sheet == O&G for Oil & Gas / sheet == P&E for Power
representative_clients = client_selection(portfolio)
ngfs = pd.read_csv('Phase_IV_Data.csv')
ngfs = ngfs.drop('id', axis = 1)
ngfs = ngfs.drop_duplicates()
carbon_price = ngfs.loc[(ngfs.scenario == SCENARIO) & (ngfs.variable == CARBON_PRICE)]

#%%
representative_clients = portfolio.loc[portfolio.id.isin(representative_clients)].copy()
representative_clients['price_2030'] = carbon_price.loc[carbon_price.year == 2030, 'value'].values[0] - carbon_price.loc[carbon_price.year == START_YEAR, 'value'].values[0]
representative_clients['price_2040'] = carbon_price.loc[carbon_price.year == 2040, 'value'].values[0] - carbon_price.loc[carbon_price.year == START_YEAR, 'value'].values[0]

calibration_points = climate_shock(representative_clients, master_scale)

# calibration_points.to_excel("Calibration_Points_Phase_iii_ogunep.xlsx", index = False)

RFPs = calc_RFPs(SECTOR, SCENARIO, ngfs)

# heatmap = pd.read_excel('Heatmaps_NZ_ZP_Transition.xlsx', sheet_name='P&E-NZ') #Change sheet name O&G-NZ for Oil & Gas / P&E-NZ for Power
heatmap = pd.read_excel('Heatmaps_NZ_ZP_Transition.xlsx', sheet_name='P&E-NZ(RENEW)')

#%%
sensitivities = pd.DataFrame({
    'Value':['direct_cost', 'indirect_cost', 'capital_expenditure', 'revenue'],
    'No impact':[0,0,0,0],
    'Low':[1,1,1,1],
    'Mod low':[1,1,1,1],
    'Moderate':[1,1,1,1],
    'Mod high':[1,1,1,1],
    'High':[1,1,1,1]
})

#%%
sensitivities.index = sensitivities["Value"]

#%%
df_decayC = calc_decay(RFPs, heatmap, sensitivities)

#%%
alpha_opt = minimize(error_alpha, 1, args=(calibration_points,df_decayC, master_scale), constraints=({'type':'ineq','fun':lambda x: MAX_ALPHA-x})).x[0]

#%%
if (RFPs.loc["revenue"].mean() < 0):
    cons = (
        #Low <= ModLow
        {"type":"ineq", "fun": lambda x: x[1] - x[0]},
        {"type":"ineq", "fun": lambda x: x[6] - x[5]},
        {"type":"ineq", "fun": lambda x: x[11] - x[10]},
        {"type":"ineq", "fun": lambda x: x[16] - x[15]},
        #ModLow <= Mod
        {"type":"ineq", "fun": lambda x: x[2] - x[1]},
        {"type":"ineq", "fun": lambda x: x[7] - x[6]},
        {"type":"ineq", "fun": lambda x: x[12] - x[11]},
        {"type":"ineq", "fun": lambda x: x[17] - x[16]},
        #Mod <= ModHigh
        {"type":"ineq", "fun": lambda x: x[3] - x[2]},
        {"type":"ineq", "fun": lambda x: x[8] - x[7]},
        {"type":"ineq", "fun": lambda x: x[13] - x[12]},
        {"type":"ineq", "fun": lambda x: x[18] - x[17]},
        #ModHigh <= High
        {"type":"ineq", "fun": lambda x: x[4] - x[3]},
        {"type":"ineq", "fun": lambda x: x[9] - x[8]},
        {"type":"ineq", "fun": lambda x: x[14] - x[13]},
        {"type":"ineq", "fun": lambda x: x[19] - x[18]},
        #COST_FLOOR
        {"type":"ineq", "fun": lambda x: x[0] - COST_FLOOR},
        {"type":"ineq", "fun": lambda x: x[1] - COST_FLOOR},
        {"type":"ineq", "fun": lambda x: x[2] - COST_FLOOR},
        {"type":"ineq", "fun": lambda x: x[3] - COST_FLOOR},
        {"type":"ineq", "fun": lambda x: x[4] - COST_FLOOR},
        {"type":"ineq", "fun": lambda x: x[5] - COST_FLOOR},
        {"type":"ineq", "fun": lambda x: x[6] - COST_FLOOR},
        {"type":"ineq", "fun": lambda x: x[7] - COST_FLOOR},
        {"type":"ineq", "fun": lambda x: x[8] - COST_FLOOR},
        {"type":"ineq", "fun": lambda x: x[9] - COST_FLOOR},
        {"type":"ineq", "fun": lambda x: x[10] - COST_FLOOR},
        {"type":"ineq", "fun": lambda x: x[11] - COST_FLOOR},
        {"type":"ineq", "fun": lambda x: x[12] - COST_FLOOR},
        {"type":"ineq", "fun": lambda x: x[13] - COST_FLOOR},
        {"type":"ineq", "fun": lambda x: x[14] - COST_FLOOR},
        #Lower_Bound
        {"type":"ineq", "fun": lambda x: x[0] - LOWER_BOUND},
        {"type":"ineq", "fun": lambda x: x[1] - LOWER_BOUND},
        {"type":"ineq", "fun": lambda x: x[2] - LOWER_BOUND},
        {"type":"ineq", "fun": lambda x: x[3] - LOWER_BOUND},
        {"type":"ineq", "fun": lambda x: x[4] - LOWER_BOUND},
        {"type":"ineq", "fun": lambda x: x[5] - LOWER_BOUND},
        {"type":"ineq", "fun": lambda x: x[6] - LOWER_BOUND},
        {"type":"ineq", "fun": lambda x: x[7] - LOWER_BOUND},
        {"type":"ineq", "fun": lambda x: x[8] - LOWER_BOUND},
        {"type":"ineq", "fun": lambda x: x[9] - LOWER_BOUND},
        {"type":"ineq", "fun": lambda x: x[10] - LOWER_BOUND},
        {"type":"ineq", "fun": lambda x: x[11] - LOWER_BOUND},
        {"type":"ineq", "fun": lambda x: x[12] - LOWER_BOUND},
        {"type":"ineq", "fun": lambda x: x[13] - LOWER_BOUND},
        {"type":"ineq", "fun": lambda x: x[14] - LOWER_BOUND},
        {"type":"ineq", "fun": lambda x: x[15] - LOWER_BOUND},
        {"type":"ineq", "fun": lambda x: x[16] - LOWER_BOUND},
        {"type":"ineq", "fun": lambda x: x[17] - LOWER_BOUND},
        {"type":"ineq", "fun": lambda x: x[18] - LOWER_BOUND},
        {"type":"ineq", "fun": lambda x: x[19] - LOWER_BOUND},
        #Upper_Bound
       {"type":"ineq", "fun": lambda x: UPPER_BOUND - x[0]},
        {"type":"ineq", "fun": lambda x: UPPER_BOUND - x[1]},
        {"type":"ineq", "fun": lambda x: UPPER_BOUND - x[2]},
        {"type":"ineq", "fun": lambda x: UPPER_BOUND - x[3]},
        {"type":"ineq", "fun": lambda x: UPPER_BOUND - x[4]},
        {"type":"ineq", "fun": lambda x: UPPER_BOUND - x[5]},
        {"type":"ineq", "fun": lambda x: UPPER_BOUND - x[6]},
        {"type":"ineq", "fun": lambda x: UPPER_BOUND - x[7]},
        {"type":"ineq", "fun": lambda x: UPPER_BOUND - x[8]},
        {"type":"ineq", "fun": lambda x: UPPER_BOUND - x[9]},
        {"type":"ineq", "fun": lambda x: UPPER_BOUND - x[10]},
        {"type":"ineq", "fun": lambda x: UPPER_BOUND - x[11]},
        {"type":"ineq", "fun": lambda x: UPPER_BOUND - x[12]},
        {"type":"ineq", "fun": lambda x: UPPER_BOUND - x[13]},
        {"type":"ineq", "fun": lambda x: UPPER_BOUND - x[14]},
        {"type":"ineq", "fun": lambda x: UPPER_BOUND - x[15]},
        {"type":"ineq", "fun": lambda x: UPPER_BOUND - x[16]},
        {"type":"ineq", "fun": lambda x: UPPER_BOUND - x[17]},
        {"type":"ineq", "fun": lambda x: UPPER_BOUND - x[18]},
        {"type":"ineq", "fun": lambda x: UPPER_BOUND - x[19]},
        #Moderate
        {"type":"ineq", "fun": lambda x: 1 - x[2]},
        {"type":"ineq", "fun": lambda x: 1 - x[7]},
        {"type":"ineq", "fun": lambda x: 1 - x[12]},
        {"type":"ineq", "fun": lambda x: 1 - x[17]},
        {"type":"ineq", "fun": lambda x: -1 + x[2]},
        {"type":"ineq", "fun": lambda x: -1 + x[7]},
        {"type":"ineq", "fun": lambda x: -1 + x[12]},
        {"type":"ineq", "fun": lambda x: -1 + x[17]},
        #MIN_INCREMENT
        {"type":"ineq", "fun": lambda x: x[1]*(1-MIN_INCREMENT) - x[0]},
        {"type":"ineq", "fun": lambda x: x[6]*(1-MIN_INCREMENT) - x[5]},
        {"type":"ineq", "fun": lambda x: x[11]*(1-MIN_INCREMENT) - x[10]},
        {"type":"ineq", "fun": lambda x: x[16]*(1-MIN_INCREMENT) - x[15]},
        {"type":"ineq", "fun": lambda x: x[2]*(1-MIN_INCREMENT) - x[1]},
        {"type":"ineq", "fun": lambda x: x[7]*(1-MIN_INCREMENT) - x[6]},
        {"type":"ineq", "fun": lambda x: x[12]*(1-MIN_INCREMENT) - x[11]},
        {"type":"ineq", "fun": lambda x: x[17]*(1-MIN_INCREMENT) - x[16]},
        {"type":"ineq", "fun": lambda x: x[3] - x[2]*(1+MIN_INCREMENT)},
        {"type":"ineq", "fun": lambda x: x[8] - x[7]*(1+MIN_INCREMENT)},
        {"type":"ineq", "fun": lambda x: x[13] - x[12]*(1+MIN_INCREMENT)},
        {"type":"ineq", "fun": lambda x: x[18] - x[17]*(1+MIN_INCREMENT)},
        {"type":"ineq", "fun": lambda x: x[4] - x[3]*(1+MIN_INCREMENT)},
        {"type":"ineq", "fun": lambda x: x[9] - x[8]*(1+MIN_INCREMENT)},
        {"type":"ineq", "fun": lambda x: x[14] - x[13]*(1+MIN_INCREMENT)},
        {"type":"ineq", "fun": lambda x: x[19] - x[18]*(1+MIN_INCREMENT)},
    )
else:
    cons = (
        #Low <= ModLow
        {"type":"ineq", "fun": lambda x: x[1] - x[0]},
        {"type":"ineq", "fun": lambda x: x[6] - x[5]},
        {"type":"ineq", "fun": lambda x: x[11] - x[10]},
        {"type":"ineq", "fun": lambda x: x[15] - x[16]},
        #ModLow <= Mod
        {"type":"ineq", "fun": lambda x: x[2] - x[1]},
        {"type":"ineq", "fun": lambda x: x[7] - x[6]},
        {"type":"ineq", "fun": lambda x: x[12] - x[11]},
        {"type":"ineq", "fun": lambda x: x[16] - x[17]},
        #Mod <= ModHigh
        {"type":"ineq", "fun": lambda x: x[3] - x[2]},
        {"type":"ineq", "fun": lambda x: x[8] - x[7]},
        {"type":"ineq", "fun": lambda x: x[13] - x[12]},
        {"type":"ineq", "fun": lambda x: x[17] - x[18]},
        #ModHigh <= High
        {"type":"ineq", "fun": lambda x: x[4] - x[3]},
        {"type":"ineq", "fun": lambda x: x[9] - x[8]},
        {"type":"ineq", "fun": lambda x: x[14] - x[13]},
        {"type":"ineq", "fun": lambda x: x[18] - x[19]},
        #COST_FLOOR
        {"type":"ineq", "fun": lambda x: x[0] - COST_FLOOR},
        {"type":"ineq", "fun": lambda x: x[1] - COST_FLOOR},
        {"type":"ineq", "fun": lambda x: x[2] - COST_FLOOR},
        {"type":"ineq", "fun": lambda x: x[3] - COST_FLOOR},
        {"type":"ineq", "fun": lambda x: x[4] - COST_FLOOR},
        {"type":"ineq", "fun": lambda x: x[5] - COST_FLOOR},
        {"type":"ineq", "fun": lambda x: x[6] - COST_FLOOR},
        {"type":"ineq", "fun": lambda x: x[7] - COST_FLOOR},
        {"type":"ineq", "fun": lambda x: x[8] - COST_FLOOR},
        {"type":"ineq", "fun": lambda x: x[9] - COST_FLOOR},
        {"type":"ineq", "fun": lambda x: x[10] - COST_FLOOR},
        {"type":"ineq", "fun": lambda x: x[11] - COST_FLOOR},
        {"type":"ineq", "fun": lambda x: x[12] - COST_FLOOR},
        {"type":"ineq", "fun": lambda x: x[13] - COST_FLOOR},
        {"type":"ineq", "fun": lambda x: x[14] - COST_FLOOR},
        #Lower_Bound
        {"type":"ineq", "fun": lambda x: x[0] - LOWER_BOUND},
        {"type":"ineq", "fun": lambda x: x[1] - LOWER_BOUND},
        {"type":"ineq", "fun": lambda x: x[2] - LOWER_BOUND},
        {"type":"ineq", "fun": lambda x: x[3] - LOWER_BOUND},
        {"type":"ineq", "fun": lambda x: x[4] - LOWER_BOUND},
        {"type":"ineq", "fun": lambda x: x[5] - LOWER_BOUND},
        {"type":"ineq", "fun": lambda x: x[6] - LOWER_BOUND},
        {"type":"ineq", "fun": lambda x: x[7] - LOWER_BOUND},
        {"type":"ineq", "fun": lambda x: x[8] - LOWER_BOUND},
        {"type":"ineq", "fun": lambda x: x[9] - LOWER_BOUND},
        {"type":"ineq", "fun": lambda x: x[10] - LOWER_BOUND},
        {"type":"ineq", "fun": lambda x: x[11] - LOWER_BOUND},
        {"type":"ineq", "fun": lambda x: x[12] - LOWER_BOUND},
        {"type":"ineq", "fun": lambda x: x[13] - LOWER_BOUND},
        {"type":"ineq", "fun": lambda x: x[14] - LOWER_BOUND},
        {"type":"ineq", "fun": lambda x: x[15] - LOWER_BOUND},
        {"type":"ineq", "fun": lambda x: x[16] - LOWER_BOUND},
        {"type":"ineq", "fun": lambda x: x[17] - LOWER_BOUND},
        {"type":"ineq", "fun": lambda x: x[18] - LOWER_BOUND},
        {"type":"ineq", "fun": lambda x: x[19] - LOWER_BOUND},
        #Upper_Bound
        {"type":"ineq", "fun": lambda x: UPPER_BOUND - x[0]},
        {"type":"ineq", "fun": lambda x: UPPER_BOUND - x[1]},
        {"type":"ineq", "fun": lambda x: UPPER_BOUND - x[2]},
        {"type":"ineq", "fun": lambda x: UPPER_BOUND - x[3]},
        {"type":"ineq", "fun": lambda x: UPPER_BOUND - x[4]},
        {"type":"ineq", "fun": lambda x: UPPER_BOUND - x[5]},
        {"type":"ineq", "fun": lambda x: UPPER_BOUND - x[6]},
        {"type":"ineq", "fun": lambda x: UPPER_BOUND - x[7]},
        {"type":"ineq", "fun": lambda x: UPPER_BOUND - x[8]},
        {"type":"ineq", "fun": lambda x: UPPER_BOUND - x[9]},
        {"type":"ineq", "fun": lambda x: UPPER_BOUND - x[10]},
        {"type":"ineq", "fun": lambda x: UPPER_BOUND - x[11]},
        {"type":"ineq", "fun": lambda x: UPPER_BOUND - x[12]},
        {"type":"ineq", "fun": lambda x: UPPER_BOUND - x[13]},
        {"type":"ineq", "fun": lambda x: UPPER_BOUND - x[14]},
        {"type":"ineq", "fun": lambda x: UPPER_BOUND - x[15]},
        {"type":"ineq", "fun": lambda x: UPPER_BOUND - x[16]},
        {"type":"ineq", "fun": lambda x: UPPER_BOUND - x[17]},
        {"type":"ineq", "fun": lambda x: UPPER_BOUND - x[18]},
        {"type":"ineq", "fun": lambda x: UPPER_BOUND - x[19]},
        #Moderate
        {"type":"ineq", "fun": lambda x: 1 - x[2]},
        {"type":"ineq", "fun": lambda x: 1 - x[7]},
        {"type":"ineq", "fun": lambda x: 1 - x[12]},
        {"type":"ineq", "fun": lambda x: 1 - x[17]},
        {"type":"ineq", "fun": lambda x: -1 + x[2]},
        {"type":"ineq", "fun": lambda x: -1 + x[7]},
        {"type":"ineq", "fun": lambda x: -1 + x[12]},
        {"type":"ineq", "fun": lambda x: -1 + x[17]},
        #MIN_INCREMENT
        {"type":"ineq", "fun": lambda x: x[1]*(1-MIN_INCREMENT) - x[0]},
        {"type":"ineq", "fun": lambda x: x[6]*(1-MIN_INCREMENT) - x[5]},
        {"type":"ineq", "fun": lambda x: x[11]*(1-MIN_INCREMENT) - x[10]},
        {"type":"ineq", "fun": lambda x: x[15] - x[16]*(1+MIN_INCREMENT)},
        {"type":"ineq", "fun": lambda x: x[2]*(1-MIN_INCREMENT) - x[1]},
        {"type":"ineq", "fun": lambda x: x[7]*(1-MIN_INCREMENT) - x[6]},
        {"type":"ineq", "fun": lambda x: x[12]*(1-MIN_INCREMENT) - x[11]},
        {"type":"ineq", "fun": lambda x: x[16] - x[17]*(1+MIN_INCREMENT)},
        {"type":"ineq", "fun": lambda x: x[3] - x[2]*(1+MIN_INCREMENT)},
        {"type":"ineq", "fun": lambda x: x[8] - x[7]*(1+MIN_INCREMENT)},
        {"type":"ineq", "fun": lambda x: x[13] - x[12]*(1+MIN_INCREMENT)},
        {"type":"ineq", "fun": lambda x: x[17]*(1-MIN_INCREMENT) - x[18]},
        {"type":"ineq", "fun": lambda x: x[4] - x[3]*(1+MIN_INCREMENT)},
        {"type":"ineq", "fun": lambda x: x[9] - x[8]*(1+MIN_INCREMENT)},
        {"type":"ineq", "fun": lambda x: x[14] - x[13]*(1+MIN_INCREMENT)},
        {"type":"ineq", "fun": lambda x: x[18]*(1-MIN_INCREMENT) - x[19]},
    )
#%%
ans = minimize(error_sensitivities, [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], args=(RFPs, heatmap, calibration_points, alpha_opt, master_scale),constraints=cons, method="COBYLA").x

df_ans = pd.DataFrame({
        'Value':['direct_cost', 'indirect_cost', 'capital_expenditure', 'revenue'],
        'No impact':[0,0,0,0],
        'Low':[ans[0], ans[5], ans[10],ans[15]],
        'Mod low':[ans[1], ans[6], ans[11],ans[16]],
        'Moderate':[ans[2], ans[7], ans[12],ans[17]],
        'Mod high':[ans[3], ans[8], ans[13],ans[18]],
        'High':[ans[4], ans[9], ans[14],ans[19]]
    })
#%%
df_ans.index = df_ans["Value"]
#%%
df_decayC = calc_decay(RFPs,heatmap, df_ans)
#%%
df_ans = pd.DataFrame(columns= ['pd', 'lgd', 'ead', 'id', 'year','subsector'])
#%%
for i in range(len(portfolio)):
    try:
        ttc_rating = portfolio.loc[i, 'ttc_rating']
        if ttc_rating < master_scale['Lower Rating'].min() or ttc_rating > master_scale['Upper Rating'].max():
            raise ValueError(f"ttc_rating out of valid range: {ttc_rating}")
        
        outstanding = portfolio.loc[i,'outstanding']
        start_pd = get_pd_from_ttc(portfolio.loc[i,'ttc_rating'], master_scale)
        subsector = portfolio.loc[i,'subsector']
        client_id = portfolio.loc[i,'id']
        start_lgd = portfolio.loc[i,'lgd']
        
        #Retrieve the decay value for the start year
        decay_row = df_decayC[(df_decayC['subsector'] == subsector) & (df_decayC['year'] == START_YEAR)]
        if decay_row.empty:
            print(f"No decay data found for subsector: {subsector}, year: {START_YEAR}")
            continue
        pd_ref = norm.cdf(norm.ppf(start_pd) - decay_row['decay_value'].values[0] / alpha_opt)

        for j in range(START_YEAR,END_YEAR + 1):
            decay_row = df_decayC[(df_decayC['subsector'] == subsector) & (df_decayC['year'] == j)]
            if decay_row.empty:
                print(f"No decay data found for subsector: {subsector}, year: {j}")
                continue
            PD = norm.cdf(norm.ppf(start_pd) - decay_row['decay_value'].values[0] / alpha_opt)
            LGD = norm.cdf(norm.ppf(PD) - norm.ppf(pd_ref)+norm.ppf(pd_ref*start_lgd))/PD

            df_ans.loc[len(df_ans)] = [PD,LGD,outstanding,client_id,j,subsector]

    except ValueError as e:
        print(f"Error at index {i}: {e}")
    
#%%Formatting of Output
new_order = ["id", "year", "subsector", "ead", "pd", "lgd"]
df_output = df_ans[new_order]

df_output.columns = [col.upper() for col in df_output.columns]

print(df_output)

#Get current directory
current_directory = os.getcwd()
#File path for Excel file
file_path = os.path.join(current_directory, "CIB_Output_Transition_.xlsx") #
df_output.to_excel(file_path, index = False)

print("Excel file saved at:",file_path)
