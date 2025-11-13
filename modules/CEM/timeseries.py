import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
from mhkit import wave
import pandas as pd
import os
import yaml
import re
from pathlib import Path
from scipy.special import hankel1 as Hankel

# Determine root_dir
if "__file__" in globals():
    # Running as a script
    script_dir = os.path.dirname(os.path.abspath(__file__))
else:
    script_dir = os.getcwd()

# Traverse up to WEC-DECIDER root
root_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))  
cem_dir = os.path.join(root_dir, "modules", "CEM")

data_type = '3-hour'
year = [2010]
parameters = [
    'omni-directional_wave_power',
    'significant_wave_height',
    'energy_period',
    'directionality_coefficient',
    'maximum_energy_direction',
    'mean_absolute_period',
    'mean_zero-crossing_period',
    'peak_period',
    'spectral_width'
]
lat_lon = (43.5, -70) # off coast of Maine
data, metadata = wave.io.hindcast.hindcast.request_wpto_point_data(data_type, parameters, lat_lon, year)
data.head()

data_mod = data.copy()

rho_w = 1025  # kg/m^3, density of water
g = 9.81  # m/s^2, acceleration due to gravity
#J_calc = data_mod["significant_wave_height_0"]**2 * rho_w * g**2 / (64 * np.pi) * data_mod["energy_period_0"]
J_calc = data_mod["significant_wave_height_0"]**2 * rho_w * g**2 / (64 * np.pi) * data_mod["mean_zero-crossing_period_0"]
fudge = 1 - .007 * (data_mod["mean_zero-crossing_period_0"] - (2*np.pi)) 
data_mod["Power_density_predicted"] = J_calc / fudge #/ data_mod["spectral_width_0"] #* data_mod["directionality_coefficient_0"]
data_mod["ratio_power_density"] = data_mod["omni-directional_wave_power_0"] / data_mod["Power_density_predicted"]

data_mod.head()

#convert timeseries to matrix
Hs_hourly = np.interp(np.arange(0, 8760), np.arange(0, 8760, 3), data_mod[:,3])
T_hourly = np.interp(np.arange(0, 8760), np.arange(0, 8760, 3), data_mod[:,4])


def load_case_results(this_case_folder):
    #case_result_folder = os.path.join(this_case_folder,'results','results_p1')
    #carbon_file   = os.path.join(case_result_folder,'emissions.csv')
    #carbon_plant_file = os.path.join(case_result_folder,'emissions_plant.csv')

    carbon_df        = pd.read_csv(carbon_file,   index_col='Total') # units: tonnes CO2
    carbon_plant_df = pd.read_csv(carbon_plant_file, index_col='Total')
    
    carbon = carbon_df['Total'].loc['AnnualSum']
    carbon_plant = carbon_plant_df['Total'].loc['AnnualSum']

    outputs = [carbon, carbon_plant]
    return outputs