import pandas as pd
from wave_conditions.trim_jpd import *

def parameters():
    ksi2pa = 6894757  # 1000 pounds/in2 to Pascal
    in2m = 0.0254  # inch to meter
    yd2m = 0.9144  # yard to meter
    lb2kg = 1/2.2  # pound to kilogram

    file = 'F:/study/project/WEC-DECIDER/Humboldt_California_Wave Resource _SAM CSV.csv'
    jpd = pd.read_csv(file, skiprows=2, header=None).values
    trimmed_jpd = trim_jpd(jpd)

    p = {
        'rho_w': 1000,  # water density (kg/m3)
        'g': 9.8,  # acceleration of gravity (m/s2)
        'JPD': trimmed_jpd[1:, 1:],  # joint probability distribution of wave (%)
        'Hs': trimmed_jpd[1:, 0],  # wave height (m)
        'Hs_struct': 11.9,  # 100 year wave height (m)
        'T': trimmed_jpd[0, 1:],  # wave period (s)
        'T_struct': 17.1,  # 100 year wave period (s)
        'sigma_y': np.array([36, 4.5, 30]) * ksi2pa,  # yield strength (Pa)
        'rho_m': np.array([8000, 2400, 8000]),  # material density (kg/m3)
        'E': np.array([200e9, 5000 * np.sqrt(4.5 * ksi2pa), 200e9]),  # young's modulus (Pa)
        'cost_m': np.array([4.28, 125 / yd2m**3 / 2400, 1.84 / lb2kg]),  # material cost ($/kg)
        'm_scale': 1.25,  # factor to account for mass of neglected stiffeners (-)
        't_ft': 0.50 * in2m,  # float top thickness (m)
        't_fr': 0.44 * in2m,  # float radial wall thickness (m)
        't_fc': 0.44 * in2m,  # float circumferential gusset thickness (m)
        't_fb': 0.56 * in2m,  # float bottom thickness (m)
        't_sr': 1.00 * in2m,  # vertical column thickness (m)
        't_dt': 1.00 * in2m,  # damping plate support tube radial wall thickness (m)
        'D_dt': 48.00 * in2m,  # damping plate support tube diameter (m)
        'theta_dt': np.arctan(17.5/15),  # angle from horizontal of damping plate support tubes (rad)
        'FOS_min': 1.5,  # minimum FOS (-)
        'D_d_min': 30,  # minimum damping plate diameter
        'FCR': 0.113,  # fixed charge rate (-)
        'N_WEC': 100,  # number of WECs in array (-)
        'D_d_over_D_s': 30/6,  # normalized damping plate diameter (-)
        'T_s_over_D_s': 35/6,  # normalized spar draft (-)
        'h_d_over_D_s': 1 * in2m / 6,  # normalized damping plate thickness (-)
        'T_f_over_h_f': 2/4,  # normalized float draft (-)
        'LCOE_max': .5,  # maximum LCOE ($/kWh)
        'power_max': float('inf'),  # maximum power (W)
        'eff_pto': 0.80,  # PTO efficiency (-)
        'eff_array': 0.95 * 0.98  # array availability and transmission efficiency (-)
    }

    return p

