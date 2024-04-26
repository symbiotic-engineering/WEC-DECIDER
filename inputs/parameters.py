import pandas as pd
import gridstatus
import requests
import io
from inputs.wave_conditions.trim_jpd import *

def parameters():
    ksi2pa = 6894757  # 1000 pounds/in2 to Pascal
    in2m = 0.0254  # inch to meter
    yd2m = 0.9144  # yard to meter
    lb2kg = 1/2.2  # pound to kilogram
    MJ2kg = 0.5588 # Megajoule heating to kilogram steel
    kg2msq = 0.285095 # kilogram to meter squared fiberglass
    kg2mi = 61.66 # kilogram diesel to miles traveled

    s_1_kg = 0.018301799 # drilling steel eco cost (euro/kg)
    s_2_kg = 0.018301799 # milling steel eco cost (euro/kg)
    s_3_kg = 0.1639439696582 # rolling steel eco cost (euro/kg)
    s_4_kg = 0.026435889 # sorting steel eco cost (euro/kg)
    s_5_kg = -0.060109785 # recyling steel eco cost (euro/kg)
    s_6_MJ = 0.017077496 # melting steel eco cost (euro/MJ)
    s_6_kg = 0.017077496 * MJ2kg # melting steel eco cost (euro/kg)

    f_1_kg = 0.280160595 # processing glass fiber eco cost (euro/kg)
    f_2_kg = 0.091234532 # incineration fiberglass eco cost (euro/kg)
    f_3_kg = 1.044916856 # processing epoxy resin eco cost (euro/kg)
    fglayers = 15.79 # scaling to 3mm fiberglass

    d_1_kg = 0.980234899 # diesel eco cost (euro/kg)

    file = '/Users/jiaruiyang/Documents/GitHub/WEC-DECIDER/inputs/wave_conditions/Humboldt_California_Wave Resource _SAM CSV.csv'
    jpd = pd.read_csv(file, skiprows=2, header=None).values
    trimmed_jpd = trim_jpd(jpd)   
    
    caiso = gridstatus.CAISO()
    start = pd.Timestamp("Jan 1, 2021").normalize()
    end = pd.Timestamp("Dec 31, 2021").normalize()
    lmp = caiso.get_lmp(start=start, end=end, market='REAL_TIME_HOURLY', 
        locations=["EUREKAA_6_N001"])
    lmp.index = pd.to_datetime(lmp.index,utc=True).tz_convert('US/Pacific') 
    
    # get wave power data - to be replaced with mhkit in the future
    url = 'https://raw.githubusercontent.com/NREL/SAM/develop/deploy/wave_resource_ts/lat40.84_lon-124.25__2010.csv'
    download = requests.get(url).content
    file = io.StringIO(download.decode('utf-8'))
    parser = lambda y,m,d,H,M: pd.datetime.strptime(f"{y}.{m}.{d}.{H}.{M}", "%Y.%m.%d.%H.%M")
    wave_data = pd.read_csv(file, skiprows = 2, parse_dates={"Time":[0,1,2,3,4]}, date_parser=parser)
    wave_data = wave_data[['Time','Significant Wave Height','Energy Period']].set_index("Time")
    wave_data.index = wave_data.index.tz_localize('US/Pacific') + pd.offsets.DateOffset(years=11) # fake it starting in 2021
    
    p = {
        'rho_w': 1000.0,  # water density (kg/m3)
        'g': 9.8,  # acceleration of gravity (m/s2)
        'JPD': trimmed_jpd[1:, 1:].astype(float),  # joint probability distribution of wave (%)
        'Hs': trimmed_jpd[1:, 0].astype(float),  # wave height (m)
        'Hs_struct': np.array([11.9]),  # 100 year wave height (m)
        'T': trimmed_jpd[0, 1:].astype(float),  # wave period (s)
        'T_struct': np.array([17.1]),  # 100 year wave period (s)
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
        'D_d_min': 30.0,  # minimum damping plate diameter
        'FCR': 0.113,  # fixed charge rate (-)
        'N_WEC': 100,  # number of WECs in array (-)
        'D_d_over_D_s': 30/6,  # normalized damping plate diameter (-)
        'T_s_over_D_s': 35/6,  # normalized spar draft (-)
        'h_d_over_D_s': 1 * in2m / 6,  # normalized damping plate thickness (-)
        'T_f_over_h_f': 2/4,  # normalized float draft (-)
        'LCOE_max': .5,  # maximum LCOE ($/kWh)
        'power_max': float('inf'),  # maximum power (W)
        'eff_pto': 0.80,  # PTO efficiency (-)
        'eff_array': 0.95 * 0.98,  # array availability and transmission efficiency (-)
        's_points': s_1_kg + s_2_kg + s_3_kg + s_4_kg + s_5_kg + s_6_kg, # steel eco cost (euros/kg)
        'f_points': (f_1_kg + f_2_kg + f_3_kg) * kg2msq * fglayers, # fiberglass eco cost (euro/m^2)
        'd_points': d_1_kg * kg2mi, # travel eco cost (euro/mi)
        'SCC': 0.133, # social cost of carbon (euros/kg CO2)
        'Year' : 2030,
        'Location' : float('NE'),
        'Demand_Scenario' : 2, #1 is low, 2 is moderate, 3 is high
        'Carbon_Constraint' : 1, # 1 is on
        'LMP': lmp,
        'wave_data' : wave_data      
    }

    return p
    

    

