import pandas as pd
import numpy as np

# verified
def nominal_c_v():

    filename = 'RM3-CBS.xlsx'  # Spreadsheet containing RM3 "actual" power data
    sheet = 'Performance & Economics'  # Name of relevant sheet

    # Read the power matrix and joint probability distribution (JPD) from the Excel file
    P_matrix = pd.read_excel(filename, sheet_name=sheet, usecols="E:S", skiprows=96, nrows=14) * 1000
    JPD = pd.read_excel(filename, sheet_name=sheet, usecols="E:S", skiprows=23, nrows=14)

    # Weighted power calculation
    P_weighted = P_matrix.to_numpy() * JPD.to_numpy() / np.sum(JPD.to_numpy())
    P_elec = np.sum(P_weighted)

    # Coefficient of variance (normalized standard deviation) of power
    P_var = np.sqrt(np.average((P_matrix.to_numpy().flatten() - P_elec) ** 2, weights=JPD.to_numpy().flatten())) / P_elec
    c_v = P_var * 100  # Convert to percentage

    return c_v


print(nominal_c_v())