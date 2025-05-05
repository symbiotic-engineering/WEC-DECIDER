import pandas as pd
from wave_conditions.trim_jpd import *
from new_mdocean.inputs.validation.get_spar_exc import get_spar_exc, get_hydro
import math

def parameters(mode:str=""):
    ksi2pa = 6894757  # 1000 pounds/in2 to Pascal
    in2m = 0.0254  # inch to meter
    yd2m = 0.9144  # yard to meter
    lb2kg = 1/2.2  # pound to kilogram

    if mode == 'wecsim':
        T_s_over_D_s = 29/6
        h_d_over_D_s = 0.1 / 6
        T_f_2_over_h_f = 3 / 5
        D_f_b_over_D_f = 10 / 20
        T_f_1_over_T_f_2 = 2 / 3
        D_f_in_over_D_s = 6 / 6
        h = 250 # must be above .4*max(jpd_Te)^2*g/(2*pi) = 213.5 to be all deep water
        power_coeffs = [1,0,0,1]
        power_scale_multibody = 1

    else:
        T_s_over_D_s = 35/6
        h_d_over_D_s = 1 * in2m / 6
        T_f_2_over_h_f = 3.2 / 5.2
        D_f_b_over_D_f = 6.5 / 20
        T_f_1_over_T_f_2 = 2 / 3.2
        D_f_in_over_D_s = 6.5 / 6
        h = 45
        power_coeffs = [22.4, 1, -15, 86]
        power_scale_multibody = 0.595

    # nrows switched from 15 to 14 to match the matrix. - Jordan
    mat = pd.read_excel('inputs/validation/RM3-CBS.xlsx', sheet_name='Performance & Economics', usecols='D:S', skiprows=21, nrows=15)
    mat = mat.to_numpy()
    mat[1, -1] = np.finfo(float).eps
    jpd_full = trim_jpd(mat)
    jpd_Hs = jpd_full[1:, 0]  # Rows 2:end, column 1 (Python is 0-indexed)
    jpd = jpd_full[1:, 1:] /(np.sum(jpd_full[1:,1:])) *100  # Rows 2:end, columns 2:end
    jpd_Te = jpd_full[1, 2:]  # Row 1, columns 2:end

    spar_exc = get_spar_exc(g = 9.8)

    # To do, output T into a table
    T = {
        'rho_w' : 1000,
        'g' : 9.8,
        'h' : 45,
        'JPD' : jpd,
        'Hs': jpd_Hs,
        'T': jpd_Te,
        'Hs_struct': np.array([5, 7, 9, 11.22, 9, 7, 5]) * 1.9 * np.sqrt(2),
        'T_struct': np.array([5.57, 8.76, 12.18, 17.26, 21.09, 24.92, 31.70]),
        'sigma_y':  np.array([36, 4.5, 30]) * ksi2pa,
        'sigma_e': np.array([58 * 0.45, 0, 75 * 0.45]) * ksi2pa,
        'rho_m': np.array([7850, 2400, 7900]),
        'E': np.array([200e9, 5000 * np.sqrt(4.5 * ksi2pa),200e9]),
        'cost_perkg_mult': np.array([
            4.28,
            125 / yd2m ** 3 / 2400,
            1.84 / lb2kg
        ]) / 4.28,
        'nu': np.array([0.36, 0, 0.29]),
        'FOS_min': 1.5,
        't_f_t_over_t_f_b': np.array([0.50 / 0.56]),
        't_f_r_over_t_f_b': np.array([0.44 / 0.56]),
        't_f_c_over_t_f_b': np.array([0.44 / 0.56]),
        'D_f_tu': np.array([20 * in2m]),
        't_f_tu': np.array([0.5 * in2m]),
        'w_over_h_stiff_f': np.array([1/16]),
        'num_sections_f': np.array([12]),
        't_d_tu': np.array([1.00 * in2m]),
        'D_d_tu': np.array([48.00 * in2m]),
        'theta_d_tu': np.array([math.atan(17.5 / 15)]),
        'h_over_h1_stiff_d': np.array([12.5, 0.5, 22, 1]) / 22,
        'w_over_h1_stiff_d': np.array([0.5, 10, 1, 12]) / 22,
        'FOS_mult_d': np.array([7.5]),
        'num_terms_plate': np.array([100]),
        'radial_mesh_plate': np.array([20]),
        'num_stiff_d': np.array([24]),

        # Economics
        'm_scale': np.array([1.1]),
        'FCR': np.array([0.113]),
        'N_WEC': np.array([100]),
        'LCOE_max': np.array([1.0]),
        'eff_array': np.array([0.95 * 0.98]),
        'cost_perN_mult': np.array([1.0]),
        'cost_perW_mult': np.array([1.0]),

        #Geometric ratios of bulk dimensions
        'D_d_min': np.array([30.0]),
        'D_d_over_D_s': np.array([30 / 6]),
        'T_s_over_D_s':  T_s_over_D_s,
        'h_d_over_D_s': h_d_over_D_s,
        'T_f_2_over_h_f': T_f_2_over_h_f,
        'T_f_1_over_T_f_2': T_f_1_over_T_f_2,
        'D_f_b_over_D_f': D_f_b_over_D_f,
        'D_f_in_over_D_s': D_f_in_over_D_s,

        # Dynamics: device parameters
        'C_d_float': np.array([1.0]),
        'C_d_spar': np.array([1.0]),
        'eff_pto': np.array([0.8]),
        'power_scale_coeffs': power_coeffs,
        'power_scale_multibody': power_scale_multibody,

        # Dynamics: simulation type,
        'control_type': 'damping',
        'use_MEEM': True,
        'use_multibody': True,

        # Dynamics: numerics and convergence
        'X_tol': np.array([1e-2]),
        'phase_X_tol': np.array([np.deg2rad(3)]),
        'max_drag_iters': np.array([40]),
        'harmonics': np.array([10]),
        'besseli_argmax': np.array([700.5]),

        # Dynamics: hydro coefficient data for nominal design

        'spar_excitation_coeffs': spar_exc,
        'hydro' : get_hydro(),
        'F_heave_mult' : np.array([0.98])
    }
    p = T.copy() # since T is also a dictionary.
    return p