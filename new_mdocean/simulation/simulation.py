import numpy as np
import warnings



# This function is required for find_nominal_inputs
def simulation(X, p):
    # Prevent negative numbers
    X = max(X, 1e-3);

    # Assemble inputs dictionary
    in_dict = p.copy()  # Start with base parameters

    # Direct parameter assignments (MATLAB: in.D_s = X(1))
    in_dict.update({
        'D_s': X[0],  # Inner diameter of float (m)
        'D_f': X[1],  # Outer diameter of float (m)
        'T_f_2': X[2],  # Draft of float (m)
        'h_s': X[3],  # Total height of spar (m)
        'h_fs_clear': X[4],  # Vertical clearance (m)
        'F_max': X[5] * 1e6,  # Max powertrain force (N)
        'P_max': X[6] * 1e3,  # Maximum power (W)
        't_f_b': X[7] * 1e-3,  # Float bottom thickness (m)
        't_s_r': X[8] * 1e-3,  # Vertical column thickness (m)
        't_d': X[9] * 1e-3,  # Damping plate thickness (m)
        'h_stiff_f': X[10],  # Float stiffener height (m)
        'h1_stiff_d': X[11],  # Damping plate stiffener height (m)
        'M': X[12]  # Material index
    })

    # Derived parameters (MATLAB ratio calculations)
    in_dict['t_f_r'] = in_dict['t_f_b'] * p['t_f_r_over_t_f_b']
    in_dict['t_f_c'] = in_dict['t_f_b'] * p['t_f_c_over_t_f_b']
    in_dict['t_f_t'] = in_dict['t_f_b'] * p['t_f_t_over_t_f_b']

    in_dict['w_stiff_f'] = p['w_over_h_stiff_f'] * in_dict['h_stiff_f']
    in_dict['h_stiff_d'] = p['h_over_h1_stiff_d'] * in_dict['h1_stiff_d']
    in_dict['w_stiff_d'] = p['w_over_h1_stiff_d'] * in_dict['h1_stiff_d']

    # Geometric similarity calculations
    in_dict['D_d'] = p['D_d_over_D_s'] * in_dict['D_s']
    in_dict['T_s'] = p['T_s_over_D_s'] * in_dict['D_s']
    in_dict['h_d'] = p['h_d_over_D_s'] * in_dict['D_s']

    in_dict['h_f'] = in_dict['T_f_2'] / p['T_f_2_over_h_f']
    in_dict['T_f_1'] = p['T_f_1_over_T_f_2'] * in_dict['T_f_2']
    in_dict['D_f_b'] = p['D_f_b_over_D_f'] * in_dict['D_f']

    # Ensure valid diameter
    D_f_in = p['D_f_in_over_D_s'] * in_dict['D_s']
    in_dict['D_f_in'] = min(D_f_in, in_dict['D_f'] - 1e-6)

    # Geometry module
    geom_output = geometry(
        in_dict['D_s'], in_dict['D_f'], in_dict['D_f_in'], in_dict['D_f_b'],
        in_dict['T_f_1'], in_dict['T_f_2'], in_dict['h_f'], in_dict['h_s'],
        in_dict['h_fs_clear'], in_dict['D_f_tu'], in_dict['t_f_t'],
        in_dict['t_f_r'], in_dict['t_f_c'], in_dict['t_f_b'], in_dict['t_f_tu'],
        in_dict['t_s_r'], in_dict['t_d_tu'], in_dict['D_d'], in_dict['D_d_tu'],
        in_dict['theta_d_tu'], in_dict['T_s'], in_dict['h_d'], in_dict['t_d'],
        in_dict['h_stiff_f'], in_dict['w_stiff_f'], in_dict['num_sections_f'],
        in_dict['h_stiff_d'], in_dict['w_stiff_d'], in_dict['num_stiff_d'],
        in_dict['M'], in_dict['rho_m'], in_dict['rho_w'], in_dict['m_scale']
    )

    V_d, m_m, m_f_tot, m_s_tot, A_c, A_lat_sub, I, T, V_f_pct, V_s_pct, GM, A_dt, L_dt = geom_output

    # Dynamics module
    dyn_output = dynamics(in_dict, m_f_tot, m_s_tot, V_d, T)
    (F_heave_storm, F_surge_storm, F_heave_op, F_surge_op,
     F_ptrain_max, P_var, P_avg_elec, P_matrix_elec,
     X_constraints, *_) = dyn_output  # Simplified unpacking

    # Structures module
    struct_output = structures(
        F_heave_storm, F_surge_storm, F_heave_op, F_surge_op,
        in_dict['h_s'], in_dict['T_s'], in_dict['D_s'], in_dict['D_f'],
        in_dict['D_f_in'], in_dict['num_sections_f'], in_dict['D_f_tu'],
        in_dict['D_d'], L_dt, in_dict['theta_d_tu'], in_dict['D_d_tu'],
        in_dict['t_s_r'], I, A_c, A_lat_sub, in_dict['t_f_b'], in_dict['t_f_t'],
        in_dict['t_d'], in_dict['t_d_tu'], in_dict['h_d'], A_dt,
        in_dict['h_stiff_f'], in_dict['w_stiff_f'], in_dict['h_stiff_d'],
        in_dict['w_stiff_d'], in_dict['M'], in_dict['rho_w'], in_dict['g'],
        in_dict['sigma_y'], in_dict['sigma_e'], in_dict['E'], in_dict['nu'],
        in_dict['num_terms_plate'], in_dict['radial_mesh_plate'], in_dict['num_stiff_d']
    )
    FOS_float, FOS_spar, FOS_damping_plate, FOS_spar_local = struct_output


    # Economics module
    econ_output = econ(
        m_m, in_dict['M'], in_dict['cost_perkg_mult'], in_dict['N_WEC'],
        P_avg_elec, in_dict['FCR'], in_dict['cost_perN_mult'],
        in_dict['cost_perW_mult'], in_dict['F_max'], in_dict['P_max'],
        in_dict['eff_array']
    )
    LCOE, capex_design = econ_output
    J_capex_design = capex_design / 1e6  # Convert to $M

    # Assemble constraints
    num_g = 20 + len(p['JPD']) + len(p['T_struct'])
    g = np.zeros(num_g)
    g[0] = V_f_pct  # Float too heavy
    g[1] = 1 - V_f_pct  # Float too light
    g[2] = V_s_pct  # Spar too heavy
    g[3] = 1 - V_s_pct  # Spar too light
    g[4] = GM  # Pitch stability
    g[5] = FOS_float[0] / p['FOS_min'] - 1  # Float max force
    g[6] = FOS_float[1] / p['FOS_min'] - 1  # Float fatigue
    g[7] = FOS_spar[0] / p['FOS_min'] - 1  # Spar max force
    g[8] = FOS_spar[1] / p['FOS_min'] - 1  # Spar fatigue
    g[9] = FOS_damping_plate[0] * in_dict['FOS_mult_d'] / p['FOS_min'] - 1
    g[10] = FOS_damping_plate[1] * in_dict['FOS_mult_d'] / p['FOS_min'] - 1
    g[11] = FOS_spar_local[0] / p['FOS_min'] - 1
    g[12] = FOS_spar_local[1] / p['FOS_min'] - 1
    g[13] = P_avg_elec / 1e6  # Positive power
    g[14] = p['LCOE_max'] / LCOE - 1  # Cost threshold
    g[15] = F_ptrain_max / in_dict['F_max'] - 1
    g[16:20] = X_constraints[:4]
    g[20:] = X_constraints[4:]

    # Validity checks
    all_values = np.concatenate([g, [LCOE, P_var]])
    if not (np.all(np.isfinite(all_values)) and
            np.all(np.isreal(all_values))):
        warnings.warn('Inf, NaN, or imaginary constraint/objective detected')

    # Validation data (if requested)
    val = None
    # (Implementation would require recalling modules with extended outputs)

    return LCOE, J_capex_design, P_matrix_elec, g, val








