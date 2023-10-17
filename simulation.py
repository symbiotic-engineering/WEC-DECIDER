from modules.structure import *
from modules.econ import *
from modules.geometry import *
from modules.dynamics_python.dynamics import *


def simulation(X, p):
    X = np.maximum(X, 1e-3)
    val = {}
    # Assemble inputs
    in_params = p.copy()
    in_params['D_f'] = X[0]
    D_s_over_D_f = X[1]
    h_f_over_D_f = X[2]
    T_s_over_h_s = X[3]
    in_params['F_max'] = X[4] * 1e6
    in_params['B_p'] = X[5] * 1e6
    in_params['w_n'] = X[6]
    in_params['M'] = X[7]

    # Variable ratios defined by design variables
    in_params['D_s'] = D_s_over_D_f * in_params['D_f']
    in_params['h_f'] = h_f_over_D_f * in_params['D_f']
    in_params['T_f'] = p['T_f_over_h_f'] * in_params['h_f']
    D_d = p['D_d_over_D_s'] * in_params['D_s']
    in_params['T_s'] = p['T_s_over_D_s'] * in_params['D_s']
    in_params['h_d'] = p['h_d_over_D_s'] * in_params['D_s']
    in_params['h_s'] = 1 / T_s_over_h_s * in_params['T_s']

    # Run modules

    V_d, m_m, m_f_tot, A_c, A_lat_sub, r_over_t, I, T, V_f_pct, V_s_pct, GM = geometry(
        in_params['D_s'], in_params['D_f'], in_params['T_f'], in_params['h_f'], in_params['h_s'],
        in_params['t_ft'], in_params['t_fr'], in_params['t_fc'], in_params['t_fb'], in_params['t_sr'],
        in_params['t_dt'], D_d, in_params['D_dt'], in_params['theta_dt'], in_params['T_s'], in_params['h_d'],
        in_params['M'], in_params['rho_m'], in_params['rho_w'], in_params['m_scale'])

    m_f_tot = max(m_f_tot, 1e-3)
    F_heave_max, F_surge_max, F_ptrain_max, P_var, P_elec, P_matrix, h_s_extra = dynamics(
        in_params, m_f_tot, V_d, T)

    FOS1Y, FOS2Y, FOS3Y, FOS_buckling = structures(
        F_heave_max, F_surge_max, in_params['M'], in_params['h_s'], in_params['T_s'],
        in_params['rho_w'], in_params['g'], in_params['sigma_y'], A_c, A_lat_sub, r_over_t, I, in_params['E'])

    LCOE = econ(m_m, in_params['M'], in_params['cost_m'], in_params['N_WEC'], P_elec,
                in_params['FCR'], in_params['eff_array'])

    # Assemble constraints g(x) >= 0
    g = np.zeros(14)
    g[0] = V_f_pct
    # ... (same structure for other g values)

    criteria = np.all(np.isfinite(g)) and np.all(~np.isnan(g)) and np.all(np.isreal(g))

    if not criteria:
        print("Warning: Inf, NaN, or imaginary constraint detected")

    # Compute additional outputs if required
    # ... (similar structure as above)

    return LCOE, P_var, P_matrix, g, val if 'val' in locals() else None
