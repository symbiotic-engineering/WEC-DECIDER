from modules.dynamics_python.dynamics_simple import  *
from modules.dynamics_python.get_abc_symbolic import *
from modules.dynamics_python.pick_which_root import *
def dynamics(in_params,m_float,V_d,draft):
    # Use probabilistic sea states for power
    T, Hs = np.meshgrid(in_params['T'], in_params['Hs'])
    P_matrix, h_s_extra, P_unsat = get_power_force(in_params, T, Hs, m_float, V_d, draft)

    # Account for powertrain electrical losses
    P_matrix *= in_params['eff_pto']

    # Saturate maximum power
    P_matrix = np.minimum(P_matrix, in_params['power_max'])

    # Weight power across all sea states
    P_weighted = P_matrix * in_params['JPD'] / 100
    P_elec = np.sum(P_weighted)

    assert np.isreal(P_elec)

    # Use max sea states for structural forces and max amplitude
    _, _, _, F_heave_max, F_surge_max, F_ptrain_max = get_power_force(
        in_params, in_params['T_struct'], in_params['Hs_struct'], m_float, V_d, draft)

    # Coefficient of variance (normalized standard deviation) of power
    P_var = np.std(P_matrix, ddof=0, weights=in_params['JPD']) / P_elec
    P_var *= 100  # Convert to percentage

    return F_heave_max, F_surge_max, F_ptrain_max, P_var, P_elec, P_matrix, h_s_extra, P_unsat


def get_power_force(in_params, T, Hs, m_float, V_d, draft):
    # Get unsaturated response
    w, A, B_h, K_h, Fd, k_wvn = dynamics_simple(Hs, T, in_params['D_f'], in_params['T_f'], in_params['rho_w'],
                                                in_params['g'])
    m = m_float + A
    b = B_h + in_params['B_p']
    k = in_params['w_n'] ** 2 * m
    K_p = k - K_h
    X_unsat = get_response(w, m, b, k, Fd)

    # Confirm unsaturated response doesn't exceed maximum capture width
    P_unsat = 0.5 * in_params['B_p'] * w ** 2 * X_unsat ** 2

    F_ptrain_over_x = np.sqrt((in_params['B_p'] * w) ** 2 + (K_p) ** 2)
    F_ptrain_unsat = F_ptrain_over_x * X_unsat

    # Get saturated response
    r = np.minimum(in_params['F_max'] / F_ptrain_unsat, 1)
    alpha = (2 / np.pi) * (1 / r * np.arcsin(r) + np.sqrt(1 - r ** 2))
    f_sat = alpha * r
    mult = get_multiplier(f_sat, m, b, k, w, b / in_params['B_p'], k / K_p)
    b_sat = B_h + mult * in_params['B_p']
    k_sat = K_h + mult * K_p
    X_sat = get_response(w, m, b_sat, k_sat, Fd)

    # Calculate power
    P_matrix = 0.5 * (mult * in_params['B_p']) * w ** 2 * X_sat ** 2

    X_max = np.max(X_sat)
    h_s_extra = (in_params['h_s'] - in_params['T_s'] - (in_params['h_f'] - in_params['T_f']) - X_max) / in_params['h_s']

    # Calculate forces
    F_ptrain = mult * F_ptrain_over_x * X_sat
    F_ptrain_max = np.max(F_ptrain)
    F_err_1 = np.abs(F_ptrain / (in_params['F_max'] * alpha) - 1)
    F_err_2 = np.abs(F_ptrain / (f_sat * F_ptrain_unsat) - 1)

    # 0.1 percent error
    if np.any(f_sat < 1):
        assert np.all(F_err_1[f_sat < 1] < 1e-3)
    assert np.all(F_err_2 < 1e-3)

    F_heave_fund = np.sqrt((mult * in_params['B_p'] * w) ** 2 + (mult * K_p - m_float * w ** 2) ** 2) * X_sat
    F_heave = np.minimum(F_heave_fund, in_params['F_max'] + m_float * w ** 2 * X_sat)

    F_surge = np.max(Hs) * in_params['rho_w'] * in_params['g'] * V_d * (1 - np.exp(-np.max(k_wvn) * draft))

    return P_matrix, h_s_extra, P_unsat, F_heave, F_surge, F_ptrain_max


def get_response(w, m, b, k, Fd):
    imag_term = b * w
    real_term = k - m * w ** 2
    X_over_F_mag = 1 / np.sqrt(real_term ** 2 + imag_term ** 2)
    X = X_over_F_mag * Fd
    return X


def get_multiplier(f_sat, m, b, k, w, r_b, r_k):
    # m, k, and r_k are scalars.
    # All other inputs are 2D arrays, the dimension of the sea state matrix.

    # speedup: only do math for saturated sea states, since unsat will = 1

    idx_no_sat = f_sat == 1
    f_sat[idx_no_sat] = np.nan
    b[idx_no_sat] = np.nan
    w[idx_no_sat] = np.nan
    r_b[idx_no_sat] = np.nan
    a_quad, b_quad, c_quad = get_abc_symbolic(f_sat, m, b, k, w, r_b, r_k)
    # solve the quadratic formula
    determinant = np.sqrt(b_quad ** 2 - 4 * a_quad * c_quad)
    num = -b_quad + determinant
    # creating a second dimension to hold the second root value
    num = np.stack((num, -b_quad - determinant), axis=-1)
    den = 2 * a_quad
    roots = num / den
    # choose which of the two roots to use
    mult = pick_which_root(roots, idx_no_sat, a_quad, b_quad, c_quad)
    assert np.all(~np.isnan(mult))

    return mult
