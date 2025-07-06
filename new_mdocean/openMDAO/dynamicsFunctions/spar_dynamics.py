import numpy as np
from scipy.interpolate import interp1d

def spar_dynamics(Ds_over_Dd, D_d, T_s, spar_coeffs, k, mult):
    """
    Compute added mass, excitation, and damping coefficients for a spar buoy.

    Parameters:
    Ds_over_Dd : float
        Ratio of spar diameter to draft diameter.
    D_d : float
        Draft diameter.
    T_s : float
        Spar submergence depth.
    spar_coeffs : dict-like
        Dictionary with keys: 'k', 'gamma_over_rho_g', 'gamma_phase',
        'A_c_over_rho', 'B_c_over_rho_w', and 'T_s'.
    k : ndarray
        Wave number array.
    mult : ndarray
        Finite-depth multiplier from group velocity calculation.

    Returns:
    A_s_over_rho, gamma_s_over_rho_g, B_s_over_rho_w,
    gamma_s_phase, A_c_over_rho, B_c_over_rho_w
    """
    r = Ds_over_Dd
    root_r = np.sqrt(1 - r**2)
    ratio_term = (1/3) - (1/4 * r**2 * root_r) - (1/12 * (1 - root_r)**2 * (2 + root_r))
    A_s_over_rho = D_d**3 * ratio_term

    # Interpolation requires 1D arrays
    kD = k * D_d
    depth_multiplier = np.exp(-k * (T_s - spar_coeffs['T_s']))

    interp = lambda y: interp1d(spar_coeffs['k'] * D_d, spar_coeffs[y], bounds_error=False, fill_value="extrapolate")(kD)

    gamma_s_over_rho_g = interp('gamma_over_rho_g') * depth_multiplier
    gamma_s_phase = interp('gamma_phase')
    A_c_over_rho = interp('A_c_over_rho')
    B_c_over_rho_w = interp('B_c_over_rho_w')

    # Radiation damping
    B_s_over_rho_w = (k / 2) * (gamma_s_over_rho_g ** 2) / mult

    return A_s_over_rho, gamma_s_over_rho_g, B_s_over_rho_w, gamma_s_phase, A_c_over_rho, B_c_over_rho_w

def calculate_x_s(x_s_guess, D_d, C_d, K_s, K_p, m_s,
                  B_p, B_33_rad_spar, w, x_f, F_s, mu, beta):
    """
    Iterative solver for heave displacement of the spar buoy.

    Parameters:
    x_s_guess : ndarray
        Initial guess for x_s.
    D_d, C_d, K_s, K_p, m_s, B_p, B_33_rad_spar, w, x_f, F_s, mu, beta : float or ndarray
        Physical and hydrodynamic parameters.

    Returns:
    x_s : ndarray
        Magnitude of spar displacement.
    x_s_error : ndarray
        Error for convergence checking.
    """
    KC = 2 * np.pi * x_s_guess / D_d
    B_33_drag_spar = (1/3) * mu * beta * D_d * KC * C_d
    B_s = B_33_drag_spar + B_33_rad_spar

    s = 1j * w
    x_s_over_F_s = 1 / (K_p + K_s + (B_p + B_s) * s + m_s * s**2)
    x_s_over_x_f = (K_p + B_p * s) * x_s_over_F_s

    x_s_complex = x_s_over_F_s * F_s + x_s_over_x_f * x_f
    x_s = np.abs(x_s_complex)
    x_s_error = x_s_guess - x_s

    return x_s, x_s_error
