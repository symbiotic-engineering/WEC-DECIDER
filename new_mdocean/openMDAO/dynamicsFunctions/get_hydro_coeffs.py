import numpy as np
from scipy.interpolate import interp1d

def get_hydro_coeffs(r, k, draft, hydro):
    """
    Interpolates hydrodynamic coefficients from WAMIT-like results.

    Parameters:
    r      : float
        Radius of the structure (not directly used here).
    k      : ndarray
        Wave numbers (rad/m).
    draft  : float
        Submergence depth of the structure.
    hydro  : dict
        Dictionary containing 'w', 'A', 'B', 'ex_ma', and 'ex_ph' 3D arrays
        indexed as [DOF, DOF, frequency index].

    Returns:
    Tuple of interpolated hydrodynamic coefficients:
        A_f_over_rho, A_s_over_rho, A_c_over_rho,
        B_f_over_rho_w, B_s_over_rho_w, B_c_over_rho_w,
        gamma_f_over_rho_g, gamma_s_over_rho_g,
        gamma_phase_f, gamma_phase_s
    """
    g = 9.8  # gravity for wave number calculation from w
    k_wamit = hydro['w']**2 / g

    def interp(y):
        return interp1d(k_wamit, y, bounds_error=False, fill_value="extrapolate")(k)

    # Extract relevant slices from 3D tensors and flatten
    A_f_wamit = hydro['A'][2, 2, :].flatten()  # DOF 3 = index 2
    A_s_wamit = hydro['A'][8, 8, :].flatten()  # DOF 9 = index 8
    A_c_wamit = hydro['A'][2, 8, :].flatten()  # DOF 3-9 coupling

    B_f_wamit = hydro['B'][2, 2, :].flatten()
    B_s_wamit = hydro['B'][8, 8, :].flatten()
    B_c_wamit = hydro['B'][2, 8, :].flatten()

    gamma_f_wamit = hydro['ex_ma'][2, 0, :].flatten()
    gamma_s_wamit = hydro['ex_ma'][8, 0, :].flatten()
    gamma_phase_f_wamit = -hydro['ex_ph'][2, 0, :].flatten()
    gamma_phase_s_wamit = -hydro['ex_ph'][8, 0, :].flatten()

    # Interpolate all quantities
    A_f_over_rho       = interp(A_f_wamit)
    A_s_over_rho       = interp(A_s_wamit)
    A_c_over_rho       = interp(A_c_wamit)

    B_f_over_rho_w     = interp(B_f_wamit)
    B_s_over_rho_w     = interp(B_s_wamit)
    B_c_over_rho_w     = interp(B_c_wamit)

    gamma_f_over_rho_g = interp(gamma_f_wamit)
    gamma_s_over_rho_g = interp(gamma_s_wamit)
    gamma_phase_f      = interp(gamma_phase_f_wamit)
    gamma_phase_s      = interp(gamma_phase_s_wamit)

    return (
        A_f_over_rho, A_s_over_rho, A_c_over_rho,
        B_f_over_rho_w, B_s_over_rho_w, B_c_over_rho_w,
        gamma_f_over_rho_g, gamma_s_over_rho_g,
        gamma_phase_f, gamma_phase_s
    )
