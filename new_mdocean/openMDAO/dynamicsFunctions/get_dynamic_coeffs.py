import numpy as np

def get_dynamic_coeffs(Hs, T, D_f, T_f, D_s, D_d, T_s, h,
                       m_float, m_spar, spar_excitation_coeffs,
                       C_d_float, C_d_spar,
                       rho_w, g,
                       use_MEEM, harmonics, hydro):
    w = 2 * np.pi / T  # angular frequency
    # to do
    k = dispersion(w, h, g)  # wave number

    # Radii and drafts
    a2 = D_f / 2
    a1 = D_s / 2
    a3 = D_d / 2
    d2 = T_f
    d1 = T_s

    # Waterplane areas
    A_w_f = np.pi * (a2 ** 2 - a1 ** 2)
    A_w_s = (np.pi / 4) * D_s ** 2

    # Hydrodynamic coefficients
    if use_MEEM:
        (A_f_over_rho, A_s_over_rho, A_c_over_rho,
         B_f_over_rho_w, B_s_over_rho_w, B_c_over_rho_w,
         gamma_f_over_rho_g, gamma_s_over_rho_g,
         gamma_phase_f, gamma_phase_s) = get_hydro_coeffs_MEEM(
            a2, k, d2, a1, d1, a3, h, g, w, harmonics, spar_excitation_coeffs)
    else:
        (A_f_over_rho, A_s_over_rho, A_c_over_rho,
         B_f_over_rho_w, B_s_over_rho_w, B_c_over_rho_w,
         gamma_f_over_rho_g, gamma_s_over_rho_g,
         gamma_phase_f, gamma_phase_s) = get_hydro_coeffs(
            a2, k, d2, hydro)

    # Added masses
    A_f = rho_w * A_f_over_rho
    A_s = rho_w * A_s_over_rho
    A_c = rho_w * A_c_over_rho

    # Total masses
    m_f = A_f + m_float
    m_s = A_s + m_spar
    m_c = A_c

    # Radiation damping
    B_h_f = rho_w * w * B_f_over_rho_w
    B_h_s = rho_w * w * B_s_over_rho_w
    B_h_c = rho_w * w * B_c_over_rho_w

    # Excitation amplitudes
    gamma_f = rho_w * g * gamma_f_over_rho_g
    gamma_s = rho_w * g * gamma_s_over_rho_g

    # Hydrostatic stiffness
    K_h_f = rho_w * g * A_w_f
    K_h_s = rho_w * g * A_w_s

    # Equivalent regular wave height
    H = Hs / np.sqrt(2)

    # Excitation forces
    F_f_mag = gamma_f * H / 2
    F_s_mag = gamma_s * H / 2

    # Excitation phases
    F_f_phase = gamma_phase_f
    F_s_phase = gamma_phase_s

    # Drag coefficients
    A_drag_s = (np.pi / 4) * D_d ** 2
    drag_const_f = (4 / (3 * np.pi)) * rho_w * A_w_f * C_d_float
    drag_const_s = (4 / (3 * np.pi)) * rho_w * A_drag_s * C_d_spar

    # Incident wave velocity magnitude
    mag_v0_f = (H / 2) * g * k / w * np.exp(-k * T_f / 2)
    mag_v0_s = (H / 2) * g * k / w * np.exp(-k * T_s / 2)

    return (m_f, B_h_f, K_h_f, F_f_mag, F_f_phase,
            m_s, B_h_s, K_h_s, F_s_mag, F_s_phase,
            m_c, B_h_c, drag_const_f, drag_const_s,
            mag_v0_f, mag_v0_s, w, k,
            A_f_over_rho, A_s_over_rho, A_c_over_rho,
            B_f_over_rho_w, B_s_over_rho_w, B_c_over_rho_w,
            gamma_f_over_rho_g, gamma_s_over_rho_g,
            gamma_phase_f, gamma_phase_s)