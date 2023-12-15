import numpy as np

## Variable Definitions
# D: diameter
# T: draft
# h: height
# t: material thickness
# _f: float
# _s: spar
# _d: damping plate
# _dt: damping plate tubular support

#               _____                               -
#               |   |                               |
#               |   |                               |
#            _____Df_____                       -   |
#           |           |                       |   |
# ----------|           |---------  -   -       hf  |
#           |           |           Tf  |       |   |
#           _____________           -   |       -   |
#               |   |                   |           |
#               |   |                   |           hs
#               |Ds |                   Ts          |
#               |   |                   |           |
#             / |   | \                 |           |
#        Ldt/   |   |   \               |           |
#         /     |   |     \             |           |
#       _________Dd__________           -   -       -
#       |                   |               hd
#       _____________________               -


# Not shown in diagram:
# t_ft - axial thickness of the float top plate
# t_fb - axial thickness of the float bottom plate
# t_fc - circumferential thickness of the float gussets
# t_fr - radial thickness of the float walls
# t_sr - radial thickness of the spar walls
# t_dt - radial thickness of damping plate support tube walls

def geometry(D_s, D_f, T_f, h_f, h_s, t_ft, t_fr, t_fc, t_fb, t_sr, t_dt,
             D_d, D_dt, theta_dt, T_s, h_d, M, rho_m, rho_w, m_scale):

    num_gussets = 24
    num_gussets_loaded_lateral = 2
    # convert index variable M to int instead of float
    M = int(M)

    # Float cross-sectional and lateral area
    A_f_c = np.pi * (D_f + D_s) * t_fr + num_gussets * t_fc * (D_f - D_s) / 2
    A_f_l = num_gussets_loaded_lateral * t_fc * T_f

    # Float material volume and mass
    V_top_plate = np.pi * (D_f / 2) ** 2 * t_ft
    V_bot_plate = np.pi * (D_f / 2) ** 2 * t_fb
    V_rims_gussets = A_f_c * h_f
    V_sf_m = V_top_plate + V_bot_plate + V_rims_gussets

    m_f_m = V_sf_m * rho_m[M] * m_scale
    #print("ad",rho_m)
    # Float hydrostatic calculations
    A_f = np.pi / 4 * (D_f ** 2 - D_s ** 2)
    V_f_d = A_f * T_f
    m_f_tot = V_f_d * rho_w

    # Ballast
    m_f_b = m_f_tot - m_f_m
    V_f_b = m_f_b / rho_w
    V_f_tot = A_f * h_f
    V_f_pct = V_f_b / V_f_tot

    I_f = np.pi / 64 * D_f ** 4

    # Spar (vertical column and damping plate)
    V_vc_d = np.pi / 4 * D_s ** 2 * T_s
    V_d_d = np.pi / 4 * D_d ** 2 * h_d
    V_s_d = V_vc_d + V_d_d
    m_s_tot = rho_w * V_s_d

    # Vertical column material use
    D_vc_i = D_s - 2 * t_sr
    A_vc_c = np.pi / 4 * (D_s ** 2 - D_vc_i ** 2)
    V_vc_m = A_vc_c * h_s

    # Damping plate material use
    A_d = np.pi / 4 * D_d ** 2
    num_supports = 4
    L_dt = D_d / (2 * np.cos(theta_dt))
    D_dt_i = D_dt - 2 * t_dt
    A_dt = np.pi / 4 * (D_dt ** 2 - D_dt_i ** 2)
    V_d_m = A_d * h_d + num_supports * A_dt * L_dt

    # Total spar material use and mass
    m_vc_m = V_vc_m * rho_m[M] * m_scale
    m_d_m = V_d_m * rho_m[M] * m_scale
    m_s_m = m_vc_m + m_d_m

    # Spar ballast
    m_s_b = m_s_tot - m_s_m
    V_s_b = m_s_b / rho_w
    V_s_tot = np.pi / 4 * D_s ** 2 * h_s
    V_s_pct = V_s_b / V_s_tot

    I_vc = np.pi * (D_s ** 4 - D_vc_i ** 4) / 64
    A_vc_l = 1 / 2 * np.pi * D_s * T_s

    # Reaction plate
    A_d_c = np.pi / 4 * (D_d ** 2 - D_s ** 2)
    A_d_l = 1 / 2 * np.pi * D_d * h_d
    I_rp = np.pi * D_d ** 4 / 64

    # Totals
    A_c = np.array([A_f_c, A_vc_c, A_d_c])
    A_lat_sub = np.array([A_f_l, A_vc_l, A_d_l])
    r_over_t = np.array([0, D_s / (2 * t_sr), 0])
    I = np.array([I_f, I_vc, I_rp])
    T = np.array([T_f, T_s, h_d])
    m_m = m_f_m + m_s_m

    V_d = np.array([V_f_d, V_vc_d, V_d_d])
    mass = np.array([m_f_m, m_vc_m, m_d_m])

    # Metacentric Height Calculation
    CB_f = h_d + T_s - T_f / 2
    CB_vc = h_d + T_s / 2
    CB_d = h_d / 2
    CBs = np.array([CB_f, CB_vc, CB_d])
    # centers of gravity, measured from keel (assume even mass distribution)
    CG_f = h_d + T_s - T_f + h_f / 2
    CG_vc = h_d + h_s / 2
    CG_d = h_d / 2
    CGs = np.array([CG_f, CG_vc, CG_d])

    # center of buoyancy above the keel
    KB = np.dot(CBs, V_d) / np.sum(V_d)

    # center of gravity above the keel
    KG = np.dot(CGs, mass) / np.sum(mass)

    BM = I_f / sum(V_d)  # moment due to buoyant rotational stiffness
    GM = KB + BM - KG



    #print('GM', GM)
    return V_d, m_m, m_f_tot, A_c, A_lat_sub, r_over_t, I, T, V_f_pct, V_s_pct, GM, mass

