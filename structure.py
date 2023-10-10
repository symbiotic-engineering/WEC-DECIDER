import numpy as np

def structures(F_heave, F_surge, M, h_s, T_s, rho_w, g, sigma_y, A_c, A_lat_sub, r_over_t, I, E):
    depth = np.array([0, T_s, T_s]) # max depth
    P_hydrostatic = rho_w * g * depth
    sigma_surge = F_surge / A_lat_sub
    sigma_rr = P_hydrostatic + sigma_surge     # radial compression
    sigma_tt = P_hydrostatic * r_over_t       # hoop stress
    sigma_zz = F_heave / A_c                  # axial compression
    sigma_rt = sigma_surge                     # shear
    sigma_tz = np.array([0, 0, 0])
    sigma_zr = np.array([0, 0, 0])

    return sigma_rr, sigma_tt, sigma_zz, sigma_rt