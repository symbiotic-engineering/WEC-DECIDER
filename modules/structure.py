import numpy as np

def structures(F_heave, F_surge, M, h_s, T_s, rho_w, g, sigma_y, A_c, A_lat_sub, r_over_t, I, E):
    # Stress calculations

    depth = np.array([0, T_s, T_s])
    P_hydrostatic = rho_w * g * depth
    sigma_surge = F_surge / A_lat_sub
    sigma_rr = P_hydrostatic + sigma_surge  # radial compression
    sigma_tt = P_hydrostatic * r_over_t  # hoop stress
    sigma_zz = F_heave / A_c  # axial compression
    sigma_rt = sigma_surge  # shear
    sigma_tz = np.array([0, 0, 0])
    sigma_zr = np.array([0, 0, 0])

    # Calculate von Mises stress
    sigma_vm = von_mises(sigma_rr, sigma_tt, sigma_zz, sigma_rt, sigma_tz, sigma_zr)

    # Buckling calculation
    K = 2  # fixed-free - top is fixed by float angular stiffness, bottom is free
    L = h_s

    F_buckling = np.pi ** 2 * E[M] * I[1] / (K * L) ** 2

    # Factor of Safety (FOS) Calculations
    FOS_yield = sigma_y[M] / sigma_vm

    #added [0]
    FOS1Y = FOS_yield[0][0]
    FOS2Y = FOS_yield[0][1]
    FOS3Y = FOS_yield[0][2]
    FOS_buckling = F_buckling / F_heave
    #print("output structure")
    #print(FOS1Y.shape, FOS2Y.shape, FOS3Y.shape, FOS_buckling.shape)
    return FOS1Y, FOS2Y, FOS3Y, FOS_buckling


def von_mises(s_11, s_22, s_33, s_12, s_23, s_31):
    principal_term = 0.5 * ((s_11 - s_22) ** 2 + (s_22 - s_33) ** 2 + (s_33 - s_11) ** 2)
    shear_term = 3 * (s_12 ** 2 + s_23 ** 2 + s_31 ** 2)


    s_vm = np.sqrt(principal_term + shear_term)
    #print("s_vm",s_vm)
    return s_vm
