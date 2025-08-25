import numpy as np
from ..meem.run_MEEM import run_MEEM
from ..meem.group_velocity import group_velocity
import openflash
from spar_dynamics import spar_dynamics

def get_hydro_coeffs_MEEM(a2, m0, d2, a1, d1, a3, h, g, w, harmonics, spar_excitation_coeffs):
    # Fixed flags
    heaving_IC = False
    heaving_OC = True
    auto_BCs = False
    spatial_res = 0
    show_A = False
    plot_phi = False

    N_num = harmonics
    M_num = harmonics
    K_num = harmonics

    # Clean and filter unique m0 values (ignoring NaNs)
    m0_meem = np.unique(m0[~np.isnan(m0)])
    if m0_meem.ndim > 1 and m0_meem.shape[0] > m0_meem.shape[1]:
        m0_meem = m0_meem.T  # ensure row vector

    # Run MEEM solver
    #m = openflash.MEEMProblem
    #e = openflash.MEEMEngine
    mu_nondim, lambda_nondim, gamma_phase_f = run_MEEM(
        heaving_IC, heaving_OC, auto_BCs,
        N_num, M_num, K_num,
        a1 / h, a2 / h, d1 / h, d2 / h, 1.0, m0_meem * h,
        spatial_res, show_A, plot_phi
    )

    # Expand to match shape of input (assumes m0 is shape [Hs, freqs])
    num_Hs = m0.shape[0]
    mu_nondim = np.tile(mu_nondim, (num_Hs, 1))
    lambda_nondim = np.tile(lambda_nondim, (num_Hs, 1))
    gamma_phase_f = np.tile(gamma_phase_f, (num_Hs, 1))

    # Normalize added mass and damping
    normalize = np.pi * a2 ** 3
    A_f_over_rho   = mu_nondim * normalize
    B_f_over_rho_w = lambda_nondim * normalize

    # Finite depth group velocity multiplier
    _, mult = group_velocity(w, m0, g, h)

    # Haskind relationship (finite depth)
    gamma_f_over_rho_g = np.sqrt(2 * mult * B_f_over_rho_w / m0)

    # Approximated spar dynamics (to supplement MEEM for internal spar motion)
    (A_s_over_rho, gamma_s_over_rho_g,
     B_s_over_rho_w, gamma_phase_s,
     A_c_over_rho, B_c_over_rho_w) = spar_dynamics(
        a1 / a3, 2 * a3, d1, spar_excitation_coeffs, m0, mult
    )

    return (A_f_over_rho, A_s_over_rho, A_c_over_rho,
            B_f_over_rho_w, B_s_over_rho_w, B_c_over_rho_w,
            gamma_f_over_rho_g, gamma_s_over_rho_g,
            gamma_phase_f, gamma_phase_s)
