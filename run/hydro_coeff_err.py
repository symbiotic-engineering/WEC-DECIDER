import numpy as np
import matplotlib.pyplot as plt


def hydro_coeff_err(plot_on=True):
    hydro = {}
    hydro = readWAMIT(hydro, 'rm3.out')  # Assuming WECSim provides a Python version

    w_max = 1.5
    w = hydro['w'][hydro['w'] < w_max]

    A = hydro['A'][2, 2, hydro['w'] < w_max]
    B = hydro['B'][2, 2, hydro['w'] < w_max]
    gamma = hydro['ex_ma'][2, 0, hydro['w'] < w_max]

    r = 10
    draft = 2
    g = 9.8

    k = w ** 2 / g

    A_MDOcean, B_MDOcean, gamma_MDOcean = get_hydro_coeffs(r, k, draft)
    A_MDOcean = np.ones(w.shape) * A_MDOcean

    # mean error
    err_A = np.abs(A - A_MDOcean) / A
    err_B = np.abs(B - B_MDOcean) / B
    err_G = np.abs(gamma - gamma_MDOcean) / gamma
    mA = np.mean(err_A)
    mB = np.mean(err_B)
    mG = np.mean(err_G)

    mean_abs_err = [mA, mB, mG]

    # R^2
    R2_A = np.corrcoef(A, A_MDOcean)[0, 1] ** 2
    R2_B = np.corrcoef(B, B_MDOcean)[0, 1] ** 2
    R2_G = np.corrcoef(gamma, gamma_MDOcean)[0, 1] ** 2

    R2 = [R2_A, R2_B, R2_G]

    if plot_on:
        # coeff comparison validation figure
        plt.figure()
        plt.plot(w, A, '--', w, B, '--', w, gamma, '--', linewidth=3)
        plt.plot(w, A_MDOcean, w, B_MDOcean, w, gamma_MDOcean)
        plt.ylim(0, 2500)
        plt.legend(['Added Mass A/ρ', 'Radiation Damping B/(ρω)', 'Excitation Force γ/(ρg)', 'Simulation (Analytical)',
                    'Actual (WAMIT BEM)'])
        plt.title('Normalized Hydrodynamic Coefficients')
        plt.xlabel('Wave frequency ω (rad/s)')

        # check B formula
        plt.figure()
        plt.plot(w, B / gamma ** 2 * 10000, w, w ** 2 / (2 * 9.8) * 10000, '--')
        plt.legend(['B/γ^2*ρg^2/ω', 'ω^2/2g'])

        plt.show()

    return mean_abs_err, R2

