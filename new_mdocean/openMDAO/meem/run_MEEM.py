import numpy as np
import os


def run_MEEM(heaving_IC, heaving_OC, auto_BCs,
             N_num, M_num, K_num,
             a1_mat, a2_mat, d1_mat,
             d2_mat, h_mat, m0_mat,
             spatial_res, show_A, plot_phi):
    """
    Run MEEM for a batch of numeric input combinations.
    """
    # Prepare input arrays
    inputs = [np.atleast_1d(x) for x in (a1_mat, a2_mat, d1_mat, d2_mat, h_mat, m0_mat)]
    shapes = [x.shape for x in inputs]
    sizes = [x.size for x in inputs]

    # Check for consistency in nonscalar inputs
    nonscalar_sizes = [s for s in sizes if s > 1]
    if nonscalar_sizes:
        assert all(s == nonscalar_sizes[0] for s in nonscalar_sizes), \
            "All nonscalar inputs must have the same number of elements"
        num_runs = nonscalar_sizes[0]
    else:
        num_runs = 1

    # Setup file name
    if heaving_IC and heaving_OC:
        heaving = 'both'
    elif heaving_IC:
        heaving = 'inner'
    else:
        heaving = 'outer'
    fname = f'N{N_num}_M{M_num}_K{K_num}_heaving_{heaving}'

    # Create symbolic expressions if not already available
    A_path = f'simulation/modules/MEEM/generated/A_b_c_matrix_{fname}.py'
    hydro_path = f'simulation/modules/MEEM/generated/hydro_potential_velocity_fields_{fname}.py'
    if not (os.path.exists(A_path) and os.path.exists(hydro_path)):
        create_symbolic_expressions(heaving_IC, heaving_OC, auto_BCs, N_num, M_num, K_num, fname)

    # Initialize results
    mu_nondim = np.zeros(num_runs)
    lambda_nondim = np.zeros(num_runs)
    exc_phases = np.zeros(num_runs)

    # Loop through runs
    for i in range(num_runs):
        idxs = [0 if size == 1 else i for size in sizes]
        a1, a2 = inputs[0][idxs[0]], inputs[1][idxs[1]]
        d1, d2 = inputs[2][idxs[2]], inputs[3][idxs[3]]
        h, m0 = inputs[4][idxs[4]], inputs[5][idxs[5]]

        valid = d1 < h and d2 < h and a1 < a2 and d2 < d1

        if valid:
            mu, lam, phase = compute_and_plot(a1, a2, d1, d2, h, m0, spatial_res,
                                              K_num, show_A, plot_phi, fname)
            mu_nondim[i], lambda_nondim[i], exc_phases[i] = mu, lam, phase
        else:
            mu_nondim[i] = lambda_nondim[i] = 1e-9
            exc_phases[i] = 0
            print("Warning: Invalid geometry. Setting hydro coeffs to very small values.")

    # Reshape if needed
    if nonscalar_sizes:
        target_shape = next(shape for shape in shapes if np.prod(shape) == num_runs)
        mu_nondim = mu_nondim.reshape(target_shape)
        lambda_nondim = lambda_nondim.reshape(target_shape)
        exc_phases = exc_phases.reshape(target_shape)

    # Check for NaNs or invalid values
    if np.isnan(mu_nondim).any():
        raise ValueError("MEEM got NaN as result")

    lambda_nondim = np.where(lambda_nondim < 0, 1e-9, lambda_nondim)
    if (lambda_nondim < 0).any():
        print("Warning: Negative damping. Setting to small positive value.")

    return mu_nondim, lambda_nondim, exc_phases