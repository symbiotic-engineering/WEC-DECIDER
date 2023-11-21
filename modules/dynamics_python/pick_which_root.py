import numpy as np
from scipy.signal import medfilt


def pick_which_root(roots, idx_no_sat, a_quad, b_quad, c_quad):
    which_soln = (roots == np.real(roots)) & (roots > 0) & (roots <= 1)
    both_ok = np.sum(which_soln, axis=2) == 2
    # Check for the third dimension and act accordingly

    # temporarily mark the non - saturated solutions
    # as having one solution, to ensure the
    # logic below works correctly

    #Jordan's change

    #which_soln[idx_no_sat] = True
    which_soln[idx_no_sat, 0] = True

    if np.any(both_ok): # two solutions
        mult = handle_two_solns(both_ok, which_soln, roots, idx_no_sat, a_quad, b_quad, c_quad)
    else:
        num_solns = np.sum(which_soln, axis=-1)
        if not np.all(num_solns == 1):
            which_soln[num_solns == 0] = (roots[num_solns == 0] > 0) & (roots[num_solns == 0] <= 1.001)
            num_solns[num_solns == 0] = np.sum(which_soln[num_solns == 0], axis=2)
            if not np.all(num_solns == 1):
                print('Some sea states have no valid quadratic solution, so their energy is zeroed.')
        mult = get_relevant_soln(which_soln, roots, idx_no_sat)

    return mult





# pick the specified roots using multidimensional logical indexing
def get_relevant_soln(which_soln, roots, idx_no_sat):
    mult = np.zeros(idx_no_sat.shape)

    idx_3d_first_sol = np.copy(which_soln)
    idx_3d_first_sol[:, :, 1] = False
    idx_3d_second_sol = np.copy(which_soln)
    idx_3d_second_sol[:, :, 0] = False
    idx_2d_first_sol = which_soln[:, :, 0]
    idx_2d_second_sol = which_soln[:, :, 1]

    mult[idx_2d_first_sol] = roots[idx_3d_first_sol]
    mult[idx_2d_second_sol] = roots[idx_3d_second_sol]
    mult[idx_no_sat] = 1

    return mult


def handle_two_solns(both_ok, which_soln, roots, idx_no_sat, a, b, c):
    row, col = np.where(both_ok)
    which_soln[row, col, 1] = False

    mult_1 = get_relevant_soln(which_soln, roots, idx_no_sat)

    # In the provided MATLAB function, logic to handle the case of two roots has been commented out.
    # The Python function currently just uses the first solution when two are available.
    # If you need to include the logic for handling two roots, please provide the full active MATLAB code.

    return mult_1


def compare_outliers(array_1, array_2, relevant_idx):
    window = 6
    outliers_1 = medfilt(array_1, kernel_size=window) != array_1
    outliers_2 = medfilt(array_2, kernel_size=window) != array_2

    outliers_1_relevant = outliers_1[relevant_idx]
    outliers_2_relevant = outliers_2[relevant_idx]

    one_all_outliers = np.all(outliers_1_relevant)
    two_all_outliers = np.all(outliers_2_relevant)
    one_all_ok = np.all(~outliers_1_relevant)
    two_all_ok = np.all(~outliers_2_relevant)

    use_1 = (two_all_outliers and not one_all_outliers) or (one_all_ok and not two_all_ok)
    use_2 = (one_all_outliers and not two_all_outliers) or (two_all_ok and not one_all_ok)

    return use_1, use_2




