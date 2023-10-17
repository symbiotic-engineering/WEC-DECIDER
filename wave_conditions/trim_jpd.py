import numpy as np

def trim_jpd(jpd):
    jpd_entries = jpd[1:, 1:]
    rows_all_zeros = np.all(jpd_entries == 0, axis=1)
    cols_all_zeros = np.all(jpd_entries == 0, axis=0)
    rows_keep = np.insert(~rows_all_zeros, 0, True)
    cols_keep = np.insert(~cols_all_zeros, 0, True)
    trimmed = jpd[rows_keep, :][:, cols_keep]

    min_T = trimmed[0, 1]
    if min_T < 3.5:
        raise ValueError("The series expansion used to compute the froude krylov force coefficient in dynamics.m requires 2 / (T * R^(5/8)) << 1. If pushing up against this limit, it is recommended to plot gamma vs omega to confirm smooth behavior at high frequencies.")

    return trimmed

