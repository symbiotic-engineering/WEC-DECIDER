import numpy as np

def trim_jpd(jpd):
    # Take the submatrix excluding the first row and first column
    jpd_entries = jpd[1:, 1:]

    # Find rows and columns that are all zeros
    rows_all_zeros = np.all(jpd_entries == 0, axis=1)
    cols_all_zeros = np.all(jpd_entries == 0, axis=0)

    # Build mask: keep first row/column + nonzero rows/cols
    rows_keep = np.concatenate(([True], ~rows_all_zeros))
    cols_keep = np.concatenate(([True], ~cols_all_zeros))

    # Apply the mask
    trimmed = jpd[np.ix_(rows_keep, cols_keep)]

    return trimmed