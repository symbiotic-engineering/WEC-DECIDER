import numpy as np

def lin_ineq_constraints(p, param_name=None):
    """
    Generate linear inequality constraints A @ x < b and optionally their derivatives dAdp, dbdp.

    Parameters:
    - p: an object or dict containing required attributes.
    - param_name: string (optional), the parameter name for which derivatives are computed.

    Returns:
    - A: inequality constraint matrix
    - b: inequality bound vector
    - dAdp: derivative of A with respect to the parameter (if param_name is given)
    - dbdp: derivative of b with respect to the parameter (if param_name is given)
    """

    pi = np.pi
    MEEM = pi * p['harmonics'] / (p['besseli_argmax'] * 2)

    # Build A matrix
    A = np.array([
        [-p['D_d_over_D_s'],         0,                     0,                      0, 0, 0, 0, 0, 0,   0,   0, 0, 0],
        [ p['D_f_in_over_D_s'],     -1,                     0,                      0, 0, 0, 0, 0, 0,   0,   0, 0, 0],
        [ p['h_d_over_D_s'] - p['T_s_over_D_s'], 0,          1,                      0, 0, 0, 0, 0, 0,   0,   0, 0, 0],
        [ p['T_s_over_D_s'],         0,    1/p['T_f_2_over_h_f'] - 1,              -1, 0, 0, 0, 0, 0,   0,   0, 0, 0],
        [0,                         MEEM,                   1,                      0, 0, 0, 0, 0, 0,   0,   0, 0, 0],
        [MEEM + p['T_s_over_D_s'],   0,                     0,                      0, 0, 0, 0, 0, 0,   0,   0, 0, 0],
        [-p['h_d_over_D_s'],         0,                     0,                      0, 0, 0, 0, 0, 0, 1e-3,  0, 0, 0],
        [0,                          0,      -0.5/p['T_f_2_over_h_f'],             0, 0, 0, 0, 0, 0,   0, 0.1, 0, 0]
    ])

    # b vector
    b = np.array([
        -p['D_d_min'],
        -0.01,
        0,
        0,
        p['h'],
        p['h'],
        0,
        0
    ])

    # Optional derivatives
    dAdp = None
    dbdp = None

    if param_name is not None:
        dAdp = np.zeros_like(A)
        dbdp = np.zeros_like(b)

        if param_name == 'D_d_over_D_s':
            dAdp[0, 0] = -1
        elif param_name == 'T_s_over_D_s':
            dAdp[2, 0] = -1
            dAdp[3, 0] = 1
            dAdp[5, 0] = 1
        elif param_name == 'T_f_2_over_h_f':
            dAdp[3, 2] = -1 / p['T_f_2_over_h_f']**2
            dAdp[7, 2] = 0.5 / p['T_f_2_over_h_f']**2
        elif param_name == 'harmonics':
            # derivative of MEEM = pi * harmonics / (besseli_argmax * 2)
            dMEEM_dharmonics = pi / (p['besseli_argmax'] * 2)
            dAdp[5, 0] = dMEEM_dharmonics
            dAdp[4, 1] = dMEEM_dharmonics
        elif param_name == 'D_d_min':
            dbdp[0] = -1
        elif param_name == 'h':
            dbdp[4:6] = 1
        elif param_name == 'h_d_over_D_s':
            dAdp[2, 0] = 1
            dAdp[6, 0] = -1

    return (A, b, dAdp, dbdp) if param_name else (A, b)
