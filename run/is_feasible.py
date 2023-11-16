import numpy as np
def is_feasible(g, b):
    tol = -0.01
    feasible = all(np.array(g) >= tol)
    print(b['constraint_names'])
    if 'constraint_names' in b:
        const_names = b['constraint_names']
        failed = ' '
        for i, value in enumerate(g):
            if value < tol:
                failed += const_names[i]
    else:
        failed = None

    return feasible, failed