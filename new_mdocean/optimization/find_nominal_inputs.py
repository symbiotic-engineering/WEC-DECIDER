import numpy as np
from scipy.optimize import minimize

def find_nominal_inputs(b,p):
    # Objective function: Maximize F_max → minimize -F_max
    def objective(F_max):
        return -F_max[0]  # SciPy defaults to minimization

    # Objective gradient (required for gradient-based optimization)
    def jac(F_max):
        return np.array([-1.0])  # Derivative of -F_max is -1

    # Nonlinear constraint (g >= 0 equivalent in MATLAB)
    def constraint(F_max):
        # Rebuild parameter vector X (matches MATLAB's [b.X_noms; 1])
        X = np.concatenate([b['X_noms'], [1.0]])  # Assume b is a dict
        # Find index of 'F_max' in variable names
        idx_F = np.where(np.array(b['var_names']) == 'F_max')[0][0]
        X[idx_F] = F_max[0]

        # Call simulation function (ensure Python implementation exists)
        _, _, _, g = simulation(X, p)
        # Locate constraint named 'irrelevant_max_force'
        idx_F_constr = np.where(np.array(b['constraint_names']) == 'irrelevant_max_force')[0][0]

        # SciPy requires inequality constraints as ">= 0"
        return -g[idx_F_constr]  # Original MATLAB logic: enforce g >= 0 → return -g <= 0

        # SciPy constraint configuration
    cons = {
        'type': 'ineq',  # Inequality constraint (g >= 0)
        'fun': constraint
    }

    # Initial guess and bounds (mirror MATLAB's x0/lb/ub)
    x0 = np.array([b['F_max_nom']])  # Initial value
    bounds = [(b['F_max_min'], b['F_max_max'])]  # (lower, upper)

    # Run optimization (SLSQP handles bounds/constraints)
    result = minimize(
        fun=objective,
        x0=x0,
        jac=jac,
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        options={'disp': True}  # Show optimization messages
    )

    return result.x[0]  # Optimal F_max value


def constr_func(F_max_in, p, b):
    # MATLAB: X = [b.X_noms; 1]
    X = np.concatenate([b['X_noms'], [1.0]])  # Ensure numerical type

    # MATLAB: idx_F = strcmp(b.var_names,'F_max')
    var_names_arr = np.array(b['var_names'])
    idx_F = np.where(var_names_arr == 'F_max')[0][0]  # First occurrence

    # MATLAB: X(idx_F) = F_max_in
    X[idx_F] = F_max_in[0]  # Scipy passes inputs as arrays

    # MATLAB: [~, ~, ~, g] = simulation(X, p)
    _, _, _, g = simulation(X, p)  # Ensure simulation returns NumPy array

    # MATLAB: idx_F_constr = strcmp(b.constraint_names,'irrelevant_max_force')
    constraint_names_arr = np.array(b['constraint_names'])
    idx_F_constr = np.where(constraint_names_arr == 'irrelevant_max_force')[0][0]

    # MATLAB: g_out = -g(idx_F_constr)
    inequality = -g[idx_F_constr]  # Convert to <= 0 format

    # MATLAB: g_eq = [] (no equality constraints)
    equality = np.array([])  # Empty array for equality constraints

    return inequality, equality



