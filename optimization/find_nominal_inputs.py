from scipy.optimize import minimize
from inputs.parameters import *
from inputs.validation.validation_inputs import *
from simulation import *
def find_nominal_inputs(b, display_on):
    # Setup
    p = parameters()
    p["N_WEC"] = 1
    v = validation_inputs()

    p["power_max"] = v["power_max"]

    # y = [avg power, max structural force, max powertrain force / force limit]
    y_desired = np.array([v["power_avg"], v["force_heave"], 1])

    # x = [F_max_nom, B_p_nom, w_n_nom]
    x_min = np.array([b["F_max_min"], b["B_p_min"], b["w_n_min"]])
    x_max = np.array([b["F_max_max"], b["B_p_max"], b["w_n_max"]])
    x0 = np.array([b["F_max_nom"], b["B_p_nom"], b["w_n_nom"]])


    def errFunc(x):
        X = b["X_noms"].copy()
        X = np.concatenate((X, [0]))
        X[4:7] = x  # Adjust index for Python's 0-based indexing

        _, _, _, g, val = simulation(X, p)
        y = np.array([val["power_avg"], val["force_heave"][0][0], g[13] + 1])  # Adjust index for Python's 0-based indexing

        err = np.abs(y - y_desired) / y_desired
        return np.linalg.norm(err)

    # Optimization
    bounds = list(zip(x_min, x_max))

    res = minimize(errFunc, x0, bounds=bounds)

    F_max_nom, B_p_nom, w_n_nom = res.x
    if display_on:
        pass
        # Check feasibility and other outputs (adjusted for Python)
        # Display results using Python's print function or other appropriate methods

    return F_max_nom, B_p_nom, w_n_nom

# Additional functions like `parameters`, `validation_inputs`, `simulation`, etc., need to be defined in Python.
