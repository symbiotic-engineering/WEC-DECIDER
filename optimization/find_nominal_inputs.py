from scipy.optimize import minimize
from inputs.parameters import *
from inputs.validation.validation_inputs import *
from openMDAO.runOpenMdao import waveEnergy_run_model
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
        print("hi from err_func")
        X = b["X_noms"].copy()
        X = np.concatenate((X, [0]))
        X[4:7] = x  # Adjust index for Python's 0-based indexing
        b_copy = b.copy()
        b_copy["X_noms"] = np.concatenate((X, [0]))
        b_copy["X_noms"][4:7] = x
        #_, _, _, g, val = simulation(X, p) #run_model
        result = waveEnergy_run_model(b_copy,p)
        result.run_model()
        g, val = retrieve_g_val_from_openMdao(result)

        y = np.array([val["power_avg"], val["force_heave"][0][0], g[13] + 1])  # Adjust index for Python's 0-based indexing

        err = np.abs(y - y_desired) / y_desired
        return np.linalg.norm(err)

    # Optimization
    bounds = list(zip(x_min, x_max))

    res = minimize(errFunc, x0, bounds=bounds,options={'maxiter':1})# starting print # don't need to replace it

    F_max_nom, B_p_nom, w_n_nom = res.x

    if display_on:
        pass
        # Check feasibility and other outputs (adjusted for Python)
        # Display results using Python's print function or other appropriate methods

    return F_max_nom, B_p_nom, w_n_nom

# Additional functions like `parameters`, `validation_inputs`, `simulation`, etc., need to be defined in Python.

def retrieve_g_val_from_openMdao(model):
    g = np.zeros(14)
    g[0] = model.get_val('outcomeComponent.g_0')[0]
    g[1] = model.get_val('outcomeComponent.g_1')[0]
    g[2] = model.get_val('outcomeComponent.g_2')[0]
    g[3] = model.get_val('outcomeComponent.g_3')[0]
    g[4] = model.get_val('outcomeComponent.g_4')[0]
    g[5] = model.get_val('outcomeComponent.g_5')[0]
    g[6] = model.get_val('outcomeComponent.g_6')[0]
    g[7] = model.get_val('outcomeComponent.g_7')[0]
    g[8] = model.get_val('outcomeComponent.g_8')[0]
    g[9] = model.get_val('outcomeComponent.g_9')[0]
    g[10] = model.get_val('outcomeComponent.g_10')[0]
    g[11] = model.get_val('outcomeComponent.g_11')[0]
    g[12] = model.get_val('outcomeComponent.g_12')[0]
    g[13] = model.get_val('outcomeComponent.g_13')[0]
    mass = model.get_val('geometryComponent.mass')
    capex = model.get_val('econComponent.capex')[0]
    opex = model.get_val('econComponent.opex')[0]
    LCOE = model.get_val('econComponent.LCOE')[0]
    P_elec = model.get_val('dynamicsComponent.P_elec')[0]
    P_matrix = model.get_val('dynamicsComponent.P_matrix')
    F_heave_max = model.get_val('dynamicsComponent.F_heave_max')
    FOS_buckling = model.get_val('structureComponent.FOS_buckling')
    P_var = model.get_val('dynamicsComponent.P_var')[0]
    P_unsat = model.get_val('dynamicsComponent.P_unsat')
    val = {
        'mass_f': mass[0],
        'mass_vc': mass[1],
        'mass_rp': mass[2],
        'mass_tot': sum(mass),
        'capex': capex,
        'opex': opex,
        'LCOE': LCOE,
        'power_avg': P_elec,
        'power_max': np.max(P_matrix),
        'force_heave': np.array([F_heave_max]), #to match the old result
        'FOS_b': np.array([FOS_buckling]),
        'c_v': P_var,
        'power_unsat': P_unsat
    }

    return g, val if 'val' in locals() else None