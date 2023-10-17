import numpy as np
import matplotlib.pyplot as plt
from is_feasible import *

def validate_nominal_RM3():
    p = parameters()
    p['N_WEC'] = 1
    p['power_max'] = 286000
    p['LCOE_max'] = 10  # set large max LCOE to avoid failing feasibility check

    b = var_bounds()
    X = np.concatenate((b['X_noms'], [1]))

    _, _, _, g, simulated = simulation(X, p)

    feasible, failed = is_feasible(g, b)

    if actual or tab:
        actual = validation_inputs()
        fig, axes = plt.subplots(1, 3)

        fields = list(actual.keys())
        for i, field in enumerate(fields):
            if field in ['capex', 'opex', 'LCOE']:
                # for economic validation, sweep N_WEC
                N_WEC = [1, 10, 50, 100]
                tmp = simulated.copy()
                for j in range(1, len(N_WEC)):
                    p['N_WEC'] = N_WEC[j]
                    _, _, _, _, tmp[j] = simulation(X, p)
                simulated[field] = [entry[field] for entry in tmp]

                axes[i].semilogx(N_WEC, simulated[field], N_WEC, actual[field])
                axes[i].set_xlabel('N_{WEC}')
                axes[i].set_title(field)
                axes[i].legend(['Simulated', 'Actual'])

            sim = simulated[field]
            act = actual[field]
            pct_error[field] = abs(sim - act) / act

        improvePlot()

        if tab:
            # Assuming a struct2table equivalent exists or is created
            results = simulated.copy()
            del results['power_unsat']
            results = [results, actual, pct_error]
            tab = struct2table(results, rownames=['Simulation', 'RM3 actual', 'Error'])

    return feasible, failed, simulated, actual, tab if tab else None

