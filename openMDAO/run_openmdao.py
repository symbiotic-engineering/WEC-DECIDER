from inputs.parameters import parameters
from inputs.var_bounds import var_bounds
import time
import numpy as np
from sharedVariables import *
def main():
    p = parameters()
    b = var_bounds(p)

    # matlab inputs
    """
    b['X_noms'] = [
        20.000000000000000,
        0.300000000000000,
        0.200000000000000,
        0.795454545454545,
        9.130448602231283,
        0.279711019370833,
        1.487887634547810
    ]

    """
    # change from [1] to [0]
    X = np.concatenate((b['X_noms'], [0]))

    #LCOE, P_var, _, g, _ = simulation(X, p)
    ivc = openmdao_ivc(X,p)

    prob = om.Problem()
    model = prob.model
    model.add_subsystem('ivc', ivc)
    prob.setup()

    # After setup, you can list outputs




    """
    feasible, failed = is_feasible(g, b)
    print("LCOE:", LCOE)
    print("P_var:", P_var)
    print("g:", g)
    print("feasible", feasible)
    print("failed", failed)
    # Timing the simulation
    start_time = time.time()
    _ = simulation(X, p)  # Assuming num_outputs isn't needed in Python translation
    runtime = time.time() - start_time
    print(f"Runtime: {runtime:.4f} seconds")
    """

    """
    # Plotting
    #plot_power_matrix(X, p)
    #plt.figure()
    #power_PDF(X, p)
    #plt.show()
    """


if __name__ == "__main__":
    main()
