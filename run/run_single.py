import numpy as np
import matplotlib.pyplot as plt
import time
from simulation import *
from is_feasible import *
from parameters import *
from var_bounds import *
# Assuming parameters, var_bounds, simulation, is_feasible, plot_power_matrix,
# and power_PDF are already translated and implemented as Python functions.

def main():
    p = parameters()
    b = var_bounds(p)

    #change from [1] to [0]
    #X = np.concatenate((b['X_noms'], [1]))
    X = np.concatenate((b['X_noms'], [0]))
    #print("X",X)
    #print(X)

    LCOE, P_var, _, g, _= simulation(X, p)

    feasible, failed = is_feasible(g, b)
    print("LCOE:",LCOE)
    print("P_var:",P_var)
    print("g:",g)
    print("feasible",feasible)
    print("failed", failed)
    # Timing the simulation
    start_time = time.time()
    _ = simulation(X, p)  # Assuming num_outputs isn't needed in Python translation
    runtime = time.time() - start_time
    print(f"Runtime: {runtime:.4f} seconds")
    
    
    """
    # Plotting
    #plot_power_matrix(X, p)
    #plt.figure()
    #power_PDF(X, p)
    #plt.show()
    """

if __name__ == "__main__":
    main()
