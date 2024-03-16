import openmdao.api as om
import numpy as np

# Shared Variables

def openmdao_ivc(X, p):
    # Initialize IndepVarComp
    ivc = om.IndepVarComp()

    X = np.maximum(X, 1e-3)
    in_params = p.copy()

    # Design Variables
    ivc.add_output('D_f', 6)
    ivc.add_output('D_s_over_D_f', 0.01)
    ivc.add_output('h_f_over_D_f', 0.1)
    ivc.add_output('T_s_over_h_s', 0.01)
    #ivc.add_output('F_max', X[4] * 1e6)
    ivc.add_output('F_max', 10000)
    ivc.add_output('B_p', 0.1 * 1e6 )
    ivc.add_output('w_n', 40)
    ivc.add_output('M', X[7])



    existed = ['D_f', 'D_s_over_D_f', 'h_f_over_D_f', 'T_s_over_h_s', 'F_max', 'B_p', 'w_n', 'M']
    for key, value in in_params.items():
        if key not in existed:
            ivc.add_output(key, value)

    return ivc
