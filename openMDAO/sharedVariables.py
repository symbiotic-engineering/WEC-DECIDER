import openmdao.api as om
import numpy as np

# Shared Variables

def openmdao_ivc(X, p):
    # Initialize IndepVarComp
    ivc = om.IndepVarComp()

    X = np.maximum(X, 1e-3)
    in_params = p.copy()
    #[20, 0.3, 0.2, 0.7954545454545454, 5, 0.5, 0.8]
    # Design Variables
    ivc.add_output('D_f', 20)
    ivc.add_output('D_s_over_D_f', 0.3)
    ivc.add_output('h_f_over_D_f', 0.2)
    ivc.add_output('T_s_over_h_s', 0.7954545454545454)
    #ivc.add_output('F_max', X[4] * 1e6)
    ivc.add_output('F_max', 5 * 1e6)
    ivc.add_output('B_p', 0.5 * 1e6 )
    ivc.add_output('w_n', 0.8)
    ivc.add_output('M', int(X[7]))



    existed = ['D_f', 'D_s_over_D_f', 'h_f_over_D_f', 'T_s_over_h_s', 'F_max', 'B_p', 'w_n', 'M']
    for key, value in in_params.items():
        if key not in existed:
            ivc.add_output(key, value)

    return ivc
