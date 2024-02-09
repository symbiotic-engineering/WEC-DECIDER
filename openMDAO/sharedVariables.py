import openmdao.api as om
import numpy as np

# Shared Variables

def openmdao_ivc(X, p):
    # Initialize IndepVarComp
    ivc = om.IndepVarComp()

    X = np.maximum(X, 1e-3)
    val = {}
    in_params = p.copy()
    in_params['D_f'] = X[0]
    D_s_over_D_f = X[1]
    h_f_over_D_f = X[2]
    T_s_over_h_s = X[3]
    in_params['F_max'] = X[4] * 1e6
    in_params['B_p'] = X[5] * 1e6
    in_params['w_n'] = X[6]
    # Change float to Int
    in_params['M'] = int(X[7])
    # Variable ratios defined by design variables
    in_params['D_s'] = D_s_over_D_f * in_params['D_f']
    in_params['h_f'] = h_f_over_D_f * in_params['D_f']
    in_params['T_f'] = p['T_f_over_h_f'] * in_params['h_f']
    D_d = p['D_d_over_D_s'] * in_params['D_s']
    in_params['T_s'] = p['T_s_over_D_s'] * in_params['D_s']
    in_params['h_d'] = p['h_d_over_D_s'] * in_params['D_s']
    in_params['h_s'] = 1 / T_s_over_h_s * in_params['T_s']

    #print(in_params)
    #exit(255)

    # Design Variables
    ivc.add_output('D_f', X[0])
    ivc.add_output('D_s_over_D_f', X[1])
    ivc.add_output('h_f_over_D_f', X[2])
    ivc.add_output('T_s_over_h_s', X[3])
    ivc.add_output('F_max', X[4] * 1e6)
    ivc.add_output('B_p', X[5] * 1e6)
    ivc.add_output('w_n', X[6])
    ivc.add_output('M', int(X[7]))
    #outputs = ivc.list_outputs()
    #print("outputs", outputs,ivc.list_outputs())
    #exited_outputs =

    ivc.add_output('D_d', D_d)
    existed = ['D_f', 'D_s_over_D_f', 'h_f_over_D_f', 'T_s_over_h_s', 'F_max', 'B_p', 'w_n', 'M']

    for key, value in in_params.items():
        if key not in existed:
            ivc.add_output(key, value)





    return ivc
    # Variable ratios defined by design variables

    # ivc.add_output('D_s', X[1] *  in_params['D_f'])
