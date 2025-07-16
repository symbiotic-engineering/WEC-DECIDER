import openmdao.api as om
import numpy as np

# Shared Variables

def openmdao_ivc(X, p, D_f=None, D_s_over_D_f=None, h_f_over_D_f=None, T_s_over_h_s=None, F_max=None, B_p=None, w_n=None, M = None):
    # Initialize IndepVarComp
    ivc = om.IndepVarComp()

    X = np.maximum(X, 1e-3)
    in_params = p.copy()


    if not D_f:
        D_f = X[0]
    if not D_s_over_D_f:
        D_s_over_D_f = X[1]
    if not h_f_over_D_f:
        h_f_over_D_f = X[2]
    if not T_s_over_h_s:
        T_s_over_h_s = X[3]
    if not F_max:
        F_max = X[4]
    if not B_p:
        B_p = X[5]
    if not w_n:
        w_n = X[6]
    if not M:
        M = int(X[7])

    # Design Variables

    ivc.add_output('D_f', D_f)
    ivc.add_output('D_s_over_D_f', D_s_over_D_f)
    ivc.add_output('h_f_over_D_f', h_f_over_D_f)
    ivc.add_output('T_s_over_h_s', T_s_over_h_s)
    ivc.add_output('F_max', F_max)
    ivc.add_output('B_p', B_p)
    ivc.add_output('w_n', w_n)
    ivc.add_output('M', M)



    existed = ['D_f', 'D_s_over_D_f', 'h_f_over_D_f', 'T_s_over_h_s', 'F_max', 'B_p', 'w_n', 'M']
    for key, value in in_params.items():
        if key not in existed:
            ivc.add_output(key, value)

    return ivc
