import openmdao.api as om
import openmdao.visualization.opt_report.opt_report as omviz
from waveEnergy import waveEnergy
#from inputs.var_bounds import var_bounds

def waveEnergy_run_driver(b,p = None,D_f=None, D_s_over_D_f=None, h_f_over_D_f=None, T_s_over_h_s=None, F_max=None, B_p=None,w_n=None, M=0, max_iter = 1000, tol = 1e-8):
    #for M in range(M_min,M_max): April 13th another function
    model = waveEnergy(b = b, p = p,D_f = D_f, D_s_over_D_f=D_s_over_D_f, h_f_over_D_f=h_f_over_D_f, T_s_over_h_s=T_s_over_h_s, F_max=F_max, B_p=B_p, w_n=w_n, M=M)
    top = om.Problem(model=model)
    top.driver = om.ScipyOptimizeDriver()
    top.driver.options['optimizer'] = 'SLSQP'

    top.model.add_design_var('ivc.D_f', lower=b['D_f_min'], upper=b['D_f_max'])
    top.model.add_design_var('ivc.D_s_over_D_f', lower=b['D_s_ratio_min'], upper=b['D_s_ratio_max'])
    top.model.add_design_var('ivc.h_f_over_D_f', lower=b['h_f_ratio_min'], upper=b['h_f_ratio_max'])
    top.model.add_design_var('ivc.T_s_over_h_s', lower=b['T_s_ratio_min'], upper=b['T_s_ratio_max'])
    top.model.add_design_var('ivc.F_max', lower=b['F_max_min'], upper=b['F_max_max'])  # new Value = (initial + adder ) * scaler
    top.model.add_design_var('ivc.B_p', lower=b['B_p_min'], upper=b['B_p_max'])
    top.model.add_design_var('ivc.w_n', lower=b['w_n_min'], upper=b['w_n_max'])
    # top.model.add_design_var('ivc.M', lower=0, upper=2)

    top.driver.options['maxiter'] = 3 #max_iter  # Increase max iterations
    top.driver.options['tol'] = 1 #tol
    #outcome
    top.model.add_objective('outcomeComponent.LCOE', scaler=1)
    # add constraints.
    top.model.add_constraint('outcomeComponent.g_0', lower=0)
    top.model.add_constraint('outcomeComponent.g_1', lower=0)
    top.model.add_constraint('outcomeComponent.g_2', lower=0)
    top.model.add_constraint('outcomeComponent.g_3', lower=0)
    top.model.add_constraint('outcomeComponent.g_4', lower=0)
    top.model.add_constraint('outcomeComponent.g_5', lower=0)
    top.model.add_constraint('outcomeComponent.g_6', lower=0)
    top.model.add_constraint('outcomeComponent.g_7', lower=0)
    top.model.add_constraint('outcomeComponent.g_8', lower=0)
    top.model.add_constraint('outcomeComponent.g_9', lower=0)
    top.model.add_constraint('outcomeComponent.g_10', lower=0)
    top.model.add_constraint('outcomeComponent.g_11', lower=0)
    top.model.add_constraint('outcomeComponent.g_12', lower=0)
    top.model.add_constraint('outcomeComponent.g_13', lower=0)

    top.setup()
    print('----------------------------SETUP DONE, STARTING OPTIMIZATION---------------------------')
    top.run_driver()
    top.model.list_outputs(val=True)
    omviz.opt_report(top)

    return top

def waveEnergy_run_model(b,p = None,D_f=None, D_s_over_D_f=None, h_f_over_D_f=None, T_s_over_h_s=None, F_max=None, B_p=None, w_n=None, M=0):
    model = waveEnergy(b = b, p = p,D_f=D_f, D_s_over_D_f=D_s_over_D_f, h_f_over_D_f=h_f_over_D_f, T_s_over_h_s=T_s_over_h_s,
                           F_max=F_max, B_p=B_p, w_n=w_n, M=M)
    top = om.Problem(model=model)
    top.driver = om.ScipyOptimizeDriver()
    top.driver.options['optimizer'] = 'SLSQP'
    """
    top.model.add_design_var('ivc.D_f', lower=b['D_f_min'], upper=b['D_f_max'])
    top.model.add_design_var('ivc.D_s_over_D_f', lower=b['D_s_ratio_min'], upper=b['D_s_ratio_max'])
    top.model.add_design_var('ivc.h_f_over_D_f', lower=b['h_f_ratio_min'], upper=b['h_f_ratio_max'])
    top.model.add_design_var('ivc.T_s_over_h_s', lower=b['T_s_ratio_min'], upper=b['T_s_ratio_max'])
    top.model.add_design_var('ivc.F_max', lower=b['F_max_min'], upper=b['F_max_max'])  # new Value = (initial + adder ) * scaler
    top.model.add_design_var('ivc.B_p', lower=b['B_p_min'], upper=b['B_p_max'])
    top.model.add_design_var('ivc.w_n', lower=b['w_n_min'], upper=b['w_n_max'])
    # top.model.add_design_var('ivc.M', lower=0, upper=2)

    #top.driver.options['maxiter'] = max_iter  # Increase max iterations
    #top.driver.options['tol'] = tol
    #top.model.add_objective('outcomeComponent.LCOE', scaler=1)
    # add constraints.
    
    top.model.add_constraint('outcomeComponent.g_0', lower=0)
    top.model.add_constraint('outcomeComponent.g_1', lower=0)
    top.model.add_constraint('outcomeComponent.g_2', lower=0)
    top.model.add_constraint('outcomeComponent.g_3', lower=0)
    top.model.add_constraint('outcomeComponent.g_4', lower=0)
    top.model.add_constraint('outcomeComponent.g_5', lower=0)
    top.model.add_constraint('outcomeComponent.g_6', lower=0)
    top.model.add_constraint('outcomeComponent.g_7', lower=0)
    top.model.add_constraint('outcomeComponent.g_8', lower=0)
    top.model.add_constraint('outcomeComponent.g_9', lower=0)
    top.model.add_constraint('outcomeComponent.g_10', lower=0)
    top.model.add_constraint('outcomeComponent.g_11', lower=0)
    top.model.add_constraint('outcomeComponent.g_12', lower=0)
    top.model.add_constraint('outcomeComponent.g_13', lower=0)
    """
    top.setup()
    #top.run_model()
    #top.model.list_outputs(val=True)

    return top

def for_loop_waveEnergy_driver(b,p = None,D_f=None, D_s_over_D_f=None, h_f_over_D_f=None, T_s_over_h_s=None, F_max=None, B_p=None, w_n=None, M_min = 0, M_max = 1, max_iter = 1000, tol = 1e8):
    waveEnergy_driver_collections = []
    for M_value in range(M_min,M_max):
        problem = waveEnergy_run_driver(b,p,D_f,D_s_over_D_f,h_f_over_D_f,T_s_over_h_s,F_max,B_p,w_n,M=M_value, max_iter=max_iter, tol=tol)
        waveEnergy_driver_collections.append(problem)
    return waveEnergy_driver_collections

def for_loop_waveEnergy_model(b,p = None,D_f=None, D_s_over_D_f=None, h_f_over_D_f=None, T_s_over_h_s=None, F_max=None, B_p=None, w_n=None, M_min = 0, M_max = 1):
    waveEnergy_model_collections = []
    for M_value in range(M_min, M_max):
        problem = waveEnergy_run_model(b,p = None,D_f=None, D_s_over_D_f=None, h_f_over_D_f=None, T_s_over_h_s=None, F_max=None, B_p=None, w_n=None, M=M_value)
        waveEnergy_model_collections.append(problem)
    return waveEnergy_model_collections
