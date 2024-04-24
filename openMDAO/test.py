from runOpenMdao import  *
from inputs.var_bounds import var_bounds,parameters

b = var_bounds()
p = parameters()
#top = waveEnergy_run_driver(b,p,D_f=30,D_s_over_D_f=0.252,T_s_over_h_s=0.41,F_max=100,w_n=10)
top = waveEnergy_run_model(b,p)
top.run_model()
top.model.list_outputs()
"""
#print(b['X_noms'])

M_min = 0
M_max = 2

collection = for_loop_waveEnergy_driver(b,M_min=0,M_max=2)
for temp in collection:
    temp.run_driver()
    temp.model.list_outputs(val=True)

"""
