import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from runOpenMdao import  *
from inputs.var_bounds import var_bounds,parameters
from omxdsm import write_xdsm

b = var_bounds()
p = parameters()
#top = waveEnergy_run_driver(b,p,D_f=30,D_s_over_D_f=0.252,T_s_over_h_s=0.41,F_max=100,w_n=10)
top = waveEnergy_run_driver(b,p)
write_xdsm(top, filename='waveEnergy', out_format='pdf', show_browser=True, equations=True, include_solver=True,
           quiet=False, output_side='left', include_indepvarcomps=True, class_names=False)
#top.run_driver()
#top.model.list_outputs()
"""
#print(b['X_noms'])

M_min = 0
M_max = 2

collection = for_loop_waveEnergy_driver(b,M_min=0,M_max=2)
for temp in collection:
    temp.run_driver()
    temp.model.list_outputs(val=True)

"""
