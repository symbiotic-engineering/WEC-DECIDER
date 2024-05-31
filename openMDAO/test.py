import os
import sys
from table import generate_csv
from plots import plot_openmdao_outputs
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from runOpenMdao import  *
from inputs.var_bounds import var_bounds,parameters
from omxdsm import write_xdsm

b = var_bounds()
p = parameters()
#top = waveEnergy_run_driver(b,p,D_f=30,D_s_over_D_f=0.252,T_s_over_h_s=0.41,F_max=100,w_n=10)

top = waveEnergy_run_driver(b,p, M=1, dynamic_version='new')

"""
#write_xdsm(top, filename='waveEnergy', out_format='pdf', show_browser=True, equations=True, include_solver=True,
#           quiet=True, output_side='left', include_indepvarcomps=True, class_names=False)

# Extract outputs
outputs = top.model.list_outputs(val=True, print_arrays=True)
output_dict = {}
for name, meta in outputs:
    value = meta['val']
    if value.size == 1:
        output_dict[name] = value.item()
    else:
        output_dict[name] = value.tolist()

design_var_names = [
    'ivc.D_f',
    'ivc.D_s_over_D_f',
    'ivc.h_f_over_D_f',
    'ivc.T_s_over_h_s',
    'ivc.F_max',
    'ivc.B_p',
    'ivc.w_n'
]
plot_openmdao_outputs(output_dict,design_var_names,'econComponent.LCOE')
generate_csv(outputs,"outputs.csv")
"""