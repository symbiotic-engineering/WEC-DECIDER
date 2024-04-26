
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from runOpenMdao import for_loop_waveEnergy_model
from inputs.var_bounds import var_bounds

b = var_bounds()
#print(b['X_noms'])

M_min = 0
M_max = 2

collection = for_loop_waveEnergy_model(b,M_min=0,M_max=2)
for temp in collection:
    temp.run_model()
    temp.model.list_outputs(val=True)