import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from inputs.var_bounds import var_bounds
from inputs.parameters import parameters
from openMDAO.runOpenMdao import waveEnergy_run_model

b = var_bounds()
p = parameters()
model = waveEnergy_run_model(b,p)