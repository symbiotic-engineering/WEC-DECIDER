import openmdao.api as om
import numpy as np

class cemComponent(om.ExplicitComponent):
    def setup(self):

        #adding inputs:
        some_float_value = 0.0
        self.add_input(name = 'wec_cost', val = 0.0, desc="this is my first float variable")
        self.add_input(name = 'wec_power_hourly', val = np.zeros((4,5)), shape = (4,5), desc="this is my first float matrix variable" )

        #adding output:
        self.add_output(name = 'system_cost_avoided', desc = "this is my first float variable output")
        self.add_output(name = 'carbon_emissions_avoided', shape=(4,5), desc = "this is my first float matrix variable output")

        # Partial derivatives required for optimization
        self.declare_partials('*', '*', method='fd')


    def compute(self, inputs, outputs):
        #retrieve inputs
        firstFloatInput = inputs['wec_cost'][0] #[0] index, retrieve float number, otherwise it is an array
        firstFloatMatrixInput = inputs['wec_power_hourly']

        firstFloatOutput = 2 * firstFloatInput
        firstFloatMatrixOutput = firstFloatOutput * firstFloatMatrixInput

        #assign outputs
        outputs['system_cost_avoided'] = firstFloatOutput
        outputs['carbon_emissions_avoided'] = firstFloatMatrixOutput


#componentTest
prob = om.Problem()
#subsystem name as test
prob.model.add_subsystem('test', cemComponent())
prob.setup()
firstFloatInput = 0.2
firstFloatMatrixInput = np.ones((4,5))

prob.set_val('test.firstFloatInput', firstFloatInput)
prob.set_val('test.firstFloatMatrixInput', firstFloatMatrixInput)

prob.run_model()

prob.model.list_inputs()
"""
varname                  val                 prom_name                 
-----------------------  ------------------  --------------------------
test
  firstFloatInput        [0.2]               test.firstFloatInput      
  firstFloatMatrixInput  |4.47213595|        test.firstFloatMatrixInput
"""

prob.model.list_outputs()
"""
varname                   val                   prom_name                  
------------------------  --------------------  ---------------------------
test
  firstFloatOutput        [0.4]                 test.firstFloatOutput      
  firstFloatMatrixOutput  |1.78885438|          test.firstFloatMatrixOutput
"""

print(prob.get_val('test.firstFloatMatrixOutput')) # to get matrix output

"""
[[0.4 0.4 0.4 0.4 0.4]
 [0.4 0.4 0.4 0.4 0.4]
 [0.4 0.4 0.4 0.4 0.4]
 [0.4 0.4 0.4 0.4 0.4]]


"""