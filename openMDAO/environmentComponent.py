import openmdao.api as om
import numpy as np
class environmentComponent(om.ExplicitComponent):
    def setup(self):

        #adding inputs:
        #some_float_value = 0.0
        self.add_input(name = 'steel', val = 0.0, desc="amoutn of steel (kg)")
        self.add_input(name = 'distance', val = 0.0, desc="distance from shore (miles)")
        self.add_input(name = 'fiberglass', val = 0.0, desc="amount of fiberglass (m^2)")
        self.add_input(name = 's_points', val = 0.17640291442619996, desc="steel eco cost (euro/kg)")
        self.add_input(name = 'f_points', val = 6.375740909087548, desc="fiberglass eco cost (euro/m^2)")
        self.add_input(name = 'd_points', val = 60.44128387234, desc="travel eco cost (euro/mi)")
        self.add_input(name = 'CEM_output', val = 0.0, desc="avoided carbon (Mtons CO2)")
        self.add_input(name = 'SCC', val = 0.133, desc="social cost of carbon (euros/kg CO2)")

        #adding output:
        self.add_output(name = 'net_eco_value', desc = "total eco value") #USD

        # Partial derivatives required for optimization
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        #retrieve inputs
        CEM_output = inputs['CEM_output'][0] #[0] index, retrieve float number, otherwise it is an array
        SCC = inputs['SCC'][0] #[0] index, retrieve float number, otherwise it is an array
        s_points = inputs['s_points'][0] #[0] index, retrieve float number, otherwise it is an array
        steel = inputs['steel'][0] #[0] index, retrieve float number, otherwise it is an array
        f_points = inputs['f_points'][0]
        fiberglass = inputs['fiberglass'][0]
        d_points = inputs['d_points'][0]
        distance = inputs['distance'][0]
<<<<<<< HEAD
        #firstFloatMatrixInput = inputs['firstFloatMatrixInput']

        CEM_points = CEM_output * SCC

        eco_value =  -(s_points * steel + f_points * fiberglass + d_points * distance) + CEM_points
        #firstFloatMatrixOutput = firstFloatOutput * firstFloatMatrixInput

        #assign outputs
        outputs['eco_value'] = eco_value
        #outputs['firstFloatMatrixOutput'] = firstFloatMatrixOutput
=======

        CEM_points = CEM_output * SCC

        eco_cost = s_points * steel + f_points * fiberglass + d_points * distance #euros
        eco_value = CEM_points #euros
        net_eco_value = (eco_value - eco_cost) * euro2USD #USD

        #assign outputs
        outputs['net_eco_value'] = net_eco_value

>>>>>>> origin/Environ_CEM_Features

"""
#componentTest
prob = om.Problem()
#subsystem name as test
prob.model.add_subsystem('test', environmentComponent())
prob.setup()
firstFloatInput = 0.2

prob.set_val('test.steel', firstFloatInput)
prob.set_val('test.distance', firstFloatInput)
prob.set_val('test.fiberglass', firstFloatInput)
prob.set_val('test.s_points', firstFloatInput)
prob.set_val('test.f_points', firstFloatInput)
prob.set_val('test.d_points', firstFloatInput)
prob.set_val('test.CEM_output', firstFloatInput)
prob.set_val('test.SCC', firstFloatInput)

<<<<<<< HEAD
#prob.set_val('test.firstFloatMatrixInput', firstFloatMatrixInput)

=======
>>>>>>> origin/Environ_CEM_Features
prob.run_model()

prob.model.list_inputs()
"""
"""
varname                  val                 prom_name                 
-----------------------  ------------------  --------------------------
test
  firstFloatInput        [0.2]               test.firstFloatInput      
  firstFloatMatrixInput  |4.47213595|        test.firstFloatMatrixInput
"""

#prob.model.list_outputs()
"""
varname                   val                   prom_name                  
------------------------  --------------------  ---------------------------
test
  firstFloatOutput        [0.4]                 test.firstFloatOutput      
  firstFloatMatrixOutput  |1.78885438|          test.firstFloatMatrixOutput
"""

#print(prob.get_val('test.firstFloatMatrixOutput')) # to get matrix output

"""
[[0.4 0.4 0.4 0.4 0.4]
 [0.4 0.4 0.4 0.4 0.4]
 [0.4 0.4 0.4 0.4 0.4]
 [0.4 0.4 0.4 0.4 0.4]]


"""
