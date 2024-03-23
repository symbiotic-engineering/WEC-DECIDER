import openmdao.api as om
import numpy as np

class environmentComponent(om.ExplicitComponent):
    def setup(self):
        # 43 in_params
        self.add_input('rho_w', val=0.0, desc="water density (kg/m3)")
        self.add_input('g', val=0.0, desc="acceleration of gravity (m/s2)")
        self.add_input('JPD', val=np.zeros((14, 15)), desc="joint probability distribution of wave (%)")
        self.add_input('Hs', val=np.zeros(14, ), desc="wave height (m)")
        self.add_input('Hs_struct', val=np.zeros(1, ), desc="100 year wave height (m)")
        self.add_input('T', val=np.zeros(15, ), desc="wave period (s)")
        self.add_input('T_struct', val=np.zeros(1, ), desc="100 year wave period (s)")
        self.add_input('power_max', val=0, desc="maximum power (W)")
        self.add_input('eff_pto', val=0, desc="PTO efficiency (-)")
        self.add_input('D_f', 0)
        self.add_input('F_max', 0)
        self.add_input('B_p', 0)
        self.add_input('w_n', 0)
        self.add_input('h_f', 0)
        self.add_input('T_f', 0)
        self.add_input('T_s', 0)
        self.add_input('h_s', 0)

        # other input
        self.add_input("m_float", val=0.0)
        self.add_input("V_d", shape=(3,))
        self.add_input("draft", shape=(3,))

        # return F_heave_max, F_surge_max, F_ptrain_max, P_var, P_elec, P_matrix, h_s_extra, P_unsat
        self.add_output('F_heave_max')
        self.add_output('F_surge_max', shape=(3,))
        self.add_output('F_ptrain_max')
        self.add_output('P_var')
        self.add_output('P_elec')
        self.add_output('P_matrix', shape=(14, 15))
        self.add_output('h_s_extra')
        self.add_output('P_unsat', shape=(14, 15))

        self.declare_partials('*', '*', method='fd')


    def compute(self, inputs, outputs):
        #retrieve inputs
        firstFloatInput = inputs['firstFloatInput'][0] #[0] index, retrieve float number, otherwise it is an array
        firstFloatMatrixInput = inputs['firstFloatMatrixInput']

        firstFloatOutput = 2 * firstFloatInput
        firstFloatMatrixOutput = firstFloatOutput * firstFloatMatrixInput

        #assign outputs
        outputs['firstFloatOutput'] = firstFloatOutput
        outputs['firstFloatMatrixOutput'] = firstFloatMatrixOutput


#componentTest
prob = om.Problem()
#subsystem name as test
prob.model.add_subsystem('test', environmentComponent())
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