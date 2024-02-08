import openmdao.api as om

# Shared Variables
class SharedVariables(om.IndepVarComp):
    def initialize(self):
        pass

    def setup(self):
        # Define all shared variables here
        self.add_output('D_f', 2.00000000e+01)  # Example initial value
        self.add_output('D_s_over_D_f', 3.00000000e-01)
        self.add_output('h_f_over_D_f',2.00000000e-01)
        self.add_output('T_s_over_h_s', 7.95454545e-01)
        self.add_output('F_max', )
