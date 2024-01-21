import openmdao.api as om
import numpy as np
class structureComponent(om.ExplicitComponent):

    def setup(self):
        #Inputs
        self.add_input('F_heave', np.zeros((1,1)))
        self.add_input('F_surge', np.zeros(3,))
        self.add_input('M', 0)
        self.add_input('h_s', 0)
        self.add_input('T_s', 0)
        self.add_input('rho_w', 0)
        self.add_input('g', 0)
        self.add_input('sigma_y', np.zeros(3,))
        self.add_input('A_c', np.zeros(3,))
        self.add_input('A_lat_sub', np.zeros(3,))
        self.add_input('r_over_t', np.zeros(3,))
        self.add_input('I', np.zeros(3,))
        self.add_input('E', np.zeros(3,))

        #Outputs
        self.add_output('FOS1Y')
        self.add_output('FOS2Y')
        self.add_output('FOS3Y')
        self.add_output('FOS_buckling')

    # using wild cards to say that this component provides derivatives of all outputs with respect to all inputs.
    def setup_partials(self):
        self.declare_partials('*', '*')

    def von_mises(self, s_11, s_22, s_33, s_12, s_23, s_31):
        principal_term = 0.5 * ((s_11 - s_22) ** 2 + (s_22 - s_33) ** 2 + (s_33 - s_11) ** 2)
        shear_term = 3 * (s_12 ** 2 + s_23 ** 2 + s_31 ** 2)
        s_vm = np.sqrt(principal_term + shear_term)
        return s_vm



    def compute(self, inputs, outputs):
        # Stress calculations
        """
        self.add_input('M', 0)
        self.add_input('h_s', 0)
        self.add_input('T_s', 0)
        self.add_input('rho_w', 0)
        self.add_input('g', 0)

        """
        # Retrieve Inputs
        T_s = inputs['T_s'][0]
        rho_w = inputs['rho_w'][0]
        g = inputs['g'][0]
        F_surge = inputs['F_surge']
        A_lat_sub = inputs['A_lat_sub']
        r_over_t = inputs['r_over_t']
        F_heave = inputs['F_heave']
        A_c = inputs['A_c']
        h_s = inputs['h_s'][0]
        sigma_y = inputs['sigma_y']
        E = inputs['E']
        M = int(inputs['M'][0])
        I = inputs['I']


        # Stress calculations

        depth = np.array([0, T_s, T_s])
        P_hydrostatic = rho_w * g * depth
        sigma_surge = F_surge / A_lat_sub
        sigma_rr = P_hydrostatic + sigma_surge  # radial compression
        sigma_tt = P_hydrostatic * r_over_t  # hoop stress
        sigma_zz = F_heave / A_c  # axial compression
        sigma_rt = sigma_surge  # shear
        sigma_tz = np.array([0, 0, 0])
        sigma_zr = np.array([0, 0, 0])

        # Calculate von Mises stress
        sigma_vm = self.von_mises(sigma_rr, sigma_tt, sigma_zz, sigma_rt, sigma_tz, sigma_zr)

        # Buckling calculation
        K = 2  # fixed-free - top is fixed by float angular stiffness, bottom is free
        L = h_s


        F_buckling = np.pi ** 2 * E[M] * I[1] / (K * L) ** 2

        # Factor of Safety (FOS) Calculations
        FOS_yield = sigma_y[M] / sigma_vm

        # added [0]
        outputs['FOS1Y'] = FOS1Y = FOS_yield[0][0]
        outputs['FOS2Y'] = FOS2Y = FOS_yield[0][1]
        outputs['FOS3Y'] = FOS3Y = FOS_yield[0][2]
        outputs['FOS_buckling'] = FOS_buckling = F_buckling / F_heave






prob = om.Problem()
promotesInputs = ['F_heave', 'F_surge', 'M', 'h_s', 'T_s', 'rho_w', 'g', 'sigma_y' ,'A_c', 'A_lat_sub' ,'r_over_t', 'I', 'E']
prob.model.add_subsystem('test', structureComponent(), promotes_inputs= promotesInputs )

prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'SLSQP'

prob.model.add_design_var('F_heave', lower=0., upper=1)
prob.model.add_objective('test.FOS1Y', scaler=1)
prob.setup()


#prob.set_val('F_heave', np.array([[8499999.90183192]]))
prob.set_val('F_surge', np.array([1.81215933e+06, 4.41507300e+07, 7.32551764e+02]))
prob.set_val('M', 0)
prob.set_val('h_s', 44.0)
prob.set_val('T_s', 35.0)
prob.set_val('rho_w', 1000.0)
prob.set_val('g', 9.8)
prob.set_val('sigma_y', np.array([2.48211252e+08, 3.10264065e+07, 2.06842710e+08]))
prob.set_val('A_c', np.array([2.79043943e+00, 4.76751890e-01, 6.78584013e+02]))
prob.set_val('A_lat_sub', np.array([4.47040000e-02, 3.29867229e+02, 1.19694680e+00]))
prob.set_val('r_over_t', np.array([0. ,118.11023622, 0.]))
prob.set_val('I', np.array([7.85398163e+03, 2.12729616e+00, 3.97607820e+04]))
prob.set_val('E', np.array([2.00000000e+11,2.78506762e+07,2.00000000e+11]))

print(prob.get_val('test.FOS_buckling'))
print(prob.get_val('T_s'))
prob.run_model();
prob.model.list_inputs(val=True)
# output structure
# 3.088498840031996 7.1377643021609884 735.3862533286745 [[63.7930595]]
prob.model.list_outputs(val = True)