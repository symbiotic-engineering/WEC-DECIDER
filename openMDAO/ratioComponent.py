import openmdao.api as om
import numpy as np
from inputs.parameters import parameters
from inputs.var_bounds import var_bounds


class ratioComponent(om.ExplicitComponent):

    def setup(self):
        # define inputs

        self.add_input('D_f')
        self.add_input('D_s_over_D_f')
        self.add_input('h_f_over_D_f')
        self.add_input('T_s_over_h_s')
        self.add_input('T_f_over_h_f')
        self.add_input('D_d_over_D_s')
        self.add_input('T_s_over_D_s')
        self.add_input('h_d_over_D_s')

        #define outputs
        #self.add_output('D_f')
        #self.add_output('F_max')
        #self.add_output('B_p')
        #self.add_output('w_n')
        #self.add_output('M')

        self.add_output('D_s')
        self.add_output('h_f')
        self.add_output('T_f')
        self.add_output('D_d')
        self.add_output('T_s')
        self.add_output('h_d')
        self.add_output('h_s')


        self.declare_partials('*', '*', method='fd')

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        D_f = inputs['D_f'][0]
        D_s_over_D_f = inputs['D_s_over_D_f'][0]
        h_f_over_D_f = inputs['h_f_over_D_f'][0]
        T_s_over_h_s = inputs['T_s_over_h_s'][0]
        T_f_over_h_f = inputs['T_f_over_h_f'][0]
        D_d_over_D_s = inputs['D_d_over_D_s'][0]
        T_s_over_D_s = inputs['T_s_over_D_s'][0]
        h_d_over_D_s = inputs['h_d_over_D_s'][0]


        D_s = D_s_over_D_f * D_f
        h_f = h_f_over_D_f * D_f
        T_f = T_f_over_h_f * h_f
        D_d = D_d_over_D_s * D_s
        T_s = T_s_over_D_s * D_s
        h_d = h_d_over_D_s * D_s
        h_s = 1 / T_s_over_h_s * T_s


        #inputs['D_f'] = 1
        #outputs['F_max'] = F_max
        #outputs['B_p'] = B_p
        #outputs['w_n'] = w_n
        #outputs['M'] = M

        outputs['D_s'] = D_s
        outputs['h_f'] = h_f
        outputs['T_f'] = T_f
        outputs['D_d'] = D_d
        outputs['T_s'] = T_s
        outputs['h_d'] = h_d
        outputs['h_s'] = h_s


"""
prob = om.Problem()
prob.model.add_subsystem('test', inputComponent())
prob.setup()
prob.run_model()
prob.run_driver()
prob.model.list_inputs(val=True,prom_name=False)
prob.model.list_outputs(val=True,prom_name=False)

p = parameters()
b = var_bounds(p)
X = np.concatenate((b['X_noms'], [0]))
from sharedVariables import openmdao_ivc
ivc = openmdao_ivc(X, p)

prob2 = om.Problem()
prob2.model.add_subsystem('test2', ivc)
prob2.setup()
prob2.run_model()
prob2.run_driver()
#prob2.model.list_inputs(val=True)
prob2.model.list_outputs(val=True,prom_name=False)

"""