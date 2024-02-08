import openmdao.api as om
import numpy as np
from omxdsm import write_xdsm

class econComponent(om.ExplicitComponent):

    def setup(self):
        # define inputs
        self.add_input('m_m', 0)
        self.add_input('M', 0)
        self.add_input('cost_m', np.zeros((3,)))
        self.add_input('N_WEC', 0)
        self.add_input('P_elec', 0)
        self.add_input('FCR', 0)
        self.add_input('efficiency', 0)

        # define outputs
        self.add_output('LCOE', 0)
        self.add_output('capex', 0)
        self.add_output('opex', 0)

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        m_m = inputs['m_m'][0]
        cost_m = inputs['cost_m']
        N_WEC = inputs['N_WEC'][0]
        M = int(inputs['M'][0])
        P_elec = inputs['P_elec'][0]
        efficiency = inputs['efficiency'][0]
        FCR = inputs['FCR'][0]


        structural_cost = np.multiply(m_m, cost_m)

        devicestructure = N_WEC * structural_cost[M]
        # Costs taken from 'CBS (Total)' tab of the RM3 cost breakdown structure
        development = 4553000
        infrastructure = 990000
        mooring = N_WEC * 525000
        pto = N_WEC * 623000
        profitmargin = 356000
        installation = 5909000
        contingency = 1590000
        capex = development + infrastructure + mooring + devicestructure + pto \
                + profitmargin + installation + contingency

        operations = N_WEC * 27000
        postinstall = 710000
        shoreoperations = 142000
        replacement = N_WEC * 54000
        consumables = N_WEC * 8000
        insurance = (.8 + .2 * N_WEC) * 227000
        opex = operations + postinstall + shoreoperations + replacement \
               + consumables + insurance

        hr_per_yr = 8766
        P_avg = N_WEC * P_elec * efficiency
        aep = P_avg * hr_per_yr / 1000
        LCOE = (FCR * capex + opex) / aep

        outputs['LCOE'] = LCOE
        outputs['capex'] = capex
        outputs['opex'] = opex
"""
prob = om.Problem()

promotesInputs = ["m_m", "M", "cost_m", "N_WEC", "P_elec", "FCR", "efficiency"]

prob.model.add_subsystem('test', econComponent(), promotes_inputs= promotesInputs )

prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'SLSQP'

prob.model.add_design_var('m_m')
prob.model.add_objective('test.LCOE', scaler=-1)
prob.setup()

prob.set_val('m_m', 673340.9777013776)
prob.set_val('M', 0)
prob.set_val('cost_m', np.array([[4.28, 0.06812243, 4.048]]))
prob.set_val('N_WEC', 100)
prob.set_val('P_elec', 86820.38081528545)
prob.set_val('FCR', 0.113)
prob.set_val('efficiency', 0.9309999999999999)

print(prob.get_val('test.LCOE'))
print(prob.get_val('m_m'))
prob.run_model()
prob.model.list_inputs(val=True)
# output structure
# 3.088498840031996 7.1377643021609884 735.3862533286745 [[63.7930595]]
prob.model.list_outputs(val = True)
write_xdsm(prob, filename='sellar_pyxdsm', out_format='html', show_browser=True,
               quiet=False, output_side='left')

"""