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

        self.declare_partials('*', '*', method='fd')

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        m_m = inputs['m_m'][0]
        cost_m = inputs['cost_m']
        N_WEC = inputs['N_WEC'][0]
        M = int(inputs['M'][0])
        P_elec = inputs['P_elec'][0]
        efficiency = inputs['efficiency'][0]
        FCR = inputs['FCR'][0] # fixed charge rate


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
