import openmdao.api as om
import numpy as np

class outputComponent(om.ExplicitComponent):
    #outputComponent = om.IndepVarComp()
    def setup(self):
        #output variables
        self.add_input('LCOE', val = 0.0,  desc = 'Levelized cost of energy')
        self.add_input('P_var', val = 0.0, desc = 'Variance of Error')


        # constraints
        self.add_input('V_f_pct',val = 0.0)
        self.add_input('V_s_pct', val = 0.0 )
        self.add_input('GM', val = 0.0)
        self.add_input('FOS1Y', val = 0.0 )
        self.add_input('FOS2Y', val = 0.0)
        self.add_input('FOS3Y', val = 0.0)
        self.add_input('FOS_buckling', val = 0.0)
        self.add_input('FOS_min', val = 0.0)
        self.add_input('P_elec', val = 0.0)
        self.add_input('D_d', val = 0.0)
        self.add_input('D_d_min', val = 0.0)
        self.add_input('h_s_extra', val = 0.0)
        self.add_input('LCOE_max', val = 0.0)
        self.add_input('F_ptrain_max', val = 0.0)
        self.add_input('F_max', val = 0.0)


        self.add_output('g_0', desc='Prevent float too heavy')
        self.add_output('g_1', desc='Prevent float too light')
        self.add_output('g_2', desc='Prevent spar too heavy')
        self.add_output('g_3', desc='Prevent spar too light')
        self.add_output('g_4', desc='Stability')
        self.add_output('g_5', desc='Float survives max force')
        self.add_output('g_6', desc='Spar survives max force')
        self.add_output('g_7', desc='Damping plate survives max force')
        self.add_output('g_8', desc='Spar survives max force in buckling')
        self.add_output('g_9', desc='Positive power')
        self.add_output('g_10', desc='Damping plate diameter')
        self.add_output('g_11', desc='Prevent float rising above top of spar')
        self.add_output('g_12', desc='LCOE threshold')
        self.add_output('g_13', desc='Max Force')

        # Partial derivatives required for optimization
        self.declare_partials('*', '*', method='fd')


    def compute(self, inputs, outputs):
        V_f_pct = inputs['V_f_pct']
        V_s_pct = inputs['V_s_pct']
        GM = inputs['GM']
        FOS1Y = inputs['FOS1Y']
        FOS2Y = inputs['FOS2Y']
        FOS3Y = inputs['FOS3Y']
        FOS_min = inputs['FOS_min']
        FOS_buckling = inputs['FOS_buckling']
        P_elec = inputs['P_elec']
        D_d = inputs['D_d']
        D_d_min = inputs['D_d_min']
        h_s_extra = inputs['h_s_extra']
        LCOE_max = inputs['LCOE_max']
        LCOE = inputs['LCOE']
        F_ptrain_max = inputs['F_ptrain_max']
        F_max = inputs['F_max']

        # copy the original code to here
        g = np.zeros(14)
        g[0] = V_f_pct  # Prevent float too heavy
        g[1] = 1 - V_f_pct  # Prevent float too light
        g[2] = V_s_pct  # Prevent spar too heavy
        g[3] = 1 - V_s_pct  # Prevent spar too light
        g[4] = GM  # Stability
        g[5] = FOS1Y / FOS_min - 1  # Float survives max force
        g[6] = FOS2Y / FOS_min - 1  # Spar survives max force
        g[7] = FOS3Y / FOS_min - 1  # Damping plate survives max force
        g[8] = FOS_buckling / FOS_min - 1  # Spar survives max force in buckling
        g[9] = P_elec  # Positive power
        g[10] = D_d / D_d_min - 1  # Damping plate diameter
        g[11] = h_s_extra  # Prevent float rising above top of spar
        g[12] = LCOE_max / LCOE - 1  # LCOE threshold
        g[13] = F_ptrain_max / F_max - 1  # Max force

        outputs['g_0'] = g[0]
        outputs['g_1'] = g[1]
        outputs['g_2'] = g[2]
        outputs['g_3'] = g[3]
        outputs['g_4'] = g[4]
        outputs['g_5'] = g[5]
        outputs['g_6'] = g[6]
        outputs['g_7'] = g[7]
        outputs['g_8'] = g[8]
        outputs['g_9'] = g[9]
        outputs['g_10'] = g[10]
        outputs['g_11'] = g[11]
        outputs['g_12'] = g[12]
        outputs['g_13'] = g[13]

        """
                # Assemble constraints g(x) >= 0
                g = np.zeros(14)
                g[0] = V_f_pct  # Prevent float too heavy
                g[1] = 1 - V_f_pct  # Prevent float too light
                g[2] = V_s_pct  # Prevent spar too heavy
                g[3] = 1 - V_s_pct  # Prevent spar too light
                g[4] = GM  # Stability
                g[5] = FOS1Y / p['FOS_min'] - 1  # Float survives max force
                g[6] = FOS2Y / p['FOS_min'] - 1  # Spar survives max force
                g[7] = FOS3Y / p['FOS_min'] - 1  # Damping plate survives max force
                g[8] = FOS_buckling / p['FOS_min'] - 1  # Spar survives max force in buckling
                g[9] = P_elec  # Positive power
                g[10] = D_d / p['D_d_min'] - 1  # Damping plate diameter
                g[11] = h_s_extra  # Prevent float rising above top of spar
                g[12] = p['LCOE_max'] / LCOE - 1  # LCOE threshold
                g[13] = F_ptrain_max / in_params['F_max'] - 1  # Max force
        """


