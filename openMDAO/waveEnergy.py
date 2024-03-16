import openmdao.api as om

from geometryComponent import geometryComponent
from dynamicsComponent import dynamicsComponent
from structureComponent import structureComponent
from econComponent import econComponent
from ratioComponent import ratioComponent

from inputs.parameters import parameters
from inputs.var_bounds import var_bounds
from sharedVariables import openmdao_ivc
import numpy as np

from omxdsm import write_xdsm
class waveEngergy(om.Group):

    def setup(self):
        p = parameters()
        b = var_bounds(p)
        X = np.concatenate((b['X_noms'], [0]))
        ivc = openmdao_ivc(X, p)
        self.add_subsystem('ivc', ivc)
        self.add_subsystem('ratioComponent', ratioComponent())
        self.add_subsystem('geometryComponent', geometryComponent())
        self.add_subsystem('dynamicsComponent', dynamicsComponent())
        self.add_subsystem('structureComponent', structureComponent())
        self.add_subsystem('econComponent', econComponent())


    def configure(self):
        #ivc to ratioComponent
        ivc_to_ratio = ['D_f','D_s_over_D_f','h_f_over_D_f','T_s_over_h_s', 'T_f_over_h_f', 'D_d_over_D_s','T_s_over_D_s', 'h_d_over_D_s']
        for var_name in ivc_to_ratio:
            self.connect(f"ivc.{var_name}", f"ratioComponent.{var_name}")

        #ivc to geometryComponent

        # connect ivc to geo
        ivc_to_geo = ['D_f',   't_ft', 't_fr', 't_fc','t_fb', 't_sr', 't_dt', 'D_dt', 'theta_dt','M', 'rho_m', 'rho_w', 'm_scale']

        for var_name in ivc_to_geo:
            self.connect(f"ivc.{var_name}", f"geometryComponent.{var_name}")

        # ratop tp geometryComponent
        ratio_to_geo = ['D_s','T_f', 'h_f', 'h_s', 'D_d',  'T_s', 'h_d']
        for var_name in ratio_to_geo:
            self.connect(f"ratioComponent.{var_name}", f"geometryComponent.{var_name}")


        # Dynamics
        #ivc to dynamics
        ivc_to_dynam = ['rho_w', 'g', 'JPD', 'Hs', 'Hs_struct', 'T', 'T_struct', 'power_max', 'eff_pto', 'D_f', 'F_max', 'B_p',
                        'w_n']

        for var_name in ivc_to_dynam:
            self.connect(f"ivc.{var_name}", f"dynamicsComponent.{var_name}")

        #ratio to dynam
        ratio_to_dymn = ['h_f', 'T_f','T_s', 'h_s']
        for var_name in ratio_to_dymn:
            self.connect(f"ratioComponent.{var_name}", f"dynamicsComponent.{var_name}")

        #geo_to_dynam
        self.connect('geometryComponent.m_f_tot', 'dynamicsComponent.m_float')
        self.connect('geometryComponent.V_d', 'dynamicsComponent.V_d')
        self.connect('geometryComponent.T', 'dynamicsComponent.draft')


        # Structure
        # ivc to structure
        ivc_to_struct = ['M',  'rho_w', 'g', 'sigma_y', 'E']

        for var_name in ivc_to_struct:
            self.connect(f"ivc.{var_name}", f"structureComponent.{var_name}")

        ratio_to_struct = ['h_s', 'T_s']

        for var_name in ratio_to_struct:
            self.connect(f"ratioComponent.{var_name}", f"structureComponent.{var_name}")

        # geo to structure
        self.connect('geometryComponent.A_c', 'structureComponent.A_c')
        self.connect('geometryComponent.A_lat_sub', 'structureComponent.A_lat_sub')
        self.connect('geometryComponent.r_over_t', 'structureComponent.r_over_t')
        self.connect('geometryComponent.I', 'structureComponent.I')

        # dynam to structure
        self.connect('dynamicsComponent.F_heave_max', 'structureComponent.F_heave')
        self.connect('dynamicsComponent.F_surge_max','structureComponent.F_surge')


        #ivc to econ
        ivc_to_eco = ['M', 'cost_m', 'N_WEC', 'FCR']

        for var_name in ivc_to_eco:
            self.connect(f"ivc.{var_name}", f"econComponent.{var_name}")
        self.connect('ivc.eff_array','econComponent.efficiency')
        # geo to econ
        self.connect('geometryComponent.m_m', 'econComponent.m_m')

        # dynm to econ
        self.connect('dynamicsComponent.P_elec', 'econComponent.P_elec')

        #self.connect('dynamicsComponent.P_var','outcome.P_var')
        #self.connect('econComponent.LCOE', 'outcome.LCOE')
        return

top = om.Problem(model=waveEngergy())

top.driver = om.ScipyOptimizeDriver()
top.driver.options['optimizer'] = 'SLSQP'
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
#add constraints.
top.model.add_constraint()

top.model.add_design_var('ivc.D_f',  lower = 6, upper = 40)
top.model.add_design_var('ivc.D_s_over_D_f',lower = 0.01, upper = 0.99,adder= 0.01)
top.model.add_design_var('ivc.h_f_over_D_f',lower = 0.1, upper = 10)
top.model.add_design_var('ivc.T_s_over_h_s',lower = 0.01, upper = 0.99)
top.model.add_design_var('ivc.F_max',lower = 9 * 1e6, upper = 10 * 1e6, adder= 10000, scaler = 1.0) #new Value = (initial + adder ) * scaler
top.model.add_design_var('ivc.B_p',lower = 0.1 * 1e6, upper = 50 * 1e6, scaler = 1.0 )
top.model.add_design_var('ivc.w_n',lower=0.01, upper=40)
top.model.add_design_var('ivc.M', lower=0, upper=2)



top.driver.options['maxiter'] = 100  # Increase max iterations
top.driver.options['tol'] = 1e-6
top.model.add_objective('econComponent.LCOE',scaler=1)

top.setup()
top.run_driver()
top.model.list_outputs(val=True)
#om.n2(top)


#write_xdsm(top, filename='waveEnergy', out_format='pdf', show_browser=True, equations=True, include_solver=True,
#           quiet=False, output_side='left', include_indepvarcomps=True, class_names=False)
