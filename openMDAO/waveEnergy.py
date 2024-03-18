import openmdao.api as om

from ratioComponent import ratioComponent
from geometryComponent import geometryComponent
from dynamicsComponent import dynamicsComponent
from structureComponent import structureComponent
from econComponent import econComponent
from outcomeComponent import outputComponent


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
        self.add_subsystem('outcomeComponent', outputComponent())


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


        #Connect to outcome component
        self.connect('econComponent.LCOE', 'outcomeComponent.LCOE')
        self.connect('dynamicsComponent.P_var','outcomeComponent.P_var')
        self.connect('geometryComponent.V_f_pct', 'outcomeComponent.V_f_pct')
        self.connect('geometryComponent.V_s_pct','outcomeComponent.V_s_pct')
        self.connect('geometryComponent.GM', 'outcomeComponent.GM')
        self.connect('structureComponent.FOS1Y', 'outcomeComponent.FOS1Y')
        self.connect('structureComponent.FOS2Y', 'outcomeComponent.FOS2Y')
        self.connect('structureComponent.FOS3Y', 'outcomeComponent.FOS3Y')
        self.connect('structureComponent.FOS_buckling', 'outcomeComponent.FOS_buckling')
        self.connect('ivc.FOS_min', 'outcomeComponent.FOS_min')
        self.connect('dynamicsComponent.P_elec', 'outcomeComponent.P_elec')
        self.connect('ratioComponent.D_d','outcomeComponent.D_d')
        self.connect('ivc.D_d_min','outcomeComponent.D_d_min')
        self.connect('dynamicsComponent.h_s_extra', 'outcomeComponent.h_s_extra')
        self.connect('ivc.LCOE_max', 'outcomeComponent.LCOE_max')
        self.connect('dynamicsComponent.F_ptrain_max', 'outcomeComponent.F_ptrain_max')
        self.connect('ivc.F_max', 'outcomeComponent.F_max')
        return

top = om.Problem(model=waveEngergy())

top.driver = om.ScipyOptimizeDriver()
top.driver.options['optimizer'] = 'SLSQP'






top.model.add_design_var('ivc.D_f',  lower = 6, upper = 40)
top.model.add_design_var('ivc.D_s_over_D_f',lower = 0.01, upper = 0.99)
top.model.add_design_var('ivc.h_f_over_D_f',lower = 0.1, upper = 10)
top.model.add_design_var('ivc.T_s_over_h_s',lower = 0.01, upper = 0.99)
top.model.add_design_var('ivc.F_max',lower = 0.01 * 1e6, upper = 10 * 1e6) #new Value = (initial + adder ) * scaler
top.model.add_design_var('ivc.B_p',lower = 0.1 * 1e6, upper = 50 * 1e6)
top.model.add_design_var('ivc.w_n',lower=0.01, upper=40)
top.model.add_design_var('ivc.M', lower=0, upper=2)



top.driver.options['maxiter'] = 300  # Increase max iterations
top.driver.options['tol'] = 1e-6
top.model.add_objective('outcomeComponent.LCOE',scaler=1)
#add constraints.
top.model.add_constraint('outcomeComponent.g_0', lower= 0)
top.model.add_constraint('outcomeComponent.g_1', lower= 0)
top.model.add_constraint('outcomeComponent.g_2', lower= 0)
top.model.add_constraint('outcomeComponent.g_3', lower= 0)
top.model.add_constraint('outcomeComponent.g_4', lower= 0)
top.model.add_constraint('outcomeComponent.g_5', lower= 0)
top.model.add_constraint('outcomeComponent.g_6', lower= 0)
top.model.add_constraint('outcomeComponent.g_7', lower= 0)
top.model.add_constraint('outcomeComponent.g_8', lower= 0)
top.model.add_constraint('outcomeComponent.g_9', lower= 0)
top.model.add_constraint('outcomeComponent.g_10', lower= 0)
top.model.add_constraint('outcomeComponent.g_11', lower= 0)
top.model.add_constraint('outcomeComponent.g_12', lower= 0)
top.model.add_constraint('outcomeComponent.g_13', lower= 0)

top.setup()
top.run_driver()
top.model.list_outputs(val=True)
#om.n2(top)


#write_xdsm(top, filename='waveEnergy', out_format='pdf', show_browser=True, equations=True, include_solver=True,
#           quiet=False, output_side='left', include_indepvarcomps=True, class_names=False)
