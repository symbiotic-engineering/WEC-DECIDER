import openmdao.api as om

from geometryComponent import geometryComponent
from dynamicsComponent import dynamicsComponent
from structureComponent import structureComponent
from econComponent import econComponent

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
        self.add_subsystem('geometryComponent', geometryComponent())
        self.add_subsystem('dynamicsComponent', dynamicsComponent())
        self.add_subsystem('structureComponent', structureComponent())
        self.add_subsystem('econComponent', econComponent())


    def configure(self):
        #ivc to geometryComponent

        # connect ivc to geo
        ivc_to_geo = [
            'D_s', 'D_f', 'T_f', 'h_f', 'h_s', 't_ft', 't_fr', 't_fc',
            't_fb', 't_sr', 't_dt', 'D_d', 'D_dt', 'theta_dt', 'T_s',
            'h_d', 'M', 'rho_m', 'rho_w', 'm_scale'
        ]

        for var_name in ivc_to_geo:
            self.connect(f"ivc.{var_name}", f"geometryComponent.{var_name}")

        # Dynamics
        #ivc to dynamics
        ivc_to_dynam = ['rho_w', 'g', 'JPD', 'Hs', 'Hs_struct', 'T', 'T_struct', 'power_max', 'eff_pto', 'D_f', 'F_max', 'B_p',
                        'w_n', 'h_f', 'T_f', 'T_s', 'h_s']

        for var_name in ivc_to_dynam:
            self.connect(f"ivc.{var_name}", f"dynamicsComponent.{var_name}")

        #geo_to_dynam
        self.connect('geometryComponent.m_f_tot', 'dynamicsComponent.m_float')
        self.connect('geometryComponent.V_d', 'dynamicsComponent.V_d')
        self.connect('geometryComponent.T', 'dynamicsComponent.draft')

        # Structure
        # ivc to structure
        ivc_to_struct = ['M', 'h_s', 'T_s', 'rho_w', 'g', 'sigma_y', 'E']

        for var_name in ivc_to_struct:
            self.connect(f"ivc.{var_name}", f"structureComponent.{var_name}")

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
        return

top = om.Problem(model=waveEngergy())
top.setup()

top.model.add_design_var('ivc.D_f')
top.model.add_design_var('ivc.D_s_over_D_f')
top.model.add_design_var('ivc.h_f_over_D_f')
top.model.add_design_var('ivc.T_s_over_h_s')
top.model.add_design_var('ivc.F_max')
top.model.add_design_var('ivc.B_p')
top.model.add_design_var('ivc.w_n')
top.model.add_design_var('ivc.M')

top.model.add_objective('econComponent.LCOE')
top.run_model()
#top.model.add_objective()
#print(top.model.get_val('t_sr'))
#print(top.model.get_val('econComponent.LCOE'))
#print(top.model.get_val('dynamicsComponent.P_var'))
#print(top.model.get_val('econComponent.LCOE'))


write_xdsm(top, filename='waveEnergy', out_format='pdf', show_browser=True, equations=True, include_solver=True,
           quiet=False, output_side='left', include_indepvarcomps=True, class_names=False)

#class waveEnergy(om.Group):
#    pass


"""
    om.IndepVarComp
    om.ExecComp ("b * 2")
    om.ExplicitComponent
"""
"""
    # input
    in_params = p.copy()
    in_params['D_f'] = X[0]
    D_s_over_D_f = X[1]
    h_f_over_D_f = X[2]
    T_s_over_h_s = X[3]
    in_params['F_max'] = X[4] * 1e6
    in_params['B_p'] = X[5] * 1e6
    in_params['w_n'] = X[6]
    #Change float to Int
    in_params['M'] = int(X[7])
    
    #output
    LCOE , P_Var, G (output)
    """
