import openmdao.api as om

from ratioComponent import ratioComponent
from geometryComponent import geometryComponent
from dynamicsComponent import dynamicsComponent
from structureComponent import structureComponent
from environmentComponent import environmentComponent
from econComponent import econComponent
from outcomeComponent import outputComponent
from inputs.parameters import parameters
from sharedVariables import openmdao_ivc
import numpy as np
from dynamicsNewComponent import DynamicsNewComponent

from omxdsm import write_xdsm
class waveEnergy(om.Group):
    def __init__(self, b, p=None, D_f=None, D_s_over_D_f=None, h_f_over_D_f=None, T_s_over_h_s=None, F_max=None, B_p=None, w_n=None,M=0):
        super().__init__()
        self.p = p
        self.b = b
        self.D_f = D_f
        self.D_s_over_D_f = D_s_over_D_f
        self.h_f_over_D_f = h_f_over_D_f
        self.T_s_over_h_s = T_s_over_h_s
        self.F_max = F_max
        self.B_p = B_p
        self.w_n = w_n
        self.M = M
        #self.parameter2 = parameter2
    def setup(self):
        if self.p == None:
            self.p = parameters()
        X = np.concatenate((self.b['X_noms'], [0]))
        ivc = openmdao_ivc(X, self.p, D_f=self.D_f, D_s_over_D_f=self.D_s_over_D_f, h_f_over_D_f=self.h_f_over_D_f, T_s_over_h_s = self.T_s_over_h_s, F_max = self.F_max, B_p = self.B_p, M = self.M)
        self.add_subsystem('ivc', ivc)
        self.add_subsystem('ratioComponent', ratioComponent())
        self.add_subsystem('geometryComponent', geometryComponent())
        self.add_subsystem('dynamicsComponent', DynamicsNewComponent())
        self.add_subsystem('structureComponent', structureComponent())
        self.add_subsystem('environmentComponent', environmentComponent())
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
        ivc_to_dynam = ['rho_w', 'g', 'JPD', 'Hs', 'Hs_struct', 'T', 'T_struct', 'power_max', 'eff_pto', 'D_f', 'F_max', 'B_p', 'w_n']

        for var_name in ivc_to_dynam:
            self.connect(f"ivc.{var_name}", f"dynamicsComponent.{var_name}")

        #ratio to dynam
        ratio_to_dymn = ['h_f', 'T_f','T_s', 'h_s', 'D_s']
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

        #ivc to environment
        ivc_to_env = ['distance', 's_points','f_points', 'd_points', 'SCC']
        for var_name in ivc_to_env:
            self.connect(f"ivc.{var_name}", f"environmentComponent.{var_name}")

        #geo to environment
        self.connect('geometryComponent.m_m', 'environmentComponent.steel')
        self.connect('geometryComponent.A_fiberglass','environmentComponent.fiberglass')

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
        self.connect('environmentComponent.eco_value','outcomeComponent.eco_value')
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

#om.n2(top)


#write_xdsm(top, filename='waveEnergy', out_format='pdf', show_browser=True, equations=True, include_solver=True,
#           quiet=False, output_side='left', include_indepvarcomps=True, class_names=False)
