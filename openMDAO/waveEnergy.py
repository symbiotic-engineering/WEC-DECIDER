import openmdao.api as om

from ratioComponent import ratioComponent
from geometryComponent import geometryComponent
from dynamicsComponent import dynamicsComponent
from hydroComponent import hydroComponent
from heavenDynamicsComponent import heavenDynamicsComponent
from surgeAndVariationComponent import surgeAndVariationComponent
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
    def __init__(self, b, p=None, D_f=None, D_s_over_D_f=None, h_f_over_D_f=None, T_s_over_h_s=None, F_max=None, B_p=None, w_n=None,M=0, dynamic_version='old'):
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
        self.dynamic_version = dynamic_version
        #self.parameter2 = parameter2
    def setup(self):
        if self.p == None:
            self.p = parameters()
        X = np.concatenate((self.b['X_noms'], [0]))
        ivc = openmdao_ivc(X, self.p, D_f=self.D_f, D_s_over_D_f=self.D_s_over_D_f, h_f_over_D_f=self.h_f_over_D_f, T_s_over_h_s = self.T_s_over_h_s, F_max = self.F_max, B_p = self.B_p, M = self.M)
        self.add_subsystem('ivc', ivc)
        self.add_subsystem('ratioComponent', ratioComponent())
        self.add_subsystem('geometryComponent', geometryComponent())
        self.add_subsystem('hydroComponent', hydroComponent())
        self.add_subsystem('heavenDynamicsComponent', heavenDynamicsComponent())
        self.add_subsystem('surgeAndVariationComponent', surgeAndVariationComponent())
        #print(self.dynamic_version)
        """
        if self.dynamic_version == 'old':
            print("old dynam")
            self.add_subsystem('dynamicsComponent', dynamicsComponent())
        """

        #print("new dynam")
        #self.add_subsystem('dynamicsComponent', DynamicsNewComponent())
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
        ivc_to_hydro = ['D_f']
        for var_name in ivc_to_hydro:
            self.connect(f"ivc.{var_name}", f"hydroComponent.{var_name}")

        #ratio_to_hydro
        ratio_to_hydro = ['h_f','D_s','T_f',]
        for var_name in ratio_to_hydro:
            self.connect(f"ratioComponent.{var_name}", f"hydroComponent.{var_name}")

        #ivc to heavendynam
        ivc_to_heavendynam = ['g','rho_w','F_max','Hs_struct','T_struct']
        for var_name in ivc_to_heavendynam:
            self.connect(f"ivc.{var_name}", f"heavenDynamicsComponent.{var_name}")

        #hydro to heavendynam
        #self.connect('hydroComponent.RM3','heavenDynamicsComponent.RM3')
        self.connect('hydroComponent.ndof', 'heavenDynamicsComponent.ndof')
        self.connect('hydroComponent.added_mass', 'heavenDynamicsComponent.added_mass')
        self.connect('hydroComponent.radiation_damping', 'heavenDynamicsComponent.radiation_damping')
        self.connect('hydroComponent.diffraction_force', 'heavenDynamicsComponent.diffraction_force')
        self.connect('hydroComponent.Froude_Krylov_force', 'heavenDynamicsComponent.Froude_Krylov_force')
        self.connect('hydroComponent.excitation_force', 'heavenDynamicsComponent.excitation_force')
        self.connect('hydroComponent.inertia_matrix', 'heavenDynamicsComponent.inertia_matrix')
        self.connect('hydroComponent.hydrostatic_stiffness', 'heavenDynamicsComponent.hydrostatic_stiffness')

        #self.connect('hydroComponent.g', 'heavenDynamicsComponent.g')
        #self.connect('hydroComponent.rho', 'heavenDynamicsComponent.rho')
        self.connect('hydroComponent.water_depth', 'heavenDynamicsComponent.water_depth')
        self.connect('hydroComponent.forward_speed', 'heavenDynamicsComponent.forward_speed')
        self.connect('hydroComponent.wave_direction', 'heavenDynamicsComponent.wave_direction')
        self.connect('hydroComponent.omega', 'heavenDynamicsComponent.omega')
        self.connect('hydroComponent.period', 'heavenDynamicsComponent.period')

        #ivc to surgeAndVariation
        ivc_to_surgeAndVariation = ['F_max', 'D_f', 'Hs_struct', 'T','T_struct', 'rho_w','g', 'JPD','B_p','w_n']
        for var_name in ivc_to_surgeAndVariation:
            self.connect(f"ivc.{var_name}", f"surgeAndVariationComponent.{var_name}")
        #ratio to surgeAndVariation
        ratio_to_surgeAndVariation = ['h_f','T_f','T_s', 'h_s']
        for var_name in ratio_to_surgeAndVariation:
            self.connect(f"ratioComponent.{var_name}", f"surgeAndVariationComponent.{var_name}")

        #heavendynam to surgeAndVariation
        heavendynam_to_surgeAndVariation = ['P_matrix','P_elec']
        for var_name in heavendynam_to_surgeAndVariation:
            self.connect(f"heavenDynamicsComponent.{var_name}", f"surgeAndVariationComponent.{var_name}")

        #geo_to_surgeAndVariation
        self.connect('geometryComponent.m_f_tot', 'surgeAndVariationComponent.m_float')
        self.connect('geometryComponent.V_d', 'surgeAndVariationComponent.V_d')
        self.connect('geometryComponent.T', 'surgeAndVariationComponent.draft')


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

        # surgeAndVariation, heavenDynamicsComponent to structure
        self.connect('heavenDynamicsComponent.F_heave_max', 'structureComponent.F_heave')
        self.connect('surgeAndVariationComponent.F_surge_max','structureComponent.F_surge')

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

        # surgeAndVariationComponent to econ
        self.connect('heavenDynamicsComponent.P_elec', 'econComponent.P_elec')


        #Connect to outcome component
        self.connect('environmentComponent.eco_value','outcomeComponent.eco_value')
        self.connect('econComponent.LCOE', 'outcomeComponent.LCOE')
        self.connect('surgeAndVariationComponent.P_var','outcomeComponent.P_var')
        self.connect('geometryComponent.V_f_pct', 'outcomeComponent.V_f_pct')
        self.connect('geometryComponent.V_s_pct','outcomeComponent.V_s_pct')
        self.connect('geometryComponent.GM', 'outcomeComponent.GM')
        self.connect('structureComponent.FOS1Y', 'outcomeComponent.FOS1Y')
        self.connect('structureComponent.FOS2Y', 'outcomeComponent.FOS2Y')
        self.connect('structureComponent.FOS3Y', 'outcomeComponent.FOS3Y')
        self.connect('structureComponent.FOS_buckling', 'outcomeComponent.FOS_buckling')
        self.connect('ivc.FOS_min', 'outcomeComponent.FOS_min')
        self.connect('heavenDynamicsComponent.P_elec', 'outcomeComponent.P_elec')
        self.connect('ratioComponent.D_d','outcomeComponent.D_d')
        self.connect('ivc.D_d_min','outcomeComponent.D_d_min')
        self.connect('surgeAndVariationComponent.h_s_extra', 'outcomeComponent.h_s_extra')
        self.connect('ivc.LCOE_max', 'outcomeComponent.LCOE_max')
        self.connect('surgeAndVariationComponent.F_ptrain_max', 'outcomeComponent.F_ptrain_max')
        self.connect('ivc.F_max', 'outcomeComponent.F_max')
        return

#om.n2(top)


#write_xdsm(top, filename='waveEnergy', out_format='pdf', show_browser=True, equations=True, include_solver=True,
#           quiet=False, output_side='left', include_indepvarcomps=True, class_names=False)
