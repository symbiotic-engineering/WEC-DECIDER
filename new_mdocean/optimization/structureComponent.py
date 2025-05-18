import openmdao.api as om
import numpy as np
class structureComponent(om.ExplicitComponent):
    """
function [FOS1Y,FOS2Y,FOS3Y,FOS_buckling] = structures(...
          	F_heave_storm, F_surge_storm, F_heave_op, F_surge_op, ...                       % forces
            h_s, T_s, D_s, D_f, D_f_in, num_sections, D_f_tu, D_d, L_dt, theta_dt,D_d_tu,...% bulk dimensions
            t_s_r, I, A_c, A_lat_sub, t_bot, t_top, t_d, t_d_tu, h_d, A_dt, ...             % structural dimensions
            h_stiff_f, w_stiff_f, h_stiff_d, w_stiff_d, ...                                 % stiffener dimensions
            M, rho_w, g, sigma_y, sigma_e, E, nu, ...                                       % constants
            num_terms_plate, radial_mesh_plate, num_stiff_d)                                % plate hyperparameters
    """
    def setup(self):
        #Inputs

        # forces
        self.add_input('F_heave_storm', 0)
        self.add_input('F_surge_storm', 0)
        self.add_input('F_heave_op', 0)

        # bulk dimensions
        self.add_input('h_s', 0)
        self.add_input('T_s', 0)
        self.add_input('D_s', 0)
        self.add_input('D_f', 0)
        self.add_input('D_f_in', 0)
        self.add_input('num_sections', 0)
        self.add_input('D_f_tu', 0)
        self.add_input('D_d', 0)
        self.add_input('L_dt', 0)
        self.add_input('theta_dt', 0)
        self.add_input('D_d_tu', 0)

        # structural dimensions
        # t_s_r, I, A_c, A_lat_sub, t_bot, t_top, t_d, t_d_tu, h_d, A_dt,
        self.add_input('t_s_r', 0)
        self.add_input('I', np.zeros(3,))
        self.add_input('A_c', np.zeros(3,))
        self.add_input('A_lat_sub', np.zeros(3,))
        self.add_input('t_bot',0)
        self.add_input('t_top', 0)
        self.add_input('t_d', 0)
        self.add_input('t_d_tu', 0)
        self.add_input('h_d', 0)
        self.add_input('A_dt', 0)

        # stiffener dimensions
        # h_stiff_f, w_stiff_f, h_stiff_d, w_stiff_d, ...
        self.add_input('h_stiff_f', 0)
        self.add_input('w_stiff_f', 0)
        self.add_input('h_stiff_d', 0)
        self.add_input('w_stiff_d', 0)

        # Constant,
        # M, rho_w, g, sigma_y, sigma_e, E, nu, ...
        self.add_input('M', 0)
        self.add_input('rho_w', 0)
        self.add_input('g', 0)
        self.add_input('sigma_y', 0)
        self.add_input('sigma_e', 0)
        self.add_input('E', 0)
        self.add_input('nu', 0)

        # plate hyperparameters
        #  num_terms_plate, radial_mesh_plate, num_stiff_d
        self.add_input('num_terms_plate', 0)
        self.add_input('radial_mesh_plate', 0)
        self.add_input('num_stiff_d', 0)

        #Outputs
        self.add_output('FOS1Y')
        self.add_output('FOS2Y')
        self.add_output('FOS3Y')
        self.add_output('FOS_buckling')

        self.declare_partials('*', '*', method='fd')

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




