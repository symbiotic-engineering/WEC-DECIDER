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
        self.add_input('F_surge_op', 0)

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

    def structures_one_case(
            F_heave, F_surge, sigma_max,
            h_s, T_s, D_s, D_f, D_f_in, num_sections, D_f_tu, D_d, L_dt, theta_dt, D_d_tu,
            t_s_r, I, A_c, A_lat_sub, t_bot, t_top, t_d, t_d_tu, h_d, A_dt,
            h_stiff_f, w_stiff_f, h_stiff_d, w_stiff_d,
            rho_w, g, E, nu, num_terms_plate, radial_mesh_plate, num_stiff_d
    ):
        # Placeholder for the function logic
        # You would implement the actual computation here
        FOS1Y = None
        FOS2Y = None
        FOS3Y = None
        FOS_spar_local = None

        return FOS1Y, FOS2Y, FOS3Y, FOS_spar_local

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
        F_heave_storm = inputs['F_heave_storm']
        F_surge_storm = inputs['F_surge_storm']
        F_heave_op = inputs['F_heave_op']
        F_surge_op = inputs['F_surge_op']

        # bulk dimensions
        h_s = inputs['h_s']
        T_s = inputs['T_s']
        D_s = inputs['D_s']
        D_f = inputs['D_f']
        D_f_in = inputs['D_f_in']
        num_sections = inputs['num_sections']
        D_f_tu = inputs['D_f_tu']
        D_d = inputs['D_d']
        L_dt = inputs['L_dt']
        theta_dt = inputs['theta_dt']
        D_d_tu = inputs['D_d_tu']

        # structural dimensions
        t_s_r = inputs['t_s_r']
        I = inputs['I']
        A_c = inputs['A_c']
        A_lat_sub=  inputs['A_lat_sub']
        t_bot = inputs['t_bot']
        t_top = inputs['t_top']
        t_d = inputs['t_d']
        t_d_tu = inputs['t_d_tu']
        h_d = inputs['h_d']
        A_dt = inputs['A_dt']

        # stiffener dimensions
        h_stiff_f = inputs['h_stiff_f']
        w_stiff_f = inputs['w_stiff_f']
        h_stiff_d = inputs['h_stiff_d']
        w_stiff_d = inputs['w_stiff_d']

        # Constant
        M = int(inputs['M'])
        rho_w = inputs['rho_w']
        g = inputs['g']
        sigma_y = inputs['sigma_y']
        sigma_e = inputs['sigma_y']
        E = inputs['E']
        nu = inputs['nu']

        # plate hyperparameters
        num_terms_plate = inputs['num_terms_plate']
        radial_mesh_plate = inputs['radial_mesh_plate']
        num_stiff_d = inputs['num_stiff_d']

        F_heave_peak = max(F_heave_storm, F_heave_op)
        F_surge_peak = max(F_surge_storm, F_surge_op)

        # Inputs for both DLCs
        shared_inputs = [
            h_s, T_s, D_s, D_f, D_f_in, num_sections, D_f_tu, D_d, L_dt, theta_dt, D_d_tu,
            t_s_r, I, A_c, A_lat_sub, t_bot, t_top, t_d, t_d_tu, h_d, A_dt,
            h_stiff_f, w_stiff_f, h_stiff_d, w_stiff_d,
            rho_w, g, E[M], nu[M], num_terms_plate, radial_mesh_plate, num_stiff_d
        ]

        # DLC 1: peak
        sigma_buckle = sigma_y[M]  # TODO: for ultimate, implement ABS buckling formulas
        sigma_u = np.sqrt(sigma_y[M] * sigma_buckle)

        FOS1Y, FOS2Y, FOS3Y, FOS_buckling = self.structures_one_case(
            F_heave_peak, F_surge_peak, sigma_u, *shared_inputs
        )

        # DLC 2: endurance limit (long cycle fatigue)
        FOS1Y[1], FOS2Y[1], FOS3Y[1], FOS_buckling[1] = self.structures_one_case(
            F_heave_op, F_surge_op, sigma_e[M], *shared_inputs
        )


        # added [0]
        outputs['FOS1Y'] = FOS1Y
        outputs['FOS2Y'] = FOS2Y
        outputs['FOS3Y'] = FOS3Y
        outputs['FOS_buckling'] = FOS_buckling






