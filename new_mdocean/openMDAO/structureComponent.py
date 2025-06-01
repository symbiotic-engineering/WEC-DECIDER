import openmdao.api as om
import numpy as np
from scipy.interpolate import interp1d

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

    # float_plate_stress.m
    def float_plate_stress(self, D_f, D_f_in, F_heave, num_sections, t_bot, t_top, h_stiff, w_stiff, D_f_tu, nu):
        # Area of annular plate
        A = np.pi / 4 * (D_f ** 2 - D_f_in ** 2)
        P_hydrodynamic = F_heave / A
        W = F_heave / num_sections

        b_out = np.pi * D_f / num_sections
        b_in = np.pi * D_f_in / num_sections
        b_avg = np.sqrt(b_out * b_in)
        h = (D_f - D_f_in) / 2

        # Trapezoidal interpolation (see notebook reference)
        m = (b_out - b_in) / (2 * h)
        if h >= b_avg:
            shorter_length = b_in * (m + np.sqrt(1 + m ** 2))
            longer_length = h
        else:
            shorter_length = h
            longer_length = (b_out + b_in) / 2 + h * (1 - np.sqrt(1 + m ** 2))

        length_ratio = longer_length / shorter_length
        r0 = 1.02  # This is hardcoded per original code
        width_plate = b_out  # FIXME: Should combine b_out and b_in?

        # Call the helper functions (TODO)
        sigma_vm_bot = self.bottom_plate_stress(length_ratio, shorter_length, P_hydrodynamic, t_bot, h_stiff, w_stiff,
                                           width_plate)
        sigma_vm_top = self.top_plate_stress(length_ratio, shorter_length, W, t_top, h_stiff, w_stiff, r0, width_plate, nu)

        return sigma_vm_bot, sigma_vm_top

    def bottom_plate_stress(self,length_ratio, shorter_length, P_hydrodynamic, t_bot, h_stiff, w_stiff, width_plate):
        # Data from Timoshenko table 35 p219
        length_ratio_data = np.concatenate((np.arange(1, 2.1, 0.1), [1000]))
        alpha_shorter_data = -0.0001 * np.array([513, 581, 639, 687, 726, 757, 780, 799, 812, 822, 829, 833])
        alpha_longer_data = -0.0001 * np.array([513, 538, 554, 563, 568, 570, 571, 571, 571, 571, 571, 571])

        alpha_shorter_fcn = interp1d(length_ratio_data, alpha_shorter_data, kind='linear', fill_value='extrapolate')
        alpha_longer_fcn = interp1d(length_ratio_data, alpha_longer_data, kind='linear', fill_value='extrapolate')

        M_shorter = alpha_shorter_fcn(length_ratio) * P_hydrodynamic * shorter_length ** 2
        M_longer = alpha_longer_fcn(length_ratio) * P_hydrodynamic * shorter_length ** 2

        sigma_shorter = self.stiffened_plate_stress(t_bot, h_stiff, width_plate, w_stiff, M_shorter)
        sigma_longer = self.stiffened_plate_stress(t_bot, h_stiff, width_plate, w_stiff, M_longer)

        sigma_zz = P_hydrodynamic

        # Von Mises stress calculation
        s_11 = sigma_shorter
        s_22 = 0
        s_33 = sigma_zz
        sigma_vm = np.sqrt(0.5 * ((s_11 - s_22) ** 2 + (s_22 - s_33) ** 2 + (s_33 - s_11) ** 2))

        return sigma_vm

    def top_plate_stress(self, length_ratio, shorter_length, W, t_top, h_stiff, w_stiff, r0, width_plate, nu):
        # Interpolation data
        length_ratio_data = np.array(list(np.arange(1, 2.2, 0.2)) + [1000])
        beta_1_data = np.array([-0.238, -0.078, 0.011, 0.053, 0.068, 0.067, 0.067])
        beta_2_data = np.array([0.7542, 0.8940, 0.9624, 0.9906, 1.0000, 1.004, 1.008])

        beta_1 = np.interp(length_ratio, length_ratio_data, beta_1_data)
        beta_2 = np.interp(length_ratio, length_ratio_data, beta_2_data)

        # Roark's table 11.4 case 8b page 514
        # Note: This assumes force in middle; may not apply for edge forces (~no bending moment)
        # Check references 26 and 21 for more general (off-center) equations
        sigma_edge_no_stiff = 3 * W / (2 * np.pi * t_top ** 2) * (
                    (1 + nu) * np.log(2 * shorter_length / (np.pi * r0)) + beta_1)
        sigma_cent_no_stiff = -beta_2 * W / t_top ** 2

        M_edge = sigma_edge_no_stiff * t_top ** 2 / 6
        M_cent = sigma_cent_no_stiff * t_top ** 2 / 6

        M_max = max(abs(M_edge), abs(M_cent))

        sigma_max = self.stiffened_plate_stress(t_top, h_stiff, width_plate, w_stiff, M_max)
        sigma_vm_top = sigma_max

        return sigma_vm_top

    def stiffened_plate_stress(self,t_plate, h_stiff, width_plate, width_stiff, moment_per_length):
        h_eq, y_max = self.get_stiffener_equivalent_properties(t_plate, h_stiff, width_plate, width_stiff)
        sigma = self.get_plate_stress(moment_per_length, y_max, h_eq)
        return sigma


    # get_plate_stress.m
    def get_plate_stress(self, moment_per_length, y_max, h_eq):
        # Formula from notebook p21 12/15/24
        M = moment_per_length
        sigma = 12 * M * y_max / h_eq ** 3
        return sigma

    # get_stiffener_equivalent_properties.m
    def get_stiffener_equivalent_properties(self, t_plate, h_stiff, width_plate, width_stiff):
        # Determine shape type
        if len(h_stiff) == 1 and len(width_stiff) == 1:
            shape = 'tee'
        elif len(h_stiff) == 2 and len(width_stiff) == 2:
            shape = 'I'
        elif len(h_stiff) == 4 and len(width_stiff) == 4:
            shape = 'doubleI'
        else:
            raise ValueError('invalid input length')
        centroid, h_eq_over_h_3, height_max = None,None, None
        if shape == 'tee':
            centroid, h_eq_over_h_3, height_max = self.T_beam_properties(t_plate, h_stiff, width_plate, width_stiff)
        elif shape == 'I':
            centroid, h_eq_over_h_3, height_max = self.I_beam_properties(t_plate, h_stiff, width_plate, width_stiff)
        elif shape == 'doubleI':
            centroid, h_eq_over_h_3, height_max = self.double_I_beam_properties(t_plate, h_stiff, width_plate, width_stiff)

        # Ensure h_eq_over_h_3 is not less than 1
        h_eq_over_h_3 = np.maximum(h_eq_over_h_3, 1)

        h = t_plate
        h_eq_over_h = h_eq_over_h_3 ** (1 / 3)
        h_eq = h_eq_over_h * h

        y_max = max(centroid, height_max - centroid)  # max distance from neutral axis

        I = (1 / 12) * width_plate * h_eq ** 3  # Moment of inertia
        S_eq = I / y_max  # Section modulus

        return h_eq, y_max, S_eq

    def T_beam_properties(self, t_plate, h_stiff, width_plate, width_stiff):
        # Inputs:
        # t_plate: plate thickness (scalar)
        # h_stiff: stiffener height (scalar or array)
        # width_plate: plate width (scalar)
        # width_stiff: stiffener width (scalar or array)

        h = t_plate  # plate thickness
        H = np.array(h_stiff)  # stiffener height (ensure array)
        a = width_plate / 2
        b = np.array(width_stiff)  # stiffener width (ensure array)

        # MIT 2.080 Eq. 7.67 (with corrections): neutral axis location
        eta_over_h = 0.5 * (1 - (b / (2 * a)) * (H / h) ** 2) / (1 + (b / (2 * a)) * (H / h))
        eta = eta_over_h * h  # neutral axis from stiffened face
        centroid = h - eta  # centroid from unstiffened face

        # MIT 2.080 Eq. 7.69 (with corrections): equivalent height cubed
        term1 = 1 - 3 * eta_over_h + 3 * eta_over_h ** 2
        term2 = (H / h) ** 3 + 3 * (H / h) ** 2 * eta_over_h + 3 * (H / h) * eta_over_h ** 2
        h_eq_over_h_3 = 4 * (term1 + (b / (2 * a)) * term2)

        height_max = h + H  # total height (plate + stiffener)

        return centroid, h_eq_over_h_3, height_max
    def I_beam_properties(self, t_plate, h_stiff, width_plate, width_stiff):
        # First moments and areas
        M0, A0 = self.first_moment_of_area_rectangle(t_plate, width_plate, t_plate / 2)
        M1, A1 = self.first_moment_of_area_Ibeam(t_plate, h_stiff[:2], width_stiff[:2])

        moment = M0 + M1
        area = A0 + A1
        centroid = moment / area  # Centroid from bottom of plate

        # Second moments
        I0 = self.second_moment_of_area_rectangle(t_plate, width_plate, 0)
        I1 = self.second_moment_of_area_Ibeam(t_plate, h_stiff[:2], width_stiff[:2])
        I = I0 + I1  # Second moment of area about bottom

        # Equivalent height ratio
        h_eq_over_h_3 = self.equiv_I_beam(area, centroid, I, width_plate, t_plate)

        height_max = t_plate + sum(h_stiff)  # Total height

        return centroid, h_eq_over_h_3, height_max

    def double_I_beam_properties(self, t_plate, h_stiff, width_plate, width_stiff):
        # Get first moment of area and area of the plate
        M0, A0 = self.first_moment_of_area_rectangle(t_plate, width_plate, t_plate / 2)
        M1, A1 = self.first_moment_of_area_Ibeam(t_plate, h_stiff[0:2], width_stiff[0:2])
        M2, A2 = self.first_moment_of_area_Ibeam(t_plate, h_stiff[2:4], width_stiff[2:4])

        moment = M0 + M1 + M2
        area = A0 + A1 + A2
        centroid = moment / area  # centroid from bottom of plate

        # Second moment of area
        I0 = self.second_moment_of_area_rectangle(t_plate, width_plate, 0)
        I1 = self.second_moment_of_area_Ibeam(t_plate, h_stiff[0:2], width_stiff[0:2])
        I2 = self.second_moment_of_area_Ibeam(t_plate, h_stiff[2:4], width_stiff[2:4])
        I = I0 + I1 + I2

        # Equivalent I-beam height ratio
        h_eq_over_h_3 = self.equiv_I_beam(area, centroid, I, width_plate, t_plate)

        # Maximum height
        height_max = t_plate + max(sum(h_stiff[0:2]), sum(h_stiff[2:4]))

        return centroid, h_eq_over_h_3, height_max

    def first_moment_of_area_rectangle(self, height, width, height_start):
        M = width * height * (height_start + height / 2)  # First moment of area
        A = width * height  # Area of rectangle
        return M, A

    def first_moment_of_area_Ibeam(self, t_plate, h_stiff, width_stiff):
        M1, A1 = self.first_moment_of_area_rectangle(h_stiff[0], width_stiff[0], t_plate + 0.5 * h_stiff[0])
        M2, A2 = self.first_moment_of_area_rectangle(h_stiff[1], width_stiff[1], t_plate + h_stiff[0] + 0.5 * h_stiff[1])
        M = M1 + M2
        A = A1 + A2
        return M, A
    def equiv_I_beam(self, area, centroid, I, width_plate, t_plate):
        I_centroid = I - area * centroid ** 2  # Parallel axis theorem
        h_eq_3 = 12 * I_centroid / width_plate
        h_eq_over_h_3 = h_eq_3 / t_plate ** 3
        return h_eq_over_h_3

    def second_moment_of_area_rectangle(self,height, width, height_start):
        I = width / 3 * ((height_start + height) ** 3 - (height_start) ** 3)
        return I

    def second_moment_of_area_Ibeam(self, t_plate, h_stiff, width_stiff):
        I1 = self.second_moment_of_area_rectangle(h_stiff[0], width_stiff[0], t_plate)
        I2 = self.second_moment_of_area_rectangle(h_stiff[1], width_stiff[1], t_plate + h_stiff[0])
        I = I1 + I2
        return I

    def spar_combined_buckling(self, F, E, I, L, D, A, t, q, sigma_0, nu):
        # Euler buckling calculation
        K = 2  # fixed-free case
        F_buckling = (np.pi ** 2) * E * I / (K * L) ** 2
        sigma_EA = F_buckling / A

        # Hoop stress calculation
        sigma_theta = q * D / (2 * t)

        # Local buckling of plate element
        k_s = 1.33  # fixed-free, uniform compression
        s = D  # assumed for now, needs verification
        P_r = 0.6  # steel proportional limit
        sigma_Ex = k_s * (np.pi ** 2) * E / (12 * (1 - nu ** 2)) * (t / s) ** 2

        # Buckling stress limit
        if sigma_Ex <= P_r * sigma_0:
            sigma_Cx = sigma_Ex
        else:
            sigma_Cx = sigma_0 * (1 - P_r * (1 - P_r) * sigma_0 / sigma_Ex)

        # Failure stress: yield or buckling
        compact = True  # assumption
        sigma_F = sigma_0 if compact else sigma_Cx

        # Combined buckling and hoop stress check
        sigma_EA_thresh = P_r * sigma_F * (1 - sigma_theta / sigma_F)
        pure_buckling = sigma_EA <= sigma_EA_thresh

        if pure_buckling:
            sigma_C_A_theta = sigma_EA
        else:
            zeta = 1 - P_r * (1 - P_r) * sigma_F / sigma_EA - sigma_theta / sigma_F
            omega = 0.5 * sigma_theta / sigma_F * (1 - 0.5 * sigma_theta / sigma_F)
            Lambda = 0.5 * (zeta + np.sqrt(zeta ** 2 + 4 * omega))
            sigma_C_A_theta = sigma_F * Lambda

        sigma_ac = F / A + q
        FOS_spar = sigma_C_A_theta / sigma_ac

        # Local buckling (placeholder)
        FOS_spar_local = 3  # TODO: implement local buckling calculation

        return FOS_spar, FOS_spar_local

    import numpy as np

    def distributed_plate_nondim(self, a, b, F_heave, nu, rho):
        # Inputs: a = D_d/2, b = D_s/2, F_heave = vertical load, nu = Poisson's ratio, rho = nondim radial vector
        # Outputs: nondimensional deflection (w), radial moment (Mr), circumferential moment (Mt)

        A = np.pi * a ** 2  # area of plate
        q = F_heave / A  # uniform pressure
        r0 = b

        C2 = 1 / 4 * (1 - (b / a) ** 2 * (1 + 2 * np.log(a / b)))
        C3 = b / (4 * a) * (((b / a) ** 2 + 1) * np.log(a / b) + (b / a) ** 2 - 1)
        C8 = 1 / 2 * (1 + nu + (1 - nu) * (b / a) ** 2)
        C9 = b / a * ((1 + nu) / 2 * np.log(a / b) + (1 - nu) / 4 * (1 - (b / a) ** 2))
        L11 = 1 / 64 * (1 + 4 * (r0 / a) ** 2 - 5 * (r0 / a) ** 4 - 4 * (r0 / a) ** 2 * (2 + (r0 / a) ** 2) * np.log(
            a / r0))
        L17 = 1 / 4 * (1 - ((1 - nu) / 4) * (1 - (r0 / a) ** 4) - (r0 / a) ** 2 * (1 + (1 + nu) * np.log(a / r0)))

        r = rho * a
        F2 = 1 / 4 * (1 - (b / r) ** 2 * (1 + 2 * np.log(r / b)))
        F3 = b / (4 * r) * (((b / r) ** 2 + 1) * np.log(r / b) + (b / r) ** 2 - 1)
        F8 = 1 / 2 * (1 + nu + (1 - nu) * (b / r) ** 2)
        F9 = b / r * (0.5 * (1 + nu) * np.log(r / b) + 0.25 * (1 - nu) * (1 - (b / r) ** 2))

        ratio = r0 / r
        # bracket = (r > r0).astype(float)
        bracket = np.where(r > r0, 1.0, 0.0)
        G11 = 1 / 64 * (1 + 4 * ratio ** 2 - 5 * ratio ** 4 - 4 * ratio ** 2 * (2 + ratio ** 2) * np.log(
            1 / ratio)) * bracket
        G17 = 1 / 4 * (
                    1 - ((1 - nu) / 4) * (1 - ratio ** 4) - ratio ** 2 * (1 + (1 + nu) * np.log(1 / ratio))) * bracket

        Mrb = -q * a ** 2 / C8 * (C9 * (a ** 2 - r0 ** 2) / (2 * a * b) - L17)
        Qb = q / (2 * b) * (a ** 2 - r0 ** 2)

        # Deflection and moments
        y_over_D = Mrb * r ** 2 * F2 + Qb * r ** 3 * F3 - q * r ** 4 * G11
        w_nondim = y_over_D * 2 * np.pi / (F_heave * a ** 2)

        Mr = Mrb * F8 + Qb * r * F9 - q * r ** 2 * G17
        Mt = nu * Mr  # Approximate (ignoring tilt angle term)

        Mr_nondim = Mr * 2 * np.pi / F_heave
        Mt_nondim = Mt * 2 * np.pi / F_heave

        # Adjust signs for positive load convention
        return -w_nondim, -Mr_nondim, -Mt_nondim

    def damping_plate_structures(self,F_heave, D_d, D_s, P_hydrostatic, t_d, A_dt, theta_dt, L_dt, h_d, A_c, E, nu,
                                 h_stiff, width_stiff, D_d_tu, t_d_tu, N, radial_mesh_plate, num_stiffeners):
        a = D_d / 2  # outer radius
        b = D_s / 2  # inner radius

        # Radial and angular points for evaluation
        rho = np.linspace(b / a, 1, radial_mesh_plate)
        theta = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2])

        # Non-dimensional solutions for distributed and concentrated load
        delta_plate_dis_nondim_vec, Mr_dis_nondim_vec, Mt_dis_nondim_vec = self.distributed_plate_nondim(a, b, F_heave, nu,
                                                                                                    rho)
        delta_plate_con_nondim_vec, Mr_con_nondim_vec = self.concentrated_plate_nondim(b / a, nu, theta, rho, N)

        # Deflection at outer edge for compatibility
        delta_plate_dis_nondim = delta_plate_dis_nondim_vec[-1]
        delta_plate_con_nondim = np.sum(delta_plate_con_nondim_vec[-1, :])
        Mr_con_nondim = np.sum(Mr_con_nondim_vec, axis=1)

        # Stiffener properties
        num_unique_stiffeners = len(h_stiff) // 2
        num_stiffener_repeats = num_stiffeners / num_unique_stiffeners
        r = rho * a
        circumf = 2 * np.pi * r
        width_plate = circumf / num_stiffener_repeats

        h_eq_vec, y_max_vec, _ = self.get_stiffener_equivalent_properties(t_d, h_stiff, width_plate, width_stiff)
        h_eq = np.mean(h_eq_vec)  # approx: average equivalent height

        # Equivalent plate bending stiffness
        D_eq = E * h_eq ** 3 / (12 * (1 - nu ** 2))

        # Tube bending stiffness
        I_tube = np.pi / 64 * (D_d_tu ** 4 - (D_d_tu - 2 * t_d_tu) ** 4)
        K_tube = 6 * E * I_tube / (L_dt ** 2 * (D_d - D_s))

        # Compatibility: deflection match for plate-tube system
        F_tube = F_heave * delta_plate_dis_nondim / (D_eq / (a ** 2 * K_tube) - delta_plate_con_nondim)

        # Moment superposition: distributed + concentrated
        Mr_con = Mr_con_nondim.T @ F_tube
        Mr_dis = Mr_dis_nondim_vec * F_heave
        Mr = Mr_dis + Mr_con

        # Radial stress: use plate bending stress
        sigma_r_vec = self.get_plate_stress(Mr, y_max_vec, h_eq_vec)
        sigma_r = np.max(np.abs(sigma_r_vec))

        sigma_vm = sigma_r  # von Mises approx: ignore Mt

        return sigma_vm

    def concentrated_plate_nondim(self, lam, nu, theta, rho, N):
        assert np.isscalar(lam) and np.isscalar(nu) and np.isscalar(N)

        RHO, n = np.meshgrid(rho, np.arange(2, N + 1))
        d0 = -1 / 4
        c0 = lam / 4 * ((1 + 2 * np.log(lam)) * (1 + nu) - (3 + nu)) / (lam ** -1 * (1 + nu) + lam * (1 - nu))
        b0 = c0 / 2 * (1 - nu) / (1 + nu) + (3 + nu) / (8 * (1 + nu))
        a0 = -lam ** 2 * b0 - c0 * np.log(lam) - d0 * lam ** 2 * np.log(lam)

        d1 = 1 / 2
        b1 = -1 / 4 * (1 + nu + lam ** 2 * (1 - nu)) / (3 + nu + lam ** 4 * (1 - nu))
        c1 = -b1 * (3 + nu) / (1 - nu) - 1 / 4 * (1 + nu) / (1 - nu)
        a1 = -b1 * lam ** 2 - c1 * lam ** -2 - d1 * np.log(lam)

        A = (3 + nu) / (1 - nu)
        B = (1 - lam ** 2) ** 2 * (n ** 2 - 1) + (lam ** (-2 * n + 2) + A) * (lam ** (2 * n + 2) + A)
        dn = -((1 - lam ** 2) * (n - 1) + lam ** (2 * n + 2) + A) / (B * n * (n - 1) * (1 - nu))
        bn = ((1 - lam ** 2) * (n + 1) - lam ** (-2 * n + 2) - A) / (B * n * (n + 1) * (1 - nu))
        an = -lam ** 2 * (bn * (n + 1) / n + dn * lam ** (-2 * n) / n)
        cn = -lam ** 2 * (dn * (n - 1) / n - bn * lam ** (2 * n) / n)

        if np.isscalar(rho):
            abcd = np.vstack(([a0, b0, c0, d0], [a1, b1, c1, d1], np.column_stack([an, bn, cn, dn])))
        else:
            abcd = None

        rho = np.asarray(rho)
        Rho_zero = a0 + b0 * rho ** 2 + c0 * np.log(rho) + d0 * rho ** 2 * np.log(rho)
        Rho_one = a1 * rho + b1 * rho ** 3 + c1 * rho ** -1 + d1 * rho * np.log(rho)
        Rho_two_N = an * RHO ** n + bn * RHO ** (n + 2) + cn * RHO ** (-n) + dn * RHO ** (-n + 2)

        w_nondim = Rho_zero[:, None] + Rho_one[:, None] * np.cos(theta) + (Rho_two_N * np.cos(n.T * theta)).sum(axis=0)

        e0 = 2 * b0 * (1 + nu) - c0 * rho ** -2 * (1 - nu) + d0 * (3 + nu + 2 * (1 + nu) * np.log(rho))
        e1 = 2 * b1 * rho * (3 + nu) + 2 * c1 * rho ** -3 * (1 - nu) + d1 * rho ** -1 * (1 + nu)

        en_a = an * RHO ** (n - 2) * n * (n - 1) * (1 - nu)
        en_b = bn * RHO ** n * (n + 1) * (n + 2 - nu * (n - 2))
        en_c = cn * RHO ** (-n - 2) * n * (n + 1) * (1 - nu)
        en_d = dn * RHO ** (-n) * (n - 1) * (n - 2 - nu * (n + 2))
        en = en_a + en_b + en_c + en_d

        Mr_nondim = e0[:, None] + e1[:, None] * np.cos(theta) + (en * np.cos(n.T * theta)).sum(axis=0)

        # Sign convention
        return -w_nondim, -Mr_nondim, abcd

    def structures_one_case(self,
            F_heave, F_surge, sigma_max,
            h_s, T_s, D_s, D_f, D_f_in, num_sections, D_f_tu, D_d, L_dt, theta_dt, D_d_tu,
            t_s_r, I, A_c, A_lat_sub, t_bot, t_top, t_d, t_d_tu, h_d, A_dt,
            h_stiff_f, w_stiff_f, h_stiff_d, w_stiff_d,
            rho_w, g, E, nu, num_terms_plate, radial_mesh_plate, num_stiff_d
    ):
        # Placeholder for the function logic
        # You would implement the actual computation here
        depth = T_s
        P_hydrostatic = rho_w * g * depth

        # Float plate stress (using helper function Done)
        sigma_float_bot, sigma_float_top = self.float_plate_stress(
            D_f, D_f_in, F_heave, num_sections, t_bot, t_top, h_stiff_f, w_stiff_f, D_f_tu, nu
        )

        # Spar Buckling (using helper function Done)
        FOS_spar, FOS_spar_local = self.spar_combined_buckling(
            F_heave, E, I[1], h_s, D_s, A_c[1], t_s_r, P_hydrostatic, sigma_max, nu
        )

        # Damping plate stress (using helper function Done)
        radial_stress_damping_plate = self.damping_plate_structures(
            F_heave, D_d, D_s, P_hydrostatic, t_d, A_dt, theta_dt, L_dt, h_d, A_c, E, nu,
            h_stiff_d, w_stiff_d, D_d_tu, t_d_tu, num_terms_plate, radial_mesh_plate, num_stiff_d
        )

        # Factor of Safety calculations
        FOS1Y = sigma_max / sigma_float_bot
        FOS2Y = FOS_spar
        FOS3Y = sigma_max / radial_stress_damping_plate



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






