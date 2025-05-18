import openmdao.api as om
import numpy as np
class geometryComponent(om.ExplicitComponent):
    def setup(self):
        """
        geometry(D_s, D_f, D_f_in, D_f_b, ...
                                            T_f_1, T_f_2, h_f, h_s, h_fs_clear, D_f_tu, ...
                                            t_f_t, t_f_r, t_f_c, t_f_b, t_f_tu, t_s_r, t_d_tu, ...
                                            D_d, D_d_tu, theta_d_tu, T_s, h_d, t_d, ...
                                            h_stiff_f, w_stiff_f, num_sect_f, ...
                                            h_stiff_d, w_stiff_d, num_stiff_d, ...
                                            M, rho_m, rho_w, m_scale)
        
        """
        #Inputs
        self.add_input('D_s', 0)
        self.add_input('D_f', 0)
        self.add_input('D_f_in', 0)
        self.add_input('D_f_b', 0)
        self.add_input('T_f_1', 0)
        self.add_input('T_f_2', 0)
        self.add_input('h_f', 0)
        self.add_input('h_s', 0)
        self.add_input('h_fs_clear', 0)
        self.add_input('D_f_tu', 0)
        self.add_input('t_f_t', 0)
        self.add_input('t_f_r', 0)
        self.add_input('t_f_c', 0)
        self.add_input('t_f_b', 0)
        self.add_input('t_f_tu', 0)
        self.add_input('t_s_r', 0)
        self.add_input('t_d_tu', 0)
        self.add_input('D_d', 0)
        self.add_input('D_d_tu', 0)
        self.add_input('theta_d_tu', 0)
        self.add_input('T_s', 0)
        self.add_input('h_d', 0)
        self.add_input('t_d', 0)
        self.add_input('h_stiff_f', 0)
        self.add_input('w_stiff_f', 0)
        self.add_input('num_sect_f', 0)
        self.add_input('h_stiff_d', 0)
        self.add_input('w_stiff_d', 0)
        self.add_input('num_stiff_d', 0)
        self.add_input('M', 0)
        self.add_input('rho_m', 0)
        self.add_input('rho_w', 0)
        self.add_input('m_scale', 0)

        #output
        self.add_output('V_d', np.zeros((3,)))
        self.add_output('m_m', 0)
        self.add_output('m_f_tot', 0)
        self.add_output('m_s_tot', 0)
        self.add_output('A_c', np.zeros((3,)))
        self.add_output('A_lat_sub', np.zeros((3,)))
        self.add_output('I', np.zeros((3,)))
        self.add_output('T', np.zeros((3,)))
        self.add_output('V_f_pct', 0)
        self.add_output('V_s_pct', 0)
        self.add_output('GM', 0)
        self.add_output('A_dt', 0)
        self.add_output('L_dt', 0)
        self.add_output('mass', np.zeros((3,)))
        self.add_output('CB_f_from_waterline', 0)
        self.add_output('CG_f_from_waterline', 0)

        #self.declare_partials('*', '*', method='fd')

    # using wild cards to say that this component provides derivatives of all outputs with respect to all inputs.
    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        #Retrieve inputs
        """
                D_s, D_f, T_f, h_f, h_s, t_ft, t_fr, t_fc, t_fb, t_sr, t_dt,
                     D_d, D_dt, theta_dt, T_s, h_d, M, rho_m, rho_w, m_scale
        """
        D_s = inputs['D_s'][0]
        D_f = inputs['D_f'][0]
        D_f_in = inputs['D_f_in'][0]
        D_f_b = inputs['D_f_b'][0]
        T_f_1 = inputs['T_f_1'][0]
        T_f_2 = inputs['T_f_2'][0]
        h_f = inputs['h_f'][0]
        h_s = inputs['h_s'][0]
        h_fs_clear = inputs['h_fs_clear'][0]
        D_f_tu = inputs['D_f_tu'][0]
        t_f_t = inputs['t_f_t'][0]
        t_f_r = inputs['t_f_r'][0]
        t_f_c = inputs['t_f_c'][0]
        t_f_b = inputs['t_f_b'][0]
        t_f_tu = inputs['t_f_tu'][0]
        t_s_r = inputs['t_s_r'][0]
        t_d_tu = inputs['t_d_tu'][0]
        D_d = inputs['D_d'][0]
        D_d_tu = inputs['D_d_tu'][0]
        theta_d_tu = inputs['theta_d_tu'][0]
        T_s = inputs['T_s'][0]
        h_d = inputs['h_d'][0]
        t_d = inputs['t_d'][0]
        h_stiff_f = inputs['h_stiff_f'][0]
        w_stiff_f = inputs['w_stiff_f'][0]
        num_sect_f = inputs['num_sect_f'][0]
        h_stiff_d = inputs['h_stiff_d'][0]
        w_stiff_d = inputs['w_stiff_d'][0]
        num_stiff_d = inputs['num_stiff_d'][0]
        M = inputs['M'][0]
        rho_m = inputs['rho_m'][0]
        rho_w = inputs['rho_w'][0]
        m_scale = inputs['m_scale'][0]


        # Float
        num_gussets = 2 * num_sect_f;
        num_gussets_loaded_lateral = 2
        # convert index variable M to int instead of float
        M = int(M)

        # float cross-sectional and lateral area for structural purposes
        D_f_mean = (D_f + D_f_b) / 2
        A_f_cross_top = np.pi * (D_f + D_f_in) * t_f_r + num_gussets * t_f_c * (
                    D_f - D_f_in) / 2  # ring with diameter D_f, ring with diameter D_f_in, and gussets
        A_f_cross_bot = np.pi * D_f_in * t_f_r + num_gussets * t_f_c * (
                    D_f_mean - D_f_in) / 2  # ring with diameter D_s and bottom part of gussets
        A_f_l = num_gussets_loaded_lateral * t_f_c * T_f_2

        # float material volume and mass - see drawings on p168-169 of notebook 11/26/24
        V_top_plate = np.pi * ((D_f / 2) ** 2 - (D_f_in / 2) ** 2) * t_f_t
        V_bot_plate = np.pi * (D_f_b / 2) ** 2 * t_f_b
        slant_height = np.sqrt((D_f / 2 - D_f_b / 2) ** 2 + (T_f_2 - T_f_1) ** 2)
        V_bot_slant = (np.pi / 2) * (D_f + D_f_b) * slant_height * t_f_b  # lateral area of frustum
        V_outer_rim = np.pi * D_f * t_f_r * (h_f - (T_f_2 - T_f_1))
        V_inner_rim = np.pi * D_f_in * t_f_t * h_f
        A_gusset = (h_f - (T_f_2 - T_f_1)) * (D_f - D_f_in) / 2 + (T_f_2 - T_f_1) * (D_f_mean - D_f_in) / 2
        V_gussets = num_gussets * A_gusset * t_f_c

        # float support tubes for attaching PTO
        num_float_tubes = 6
        horiz_leg_tube = D_f / 2
        vert_leg_tube = (1 + D_f / (D_f - D_s)) * (
                    h_fs_clear + h_s - h_f + T_f_2 - T_f_1 - T_s)  # vertical clearance between float tubes and spar at rest
        L_ft = np.sqrt(horiz_leg_tube ** 2 + vert_leg_tube ** 2)  # length of float tube
        V_f_tubes = num_float_tubes * (D_f_tu ** 2 - (D_f_tu - 2 * t_f_tu) ** 2) * np.pi / 4 * L_ft

        # float stiffeners
        num_stiff_per_sect_f = 2  # top and bottom - neglecting the side stiffeners
        A_stiff_f = num_stiff_per_sect_f * num_sect_f * h_stiff_f * w_stiff_f
        len_stiff_f = (D_f - D_f_in) / 2 * 0.75  # goes ~3/4 of the way along the float
        V_stiff_f = len_stiff_f * A_stiff_f

        V_sf_m = (V_top_plate + V_bot_plate + V_outer_rim + V_inner_rim +
                  V_bot_slant + V_gussets + V_f_tubes + V_stiff_f)

        m_f_m = V_sf_m * rho_m[M] * m_scale  # mass of float material without ballast

        # Float hydrostatic calculations
        A_f = np.pi / 4 * (D_f ** 2 - D_f_in ** 2)
        V_f_cyl = A_f * T_f_1  # displaced volume of float: hollow cylinder portion
        V_f_fr = (np.pi / 12) * (T_f_2 - T_f_1) * (
                    D_f ** 2 + D_f_b ** 2 + D_f * D_f_b)  # displaced volume: non-hollow frustum portion
        V_f_fr_mid = (np.pi / 4) * D_f_in ** 2 * (T_f_2 - T_f_1)  # center cylinder volume to subtract from frustum
        V_f_fru_hol = V_f_fr - V_f_fr_mid  # hollow frustum portion volume
        V_f_d = V_f_cyl + V_f_fru_hol  # total displaced volume of float
        m_f_tot = V_f_d * rho_w  # total mass displaced by float (buoyancy)

        # Ballast (calculations)
        m_f_b = m_f_tot - m_f_m  # mass of ballast on float
        V_f_b = m_f_b / rho_w  # volume of ballast on float
        V_f_tot = V_f_d + A_f * (h_f - T_f_2)  # total volume available on float
        V_f_pct = V_f_b / V_f_tot  # percentage of available volume used by ballast

        I_f = np.pi / 64 * D_f ** 4  # area moment of inertia of float

        # Spar (vertical column and damping plate)
        V_vc_d = np.pi / 4 * D_s ** 2 * (T_s - h_d)  # vertical column volume displaced
        V_d_d = np.pi / 4 * D_d ** 2 * h_d  # damping plate volume displaced
        V_s_d = V_vc_d + V_d_d  # spar volume displaced (both column and damping plate)
        m_s_tot = rho_w * V_s_d  # total spar mass

        # vertical column material use
        D_vc_i = D_s - 2 * t_s_r  # Inner diameter of spar column
        A_vc_c = np.pi / 4 * (D_s ** 2 - D_vc_i ** 2)  # Cross-sectional area of spar column
        V_vc_m_body = A_vc_c * (h_s - h_d)  # Volume of vertical column body material
        A_vc_caps = np.pi / 4 * D_vc_i ** 2  # Cap area
        t_vc_caps = (0.5 + 2.5) * 0.0254  # Cap thickness (inches to meters)
        V_vc_caps = A_vc_caps * t_vc_caps  # Volume of caps
        num_stiff_vc = 12  # Number of stiffeners
        A_vc_stiff = 0.658 + 2 * 0.652 + 0.350  # Stiffener cross-sectional area (m²)
        t_vc_stiff = 0.0254  # Stiffener thickness (m)
        V_vc_m_stiff = num_stiff_vc * A_vc_stiff * t_vc_stiff  # Volume of stiffeners
        V_vc_m = V_vc_m_body + V_vc_m_stiff + V_vc_caps  # Total volume of column material


        # damping plate material use
        A_d = np.pi / 4 * D_d ** 2  # Damping plate area
        num_supports = 4  # Number of support tubes
        L_dt = (D_d - D_s) / (2 * np.cos(theta_d_tu))  # Length of each diagonal support tube
        D_dt_i = D_d_tu - 2 * t_d_tu  # Inner diameter of damping tube
        A_dt = np.pi / 4 * (D_d_tu ** 2 - D_dt_i ** 2)  # Cross-sectional area of support tube
        num_unique_stiffeners = len(h_stiff_d) // 2  # Unique stiffeners (symmetrical layout)
        num_stiff_repeats = num_stiff_d / num_unique_stiffeners  # Number of times each stiffener shape is used
        A_stiff_d = num_stiff_repeats * np.sum(h_stiff_d * w_stiff_d)  # Total stiffener area
        V_d_m = A_d * t_d + num_supports * A_dt * L_dt + A_stiff_d * (D_d - D_s) / 2  # Total material volume

        # total spar material use and mass
        m_vc_m = V_vc_m * rho_m[M] * m_scale  # Mass of spar vertical column material
        m_d_m = V_d_m * rho_m[M] * m_scale  # Mass of damping plate material
        m_s_m = m_vc_m + m_d_m  # Total mass of spar material

        # spar ballast
        m_s_b = m_s_tot - m_s_m  # mass of spar ballast
        V_s_b = m_s_b / rho_w  # volume of spar ballast
        V_pto = 0.75 * np.pi * 3 ** 2 * 12  # volume of PTO (assume constant)
        V_s_tot = np.pi / 4 * D_s ** 2 * h_s - V_pto  # total volume available on spar for ballast
        V_s_pct = V_s_b / V_s_tot  # percent of available volume used by ballast on spar

        I_vc = np.pi * (D_s ** 4 - D_vc_i ** 4) / 64  # area moment of inertia
        A_vc_l = 0.5 * np.pi * D_s * (T_s - h_d)  # lateral area

        # Reaction plate
        A_d_c = np.pi / 4 * D_d ** 2  # cross sectional area
        A_d_l = 0.5 * np.pi * D_d * h_d  # lateral area
        I_rp = np.pi * D_d ** 4 / 64  # area moment of inertia

        # Totals
        A_c = [A_f_cross_top, A_vc_c, A_d_c]  # cross sectional areas
        A_lat_sub = [A_f_l, A_vc_l, A_d_l]  # lateral submerged areas
        I = [I_f, I_vc, I_rp]  # moments of inertia
        T = [T_f_2, T_s, h_d]  # drafts: used to calculate F_surge
        m_m = m_f_m + m_s_m  # total mass of material

        V_d = [V_f_d, V_vc_d, V_d_d]  # volume displaced
        mass = [m_f_m, m_vc_m, m_d_m]  # material mass of each structure

        # Metacentric Height Calculation
        # see dev / cob_com_frustum.mlx for derivation of float COB and COM

        # Center of Buoyancy (CB) for float
        D_term_1 = -D_f_b ** 2 - 2 * D_f_b * D_f + 3 * D_f ** 2
        D_term_2 = 2 * D_f ** 2 - 2 * D_f_b ** 2
        D_term_3 = -6 * D_f_in ** 2 + 3 * D_f_b ** 2 + 2 * D_f_b * D_f + D_f ** 2
        CB_f_integral = D_term_1 * T_f_1 ** 2 + D_term_2 * T_f_1 * T_f_2 + D_term_3 * T_f_2 ** 2
        CB_f_from_waterline = (np.pi / 48) * CB_f_integral / V_f_d

        # Center of Gravity (CG) for float
        T_term_1 = T_f_1 ** 2 + 2 * T_f_1 * T_f_2 - 3 * T_f_2 ** 2
        T_term_2 = 2 * T_f_1 ** 2 - 2 * T_f_2 ** 2
        T_term_3 = -3 * T_f_1 ** 2 - 2 * T_f_1 * T_f_2 + 5 * T_f_2 ** 2 - 12 * T_f_2 * h_f + 6 * h_f ** 2
        T_term_4 = 12 * T_f_2 * h_f - 6 * h_f ** 2
        T_term_5 = 4 * (T_f_1 - T_f_2)
        T_term_6 = 4 * (-2 * T_f_1 + 2 * T_f_2 - 3 * h_f)
        T_term_7 = 12 * h_f

        CG_f_num = T_term_1 * D_f_b ** 2 + T_term_2 * D_f_b * D_f + T_term_3 * D_f ** 2 + T_term_4 * D_f_in ** 2
        CG_f_den = T_term_5 * D_f_b ** 2 + T_term_5 * D_f_b * D_f + T_term_6 * D_f ** 2 + T_term_7 * D_f_in ** 2
        CG_f_from_waterline = CG_f_num / CG_f_den

        # centers of buoyancy, measured from keel (bottom of damping plate)
        CB_f = T_s - CB_f_from_waterline
        CB_vc = h_d + (T_s - h_d) / 2
        CB_d = h_d / 2
        CBs = np.array([CB_f, CB_vc, CB_d])

        # centers of gravity, measured from keel - assume even mass distribution
        CG_f = T_s - CG_f_from_waterline
        CG_vc = h_d + (h_s - h_d) / 2
        CG_d = h_d / 2
        CGs = np.array([CG_f, CG_vc, CG_d])

        KB = np.dot(CBs, V_d) / np.sum(V_d)  # center of buoyancy above the keel
        KG = np.dot(CGs, mass) / np.sum(mass)  # center of gravity above the keel
        BM = I_f / np.sum(V_d)  # moment due to buoyant rotational stiffness
        GM = KB + BM - KG  # metacentric height

        outputs['V_d'] = V_d
        outputs['m_m'] = m_m
        outputs['m_f_tot'] = m_f_tot
        outputs['m_s_tot'] = m_s_tot
        outputs['A_c'] = A_c
        outputs['A_lat_sub'] = A_lat_sub
        outputs['I'] = I
        outputs['T'] = T
        outputs['V_f_pct'] = V_f_pct
        outputs['V_s_pct'] = V_s_pct
        outputs['GM'] = GM
        outputs['A_dt'] = A_dt
        outputs['L_dt'] = L_dt
        outputs['mass'] = mass
        outputs['CB_f_from_waterline'] = CB_f_from_waterline
        outputs['CG_f_from_waterline'] = CG_f_from_waterline
