import openmdao.api as om
import numpy as np
class surgeComponent(om.ExplicitComponent):

    def setup(self):
        self.add_input('F_max', 0, desc="maximum force (N)")
        self.add_input('D_f', 0, desc="outer diameter of float (m)")
        self.add_input('h_f', 0, desc="height of straight section of float, before the frustum (m)")
        self.add_input('Hs_struct', val=np.zeros(1, ), desc="100 year wave height (m)")
        self.add_input('T_struct', val=np.zeros(1, ), desc="100 year wave period (s)")
        self.add_input('rho_w', val=0.0, desc="water density (kg/m3)")
        self.add_input('g', val=0.0, desc="acceleration of gravity (m/s2)")
        self.add_input('JPD', val=np.zeros((14, 15)), desc="joint probability distribution of wave (%)")
        self.add_input('Hs', val=np.zeros(14, ), desc="wave height (m)")
        self.add_input('T', val=np.zeros(15, ), desc="wave period (s)")
        self.add_input('power_max', val=0, desc="maximum power (W)")
        self.add_input('eff_pto', val=0, desc="PTO efficiency (-)")
        self.add_input('B_p', 0)
        self.add_input('w_n', 0)
        self.add_input('T_f', 0)
        self.add_input('T_s', 0)
        self.add_input('h_s', 0)


        self.add_input("m_float", val=0.0)
        self.add_input("V_d", shape=(3,))
        self.add_input("draft", shape=(3,))

        self.add_output('F_surge_max', shape=(3,))
        self.add_output('h_s_extra')
    def setup_partials(self):
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        f_max = inputs['F_max'] * 1e6
        D_f = inputs['D_f']
        h_f = inputs['h_f']
        Hs_struct = inputs['Hs_struct']
        # Hs = inputs['Hs_struct'][0]
        # Tp = inputs['T_struct'][0]
        T_struct = inputs['T_struct']
        rho_w = inputs['rho_w']
        g = inputs['g']
        JPD = inputs['JPD']
        Hs = inputs['Hs']
        T = inputs['T']
        power_max = inputs['power_max']
        eff_pto = inputs['eff_pto']
        B_p = inputs['B_p'] * 1e6
        w_n = inputs['w_n']
        T_f = inputs['T_f']
        T_s = inputs['T_s']
        h_s = inputs['h_s']

        m_float = inputs['m_float']
        V_d = inputs['V_d']
        draft = inputs['draft']


        P_matrix, h_s_extra, P_unsat, _, _, _ = self.get_power_force(D_f, T_f, rho_w, g, B_p, w_n, f_max, h_s, T_s, h_f,
                                                                     T.copy(), Hs.copy(), m_float.copy(), V_d.copy(),
                                                                     draft.copy())

        # Account for powertrain electrical losses
        P_matrix *= eff_pto

        # Saturate maximum power
        P_matrix = np.minimum(P_matrix, power_max)
        # Weight power across all sea states
        P_weighted = P_matrix * JPD / 100
        P_elec = np.sum(P_weighted)

        assert np.isreal(P_elec)

        # Use max sea states for structural forces and max amplitude
        _, _, _, F_heave_max, F_surge_max, F_ptrain_max = self.get_power_force(
            D_f, T_f, rho_w, g, B_p, w_n, f_max, h_s, T_s, h_f, T_struct, Hs_struct, m_float, V_d, draft)

        average, P_var = self.weighted_avg_and_std(P_matrix, JPD) / P_elec
        P_var *= 100  # Convert to percentage

        outputs['F_surge_max'] = F_surge_max
        outputs['h_s_extra'] = h_s_extra
        outputs['P_var'] = P_var

    def get_power_force(self, D_f, T_f, rho_w, g, B_p, w_n, F_max, h_s, T_s, h_f, T, Hs, m_float, V_d, draft):
        # Get unsaturated response
        w, A, B_h, K_h, Fd, k_wvn = self.dynamics_simple(Hs, T, D_f, T_f, rho_w, g)

        m = m_float + A

        b = B_h + B_p
        k = w_n ** 2 * m
        K_p = k - K_h
        X_unsat = self.get_response(w, m, b, k, Fd)
        # Confirm unsaturated response doesn't exceed maximum capture width
        P_unsat = 0.5 * B_p * w ** 2 * X_unsat ** 2

        F_ptrain_over_x = np.sqrt((B_p * w) ** 2 + (K_p) ** 2)
        F_ptrain_unsat = F_ptrain_over_x * X_unsat

        # Get saturated response
        r = np.minimum(F_max / F_ptrain_unsat, 1)
        alpha = (2 / np.pi) * (1 / r * np.arcsin(r) + np.sqrt(1 - r ** 2))
        f_sat = alpha * r
        # add copy() to each np array as parameters
        mult = self.get_multiplier(np.copy(f_sat), np.copy(m), np.copy(b), np.copy(k), np.copy(w), b / B_p, k / K_p)
        b_sat = B_h + mult * B_p
        k_sat = K_h + mult * K_p
        X_sat = self.get_response(w, m, b_sat, k_sat, Fd)

        # Calculate power
        P_matrix = .5 * (mult * B_p) * w ** 2 * X_sat ** 2

        X_max = np.max(X_sat)
        h_s_extra = (h_s - T_s - (h_f - T_f) - X_max) / h_s

        # Calculate forces
        F_ptrain = mult * F_ptrain_over_x * X_sat
        F_ptrain_max = np.max(F_ptrain)
        F_err_1 = np.abs(F_ptrain / (F_max * alpha) - 1)
        F_err_2 = np.abs(F_ptrain / (f_sat * F_ptrain_unsat) - 1)

        # 0.1 percent error
        if np.any(f_sat < 1):
            assert np.all(F_err_1[f_sat < 1] < 1e-3)

        assert np.all(F_err_2 < 1e-3)

        F_heave_fund = np.sqrt((mult * B_p * w) ** 2 + (mult * K_p - m_float * w ** 2) ** 2) * X_sat
        F_heave = np.minimum(F_heave_fund, F_max + m_float * w ** 2 * X_sat)
        F_surge = np.max(Hs) * rho_w * g * V_d * (1 - np.exp(-np.max(k_wvn) * draft))
        return P_matrix, h_s_extra, P_unsat, F_heave, F_surge, F_ptrain_max

    def dynamics_simple(self, Hs, T, D_f, T_f, rho_w, g):
        w = 2 * np.pi / T  # angular frequency
        k = w ** 2 / g  # wave number (dispersion relation for deep water)
        r = D_f / 2  # radius
        draft = T_f  # draft below waterline
        A_w = np.pi * r ** 2  # waterplane area
        A_over_rho, B_over_rho_w, gamma_over_rho_g = self.get_hydro_coeffs(r, k, draft)
        A = rho_w * A_over_rho  # added mass
        B = rho_w * w * B_over_rho_w  # radiation damping
        gamma = rho_w * g * gamma_over_rho_g  # froude krylov coefficient
        K = rho_w * g * A_w  # hydrostatic stiffness
        Fd = gamma * Hs  # excitation force of wave

        return w, A, B, K, Fd, k

    def get_response(self, w, m, b, k, Fd):
        imag_term = b * w
        real_term = k - m * w ** 2
        X_over_F_mag = 1 / np.sqrt(real_term ** 2 + imag_term ** 2)
        X = X_over_F_mag * Fd
        return X
    def get_hydro_coeffs(self, r, k, draft):
        # add the follow line to convert k as float:
        # k = k.astype(float)
        # Froude Krylov force coefficient (diffraction is neglected)
        tuning_factor = 4.5  # tune to more closely match WAMIT results which include diffraction

        r_k_term = r ** 2 - (k ** 2 * r ** 4) / 8 + (k ** 4 * r ** 6) / 192 - (k ** 6 * r ** 8) / 9216 \
                   + (k ** 8 * r ** 10) / 737280 - (k ** 10 * r ** 12) / 88473600

        r_k_term = np.abs(r_k_term)  # get rid of any negatives that result at high frequencies

        gamma_over_rho_g = np.pi * np.exp(-k * draft * tuning_factor) * r_k_term

        # Added mass
        A_over_rho = 0.5 * 4 / 3 * np.pi * r ** 3 * 0.63

        # Radiation damping
        B_over_rho_w = k / 2 * gamma_over_rho_g ** 2  # Haskind relationship, using the deep water group velocity

        return A_over_rho, B_over_rho_w, gamma_over_rho_g

    def get_multiplier(self, f_sat, m, b, k, w, r_b, r_k):
        # m, k, and r_k are scalars.
        # All other inputs are 2D arrays, the dimension of the sea state matrix.

        # speedup: only do math for saturated sea states, since unsat will = 1
        idx_no_sat = f_sat == 1
        f_sat[idx_no_sat] = np.nan
        b[idx_no_sat] = np.nan
        w[idx_no_sat] = np.nan
        r_b[idx_no_sat] = np.nan
        a_quad, b_quad, c_quad = self.get_abc_symbolic(f_sat, m, b, k, w, r_b, r_k)
        # solve the quadratic formula
        if idx_no_sat.shape == (1,):
            idx_no_sat = idx_no_sat.reshape((1, 1))
        if a_quad.shape == (1,):
            a_quad = a_quad.reshape((1, 1))
        if b_quad.shape == (1,):
            b_quad = b_quad.reshape((1, 1))
        if c_quad.shape == (1,):
            c_quad = c_quad.reshape((1, 1))
        determinant = np.sqrt(b_quad ** 2 - 4 * a_quad * c_quad)

        num = -b_quad + determinant
        # creating a second dimension to hold the second root value
        num = np.stack((num, -b_quad - determinant), axis=-1)
        den = 2 * a_quad

        den = den[:, :, None]

        roots = num / den

        # choose which of the two roots to use
        mult = self.pick_which_root(roots, idx_no_sat, a_quad, b_quad, c_quad)
        assert np.all(~np.isnan(mult))

        return mult

    def get_abc_symbolic(self, f, m, b, k, w, r_b, r_k):
        """
        % GET_ABC_SYMBOLIC
        % [A_Q, B_Q, C_Q] = GET_ABC_SYMBOLIC(F, M, B, K, W, R_B, R_K)

        12 - Jan - 2023 02: 31:32
        """
        t2 = b ** 2
        t3 = k ** 2
        t4 = m ** 2
        t5 = r_b ** 2
        t6 = r_k ** 2
        t7 = w ** 2
        t8 = t7 ** 2
        t9 = t3 * t5
        t12 = t2 * t6 * t7
        t13 = k * m * r_k * t5 * t7 * 2.0
        t10 = r_k * t9 * 2.0
        t11 = -t9
        t14 = r_b * t12 * 2.0
        t15 = -t12

        a_q = t11 + t15 + 1.0 / f ** 2 * t5 * t6 * (t3 + t2 * t7 + t4 * t8 - k * m * t7 * 2.0)
        b_q = t9 * 2.0 - t10 + t12 * 2.0 + t13 - t14
        c_q = t10 + t11 - t13 + t14 + t15 + t6 * t11 + t5 * t15 - t4 * t5 * t6 * t8 + k * m * t5 * t6 * t7 * 2.0

        return a_q, b_q, c_q
    def pick_which_root(self, roots, idx_no_sat, a_quad, b_quad, c_quad):
        which_soln = (roots == np.real(roots)) & (roots > 0) & (roots <= 1)
        both_ok = np.sum(which_soln, axis=2) == 2
        # Check for the third dimension and act accordingly

        # temporarily mark the non - saturated solutions
        # as having one solution, to ensure the
        # logic below works correctly

        # Jordan's change

        # which_soln[idx_no_sat] = True
        which_soln[idx_no_sat, 0] = True

        if np.any(both_ok):  # two solutions
            mult = self.handle_two_solns(both_ok, which_soln, roots, idx_no_sat, a_quad, b_quad, c_quad)
        else:
            num_solns = np.sum(which_soln, axis=-1)
            if not np.all(num_solns == 1):
                which_soln[num_solns == 0] = (roots[num_solns == 0] > 0) & (roots[num_solns == 0] <= 1.001)
                num_solns[num_solns == 0] = np.sum(which_soln[num_solns == 0], axis=2)
                if not np.all(num_solns == 1):
                    print('Some sea states have no valid quadratic solution, so their energy is zeroed.')
            mult = self.get_relevant_soln(which_soln, roots, idx_no_sat)

        return mult

    def handle_two_solns(self, both_ok, which_soln, roots, idx_no_sat, a, b, c):
        row, col = np.where(both_ok)
        which_soln[row, col, 1] = False

        mult_1 = self.get_relevant_soln(which_soln, roots, idx_no_sat)

        # In the provided MATLAB function, logic to handle the case of two roots has been commented out.
        # The Python function currently just uses the first solution when two are available.
        # If you need to include the logic for handling two roots, please provide the full active MATLAB code.

        return mult_1

    def get_relevant_soln(self, which_soln, roots, idx_no_sat):
        mult = np.zeros(idx_no_sat.shape)

        idx_3d_first_sol = np.copy(which_soln)
        idx_3d_first_sol[:, :, 1] = False
        idx_3d_second_sol = np.copy(which_soln)
        idx_3d_second_sol[:, :, 0] = False
        idx_2d_first_sol = which_soln[:, :, 0]
        idx_2d_second_sol = which_soln[:, :, 1]

        mult[idx_2d_first_sol] = roots[idx_3d_first_sol]
        mult[idx_2d_second_sol] = roots[idx_3d_second_sol]
        mult[idx_no_sat] = 1

        return mult

    def weighted_avg_and_std(self, values, weights):
        """
        Return the weighted average and standard deviation.

        They weights are in effect first normalized so that they
        sum to 1 (and so they must not all be 0).

        values, weights -- NumPy ndarrays with the same shape.
        """
        average = np.average(values, weights=weights)
        # Fast and numerically precise:
        variance = np.average((values - average) ** 2, weights=weights)
        return (average, np.sqrt(variance))