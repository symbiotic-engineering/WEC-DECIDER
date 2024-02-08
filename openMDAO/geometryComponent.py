import openmdao.api as om
import numpy as np
class geometryComponent(om.ExplicitComponent):
    def setup(self):
        """
        D_s, D_f, T_f, h_f, h_s, t_ft, t_fr, t_fc, t_fb, t_sr, t_dt,
             D_d, D_dt, theta_dt, T_s, h_d, M, rho_m, rho_w, m_scale


        return V_d, m_m, m_f_tot, A_c, A_lat_sub, r_over_t, I, T, V_f_pct, V_s_pct, GM, mass

        """
        #Inputs
        self.add_input('D_s', 0)
        self.add_input('D_f', 0)
        self.add_input('T_f', 0)
        self.add_input('h_f', 0)
        self.add_input('h_s', 0)
        self.add_input('t_ft', 0)
        self.add_input('t_fr', 0)
        self.add_input('t_fc', 0)
        self.add_input('t_fb', 0)
        self.add_input('t_sr', 0)
        self.add_input('t_dt', 0)
        self.add_input('D_d', 0)
        self.add_input('D_dt', 0)
        self.add_input('theta_dt')
        self.add_input('T_s', 0)
        self.add_input('h_d', 0)
        self.add_input('M', 0)
        self.add_input('rho_m', np.zeros((3,)))
        self.add_input('rho_w', 0)
        self.add_input('m_scale', 0)



        #output
        self.add_output('V_d', np.zeros((3,)))
        self.add_output('m_m', 0)
        self.add_output('m_f_tot', 0)
        self.add_output('A_c', np.zeros((3,)))
        self.add_output('A_lat_sub', np.zeros((3,)))
        self.add_output('r_over_t', np.zeros((3,)))
        self.add_output('I', np.zeros((3,)))
        self.add_output('T', np.zeros((3,)))
        self.add_output('V_f_pct', 0)
        self.add_output('V_s_pct', 0)
        self.add_output('GM', 0)
        self.add_output('mass', np.zeros((3,)))

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
        T_f = inputs['T_f'][0]
        h_f = inputs['h_f'][0]
        h_s = inputs['h_s'][0]
        t_ft = inputs['t_ft'][0]
        t_fr = inputs['t_fr'][0]
        t_fc = inputs['t_fc'][0]
        t_fb = inputs['t_fb'][0]
        t_sr = inputs['t_sr'][0]
        t_dt = inputs['t_dt'][0]
        D_d = inputs['D_d'][0]
        D_dt = inputs['D_dt'][0]
        theta_dt = inputs['theta_dt'][0]
        T_s = inputs['T_s'][0]
        h_d = inputs['h_d'][0]
        M = inputs['M'][0]
        rho_m = inputs['rho_m']
        rho_w = inputs['rho_w'][0]
        m_scale = inputs['m_scale'][0]

        num_gussets = 24
        num_gussets_loaded_lateral = 2
        # convert index variable M to int instead of float
        M = int(M)

        # Float cross-sectional and lateral area
        A_f_c = np.pi * (D_f + D_s) * t_fr + num_gussets * t_fc * (D_f - D_s) / 2
        A_f_l = num_gussets_loaded_lateral * t_fc * T_f

        # Float material volume and mass
        V_top_plate = np.pi * (D_f / 2) ** 2 * t_ft
        V_bot_plate = np.pi * (D_f / 2) ** 2 * t_fb
        V_rims_gussets = A_f_c * h_f
        V_sf_m = V_top_plate + V_bot_plate + V_rims_gussets

        m_f_m = V_sf_m * rho_m[M] * m_scale
        # print("ad",rho_m)
        # Float hydrostatic calculations
        A_f = np.pi / 4 * (D_f ** 2 - D_s ** 2)
        V_f_d = A_f * T_f
        m_f_tot = V_f_d * rho_w

        # Ballast
        m_f_b = m_f_tot - m_f_m
        V_f_b = m_f_b / rho_w
        V_f_tot = A_f * h_f
        V_f_pct = V_f_b / V_f_tot

        I_f = np.pi / 64 * D_f ** 4

        # Spar (vertical column and damping plate)
        V_vc_d = np.pi / 4 * D_s ** 2 * T_s
        V_d_d = np.pi / 4 * D_d ** 2 * h_d
        V_s_d = V_vc_d + V_d_d
        m_s_tot = rho_w * V_s_d

        # Vertical column material use
        D_vc_i = D_s - 2 * t_sr
        A_vc_c = np.pi / 4 * (D_s ** 2 - D_vc_i ** 2)
        V_vc_m = A_vc_c * h_s

        # Damping plate material use
        A_d = np.pi / 4 * D_d ** 2
        num_supports = 4
        L_dt = D_d / (2 * np.cos(theta_dt))
        D_dt_i = D_dt - 2 * t_dt
        A_dt = np.pi / 4 * (D_dt ** 2 - D_dt_i ** 2)
        V_d_m = A_d * h_d + num_supports * A_dt * L_dt

        # Total spar material use and mass
        m_vc_m = V_vc_m * rho_m[M] * m_scale
        m_d_m = V_d_m * rho_m[M] * m_scale
        m_s_m = m_vc_m + m_d_m

        # Spar ballast
        m_s_b = m_s_tot - m_s_m
        V_s_b = m_s_b / rho_w
        V_s_tot = np.pi / 4 * D_s ** 2 * h_s
        V_s_pct = V_s_b / V_s_tot

        I_vc = np.pi * (D_s ** 4 - D_vc_i ** 4) / 64
        A_vc_l = 1 / 2 * np.pi * D_s * T_s

        # Reaction plate
        A_d_c = np.pi / 4 * (D_d ** 2 - D_s ** 2)
        A_d_l = 1 / 2 * np.pi * D_d * h_d
        I_rp = np.pi * D_d ** 4 / 64

        # Totals
        A_c = np.array([A_f_c, A_vc_c, A_d_c])
        A_lat_sub = np.array([A_f_l, A_vc_l, A_d_l])
        r_over_t = np.array([0, D_s / (2 * t_sr), 0])
        I = np.array([I_f, I_vc, I_rp])
        T = np.array([T_f, T_s, h_d])
        m_m = m_f_m + m_s_m

        V_d = np.array([V_f_d, V_vc_d, V_d_d])
        mass = np.array([m_f_m, m_vc_m, m_d_m])

        # Metacentric Height Calculation
        CB_f = h_d + T_s - T_f / 2
        CB_vc = h_d + T_s / 2
        CB_d = h_d / 2
        CBs = np.array([CB_f, CB_vc, CB_d])
        # centers of gravity, measured from keel (assume even mass distribution)
        CG_f = h_d + T_s - T_f + h_f / 2
        CG_vc = h_d + h_s / 2
        CG_d = h_d / 2
        CGs = np.array([CG_f, CG_vc, CG_d])

        # center of buoyancy above the keel
        KB = np.dot(CBs, V_d) / np.sum(V_d)

        # center of gravity above the keel
        KG = np.dot(CGs, mass) / np.sum(mass)

        BM = I_f / sum(V_d)  # moment due to buoyant rotational stiffness
        GM = KB + BM - KG

        #return V_d, m_m, m_f_tot, A_c, A_lat_sub, r_over_t, I, T, V_f_pct, V_s_pct, GM, mass
        outputs['V_d'] = V_d
        outputs['m_m'] = m_m
        outputs['m_f_tot'] = m_f_tot
        outputs['A_c'] = A_c
        outputs['A_lat_sub'] = A_lat_sub
        outputs['r_over_t'] = r_over_t
        outputs['I'] = I
        outputs['T'] = T
        outputs['V_f_pct'] = V_f_pct
        outputs['V_s_pct'] = V_s_pct
        outputs['GM'] = GM
        outputs['mass'] = mass

"""
prob = om.Problem()

promotesInputs = ["D_s", "D_f", "T_f", "h_f", "h_s", "t_ft", "t_fr", "t_fc",
    "t_fb", "t_sr", "t_dt", "D_d", "D_dt", "theta_dt", "T_s",
    "h_d", "M", "rho_m", "rho_w", "m_scale"]

prob.model.add_subsystem('test', geometryComponent(), promotes_inputs= promotesInputs )

prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'SLSQP'

prob.model.add_design_var('D_s')
prob.model.add_objective('test.mass', scaler=-1)
prob.setup()

prob.set_val('D_s', 6.0)
prob.set_val('D_f', 20.0)
prob.set_val('T_f', 2.0)
prob.set_val('h_f', 4.0)
prob.set_val('t_ft', 0.0127)
prob.set_val('t_fr', 0.011176)
prob.set_val('t_fc', 0.011176)
prob.set_val('t_fb', 0.014224)
prob.set_val('t_sr', 0.0254)
prob.set_val('t_dt', 0.0254)
prob.set_val('D_d', 30.0)
prob.set_val('D_dt', 1.2191999999999998)
prob.set_val('theta_dt', 0.8621700546672264)
prob.set_val('T_s', 35.0)
prob.set_val('h_d', 0.0254)
prob.set_val('M', 0)
prob.set_val('rho_m', np.array([8000, 2400, 8000]))
prob.set_val('rho_w', 1000)
prob.set_val('m_scale', 1.25)


print(prob.get_val('test.mass'))
print(prob.get_val('D_s'))
prob.run_model()
prob.model.list_inputs(val=True)
# output structure
# 3.088498840031996 7.1377643021609884 735.3862533286745 [[63.7930595]]
prob.model.list_outputs(val = True)
"""