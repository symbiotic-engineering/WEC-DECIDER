import openmdao.api as om
import autograd.numpy as np
import wecopttool as wot
import capytaine as cpy
import math
class heavenDynamicsComponent(om.ExplicitComponent):

    def setup(self):
        self.add_input('g', val=0.0, desc="acceleration of gravity (m/s2)")
        self.add_input('rho_w', val=0.0, desc="water density (kg/m3)")
        self.add_input('mass', 208000, desc="mass of RM3 (kg)")  # missing
        self.add_input('F_max', 0, desc="maximum force (N)")
        self.add_input('x_max', 0.04, desc="maximum position (m)")  # missing
        self.add_input('Vs_max', 1.5e5, desc="maximum voltage (V)")  # missing
        self.add_input('Hs_struct', val=np.zeros(1, ), desc="100 year wave height (m)")
        self.add_input('T_struct', val=np.zeros(1, ), desc="100 year wave period (s)")
        self.add_input('RM3',desc="RM3 generate from hydro")


        self.add_output("P_elec")
        self.add_output("F_heave_max")
        self.add_output('P_matrix', shape=(14, 15))
    def setup_partials(self):
        self.declare_partials('*', '*', method='fd')


    def compute(self, inputs, outputs):
        g = inputs['g']
        rho_w = inputs['rho_w']
        mass = inputs['mass']
        f_max = inputs['F_max'] * 1e6
        x_max = inputs['x_max']
        Vs_max = inputs['Vs_max']
        Hs_struct = inputs['Hs_struct']
        T_struct = inputs['T_struct']
        RM3 = inputs['RM3'][0]

        #RM3 = self.make_RM3(h_f[0], h_f_2[0], D_s[0], D_f[0], T_f[0], int(mesh_density))
        # RM3 another model hydro
        P_elec, f_heave = self.inner_function(g[0], rho_w[0], mass[0], f_max[0], x_max[0], Vs_max[0], RM3, Hs_struct[0],
                                              T_struct[0], waves_are_irreg=False)
        #missing P_matrix
        outputs['P_elec'] = P_elec
        outputs['F_heave_max'] = f_heave
        outputs['P_matrix'] = np.zeros((14,15)) #TO-DO

    def body_from_profile(self, x, y, z, nphi):
        xyz = np.array([np.array([x / math.sqrt(2), y / math.sqrt(2), z]) for x, y, z in
                        zip(x, y, z)])  # /sqrt(2) to account for the scaling
        body = cpy.FloatingBody(cpy.AxialSymmetricMesh.from_profile(xyz, nphi=nphi))
        return body


    def make_RM3(self, h_f, h_f_2, D_s, D_f, T_f, mesh_density):
        cpy.set_logging('ERROR')  # to get rid off the warnings
        freeboard = h_f_2 - T_f
        # normal vectors have to be facing outwards
        z1 = np.linspace(-h_f_2 + freeboard, -h_f + freeboard, mesh_density)
        x1 = np.linspace(D_s / 2, D_f / 2, mesh_density)
        y1 = np.linspace(D_s / 2, D_f / 2, mesh_density)
        bottom_frustum = self.body_from_profile(x1, y1, z1, mesh_density ** 2)
        z3 = np.linspace(-h_f + freeboard, freeboard, mesh_density)
        x3 = np.full_like(z3, D_f / 2)
        y3 = np.full_like(z3, D_f / 2)
        outer_surface = self.body_from_profile(x3, y3, z3, mesh_density ** 2)
        z4 = np.linspace(freeboard, +freeboard, mesh_density)
        x4 = np.linspace(D_f / 2, D_s / 2, mesh_density)
        y4 = np.linspace(D_f / 2, D_s / 2, mesh_density)
        top_surface = self.body_from_profile(x4, y4, z4, mesh_density ** 2)
        z2 = np.linspace(freeboard, -h_f_2 + freeboard, mesh_density)
        x2 = np.full_like(z2, D_s / 2)
        y2 = np.full_like(z2, D_s / 2)
        inner_surface = self.body_from_profile(x2, y2, z2, mesh_density ** 2)
        RM3 = bottom_frustum.join_bodies(outer_surface, top_surface, inner_surface).keep_immersed_part()
        RM3.center_of_mass = [0, 0, -(0.5 * h_f * h_f + (h_f + (h_f_2 - h_f) / 3) * (h_f_2 - h_f) * 0.5) / (
                    h_f + (h_f_2 - h_f) * 0.5) - T_f]
        RM3.rotation_center = RM3.center_of_mass
        return RM3
    def inner_function(self, g, rho, mass, f_max, x_max, Vs_max, fb, Hs, Tp, waves_are_irreg=False):
        # g = 9.8
        # rho = 1000 #rho_w
        fb.add_translation_dof(name="Heave")
        ndof = fb.nb_dofs
        fb.mass = np.atleast_2d(mass)
        f1 = 0.05  # Hz
        nfreq = 10
        freq = wot.frequency(f1, nfreq, False)  # False -> no zero frequency
        bem_data = wot.run_bem(fb, freq, rho=rho, g=g)
        name = ["PTO_Heave", ]
        kinematics = np.eye(ndof)
        controller = None
        loss = None
        pto_impedance = None
        pto = wot.pto.PTO(ndof, kinematics, controller, pto_impedance, loss, name)

        def force_on_wec_with_bumpstop(wec, x_wec, x_opt, waves, nsubsteps=1):
            pos = pto.position(wec, x_wec, x_opt, waves, nsubsteps)
            vel = pto.velocity(wec, x_wec, x_opt, waves, nsubsteps)
            b = 14  #
            k = 10e6  # N/m
            bumpstop = -k * (np.abs(pos) + x_max) * np.sign(pos) - b * vel
            condition = [(pos > x_max) | (pos < -x_max)]
            bumpstop = np.where(condition, bumpstop, 0.0)
            f_tot = pto.force_on_wec(wec, x_wec, x_opt, waves, nsubsteps) + bumpstop
            return f_tot

        f_add = {
            'PTO+bumpstop': force_on_wec_with_bumpstop
        }
        # contraints
        nsubsteps = 4

        def const_f_pto(wec, x_wec, x_opt, waves):  # Format for scipy.optimize.minimize
            f = pto.force_on_wec(wec, x_wec, x_opt, waves, nsubsteps)
            return f_max - np.abs(f.flatten())

        def const_Vs(wec, x_wec, x_opt, waves):
            vel = pto.velocity(wec, x_wec, x_opt, waves, nsubsteps)
            xdot = vel
            V = -1 * xdot
            f = pto.force_on_wec(wec, x_wec, x_opt, waves, nsubsteps)
            I = f
            L = 1
            p = 1
            G = 1
            omega = G * xdot
            #print("test")
            #print(V,L,p,omega,I)
            Vs = np.sqrt(V ** 2 + (L * p * omega * I) ** 2)
            return Vs_max - Vs.flatten()

        ineq_cons1 = {'type': 'ineq',
                      'fun': const_f_pto,
                      }
        ineq_cons2 = {'type': 'ineq',
                      'fun': const_Vs,
                      }
        constraints = [
            ineq_cons1,
            ineq_cons2
        ]
        wec = wot.WEC.from_bem(
            bem_data,
            constraints=constraints,
            f_add=f_add,
        )
        # regular waves
        wavefreq = 0.3
        amplitude = 1
        phase = 0
        wavedir = 0
        waves_regular = wot.waves.regular_wave(f1, nfreq, wavefreq, amplitude, phase, wavedir)

        # irregular waves
        # number of realizations to reach 20 minutes of total simulation time
        minutes_needed = 20
        nrealizations = minutes_needed * 60 * f1
        print(f'Number of realizations for a 20 minute total simulation time: {nrealizations}')
        nrealizations = 2  # overwrite nrealizations to reduce run-time
        fp = 1 / Tp
        spectrum = lambda f: wot.waves.pierson_moskowitz_spectrum(f, fp, Hs)
        efth = wot.waves.omnidirectional_spectrum(f1, nfreq, spectrum, "Pierson-Moskowitz")
        waves_irregular = wot.waves.long_crested_wave(efth, nrealizations)

        obj_fun = pto.average_power
        nstate_opt = 2 * nfreq

        options = {'maxiter': 1000}
        scale_x_wec = 1e4
        scale_x_opt = 1e-3
        scale_obj = 1e-3

        if waves_are_irreg:
            waves = waves_irregular
        else:
            waves = waves_regular

        results = wec.solve(
            waves,
            obj_fun,
            nstate_opt,
            optim_options=options,
            x_wec_0=np.ones(nfreq * 2) * 1e-4,
            x_opt_0=np.ones(nfreq * 2) * 1e0,
            scale_x_wec=scale_x_wec,
            scale_x_opt=scale_x_opt,
            scale_obj=scale_obj,
        )
        x_wec, x_opt = wot.decompose_state(results[0].x, ndof=ndof, nfreq=nfreq)
        inertia = wec.inertia(wec, x_wec, x_opt, waves)
        ptoPlusBumpstop = force_on_wec_with_bumpstop(wec, x_wec, x_opt, waves)
        f_heave = np.max(np.abs(np.add(inertia, ptoPlusBumpstop)))
        return -results[0].fun, f_heave
"""
prob = om.Problem()
prob.model.add_subsystem('test', heavenDynamicsComponent())
prob.setup()
#print(prob.model.list_inputs())
prob.set_val('test.h_f', 4.0)  # Ensure you reference inputs correctly, especially if within a subsystem
prob.set_val('test.h_f_2', 5.2)
prob.set_val('test.D_f', 6.0)
prob.set_val('test.D_s', 20.0)
prob.set_val('test.T_f', 5.2)
prob.set_val('test.g', 9.81)
prob.set_val('test.rho_w', 1000.0)
prob.set_val('test.mass', 208000)
prob.set_val('test.mesh_density', 8)
prob.set_val('test.F_max', 1e7)
prob.set_val('test.x_max', 1.0)
prob.set_val('test.Vs_max', 1e6)
prob.set_val('test.Hs_struct', 3.0)
prob.set_val('test.T_struct', 8.0)


prob.run_model()
#prob.model.add_objective('test.F_heave_max', scaler=-1)
#prob.setup()
prob.model.list_inputs(
)
prob.model.list_outputs()
#
"""