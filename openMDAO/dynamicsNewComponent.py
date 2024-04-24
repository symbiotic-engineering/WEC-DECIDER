import openmdao.api as om
import numpy as np
import autograd.numpy as np
import capytaine as cpy
from scipy.optimize import brute
import wecopttool as wot
import math

class DynamicsNewComponent(om.ExplicitComponent):
    def setup(self):
        #11 inputs
        self.add_input('f_max', 0, desc="maximum force (N)")
        self.add_input('x_max', 0, desc="maximum position (m)")
        self.add_input('Vs_max', 0, desc="maximum voltage (V)")
        self.add_input('D_f', 0, desc="outer diameter of float (m)")
        self.add_input('D_s', 0, desc="diameter of spar (inner diameter of float) (m)")
        self.add_input('h_f', 0, desc="height of straight section of float, before the frustum (m)")
        self.add_input('h_f_2', 0, desc="height of entire float, including the frustum at the bottom (m)")
        self.add_input('mesh_density', 0)
        self.add_input('mass', 0, desc="mass of RM3 (kg)")
        self.add_input('Hs', 0)
        self.add_input('Tp', 0)

        # 2 outputs
        self.add_output('P_elec')
        self.add_output('f_heave')

    def setup_partials(self):
        self.declare_partials('*', '*', method='fd')
    
    def compute(self, inputs, outputs):
        #retrieve inputs
        f_max = inputs['f_max'][0]
        x_max = inputs['x_max'][0]
        Vs_max = inputs['Vs_max'][0]
        D_f = inputs['D_f'][0]
        D_s = inputs['D_s'][0]
        h_f = inputs['h_f'][0]
        h_f_2 = inputs['h_f_2'][0]
        mesh_density = int(inputs['mesh_density'][0])
        mass = inputs['mass'][0]
        Hs = inputs['Hs'][0]
        Tp = inputs['Tp'][0]
        #compute
        RM3 = self.make_RM3(h_f, h_f_2, D_s, D_f, mesh_density)
        P_elec, f_heave = self.inner_function(mass, f_max, x_max, Vs_max, RM3, Hs, Tp, waves_are_irreg=False)
        #assign outputs
        outputs['P_elec'] = P_elec
        outputs['f_heave'] = f_heave
    
    def body_from_profile(self, x, y, z, nphi):
        xyz = np.array([np.array([x/math.sqrt(2),y/math.sqrt(2),z]) for x,y,z in zip(x,y,z)])    # /sqrt(2) to account for the scaling
        body = cpy.FloatingBody(cpy.AxialSymmetricMesh.from_profile(xyz, nphi=nphi))
        return body
    
    def make_RM3(self, h_f, h_f_2, D_s, D_f, mesh_density):
        #normal vectors have to be facing outwards
        z1 = np.linspace(-h_f_2,-h_f,mesh_density)
        x1 = np.linspace(D_s/2, D_f/2, mesh_density)
        y1 = np.linspace(D_s/2, D_f/2,mesh_density)
        bottom_frustum = self.body_from_profile(x1,y1,z1,mesh_density**2)
        z2 = np.linspace(0, -h_f_2, mesh_density)
        x2 = np.full_like(z2, D_s/2)
        y2 = np.full_like(z2, D_s/2)
        inner_surface = self.body_from_profile(x2,y2,z2,mesh_density**2)
        z3 = np.linspace(-h_f, 0, 1+int(mesh_density/2))
        x3 = np.full_like(z3, D_f/2)
        y3 = np.full_like(z3, D_f/2)
        outer_surface = self.body_from_profile(x3,y3,z3,mesh_density**2)
        z4 = np.linspace(0,0,mesh_density)
        x4 = np.linspace(D_f/2, D_s/2, mesh_density)
        y4 = np.linspace(D_f/2, D_s/2, mesh_density)
        top_surface = self.body_from_profile(x4,y4,z4, mesh_density**2)
        RM3 = bottom_frustum + outer_surface + top_surface + inner_surface
        return RM3
    
    def inner_function(self, mass, f_max, x_max, Vs_max, fb, Hs, Tp, waves_are_irreg=False):
        g = 9.8
        rho = 1000
        fb.add_translation_dof(name="Heave")
        ndof = fb.nb_dofs
        fb.mass = np.atleast_2d(mass)
        f1 = 0.05# Hz
        nfreq = 8
        freq = wot.frequency(f1, nfreq, False) # False -> no zero frequency
        bem_data = wot.run_bem(fb, freq, rho=rho, g=g)
        name = ["PTO_Heave",]
        kinematics = np.eye(ndof)
        controller = None
        loss = None
        pto_impedance = None
        pto = wot.pto.PTO(ndof, kinematics, controller, pto_impedance, loss, name)
        def force_on_wec_with_bumpstop(wec, x_wec, x_opt, waves, nsubsteps=1):
            pos = pto.position(wec, x_wec, x_opt, waves, nsubsteps)
            vel = pto.velocity(wec, x_wec, x_opt, waves, nsubsteps)
            b = 14    #
            k = 10e6  # N/m
            bumpstop = -k * (np.abs(pos) + x_max) * np.sign(pos) - b * vel
            condition = [(pos > x_max) | (pos < -x_max)]
            bumpstop = np.where(condition, bumpstop, 0.0)
            f_tot = pto.force_on_wec(wec, x_wec, x_opt, waves, nsubsteps) + bumpstop
            return f_tot
        f_add = {
                'PTO+bumpstop' : force_on_wec_with_bumpstop
                }
        #contraints
        nsubsteps = 4
        def const_f_pto(wec, x_wec, x_opt, waves): # Format for scipy.optimize.minimize
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
            Vs = np.sqrt(V**2+(L*p*omega*I)**2)
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
        #regular waves
        wavefreq = 0.3
        amplitude = 1
        phase = 0
        wavedir = 0
        waves_regular = wot.waves.regular_wave(f1, nfreq, wavefreq, amplitude, phase, wavedir)
        
        #irregular waves
        # number of realizations to reach 20 minutes of total simulation time
        minutes_needed = 20
        nrealizations = minutes_needed*60*f1
        print(f'Number of realizations for a 20 minute total simulation time: {nrealizations}')
        nrealizations = 2 # overwrite nrealizations to reduce run-time
        fp = 1/Tp
        spectrum = lambda f: wot.waves.pierson_moskowitz_spectrum(f, fp, Hs)
        efth = wot.waves.omnidirectional_spectrum(f1, nfreq, spectrum, "Pierson-Moskowitz")
        waves_irregular = wot.waves.long_crested_wave(efth, nrealizations)

        obj_fun = pto.average_power
        nstate_opt = 2*nfreq

        options = {'maxiter': 1000}
        scale_x_wec = 1e1
        scale_x_opt = 1e-4
        scale_obj = 1e-3

        def callbackF(wec, x_wec, x_opt, waves):
            global Nfeval
            print ('{0:4d}   {1: 3.6f}'.format(Nfeval, obj_fun(wec, x_wec, x_opt, waves)))
            Nfeval += 1

        if waves_are_irreg:
            waves = waves_irregular
        else:
            waves = waves_regular

        results = wec.solve(
            waves,
            obj_fun,
            nstate_opt,
            optim_options=options,
            x_wec_0=np.ones(nfreq*2) *1e-7,
            x_opt_0=np.ones(nfreq*2) *1e0,
            scale_x_wec=scale_x_wec,
            scale_x_opt=scale_x_opt,
            scale_obj=scale_obj,
            )
        x_wec, x_opt = wot.decompose_state(results[0].x, ndof=ndof, nfreq=nfreq)
        inertia = wec.inertia(wec,x_wec, x_opt, waves)
        ptoPlusBumpstop = force_on_wec_with_bumpstop(wec, x_wec, x_opt, waves)
        f_heave = np.max(np.abs(np.add(inertia, ptoPlusBumpstop)))
        return results[0].fun, f_heave
    

prob = om.Problem()
#componentTest
prob.model.add_subsystem('test', DynamicsNewComponent())
prob.setup()
print(prob.model.list_inputs())
# prob.driver = om.ScipyOptimizeDriver()
# prob.driver.options['optimizer'] = 'SLSQP'
# prob.model.add_design_var('f_max')
# prob.model.add_objective('f_heave')
prob.set_val('test.f_max', 1e6)
prob.set_val('test.x_max',  1)
prob.set_val('test.Vs_max', 1.5e5)
prob.set_val('test.D_f', 20.0)
prob.set_val('test.D_s', 6.0)
prob.set_val('test.h_f', 4.0)
prob.set_val('test.h_f_2', 5.2)
prob.set_val('test.mesh_density', 5)
prob.set_val('test.mass', 208000)
prob.set_val('test.Hs', 3.0)
prob.set_val('test.Tp', 8.0)


prob.run_model()
prob.model.list_inputs(val=True)
prob.model.list_outputs(val=True)
print(prob.get_val('test.P_elec'))
print(prob.get_val('test.f_heave'))
