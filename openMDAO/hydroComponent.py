import openmdao.api as om
import autograd.numpy as np
import capytaine as cpy
import math
import xarray as xr
import wecopttool as wot


#Make RM3 - new Dynamic - output(RM3)
class hydroComponent(om.ExplicitComponent):
    def setup(self):
        self.add_input('h_f', 0, desc="height of straight section of float, before the frustum (m)")
        self.add_input('h_f_2', 20, desc="height of entire float, including the frustum at the bottom (m)")  # missing
        self.add_input('D_f', 0, desc="outer diameter of float (m)")
        self.add_input('D_s', 0, desc="diameter of spar (inner diameter of float) (m)")  # mssing in the old dynamic
        self.add_input('T_f', 0)
        self.add_input('mesh_density', 8)  # missing


        #self.add_output("RM3")
        self.add_output('ndof')
        self.add_output('added_mass', shape= (10,1,1))
        self.add_output('radiation_damping',shape= (10,1,1))
        self.add_output('diffraction_force',shape= (10,1,1))
        self.add_output('Froude_Krylov_force', shape= (10,1,1))
        self.add_output('excitation_force', shape= (10,1,1))
        self.add_output('inertia_matrix', shape= (1,1))
        self.add_output('hydrostatic_stiffness', shape= (1,1))


        #self.add_output('g')
        #self.add_output('rho')
        self.add_output('water_depth')
        self.add_output('forward_speed')
        self.add_output('wave_direction')
        self.add_output('omega', shape=(10,))
        self.add_output('period', shape = (10,))
        #se


    def setup_partials(self):
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        h_f = inputs['h_f']
        h_f_2 = inputs['h_f_2']
        D_f = inputs['D_f']
        D_s = inputs['D_s']
        T_f = inputs['T_f']
        mesh_density = int(inputs['mesh_density'][0])

        RM3 = self.make_RM3(h_f[0], h_f_2[0], D_s[0], D_f[0], T_f[0], int(mesh_density))
        RM3.add_translation_dof(name="Heave")
        outputs['ndof'] = RM3.nb_dofs
        kinematics = np.eye(RM3.nb_dofs)
        #print("k", kinematics)
        #print(RM3.nb_dofs)
        #exit(123)
        RM3.mass = np.atleast_2d(208000)
        f1 = 0.05  # Hz
        nfreq = 10
        freq = wot.frequency(f1, nfreq, False)  # False -> no zero frequency
        bem_data = wot.run_bem(RM3,freq,rho= 1000,g = 9.8)
        for var_name, var_data in bem_data.items():
            outputs[var_name] = var_data
        string_fields = ["g", "rho", "body_name", "radiating_dof", "influenced_dof"]
        for var_name, var_data in bem_data.coords.items():
            if var_name not in string_fields:
                outputs[var_name] = var_data


        def inner_function(fb):
            mass = 208000
            f_max = 5000000.0
            x_max = 0.04
            Vs_max = 150000.0
            g = 9.8
            rho = 1000  # rho_w
            Hs = 11.9
            Tp = 17.1
            waves_are_irreg = False
            fb.add_translation_dof(name="Heave")
            ndof = fb.nb_dofs
            fb.mass = np.atleast_2d(mass)
            f1 = 0.05  # Hz
            nfreq = 10
            freq = wot.frequency(f1, nfreq, False)  # False -> no zero frequency
            bem_data = wot.run_bem(fb, freq, rho=rho, g=g)
            name = ["PTO_Heave", ]
            # print(ndof)
            kinematics = np.eye(ndof)
            controller = None
            loss = None
            pto_impedance = None
            pto = wot.pto.PTO(ndof, kinematics, controller, pto_impedance, loss, name)

            # print(pto)

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
                # print("test")
                # print(V,L,p,omega,I)
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
            # print(wec)
            # exit(103)
            # regular waves
            wavefreq = 0.3
            amplitude = 1
            phase = 0
            # wavedir = 0
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

            options = {'maxiter': 3}
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
            print("1 result",wec)
            x_wec, x_opt = wot.decompose_state(results[0].x, ndof=ndof, nfreq=nfreq)
            print(1,x_wec,x_opt)
            inertia = wec.inertia(wec, x_wec, x_opt, waves)
            ptoPlusBumpstop = force_on_wec_with_bumpstop(wec, x_wec, x_opt, waves)
            f_heave = np.max(np.abs(np.add(inertia, ptoPlusBumpstop)))
            return -results[0].fun, f_heave,wec

        def inner_function_2(bem_data):
                ndof = 1
                mass = 208000
                f_max = 5000000.0
                x_max = 0.04
                Vs_max = 150000.0
                g = 9.8
                rho = 1000  # rho_w
                Hs = 11.9
                Tp = 17.1
                #print(Hs, Tp, waves_are_irreg)
                # print(Tp)
                # exit(101)

                g = 9.8
                rho = 1000  # rho_w
                # fb.add_translation_dof(name="Heave")
                # ndof = fb.nb_dofs
                # fb.mass = np.atleast_2d(mass)
                f1 = 0.05  # Hz
                nfreq = 10
                # freq = wot.frequency(f1, nfreq, False)  # False -> no zero frequency
                # bem_data = wot.run_bem(fb, freq, rho=rho, g=g)
                name = ["PTO_Heave", ]
                # print(ndof)
                kinematics = np.eye(ndof)
                controller = None
                loss = None
                pto_impedance = None
                pto = wot.pto.PTO(ndof, kinematics, controller, pto_impedance, loss, name)

                # print(pto)

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
                    # print("test")
                    # print(V,L,p,omega,I)
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
                # print(wec)
                # exit(103)
                # regular waves
                wavefreq = 0.3
                amplitude = 1
                phase = 0
                # wavedir = 0
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

                options = {'maxiter': 3}
                scale_x_wec = 1e4
                scale_x_opt = 1e-3
                scale_obj = 1e-3
                waves_are_irreg = False
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
                print("2 result",wec)
                x_wec, x_opt = wot.decompose_state(results[0].x, ndof=ndof, nfreq=nfreq)
                print(2, x_wec, x_opt)
                inertia = wec.inertia(wec, x_wec, x_opt, waves)
                ptoPlusBumpstop = force_on_wec_with_bumpstop(wec, x_wec, x_opt, waves)
                f_heave = np.max(np.abs(np.add(inertia, ptoPlusBumpstop)))
                return -results[0].fun, f_heave, wec


        print("starting test")

        _,_,wec1 = inner_function(RM3)
        print(inner_function(RM3))

        new_coords = {
            'g': [9.8],
            'rho': [1000.0],
            'body_name': 'axisymmetric_mesh+axisymmetric_mesh+axisymmetric_mesh+axisymmetric_mesh_immersed',
            'water_depth': outputs['water_depth'],
            'forward_speed': outputs['forward_speed'],
            'wave_direction': outputs['wave_direction'],
            'omega': xr.DataArray(outputs['omega'], dims=['omega']),
            'radiating_dof': xr.DataArray(np.array(['Heave']), dims=['radiating_dof']),
            'influenced_dof': xr.DataArray(np.array(['Heave']), dims=['influenced_dof']),
            'period': xr.DataArray(outputs['period'], dims=['omega'])
        }



        new_data_vars = {
            'added_mass': (('omega', 'radiating_dof', 'influenced_dof'), outputs['added_mass']),
            'radiation_damping': (('omega', 'radiating_dof', 'influenced_dof'), outputs['radiation_damping']),
            'diffraction_force': (
            ('omega', 'wave_direction', 'influenced_dof'), outputs['diffraction_force'].astype(np.complex128)),
            'Froude_Krylov_force': (
            ('omega', 'wave_direction', 'influenced_dof'), outputs['Froude_Krylov_force'].astype(np.complex128)),
            'excitation_force': (
            ('omega', 'wave_direction', 'influenced_dof'), outputs['excitation_force'].astype(np.complex128)),
            'inertia_matrix': (('influenced_dof', 'radiating_dof'), outputs['inertia_matrix']),
            'hydrostatic_stiffness': (('influenced_dof', 'radiating_dof'), outputs['hydrostatic_stiffness'])
        }
        print("start datavars")

        new_array = xr.Dataset(new_coords, new_data_vars)
        _, _, wec2 = inner_function_2(new_array)
        print(inner_function_2(new_array))
        #print(wec1.ns == wec2.nstate_wec)
        attributes_to_check = [
            '_freq', '_time', '_time_mat', '_derivative_mat',
            '_derivative2_mat', '_forces', '_constraints',
            '_inertia_in_forces', '_inertia_matrix', '_ndof',
            '_inertia', '_dof_names'
        ]
        attr_diff = []
        for attr in attributes_to_check:
            if not np.array_equal(getattr(wec1, attr), getattr(wec2, attr)):
                attr_diff.append(attr)
                print("helloworld1")
                #return False

        # Check if the dynamic properties are equal
        dynamic_properties_to_check = [
            'forces', 'constraints', 'inertia_in_forces',
            'inertia_matrix', 'inertia', 'dof_names',
            'ndof', 'frequency', 'f1', 'nfreq', 'omega',
            'period', 'w1', 'time', 'time_mat', 'derivative_mat',
            'derivative2_mat', 'dt', 'tf', 'nt', 'ncomponents', 'nstate_wec'
        ]
        prop_diff = []
        for prop in dynamic_properties_to_check:
            if not np.array_equal(getattr(wec1, prop), getattr(wec2, prop)):
                prop_diff.append(prop)
                print("helloworld2")
                #return False

        #print(attr_diff,prop_diff)
        #print(wec1.forces,wec2.forces)
        #exit(250)





    #exit(100)
    """
        new_coords = {
            'g': outputs['g'],
            'rho': outputs['rho'],
            'body_name': 'axisymmetric_mesh+axisymmetric_mesh+axisymmetric_mesh+axisymmetric_mesh_immersed',
            'water_depth': outputs['water_depth'],
            'forward_speed': outputs['forward_speed'],
            'wave_direction': outputs['wave_direction'],
            'omega': xr.DataArray(outputs['omega'], dims=['omega']),
            'radiating_dof': xr.DataArray(np.array(['Heave']), dims=['radiating_dof']),
            'influenced_dof': xr.DataArray(np.array(['Heave']), dims=['influenced_dof']),
            'period': xr.DataArray(outputs['period'], dims=['omega'])
        }

        print(new_coords)
        print(bem_data.coords['radiating_dof'] == new_coords['radiating_dof'])


        new_data_vars = {
            'added_mass': (('omega', 'radiating_dof', 'influenced_dof'),outputs['added_mass']),
            'radiation_damping': (('omega', 'radiating_dof', 'influenced_dof'),outputs['radiation_damping']),
            'diffraction_force': (('omega', 'wave_direction', 'influenced_dof'),outputs['diffraction_force'].astype(np.complex128)),
            'Froude_Krylov_force': (('omega', 'wave_direction', 'influenced_dof'),outputs['Froude_Krylov_force'].astype(np.complex128)),
            'excitation_force': (('omega', 'wave_direction', 'influenced_dof'),outputs['excitation_force'].astype(np.complex128)),
            'inertia_matrix': (('influenced_dof', 'radiating_dof'), outputs['inertia_matrix']),
            'hydrostatic_stiffness': (('influenced_dof', 'radiating_dof'),outputs['hydrostatic_stiffness'])
        }
        print("start datavars")

        new_array = xr.Dataset(new_coords, new_data_vars)
        print(new_array)
        print(type(new_array))
        print("end")
        print("test")
        import wecopttool as wot
        wec1 = wot.WEC.from_bem(
            bem_data
            #constraints=constraints,
            #f_add=f_add,
        )
        print(wec1)
        print(type(wec1))
        print(bem_data.attrs)
        wec2 = wot.WEC.from_bem(
            new_array
            # constraints=constraints,
            # f_add=f_add,
        )
        print("wec1")
        print(wec1.time_mat)
        print("wec2")
        print(wec2.time_mat)

        #print(wec1==wec2)
        print(type(bem_data.data_vars))
        print("items")


        #print(bem_data.data_vars == bem_data.items())
        RM3 = 0
        #exit(241)
        #outputs['RM3'] = RM3
        """
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

    def body_from_profile(self, x, y, z, nphi):
        xyz = np.array([np.array([x/math.sqrt(2),y/math.sqrt(2),z]) for x,y,z in zip(x,y,z)])    # /sqrt(2) to account for the scaling
        body = cpy.FloatingBody(cpy.AxialSymmetricMesh.from_profile(xyz, nphi=nphi))
        return body