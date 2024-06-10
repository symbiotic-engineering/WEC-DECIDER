import openmdao.api as om
import autograd.numpy as np
import capytaine as cpy
import math
import xarray as xr
import wecopttool


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
        #print(RM3.nb_dofs)
        #exit(123)
        RM3.mass = np.atleast_2d(208000)
        f1 = 0.05  # Hz
        nfreq = 10
        freq = wecopttool.frequency(f1, nfreq, False)  # False -> no zero frequency
        bem_data = wecopttool.run_bem(RM3,freq,rho= 1000,g = 9.8)
        print("start items")

        for var_name, var_data in bem_data.items():
           outputs[var_name] = var_data
        string_fields = ["g","rho", "body_name", "radiating_dof", "influenced_dof"]
        for var_name, var_data in bem_data.coords.items():
            if var_name not in string_fields:
                outputs[var_name] = var_data

        print('inertia_matrix',outputs['water_depth'])
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