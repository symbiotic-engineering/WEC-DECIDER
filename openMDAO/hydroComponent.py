import openmdao.api as om
import autograd.numpy as np
import capytaine as cpy
import math

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


        self.add_output("RM3")

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
        print("RM3")
        print(type(RM3))
        RM3.add_translation_dof(name="Heave")
        RM3.ndof = RM3.nb_dofs
        print(RM3.nb_dofs)
        print(wecopttool.run_bem(RM3,rho= 1000,g = 9.8))
        RM3 = 0
        exit(241)
        outputs['RM3'] = RM3

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