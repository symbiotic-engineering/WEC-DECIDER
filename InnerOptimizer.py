import autograd.numpy as np
import capytaine as cpy
import matplotlib.pyplot as plt
from scipy.optimize import brute
import wecopttool as wot

def body_from_profile(x,y,z,nphi):
    xyz = np.array([np.array([x,y,z]) for x,y,z in zip(x,y,z)])
    body = cpy.FloatingBody(cpy.AxialSymmetricMesh.from_profile(xyz, nphi=nphi))
    return body

def inner_function(f_max, p_max, v_max, fb, wavefreq, amplitude):

    fb.add_translation_dof(name="Heave")
    ndof = fb.nb_dofs
    stiffness = wot.hydrostatics.stiffness_matrix(fb).values
    mass = wot.hydrostatics.inertia_matrix(fb).values
    f1 = 0.05
    nfreq = 50
    freq = wot.frequency(f1, nfreq, False) # False -> no zero frequency
    bem_data = wot.run_bem(fb, freq)
    name = ["PTO_Heave",]
    kinematics = np.eye(ndof)
    controller = None
    loss = None
    pto_impedance = None
    pto = wot.pto.PTO(ndof, kinematics, controller, pto_impedance, loss, name)
    f_add = {'PTO': pto.force_on_wec}
    nsubsteps = 4
    def const_f_pto(wec, x_wec, x_opt, waves): # Format for scipy.optimize.minimize
        f = pto.force_on_wec(wec, x_wec, x_opt, waves, nsubsteps)
        return f_max - np.abs(f.flatten())
    def const_p_pto(wec, x_wec, x_opt, waves): # Format for scipy.optimize.minimize
        p = pto.position(wec, x_wec, x_opt, waves, nsubsteps)
        return p_max - np.abs(p.flatten())
    def const_v_pto(wec, x_wec, x_opt, waves): # Format for scipy.optimize.minimize
        v = pto.velocity(wec, x_wec, x_opt, waves, nsubsteps)
        return v_max - np.abs(v.flatten())
    ineq_cons1 = {'type': 'ineq',
                 'fun': const_f_pto,
                 }
    ineq_cons2 = {'type': 'ineq',
                 'fun': const_p_pto,
                 }
    ineq_cons3 = {'type': 'ineq',
                 'fun': const_v_pto,
                 }
    constraints = [ineq_cons1,ineq_cons2,ineq_cons3]
    wec = wot.WEC.from_bem(
        bem_data,
        inertia_matrix=mass,
        hydrostatic_stiffness=stiffness,
        constraints=constraints,
        friction=None,
        f_add=f_add,)
    phase = 30
    wavedir = 0
    waves = wot.waves.regular_wave(f1, nfreq, wavefreq, amplitude, phase, wavedir)


    obj_fun = pto.mechanical_average_power
    nstate_opt = 2*nfreq


    options = {'maxiter': 200}
    scale_x_wec = 1e1
    scale_x_opt = 1e-3
    scale_obj = 1e-2
    results = wec.solve(
        waves,
        obj_fun,
        nstate_opt,
        optim_options=options,
        scale_x_wec=scale_x_wec,
        scale_x_opt=scale_x_opt,
        scale_obj=scale_obj,
        )
    return results.fun

def make_RM3():
    h_f = 4.0           # height of straight section of float, before the frustum
    h_f_2 = 5.2         # height of entire float, including the frustum at the bottom
    D_s = 6.0           # diameter of spar (inner diameter of float)
    D_f = 20.0          # outer diameter of float
    mesh_density = 5

    z1 = np.linspace(-h_f,-h_f_2,mesh_density)
    x1 = np.linspace(D_f/2, D_s/2,mesh_density)
    y1 = np.linspace(D_f/2, D_s/2,mesh_density)
    bottom_frustum = body_from_profile(x1,y1,z1,mesh_density**2)

    z2 = np.linspace(0,-h_f_2,mesh_density)
    x2 = np.full_like(z2, D_s/2)
    y2 = np.full_like(z2, D_s/2)
    inner_surface = body_from_profile(x2,y2,z2,mesh_density**2)

    z3 = np.linspace(0,-h_f,1+int(mesh_density/2))
    x3 = np.full_like(z3, D_f/2)
    y3 = np.full_like(z3, D_f/2)
    outer_surface = body_from_profile(x3,y3,z3,mesh_density**2)
    # the radius is 15 instead of 10 for some reason!?

    z4 = np.linspace(0,0,mesh_density)
    x4 = np.linspace(D_s/2, D_f/2, mesh_density)
    y4 = np.linspace(D_s/2, D_f/2, mesh_density)
    top_surface = body_from_profile(x4,y4,z4, mesh_density**2)

    RM3 = bottom_frustum + inner_surface + outer_surface + top_surface

    print('RM3 created')
    #frustum.show_matplotlib()
    #inner_surface.show_matplotlib()
    #outer_surface.show_matplotlib()
    #top_surface.show_matplotlib()
    RM3.show_matplotlib()

    return RM3

if __name__ == '__main__':
    RM3 = make_RM3()
    #wavebot = cpy.FloatingBody.from_meshio(RM3, name="WaveBot")

    #inner_function(f_max = 2000.0, p_max = 100.0, v_max = 10000.0, fb=RM3, wavefreq = 0.3, amplitude = 1)

