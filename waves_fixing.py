import autograd.numpy as np
import capytaine as cpy
import matplotlib.pyplot as plt
from scipy.optimize import brute
import wecopttool as wot
import math

def body_from_profile(x,y,z,nphi):
    xyz = np.array([np.array([x/math.sqrt(2),y/math.sqrt(2),z]) for x,y,z in zip(x,y,z)])    # /sqrt(2) to account for the scaling
    body = cpy.FloatingBody(cpy.AxialSymmetricMesh.from_profile(xyz, nphi=nphi))
    return body

def inner_function(f_max, p_max, v_max, fb, wavefreq, amplitude):

    fb.add_translation_dof(name="Heave")
    ndof = fb.nb_dofs
    
    stiffness = wot.hydrostatics.stiffness_matrix(fb).values
    mass = 208000

    f1 = 0.02 # Hz
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
    
    #Contraints
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
        f_add=f_add,
    )
    
    
    #regular waves
    phase = 0
    wavedir = 0
    waves = {}
    waves['regular'] = wot.waves.regular_wave(f1, nfreq, wavefreq, amplitude, phase, wavedir)
    
    #irregular waves
    wave_cases = {
        'south_max_90': {'Hs': 0.21, 'Tp': 3.09},
        'south_max_annual': {'Hs': 0.13, 'Tp': 2.35},
        'south_max_occurrence': {'Hs': 0.07, 'Tp': 1.90},
        'south_min_10': {'Hs': 0.04, 'Tp': 1.48},
        'north_max_90': {'Hs': 0.25, 'Tp': 3.46},
        'north_max_annual': {'Hs': 0.16, 'Tp': 2.63},
        'north_max_occurrence': {'Hs': 0.09, 'Tp': 2.13},
        'north_min_10': {'Hs': 0.05, 'Tp': 1.68},
        'testing' : {'Hs': 0.2, 'Tp': 5}
        }
    def irregular_wave(hs, tp):
        fp = 1/tp
        spectrum = lambda f: wot.waves.pierson_moskowitz_spectrum(f, fp, hs)
        efth = wot.waves.omnidirectional_spectrum(f1, nfreq, spectrum, "Pierson-Moskowitz")
        return wot.waves.long_crested_wave(efth)

    for case, data in wave_cases.items():
        waves[case] = irregular_wave(data['Hs'], data['Tp'])


    
    
    obj_fun = pto.mechanical_average_power
    nstate_opt = 2*nfreq


    options = {'maxiter': 200}
    scale_x_wec = 1e1
    scale_x_opt = 1e-3
    scale_obj = 1e-2
    
    results = wec.solve(
        waves['testing'],
        obj_fun,
        nstate_opt,
        optim_options=options,
        scale_x_wec=scale_x_wec,
        scale_x_opt=scale_x_opt,
        scale_obj=scale_obj,
        )
    
    # graph for hyrdodynamics coefficients
    #rho=1030
    #Added_mass_norm = bem_data['added_mass']/rho
    #radiation_dampin_normg = bem_data['radiation_damping']/(rho*f1*2*math.pi)
    
    #fig, axes = plt.subplots(3,1)
    #Added_mass_norm.plot(ax = axes[0])
    #radiation_dampin_normg.plot(ax = axes[1])
    #axes[2].plot(bem_data['omega'],np.abs(np.squeeze(bem_data['diffraction_force'].values)), color = 'orange')
    #axes[2].set_ylabel('abs(diffraction_force)', color = 'orange')
    #axes[2].tick_params(axis ='y', labelcolor = 'orange')
    #ax2r = axes[2].twinx()
    #ax2r.plot(bem_data['omega'], np.abs(np.squeeze(bem_data['Froude_Krylov_force'].values)), color = 'blue')
    #ax2r.set_ylabel('abs(FK_force)', color = 'blue')
    #ax2r.tick_params(axis ='y', labelcolor = 'blue')

    #for axi in axes:
        #axi.set_title('')
        #axi.label_outer()
        #axi.grid()

    #axes[-1].set_xlabel('Frequency [rad/s]')

    return results.fun


def make_RM3():
    h_f = 4.0           # height of straight section of float, before the frustum
    h_f_2 = 5.2         # height of entire float, including the frustum at the bottom
    D_s = 6.0           # diameter of spar (inner diameter of float)
    D_f = 20.0          # outer diameter of float
    mesh_density = 5

    #normal vectors have to be facing outwards
    z1 = np.linspace(-h_f_2,-h_f,mesh_density)
    x1 = np.linspace(D_s/2, D_f/2, mesh_density) 
    y1 = np.linspace(D_s/2, D_f/2,mesh_density)
    bottom_frustum = body_from_profile(x1,y1,z1,mesh_density**2)

    z2 = np.linspace(0, -h_f_2, mesh_density)
    x2 = np.full_like(z2, D_s/2)
    y2 = np.full_like(z2, D_s/2)
    inner_surface = body_from_profile(x2,y2,z2,mesh_density**2)

    z3 = np.linspace(-h_f, 0, 1+int(mesh_density/2))
    x3 = np.full_like(z3, D_f/2)
    y3 = np.full_like(z3, D_f/2)
    outer_surface = body_from_profile(x3,y3,z3,mesh_density**2)

    z4 = np.linspace(0,0,mesh_density)
    x4 = np.linspace(D_f/2, D_s/2, mesh_density)
    y4 = np.linspace(D_f/2, D_s/2, mesh_density)
    top_surface = body_from_profile(x4,y4,z4, mesh_density**2)

    RM3 = bottom_frustum + outer_surface + top_surface + inner_surface

    
    print('RM3 created')
    RM3.show_matplotlib()

    return RM3

if __name__ == '__main__':
    RM3 = make_RM3()
    #wavebot = cpy.FloatingBody.from_meshio(RM3, name="WaveBot")

    #inner_function(f_max = 2000.0, p_max = 100.0, v_max = 10000.0, fb=RM3, wavefreq = 0.3, amplitude = 1)


inner_function(f_max = 2000.0, p_max = 0.05, v_max = 0.1, fb=RM3, wavefreq = 0.233, amplitude = 0.1)


#outer loop
f_max = 2000.0 
p_max = 0.05
v_max = 0.1
len_f_ = 4
len_p_ = 4
len_v_ = 4
f_ = np.linspace(1, f_max, len_f_)
p_ = np.linspace(0.01, p_max, len_p_)
v_ = np.linspace(0.01, v_max, len_v_)

f = [f_[i] for i in range(len_f_) for j in range(len_p_) for k in range(len_v_)]
p = [p_[j] for i in range(len_f_) for j in range(len_p_) for k in range(len_v_)]
v = [v_[k] for i in range(len_f_) for j in range(len_p_) for k in range(len_v_)]
X = [ inner_function(f_[i], p_[j], v_[k], fb=RM3, wavefreq = 0.3, amplitude = 1) for i in range(len_f_) for j in range(len_p_) for k in range(len_v_) ]

    
ax = plt.subplot(projection="3d")
sc = ax.scatter(f, v, p, c=X, marker='o', s=25, cmap="autumn")
plt.colorbar(sc)
ax.set_xlabel("f_max")
ax.set_ylabel("p_max")
ax.set_zlabel("v_max")
plt.show()