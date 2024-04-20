# %%
import autograd.numpy as np
import capytaine as cpy
import matplotlib.pyplot as plt
from scipy.optimize import brute
import wecopttool as wot
import math
import pandas as pd

# %%
def body_from_profile(x,y,z, nphi):
    xyz = np.array([np.array([x/math.sqrt(2),y/math.sqrt(2),z]) for x,y,z in zip(x,y,z)])    # /sqrt(2) to account for the scaling
    body = cpy.FloatingBody(cpy.AxialSymmetricMesh.from_profile(xyz, nphi=nphi))
    return body

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


# %%
def inner_function(f_max, x_max, Vs_max, fb, wavefreq, amplitude, x_wec_0, x_opt_0):
    
    g = 9.8
    rho = 1000
    fb.add_translation_dof(name="Heave")
    ndof = fb.nb_dofs
    fb.mass = np.atleast_2d(208000)
    
    f1 = 0.05# Hz
    nfreq = 20
    freq = wot.frequency(f1, nfreq, False) # False -> no zero frequency
    bem_data = wot.run_bem(fb, freq, rho=rho, g=g)
    
    name = ["PTO_Heave"]
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

    # def f_bumpstop(wec, x_wec, x_opt, waves, nsubsteps=1):
    #     pos = pto.position(wec, x_wec, x_opt, waves, nsubsteps)
    #     vel = pto.velocity(wec, x_wec, x_opt, waves, nsubsteps)
    #     b = 14    # 
    #     k = 10e6  # N/m        

    #     # bumpstop = np.array([])
    #     # for i in range(len(pos)):
    #     #     if pos[i] > x_max: 
    #     #         bump = k * (pos[i] - x_max) + b * vel[i]
    #     #     elif pos[i] < -x_max: 
    #     #         bump = k * (pos[i] + x_max) + b * vel[i]
    #     #     else: 
    #     #         bump = [0.0]
    #     #     bumpstop = np.concatenate((bumpstop, bump), axis = 0)
        
    #     # bumpstop = bumpstop.reshape(-1, 1)
       
    #     # f_bumpstop = np.dot(time_matrix, bumpstop)

    #     bumpstop = -k * (np.abs(pos) + x_max) * np.sign(pos) - b * vel
    #     condition = [(pos > x_max) | (pos < -x_max)]
    #     bumpstop = np.where(condition, bumpstop, 0.0)

        return bumpstop
    

    f_add = {
            'PTO+bumpstop' : force_on_wec_with_bumpstop
            # 'PTO': pto.force_on_wec,
            # 'bumpstop': f_bumpstop,
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
        f_add=f_add
        )
    
    #regular waves
    phase = 0
    wavedir = 0
    waves_regular = wot.waves.regular_wave(f1, nfreq, wavefreq, amplitude, phase, wavedir)
    
    #irregular waves
    Hs = 3
    Tp = 8
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
    scale_x_wec = 1e3
    scale_x_opt = 1e-4
    scale_obj = 1e-5

    
    
    def callbackF(wec, x_wec, x_opt, waves):
        global Nfeval
        print ('{0:4d}   {1: 3.6f}'.format(Nfeval, obj_fun(wec, x_wec, x_opt, waves)))
        Nfeval += 1

    waves = waves_irregular
    
    results = wec.solve(
        waves,
        obj_fun,
        nstate_opt,
        optim_options=options,
        x_wec_0=x_wec_0,
        x_opt_0=x_opt_0,
        callback=callbackF,
        scale_x_wec=scale_x_wec,
        scale_x_opt=scale_x_opt,
        scale_obj=scale_obj,
        )
    # bem_data.added_mass.plot()
    # bem_data.radiation_damping.plot()
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

    # nsubstep_postprocess = 4
    # wec_fdom, wec_tdom = wec.post_process(results, waves['regular'], nsubstep_postprocess)
    # pto_fdom, pto_tdom = pto.post_process(wec, results, waves['regular'], nsubstep_postprocess)
    # fig, ax = plt.subplots(3,1)

    # #Pos, Vel, Wave Elevation
    # (wec_tdom["pos"]).plot(ax=ax[0], label = 'Pos')
    # (wec_tdom["vel"]).plot(ax=ax[0], label = 'Vel')
    # (wec_tdom["wave_elev"]).plot(ax=ax[0], label = 'Wave')
    # ax[0].legend()
    # ax[0].set_ylabel("Position [m] / Vel [m/s]"); ax[0].set_title("")

    # # Force
    # (pto_tdom["force"]).plot(ax=ax[1], label = 'PTO fore')
    # ax[1].legend()
    # ax[1].set_title("")

    # # Power
    # (pto_tdom['power'].loc['mech',:,:]/1e3).plot(ax=ax[2], label='Mech. power')
    # (pto_tdom['power'].loc['elec',:,:]/1e3).plot(ax=ax[2], linestyle='dashed',
    #                             label="Elec. power")
    # # ax[3].axhline(-1*power_max/1e3, linestyle='dotted', color='black',
    #     # label=f'Max mech. power ($\pm${power_max/1e3:.0f}kW)')
    # # ax[3].axhline(1*power_max/1e3, linestyle='dotted', color='black')
    # ax[2].grid(color='0.75', linestyle='-', linewidth=0.5, axis='x')
    # ax[2].legend(loc='upper right')
    # ax[2].set_title('')
    # ax[2].set_ylabel('Power [kW]')
    x_wec, x_opt = wot.decompose_state(results[0].x, ndof=ndof, nfreq=nfreq)
    
    inertia = wec.inertia(wec,x_wec, x_opt, waves)
    ptoPlusBumpstop = force_on_wec_with_bumpstop(wec, x_wec, x_opt, waves)
    f_heave = np.max(np.abs(np.add(inertia, ptoPlusBumpstop)))
    
    print('f_heave',np.add(inertia, ptoPlusBumpstop), f_heave)
    
    # print('INERTIA!', wec.inertia(wec,x_wec, x_opt, waves['regular']))
    # print('f_add', force_on_wec_with_bumpstop(wec, x_wec, x_opt, waves['regular'], nsubsteps=1))
    # print('sum', np.add(wec.inertia(wec,x_wec, x_opt, waves['regular']), force_on_wec_with_bumpstop(wec, x_wec, x_opt, waves['regular'], nsubsteps=1)))
    return [results[0].fun,results[0], f_heave, x_wec, x_opt]

# %%
## 1D ##

Nfeval = 1
tt2 = inner_function(f_max = 5e6, x_max = 1000,  Vs_max = 3e10, fb=RM3, wavefreq = 0.3, amplitude = 1, x_wec_0=np.ones(40)*1e-3, x_opt_0=np.ones(40)*1e4)
print(tt2[0])

# number = 5
# xrange = np.linspace(0.05, 0.1, number)
# frange = np.linspace(1e5, 1e6, number)
# temp_result = np.array([])
# X1 = np.full(number, 0.0)
# for i in range(number):
#     Nfeval = 1
#     try:
#         if i == 0:
#             temp_result = inner_function(f_max = 1e5, x_max = xrange[i], Vs_max = 1e5, fb=RM3, wavefreq = 0.3, amplitude = 1, x_wec_0=np.ones(40) *1e-7, x_opt_0=np.ones(40) *1e0)
#             X1[i] = temp_result[0]
#         else:
#             temp_result = inner_function(f_max = 1e5, x_max = xrange[i], Vs_max = 1e5, fb=RM3, wavefreq = 0.3, amplitude = 1, x_wec_0= temp_result[1][0:len(temp_result[1])//2], x_opt_0=temp_result[1][len(temp_result[1])//2:])
#             X1[i] = temp_result[0]
#         # X1[i] = inner_function(f_max = frange[i], x_max = 0.05, Vs_max = 1e5, fb=RM3, wavefreq = 0.3, amplitude = 1)
#     except:
#             pass
#     print(X1)

# fig, ax = plt.subplots()
# ax.plot(xrange, np.abs(X1))
    

# %%
print(tt2[4])

# %%
## 2D ##

f_max = 1e6
Vs_max = 3e5
len_f_ = 5
len_Vs_ = 5
len_x_ = 5
f_ = np.linspace(1e4, 2e4, len_f_)
# f_ = [1e3, 1e4, 1e5, 1e6, 1e7]
x_= np.linspace(0.16, 0.2, len_x_)
Vs_ = np.linspace(3e4, Vs_max, len_Vs_)
# Vs_ = [200, 400, 600, 800, 1000]
X2 = np.full((len_f_, len_x_), np.nan)
t = 1
temp_result = np.array([])
for i in range(len_f_):
    for j in range(len_x_):
        Nfeval = 1
        try:
            if j == 0:
                temp_result = inner_function(f_max = f_[i], x_max =  x_[j], Vs_max = 1e5, fb=RM3, wavefreq = 0.3, amplitude = 1, x_wec_0=np.ones(40) *1e-7, x_opt_0=np.ones(40) *1e0)
                X2[i,j] = temp_result[0]
            else:
                temp_result = inner_function(f_max = f_[i], x_max =  x_[j], Vs_max = 1e5, fb=RM3, wavefreq = 0.3, amplitude = 1, x_wec_0= temp_result[1][0:len(temp_result[1])//2], x_opt_0=temp_result[1][len(temp_result[1])//2:])
                X2[i,j] = temp_result[0]
        except Exception as error:
            print("An error occurred:", error)

        print("t", t)
        print('Y', X2)
        t += 1

print('x_', x_)
print('f_', f_)
print('X2', X2)

fig, ax = plt.subplots()

im = ax.pcolormesh(x_, f_, X2, shading='nearest')
fig.colorbar(im, ax=ax, label="Ave Elec Power")
ax.set_xlabel('x_max')
ax.set_ylabel('F_max')

# %%
print('x_', x_)
print('f_', f_)
print('Y', X2)

fig, ax = plt.subplots()

im = ax.pcolormesh(x_, f_, X2, shading='nearest')
fig.colorbar(im, ax=ax, label="Ave Elec Power")
ax.set_xlabel('x_max')
ax.set_ylabel('F_max')

# %%
print(np.linspace(3e4, 3e5, 5))

# %%
## 3D ##

f_max = 2e5
x_max = 0.15
Vs_max = 4e4
len_f_ = 5
len_x_ = 5
len_Vs_ = 5
f_ = np.linspace(1e5, f_max, len_f_)
x_ = np.linspace(0.1, x_max, len_x_)
Vs_ = np.linspace(3e4, Vs_max, len_Vs_)

f = [f_[i] for i in range(len_f_) for j in range(len_x_) for k in range(len_Vs_)]
x = [x_[j] for i in range(len_f_) for j in range(len_x_) for k in range(len_Vs_)]
Vs = [Vs_[k] for i in range(len_f_) for j in range(len_x_) for k in range(len_Vs_)]

t = 0
X3 = np.full(len_f_*len_x_*len_Vs_, np.nan)
temp_result = np.array([])
for i in range(len_f_):
    for j in range(len_x_):
        for k in range(len_Vs_):
            Nfeval = 1
            try:
                # if k == 0:
                    temp_result = inner_function(f_max = f_[i], x_max =  x_[j], Vs_max = Vs_[k], fb=RM3, wavefreq = 0.3, amplitude = 1, x_wec_0=np.ones(40) *1e-4, x_opt_0=np.ones(40) *1e-4)
                    X3[t] = temp_result[0]
                # else:
                #     temp_result = inner_function(f_max = f_[i], x_max =  x_[j], Vs_max = Vs_[k], fb=RM3, wavefreq = 0.3, amplitude = 1, x_wec_0= temp_result[1][0:len(temp_result[1])//2], x_opt_0=temp_result[1][len(temp_result[1])//2:])
                #     X3[t] = temp_result[0]
            except Exception as error:
                print("An error occurred:", error)
            print('t', t)
            print('X', X3)
            t += 1

ax = plt.subplot(projection="3d")
sc = ax.scatter(f, x, Vs, c=X3, marker='o', s=25, depthshade=False)
plt.colorbar(sc, pad = 0.15)
ax.set_xlabel("f_max")
ax.set_ylabel("x_max")
ax.set_zlabel("Vs_max")
plt.show()
print(t)

# %%
np.linspace(500,5000,5)


# %%
ax = plt.subplot(projection="3d")
sc = ax.scatter(f, x, Vs, c=X3, marker='o', s=25, depthshade=False)
plt.colorbar(sc, pad = 0.15)
ax.set_xlabel("f_max")
ax.set_ylabel("x_max")
ax.set_zlabel("Vs_max")
plt.show()
print(t)

# %%
print(f)

# %%
df = pd.DataFrame({'f_max': f,
                   'x_max': x,
                   'Vs_max': Vs,
                   'elec power': X3})
df.to_csv('NewBS_output.csv', index=False)  

# %%
f_mod = f
x_mod = x
Vs_mod = Vs
X3_mod = X3
X3_mod = X3_mod.reshape((5,5,5))

# %%

fig, ax = plt.subplots()

im = ax.pcolormesh(Vs_, x_, X3_mod[0], shading='nearest')
fig.colorbar(im, ax=ax, label="Ave Mech Power")
ax.set_xlabel('Vs_max')
ax.set_ylabel('x_max')
ax.set_title('f_max = 1e5')

# %%
fig, ax = plt.subplots()

im = ax.pcolormesh(Vs_, x_, X3_mod[1], shading='nearest')
fig.colorbar(im, ax=ax, label="Ave Mech Power")
ax.set_xlabel('Vs_max')
ax.set_ylabel('x_max')
ax.set_title('f_max = 3.25e5')

# %%
fig, ax = plt.subplots()

im = ax.pcolormesh(Vs_, x_, X3_mod[2], shading='nearest')
fig.colorbar(im, ax=ax, label="Ave Mech Power")
ax.set_xlabel('Vs_max')
ax.set_ylabel('x_max')
ax.set_title('f_max = 5.50e5')

# %%
fig, ax = plt.subplots()

im = ax.pcolormesh(Vs_, x_, X3_mod[3], shading='nearest')
fig.colorbar(im, ax=ax, label="Ave Mech Power")
ax.set_xlabel('Vs_max')
ax.set_ylabel('x_max')
ax.set_title('f_max = 7.75e5')

# %%
fig, ax = plt.subplots()

im = ax.pcolormesh(Vs_, x_, X3_mod[4], shading='nearest')
fig.colorbar(im, ax=ax, label="Ave Mech Power")
ax.set_title('f_max = 1e6')
ax.set_xlabel('Vs_max')
ax.set_ylabel('x_max')


plt.show()

# %%
print(np.linspace(0.1, 5, 20))


# %%
