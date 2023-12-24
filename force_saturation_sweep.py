import autograd.numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import wecopttool as wot

def inner_function(zeta_u, w_u_star, f_max_Fp, m, w, F_h, wavefreq, amplitude):

    ndof = 1
    nfreq = 1

    f1 = w / (2 * np.pi) # Hz
    
    # create dimensional coeffs from nondimensional coeffs
    w_n_u = w / w_u_star
    B_h = 2 * zeta_u * w_n_u * m
    K_h = w_n_u**2 * m
    # impedance matching, for w_star = 1 and r_b = 2
    K_p = m * w**2 - K_h
    B_p = B_h
    # max force
    w_star = 1
    r_b = 2
    w_n = w / w_star
    zeta = r_b / (r_b - 1) * w_star / w_u_star * zeta_u
    X_unsat = F_h / (m * w_n**2) / np.sqrt( (1 - w_star^2)**2 + (2*zeta*w_star)**2 )
    Fp = np.sqrt( (B_p * w)**2 + K_p**2) * X_unsat
    f_max = f_max_Fp * Fp

    # create PTO
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

    ineq_cons1 = {'type': 'ineq',
                 'fun': const_f_pto,
                 }
    constraints = [ineq_cons1]
    
    # define impedance and reshape
    impedance = m * 1j * w + B_h + K_h/(1j * w)
    print('impedance: ',impedance)
    impedance = np.reshape(impedance,(ndof,ndof,nfreq))
    #F_h = np.reshape(F_h,(ndof,nfreq))
    K_h = np.reshape(K_h,(ndof,ndof))
    
    # make xarrays
    freq_attr = {'long_name': 'Wave frequency', 'units': 'rad/s'}
    dir_attr = {'long_name': 'Wave direction', 'units': 'rad'}
    dof_attr = {'long_name': 'Degree of freedom'}
    dof_names = ["Pitch",]
    ndof = len(dof_names)
    directions = np.atleast_1d(0.0)

    dims_exc = ('omega', 'wave_direction', 'influenced_dof')
    coords_exc = [
        (dims_exc[0], w, freq_attr),
        (dims_exc[1], directions, dir_attr),
        (dims_exc[2], dof_names, dof_attr),
    ]
    attrs_exc = {'units': 'N/m', 'long_name': 'Excitation Coefficient'}
    exc_coeff = np.expand_dims(F_h, axis = [1,2])
    exc_coeff = xr.DataArray(exc_coeff, dims=dims_exc, coords=coords_exc,
                            attrs=attrs_exc, name='excitation coefficient')

    dims_imp = ('radiating_dof', 'influenced_dof', 'omega')
    coords_imp = [
        (dims_imp[0], dof_names, dof_attr),
        (dims_imp[1], dof_names, dof_attr),
        (dims_imp[2], w, freq_attr),
    ]
    attrs_imp = {'units': 'Ns/m', 'long_name': 'Intrinsic Impedance'}
    impedance = xr.DataArray(impedance, dims=dims_imp, coords=coords_imp, attrs=attrs_imp, name='Intrisnic impedance')

    wec = wot.WEC.from_impedance(
        freqs=w,
        impedance=impedance,
        exc_coeff=exc_coeff,
        hydrostatic_stiffness=K_h,
        constraints=constraints,
        f_add=f_add
    )
    
    phase = 30
    wavedir = 0
    waves = wot.waves.regular_wave(f1, nfreq, wavefreq, amplitude, phase, wavedir)

    obj_fun = pto.mechanical_average_power
    nstate_opt = 2*nfreq

    options = {'maxiter': 200}
    scale_x_wec = 1
    scale_x_opt = 1
    scale_obj = 1

    # use the unsaturated solution as a guess
    dc_pos = np.array([0])
    real_part_pos = X_unsat
    Fp_phase = np.arctan(B_p*w / K_p)
    real_part_Fp  = -Fp * np.cos(Fp_phase)
    imag_part_Fp  = -Fp * np.sin(Fp_phase)

    x_wec_0 = np.concatenate([dc_pos, real_part_pos])
    x_opt_0 = np.concatenate([real_part_Fp,imag_part_Fp])
    
    results = wec.solve(
        waves,
        obj_fun,
        nstate_opt,
        optim_options=options,
        scale_x_wec=scale_x_wec,
        scale_x_opt=scale_x_opt,
        scale_obj=scale_obj,
        x_wec_0=x_wec_0,
        x_opt_0=x_opt_0
    )
    
    
    x_wec, x_opt = wec.decompose_state(results.x)
    
    #print(x_wec)
    #print(len(x_wec))
    #print(x_opt)
    #print(len(x_opt))
    
    f = pto.force_on_wec(wec, x_wec, x_opt, waves, nsubsteps)
    p = pto.position(wec, x_wec, x_opt, waves, nsubsteps)
    v = pto.velocity(wec, x_wec, x_opt, waves, nsubsteps)
    #print(f)
    #print(p)
    #print(v)
    print('Power timeseries: ',f * v)
    print('Power: ',results.fun)

    return results.fun

def sweep_nondim_coeffs():
    # nondimensional coeffs
    zeta_u_vec = np.array([0.1,0.3,0.5])
    w_u_star_vec = np.array([0.5,0.6,0.7])
    f_max_Fp_vec = np.array([0.8,0.9,1.0])

    zeta_u_mat, w_u_star_mat, f_max_Fp_mat = np.meshgrid(zeta_u_vec, w_u_star_vec, f_max_Fp_vec)

    # dimensional coeffs
    m = np.array([1.0])
    w = np.array([1.0])
    F_h = np.array([1.0])

    # run sim
    X = np.zeros_like(zeta_u_mat)
    for i in np.arange(zeta_u_mat.size):
        idx = np.unravel_index(i,X.shape)
        try:
            X[idx] = inner_function(zeta_u_mat.ravel()[i], w_u_star_mat.ravel()[i], f_max_Fp_mat.ravel()[i], m, w, F_h, wavefreq = w/(2*np.pi), amplitude = 1)
        except:
            X[idx] = np.nan
            print(i,' is nan')

    print('X: ', X)

    # plot results
    ax = plt.subplot(projection="3d")
    sc = ax.scatter(zeta_u_mat, w_u_star_mat, f_max_Fp_mat, c=X, marker='o', s=25, cmap="viridis")
    plt.colorbar(sc)
    ax.set_xlabel("zeta_u")
    ax.set_ylabel("w_u_star")
    ax.set_zlabel("F_max/F_p")
    plt.show()

if __name__ == '__main__':
    sweep_nondim_coeffs()
