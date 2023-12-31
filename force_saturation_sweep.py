import autograd.numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.io import savemat
from matplotlib.ticker import StrMethodFormatter as strformat
import xarray as xr
import datetime
import time
import sys
import timeit
import dask

# switch to using local WecOptTool instead of conda package
sys.path.insert(1,'C:/Users/rgm222/Documents/Github/SEA-Lab/WecOptTool')
import wecopttool as wot

def inner_function(zeta_u, w_u_star, f_max_Fp, m, w, F_h, amplitude, nfreq, 
                   nsubsteps=16, use_PI=False, plot_on=False, return_extras=False):
    wavefreq = w/(2*np.pi)
    ndof = 1
    w1 = w
    f1 = w1/(2*np.pi)
    ws = np.arange(w1,w1*(nfreq+1),w1)
    F_h = F_h * np.ones_like(ws)

    freqs = ws / (2 * np.pi) # Hz
    
    # create dimensional coeffs from nondimensional coeffs
    w_n_us = ws / w_u_star
    w_n_u = w / w_u_star
    B_hs = 2 * zeta_u * w_n_us * m
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
    X_unsat = F_h[0] / (m * w_n**2) / np.sqrt( (1 - w_star^2)**2 + (2*zeta*w_star)**2 )
    Fp = np.sqrt( (B_p * w)**2 + K_p**2) * X_unsat
    f_max = f_max_Fp * Fp if f_max_Fp is not None else None

    # create PTO
    name = ["PTO_Heave",]
    kinematics = np.eye(ndof)

    # toggle to switch between saturated PI and unstructured constrained
    # True takes longer but gives more power (at least for nfreq=5. The power 
    # might be comparable for higher nfreq)
    if use_PI:
        def controller(p,w,xw,xo,wa,ns):
                return wot.pto.controller_pid(p,w,xw,xo,wa,ns,derivative=False,saturation=f_max)
        constraints = None
        nstate_opt = 2
        x_opt_0 = np.array([-B_p[0], -K_p[0]])
    else:
        controller = None
        
        def const_f_pto(wec, x_wec, x_opt, waves): # Format for scipy.optimize.minimize
            f = pto.force_on_wec(wec, x_wec, x_opt, waves, nsubsteps)
            return f_max - np.abs(f.flatten())
        ineq_cons1 = {'type': 'ineq',
                      'fun': const_f_pto,
                      }
        constraints = [ineq_cons1] if f_max is not None else None
        
        nstate_opt = 2*nfreq
        
        Fp_phase = np.arctan(B_p*ws / K_p)
        real_part_Fp  = -Fp * np.cos(Fp_phase)
        imag_part_Fp  = -Fp * np.sin(Fp_phase)
        x_opt_0 = np.concatenate([real_part_Fp,imag_part_Fp])
        
    loss = None
    pto_impedance = None
    pto = wot.pto.PTO(ndof, kinematics, controller, pto_impedance, loss, name)
    
    f_add = {'PTO': pto.force_on_wec}
    
    # define impedance and reshape
    impedance = m * 1j * ws + B_hs + K_h/(1j * ws)
    impedance = np.reshape(impedance,(nfreq,ndof,ndof))
    K_h = np.reshape(K_h,(ndof,ndof))
    
    exc_coeff, impedance = make_xarrays(ws, F_h, impedance)

    wec = wot.WEC.from_impedance(
        freqs=freqs,
        impedance=impedance,
        exc_coeff=exc_coeff,
        hydrostatic_stiffness=K_h,
        constraints=constraints,
        f_add=f_add
    )
    
    phase = 30
    wavedir = 0
    waves = wot.waves.regular_wave(f1, nfreq, wavefreq, amplitude, phase, wavedir)
    wave = waves[:,:,0]
    
    #twave = timeit.timeit(lambda: dask.base.tokenize(wave), number=500)/500
    #texc =  timeit.timeit(lambda: dask.base.tokenize(exc_coeff), number=500)/500
    #print('time wave: ',twave)
    #print('time exc: ',texc)

    obj_fun = pto.mechanical_average_power
    
    options = {'maxiter': 200}
    scale_x_wec = 1
    scale_x_opt = 1
    scale_obj = 1

    # use the unsaturated solution as a guess
    # the vectorization on this might be bad (need vectorized B_p, Fp, and X_unsat?)
    dc_pos = np.array([0])
    real_part_pos = np.full_like(ws, X_unsat)
    imag_part_pos = np.zeros_like(real_part_pos)
    x_wec_0 = np.concatenate([dc_pos, real_part_pos, imag_part_pos[:-1]])
    
    results = wec.solve(
        waves,
        obj_fun,
        nstate_opt,
        optim_options=options,
        scale_x_wec=scale_x_wec,
        scale_x_opt=scale_x_opt,
        scale_obj=scale_obj,
        x_wec_0=x_wec_0,
        x_opt_0=x_opt_0,
        use_grad=False
    )
    
    if plot_on or return_extras:
        x_wec, x_opt = wec.decompose_state(results[0].x)
        r = wec.residual(x_wec, x_opt, wave)
        print('Residual: ',r)
        
        res_wec_fd, res_wec_td = wec.post_process(results[0],wave,nsubsteps=nsubsteps)
        res_pto_fd, res_pto_td = pto.post_process(wec,results[0],wave,nsubsteps=nsubsteps)
    
    if plot_on:
        plt.figure()
        res_wec_td.pos.plot()
        res_wec_td.vel.plot()
        res_wec_td.acc.plot()
        res_pto_td.force.plot()
        res_pto_td.power.sel(type='elec').plot.line('--',x='time')
        plt.legend([res_wec_td.pos.long_name, res_wec_td.vel.long_name, 
                    res_wec_td.acc.long_name, res_pto_td.force.long_name,
                    res_pto_td.power.long_name])

    avg_pwr = -results[0].fun
    print('Power: ',avg_pwr)
    
    if return_extras == False:
        out = avg_pwr
    else:
        max_x    = 1/2 * (np.max(res_wec_td.pos) - np.min(res_wec_td.pos))
        max_xdot = np.max(np.abs(res_wec_td.vel))
        out = avg_pwr, max_x, max_xdot
    return out

def sat_unsat_wrapper(zeta_u, w_u_star, f_max_Fp, m, w, F_h, amplitude, nfreq, 
                      nsubsteps, use_PI):
    # saturated run
    avg_pwr, max_x, max_xdot = inner_function(zeta_u, w_u_star, f_max_Fp, m, w, 
                                              F_h, amplitude, nfreq, nsubsteps, 
                                              use_PI, plot_on=False, return_extras=True)
        
    # rerun but unsaturated
    avg_pwr_unsat, max_x_unsat, max_xdot_unsat = inner_function(zeta_u, w_u_star, None, m, w, 
                                              F_h, amplitude, nfreq, nsubsteps, 
                                              use_PI, plot_on=False, return_extras=True)
                                        
    # ratios
    pwr_ratio = avg_pwr / avg_pwr_unsat
    x_ratio = max_x / max_x_unsat
    xdot_ratio = max_xdot / max_xdot_unsat

    return avg_pwr, max_x, max_xdot, pwr_ratio, x_ratio, xdot_ratio

def make_xarrays(ws, F_h, impedance):
    # make xarrays
    freq_attr = {'long_name': 'Wave frequency', 'units': 'rad/s'}
    dir_attr = {'long_name': 'Wave direction', 'units': 'rad'}
    dof_attr = {'long_name': 'Degree of freedom'}
    dof_names = ["Pitch",]

    directions = np.atleast_1d(0.0)

    dims_exc = ('omega', 'wave_direction', 'influenced_dof')
    coords_exc = [
        (dims_exc[0], ws, freq_attr),
        (dims_exc[1], directions, dir_attr),
        (dims_exc[2], dof_names, dof_attr),
    ]
    attrs_exc = {'units': 'N/m', 'long_name': 'Excitation Coefficient'}
    exc_coeff = np.expand_dims(F_h, axis = [1,2])
    exc_coeff = xr.DataArray(exc_coeff, dims=dims_exc, coords=coords_exc,
                            attrs=attrs_exc, name='excitation coefficient')

    dims_imp = ('omega', 'radiating_dof', 'influenced_dof')
    coords_imp = [
        (dims_imp[0], ws, freq_attr), 
        (dims_imp[1], dof_names, dof_attr),
        (dims_imp[2], dof_names, dof_attr),
    ]
    attrs_imp = {'units': 'Ns/m', 'long_name': 'Intrinsic Impedance'}
    impedance = xr.DataArray(impedance, dims=dims_imp, coords=coords_imp, 
                             attrs=attrs_imp, name='Intrisnic impedance')
    
    return exc_coeff, impedance

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

    nfreq = 8

    # preallocate output arrays
    avg_pwr    = np.zeros_like(zeta_u_mat)
    max_x      = np.zeros_like(zeta_u_mat)
    max_xdot   = np.zeros_like(zeta_u_mat)
    pwr_ratio  = np.zeros_like(zeta_u_mat)
    x_ratio    = np.zeros_like(zeta_u_mat)
    xdot_ratio = np.zeros_like(zeta_u_mat)
    
    # run sim
    t1 = time.time()
    for i in np.arange(zeta_u_mat.size):
        idx = np.unravel_index(i,zeta_u_mat.shape)
        #try:
        zeta_u = zeta_u_mat.ravel()[i]
        w_u_star = w_u_star_mat.ravel()[i]
        f_max_Fp = f_max_Fp_mat.ravel()[i]
        
        tuple_out = sat_unsat_wrapper(zeta_u, w_u_star, f_max_Fp, m, w, F_h, 
                                   amplitude=1, nfreq=nfreq, nsubsteps=1, 
                                   use_PI=False)
        
        avg_pwr[idx]    = tuple_out[0]
        max_x[idx]      = tuple_out[1]
        max_xdot[idx]   = tuple_out[2]
        pwr_ratio[idx]  = tuple_out[3]
        x_ratio[idx]    = tuple_out[4]
        xdot_ratio[idx] = tuple_out[5]
        
        #except:
        #    X[idx] = np.nan
    t2 = time.time()
    print('Time elapsed for ',zeta_u_mat.size, ' iterations: ',t2-t1,' s = ',(t2-t1)/zeta_u_mat.size,' s per iteration')

    timestamp = datetime.datetime.now().strftime('%G%m%d'+'_'+'%H%M%S')
    fname = 'wot_sweep_results_' + timestamp + '_N=' + str(nfreq) + '.csv'
    mdict = {"avg_pwr2": avg_pwr, "max_x2": max_x, "max_xdot2": max_xdot, 
             "pwr_ratio2":pwr_ratio, "x_ratio2":x_ratio, "xdot_ratio2":xdot_ratio}
    savemat(fname, mdict)

    # plot results
    plot_nondim_sweep(zeta_u_mat, w_u_star_mat, f_max_Fp_mat, avg_pwr,'Average Electrical Power (W)')
    plot_nondim_sweep(zeta_u_mat, w_u_star_mat, f_max_Fp_mat, max_x, 'Max Displacement of WEC (m)')
    plot_nondim_sweep(zeta_u_mat, w_u_star_mat, f_max_Fp_mat, max_xdot, 'Max Speed of WEC (m/s)')
    plot_nondim_sweep(zeta_u_mat, w_u_star_mat, f_max_Fp_mat, pwr_ratio,'Electrical Power Ratio (-)')
    plot_nondim_sweep(zeta_u_mat, w_u_star_mat, f_max_Fp_mat, x_ratio, 'Displacement Ratio (-)')
    plot_nondim_sweep(zeta_u_mat, w_u_star_mat, f_max_Fp_mat, xdot_ratio, 'Speed Ratio(-)')
    plt.show()

def plot_nondim_sweep(zeta_u_mat, w_u_star_mat, f_max_Fp_mat, Z, z_title):
    plt.figure()
    ax = plt.subplot(projection="3d")
    sc = ax.scatter(zeta_u_mat, w_u_star_mat, f_max_Fp_mat, c=Z, 
                    marker='o', s=25, cmap="viridis", depthshade=False)
    plt.colorbar(sc)
    ax.set_xlabel("zeta_u")
    ax.set_ylabel("w_u_star")
    ax.set_zlabel("F_max/F_p")
    ax.set_title(z_title)

def try_different_nfreqs():
    nfreqs = np.arange(2,15,2)
    nsubsteps = np.arange(1,7,2)
    use_PI = [True,False]
    X = np.zeros((nfreqs.size,nsubsteps.size,2))
    t = np.zeros_like(X)

    # dimensional coeffs
    m = np.array([1.0])
    w = np.array([1.0])
    F_h = np.array([1.0])

    # nondimensional coeffs
    zeta_u = 0.5
    w_u_star = 0.5
    f_max_Fp = 0.5

    for idx_PI in [0,1]:
        for idx_freq in np.arange(nfreqs.size):
            for idx_sub in np.arange(nsubsteps.size):
                try:
                    t1 = time.time()
                    X[idx_freq,idx_sub,idx_PI] = inner_function(zeta_u, w_u_star, f_max_Fp, 
                                                                 m, w, F_h, amplitude=1, 
                                                                 nfreq=nfreqs[idx_freq], 
                                                                 nsubsteps=nsubsteps[idx_sub], 
                                                                 use_PI=use_PI[idx_PI], plot_on=False)
                    t2 = time.time()
                    t[idx_freq,idx_sub,idx_PI] = t2 - t1
                except:
                    X[idx_freq,idx_sub,idx_PI] = np.nan
                    t[idx_freq,idx_sub,idx_PI] = np.nan
    
        plt.figure()
        plt.pcolormesh(nsubsteps,nfreqs,X[:,:,idx_PI])
        plt.xlabel('Number of substeps')
        plt.ylabel('Number of freqs')
        plt.title('Power (W) for use_PI='+str(use_PI[idx_PI]))
        plt.colorbar()
        
        plt.figure()
        plt.pcolormesh(nsubsteps,nfreqs,t[:,:,idx_PI])
        plt.xlabel('Number of substeps')
        plt.ylabel('Number of freqs')
        plt.title('Time (s) for use_PI='+str(use_PI[idx_PI]))
        plt.colorbar()
    print('power: ',X)
    print('time: ',t)
    
    idx_true = (-1,-1,1) # highest freq, highest substep, PI=false
    P_true = X[idx_true] 
    P_error = (X-P_true)/P_true * 100
    
    t_true = t[idx_true]
    t_rel = t / t_true

    blue_white_red_subplots(nsubsteps, nfreqs, P_error, 0, 'Power Error (%)', use_PI, idx_true, "{x:.1f}%")
    blue_white_red_subplots(nsubsteps, nfreqs, t_rel,   1, 'Relative Time (-)', use_PI, idx_true, "{x:.1f}")
    plt.show()

def blue_white_red_subplots(x, y, Z, center, title, use_PI, idx_true, valfmt):
    idx_no_nan = ~np.isnan(Z)
    
    fig, axs = plt.subplots(nrows=1, ncols=2)
    
    # green star where the ground truth value is
    axs[idx_true[-1]].plot( x[idx_true[0]], y[idx_true[1]], 'g*', markersize=50)
    
    zmin = np.nanmin(Z)
    zmax = np.nanmax(Z)
    driving = np.max(np.abs(np.array([zmin,zmax])-center)) # whichever of min/max is further from center
    norm = colors.TwoSlopeNorm(vmin=center-driving, vcenter=center, vmax=center+driving)
    txt_color_lims = [center-driving/2, center+driving/2]
    print('text lims: ',txt_color_lims)
    for idx_PI in [0,1]:
        h = axs[idx_PI].pcolormesh(x,y,Z[:,:,idx_PI], norm=norm, cmap='bwr')
        #h = axs[idx_PI].pcolorfast(x,y,Z[:,:,idx_PI], norm=norm, cmap='bwr')
        
        # red X where the nan's are
        X,Y = np.meshgrid(x.astype(float),y.astype(float))
        X[idx_no_nan[:,:,idx_PI]] = np.nan
        Y[idx_no_nan[:,:,idx_PI]] = np.nan
        axs[idx_PI].plot(X,Y,'rx')
        
        annotate_heatmap(h, x, y, txt_color_lims, valfmt=valfmt)
        
        axs[idx_PI].set_xlabel('Number of substeps')
        axs[idx_PI].set_ylabel('Number of freqs')
        axs[idx_PI].set_title('use_PI='+str(use_PI[idx_PI]))
    fig.colorbar(h, ax=axs.ravel().tolist())
    fig.suptitle(title)
    

# modified from https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
def annotate_heatmap(im, x, y, txt_color_lims, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    threshold_lo = im.norm(txt_color_lims[0])
    threshold_hi = im.norm(txt_color_lims[1])

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = strformat(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for iy,yi in np.ndenumerate(y):
        for ix,xi in np.ndenumerate(x):
            norm_data = im.norm(data[iy, ix])
            if ~np.isnan(norm_data.data):
                color_idx = int(norm_data > threshold_hi or norm_data < threshold_lo)
                kw.update(color=textcolors[color_idx])
                text = im.axes.text(xi, yi, valfmt(float(data[iy, ix][0]), None), **kw)
                texts.append(text)

    return texts

if __name__ == '__main__':
    sweep_nondim_coeffs()
    #try_different_nfreqs()
