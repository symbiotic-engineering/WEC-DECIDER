#from optimization.find_nominal_inputs import *
import numpy as np
def var_bounds(mode: str = ''):
    b = {}

    b['var_names'] = ['D_s','D_f','T_f_2','h_s','h_fs_clear','F_max','P_max',
                      't_fb','t_sr','t_d','h_stiff_f','h1_stiff_d','M']
    b['var_names_pretty'] = {'D_s', 'D_f', 'T_{f,2}', 'h_s', 'h_{fs,clear}', 'F_{max}','P_{max}',
                             't_{fb}', 't_{sr}', 't_d', 'h_{stiff,f}', 'h_{1,stiff,d}', 'M'}
    b['var_descs'] = {'Spar diameter', 'Float diameter', 'Float draft', 'Spar height',
                   'Float-spar clearance height', 'Maximum force', 'Maximum power',
                   'Float bottom thickness', 'Spar radial thickness', 'Damping plate thickness',
                   'Float stiffener height', 'Damping plate stiffener height', 'Material index'}


    # diameter of spar (m)
    b['D_s_min'] = 0
    b['D_s_max'] = 30
    b['D_s_nom'] = 6
    b['D_s_wecsim'] = 6
    b['D_s_start'] = 6

    # outer diameter of float (m)
    b['D_f_min'] = 1
    b['D_f_max'] = 30
    b['D_f_wecsim'] = 20
    b['D_f_nom'] = 20
    b['D_f_start'] = 20

    # draft of float (m)
    b['T_f_2_min'] = 0.5
    b['T_f_2_max'] = 100
    b['T_f_2_nom'] = 3.2
    b['T_f_2_wecsim'] = 3
    b['T_f_2_start'] = 3

    # height of spar (m)
    b['h_s_min'] = 5
    b['h_s_max'] = 100
    b['h_s_nom'] = 44
    b['h_s_wecsim'] = 37.9
    b['h_s_start'] = 37.9

    # vertical clearance between float tube support and spar (m)
    b['h_fs_clear_min'] = 0.01
    b['h_fs_clear_max'] = 10
    b['h_fs_clear_wecsim'] = 4
    b['h_fs_clear_nom'] = 4 #p169 11/26/24
    b['h_fs_clear_start'] = 4

    # maximum powertrain force (MN)
    b['F_max_min'] = 0.01
    b['F_max_max'] = 100
    b['F_max_wecsim'] = 100
    b['F_max_nom'] = 1
    b['F_max_start'] = 5

    # maximum generator power (kW)
    b['P_max_min'] = 1
    b['P_max_max'] = 3000
    b['P_max_wecsim'] = 286
    b['P_max_nom'] = 286
    b['P_max_start'] = 286

    in2mm = 25.4
    # material thickness of float bottom (mm)
    b['t_fb_min'] = 0.05 * in2mm
    b['t_fb_max'] = 1.0 * in2mm
    b['t_fb_wecsim'] = 0.56 * in2mm
    b['t_fb_nom'] = 0.56 * in2mm
    b['t_fb_start'] = 0.56 * in2mm

    # material thickness of spar radial (mm)
    b['t_sr_min'] = 0.2 * in2mm
    b['t_sr_max'] = 2.0 * in2mm
    b['t_sr_wecsim'] = 1.0 * in2mm
    b['t_sr_nom'] = 1.0 * in2mm
    b['t_sr_start'] = 1.0 * in2mm

    # material thickness of damping plate (mm)
    b['t_d_min'] = 0.05 * in2mm
    b['t_d_max'] = 2.0 * in2mm
    b['t_d_wecsim'] = 1.0 * in2mm
    b['t_d_nom'] = 1.0 * in2mm
    b['t_d_start'] = 1.0 * in2mm

    in2m = in2mm / 1000
    # float stiffener height (m)
    b['h_stiff_f_min'] = 0
    b['h_stiff_f_max'] = 3
    b['h_stiff_f_nom'] = 16 * in2m
    b['h_stiff_f_wecsim'] = 16 * in2m
    b['h_stiff_f_start'] = 16 * in2m

    # damping plate stiffener height
    b['h1_stiff_d_min'] = 0
    b['h1_stiff_d_max'] = 2
    b['h1_stiff_d_nom'] = 22 * in2m
    b['h1_stiff_d_wecsim'] = 22 * in2m
    b['h1_stiff_d_start'] = 22 * in2m

    # material index (-)\
    b['M_min'] = 1
    b['M_max'] = 3
    b['M_wecsim'] = 1
    b['M_nom'] = 1
    b['M_start'] = 1

    b['mins_flexible'] = [False, True, True, True, False, True, True, [True, True, True], [False, False]]
    b['maxs_flexible'] = [True, True, False, False, True, True, True, [True, True, True], [False, False]]

    n_dv = len(b['var_names']) - 1

    X_mins = np.zeros((n_dv, 1))
    X_maxs = np.zeros((n_dv, 1))
    X_noms = np.zeros((n_dv, 1))
    X_noms_wecsim = np.zeros((n_dv, 1))
    X_starts = np.zeros((n_dv, 1))

    for i in range(n_dv):
        dv_name = b['var_names'][i]
        X_mins[i] = b[dv_name + '_min']
        X_maxs[i] = b[dv_name + '_max']
        X_noms[i] = b[dv_name + '_nom']
        X_noms_wecsim[i] = b[dv_name + '_wecsim' ]
        X_starts[i] = b[dv_name + '_start']



    b['X_mins'] = X_mins
    b['X_maxs'] = X_maxs

    if mode == 'wecsim':
        b['X_noms'] = X_noms_wecsim
    else:
        b['X_noms'] = X_noms

    b['X_starts'] = X_starts
    b['X_start_struct'] = dict(zip(b['var_names'][:-1], b['X_starts']))
    # constraints
    b['constraint_names'] = [
        'float_too_heavy', 'float_too_light', 'spar_too_heavy', 'spar_too_light',
        'stability', 'FOS_float_max', 'FOS_float_fatigue',
        'FOS_col_max', 'FOS_col_fatigue', 'FOS_plate_max', 'FOS_plate_fatigue',
        'FOS_col_local_max', 'FOS_col_local_fatigue',
        'pos_power', 'LCOE_max', 'irrelevant_max_force',
        'spar_height_up', 'spar_height_down', 'float_spar_hit',
        'linear_theory'
    ]

    i1 = len(b['constraint_names'])
    JPD_size = 14 * 15
    storm_size = 7

    b['constraint_names'].extend([''] * (JPD_size + storm_size + 1))
    for i in range(i1 + 1, i1 + JPD_size + storm_size + 1):
        b['constraint_names'][i] = f'prevent_slamming{i - i1}'

    b['constraint_names_pretty'] = [name.replace('_', ' ') for name in b['constraint_names']]
    b['lin_constraint_names'] = ['spar_natural_freq', 'float_spar_diam','float_spar_draft',
        'float_spar_tops', 'float_seafloor','spar_seafloor']

    b['lin_constraint_names_pretty'] = [name.replace('_', ' ') for name in b['lin_constraint_names']]

    # objective
    b['obj_names'] = {'LCOE', 'capex_design'}
    b['obj_names_pretty'] = {'LCOE', 'C_{design}'}

    # indices
    idxs_sort = np.argsort(b['var_names'][:-1])
    idxs_recover = np.zeros_like(idxs_sort)
    idxs_recover[idxs_sort] = np.arange(len(idxs_sort))  # indices to recover unsorted variabes from sorted ones
    b['idxs_sort'] = idxs_sort
    b['idxs_recover'] = idxs_recover

    # uuid
    b['filename_uuid'] = "" # string to append to generated filenames to prevent parallel overlap

    # calibrations of nominal values
    #b['F_max_nom'] = find_nominal_inputs(b, parameters(mode));
    b['F_max_nom'] = 7.0160
    index = b['var_names'].index('F_max')
    b['X_noms'][index] = b['F_max_nom']

    return b


var_bounds("")