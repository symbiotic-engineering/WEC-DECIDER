import numpy as np

def all_figures(which_figs=None, which_tabs=None):
    """
    Generate all figures for the WEC-DECIDER project.
    """
    num_figs = 10
    num_tabs = 7

    if which_figs is None:
        which_figs = np.arange(num_figs)
    if which_tabs is None:
        which_tabs = np.arange(num_tabs)
    
    if np.any(which_figs == 1):
        pass
        # created in powerpoint

    if np.any(which_figs == 2):
        pass
        # created in powerpoint

    if np.any(which_figs == 3 or which_figs == 4):
        pass
        # sin saturation demo

    if np.any(which_figs == 5):
        pass
        # todo
        # plot power matrix

    if np.any(which_figs == 6 | which_figs == 7):
        pass
        # todo
        # pareto search
        # pareto curve heuristics

    if np.any(which_figs == 8):
        pass
        # todo
        # param sweep

    if np.any(which_figs == 9 | which_figs == 10 | which_tabs == 5):
        pass
        # todo
        # compare

    if np.any(which_tabs == 1):
        pass
        # todo
        # design variables table

    if np.any(which_tabs == 2):
        pass
        # todo
        # constraints table

    if np.any(which_tabs == 3):
        pass
        # todo
        # parameters table

    if np.any(which_tabs == 4):
        pass
        # todo
        # validate nominal RM3