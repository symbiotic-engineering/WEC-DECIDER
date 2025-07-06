import numpy as np


def group_velocity(w, k, g, h):
    """
    Calculate group velocity for wave motion in water of depth h.

    Parameters:
    w : array_like
        Angular frequency (rad/s)
    k : array_like
        Wave number (rad/m)
    g : float
        Gravitational acceleration (m/s^2)
    h : float
        Water depth (m)

    Returns:
    V_g : ndarray
        Group velocity (m/s)
    mult : ndarray
        Multiplier term modifying deep-water formula
    """
    # w^2 / (g * k), equals 1 in deep water
    w2_over_gk = (w ** 2) / (g * k)

    # multiplier depends on depth; equals 1 in deep water
    mult = k * h * (1 - w2_over_gk ** 2) + w2_over_gk

    # group velocity: V_g = dω/dk = g / (2ω) * mult
    V_g = (g / (2 * w)) * mult

    return V_g, mult
