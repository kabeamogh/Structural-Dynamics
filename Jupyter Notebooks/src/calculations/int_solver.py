from typing import List, Tuple
import numpy as np
from scipy.integrate import odeint

"""Module to perform integration of differential equations for the vibrating table system."""

class IntSolverAufgabe1:
    def __init__(self):
        pass

    def state_space_settled(self, z: Tuple[float, float], t: float, d: float, m: float, c: float, e: float) -> List[float]:
        x, x_p = z  # x: displacement, x_p: velocity
        omega_0 = np.sqrt(c / m)
        delta = d / (2 * m)
        m_U = 0.012  # kg
        r = 0.01     # m
        return [x_p, -2 * delta * x_p - omega_0**2 * x + (m_U * r * e**2 / m) * np.cos(e * t)]

    def state_space_accelerated(self, z: Tuple[float, float], t: float, d: float, m: float, c: float) -> List[float]:
        x, x_p = z
        omega_0 = np.sqrt(c / m)
        delta = d / (2 * m)
        m_U = 0.012
        r = 0.01
        alpha = 5.0  # Fixed, not an input
        e_t = alpha * t
        return [x_p, -2 * delta * x_p - omega_0**2 * x + (m_U * r * e_t**2 / m) * np.cos(0.5 * alpha * t**2)]

    def integrate(self, func, t: List[float], start_deflection: float, start_velocity: float, *args) -> np.ndarray:
        z0 = (start_deflection, start_velocity)
        return odeint(func=func, y0=z0, t=t, args=args)
