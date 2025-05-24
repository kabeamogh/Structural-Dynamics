from typing import List, Tuple
import numpy as np
from scipy.integrate import odeint
from scipy import signal as signal

"""Module to perform integration of differential equations to determine
    coordinates for graphs and animation.
"""


class IntSolverAufgabe4:
    def __init__(self) -> None:
        return None

    def calculate(
        self, x: Tuple[float, float], t: List[float], c: float, d: float, m: float, e: float, M: float, r: float
    ) -> Tuple[float, float]:

        x_1, x_2 = x[0], x[1]

        x_dot = (x_2, ((M * r * e^2)/m) * cos(et) - d / m * x_2 - c / m * x_1)
        return x_dot

    def integrate(
        self,
        func,
        start_deflection: float,
        start_velocity: float,
        t: List[float],
        *args,
    ):

        y_0 = (start_deflection, start_velocity)
        x = odeint(func=func, y0=y_0, t=t, args=args)[:, 0]

        return x
