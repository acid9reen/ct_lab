from functools import reduce
from operator import matmul
from typing import Any, Callable

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp


def _create_char_pol(*roots: complex) -> Callable[[np.ndarray], np.ndarray]:
    """
    Create characteristic polynomial from roots dynamically
    Return function that look like x -> (x - root[0]) @ (x - root[1]) @ ...
    (Subtraction is not element-wise, it's like in true math!)
    """
    dim = len(roots)

    char_pol = lambda x: (
        reduce(matmul, [x - np.identity(dim) * root for root in roots])
    )

    return char_pol

def create_control_mat(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.column_stack([np.linalg.matrix_power(A, i) @ b for i in range(len(A))])


def create_theta(A: np.ndarray, C: np.ndarray, *roots: complex) -> np.ndarray:
    size = len(A)
    theta = (
        -np.eye(1, size, size - 1)
        @ np.linalg.inv(C)
        @ _create_char_pol(*roots)(A)
    )

    return np.real(theta)


def init_solve_system(
        initial_cond: np.ndarray,
        start: float = 0,
        stop: float = 10,
        num_points: int = 300,
) -> tuple[Callable, np.ndarray]:
    time = np.linspace(start, stop, num_points)

    def _solve_system(
            system: Callable,
            *system_args: Any
    ) -> np.ndarray:
        sol = solve_ivp(
            system,
            (start, stop),
            initial_cond,
            dense_output=True,
            args=system_args,
            method="RK45",
        )

        return sol.sol(time)
    return _solve_system, time


def gen_plot(
        time: np.ndarray,
        sol_linear: np.ndarray,
        sol_nonlinear: np.ndarray,
        title: str,
        size: tuple[float, float] = (10, 15),
        dpi: int = 96,
):
    y_labels = (r"$x$", r"$\varphi$", r"$\dot x$", r"$\dot \varphi$")

    fig, axs = plt.subplots(4, 1, sharex=True)
    fig.set_size_inches(*size)
    fig.suptitle(title)
    fig.set_dpi(dpi)

    for i in range(4):
        axs[i].plot(time, sol_linear[i], label="linear")
        axs[i].plot(time, sol_nonlinear[i], label="nonlinear")
        axs[i].set_xlabel("time")
        axs[i].set_ylabel(y_labels[i])
        axs[i].grid(True)
        axs[i].legend()

    fig.tight_layout()

    return fig
