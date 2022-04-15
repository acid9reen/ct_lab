from functools import reduce
from operator import matmul, mul
from typing import Any, Callable

import numpy as np
import sympy as sym
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
        label_and_xs: list[tuple[str, np.ndarray]],
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
        for label, xs in label_and_xs:
            axs[i].plot(time, xs[i], label=label)

        axs[i].set_xlabel("time")
        axs[i].set_ylabel(y_labels[i])
        axs[i].grid(True)
        axs[i].legend()

    fig.tight_layout()

    return fig


def round_expr_fac(num_digits):
    def _round_expr(array: np.ndarray):
        expr = sym.Matrix(array)
        return expr.xreplace({n : round(n, num_digits) for n in expr.atoms(sym.Number)})
    return _round_expr


def find_theta(
        A: np.ndarray,
        wanted_eigvals: tuple[complex, complex],
        th_1: sym.Symbol,
        th_2: sym.Symbol,
) -> np.ndarray:
    c = reduce(mul, wanted_eigvals)
    b = sum(wanted_eigvals)
    f_1 = -A[0][1].subs([(th_2, 1)]) * th_1 - th_2 - A[0][0].subs([(th_1, 0)]) + b
    f_2 = A[0][0].subs([(th_1, 0)]) * th_2 - c
    sol = sym.nsolve((f_1, f_2), (th_1, th_2), (0, 0))

    return sol
