import numpy as np

PEND_MASS = 0.127
CART_MASS = 1.206
MOMENT_OF_INERTIA = 0.001
LENGTH = 0.178
K_F = 1.726
K_S = 4.487
B_C = 5.4
B_P = 0.002


A_0 = np.array([
    [PEND_MASS + CART_MASS, -PEND_MASS * LENGTH],
    [-PEND_MASS * LENGTH, MOMENT_OF_INERTIA + PEND_MASS * LENGTH * LENGTH],
])
A_1 = np.array([[B_C, 0], [0, B_P]])
A_2 = np.array([[0, 0], [0, -PEND_MASS * 9.8 * LENGTH]])

A = np.block([
    [np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])],
    [-np.linalg.inv(A_0) @ A_2, -np.linalg.inv(A_0) @ A_1]
])

b = np.block([
    [np.array([[0], [0]])],
    [np.linalg.inv(A_0) @ np.array([[1], [0]])]
])

C = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])


def linear_system(
        t: np.ndarray,
        x: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        theta: np.ndarray,
) -> np.ndarray:
    return A @ x + b @ theta @ x


def nonlinear_system(
        t: np.ndarray,
        x: np.ndarray,
        theta: np.ndarray,
) -> np.ndarray:
    _, phi, xi_dot, phi_dot = x

    A_0 = np.array([
        [CART_MASS + PEND_MASS, -PEND_MASS * LENGTH * np.cos(phi)],
        [-PEND_MASS * LENGTH * np.cos(phi), MOMENT_OF_INERTIA + PEND_MASS * np.power(LENGTH, 2)],
    ])

    F = theta @ x

    b = np.array([
        [F.item() - B_C * xi_dot - PEND_MASS * LENGTH * np.power(phi_dot, 2) * np.sin(phi)],
        [-B_P * phi_dot + PEND_MASS * 10 * LENGTH * np.sin(phi)],
    ])

    y = np.linalg.inv(A_0) @ b
    res = np.vstack((np.array([[xi_dot], [phi_dot]]), y)).ravel()

    return res
