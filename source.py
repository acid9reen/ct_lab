# %%
from functools import reduce
from operator import matmul
from typing import Callable, Optional

import numpy as np
import sympy as sym
from matplotlib import pyplot as plt
from scipy import integrate

# import warnings
# warnings.filterwarnings("ignore")


# %%
PEND_MASS = 0.127
CART_MASS = 1.206
MOMENT_OF_INERTIA = 0.001
LENGTH = 0.178
K_F = 1.726
K_S = 4.487
B_C = 5.4
B_P = 0.002

# %% [markdown]
# # Функции и классы

# %%
def create_char_pol(*roots: complex) -> Callable[[np.ndarray], np.ndarray]:
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


# %%
class PendODESystem:
    def __init__(
        self,
        pend_mass,
        cart_mass,
        moment_of_inertia,
        length,
        K_f,
        K_s,
        B_c,
        B_p,
    ) -> None:

        A_0 = np.array(
            [
                [pend_mass + cart_mass, -pend_mass * length],
                [-pend_mass * length, moment_of_inertia + pend_mass * length * length],
            ]
        )

        A_1 = np.array([[B_c, 0], [0, B_p]])

        A_2 = np.array([[0, 0], [0, -pend_mass * 9.8 * length]])

        first = -np.linalg.inv(A_0) @ A_2
        second = -np.linalg.inv(A_0) @ A_1
        third = np.linalg.inv(A_0) @ np.array([[1], [0]])

        self._A = np.array(
            [
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [first[0][0], first[0][1], second[0][0], second[0][1]],
                [first[1][0], first[1][1], second[1][0], second[1][1]],
            ],
            dtype=np.complex_,
        )

        self._b = np.array([[0], [0], third[0], third[1]], dtype=np.complex_)

    @property
    def A(self):
        return self._A

    @property
    def b(self):
        return self._b


# %% [markdown]
# # Выполнение заданий

# %% [markdown]
# ## Ситнез регулятора

# %%
pend = PendODESystem(
    PEND_MASS,
    CART_MASS,
    MOMENT_OF_INERTIA,
    LENGTH,
    K_F,
    K_S,
    B_C,
    B_P,
)

# %%
print(np.array_str(pend.A.astype(float), precision=3, suppress_small=True))

# %%
C = np.column_stack(
    [
        pend.b,
        pend.A @ pend.b,
        np.linalg.matrix_power(pend.A, 2) @ pend.b,
        np.linalg.matrix_power(pend.A, 3) @ pend.b,
    ]
)

print(f"C = \n{np.array_str(C.astype(float), precision=3, suppress_small=True)}")
print(f"Shape: {C.shape}")
print(f"Rank: {np.linalg.matrix_rank(C)}")

# %%
print(
    f"Eigs: {np.array_str(np.linalg.eigvals(pend.A).astype(float), precision=3, suppress_small=True)}"
)

eigs = np.linalg.eigvals(pend.A)
eigs = list(map(float, eigs))

# %%
# перенесём 6.597 в устойчивое -6.597
theta_naive = (
    -np.array([[0, 0, 0, 1]])
    @ np.linalg.inv(C)
    @ create_char_pol(eigs[0], -eigs[1], eigs[2], eigs[3])(pend.A)
)

print(
    f"При переносе СЧ (6.597) в действительное (-6.597): theta = {theta_naive.astype(float)}\n"
)

# перенесём 0 и 6.597 в устойчивые -2.069 и -6.597
theta_real = (
    -np.array([[0, 0, 0, 1]])
    @ np.linalg.inv(C)
    @ create_char_pol(-2.069, -eigs[1], eigs[2], eigs[3])(pend.A)
)

print(
    f"При переносе СЧ (0, 6.597) в действительные (-2.069, -6.597): theta = {theta_real.astype(float)}\n"
)

# Перенесём СЧ 0 и 6.597 в пару комплексно сопряженных чисел -1-i, -1+i
theta_complex = (
    -np.array([[0, 0, 0, 1]])
    @ np.linalg.inv(C)
    @ create_char_pol(complex(-1, -1), complex(-1, 1), eigs[2], eigs[3])(pend.A)
)

print(
    f"При переносе СЧ (0, 6.597) пару комплексно сопряженных чисел (-1-i, -1+i): theta = {theta_complex.astype(float)}\n"
)


# %%
print("Проверка СЧ полкченных после пременения управления:\n")

print(
    np.array_str(
        np.linalg.eigvals(pend.A + pend.b @ theta_naive).astype(float), precision=3
    )
)

print(
    np.array_str(
        np.linalg.eigvals(pend.A + pend.b @ theta_real).astype(float), precision=3
    )
)

print(
    np.array_str(
        np.linalg.eigvals(pend.A + pend.b @ theta_complex),
        precision=3,
        suppress_small=True,
    )
)

# %%
def linear_system(
    t: np.ndarray,
    x: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    theta: np.ndarray,
    x_0: Optional[np.ndarray] = None,
) -> np.ndarray:
    if x_0 is None:
        x_0 = x

    return A @ x + b @ theta @ x_0


# %%
def nonlinear_system(
    t: np.ndarray,
    x: np.ndarray,
    theta: np.ndarray,
) -> np.ndarray:
    # Распаковываем вектор состояний
    xi, phi, xi_dot, phi_dot = x

    A_0 = np.array(
        [
            [CART_MASS + PEND_MASS, -PEND_MASS * LENGTH * np.cos(phi)],
            [
                -PEND_MASS * LENGTH * np.cos(phi),
                MOMENT_OF_INERTIA + PEND_MASS * np.power(LENGTH, 2),
            ],
        ]
    )

    F = theta @ x

    b = np.array(
        [
            [
                F
                - B_C * xi_dot
                - PEND_MASS * LENGTH * np.power(phi_dot, 2) * np.sin(phi)
            ],
            [-B_P * phi_dot + PEND_MASS * 10 * LENGTH * np.sin(phi)],
        ]
    )

    y = np.linalg.inv(A_0) @ b

    # Стакаем векторы, получаем матрицу из 4 переменных, раскатываем в одномерный вектор 1х4
    res = np.vstack((np.array([[xi_dot], [phi_dot]]), y)).ravel()

    return res


# %%
# start, stop = 0, 10

# time = np.linspace(start, stop, 300)
# y_0 = np.array([0, 0.1, 0, 0])

# sol = integrate.solve_ivp(
#     linear_system,
#     (start, stop),
#     y_0,
#     dense_output=True,
#     args=(pend.A, pend.b, theta_real),
#     method="RK45"
# )

start, stop = 0, 10

time = np.linspace(start, stop, 300)
y_0 = np.array([0, 0.1, 0, 0])

sol = integrate.solve_ivp(
    nonlinear_system,
    (start, stop),
    y_0,
    dense_output=True,
    args=(theta_real.astype(float),),
    method="RK45",
)

# %%
theta_real

# %%
np.array([[0, -42.1689, 10.8000, -6.0108]])

# %%
z = sol.sol(time)

# %%
y_labels = (r"x", r"\phi", r"\dot x", r"\dot \phi")
# plt.rcParams['text.usetex'] = True # uncomment if you have latex

fig, axs = plt.subplots(4, 1)
fig.set_size_inches(10, 15)

for i in range(4):
    axs[i].plot(time, z[i], label="linear")
    axs[i].set_xlabel("time")
    axs[i].set_ylabel(y_labels[i])
    axs[i].grid(True)
    axs[i].legend()

fig.tight_layout()
# fig.savefig('out.png', dpi=300, facecolor='white') # uncomment to save high-res picture
plt.show()


# %% [markdown]
# ## Синтез наблюдателя

# %%
pend.C = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

# %%
eigs, eigs_vectors = np.linalg.eig(pend.A.T.astype(float))
p_inverse = np.vstack((eigs_vectors[0], eigs_vectors[1], [1, 0, 0, 0], [0, 1, 0, 0]))

print(f"Rank: {np.linalg.matrix_rank(p_inverse)}")
print(np.array_str(p_inverse, precision=3, suppress_small=True))

# %%
p = np.linalg.inv(p_inverse)
A_hat = p_inverse @ pend.A.T.astype(float) @ p
c_hat = p_inverse @ pend.C.T

print(np.array_str(A_hat, precision=3, suppress_small=True))

print(np.array_str(c_hat, precision=3, suppress_small=True))

# %%
theta_1 = sym.Symbol("theta_1")
theta_2 = sym.Symbol("theta_2")

theta_for_L = np.array([[theta_1, theta_2, 0, 0], [theta_1, theta_2, 0, 0]])

p_inverse @ (pend.A.T.astype(float) @ p - pend.C.T @ theta_for_L)

# solution = sym.solve((x + 5 * y - 2, -3 * x + 6 * y - 15), (x, y))

# %%
C_observe = np.column_stack(
    [
        pend.C.T,
        pend.A.T @ pend.C.T,
        np.linalg.matrix_power(pend.A.T, 2) @ pend.C.T,
        np.linalg.matrix_power(pend.A.T, 3) @ pend.C.T,
    ]
)

print(C_observe.astype(float))
print(C_observe.shape)
print(f"Rank: {np.linalg.matrix_rank(C_observe)}")

# %%
print(f"Eigs: {np.linalg.eigvals(pend.A).astype(float)}")

eigs = np.linalg.eigvals(pend.A)
eigs = list(map(float, eigs))

# %%
# Перенесём СЧ 0 и 6.597 в пару вещественных -2.069 и -6.597
L_real = -(
    -np.array([[0, 0, 0, 1]])
    @ np.linalg.inv(C_observe)
    @ create_char_pol(-2.069, -eigs[1], eigs[2], eigs[3])(pend.A.T)
).T

print(f"При переносе СЧ в пару вещественных: L =\n {L_real.astype(float)}\n")

# Перенесём СЧ 0 и 6.597 в пару комплексно сопряженных чисел -1-i, -1+i
L_im = -(
    -np.array([[0, 0, 0, 1]])
    @ np.linalg.inv(C_observe)
    @ create_char_pol(complex(-1, -1), complex(-1, 1), eigs[2], eigs[3])(pend.A.T)
).T

print(f"При переносе СЧ в пару комплексно сопряженных: L =\n {L_im.astype(float)}")


# %% [markdown]
# ### Перенос СЧ в действительные

# %%
# TODO узнать является ли верхний правый блок нулевым

A_observe_real = np.block(
    [
        [pend.A, pend.b @ theta_real],
        [L_real @ pend.C, pend.A - L_real @ pend.C + pend.b @ theta_real],
    ]
).astype(float)

print(np.array_str(A_observe_real, precision=3, suppress_small=True))

# %%
def linear_system_observer_real(
    t: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:

    return A_observe_real @ x


start, stop = 0, 10

time = np.linspace(start, stop, 300)
y_0 = np.array([0, 0.1, 0, 0, 1, 0.1, 1, 0])

sol = integrate.solve_ivp(
    linear_system_observer_real,
    (start, stop),
    y_0,
    dense_output=True,
    args=(),
    method="RK45",
)

z = sol.sol(time)

y_labels = (r"x", r"\phi", r"\dot x", r"\dot \phi")

fig, axs = plt.subplots(4, 1)
fig.set_size_inches(10, 15)

for i in range(4):
    axs[i].plot(time, z[i], label="x")
    axs[i].plot(time, z[i + 4], label="ksi")
    axs[i].set_xlabel("time")
    axs[i].set_ylabel(y_labels[i])
    axs[i].grid(True)
    axs[i].legend()

fig.tight_layout()
# fig.savefig('out.png', dpi=300, facecolor='white') # uncomment to save high-res picture
plt.show()


# %% [markdown]
# ### Перенос СЧ в комплексные

# %%
A_observe_im = np.block(
    [
        [pend.A, pend.b @ theta_complex],
        [L_im @ pend.C, pend.A - L_im @ pend.C + pend.b @ theta_complex],
    ]
).astype(float)

print(np.array_str(A_observe_im, precision=3, suppress_small=True))

# %%
def linear_system_observer_im(
    t: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:

    return A_observe_im @ x


start, stop = 0, 10

time = np.linspace(start, stop, 300)
y_0 = np.array([0, 0.1, 0, 0, 1, 0.1, 1, 0])

sol = integrate.solve_ivp(
    linear_system_observer_im,
    (start, stop),
    y_0,
    dense_output=True,
    args=(),
    method="RK45",
)

z = sol.sol(time)

y_labels = (r"x", r"\phi", r"\dot x", r"\dot \phi")

fig, axs = plt.subplots(4, 1)
fig.set_size_inches(10, 15)

for i in range(4):
    axs[i].plot(time, z[i], label="x")
    axs[i].plot(time, z[i + 4], label="ksi")
    axs[i].set_xlabel("time")
    axs[i].set_ylabel(y_labels[i])
    axs[i].grid(True)
    axs[i].legend()

fig.tight_layout()
# fig.savefig('out.png', dpi=300, facecolor='white') # uncomment to save high-res picture
plt.show()
