# %%
from functools import partial

import numpy as np
import sympy as sym
from matplotlib import pyplot as plt
from scipy import integrate

import pend
from utils import (_create_char_pol, create_control_mat, create_theta,
                   gen_plot, init_solve_system)

# %% [markdown]
# # Выполнение заданий
# %%
# Array to string
atos = partial(np.array_str, precision=3, suppress_small=True)
# %% [markdown]
# ## Ситнез регулятора
# %%
print(atos(pend.A))
# %%
C = create_control_mat(pend.A, pend.b)

print(f"C = \n{atos(C)}")
print(f"Shape: {C.shape}")
print(f"Rank: {np.linalg.matrix_rank(C)}")
# %%
print(f"Eigs: {atos(np.linalg.eigvals(pend.A))}")

eigs = np.linalg.eigvals(pend.A)
eigs = list(map(float, eigs))
# %%
# перенесём 6.597 в устойчивое -6.597
theta_naive = create_theta(pend.A, C, eigs[0], -eigs[1], eigs[2], eigs[3])
print(f"При переносе СЧ (6.597) в действительное (-6.597): theta = {theta_naive}")

# перенесём 0 и 6.597 в устойчивые -2.069 и -6.597
theta_real = create_theta(pend.A, C, -2.069, -eigs[1], eigs[2], eigs[3])

print(f"При переносе СЧ (0, 6.597) в действительные (-2.069, -6.597): theta = {theta_real}")

# Перенесём СЧ 0 и 6.597 в пару комплексно сопряженных чисел -1-i, -1+i
theta_complex = create_theta(pend.A, C, complex(-1, -1), complex(-1, 1), eigs[2], eigs[3])

print(f"При переносе СЧ (0, 6.597) пару комплексно сопряженных чисел (-1-i, -1+i): theta = {theta_complex}")
# %%
print("Проверка СЧ полкченных после пременения управления:\n")

print(atos(np.linalg.eigvals(pend.A + pend.b @ theta_naive)))

print(atos(np.linalg.eigvals(pend.A + pend.b @ theta_real)))
print(atos(np.linalg.eigvals(pend.A + pend.b @ theta_complex)))
# %%
solver, time = init_solve_system(np.array([0, 0.1, 0, 0]), stop=5)
# %% [markdown]
# ### Naive $\theta$
# %%
sol_nonlinear = solver(pend.nonlinear_system, theta_naive)
sol_linear = solver(pend.linear_system, pend.A, pend.b, theta_naive)

fig = gen_plot(time, sol_linear, sol_nonlinear, r"Naive $\theta$")
plt.show()
# %% [markdown]
# ### Real $\theta$
# %%
sol_nonlinear = solver(pend.nonlinear_system, theta_real)
sol_linear = solver(pend.linear_system, pend.A, pend.b, theta_real)

fig = gen_plot(time, sol_linear, sol_nonlinear, r"Real $\theta$")
plt.show()
# %% [markdown]
# ### Complex $\theta$
# %%
sol_nonlinear = solver(pend.nonlinear_system, theta_complex)
sol_linear = solver(pend.linear_system, pend.A, pend.b, theta_complex)

fig = gen_plot(time, sol_linear, sol_nonlinear, r"Complex $\theta$")
plt.show()
# %% [markdown]
# ## Синтез наблюдателя
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
    @ _create_char_pol(-2.069, -eigs[1], eigs[2], eigs[3])(pend.A.T)
).T

print(f"При переносе СЧ в пару вещественных: L =\n {L_real.astype(float)}\n")

# Перенесём СЧ 0 и 6.597 в пару комплексно сопряженных чисел -1-i, -1+i
L_im = -(
    -np.array([[0, 0, 0, 1]])
    @ np.linalg.inv(C_observe)
    @ _create_char_pol(complex(-1, -1), complex(-1, 1), eigs[2], eigs[3])(pend.A.T)
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

sol = sol.sol(time)

y_labels = (r"x", r"\phi", r"\dot x", r"\dot \phi")

fig, axs = plt.subplots(4, 1)
fig.set_size_inches(10, 15)

for i in range(4):
    axs[i].plot(time, sol[i], label="x")
    axs[i].plot(time, sol[i + 4], label="ksi")
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

sol = sol.sol(time)

y_labels = (r"x", r"\phi", r"\dot x", r"\dot \phi")

fig, axs = plt.subplots(4, 1)
fig.set_size_inches(10, 15)

for i in range(4):
    axs[i].plot(time, sol[i], label="x")
    axs[i].plot(time, sol[i + 4], label="ksi")
    axs[i].set_xlabel("time")
    axs[i].set_ylabel(y_labels[i])
    axs[i].grid(True)
    axs[i].legend()

fig.tight_layout()
# fig.savefig('out.png', dpi=300, facecolor='white') # uncomment to save high-res picture
plt.show()
