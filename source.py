# %%
from functools import partial

import numpy as np
import scipy as sp
import sympy as sym
from matplotlib import pyplot as plt

import pend
from utils import (create_control_mat, create_theta, gen_plot,
                   init_solve_system, round_expr_fac, find_theta)


# %% [markdown]
#  # Выполнение заданий

# %%
sym.init_printing()

# %%
# Array to string
atos = partial(np.array_str, precision=3, suppress_small=True)
pprint = round_expr_fac(3)

# %% [markdown]
#  ## Ситнез регулятора

# %%
pprint(pend.A)

# %%
C = create_control_mat(pend.A, pend.b)

print(f"C with shape {C.shape} and rank {np.linalg.matrix_rank(C)} = ")
pprint(C)

# %%
eigs = np.linalg.eigvals(pend.A)
print("A eigvals = ")
pprint(eigs)

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
#  ### Naive $\theta$

# %%
sol_nonlinear = solver(pend.nonlinear_system, theta_naive)
sol_linear = solver(pend.linear_system, pend.A, pend.b, theta_naive)

fig = gen_plot(time, [("linear", sol_linear), ("nonlinear", sol_nonlinear)], r"Naive $\theta$")
plt.show()

# %% [markdown]
#  ### Real $\theta$

# %%
sol_nonlinear = solver(pend.nonlinear_system, theta_real)
sol_linear = solver(pend.linear_system, pend.A, pend.b, theta_real)

fig = gen_plot(time, [("linear", sol_linear), ("nonlinear", sol_nonlinear)], r"Real $\theta$")
plt.show()

# %% [markdown]
#  ### Complex $\theta$

# %%
sol_nonlinear = solver(pend.nonlinear_system, theta_complex)
sol_linear = solver(pend.linear_system, pend.A, pend.b, theta_complex)

fig = gen_plot(time, [("linear", sol_linear), ("nonlinear", sol_nonlinear)], r"Complex $\theta$")
plt.show()

# %% [markdown]
#  # Синтез наблюдателя

# %%
O = create_control_mat(pend.A.T, pend.C.T)

print(f"O with shape {O.shape} and rank {np.linalg.matrix_rank(O)} = ")
pprint(O)

# %%
eigvals, eigs = sp.linalg.eig(pend.A.T, right=False, left=True)
P_inv = np.vstack([
    eigs[:, 0],
    eigs[:, 3],
    np.array([0, 1, 0, 0]),
    np.array([0, 0, 1, 0]),
])
P = np.linalg.inv(P_inv)

print(f"Eigvals = {atos(eigvals.real)}")
print()
print(f"P^-1 = \n{atos(P_inv)}")
print()
print(f"P = \n{atos(P)}")

# %%
A_hat = P_inv @ pend.A.T @ P
b_hat = P_inv @ pend.C.T

print(f"A_hat = \n{atos(A_hat)}")
print(f"b_hat = \n{atos(b_hat)}")

# %%
th_1 = sym.Symbol(r"theta_1")
th_2 = sym.Symbol(r"theta_2")

L_hat = np.array([
    [th_1, th_2, 0, 0],
    [th_1, th_2, 0, 0]
])

print(f"L_hat = ")
pprint(L_hat)

# %% [markdown]
#  $\hat A + \hat b \hat\theta = $

# %%
eq = A_hat + b_hat @ L_hat
pprint(eq)


# %%
A = np.array([
    [eq[0][0], eq[0][1]],
    [eq[1][0], eq[1][1]],
])

print("A = ")
pprint(A)

# %%
theta_real_L = find_theta(A, (-8, -2), th_1, th_2)
theta_real_L_non_asympt = find_theta(A, (-8, 0), th_1, th_2)
theta_complex_L = find_theta(A, (complex(-1, -1), complex(-1, 1)), th_1, th_2)

# %%
theta_hat_real = sym.Matrix(L_hat).subs([(th_1, theta_real_L[0]), (th_2, theta_real_L[1])])
theta_hat_real = np.array(theta_hat_real).astype(float)
L_real = (theta_hat_real @ P_inv).T

# %%
theta_hat_complex = sym.Matrix(L_hat).subs([(th_1, theta_complex_L[0]), (th_2, theta_complex_L[1])])
theta_hat_complex = np.array(theta_hat_complex).astype(complex)
L_complex = (theta_hat_complex @ P_inv).T

# %%
theta_hat_non_asympt = sym.Matrix(L_hat).subs([(th_1, theta_real_L_non_asympt[0]), (th_2, theta_real_L_non_asympt[1])])
theta_hat_non_asympt = np.array(theta_hat_non_asympt).astype(complex)
L_real_non_asympt= (theta_hat_non_asympt @ P_inv).T

# %%
pprint(np.linalg.eigvals(pend.A.T + pend.C.T @ L_real.T))

# %%
pprint(np.linalg.eigvals(pend.A.T + pend.C.T @ L_complex.T))

# %%
pprint(np.linalg.eigvals(pend.A.T + pend.C.T @ L_real_non_asympt.T))

# %%
solver, time = init_solve_system(np.array([0.1, 0, 0, 0, 0.7, 0, 0, 0]), stop=10)

# %%
sol_observer = solver(pend.system_with_observer, pend.A, pend.b, pend.C, theta_real, -L_real)

fig = gen_plot(time, [("State", sol_observer[:4]), ("Observer", sol_observer[4:])], r"Real $\theta$, real $L$")
plt.show()

# %%
sol_observer = solver(pend.system_with_observer, pend.A, pend.b, pend.C, theta_complex, -L_complex)

fig = gen_plot(time, [("State", sol_observer[:4]), ("Observer", sol_observer[4:])], r"Real $\theta$, real $L$")
plt.show()

# %%
sol_observer = solver(pend.system_with_observer, pend.A, pend.b, pend.C, theta_complex, -L_real_non_asympt)

fig = gen_plot(time, [("State", sol_observer[:4]), ("Observer", sol_observer[4:])], r"Real $\theta$, real $L$, non asympt")
plt.show()


