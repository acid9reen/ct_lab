# %%
from functools import partial

import numpy as np
from matplotlib import pyplot as plt

import pend
from utils import create_control_mat, create_theta, gen_plot, init_solve_system

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
