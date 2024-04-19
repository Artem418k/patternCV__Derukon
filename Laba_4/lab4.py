
#1. Реалізувати наївний байєсівський класифікатор для свого варіанта.

import numpy as np

# Дані
x_train_5 = np.array([[31, 13], [42, 27], [30, 28], [8, 26], [24, 35], [34, 21], [27, 21], [37, 47], [34, 21]])
y_train_5 = np.array([-1, 1, 1, 1, 1, -1, -1, -1, -1])

# Обчислення середніх значень та дисперсій для кожного класу
mw1_5, ml1_5 = np.mean(x_train_5[y_train_5 == 1], axis=0)
mw_1_5, ml_1_5 = np.mean(x_train_5[y_train_5 == -1], axis=0)

sw1_5, sl1_5 = np.var(x_train_5[y_train_5 == 1], axis=0)
sw_1_5, sl_1_5 = np.var(x_train_5[y_train_5 == -1], axis=0)

print('Середнє: ', mw1_5, ml1_5, mw_1_5, ml_1_5)
print('Дисперсії:', sw1_5, sl1_5, sw_1_5, sl_1_5)

# Точка для класифікації
x_5 = [36, 30]

# Функції для визначення ймовірності належності до кожного класу
a_1_5 = lambda x: -(x[0] - ml_1_5) ** 2 / (2 * sl_1_5) - (x[1] - mw_1_5) ** 2 / (2 * sw_1_5)  # Класифікатор першого класу
a1_5 = lambda x: -(x[0] - ml1_5) ** 2 / (2 * sl1_5) - (x[1] - mw1_5) ** 2 / (2 * sw1_5)  # Класифікатор другого класу

# Визначення класу для точки
y_5 = np.argmax([a_1_5(x_5), a1_5(x_5)])  # Обираємо максимум

# Вивід результату
print('Номер класу (-1 - перший клас, 1 - другий клас): ', y_5)

# 2.Реалізувати байєсівський класифікатор, попередньо змоделювавши дані згідно параметри кластерів у відповідності до свого варіанта.

import numpy as np
import matplotlib.pyplot as plt

# Вхідні параметри для першого кластеру
mu_x1, mu_y1 = 31, 13
sigma_x1_squared, sigma_y1_squared = 1.0, 1.0

# Вхідні параметри для другого кластеру
mu_x2, mu_y2 = 27, 23
sigma_x2_squared, sigma_y2_squared = 2.0, 2.0

# моделювання навчальної вибірки для кожного кластера
N = 1000
x1 = np.random.multivariate_normal([mu_x1, mu_y1], [[sigma_x1_squared, 0], [0, sigma_y1_squared]], N).T
x2 = np.random.multivariate_normal([mu_x2, mu_y2], [[sigma_x2_squared, 0], [0, sigma_y2_squared]], N).T

# обчислення оцінок середнього та коваріаційних матриць для кожного кластера
mm1 = np.mean(x1.T, axis=0)
mm2 = np.mean(x2.T, axis=0)

VV1 = np.cov(x1)
VV2 = np.cov(x2)

# модель байєсівського класифікатора
Py1, L1 = 0.5, 1  # ймовірності та штрафи
Py2, L2 = 1 - Py1, 1

b = lambda x, v, m, l, py: np.log(l * py) - 0.5 * (x - m) @ np.linalg.inv(v) @ (x - m).T - 0.5 * np.log(
    np.linalg.det(v))

x = np.array([36, 24])  # вхідний вектор
a = np.argmax([b(x, VV1, mm1, L1, Py1), b(x, VV2, mm2, L2, Py2)])  # класифікація
print(a)

# виведення графіків
plt.figure(figsize=(4, 4))
plt.title(f"Класифікація: {a}")
plt.scatter(x1[0], x1[1], s=10)
plt.scatter(x2[0], x2[1], s=10)
plt.scatter(x[0], x[1], color='red', marker='x', s=100)
plt.show()

#2.1 Змінити знак коефіцієнта кореляції одного з кластерів. Повторити експеримент з класифікації. .


import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

# Вхідні параметри для першого кластеру
rho1 = 0.8
sigma_x1_squared = 1.0
mu_x1 = 31
sigma_y1_squared = 1.0
mu_y1 = 13

# Вхідні параметри для другого кластеру
rho2 = -0.7
sigma_x2_squared = 2.0
mu_x2 = 27
sigma_y2_squared = 2.0
mu_y2 = 21

# моделювання навчальної вибірки для кожного кластеру
N = 1000
x1 = np.random.multivariate_normal([mu_x1, mu_y1], [[sigma_x1_squared, rho1], [rho1, sigma_y1_squared]], N).T
x2 = np.random.multivariate_normal([mu_x2, mu_y2], [[sigma_x2_squared, rho2], [rho2, sigma_y2_squared]], N).T

# обчислення оцінок середнього та коваріаційних матриць для кожного кластеру
mm1 = np.mean(x1.T, axis=0)
mm2 = np.mean(x2.T, axis=0)

a = (x1.T - mm1).T
VV1 = np.array([[np.dot(a[0], a[0]) / N, np.dot(a[0], a[1]) / N],
                [np.dot(a[1], a[0]) / N, np.dot(a[1], a[1]) / N]])

a = (x2.T - mm2).T
VV2 = np.array([[np.dot(a[0], a[0]) / N, np.dot(a[0], a[1]) / N],
                [np.dot(a[1], a[0]) / N, np.dot(a[1], a[1]) / N]])

# модель гауссівського баєсівського класифікатора
Py1, L1 = 0.5, 1  # ймовірності появи класів
Py2, L2 = 1 - Py1, 1  # та величини штрафів невірної класифікації

b = lambda x, v, m, l, py: np.log(l * py) - 0.5 * (x - m) @ np.linalg.inv(v) @ (x - m).T - 0.5 * np.log(
    np.linalg.det(v))

x = np.array([10, 20])  # вхідний вектор у форматі (x, y)
a = np.argmax([b(x, VV1, mm1, L1, Py1), b(x, VV2, mm2, L2, Py2)])  # класифікатор
print(a)

# виведення графіків
plt.figure(figsize=(4, 4))
plt.title(f"Кореляції: rho1 = {rho1}, rho2 = {rho2}")
plt.scatter(x1[0], x1[1], s=10)
plt.scatter(x2[0], x2[1], s=10)
plt.show()

#2.2 Модифікувати код на випадок трьох кластерів. Змоделювати відповідні дані і візуалізувати результати. ПРодемонструвати працездатність класифікатора..

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

# Вхідні параметри для першого кластеру
rho1 = 0.8
sigma_x1_squared = 1.0
mu_x1 = [0, -3]
sigma_y1_squared = 1.0
mu_y1 = [0, -3]

# Вхідні параметри для другого кластеру
rho2 = 0.7
sigma_x2_squared = 2.0
mu_x2 = [0, 3]
sigma_y2_squared = 2.0
mu_y2 = [0, 3]

# Вхідні параметри для третього кластеру
rho3 = -0.5
sigma_x3_squared = 1.5
mu_x3 = [-4, 0]
sigma_y3_squared = 1.5
mu_y3 = [-4, 0]

# моделювання навчальної вибірки для кожного кластеру
N = 1000
x1 = np.random.multivariate_normal(mu_x1, [[sigma_x1_squared, rho1], [rho1, sigma_y1_squared]], N).T
x2 = np.random.multivariate_normal(mu_x2, [[sigma_x2_squared, rho2], [rho2, sigma_y2_squared]], N).T
x3 = np.random.multivariate_normal(mu_x3, [[sigma_x3_squared, rho3], [rho3, sigma_y3_squared]], N).T

# обчислення оцінок середнього та коваріаційних матриць для кожного кластеру
mm1 = np.mean(x1.T, axis=0)
mm2 = np.mean(x2.T, axis=0)
mm3 = np.mean(x3.T, axis=0)

a = (x1.T - mm1).T
VV1 = np.array([[np.dot(a[0], a[0]) / N, np.dot(a[0], a[1]) / N],
                [np.dot(a[1], a[0]) / N, np.dot(a[1], a[1]) / N]])

a = (x2.T - mm2).T
VV2 = np.array([[np.dot(a[0], a[0]) / N, np.dot(a[0], a[1]) / N],
                [np.dot(a[1], a[0]) / N, np.dot(a[1], a[1]) / N]])

a = (x3.T - mm3).T
VV3 = np.array([[np.dot(a[0], a[0]) / N, np.dot(a[0], a[1]) / N],
                [np.dot(a[1], a[0]) / N, np.dot(a[1], a[1]) / N]])

# модель гауссівського баєсівського класифікатора
Py1, L1 = 1/3, 1  # ймовірності появи класів
Py2, L2 = 1/3, 1  # та величини штрафів невірної класифікації
Py3, L3 = 1/3, 1  # (за замовчуванням)

b = lambda x, v, m, l, py: np.log(l * py) - 0.5 * (x - m) @ np.linalg.inv(v) @ (x - m).T - 0.5 * np.log(
    np.linalg.det(v))

x = np.array([10, 20])  # вхідний вектор у форматі (x, y)
a = np.argmax([b(x, VV1, mm1, L1, Py1), b(x, VV2, mm2, L2, Py2), b(x, VV3, mm3, L3, Py3)])  # класифікатор
print(a)

# виведення графіків
plt.figure(figsize=(6, 6))
plt.title(f"Кореляції: rho1 = {rho1}, rho2 = {rho2}, rho3 = {rho3}")
plt.scatter(x1[0], x1[1], s=10)
plt.scatter(x2[0], x2[1], s=10)
plt.scatter(x3[0], x3[1], s=10)
plt.show()