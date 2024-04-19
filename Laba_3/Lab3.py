#Повторити приклад апроксимації поліномом для довільної функції, що відрізняється від наведеної у прикладі 1.

import numpy as np

# Функція для передбачення значень поліноміальної моделі
def predict_poly(x, koeff):
    res = 0
    # Обчислення значень для кожного степеня x та відповідного коефіцієнта
    xx = [x ** (len(koeff) - n - 1) for n in range(len(koeff))]

    # Обчислення значення полінома за формулою
    for i, k in enumerate(koeff):
        res += k * xx[i]

    return res

# Створення даних для x та y
x = np.arange(0, 10.1, 0.1)
y = 1 / (1 + 10 * np.square(x)) # обчислення значення функції

# Тренувальні дані x_train та y_train з вашого варіанту
x_train_5 = np.array([[42, 6], [39, 18], [25, 16], [27, 13], [24, 33], [8, 44], [30, 17], [7, 48], [25, 17]])
y_train_5 = np.array([1, 1, -1, -1, -1, -1, 1, -1, -1])

N = len(x_train_5)

# Підгонка полінома 10-го степеня до тренувальних даних
z_train = np.polyfit(x_train_5[:, 0], y_train_5, 10)
print(z_train)

#2.Повторити приклад апроксимації поліномом для довільної функції, що відрізняється від наведеної у прикладі.

import numpy as np
import matplotlib.pyplot as plt

# Дані
x_train_5 = np.array([[42, 6], [39, 18], [25, 16], [27, 13], [24, 33], [8, 44], [30, 17], [7, 48], [25, 17]])
y_train_5 = np.array([1, 1, -1, -1, -1, -1, 1, -1, -1])

# Параметри моделі
N = 13  # Степінь полінома (N-1)
L = 20  # Параметр регуляризації

# Створення матриці X для вихідних даних
X_train_5 = np.array([[a ** n for n in range(N)] for a in x_train_5[:, 0]])

# Створення матриці Y для вихідних даних
Y_train_5 = y_train_5

# Обчислення коефіцієнтів за формулою w = (XT*X + lambda*I)^-1 * XT * Y
A = np.linalg.inv(X_train_5.T @ X_train_5 + L * np.eye(N))
w = Y_train_5 @ X_train_5 @ A
print(w)

# Побудова прогнозів
x_range = np.linspace(0, 50, 1000)
X_range = np.array([[a ** n for n in range(N)] for a in x_range])
y_pred = X_range @ w

# Відображення графіків
plt.plot(x_range, y_pred, label='Прогноз моделі')
plt.scatter(x_train_5[:, 0], y_train_5, c='r', label='Точки даних')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Регуляризована апроксимація поліномом з L2-регуляризатором')
plt.legend()
plt.grid(True)
plt.show()

#3.Побудувати бінарний лінійний-класифікатор з L1-регуляризатором згідно з розглянутим прикладом 3 для даних свого варіанту, проаналізувати результати і вивести результати бінарної класифікації на графіку. Імпортувати дані для навчальної вибірки згідно з індивідуальним варіантом (імпорт організувати з файлу).

import numpy as np
import matplotlib.pyplot as plt

# Сигмоїдна функція втрат
def loss(w, x, y):
    M = np.dot(w, x) * y
    return 2 / (1 + np.exp(M))

# Похідна від сигмоїдальної функції втрат по вектору w
def df(w, x, y):
    L1 = 1.0  # Коефіцієнт L1-регуляризатора
    M = np.dot(w, x) * y
    return -2 * (1 + np.exp(M)) ** (-2) * np.exp(M) * x * y + L1 * np.sign(w)

# Дані
x_train_5 = np.array([[42, 6], [39, 18], [25, 16], [27, 13], [24, 33], [8, 44], [30, 17], [7, 48], [25, 17]])
x_train_5 = np.hstack((x_train_5, np.ones((len(x_train_5), 1))))  # Додаємо стовпець константи
y_train_5 = np.array([1, 1, -1, -1, -1, -1, 1, -1, -1])

fn = len(x_train_5[0])
n_train = len(x_train_5)  # Розмір навчальної вибірки
w = np.zeros(fn)           # Початкові вагові коефіцієнти
nt = 0.00001               # Крок збіжності SGD
lm = 0.01                  # Швидкість "забування" для Q
N = 5000                   # Кількість ітерацій SGD

Q = np.mean([loss(w, x, y) for x, y in zip(x_train_5, y_train_5)])  # Показник якості
Q_plot = [Q]

# Стохастичний алгоритм градієнтного спуску
for i in range(N):
    k = np.random.randint(0, n_train - 1)       # Випадковий індекс
    ek = loss(w, x_train_5[k], y_train_5[k])    # Визначення втрат для обраного вектора
    w = w - nt * df(w, x_train_5[k], y_train_5[k])  # Коригування вагів за допомогою SGD
    Q = lm * ek + (1 - lm) * Q                  # Перерахунок показника якості
    Q_plot.append(Q)

Q = np.mean([loss(w, x, y) for x, y in zip(x_train_5, y_train_5)]) # Справжнє значення емпіричного ризику після навчання
print("Вагові коефіцієнти:", w)
print("Показник якості:", Q)

# Відображення графіка показника якості
plt.plot(Q_plot)
plt.grid(True)
plt.xlabel('Ітерації')
plt.ylabel('Показник якості')
plt.title('Динаміка показника якості під час навчання')
plt.show()

#4.Модифікувати код, де замість L1-регуляризатора має використовуватись L2-регуляризатор.

import numpy as np
import matplotlib.pyplot as plt

# Сигмоїдна функція втрат
def loss(w, x, y):
    M = np.dot(w, x) * y
    return 2 / (1 + np.exp(M))

# Похідна від сигмоїдальної функції втрат по вектору w
def df(w, x, y):
    L2 = 0.01  # Коефіцієнт L2-регуляризатора
    M = np.dot(w, x) * y
    return -2 * (1 + np.exp(M)) ** (-2) * np.exp(M) * x * y + 2 * L2 * w

# Дані
x_train_5 = np.array([[42, 6], [39, 18], [25, 16], [27, 13], [24, 33], [8, 44], [30, 17], [7, 48], [25, 17]])
x_train_5 = np.hstack((x_train_5, np.ones((len(x_train_5), 1))))  # Додаємо стовпець константи
y_train_5 = np.array([1, 1, -1, -1, -1, -1, 1, -1, -1])

fn = len(x_train_5[0])
n_train = len(x_train_5)  # Розмір навчальної вибірки
w = np.zeros(fn)           # Початкові вагові коефіцієнти
nt = 0.00001               # Крок збіжності SGD
lm = 0.01                  # Швидкість "забування" для Q
N = 5000                   # Кількість ітерацій SGD

Q = np.mean([loss(w, x, y) for x, y in zip(x_train_5, y_train_5)])  # Показник якості
Q_plot = [Q]

# Стохастичний алгоритм градієнтного спуску
for i in range(N):
    k = np.random.randint(0, n_train - 1)       # Випадковий індекс
    ek = loss(w, x_train_5[k], y_train_5[k])    # Визначення втрат для обраного вектора
    w = w - nt * df(w, x_train_5[k], y_train_5[k])  # Коригування вагів за допомогою SGD
    Q = lm * ek + (1 - lm) * Q                  # Перерахунок показника якості
    Q_plot.append(Q)

Q = np.mean([loss(w, x, y) for x, y in zip(x_train_5, y_train_5)]) # Справжнє значення емпіричного ризику після навчання
print("Вагові коефіцієнти:", w)
print("Показник якості:", Q)

# Відображення графіка показника якості
plt.plot(Q_plot)
plt.grid(True)
plt.xlabel('Ітерації')
plt.ylabel('Показник якості')
plt.title('Динаміка показника якості під час навчання')
plt.show()