import numpy as np
import matplotlib.pyplot as plt

def classifier(x_train, y_train, variant):

    change = True  # Змінна для перевірки змін у вагах
    n_train = len(x_train)  # Розмір навчального набору даних
    w = [0, -1]  # Початкове значення вектора ваги
    a = lambda x: np.sign(x[0] * w[0] + x[1] * w[1])  # Правило класифікації
    L = 0.1  # Крок зміни ваги
    e = 0.1  # Невелика додаткова величина до w0, щоб забезпечити зазор між лінією розділення та областю
    count = 0  # Лічильник ітерацій
    last_error_index = -1  # Індекс останньої помилково класифікованої спостереження
    while change and count < 100:
        change = False
        for i in range(n_train):  # Ітерація по спостереженням
            if y_train[i] * a(x_train[i]) < 0:  # Якщо помилка класифікації,
                w[0] = w[0] + L * y_train[i]  # То коригування ваги w0
                last_error_index = i
                change = True

        Q = sum([1 for i in range(n_train) if y_train[i] * a(x_train[i]) < 0])
        if Q == 0:  # Показник якості класифікації (кількість помилок)
            break  # Зупинка, якщо всі класифікуються правильно
        count += 1
    if last_error_index > -1:
        w[0] = w[0] + e * y_train[last_error_index]

    print(f"Варіант {variant}:")  # Друк варіанту
    print(w)

    line_x = list(range(max(x_train[:, 0])))  # Створення графіка роздільної лінії
    line_y = [w[0] * x for x in line_x]

    x_0 = x_train[y_train == 1]  # Формування точок для першого класу
    x_1 = x_train[y_train == -1]  # та другого класу

    plt.scatter(x_0[:, 0], x_0[:, 1], color='red')
    plt.scatter(x_1[:, 0], x_1[:, 1], color='blue')
    plt.plot(line_x, line_y, color='green')

    plt.xlim([0, 55])
    plt.ylim([0, 55])
    plt.ylabel("довжина")
    plt.xlabel("ширина")
    plt.grid(True)
    plt.show()

x_train_5 = np.array([[49, 50], [5, 29], [37, 41], [38, 38], [13, 5], [41, 27], [7, 32], [40, 18], [31, 18], [41, 31]])
y_train_5 = np.array([-1, -1, -1, -1, 1, 1, -1, -1, 1, -1])
variant = 5
classifier(x_train_5, y_train_5, variant)