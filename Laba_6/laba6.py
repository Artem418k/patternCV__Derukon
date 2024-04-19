import numpy as np
import matplotlib.pyplot as plt

# Ваші дані
x_train_5 = np.array([[27, 46], [8, 22], [5, 30], [5, 19], [15, 23], [34, 42], [10, 23], [35, 32]])
y_train_5 = np.array([-1, -1, 1, -1, 1, 1, -1, -1])

# Тестовий набір даних
x_test_5 = np.array([[35, 10], [36, 39], [41, 9], [43, 19], [32, 42], [38, 48], [12, 39], [22, 45], [20, 29], [48, 21]])
y_test_5 = np.array([-1, -1, -1, 1, 1, 1, -1, 1, -1, 1])

# Функція для знаходження евклідового відстані між двома точками
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# Функція для знаходження k найближчих сусідів
def k_nearest_neighbors(x_train, y_train, x_test, k):
    y_pred = []
    for test_point in x_test:
        distances = [euclidean_distance(test_point, train_point) for train_point in x_train]
        k_nearest_indices = np.argsort(distances)[:k]
        k_nearest_labels = [y_train[i] for i in k_nearest_indices]
        most_common_label = max(set(k_nearest_labels), key=k_nearest_labels.count)
        y_pred.append(most_common_label)
    return np.array(y_pred)

# Визначення кількості найближчих сусідів
k = 3

# Класифікація за допомогою kNN
y_pred_5 = k_nearest_neighbors(x_train_5, y_train_5, x_test_5, k)

# Оцінка точності класифікації
accuracy_5 = np.mean(y_pred_5 == y_test_5)
print(f"Точність класифікації методом k найближчих сусідів: {accuracy_5:.2f}")

# Візуалізація
plt.figure(figsize=(10, 6))
plt.scatter(x_test_5[:, 0], x_test_5[:, 1], c=y_pred_5, cmap='viridis', s=50)
plt.title('Класифікація методом k найближчих сусідів')
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.colorbar(label='Клас')
plt.show()
