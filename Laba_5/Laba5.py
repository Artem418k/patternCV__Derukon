#1.Реалізувати SVM-класифікатор з лінійним і різними нелінійним ядрами згідно з варіантом.

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

x_train_5 = np.array([[27, 46], [8, 22], [5, 30], [5, 19], [15, 23], [34, 42], [10, 23], [35, 32]])
y_train_5 = np.array([-1, -1, 1, -1, 1, 1, -1, -1])

# Linear SVM
clf_linear = svm.SVC(kernel='linear')
clf_linear.fit(x_train_5, y_train_5)

# Non-linear SVM with Radial Basis Function (RBF) kernel
clf_nonlinear = svm.SVC(kernel='rbf', gamma='auto')
clf_nonlinear.fit(x_train_5, y_train_5)

# Visualizing the results
plt.figure(figsize=(12, 5))

# Plotting linear SVM results
plt.subplot(1, 2, 1)
plt.scatter(x_train_5[:, 0], x_train_5[:, 1], c=y_train_5, cmap=plt.cm.coolwarm, s=30)
plt.scatter(clf_linear.support_vectors_[:, 0], clf_linear.support_vectors_[:, 1], s=100, facecolors='none', edgecolors='k')
plt.title('Linear SVM')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Plotting non-linear SVM results
plt.subplot(1, 2, 2)
plt.scatter(x_train_5[:, 0], x_train_5[:, 1], c=y_train_5, cmap=plt.cm.coolwarm, s=30)
plt.scatter(clf_nonlinear.support_vectors_[:, 0], clf_nonlinear.support_vectors_[:, 1], s=100, facecolors='none', edgecolors='k')
plt.title('Non-linear SVM with RBF Kernel')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()

#2.Проаналізувати іпорівняти результати класифікації, обчисливши відповідні метрики.

import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.exceptions import UndefinedMetricWarning

# Ваши дані для навчальної вибірки
x_train_5 = np.array([[27, 46], [8, 22], [5, 30], [5, 19], [15, 23], [34, 42], [10, 23], [35, 32]])
y_train_5 = np.array([-1, -1, 1, -1, 1, 1, -1, -1])

# Розділення даних на навчальний і тестовий набори
x_train, x_test, y_train, y_test = train_test_split(x_train_5, y_train_5, test_size=0.2, random_state=42)

# Вимкнення попереджень про невизначені метрики
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Linear SVM
clf_linear = svm.SVC(kernel='linear')
clf_linear.fit(x_train, y_train)
linear_predictions = clf_linear.predict(x_test)

# Non-linear SVM with RBF kernel
clf_nonlinear = svm.SVC(kernel='rbf', gamma='auto')
clf_nonlinear.fit(x_train, y_train)
nonlinear_predictions = clf_nonlinear.predict(x_test)

# Обчислення метрик для лінійного SVM
linear_accuracy = accuracy_score(y_test, linear_predictions)
linear_classification_report = classification_report(y_test, linear_predictions)

# Обчислення метрик для нелінійного SVM з RBF ядром
nonlinear_accuracy = accuracy_score(y_test, nonlinear_predictions)
nonlinear_classification_report = classification_report(y_test, nonlinear_predictions)

# Виведення результатів
print("Linear SVM:")
print("Accuracy:", linear_accuracy)
print("Classification Report:")
print(linear_classification_report)

print("\nNon-linear SVM with RBF kernel:")
print("Accuracy:", nonlinear_accuracy)
print("Classification Report:")
print(nonlinear_classification_report)