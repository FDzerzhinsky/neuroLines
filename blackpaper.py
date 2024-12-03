import numpy as np
from sklearn.model_selection import train_test_split

# Пример входных и выходных данных
X1 = np.array([0, 1, 2, 3])
X2 = np.array([[8, 9], [10, 11], [12, 13], [14, 15]])
y = np.array([0, 1, 0, 1])

# Разделение данных на обучающие и тестовые множества
X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(X1, X2, y, test_size=0.2, random_state=42)

print("X1_train:", X1_train)
print("X1_test:", X1_test)
print("X2_train:", X2_train)
print("X2_test:", X2_test)
print("y_train:", y_train)
print("y_test:", y_test)