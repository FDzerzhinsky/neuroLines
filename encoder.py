import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Пример датафрейма
data = pd.DataFrame({
    'feature1': ['A', 'B', 'A', 'C'],
    'feature2': ['X', 'Y', 'X', 'Z'],
    'feature3': ['K', 'L', 'M', 'N']
})

# Инициализация LabelEncoder для каждого столбца
label_encoders = {}
for column in data.columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Преобразование датафрейма в двухмерный np.array
encoded_array = data.values

print(encoded_array)