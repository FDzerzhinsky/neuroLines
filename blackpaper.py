import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate

# Пример двумерного ndarray с категориальными признаками
categorical_data = np.array([
    [1, 5, 1],
    [4, 5, 2],
    [7, 8, 9],
    [1, 1, 3],
    [2, 8, 2],
    [2, 2, 8],
])

# Пример числовых признаков (например, скорость ветра)
numerical_data = np.array([3.5, 7.2, 5.1, 6.3, 4.8, 9.0])

# Определите параметры эмбеддинг-слоя
input_dim = np.max(categorical_data) + 1  # Размер словаря (максимальное значение + 1)
output_dim = 8  # Размерность эмбеддингов

# Вход для категориальных данных
categorical_input = Input(shape=(categorical_data.shape[1],), name='categorical_input')
embedding_layer = Embedding(input_dim=input_dim, output_dim=output_dim, input_length=categorical_data.shape[1])(categorical_input)
flatten_layer = Flatten()(embedding_layer)

# Вход для числовых данных
numerical_input = Input(shape=(1,), name='numerical_input')

# Объединение эмбеддингов и числовых данных
concatenated = Concatenate()([flatten_layer, numerical_input])

# Полносвязный слой
dense_layer = Dense(1, activation='sigmoid')(concatenated)

# Создание модели
model = Model(inputs=[categorical_input, numerical_input], outputs=dense_layer)

# Компиляция модели
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Вывод структуры модели
model.summary()

# Пример обучения модели
labels = np.array([0, 1, 1, 0, 1, 1])  # Пример меток
model.fit([categorical_data, numerical_data], labels, epochs=10)