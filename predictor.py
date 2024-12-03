import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Dropout
class Predictor:
    def __init__(self, data):
        self.data = data

    def build(self):
        # Метод строит модель
        # Делим данные
        label = 'Устройство нанесения'

        trn_cat_data, tst_cat_data, trn_num_data, tst_num_data, trn_labels, tst_labels = train_test_split(self.data.x_cat, self.data.x_num,
                                                                  self.data.y_onehot[label],
                                                                  test_size=0.2, random_state=42)

        # Определите параметры эмбеддинг-слоя
        input_dim = np.max(trn_cat_data) + 1  # Размер словаря (максимальное значение + 1)
        output_dim = 4  # Размерность эмбеддингов

        # Вход для категориальных данных
        categorical_input = Input(shape=(trn_cat_data.shape[1],), name='categorical_input')
        embedding_layer = Embedding(input_dim=input_dim, output_dim=output_dim, input_length=trn_cat_data.shape[1])(
            categorical_input)
        flatten_layer = Flatten()(embedding_layer)

        # Вход для числовых данных
        numerical_input = Input(shape=(1,), name='numerical_input')

        # Объединение эмбеддингов и числовых данных
        concatenated = Concatenate()([flatten_layer, numerical_input])

        # Полносвязные слои с дропаутами
        dense_layer_1 = Dense(32, activation='relu')(concatenated)
        dropout_1 = Dropout(0.5)(dense_layer_1)
        dense_layer_2 = Dense(16, activation='relu')(dropout_1)
        dropout_2 = Dropout(0.25)(dense_layer_2)

        # Выходной слой
        output_layer = Dense(6, activation='sigmoid')(dropout_2)

        # Создание модели
        model = Model(inputs=[categorical_input, numerical_input], outputs=output_layer)

        # Компиляция модели
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Вывод структуры модели
        model.summary()

        # Пример обучения модели

        model.fit([trn_cat_data, trn_num_data], trn_labels, epochs=200)