# Этот файл содержит класс Preparator, который предназначен для предобработки данных
# Класс импортирует данные из файла .xlsx, берёт с указанных листов входные и выходные параметры для обучения нейросети

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
class Preparator:
    def __init__(self, path, x_sheet_name, y_sheet_name):
        self.path = path
        self.x_sheet_name = x_sheet_name
        self.y_sheet_name = y_sheet_name
        self.numerical = ['Скорость линии']
        self.x_data = None
        self.y_data = None
        self.x_cat = None
        self.x_num = None
        self.y_cat = None
        self.y_onehot = dict()
        self._load_data()
        self.encode_categorical()
        self.prepare_numerical()
        self.one_hot_encode()

    def _load_data(self):
        #   Метод для загрузки данных из файла .xlsx
        self.x_data = pd.read_excel(self.path, sheet_name=self.x_sheet_name).drop(columns=['№'])
        self.y_data = pd.read_excel(self.path, sheet_name=self.y_sheet_name).drop(columns=['№'])

    def prepare_numerical(self):
        #   Метод для подготовки и нормализации числовых данных
        for column in self.numerical:
            col_min = self.x_data[column].min()
            col_max = self.x_data[column].max()
            self.x_data[column] = (self.x_data[column] - col_min) / (col_max - col_min)
        self.x_num = self.x_data[self.numerical].values


    def encode_categorical(self):
        #   Метод для кодирования категориальных данных
        x_enc, y_enc = pd.DataFrame(), pd.DataFrame()
        label_encoder = LabelEncoder()
        x_encoders = {}
        y_encoders = {}
        for column in self.x_data.columns:
            if column not in self.numerical:
                x_enc[column] = label_encoder.fit_transform(self.x_data[column])
                x_encoders[column] = label_encoder
        for column in self.y_data.columns:
            y_enc[column] = label_encoder.fit_transform(self.y_data[column])
            y_encoders[column] = label_encoder
        self.x_cat, self.y_cat = x_enc.values, y_enc
        # self.x_enc, self.y_enc = x_enc, y_enc

    def one_hot_encode(self):
        #   Метод для one-hot кодирования выходных данных
        for column in self.y_cat.columns:
            one_hot_encoder = OneHotEncoder(sparse_output=False)
            self.y_onehot[column] = one_hot_encoder.fit_transform(self.y_cat[column].values.reshape(-1, 1))
