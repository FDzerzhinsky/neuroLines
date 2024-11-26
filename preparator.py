# Этот файл содержит класс Preparator, который предназначен для предобработки данных
# Класс импортирует данные из файла .xlsx, берёт с указанных листов входные и выходные параметры для обучения нейросети


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
class Preparator:
    def __init__(self, path, x_sheet_name, y_sheet_name):
        self.path = path
        self.x_sheet_name = x_sheet_name
        self.y_sheet_name = y_sheet_name
        self.x_data = None
        self.y_data = None
        self.x_enc = None
        self.y_enc = None
        self._load_data()
        self.encode()

    def _load_data(self):
        self.x_data = pd.read_excel(self.path, sheet_name=self.x_sheet_name).drop(columns=['№'])
        self.y_data = pd.read_excel(self.path, sheet_name=self.y_sheet_name).drop(columns=['№'])

    def encode(self):
        x_enc, y_enc = pd.DataFrame(), pd.DataFrame()
        label_encoder = LabelEncoder()
        x_encoders = {}
        y_encoders = {}
        for column in self.x_data.columns:
            if column != 'Скорость линии':
                x_enc[column] = label_encoder.fit_transform(self.x_data[column])
            else:
                x_enc[column] = self.x_data[column]
                x_encoders[column] = label_encoder
        for column in self.y_data.columns:
            y_enc[column] = label_encoder.fit_transform(self.y_data[column])
            y_encoders[column] = label_encoder
        self.x_enc, self.y_enc = x_enc.values, y_enc.values
        # self.x_enc, self.y_enc = x_enc, y_enc


