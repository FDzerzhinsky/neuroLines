# Этот файл содержит класс Preparator, который предназначен для предобработки данных
# Класс импортирует данные из файла .xlsx, берёт с указанных листов входные и выходные параметры для обучения нейросети


import pandas as pd
import numpy as np

class Preparator:
    def __init__(self, path, x_sheet_name, y_sheet_name):
        self.path = path
        self.x_sheet_name = x_sheet_name
        self.y_sheet_name = y_sheet_name
        self.x_data = None
        self.y_data = None
        self._load_data()

    def _load_data(self):
        self.x_data = pd.read_excel(self.path, sheet_name=self.x_sheet_name)
        self.y_data = pd.read_excel(self.path, sheet_name=self.y_sheet_name)
