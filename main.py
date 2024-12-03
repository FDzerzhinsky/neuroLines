import os
from preprocessor import Preparator
from predictor import Predictor
project_root = os.path.dirname(__file__)
data_path = os.path.join(project_root, 'data', 'Синтетический датасет.xlsx')

data = Preparator(data_path, 'x_list', 'y_list')

predictor = Predictor(data)
predictor.build()

# # Сохраняем данные датафреймов data.x_enc и data.y_enc в файлы .xlsx
# data.x_enc.to_excel('x_data.xlsx', index=False)
# data.y_enc.to_excel('y_data.xlsx', index=False)
