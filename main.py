import os
from preparator import Preparator

project_root = os.path.dirname(__file__)
data_path = os.path.join(project_root, 'data', 'Синтетический датасет.xlsx')

data = Preparator(data_path, 'x_list', 'y_list')


data