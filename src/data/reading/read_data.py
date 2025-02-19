import config.paths as paths
from src.data.reading.read_csv import read_csv

def read_data():
    data = read_csv(file_path=paths.Paths.DATA_FILE_1.value)
    return data