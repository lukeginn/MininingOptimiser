import pandas as pd
import logging as logger

def read_csv(file_path):
    logger.info(f"Reading CSV file from {file_path}")
    data = pd.read_csv(file_path)
    logger.info("CSV file read successfully")
    return data