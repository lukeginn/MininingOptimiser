import logging as logger

def export_data(data, path, path_suffix):
    path = str(path)
    path = path.replace(".csv", "")
    full_file_path = f"{path}_{path_suffix}.csv"
    data.to_csv(full_file_path, index=True)
    logger.info(f"Data written to {full_file_path}")

    return None