import pandas as pd
import logging as logger


def initial_preprocessing(data):

    data = renaming_columns(data)
    data = clean_data(data)
    data = converting_columns(data)
    data = aggregating_data(data)

    return data


def renaming_columns(data):
    # Make All columns Uppercase
    data.columns = map(str.upper, data.columns)

    # Replace % with PERC and make it always a suffix
    data.columns = [
        col.replace("%", "") + "_PERC" if "%" in col else col for col in data.columns
    ]

    # Remove leading whitespace from COLUMN names
    data.columns = [col.lstrip() for col in data.columns]

    # Replace spaces with underscOREs
    data.columns = [col.replace(" ", "_") for col in data.columns]

    return data


def clean_data(data):
    # Replace commas with dots in the data
    data = data.stack().str.replace(",", ".").unstack()

    return data


def converting_columns(data):
    # Converting columns to datetime
    data["DATE"] = pd.to_datetime(data["DATE"])

    # Converting columns to numeric
    data["IRON_FEED_PERC"] = pd.to_numeric(data["IRON_FEED_PERC"])
    data["SILICA_FEED_PERC"] = pd.to_numeric(data["SILICA_FEED_PERC"])
    data["STARCH_FLOW"] = pd.to_numeric(data["STARCH_FLOW"])
    data["AMINA_FLOW"] = pd.to_numeric(data["AMINA_FLOW"])
    data["ORE_PULP_FLOW"] = pd.to_numeric(data["ORE_PULP_FLOW"])
    data["ORE_PULP_PH"] = pd.to_numeric(data["ORE_PULP_PH"])
    data["ORE_PULP_DENSITY"] = pd.to_numeric(data["ORE_PULP_DENSITY"])
    data["FLOTATION_COLUMN_01_AIR_FLOW"] = pd.to_numeric(
        data["FLOTATION_COLUMN_01_AIR_FLOW"]
    )
    data["FLOTATION_COLUMN_02_AIR_FLOW"] = pd.to_numeric(
        data["FLOTATION_COLUMN_02_AIR_FLOW"]
    )
    data["FLOTATION_COLUMN_03_AIR_FLOW"] = pd.to_numeric(
        data["FLOTATION_COLUMN_03_AIR_FLOW"]
    )
    data["FLOTATION_COLUMN_04_AIR_FLOW"] = pd.to_numeric(
        data["FLOTATION_COLUMN_04_AIR_FLOW"]
    )
    data["FLOTATION_COLUMN_05_AIR_FLOW"] = pd.to_numeric(
        data["FLOTATION_COLUMN_05_AIR_FLOW"]
    )
    data["FLOTATION_COLUMN_06_AIR_FLOW"] = pd.to_numeric(
        data["FLOTATION_COLUMN_06_AIR_FLOW"]
    )
    data["FLOTATION_COLUMN_07_AIR_FLOW"] = pd.to_numeric(
        data["FLOTATION_COLUMN_07_AIR_FLOW"]
    )
    data["FLOTATION_COLUMN_01_LEVEL"] = pd.to_numeric(data["FLOTATION_COLUMN_01_LEVEL"])
    data["FLOTATION_COLUMN_02_LEVEL"] = pd.to_numeric(data["FLOTATION_COLUMN_02_LEVEL"])
    data["FLOTATION_COLUMN_03_LEVEL"] = pd.to_numeric(data["FLOTATION_COLUMN_03_LEVEL"])
    data["FLOTATION_COLUMN_04_LEVEL"] = pd.to_numeric(data["FLOTATION_COLUMN_04_LEVEL"])
    data["FLOTATION_COLUMN_05_LEVEL"] = pd.to_numeric(data["FLOTATION_COLUMN_05_LEVEL"])
    data["FLOTATION_COLUMN_06_LEVEL"] = pd.to_numeric(data["FLOTATION_COLUMN_06_LEVEL"])
    data["FLOTATION_COLUMN_07_LEVEL"] = pd.to_numeric(data["FLOTATION_COLUMN_07_LEVEL"])
    data["IRON_CONCENTRATE_PERC"] = pd.to_numeric(data["IRON_CONCENTRATE_PERC"])
    data["SILICA_CONCENTRATE_PERC"] = pd.to_numeric(data["SILICA_CONCENTRATE_PERC"])

    return data


def aggregating_data(data):
    # Here we aggregate 737453 into 4096 rows (hours). There was a reading every half a second for 172 days
    data = data.groupby(["DATE"]).mean().reset_index()

    return data
