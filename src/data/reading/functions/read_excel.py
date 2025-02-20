import numpy as np
import pandas as pd
import logging as logger


# TO-DO: Clean up and turn into a class
def reading_preprocessing_and_merging_multiple_excel_workbooks(config, file_paths):
    data = read_multiple_excel_workbooks(file_paths=[file_paths[0]])
    data = preprocess_multiple_excel_workbooks(
        data=data,
        min_non_null_values_for_empty_columns=data_config.preprocessing_excel_workbook.min_non_null_values_for_empty_columns,
        min_non_null_values_for_empty_rows=data_config.preprocessing_excel_workbook.min_non_null_values_for_empty_rows,
        exclude_sheets_for_combining_first_two_rows=data_config.preprocessing_excel_workbook.exclude_sheets_for_combining_first_two_rows,
        missing_value_words=data_config.preprocessing_excel_workbook.missing_value_words,
        unique_phrases_to_replace_threshold=data_config.preprocessing_excel_workbook.unique_phrases_to_replace_threshold,
        unique_values_to_stop_replacement_threshold=data_config.preprocessing_excel_workbook.unique_values_to_stop_replacement_threshold,
        timestamp_column_for_merging=data_config.timestamp,
    )
    data = custom_preprocessing(data)

    extra_data = read_multiple_excel_workbooks(file_paths=[file_paths[1]])
    extra_data = preprocess_excel_workbook(
        data=extra_data[file_paths[1]],
        min_non_null_values_for_empty_columns=data_config.preprocessing_excel_workbook.min_non_null_values_for_empty_columns,
        min_non_null_values_for_empty_rows=data_config.preprocessing_excel_workbook.min_non_null_values_for_empty_rows,
        exclude_sheets_for_combining_first_two_rows=data_config.preprocessing_excel_workbook.exclude_sheets_for_combining_first_two_rows,
        missing_value_words=data_config.preprocessing_excel_workbook.missing_value_words,
        unique_phrases_to_replace_threshold=data_config.preprocessing_excel_workbook.unique_phrases_to_replace_threshold,
        unique_values_to_stop_replacement_threshold=data_config.preprocessing_excel_workbook.unique_values_to_stop_replacement_threshold,
        timestamp_column_for_merging=data_config.timestamp,
    )
    extra_data = custom_preprocessing_for_extra_data(extra_data)

    extra_data2 = read_multiple_excel_workbooks(file_paths=[file_paths[2]])
    extra_data2 = custom_preprocessing_for_extra_data_2(extra_data2)
    extra_data2 = preprocess_excel_workbook(
        data=extra_data2[file_paths[2]],
        min_non_null_values_for_empty_columns=data_config.preprocessing_excel_workbook.min_non_null_values_for_empty_columns,
        min_non_null_values_for_empty_rows=data_config.preprocessing_excel_workbook.min_non_null_values_for_empty_rows,
        exclude_sheets_for_combining_first_two_rows=data_config.preprocessing_excel_workbook.exclude_sheets_for_combining_first_two_rows,
        missing_value_words=data_config.preprocessing_excel_workbook.missing_value_words,
        unique_phrases_to_replace_threshold=data_config.preprocessing_excel_workbook.unique_phrases_to_replace_threshold,
        unique_values_to_stop_replacement_threshold=data_config.preprocessing_excel_workbook.unique_values_to_stop_replacement_threshold,
        timestamp_column_for_merging=data_config.timestamp,
    )

    data = merge_excel_workbooks(
        data=data,
        extra_data=extra_data,
        extra_data2=extra_data2,
        timestamp_column=data_config.timestamp,
    )
    data = apply_custom_filters(data=data)

    return data


def read_multiple_excel_workbooks(file_paths):
    data = {}
    for file_path in file_paths:
        dataset = read_excel_workbook(file_path=file_path)
        data[file_path] = dataset
    logger.info("Multiple Excel workbooks read successfully")
    return data


def read_excel_workbook(file_path):
    logger.info(f"Reading Excel workbook from {file_path}")
    xls = pd.ExcelFile(file_path)
    # Read each sheet without using the first row as header
    data = {
        sheet_name: xls.parse(sheet_name, header=None) for sheet_name in xls.sheet_names
    }
    logger.info("Excel workbook read successfully")
    return data


def align_sheets_across_multiple_excel_workbooks(data):
    # Get dataset names
    dataset_names = list(data.keys())

    # Get the list of sheets for each dataset
    list_of_sheets = [list(data[dataset].keys()) for dataset in dataset_names]

    # Find the common sheets across all datasets
    common_sheets = set(list_of_sheets[0]).intersection(*list_of_sheets[1:])
    logger.info(f"Common sheets across all datasets: {common_sheets}")

    # Ensure the order of sheets is consistent across all datasets
    for dataset in dataset_names:
        data[dataset] = {sheet: data[dataset][sheet] for sheet in sorted(common_sheets)}

    return data


def preprocess_multiple_excel_workbooks(
    data,
    min_non_null_values_for_empty_columns=4,
    min_non_null_values_for_empty_rows=3,
    exclude_sheets_for_combining_first_two_rows=None,
    missing_value_words=["na", "nan", "missing", "none", "null"],
    unique_phrases_to_replace_threshold=3,
    unique_values_to_stop_replacement_threshold=10,
    timestamp_column_for_merging="timestamp",
):
    for dataset_name, dataset in data.items():
        data[dataset_name] = preprocess_excel_workbook(
            data=dataset,
            min_non_null_values_for_empty_columns=min_non_null_values_for_empty_columns,
            min_non_null_values_for_empty_rows=min_non_null_values_for_empty_rows,
            exclude_sheets_for_combining_first_two_rows=exclude_sheets_for_combining_first_two_rows,
            missing_value_words=missing_value_words,
            unique_phrases_to_replace_threshold=unique_phrases_to_replace_threshold,
            unique_values_to_stop_replacement_threshold=unique_values_to_stop_replacement_threshold,
            timestamp_column_for_merging=timestamp_column_for_merging,
        )
    return data


def preprocess_excel_workbook(
    data,
    min_non_null_values_for_empty_columns,
    min_non_null_values_for_empty_rows,
    exclude_sheets_for_combining_first_two_rows,
    missing_value_words,
    unique_phrases_to_replace_threshold,
    unique_values_to_stop_replacement_threshold,
    timestamp_column_for_merging,
):
    data = remove_empty_columns(
        data=data, min_non_null_values=min_non_null_values_for_empty_columns
    )
    data = remove_empty_rows(
        data=data, min_non_null_values=min_non_null_values_for_empty_rows
    )
    data = remove_header_names(data=data)
    data = add_column_names(data=data)
    data = make_first_row_the_header(data=data)
    data = combine_the_first_row_of_units_with_the_column_names(
        data=data, exclude_sheets=exclude_sheets_for_combining_first_two_rows
    )
    data = format_column_names_per_sheet(data=data)
    data = identify_and_replace_words_representing_missing_values(
        data=data, missing_value_words=missing_value_words
    )
    data = identify_and_replace_unique_phrases(
        data=data,
        unique_phrase_threshold=unique_phrases_to_replace_threshold,
        unique_values_threshold=unique_values_to_stop_replacement_threshold,
    )
    data = format_column_types(data=data)
    data = merge_sheets_together(
        data=data,
        merge_on_timestamp=True,
        timestamp_column=timestamp_column_for_merging,
    )
    check_column_types(data=data)
    data = remove_duplicate_rows(data=data)
    return data


def merge_excel_workbooks(data, extra_data, extra_data2, timestamp_column="timestamp"):
    logger.info("Merging Excel workbooks")

    # Concatenate all datasets along the rows and reset the index
    data = pd.concat(data.values(), axis=0).reset_index(drop=True)

    # Merge via the timestamp column
    data = data.merge(extra_data, on=timestamp_column, how="outer")

    # Merge via the timestamp column
    data = data.merge(extra_data2, on=timestamp_column, how="outer")

    return data


def remove_empty_columns(data, min_non_null_values):
    logger.info("Removing empty columns")
    for sheet_name, df in data.items():
        # Remove columns that are completely empty
        df = df.dropna(axis=1, how="all")

        # Remove columns that have fewer than the specified number of non-null values
        for col in df.columns:
            if df[col].nunique(dropna=True) < min_non_null_values:
                df = df.drop(columns=[col])

        data[sheet_name] = df
    return data


def remove_empty_rows(data, min_non_null_values):
    logger.info("Removing empty rows and columns with fewer than min_non_null_values")
    for sheet_name, df in data.items():
        # Remove rows that are completely empty
        df = df.dropna(axis=0, how="all")

        # Remove rows that have fewer than the specified number of non-null values
        df = df.dropna(thresh=min_non_null_values)

        data[sheet_name] = df
    return data


def remove_first_rows(data, num_rows):
    logger.info(f"Removing the first {num_rows} rows")
    for sheet_name, df in data.items():
        data[sheet_name] = df.iloc[num_rows:]
    return data


def remove_header_names(data):
    logger.info("Removing header names")
    for sheet_name, df in data.items():
        df.columns = range(df.shape[1])
        data[sheet_name] = df
    return data


def add_column_names(data):
    logger.info("Adding column names where missing")
    for sheet_name, df in data.items():
        for col in df.columns:
            if pd.isnull(df.iloc[0][col]) or pd.isnull(df.iloc[1][col]):
                # Check if the rest of the column looks like timestamps
                try:
                    pd.to_datetime(df[col], errors="raise")
                    df[col].name = "timestamp"
                    df.iloc[0, df.columns.get_loc(col)] = "timestamp"
                except (ValueError, TypeError):
                    pass
    return data


def make_first_row_the_header(data):
    logger.info("Making the first row the header")
    for sheet_name, df in data.items():
        df.columns = df.iloc[0]
        data[sheet_name] = df.drop(df.index[0])
    return data


def combine_the_first_row_of_units_with_the_column_names(data, exclude_sheets=None):
    logger.info("Combining the first row of units with the column names")

    if exclude_sheets is None:
        exclude_sheets = []

    for sheet_name, df in data.items():
        if sheet_name in exclude_sheets:
            logger.info(f"Skipping sheet {sheet_name} as it is in the exclude list")
            continue

        # Combine the first row (units) with the column names if units exist
        units = df.iloc[0]
        if not units.isnull().all():
            df.columns = [
                f"{col} ({unit})" if pd.notnull(unit) else col
                for col, unit in zip(df.columns, units)
            ]
            # Drop the first row as it is now part of the column names
            data[sheet_name] = df.drop(df.index[0])

    return data


def format_column_names_per_sheet(data):
    logger.info("Formatting column names")
    for sheet_name, df in data.items():
        df.columns = (
            df.columns.str.strip()
            .str.lower()
            .str.replace(" ", "_")
            .str.replace("/", "_")
            .str.replace("(", "_")
            .str.replace(")", "_")
            .str.replace("%", "_perc_")
            .str.replace("+", "_plus_")
            .str.replace("-", "_")
            .str.replace(",", "_")
            .str.replace(".", "_")
            .str.replace(":", "_")
            .str.replace(";", "_")
            .str.replace("=", "_")
        )
        df.columns = df.columns.str.replace(r"__+", "_", regex=True)
        df.columns = df.columns.str.rstrip("_")
    return data


def identify_and_replace_words_representing_missing_values(
    data, missing_value_words=["na", "nan", "missing", "none", "null"]
):
    logger.info(
        "Identifying and replacing words representing missing values in columns"
    )
    for sheet_name, df in data.items():
        for col in df.columns:
            # Check if the values in the column look like missing values
            logger.info(f"Replacing missing values in column {col}")
            df[col] = df[col].replace(missing_value_words, pd.NA)
            df[col] = df[col].replace(np.nan, pd.NA)
    return data


def identify_and_replace_unique_phrases(
    data, unique_phrase_threshold=1, unique_values_threshold=10
):
    logger.info("Identifying and replacing unique phrases in columns")
    for sheet_name, df in data.items():
        for col in df.columns:
            # Check if the values in the column look like datetimes
            try:
                pd.to_datetime(df[col].astype(str), errors="raise")
                continue
            except (ValueError, TypeError):
                pass

            # if col == 'ft13_jameson_vacuum__kpa___340ft013_vacuum_pressure_':
            #     print('debug here')

            # Exclude numbers and find unique phrases
            unique_phrases = (
                df[col].astype(str).str.extractall(r"(\b[A-Za-z\s]+\b)")[0].unique()
            )
            if (
                len(unique_phrases) <= unique_phrase_threshold
                and df[col].nunique() > unique_values_threshold
            ):
                for phrase in unique_phrases:
                    # Find all row indices that contain the unique phrase
                    indices_with_phrase = (
                        df[col].astype(str).str.contains(phrase, na=False)
                    )

                    # Set the entire row index for that column to NA
                    df.loc[indices_with_phrase, col] = pd.NA
    return data


def format_column_types(data):
    logger.info("Formatting column types")
    for sheet_name, df in data.items():
        for col in df.columns:

            # Check if the values in the column look like datetimes
            try:
                df[col] = pd.to_datetime(df[col].astype(str), errors="raise")
                continue
            except (ValueError, TypeError):
                pass

            # Check if the values in the column look like integers
            if df[col].astype(str).str.match(r"^-?\d+$").all():
                df[col] = df[col].astype(int)
                continue

            # Check if the values in the column look like booleans
            if df[col].astype(str).str.lower().isin(["true", "false"]).all():
                df[col] = (
                    df[col].astype(str).str.lower().map({"true": True, "false": False})
                )
                continue

            # Check if the values in the column look like floats
            try:
                df[col] = pd.to_numeric(df[col], errors="raise")
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
                continue
            except ValueError:
                pass

            # If no other type matched, convert the column to string
            df[col] = df[col].astype(str)

    return data


def check_column_types_per_sheet(data):

    for sheet_name, df in data.items():
        logger.info(f"Column types for sheet {sheet_name}:")
        check_column_types(data=df)


def check_column_types(data):

    for col in data.columns:
        if pd.api.types.is_datetime64_any_dtype(data[col]):
            logger.info(f"Column {col} is of type datetime")
        elif pd.api.types.is_float_dtype(data[col]):
            logger.info(f"Column {col} is of type float")
        elif pd.api.types.is_integer_dtype(data[col]):
            logger.info(f"Column {col} is of type integer")
        elif pd.api.types.is_bool_dtype(data[col]):
            logger.info(f"Column {col} is of type boolean")
        elif pd.api.types.is_categorical_dtype(data[col]):
            logger.info(f"Column {col} is of type category")
        else:
            logger.info(f"Column {col} is of type string")


def merge_sheets_together(data, merge_on_timestamp=False, timestamp_column="timestamp"):
    logger.info("Merging sheets together")
    if merge_on_timestamp:
        # Ensure the timestamp column exists in all sheets
        for sheet_name, df in data.items():
            if timestamp_column not in df.columns:
                raise ValueError(
                    f"Timestamp column '{timestamp_column}' not found in sheet '{sheet_name}'"
                )

        # Merge all sheets together on the timestamp column using join
        merged_data = None
        for sheet_name, df in data.items():
            if merged_data is None:
                merged_data = df.set_index(timestamp_column)
            else:
                merged_data = merged_data.join(
                    df.set_index(timestamp_column), how="outer"
                )
        merged_data = merged_data.reset_index()
    else:
        # Merge all sheets together without considering the timestamp column
        merged_data = pd.concat(data.values(), ignore_index=True)

    return merged_data


def remove_duplicate_rows(data):
    logger.info("Removing duplicate rows")
    data = data.drop_duplicates()
    return data


def apply_custom_filters(data):
    logger.info("Applying custom filters")

    print()
    second_column_name = data.columns[1]
    first_non_null_value = data[second_column_name].dropna().iloc[0]
    first_non_null_timestamp = data.loc[
        data[second_column_name] == first_non_null_value, "estampa_de_tiempo"
    ].iloc[0]
    logger.info(f"The first non-null estampa_de_tiempo is: {first_non_null_timestamp}")

    last_non_null_value = data[second_column_name].dropna().iloc[-1]
    last_non_null_timestamp = data.loc[
        data[second_column_name] == last_non_null_value, "estampa_de_tiempo"
    ].iloc[-1]
    logger.info(f"The last non-null estampa_de_tiempo is: {last_non_null_timestamp}")

    data = data[data["estampa_de_tiempo"] >= first_non_null_timestamp]
    data = data[data["estampa_de_tiempo"] <= last_non_null_timestamp]

    return data


def custom_preprocessing(data):
    logger.info("Applying custom preprocessing")

    for sheet_name, df in data.items():
        if "estampa_de_tiempo" in df.columns:
            df["estampa_de_tiempo"] = pd.to_datetime(df["estampa_de_tiempo"])
            df["estampa_de_tiempo"] = df["estampa_de_tiempo"].apply(
                lambda x: x.replace(second=0)
            )
            data[sheet_name] = df

    for sheet_name, df in data.items():
        if "estampa_de_tiempo" in df.columns:
            df["estampa_de_tiempo"] = pd.to_datetime(df["estampa_de_tiempo"])
            df["estampa_de_tiempo"] = df["estampa_de_tiempo"].apply(
                lambda x: (
                    x.replace(second=0, minute=0)
                    if x.minute < 30
                    else x.replace(second=0, minute=30)
                )
            )
            data[sheet_name] = df

    for sheet_name, df in data.items():
        df["valvula_hacia_este"] = df["valvula_hacia_este"].replace("Open", 1)
        df["valvula_hacia_este"] = df["valvula_hacia_este"].replace("Off", 0)
        df["valvula_hacia_este"] = df["valvula_hacia_este"].astype(bool).astype(int)

        df["valvula_hacia_oeste"] = df["valvula_hacia_oeste"].replace("Open", 1)
        df["valvula_hacia_oeste"] = df["valvula_hacia_oeste"].replace("Off", 0)
        df["valvula_hacia_oeste"] = df["valvula_hacia_oeste"].astype(bool).astype(int)

        data[sheet_name] = df

    return data


def custom_preprocessing_for_extra_data(data):
    logger.info("Applying custom preprocessing for extra data")

    if "turno" in data.columns and "estampa_de_tiempo" in data.columns:
        data["estampa_de_tiempo"] = data.apply(
            lambda row: (
                f"{row['estampa_de_tiempo']} 12:00:00"
                if row["turno"] == "Dia"
                else (
                    f"{row['estampa_de_tiempo']} 00:00:00"
                    if row["turno"] == "Noche"
                    else row["estampa_de_tiempo"]
                )
            ),
            axis=1,
        )
    data = data.drop(columns=["año", "mes", "turno"], errors="ignore")

    # Clean up the estampa_de_tiempo column
    data["estampa_de_tiempo"] = pd.to_datetime(
        data["estampa_de_tiempo"]
        .astype(str)
        .str.replace(r" 00:00:00 00:00:00", " 00:00:00")
    )
    data["estampa_de_tiempo"] = pd.to_datetime(
        data["estampa_de_tiempo"]
        .astype(str)
        .str.replace(r" 00:00:00 12:00:00", " 12:00:00")
    )

    # Calculate the proportions of each category in 'tipo_de_mineral' and save it to a CSV file
    proportions = data["tipo_de_mineral"].value_counts(normalize=True).reset_index()
    proportions.columns = ["tipo_de_mineral", "proporción_de_tipo_de_mineral"]
    proportions.to_csv("outputs/proporción_de_tipo_de_mineral.csv", index=False)

    # Calculate the proportions of tonelaje for each tipo_de_mineral and save it to a CSV file
    proportions_tonelaje = (
        data.groupby("tipo_de_mineral")["tonelaje"].sum().reset_index()
    )
    proportions_tonelaje["proporción_de_tonelaje_del_tipo_de_mineral"] = (
        proportions_tonelaje["tonelaje"] / proportions_tonelaje["tonelaje"].sum()
    )
    proportions_tonelaje = proportions_tonelaje[
        ["tipo_de_mineral", "proporción_de_tonelaje_del_tipo_de_mineral"]
    ]
    proportions_tonelaje = proportions_tonelaje.sort_values(
        by="proporción_de_tonelaje_del_tipo_de_mineral", ascending=False
    )
    proportions_tonelaje.to_csv(
        "outputs/proporción_de_tonelaje_del_tipo_de_mineral.csv", index=False
    )

    # Feed Blend Combination 1 Clean-up
    data["tipo_de_mineral_copy"] = data["tipo_de_mineral"]
    data["tipo_de_mineral_copy"] = data["tipo_de_mineral_copy"].replace(
        data["tipo_de_mineral_copy"][
            data["tipo_de_mineral_copy"].str.contains("p", case=False, na=False)
        ].unique(),
        "ORE_PAMPACANCHA",
    )
    data["tipo_de_mineral_copy"] = data["tipo_de_mineral_copy"].replace(
        data["tipo_de_mineral_copy"][
            ~data["tipo_de_mineral_copy"].str.contains("p", case=False, na=False)
        ].unique(),
        "ORE_CONSTANCIA",
    )

    # Feed Blend Combination 2 Clean-up
    data["tipo_de_mineral"] = data["tipo_de_mineral"].replace(
        [
            "ORE_K",
            "ORE_K_P",
            "ORE_K_p",
            "STK01_K",
            "STK1_K_PC",
            "STK02_K",
            "STK03_K",
            "STK4_K",
        ],
        "ORE_K",
    )
    data["tipo_de_mineral"] = data["tipo_de_mineral"].replace(
        ["ORE_M", "ORE_M_P"], "ORE_M"
    )
    data["tipo_de_mineral"] = data["tipo_de_mineral"].replace(
        ["ORE_H", "ORE_H_P", "STK4_H"], "ORE_H"
    )
    data["tipo_de_mineral"] = data["tipo_de_mineral"].replace(["STK01_S"], "ORE_S")
    data["tipo_de_mineral"] = data["tipo_de_mineral"].replace(
        ["-HIZN", "_HIZN", "_HIZN_P"], "HIZN"
    )
    data["tipo_de_mineral"] = data["tipo_de_mineral"].replace(
        ["_HIAU", "_HIAU_P"], "HIAU"
    )
    data["tipo_de_mineral"] = data["tipo_de_mineral"].replace(
        ["STK02", "STK2-"], "STK02"
    )
    data["tipo_de_mineral"] = data["tipo_de_mineral"].replace(
        ["STK03", "STK3-"], "STK03"
    )
    data["tipo_de_mineral"] = data["tipo_de_mineral"].replace(
        ["STK4-", "STK4_CS", "STK4_P", "STK4_PC"], "STK04"
    )  # This only appears in 2024
    data["tipo_de_mineral"] = data["tipo_de_mineral"].replace(
        ["STK-O", "STKOS", "STK0", "STK02", "STK03", "STK04"], "STK"
    )
    data["tipo_de_mineral"] = data["tipo_de_mineral"].replace(
        ["NAG-1", "NAG-1_P", "ORE-1_P", "ore_H_P", "<NA>", "HIAU"], "OTHER"
    )

    # Finding the total tonnes per shift
    tonelaje_total = data.groupby("estampa_de_tiempo")["tonelaje"].sum().reset_index()
    data = data.merge(tonelaje_total, on="estampa_de_tiempo", suffixes=("", "_total"))
    data["proporción_de_tonelaje"] = data["tonelaje"] / data["tonelaje_total"]

    # Weighted averaging all of the samples from repeated day and night to a single day and night respectively
    logger.info("Weighted averaging all of the samples to day and night")
    numeric_columns = data.select_dtypes(include=["number"]).columns
    numeric_columns = numeric_columns.difference(
        ["tonelaje", "proporción_de_tonelaje", "tonelaje_total"]
    )
    data_aggregated = (
        data.groupby("estampa_de_tiempo")
        .apply(
            lambda x: pd.Series(
                {
                    col: (x[col] * x["proporción_de_tonelaje"]).sum()
                    / x["proporción_de_tonelaje"].sum()
                    for col in numeric_columns
                }
            )
        )
        .reset_index()
    )
    data_aggregated["estampa_de_tiempo"] = pd.to_datetime(
        data_aggregated["estampa_de_tiempo"], errors="coerce"
    )
    data_aggregated = data_aggregated.sort_values(by="estampa_de_tiempo")

    # Pivot the data to get the total sum of proporción_de_tonelaje per shift for each tipo_de_mineral
    logger.info(
        "Pivoting data to get the total sum of proporción_de_tonelaje per shift for each tipo_de_mineral"
    )
    feed_blend_data_combination_1 = data.pivot_table(
        index="estampa_de_tiempo",
        columns="tipo_de_mineral_copy",
        values="proporción_de_tonelaje",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()

    feed_blend_data_combination_2 = data.pivot_table(
        index="estampa_de_tiempo",
        columns="tipo_de_mineral",
        values="proporción_de_tonelaje",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()

    data_aggregated = tonelaje_total.merge(
        data_aggregated, on="estampa_de_tiempo", how="outer"
    )
    data_aggregated = data_aggregated.merge(
        feed_blend_data_combination_1, on="estampa_de_tiempo", how="outer"
    )
    data_aggregated = data_aggregated.merge(
        feed_blend_data_combination_2, on="estampa_de_tiempo", how="outer"
    )

    return data_aggregated


def custom_preprocessing_for_extra_data_2(data):
    logger.info("Applying custom preprocessing for extra data 2")

    key1 = list(data.keys())[0]
    data = data[key1]
    key2 = list(data.keys())[0]
    data = data[key2]

    data.iloc[1:, 0] = pd.to_datetime(data.iloc[1:, 0], format="mixed").dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    data.iloc[1:, 0] = (
        pd.to_datetime(data.iloc[1:, 0])
        .apply(lambda x: x.replace(second=0))
        .dt.strftime("%Y-%m-%d %H:%M:%S")
    )
    data.iloc[1:, 0] = (
        pd.to_datetime(data.iloc[1:, 0])
        .apply(
            lambda x: (
                x.replace(second=0, minute=0)
                if x.minute < 30
                else x.replace(second=0, minute=30)
            )
        )
        .dt.strftime("%Y-%m-%d %H:%M:%S")
    )

    data = {key1: {key2: data}}

    return data
