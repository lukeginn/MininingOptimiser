import numpy as np
import pandas as pd
import logging as logger
import os
import matplotlib.pyplot as plt


def custom_plots(data, path):

    #data = preprocess_data_for_stacked_time_series_plots(data=data)
    create_stacked_time_series_plots_proportion_of_constancia_and_pampacancha(data=data, path=path)
    create_stacked_time_series_plots_proportion_of_other_ores(data=data, path=path)
    create_binned_box_and_whiskers_plot(data=data, path=path)
    create_binned_box_and_whiskers_plot_of_tonnage_bias(data=data, path=path)
    #create_7_day_time_series(data=data, path=path)
    create_7_day_time_series_2(data=data, path=path)

    return data

def create_stacked_time_series_plots_proportion_of_constancia_and_pampacancha(data, path):

    # Optional preprocessing
    suffix = '_mean' if 'ORE_PAMPACANCHA_mean' in data.columns else ''
    ORE_PAMPACANCHA = data[f'ORE_PAMPACANCHA{suffix}'].rolling(window=336, min_periods=1).mean()
    ORE_CONSTANCIA = data[f'ORE_CONSTANCIA{suffix}'].rolling(window=336, min_periods=1).mean()
    ORE_TOTAL = ORE_PAMPACANCHA + ORE_CONSTANCIA
    UPLIFT = 1/ORE_TOTAL
    ORE_PAMPACANCHA = ORE_PAMPACANCHA * UPLIFT
    ORE_CONSTANCIA = ORE_CONSTANCIA * UPLIFT

    # 30 minutes is one value. so 7 days is 7*24*2 = 336 values

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    #suffix = '_mean' if 'ORE_PAMPACANCHA_mean' in data.columns else ''
    ax.stackplot(data['estampa_de_tiempo'], 
                 ORE_PAMPACANCHA, 
                 ORE_CONSTANCIA, 
                 labels=['proporcion de pampacancha', 'proporcion de constancia'], 
                 colors=['#66b3ff', '#ff9999'])

    ax.legend(loc='upper left', facecolor='white', framealpha=1)
    ax.set_title('Series de Tiempo Apiladas: Proporciones de Tonelaje', color='black')
    ax.set_ylabel('Proporciones', color='black')
    ax.set_xlabel('Estampa de Tiempo', color='black')
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')

    # Save the plot to the specified path
    plt.savefig(f"{path}/stacked_time_series_plot_of_tph_proportions_for_constancia_and_pampacancha.png")

    # Close the plot to prevent it from showing
    plt.close()

def create_stacked_time_series_plots_proportion_of_other_ores(data, path):

    # Optional preprocessing
    suffix = '_mean' if 'ORE_K_mean' in data.columns else ''
    ORE_K = data[f'ORE_K{suffix}'].rolling(window=336, min_periods=1).mean()
    ORE_H = data[f'ORE_H{suffix}'].rolling(window=336, min_periods=1).mean()
    HIZN = data[f'HIZN{suffix}'].rolling(window=336, min_periods=1).mean()
    STK = data[f'STK{suffix}'].rolling(window=336, min_periods=1).mean()
    OTHER = data[f'OTHER{suffix}'] + data[f'ORE_S{suffix}'] + data[f'ORE_M{suffix}']
    OTHER = OTHER.rolling(window=336, min_periods=1).mean()
    ORE_TOTAL = ORE_K + ORE_H + HIZN + STK + OTHER
    UPLIFT = 1/ORE_TOTAL
    ORE_K = ORE_K * UPLIFT
    ORE_H = ORE_H * UPLIFT
    HIZN = HIZN * UPLIFT
    STK = STK * UPLIFT
    OTHER = OTHER * UPLIFT

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    #suffix = '_mean' if 'ORE_K_mean' in data.columns else ''
    ax.stackplot(data['estampa_de_tiempo'], 
                 ORE_K,
                 ORE_H,  
                 HIZN,
                 STK,
                 OTHER,
                 labels=['proporcion de Ore K', 'proporcion de Ore H', 'proporcion de HIZN', 'proporcion de STK', 'proporcion de OTHER'], 
                 colors=['#66b3ff', '#ff9999', '#99ff99', '#ffcc99', '#c2c2f0'])   

    ax.legend(loc='upper left', facecolor='white', framealpha=1)
    ax.set_title('Series de Tiempo Apiladas: Proporciones de Tonelaje', color='black')
    ax.set_ylabel('Proporciones', color='black')
    ax.set_xlabel('Estampa de Tiempo', color='black')
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')

    # Save the plot to the specified path
    plt.savefig(f"{path}/stacked_time_series_plot_of_tph_proportions_for_other_ores.png")

    # Close the plot to prevent it from showing
    plt.close()

def create_binned_box_and_whiskers_plot(data, path):    
    # Binning the data
    suffix = '_mean' if 'flujo_masa_t_h_mean' in data.columns else ''
    bin_size = 1
    data['binned'] = pd.cut(data[f'_perc_solidos{suffix}'], bins=np.arange(0, data[f'_perc_solidos{suffix}'].max() + bin_size, bin_size), labels=np.arange(0, data[f'_perc_solidos{suffix}'].max(), bin_size))

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Creating box and whisker plot without showing outliers and only median bar
    boxplot = data.boxplot(column=f'flujo_masa_t_h{suffix}', by='binned', ax=ax, grid=False, patch_artist=True, 
                           boxprops=dict(facecolor='#1f77b4', color='#1f77b4'), flierprops=dict(marker=''), 
                           medianprops=dict(color='#ff7f0e', linewidth=2))

    ax.set_title('Diagrama de Cajas y Bigotes Agrupado', color='black')
    ax.set_ylabel('Flujo de Masa (t/h)', color='black')
    ax.set_xlabel('Porcentaje de Sólidos (Binned)', color='black')
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    plt.suptitle('')  # Suppress the automatic title to keep only the custom title

    # Set x-axis limits
    ax.set_xlim(data['binned'].min(), data['binned'].max() + 2)

    # Save the plot to the specified path
    plt.savefig(f"{path}/binned_box_and_whiskers_plot_of_perc_solidos_and_flujo_masa.png")

    # Close the plot to prevent it from showing
    plt.close()

    data.drop(columns=['binned'], inplace=True)

def create_binned_box_and_whiskers_plot_of_tonnage_bias(data, path):

    if 'tonelaje_delta_t_h_mean' in data.columns:

        # Binning the data
        bin_size = 100
        data['binned'] = pd.cut(
            data['linea12_tonelaje_t_h_mean'],
            bins=np.arange(0, data['linea12_tonelaje_t_h_mean'].max() + bin_size, bin_size),
            labels=np.arange(0, data['linea12_tonelaje_t_h_mean'].max(), bin_size)
        )

        #data[['binned', 'tonelaje_delta_t_h_mean']]

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        # Creating box and whisker plot without showing outliers and only median bar
        data.boxplot(column='tonelaje_delta_t_h_mean', by='binned', ax=ax, grid=False, patch_artist=True, 
                     boxprops=dict(facecolor='#1f77b4', color='#1f77b4'), flierprops=dict(marker=''), 
                     medianprops=dict(color='#ff7f0e', linewidth=2))

        ax.set_title('Diagrama de Cajas y Bigotes Agrupado', color='black')
        ax.set_ylabel('Tonelaje de Linea 1 y 2 minus Flujo Masa (t/h)', color='black')
        ax.set_xlabel('Tonelaje de Linea 1 y 2 (t/h) (Binned)', color='black')
        ax.tick_params(axis='x', colors='black')
        ax.tick_params(axis='y', colors='black')
        plt.suptitle('')  # Suppress the automatic title to keep only the custom title

        # Set x-axis limits
        ax.set_xlim(30.5, 45.5)
        #ax.set_xlim(data['linea12_tonelaje_t_h_mean'].min(), data['linea12_tonelaje_t_h_mean'].max())

        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')

        # Save the plot to the specified path
        plt.savefig(f"{path}/binned_box_and_whiskers_plot_of_tonnage_bias.png")

        # Close the plot to prevent it from showing
        plt.close()

        data.drop(columns=['binned'], inplace=True)

def create_7_day_time_series(data, path):

    if 'linea12_tonelaje_t_h_mean' in data.columns:

        # Filter data for the last 7 days
        last_days = data['estampa_de_tiempo'].max() - pd.Timedelta(days=10)
        data = data[(data['estampa_de_tiempo'] >= last_days) & (data['estampa_de_tiempo'] < last_days + pd.Timedelta(days=7))]

        # Plotting
        fig, ax1 = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('white')
        ax1.set_facecolor('white')

        ax2 = ax1.twinx()

        ax1.plot(data['estampa_de_tiempo'], data['3000_planta_procesos_3290_relaves_5511_pu060_perc_speed_mean'], label='3000 Planta Procesos 3290 Relaves 5511 PU060 Perc Speed', color='#ff7f0e')
        ax2.plot(data['estampa_de_tiempo'], data['linea12_tonelaje_t_h_mean'], label='Linea 12 Tonelaje t/h', color='#1f77b4')

        ax1.set_xlabel('Estampa de Tiempo', color='black')
        ax1.set_ylabel('PU060 Perc Speed', color='#ff7f0e')
        ax2.set_ylabel('Linea 12 Tonelaje t/h', color='#1f77b4')

        ax1.tick_params(axis='x', colors='black')
        ax1.tick_params(axis='y', colors='#ff7f0e')
        ax2.tick_params(axis='y', colors='#1f77b4')

        #fig.legend(loc='upper left', facecolor='white', framealpha=1)
        ax1.set_title('Serie de Tiempo de PU060 Speed y Linea12 Tonelaje', color='black')

        # Save the plot to the specified path
        plt.savefig(f"{path}/time_series_plot_for_PU060_Speed_y_Tonelaje.png")

        # Close the plot to prevent it from showing
        plt.close()

    else:

        # Filter data for the last 7 days
        last_days = data['estampa_de_tiempo'].max() - pd.Timedelta(days=10)
        data = data[(data['estampa_de_tiempo'] >= last_days) & (data['estampa_de_tiempo'] < last_days + pd.Timedelta(days=7))]
        data['linea12_tonelaje_t_h'] = data['linea1_tonelaje_t_h'] + data['linea2_tonelaje_t_h']

        # Plotting
        fig, ax1 = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('white')
        ax1.set_facecolor('white')

        ax2 = ax1.twinx()

        ax1.plot(data['estampa_de_tiempo'], data['3000_planta_procesos_3290_relaves_5511_pu060_perc_speed'], label='3000 Planta Procesos 3290 Relaves 5511 PU060 Perc Speed', color='#ff7f0e')
        ax2.plot(data['estampa_de_tiempo'], data['linea12_tonelaje_t_h'], label='Linea 12 Tonelaje t/h', color='#1f77b4')

        ax1.set_xlabel('Estampa de Tiempo', color='black')
        ax1.set_ylabel('PU060 Perc Speed', color='#ff7f0e')
        ax2.set_ylabel('Linea 12 Tonelaje t/h', color='#1f77b4')

        ax1.tick_params(axis='x', colors='black')
        ax1.tick_params(axis='y', colors='#ff7f0e')
        ax2.tick_params(axis='y', colors='#1f77b4')

        #fig.legend(loc='upper left', facecolor='white', framealpha=1)
        ax1.set_title('Serie de Tiempo de PU060 Speed y Linea12 Tonelaje', color='black')

        # Save the plot to the specified path
        plt.savefig(f"{path}/time_series_plot_for_PU060_Speed_y_Tonelaje.png")

        # Close the plot to prevent it from showing
        plt.close()

def create_7_day_time_series_2(data, path):

    if 'linea12_tonelaje_t_h_mean' in data.columns:

        # Identify periods where _perc_solidos_mean was lower than 54 for more than 4 hours
        threshold = 54
        min_duration = 4 * 2  # 4 hours in half-hour increments
        below_threshold = data['_perc_solidos_mean'] < threshold
        below_threshold_periods = below_threshold.rolling(window=min_duration, min_periods=min_duration).sum() >= min_duration
        below_threshold_periods = below_threshold_periods & below_threshold

        # Extract start and end times of these periods
        periods = []
        in_period = False
        for i in range(len(below_threshold_periods)):
            if below_threshold_periods.iloc[i] and not in_period:
                start_time = data['estampa_de_tiempo'].iloc[i]
                in_period = True
            elif not below_threshold_periods.iloc[i] and in_period:
                end_time = data['estampa_de_tiempo'].iloc[i]
                periods.append((start_time, end_time))
                in_period = False
        if in_period:
            periods.append((start_time, data['estampa_de_tiempo'].iloc[-1]))

        # # Print the periods
        # for start, end in periods:
        #     print(f"Period from {start} to {end} where _perc_solidos_mean was below {threshold}")

        # Create a sub-folder for these plots
        sub_folder = os.path.join(path, 'suboptimal_control_plots')
        os.makedirs(sub_folder, exist_ok=True)

        for i, (start, end) in enumerate(periods):
            start = start - pd.Timedelta(hours=48)
            end = end + pd.Timedelta(hours=48)
            data_filtered = data[(data['estampa_de_tiempo'] >= start) & (data['estampa_de_tiempo'] <= end)]

            # Plotting
            fig, ax1 = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor('white')
            ax1.set_facecolor('white')

            ax2 = ax1.twinx()

            ax1.plot(data_filtered['estampa_de_tiempo'], data_filtered['_perc_solidos_mean'], label='%solidos', color='#A9A9A9', linestyle='--')
            ax1.plot(data_filtered['estampa_de_tiempo'], data_filtered['bed_mass_perc_mean'], label='Bed mass %', color='#696969')
            ax2.plot(data_filtered['estampa_de_tiempo'], data_filtered['flujo_descarga_m3_h_mean'], label='Flujo descarga m3/h', color='#A9A9A9')

            # Adding legends
            ax1.legend(loc='upper left', facecolor='white', framealpha=1)
            ax2.legend(loc='upper right', facecolor='white', framealpha=1)
            ax1.set_xlabel('Estampa de Tiempo', color='black')

            # Save the plot to the specified path
            plt.savefig(f"{sub_folder}/time_series_plot_for_suboptimal_control_{i+1}.png")

            # Close the plot to prevent it from showing
            plt.close()
