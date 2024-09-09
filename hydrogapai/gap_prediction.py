import pandas as pd
from .gap_utils import process_file
from datetime import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates

import random
import numpy as np

RANDOM_STATE = 42


def predict_station_gaps(
    input_file_path: str, results_folder: str, model_type="rf", hyper_opt=False
):
    """
    Predicts station gaps based on the input file path and model type.
    Parameters:
        input_file_path (str): The file path of the input file.
        model_type (str, optional): The type of model to use for prediction. Defaults to "rf".
    Returns:
        The result of the process_file function.
    """

    np.random.RandomState().seed(RANDOM_STATE)
    random.seed(RANDOM_STATE)

    # current_datetime = datetime.now().strftime("%Y%m%d_%H%M")

    # results_folder = os.path.join(
    #    os.path.dirname(input_file_path), f"results_{current_datetime}"
    # )
    os.makedirs(results_folder, exist_ok=True)

    # Create a results directory with the current date and time

    return process_file(
        input_file_path=input_file_path,
        results_folder=results_folder,
        model_type=model_type,
        hyper_opt=hyper_opt,
    )


def plot_fully(
    results_folder,
    val_full,
    real_predictions,
    input_file_name,
    model_type,
    plot_start_date,
    plot_end_date,
    rotation=45,
    tick_interval="auto",
):
    plt.figure(figsize=(15, 6))

    # Filter the data based on the provided date range
    if plot_start_date and plot_end_date:
        val_full = val_full.loc[plot_start_date:plot_end_date]
        real_predictions = real_predictions.loc[plot_start_date:plot_end_date]

    # Create a new series for real_predictions aligned with val_full
    # Initialize a series with NaNs
    aligned_predictions = pd.Series(index=val_full.index, data=np.nan)
    # Fill the aligned_predictions where val_full['obsdis'] is NaN (i.e., in the gaps)
    aligned_predictions[val_full["obsdis"].isna()] = real_predictions
    # Plot predicted values with a red line (synthetically hidden)
    plt.plot(
        val_full.index.values,
        val_full["predicted"].values,
        color="red",
        label="Predicted (synth gaps only)",
    )
    # Plot actual values with a black line
    plt.plot(
        val_full.index.values,
        val_full["obsdis"].values,
        color="black",
        label="Actual Values (with original gaps)",
    )
    # Plot the aligned real predictions with a green line, showing predictions only for the gaps
    plt.plot(
        aligned_predictions.index.values,
        aligned_predictions.values,
        color="green",
        label="Predicted (real gaps only)",
    )
    plt.xlabel("Date", fontsize=15)
    if tick_interval == "auto":
        # Automatically adjust tick locations and labels to avoid overlap
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    else:
        # Custom interval for tick locations (e.g., every 2 months)
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=tick_interval))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.ylabel("Discharge", fontsize=15)
    plt.legend(fontsize=12)
    plt.title(
        f"Actual vs Predicted Values - ({model_type.upper()})\n{input_file_name}",
        fontsize=20,
    )
    # Rotate x-axis labels and automatically adjust them to prevent overlap
    plt.xticks(rotation=rotation)
    plt.gcf().autofmt_xdate()
    # Save the plot as an image file in the results folder
    plot_file_name = os.path.join(
        results_folder,
        f"real_vs_pred_full_{model_type}_{os.path.splitext(os.path.basename(input_file_name))[0]}.png",
    )
    plt.savefig(plot_file_name, dpi=300)
    plt.close()


def plot_yearly(
    results_folder,
    val_full,
    real_predictions,
    input_file_name,
    model_type,
    rotation=45,
    tick_interval="auto",
):
    # Create a subfolder for the current file inside the results folder
    file_folder_name = os.path.splitext(input_file_name)[
        0
    ]  # Get the base filename (without extension)
    file_specific_folder = os.path.join(
        results_folder, file_folder_name
    )  # Create the path for the file-specific folder
    os.makedirs(
        file_specific_folder, exist_ok=True
    )  # Create the folder if it doesn't exist

    # Group the data by year
    val_full["year"] = val_full.index.year
    grouped_by_year = val_full.groupby("year")

    for year, group in grouped_by_year:
        plt.figure(figsize=(15, 6))

        # Align real_predictions with the year group and reindex to handle missing dates
        aligned_predictions = pd.Series(index=group.index, data=np.nan)
        common_index = group.index.intersection(
            real_predictions.index
        )  # Find common dates between group and real_predictions
        aligned_predictions.loc[common_index] = real_predictions.loc[
            common_index
        ]  # Only assign where the indices match

        # Plot predicted values with a red line
        plt.plot(
            group.index.values,
            group["predicted"].values,
            color="red",
            label="Predicted Values (synth gaps only)",
        )
        # Plot actual values with a black line
        plt.plot(
            group.index.values,
            group["obsdis"].values,
            color="black",
            label="Actual Values (with original gaps)",
        )
        # Plot the aligned real predictions with a green line, showing predictions only for the real gaps
        plt.plot(
            aligned_predictions.index.values,
            aligned_predictions.values,
            color="green",
            label="Predicted (real gaps only)",
        )

        plt.xlabel("Date", fontsize=15)

        if tick_interval == "auto":
            # Automatically adjust tick locations and labels to avoid overlap
            plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        else:
            # Custom interval for tick locations (e.g., every 2 months)
            plt.gca().xaxis.set_major_locator(
                mdates.MonthLocator(interval=tick_interval)
            )

        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.ylabel("Discharge", fontsize=15)
        plt.legend(fontsize=12)
        plt.title(
            f"Actual vs Predicted Values - {year} ({model_type.upper()})\n{input_file_name}",
            fontsize=20,
        )

        # Rotate x-axis labels and automatically adjust them to prevent overlap
        plt.xticks(rotation=rotation)
        plt.gcf().autofmt_xdate()

        # Save the plot for the specific year in the file-specific folder
        plot_file_name = os.path.join(
            file_specific_folder, f"real_vs_pred_{year}_{model_type}.png"
        )
        plt.savefig(plot_file_name, dpi=300)
        plt.close()
