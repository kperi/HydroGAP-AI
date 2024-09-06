import pandas as pd
from .gap_utils import process_file
from datetime import datetime
import os


def predict_station_gaps(
    input_file_path: str,
    model_type="rf",
):
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M")

    results_folder = os.path.join(
        os.path.dirname(input_file_path), f"results_{current_datetime}"
    )
    os.makedirs(results_folder, exist_ok=True)

    # Create a results directory with the current date and time

    return process_file(
        input_file_path=input_file_path,
        results_folder=results_folder,
        model_type=model_type,
    )
