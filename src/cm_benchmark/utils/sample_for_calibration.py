import os

# import matplotlib.patches as patches
# import matplotlib.pyplot as plt
# from PIL import Image
# import seaborn as sns
import pandas as pd
# import numpy as np

def calibrate_sample(csv_input_path):
    """
    Calibrate a sample for calibration.
    """
    df = pd.read_csv(csv_input_path)
    df_calibrated = df.copy()
    df_calibrated = df_calibrated.dropna(subset=['cmin', 'rmin', 'cmax', 'rmax'])
    df_calibrated = df_calibrated[df_calibrated['displaced'] == False]
    return df_calibrated

def export_calibrated_sample(df_calibrated, output_path):
    """
    Export a calibrated sample to a CSV file.
    """
    df_calibrated.to_csv(output_path, index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_input_path", help="Path to the sample CSV file")
    parser.add_argument("--csv_output_path", help="Path to the output CSV file")
    args = parser.parse_args()
    df_calibrated = calibrate_sample(args.csv_input_path)
    export_calibrated_sample(df_calibrated, args.csv_output_path)


