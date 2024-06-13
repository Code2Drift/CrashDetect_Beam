import pandas as pd
from copy import deepcopy
from PreProcess.signal_process import butter_LPF_Viz
import os

def csv_read2(path):
    df_ori = pd.read_csv(path)
    df_copy = deepcopy(df_ori)

    if len(df_ori["YOLO_ID"].value_counts().index) == 2:
        # Filter for car 1
        v1 = df_copy[df_copy["YOLO_ID"] == 1]
        filtered_v1 = butter_LPF_Viz(v1, cutoff=2, fs=60, order=1)

        # Filter for car 2
        v2 = df_copy[df_copy["YOLO_ID"] == 2]
        filtered_v2 = butter_LPF_Viz(v2, cutoff=2, fs=60, order=1)

        df_fil_list = [filtered_v1, filtered_v2]

    elif len(df_ori["YOLO_ID"].value_counts().index) > 2:
        if df_ori[df_ori["YOLO_ID"] == 3].shape[0] > 10:
            print("Check video response")
        else:
            # Filter for car 1
            v1 = df_copy[df_copy["YOLO_ID"] == 1]
            filtered_v1 = butter_LPF_Viz(v1, cutoff=2, fs=60, order=1)

            # Filter for car 2
            v2 = df_copy[df_copy["YOLO_ID"] == 2]
            filtered_v2 = butter_LPF_Viz(v2, cutoff=2, fs=60, order=1)

            df_fil_list = [filtered_v1, filtered_v2]

    else:
        print("Check video response")

    return df_fil_list


def process_files(path):
    # Initialize an empty list to store results
    results = []

    # Iterate over each file in the folder
    for filename in os.listdir(path):
        if filename.endswith('csv'):
            file_path = os.path.join(path, filename)
            df_list = csv_read2(file_path)  # Get the nested list of DataFrames

            # Iterate over each DataFrame in the list
            for idx, ID_df in enumerate(df_list):
                # Calculate the 50th and 75th quantiles for each column
                quantiles = ID_df.drop(columns=['YOLO_ID']).quantile([0.5, 0.75])

                # Create a dictionary to store the results
                result = {'file': filename, 'object_id': idx + 1}
                for column in ID_df.columns:
                    if column != 'YOLO_ID':
                        result[f'{column}_50'] = quantiles.loc[0.5, column]
                        result[f'{column}_75'] = quantiles.loc[0.75, column]

                # Append the result to the list
                results.append(result)

    # Create a final DataFrame from the results
    final_df = pd.DataFrame(results)

    # Round the numerical values
    final_df = final_df.round(0)

    # Select columns that start with 'Bin' or 'object'
    select_col = [col for col in final_df.columns if col.startswith('Bin') or col.startswith('object')]
    final_dfdf = final_df[select_col]

    return final_dfdf