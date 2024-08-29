import pandas as pd

def split_contiguous_blocks(df):
    # Ensure the DataFrame is sorted by the index
    df = df.sort_index()

    # Initialize variables
    blocks = []
    start_idx = None

    # Iterate through the DataFrame to find contiguous blocks where obsdis is not NaN
    for i in range(len(df)):
        if pd.notna(df.iloc[i]["obsdis"]):
            if start_idx is None:
                start_idx = i
        else:
            if start_idx is not None:
                blocks.append(df.iloc[start_idx:i])
                start_idx = None

    # Append the last block if it ends with non-NaN values
    if start_idx is not None:
        blocks.append(df.iloc[start_idx:])

    return blocks

