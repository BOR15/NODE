import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
import os

def createfile(filename, features):
    # Create an empty DataFrame with the specified columns
    df = pd.DataFrame(columns=features)

    # Save the DataFrame to a CSV file
    df.to_csv(filename, index=False)

def addlog(csv_file, row_dict):
    # Read the existing CSV file
    df = pd.read_csv(csv_file)

    # Ensure the dictionary keys match the DataFrame columns
    if not all(key in df.columns for key in row_dict.keys()):
        raise ValueError("Dictionary keys must match CSV column names")
    # Convert the dictionary to a DataFrame row
    new_row_df = pd.DataFrame([row_dict])

    # Convert the dictionary to a DataFrame row and append it
    df = pd.concat([df, new_row_df], ignore_index=True)

    # Save the updated DataFrame to the CSV file
    df.to_csv(csv_file, index=False)

def addcolumn(csv_file, name, default):
    # Read the existing CSV file
    df = pd.read_csv(csv_file)

    # Add a new column with the default value
    df[name] = default

    # Save the updated DataFrame to the CSV file
    df.to_csv(csv_file, index=False)
    

def saveplot(plt, foldername):
    df = pd.read_csv('logging/log.csv')
    number = str(len(df))
    filename = f"logging/{foldername}/{number}.png"
    # Save the plot
    plt.savefig(filename)


def id():
    return len(pd.read_csv('logging/log.csv'))


