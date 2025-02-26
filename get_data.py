"""
Download sudoku data from its original source and/or create subsets of it
dataset link: https://www.kaggle.com/datasets/rohanrao/sudoku
Madelyn Redick
February 26, 2025
"""

# uncomment this segment and run if you need to download the data from its original source
"""import kagglehub

# Download latest version
path = kagglehub.dataset_download("rohanrao/sudoku")

print("Path to dataset files:", path)"""

# uncomment the function calls to create subsets of the original data
# make sure you are in the same directory as this project when creating subsets, or adjust the path of files
import random
import pandas as pd

def create_subset(input_file, output_file, num_rows):
    """
    Selects random number of rows from a CSV file and exports them to a new CSV file

    Arguments:
        input_file (str): Path to the input CSV file
        output_file (str): Path to the output CSV file
        num_rows (int): Number of random rows to select

    Returns:
        None
    """
    # read input file as pandas dataframe
    df = pd.read_csv(input_file)

    # make sure num_rows is less than length of original dataset
    if len(df) < num_rows:
        raise ValueError("Number of rows to create subset with exceeds number of rows in original dataset")

    # get random number rows from df
    random_rows = df.sample(num_rows)

    # export random rows to new CSV file
    random_rows.to_csv(output_file, index=False)


# these function calls use the full dataset to obtain a subset from, but the full dataset could not be included 
# in the github repository because it is too large (1407.62 MB)
# create subset of 500 rows
#create_subset("sudoku.csv", "sudoku_500.csv", 500)

# create subset of 10000 rows
#create_subset("sudoku.csv", "sudoku_10000.csv", 10000)