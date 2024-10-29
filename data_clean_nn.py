#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 18:03:29 2024

@author: trekz1
"""
import pandas as pd
import numpy as np

import re



df = pd.read_csv('/Users/trekz1/Documents/Applied Data Sci/Coursework/screenTime/maps-synthetic-data-v1.1.csv')

yes_no_columns = [column for column in df.columns if df[column].isin(['Yes', 'No']).any()]  # find all columns with yes/no

def convert_yes_no(value):   # convert Yes and No values to binary indicators
    if value == 'Yes':
        return 1
    elif value == 'No':
        return 0
    else:
        return value

df1 = df.applymap(convert_yes_no)  # apply function to df



def convert_to_interval(value):  # convert all interval descriptions to numerical values
    if value == 'Less than 1 hour':
        return pd.Interval(left=0, right=1, closed='left')
    elif value == '< 1 hour':
        return pd.Interval(left=0, right=1, closed='left')
    elif value == '1-2 hours':
        return pd.Interval(left=1, right=3, closed='right')
    elif value == '3 or more hours':
        return pd.Interval(left=3, right=float('inf'), closed='right')
    elif value == 'Not at all':
        return pd.Interval(left=0, right=0, closed='both')
    elif value == '1 or more hours':
        return pd.Interval(left=1, right=float('inf'), closed='right')
    elif value == 'Less than 3 hours':
        return pd.Interval(left=0, right=3, closed='left') 
    else:
        return value  # Return the value unchanged if it does not match any pattern

# df1 = df1.applymap(convert_to_interval)
    
'''
Check for other < or > signs:
'''   
# Pattern to match strings that contain '<' or '>' optionally followed by spaces, and then any number
pattern = re.compile(r'[<>]\s*\d+')

# Create a dictionary to store unique matches for each column index
unique_matches_in_columns = {}

# Iterate over each column in the DataFrame
for col_index, column in enumerate(df.columns):
    # Only process columns with data type 'object' (string)
    if df[column].dtype == 'object':
        # Initialize a set to store unique matches for the current column
        matches_set = set()
        # Apply the pattern search row-wise and collect all unique matches found in each column
        df[column].apply(lambda x: matches_set.update(re.findall(pattern, str(x))))
        
        # If any matches were found, store them for the current column
        if matches_set:
            unique_matches_in_columns[col_index] = matches_set

# Display the results
for col_index, matches in unique_matches_in_columns.items():
    print(f"Column {col_index}: contains unique strings like {', '.join(matches)}")


# Columns identified as containing interval-like descriptions
interval_columns = ['talk_phon_wend', 'text_wend', 'talk_mob_wend', 'comp_wend', 'musi_wend',
    'read_wend', 'work_wend', 'alon_wend', 'draw_wend', 'play_wend',
    'tv_wend', 'out_win_wend', 'out_sum_wend', 'tran_wend', 'talk_phon_week',
    'text_week', 'talk_mob_week', 'comp_week', 'musi_week', 'read_week',
    'work_week', 'alon_week', 'draw_week', 'play_week', 'tv_week',
    'out_win_week', 'out_sum_week', 'tran_week', 'phone_14_wend', 'phone_14_week']

for column in interval_columns:  #  apply interval function to all columns that need it
    df1[column] = df1[column].apply(convert_to_interval)  

percentage_nans = df1.isna().mean() * 100



def transform_yes_no(df, column_indices):
    """
    Transforms 'Yes' and 'No' values in specified columns into 1 and 0.
    Throws an error if a column contains values other than 'Yes' or 'No'.

    Parameters:
    - df: pandas DataFrame.
    - column_indices: List of column indices to transform.

    Returns:
    - Transformed DataFrame.
    """
    for index in column_indices:
        column_name = df.columns[index]
        unique_values = df[column_name].unique()
        
        # Check if the column contains only 'Yes' and 'No'
        if not all(value in ['Yes', 'No', np.nan] for value in unique_values):  # Allowing NaN values
            raise ValueError(f"Column '{column_name}' contains values other than 'Yes' or 'No'.")
        
        # Transform 'Yes' and 'No' to 1 and 0
        df[column_name] = df[column_name].map({'Yes': 1, 'No': 0})
    
    return df



def transform_not_any_at_all(df, column_indices):
    """
    Transforms 'Not at all' values to 0 and 'Any at all' to 1 in specified columns.
    Throws an error if a column contains values other than these options.

    Parameters:
    - df: pandas DataFrame.
    - column_indices: List of column indices to transform.

    Returns:
    - Transformed DataFrame.
    """
    for index in column_indices:
        column_name = df.columns[index]
        unique_values = df[column_name].unique()
        
        if not all(value in ['Not at all', 'Any at all', np.nan] for value in unique_values):  # Allowing NaN values
            raise ValueError(f"Column '{column_name}' contains values other than 'Not at all' or 'Any at all'.")
        
        df[column_name] = df[column_name].map({'Not at all': 0, 'Any at all': 1})
    
    return df




def transform_time_set1(df, column_indices):
    expected_values = {"Not at all", "< 1 hour", "Less than 1 hour", "1 or more hours", None}  # Including None for potential NaN values

    for index in column_indices:
        column_name = df.columns[index]
        unique_values = set(df[column_name].dropna().unique())  # Drop NaN values for this check

        if not unique_values.issubset(expected_values):
            raise ValueError(f"Column '{column_name}' at index {index} contains unexpected values. "
                             f"Unique values are: {unique_values}")

        df[column_name] = df[column_name].map({
            "Not at all": 0,
            "< 1 hour": 0.5,
            "Less than 1 hour": 0.5,
            "1 or more hours": 1
        })
    return df



def transform_time_set2(df, column_indices):
    expected_values = {"Not at all", "Less than 1 hour", "1-2 hours", "3 or more hours", None}  # Including None for potential NaN values

    for index in column_indices:
        column_name = df.columns[index]
        unique_values = set(df[column_name].dropna().unique())  # Drop NaN values for this check

        if not unique_values.issubset(expected_values):
            raise ValueError(f"Column '{column_name}' at index {index} contains unexpected values. "
                             f"Unique values are: {unique_values}")

        df[column_name] = df[column_name].map({
            "Not at all": 0,
            "Less than 1 hour": 0.5,
            "1-2 hours": 1.5,
            "3 or more hours": 3
        })
    return df



def transform_time_set3(df, column_indices):

    expected_values = {"3 or more hours", "Less than 3 hours", None}  # Including None for potential NaN values

    for index in column_indices:
        column_name = df.columns[index]
        unique_values = set(df[column_name].dropna().unique())  # Drop NaN values for this check

        if not unique_values.issubset(expected_values):
            raise ValueError(f"Column '{column_name}' at index {index} contains unexpected values. "
                             f"Unique values are: {unique_values}")

        df[column_name] = df[column_name].map({
            "Less than 3 hours": 1.5,
            "3 or more hours": 3
        })
    return df



def transform_education_level(df, column_indices):
    """
    Transforms education level values into numerical values suitable for machine learning algorithms.
    Includes a check for unexpected values in the specified columns.

    Parameters:
    - df: pandas DataFrame.
    - column_indices: List of column indices to transform.

    Returns:
    - Transformed DataFrame.
    """
    expected_values = {'CSE/None', 'cse', 'O level', 'A level', 'Vocational', 'Degree', None}  # Including None for potential NaN values

    for index in column_indices:
        column_name = df.columns[index]
        unique_values = set(df[column_name].dropna().unique())  # Drop NaN values for this check

        if not unique_values.issubset(expected_values):
            raise ValueError(f"Column '{column_name}' at index {index} contains unexpected values. "
                             f"Unique values are: {unique_values}")

        df[column_name] = df[column_name].map({
            'CSE/None': 0,
            'cse': 0,
            'O level': 1,
            'A level': 2,
            'Vocational': 2,  # Consider adjusting based on the specific context
            'Degree': 3
        })
    return df


def transform_occupation_class(df, column_indices):
    """
    Transforms values corresponding to education levels/occupational classes into numerical values.
    Includes a check for unexpected values in the specified columns.

    Parameters:
    - df: pandas DataFrame.
    - column_indices: List of column indices to transform.

    Returns:
    - Transformed DataFrame.
    """
    expected_values = {'i', 'ii', 'III (non-manual)', 'III (manual)', 'iv', 'v', 'Armed forces', None}  # Including None for potential NaN values

    for index in column_indices:
        column_name = df.columns[index]
        unique_values = set(df[column_name].dropna().unique())  # Drop NaN values for this check

        if not unique_values.issubset(expected_values):
            raise ValueError(f"Column '{column_name}' at index {index} contains unexpected values. "
                             f"Unique values are: {unique_values}")

        df[column_name] = df[column_name].map({
            'i': 6,
            'ii': 5,
            'iii (non-manual)': 4,
            'iii (manual)': 3,
            'iv': 2,
            'v': 1,
            'Armed forces': 4  # Adjust as needed based on the specific context
        })
    return df



def transform_depression_anxiety_levels(df, column_indices):
    """
    Transforms values corresponding to depression or anxiety levels into numerical values.
    Includes a check for unexpected values in the specified columns.

    Parameters:
    - df: pandas DataFrame.
    - column_indices: List of column indices to transform.

    Returns:
    - Transformed DataFrame.
    """
    expected_values = {'~0.5%', '~15%', '~3%', '>70%', '<0.1%', '~50%', None}  # Including None for potential NaN values

    for index in column_indices:
        column_name = df.columns[index]
        unique_values = set(df[column_name].dropna().unique())  # Drop NaN values for this check

        if not unique_values.issubset(expected_values):
            raise ValueError(f"Column '{column_name}' at index {index} contains unexpected values. "
                             f"Unique values are: {unique_values}")

        df[column_name] = df[column_name].map({
            '<0.1%': 0.05,
            '~0.5%': 0.5,
            '~3%': 3,
            '~15%': 15,
            '~50%': 50,
            '>70%': 85
        })
    return df



def transform_frequency(df, column_indices):
    """
    Transforms frequency values into numerical values suitable for machine learning algorithms.
    Includes a check for unexpected values in the specified columns.

    Parameters:
    - df: pandas DataFrame.
    - column_indices: List of column indices to transform.

    Returns:
    - Transformed DataFrame.
    """
    expected_values = {'5 or more times a week', '1-4 times a week', '1-3 times a month', 'Less than once a month', 'Never', None}  # Including None for potential NaN values

    for index in column_indices:
        column_name = df.columns[index]
        unique_values = set(df[column_name].dropna().unique())  # Drop NaN values for this check

        if not unique_values.issubset(expected_values):
            raise ValueError(f"Column '{column_name}' at index {index} contains unexpected values. "
                             f"Unique values are: {unique_values}")

        df[column_name] = df[column_name].map({
            'Never': 0,
            'Less than once a month': 0.5,
            '1-3 times a month': 2.5,
            '1-4 times a week': 10,
            '5 or more times a week': 20
        })
    return df


def transform_depression_diagnosis(df, column_indices):
    """
    Transforms frequency values into numerical values suitable for machine learning algorithms.
    Includes a check for unexpected values in the specified columns.

    Parameters:
    - df: pandas DataFrame.
    - column_indices: List of column indices to transform.

    Returns:
    - Transformed DataFrame.
    """
    expected_values = {' No ICD-10 diagnosis of depression', 'Yes ICD-10 diagnosis of depression', None}  # Including None for potential NaN values

    for index in column_indices:
        column_name = df.columns[index]
        unique_values = set(df[column_name].dropna().unique())  # Drop NaN values for this check

        if not unique_values.issubset(expected_values):
            raise ValueError(f"Column '{column_name}' at index {index} contains unexpected values. "
                             f"Unique values are: {unique_values}")

        df[column_name] = df[column_name].map({
            ' No ICD-10 diagnosis of depression': 0,
            'Yes ICD-10 diagnosis of depression': 1
        })
    return df



def transform_creative_activities(df, column_indices):
    expected_values = {"sometimes", "often", None}  # Including None for potential NaN values

    for index in column_indices:
        column_name = df.columns[index]
        unique_values = set(df[column_name].dropna().unique())  # Drop NaN values for this check

        if not unique_values.issubset(expected_values):
            raise ValueError(f"Column '{column_name}' at index {index} contains unexpected values. "
                             f"Unique values are: {unique_values}")

        df[column_name] = df[column_name].map({
            "sometimes": 0,
            "often": 1
        })
    return df


def transform_frequency_tv(df, column_indices):
    """
    Transforms frequency values into numerical values suitable for machine learning algorithms.
    Includes a check for unexpected values in the specified columns.

    Parameters:
    - df: pandas DataFrame.
    - column_indices: List of column indices to transform.

    Returns:
    - Transformed DataFrame.
    """
    expected_values = {'Other', 'Yes, Some Days', 'Yes, Every Day', None}  # Including None for potential NaN values

    for index in column_indices:
        column_name = df.columns[index]
        unique_values = set(df[column_name].dropna().unique())  # Drop NaN values for this check

        if not unique_values.issubset(expected_values):
            raise ValueError(f"Column '{column_name}' at index {index} contains unexpected values. "
                             f"Unique values are: {unique_values}")

        df[column_name] = df[column_name].map({
            'Other': 0,
            'Yes, Some Days': 1,
            'Yes, Every Day': 2
        })
    return df


def transform_sex(df, column_indices):
    """
    Transforms frequency values into numerical values suitable for machine learning algorithms.
    Includes a check for unexpected values in the specified columns.

    Parameters:
    - df: pandas DataFrame.
    - column_indices: List of column indices to transform.

    Returns:
    - Transformed DataFrame.
    """
    expected_values = {'Male', 'Female', None}  # Including None for potential NaN values

    for index in column_indices:
        column_name = df.columns[index]
        unique_values = set(df[column_name].dropna().unique())  # Drop NaN values for this check

        if not unique_values.issubset(expected_values):
            raise ValueError(f"Column '{column_name}' at index {index} contains unexpected values. "
                             f"Unique values are: {unique_values}")

        df[column_name] = df[column_name].map({
            'Male': 0,
            'Female': 1
        })
    return df



def transform_birth_order(df, column_indices):
    """
    Transforms frequency values into numerical values suitable for machine learning algorithms.
    Includes a check for unexpected values in the specified columns.

    Parameters:
    - df: pandas DataFrame.
    - column_indices: List of column indices to transform.

    Returns:
    - Transformed DataFrame.
    """
    expected_values = {'A', 'B', None}  # Including None for potential NaN values

    for index in column_indices:
        column_name = df.columns[index]
        unique_values = set(df[column_name].dropna().unique())  # Drop NaN values for this check

        if not unique_values.issubset(expected_values):
            raise ValueError(f"Column '{column_name}' at index {index} contains unexpected values. "
                             f"Unique values are: {unique_values}")

        df[column_name] = df[column_name].map({
            'A': 1,
            'B': 2
        })
    return df


def transform_to_bmi(df):
    """
    Transforms 'weight_16' and 'height_16' into 'bmi_16' values in the DataFrame,
    and inserts the 'bmi_16' column immediately after 'weight_16'.
    BMI is calculated as weight in kilograms divided by height in meters squared.

    Parameters:
    - df: pandas DataFrame with 'weight_16' and 'height_16' columns.

    Returns:
    - Transformed DataFrame with a new 'bmi_16' column inserted after 'weight_16'.
    """
    # Check for non-numeric values in 'weight_16' and 'height_16'
    if not np.issubdtype(df['weight_16'].dtype, np.number) or not np.issubdtype(df['height_16'].dtype, np.number):
        raise ValueError("Non-numeric values found in 'weight_16' or 'height_16' columns.")

    # Convert height from cm to m
    height_m = df['height_16'] / 100

    # Calculate BMI
    bmi_16_values = df['weight_16'] / (height_m ** 2)

    # Find the index of 'weight_16' column
    weight_16_index = df.columns.get_loc('weight_16')

    # Insert 'bmi_16' column immediately after 'weight_16' column
    df.insert(loc=weight_16_index + 1, column='bmi_16', value=bmi_16_values)

    # Optional: drop the original 'weight_16' and 'height_16' columns
    # df = df.drop(columns=['weight_16', 'height_16'])

    return df


def transform_mat_age(df, column_indices):
    """
    Transforms values in specified columns (by indices) into numerical values based on predefined rules.
    The function handles specific non-numeric categories and includes a check for unexpected values.

    Parameters:
    - df: pandas DataFrame.
    - column_indices: List of indices of the columns to transform.

    Returns:
    - Transformed DataFrame with numerical values in specified columns.
    """
    for column_index in column_indices:
        # Ensure the column_index is within the DataFrame's column range
        if column_index >= len(df.columns):
            raise ValueError(f"Column index {column_index} is out of range for the DataFrame.")

        # Get the column name using its index
        column_name = df.columns[column_index]

        # Define a mapping for the transformation, e.g., for 'mat_age' column
        value_mapping = {'< 16': 15, '>43': 44}  # Example mapping; replace with actual transformation rules

        # Apply the transformation
        df[column_name] = df[column_name].replace(value_mapping).astype(float)

        # Check for unexpected values after transformation
        unique_values = df[column_name].dropna().unique()
        # Define the expected set of values after transformation; adjust as per your transformation rules
        expected_values = set(range(15, 45))  # Example expected values; adjust as needed

        if not set(unique_values).issubset(expected_values):
            raise ValueError(f"Column '{column_name}' at index {column_index} contains unexpected values after transformation. Unique values are: {unique_values}")

    return df



def transform_num_home(df, column_indices):
    """
    Transforms home-related values in specified columns (by indices) into numerical values suitable for machine learning.
    Handles '9 or more' by assigning it a numerical value while ensuring all other values are converted to their numeric equivalents.

    Parameters:
    - df: pandas DataFrame.
    - column_indices: List of indices of the columns to transform.

    Returns:
    - Transformed DataFrame with numerical values in specified columns.
    """
    for column_index in column_indices:
        # Ensure the column_index is within the DataFrame's column range
        if column_index >= len(df.columns):
            raise ValueError(f"Column index {column_index} is out of range for the DataFrame.")

        # Get the column name using its index
        column_name = df.columns[column_index]

        # Define a mapping for the '9 or more' category, if necessary
        value_mapping = {'9 or more': 10}

        # Apply the mapping and convert all other values to numeric, coercing errors to NaN
        df[column_name] = df[column_name].replace(value_mapping).apply(pd.to_numeric, errors='coerce')

        # Check for unexpected values after transformation (excluding NaN)
        unique_values = df[column_name].dropna().unique()
        if not all(isinstance(x, (int, float)) for x in unique_values):
            raise ValueError(f"Column '{column_name}' at index {column_index} still contains non-numeric values after transformation. Unique values are: {unique_values}")

    return df



def print_non_numerical_values(df):
    """
    Goes through all columns in a DataFrame and checks for non-numerical values.
    For columns with non-numerical values, prints all unique values.

    Parameters:
    - df: pandas DataFrame.
    """
    for column in df.columns:
        # Check if column is non-numeric
        if not pd.api.types.is_numeric_dtype(df[column]):
            # Get unique values excluding NaN
            unique_values = df[column].dropna().unique()
            print(f"Column '{column}' contains non-numerical values. Unique values are: {unique_values}")


def transform_nan_to_0(df, column_indices):
    """
    Fills NaN values with 0 in specified columns of a DataFrame.

    Parameters:
    - df: pandas.DataFrame
    - column_indices: list of int, indices of the columns to fill NaNs with 0

    Returns:
    - Modified DataFrame with NaNs replaced by 0 in the specified columns.
    """

    # Get column names based on indices
    columns_to_fill = df.columns[column_indices]

    # Fill NaNs with 0 in specified columns
    df[columns_to_fill] = df[columns_to_fill].fillna(0)
    
    return df


# ----------------------------------------------------------------------------------------------------


def main():
    # Load the dataset
    df = pd.read_csv('/Users/trekz1/Documents/Applied Data Sci/Coursework/screenTime/maps-synthetic-data-v1.1.csv')
    
    nan_to_0_indices = [10,11,77,78]
    
    # Specify the indices of columns for Yes/No transformation
    yes_no_column_indices = [3, 9, 10, 39, 40, 41, 43, 44, 45, 47, 48, 49, 64, 67, 68, 69, 76, 77, 79]  # Update these indices based on your specific columns
    
    # Specify the indices of columns for Not at all/Any at all transformation
    not_any_column_indices = [11, 15, 25, 29]  # Update these indices based on your specific columns

    # Specify the indices of columns for "Not at all", "1 or more hours" and "< 1 hour" transformation
    time_set1_column_indices = [13, 16, 19, 24, 27, 30, 33, 38, 65, 66]
    
    # Specify the indices of columns for "Not at all", "Less than 1 hour", "1-2 hours" and "3 or more hours" transformation
    time_set2_column_indices = [12, 14, 17, 18, 21, 22, 26, 28, 31, 32, 35, 36]
    
    # Specify the indices of columns for "3 or more hours", "Less than 3 hours" transformation
    time_set3_column_indices = [20, 23, 34, 37]
    
    education_level_column_indices = [52, 53]
    
    occupation_class_column_indices = [50, 51]
    
    depression_anxiety_levels_column_indices = [55, 56, 57, 58, 59, 60, 61, 62]
    
    frequency_column_indices = [63]
    
    depression_diagnosis_column_indices = [70]
    
    creative_activities_column_indices = [78]
    
    frequency_tv_column_indices = [80, 81, 82]
    
    sex_column_indices = [83]
    
    birth_order_column_indices = [84]
    
    mat_age_column_indices = [5]
    
    num_home_column_indices = [42]
    
    
    try:
                
        transformed_df = transform_yes_no(df, yes_no_column_indices)
        
        transformed_df = transform_not_any_at_all(transformed_df, not_any_column_indices)
        
        transformed_df = transform_time_set1(transformed_df, time_set1_column_indices)
        
        transformed_df = transform_time_set2(transformed_df, time_set2_column_indices)
        
        transformed_df = transform_time_set3(transformed_df, time_set3_column_indices)
        
        transformed_df = transform_education_level(transformed_df, education_level_column_indices)
        
        transformed_df = transform_occupation_class(transformed_df, occupation_class_column_indices)
        
        transformed_df = transform_depression_anxiety_levels(transformed_df, depression_anxiety_levels_column_indices)
        
        transformed_df = transform_frequency(transformed_df, frequency_column_indices)
        
        transformed_df = transform_depression_diagnosis(transformed_df, depression_diagnosis_column_indices)
        
        transformed_df = transform_creative_activities(transformed_df, creative_activities_column_indices)
        
        transformed_df = transform_frequency_tv(transformed_df, frequency_tv_column_indices)
        
        transformed_df = transform_sex(transformed_df, sex_column_indices)
        
        transformed_df = transform_birth_order(transformed_df, birth_order_column_indices)
        
        transformed_df = transform_mat_age(transformed_df, mat_age_column_indices)
        
        transformed_df = transform_num_home(transformed_df, num_home_column_indices)
        
        transformed_df = transform_to_bmi(transformed_df)
        
        transformed_df = transform_nan_to_0(df, nan_to_0_indices)
        
        # Drop columns flag and weight_16
        transformed_df = transformed_df.drop(columns=['Unnamed: 0', 'X', 'flag', 'weight_16'])
        
        # Save the transformed dataset to a new CSV file
        transformed_df.to_csv('/Users/trekz1/Documents/Applied Data Sci/Coursework/screenTime/numeric-maps-synthetic-data-v1.1.csv', index=False)
        
        print("Transformation successful. The dataset has been saved.")
        
        print_non_numerical_values(transformed_df)
        
    except ValueError as e:
        print(e)
        
        
    return transformed_df
# Run the main function and display the first few rows of the transformed dataset
transformed_dataset = main()



