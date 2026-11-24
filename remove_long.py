# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 15:02:27 2024

@author: Karim
"""

import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('D:/Phd/Part 2-1/combined_kalimat_data.csv')

# Function to filter out long sentences
def filter_long_sentences(sentence):
    return len(sentence) <= 1800

# Apply the filter to the DataFrame
df = df[df['input'].apply(filter_long_sentences)]

# If you want to reset the index after dropping rows
df.reset_index(drop=True, inplace=True)

# Save the filtered DataFrame to a new CSV file if needed
df.to_csv('filtered_kalimat_(8282).csv', index=False)
