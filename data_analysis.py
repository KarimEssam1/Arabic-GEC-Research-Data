# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 21:07:00 2025

@author: Karim
"""

import pandas as pd

def analyze_sentences(csv_file, column_name):
    try:
        # Read the CSV file
        data = pd.read_csv(csv_file,header=None)
        data.columns=[column_name]

        # Ensure the column exists
        if column_name not in data.columns:
            raise ValueError(f"Column '{column_name}' not found in the CSV file.")

        # Get the sentences from the specified column
        sentences = data[column_name].astype(str)

        # Calculate sentence lengths
        sentence_lengths = sentences.apply(len)

        # Count sentences with length less than 5
        short_sentences = sentence_lengths[sentence_lengths < 10].count()

        # Count sentences with length greater than 2000
        long_sentences = sentence_lengths[sentence_lengths > 1800].count()

        # Calculate average sentence length
        average_length = sentence_lengths.mean()

        # Report results
        print(f"Number of sentences with length < 5: {short_sentences}")
        print(f"Number of sentences with length > 2000: {long_sentences}")
        print(f"Average sentence length: {average_length:.2f}")

    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
# Replace 'your_file.csv' with the path to your CSV file and 'sentence_column' with the column name containing sentences.
csv_file = 'D:/Phd/Part 2-1/other data/arabic_dalicate_MSA.csv'
column_name = 'sentences'
analyze_sentences(csv_file, column_name)
