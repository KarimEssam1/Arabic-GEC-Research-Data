# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 15:10:52 2024

@author: Karim
"""


import os
import csv

def words_to_sentence(words):
    # Constructing the sentence
    sentence = ' '.join(words)
    # Add a period at the end if the last character is not already a punctuation mark
    if sentence[-1] not in ['.', '!', '?']:
        sentence += '.'
    return sentence

def process_folder(folder_path):
    # List all files in the folder
    files = os.listdir(folder_path)
    output_csv = os.path.join(folder_path, 'sentences.csv')
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        for file_name in files:
            # Check if the file is a text file
            if file_name.endswith('.txt'):
                file_path = os.path.join(folder_path, file_name)
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        words = file.readlines()
                        # Removing newline characters and extra spaces
                        words = [word.strip() for word in words]
                        sentence = words_to_sentence(words)
                        csv_writer.writerow([file_name, sentence])
                except FileNotFoundError:
                    print(f"File {file_name} not found. Skipping.")

# Example usage:
#folder_path = '/path/to/your/folder'  # Change '/path/to/your/folder' to the folder path you want to process
#process_folder(folder_path)


# Example usage:
#folder_path = 'D:/Phd/Part 2-1/articlesEconomy'  # Change '/path/to/your/folder' to the folder path you want to process
folder_path=['D:/Phd/Part 2-1/articlesCulture','D:/Phd/Part 2-1/articlesEconomy','D:/Phd/Part 2-1/articlesInternational','D:/Phd/Part 2-1/articlesLocal','D:/Phd/Part 2-1/articlesReligion','D:/Phd/Part 2-1/articlesSports']
#process_folder(folder_path)

for x in range (len(folder_path)):
    process_folder(folder_path[x])