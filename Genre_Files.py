import os
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from IPython import display
from matplotlib import pyplot as plt
from unidecode import unidecode

# Set random seed
random.seed(42)

# Set Pandas option
pd.set_option("mode.copy_on_write", True)


def patch_filename(f):
    """Patch filename to correct the filenames"""
    f = (
        f.replace("ã¨", "├г┬и")
        .replace("ã­", "├г┬н")
        .replace("ã©", "├г┬й")
        .replace("ã³", "├г┬│")
    )
    f = f.replace("ã¶", "├г┬╢").replace("ã¼", "├г┬╝").replace("â\xa0", "├в┬а")
    return f


def categorize_genre(genres):
    """Generate big genre based on sub-genres"""
    if any(
        genre in genres
        for genre in ["Minimalism", "Pop Art", "Pointillism", "Symbolism"]
    ):
        return 1
    elif any(
        genre in genres
        for genre in [
            "Impressionism",
            "Expressionism",
            "Post-Impressionism",
            "Surrealism",
            "Abstract Expressionism",
            "Cubism",
            "Pop Art",
            "Abstract Art",
            "Art Informel",
            "Color Field Painting",
            "Neo-Expressionism",
            "Magic Realism",
            "Lyrical Abstraction",
            "Art Nouveau Modern",
            "Fauvism",
            "Action Painting",
            "Naive Art Primitivism",
            "Ukiyo",
            "Colour Field Painting",
        ]
    ):
        return 2
    elif any(
        genre in genres
        for genre in ["Realism", "Romanticism", "Baroque", "Neoclassicism", "Rococo"]
    ):
        return 3
    elif "Renaissance" in genres:
        return 4
    else:
        return None


# Load the dataset
description = pd.read_csv("data/classes.csv")

# Correct the file name
description["filename"] = description["filename"].map(patch_filename)

# Add a new column 'big_genre' to the DataFrame
description["big_genre"] = description["genre"].apply(categorize_genre)

# Filter data based on big genres
filtered_genre_1 = description[description["big_genre"] == 1.0]
filtered_genre_2 = description[description["big_genre"] == 2.0]
filtered_genre_3 = description[description["big_genre"] == 3.0]
filtered_genre_4 = description[description["big_genre"] == 4.0]

# Extract unique values from the 'filename' column
unique_filenames_contemporary = filtered_genre_1["filename"].unique().tolist()
unique_filenames_modern = filtered_genre_2["filename"].unique().tolist()
unique_filenames_post_renaissance = filtered_genre_3["filename"].unique().tolist()
unique_filenames_renaissance = filtered_genre_4["filename"].unique().tolist()

# Sample from each genre
random_samples_modern = random.sample(unique_filenames_modern, 2900)
random_samples_post_renaissance = random.sample(unique_filenames_post_renaissance, 2900)
random_samples_renaissance = random.sample(unique_filenames_renaissance, 2900)

# Set paths
dataset_path = Path("archive")
output_path = Path("archive/four_genres/renaissance_1")

# List to store names of files that don't exist
missing_files = []

# Process and copy images
for img_path in random_samples_renaissance:
    abs_img_path = dataset_path / img_path
    if abs_img_path.exists():
        try:
            img = Image.open(abs_img_path)
            shutil.copy(
                abs_img_path, output_path / unidecode(img_path.replace("/", "_"))
            )
            # display.display(img)
        except Exception as e:
            print(f"Error processing image {abs_img_path}: {e}")
    else:
        missing_files.append(str(abs_img_path))

# Display missing files if any
if missing_files:
    print("The following files were not found:")
    for file_name in missing_files:
        print(file_name)
else:
    print("All files saved in the output directory")
