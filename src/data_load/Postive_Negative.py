import os
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from unidecode import unidecode

# Set Pandas option
pd.set_option("mode.copy_on_write", True)


def patch_filename(f):
    """Patch filename to correct the filenames."""
    f = (
        f.replace("ã¨", "├г┬и")
        .replace("ã­", "├г┬н")
        .replace("ã©", "├г┬й")
        .replace("ã³", "├г┬│")
    )
    f = f.replace("ã¶", "├г┬╢").replace("ã¼", "├г┬╝").replace("â\xa0", "├в┬а")
    return f


# Load datasets
description = pd.read_csv("data/classes.csv")
artemis = pd.read_csv("data/artemis_dataset_release_v0.csv")

# Patch filenames
description["filename"] = description["filename"].map(patch_filename)
artemis["filename"] = artemis["art_style"] + "/" + artemis["painting"] + ".jpg"
artemis["filename"] = artemis["filename"].map(patch_filename)

# Merge datasets
merged_df = pd.merge(artemis, description, on="filename", how="left")

# Define emotion map
emotion_map = {
    "contentment": 1,
    "awe": 1,
    "amusement": 1,
    "excitement": 1,
    "sadness": 0,
    "fear": 0,
    "disgust": 0,
    "anger": 0,
    "something else": 2,
}

# Apply emotion_map to the emotion column and create a column called emotion_sum
merged_df["emotion_sum"] = merged_df["emotion"].map(emotion_map)

# Filter data based on emotion_sum
filtered_data_1 = merged_df[merged_df["emotion_sum"] == 1.0]
filtered_data_0 = merged_df[merged_df["emotion_sum"] == 0.0]

# Extract unique values from the 'filename' column
unique_filenames_positive = filtered_data_1["filename"].unique().tolist()
unique_filenames_negative = filtered_data_0["filename"].unique().tolist()

# Find non-overlapping filenames
non_overlap_list1 = [
    item for item in unique_filenames_positive if item not in unique_filenames_negative
]
non_overlap_list2 = [
    item for item in unique_filenames_negative if item not in unique_filenames_positive
]

# Sample from non-overlapping filenames
random_samples_positive = random.sample(non_overlap_list1, 6200)

# Set output path
output_path = Path("archive/positive_genre_unique_random/")

# List to store names of files that don't exist
missing_files = []

for img_path in random_samples_positive:
    abs_img_path = Path("dataset_path") / img_path
    if abs_img_path.exists():
        try:
            img = Image.open(abs_img_path)
            shutil.copy(
                abs_img_path, output_path / unidecode(img_path.replace("/", "_"))
            )
            # plt.imshow(img)
            # plt.xticks([])
            # plt.yticks([])
            # plt.show()
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
