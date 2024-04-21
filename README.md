# Classifying-Art-by-Genre-and-Emotion-Evoked

This repository is dedicated to loading, preprocessing, and building models for classifying the emotions evoked by artworks.

### Data Source

This repository utilizes datasets from two primary sources: WikiArt and Artemis.

#### WikiArt
WikiArt provides a meticulously curated collection of 80,020 unique images sourced from WikiArt.org. Processed by Peter Baylies, this dataset encompasses 27 distinct artistic styles and features works from 1,119 artists. With such diversity, it offers a rich resource for analysis and machine learning applications in the realm of visual art.

#### Artemis
Artemis is focused on exploring the intricate relationship between visual art and the emotions it evokes, along with the linguistic justifications behind these emotional responses. Employing human annotators, Artemis identifies the primary emotion elicited by each artwork and provides a verbal explanation for the choice. This dataset comprises 455,000 emotion attributions and explanations, drawing from a diverse set of 80,000 artworks sourced from WikiArt.

#### Download Instructions
- **WikiArt Dataset:**
  - To download the WikiArt dataset, navigate to [this Kaggle link](https://www.kaggle.com/datasets/steubk/wikiart/data).
  - Download the 33.77GB dataset for images and the 12.51MB 'classes.csv' file.
  - Create an 'archive' folder to store the images and a 'data' folder for the 'classes.csv' file for future reference.

- **Artemis Dataset:**
  - To download the Artemis dataset, visit [this link](https://www.artemisdataset.org/#dataset).
  - Fill out the form and proceed to download the dataset.
  - Place the downloaded file into the 'data' folder created above for later use.

These datasets serve as invaluable resources for studying and analyzing emotions evoked by artworks, offering a comprehensive foundation for machine learning and artistic exploration.

### Data Preparation
To effectively utilize the code, follow these steps:

1. **Loading Image Files:** 
   - Start by utilizing `Load_Files.py` to load the image files.
   - Customize the directory for the images by modifying line 106 in the script.

2. **Generating Dataset with Positive and Negative Images:** 
   - Utilize `Positive_Negative.py` to create a dataset containing both positive and negative images.
   - Ensure the existence of a 'data' folder containing two essential files: 'classes.csv' from WikiArt and 'artemis_dataset_release_v0.csv' from Artemis.
   - Create an 'archive' folder and a subfolder named 'positive_genre_unique_random' to store all the positive images before running the code.

3. **Generating Genre Files:** 
   - Use `Genre_Files.py` to generate files for different genres.
   - Ensure the existence of a 'data' folder containing a file named 'classes.csv' from WikiArt.
   - Create an 'archive' folder and a subfolder named 'four_genres'.
   - Within 'four_genres', create four corresponding genre folders.

Following these steps will aid in effectively loading, preprocessing, and organizing the necessary data for classifying the emotions evoked by artworks.
