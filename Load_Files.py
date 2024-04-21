# Code chunk for reading and preprocessing images
# Author: Juan David Rico Molano
# Date: Feb 2024
# Source: https://www.kaggle.com/code/juandaviddev/color-palette

import os
import random
import cv2
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

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


description = pd.read_csv("data/classes.csv")

description["filename"] = description["filename"].map(patch_filename)


class ImageReader:
    """Class to read and preprocess images.

    Attributes:
        - n_images (int): Number of images to read
        - file_type (str): File type of the images
        - directory (str): Directory where the images are stored

    Preprocessing includes:
    - Reading the images
    - Rescaling the images keeping the aspect ratio
    """

    def __init__(
        self, n_images: int, file_type: str, directory: str, scale: float = 0.1
    ):
        self.images_names = []
        self.n_images = n_images
        self.file_type = file_type
        self.directory = directory
        self.images = self.read_images()
        self.images = self.rescale_images(self.images, scale)

    def read_images(self):
        """Read images from a directory and store them in a list"""
        files = []
        for dirname, _, filenames in os.walk(self.directory):
            if len(filenames) > self.n_images:
                chosen_files = random.sample(filenames, self.n_images)
            else:
                chosen_files = filenames

            for filename in chosen_files:
                if filename.endswith(self.file_type):
                    files.append(os.path.join(dirname, filename))

        images = []
        files = random.sample(files, self.n_images)
        for file in files:
            image = cv2.imread(file)
            images.append(image)
            self.images_names.append(file.split("/")[-1])
        return images

    def rescale_images(self, images: list, scale: float = 0.5):
        """Rescale images keeping the aspect ratio"""
        rescaled_images = []
        for image in images:
            width = int(image.shape[1] * scale)
            height = int(image.shape[0] * scale)
            dim = (width, height)
            rescaled_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
            rescaled_images.append(rescaled_image)
        return rescaled_images

    def plot_images(self, title: str = "Images"):
        """Plot the images"""
        n_cols = 3
        n_rows = int(np.ceil(self.n_images / n_cols))
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 15))
        for i, ax in enumerate(axs.flat):
            if i < self.n_images:
                ax.imshow(cv2.cvtColor(self.images[i], cv2.COLOR_BGR2RGB))
                ax.axis("off")
                ax.set_title(self.images_names[i])
        # set title, label, xticks, yticks and legend
        plt.suptitle(title, fontsize=20)
        plt.tight_layout()
        plt.show()


image_positive = ImageReader(
    n_images=3, file_type=".jpg", directory="archive/positive_report"
)

print("The images evoke a positive sentiment:")
image_positive.plot_images()
