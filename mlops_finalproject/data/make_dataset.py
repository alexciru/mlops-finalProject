# -*- coding: utf-8 -*-
import click
import logging
import numpy as np
from torchvision import transforms, datasets
import torch
import os
import pandas as pd
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import random


def get_sample(images_tensor, labels, n, size):
    N = images_tensor.shape[0]

    # Get a random set of indices
    indices = random.sample(range(N), n*n)

    # Get the corresponding images
    images = images_tensor[indices]
    labels = labels[indices]

    # Create a figure and axes
    fig, axs = plt.subplots(n, n, figsize=(12, 12))

    # Loop through the axes and plot each image
    for i in range(n):
        for j in range(n):
            axs[i, j].imshow(np.transpose(images[i*n+j], (1, 2, 0)))
            axs[i, j].axis('off')
            axs[i, j].set_title(labels[i*n+j].item())

    # Display the plot
    # plt.show()
    fig.savefig(f"reports/figures/processed_samples/proc_samples_{size}.png")


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
@click.option("-n", "--num-images", required=False, type=int)
@click.option("-s", "--size", required=False, type=int)
def main(input_filepath: str, output_filepath: str, num_images: int, size=32):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    # Train files -  organized in folders from labels
    transform = transforms.Compose(
        [
            transforms.Resize([size, size]),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(10),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_dataset = datasets.ImageFolder(input_filepath, transform=transform)

    if num_images is None:
        num_images = len(train_dataset)

    indices = range(0, num_images)
    subset = Subset(train_dataset, indices)
    train_loader = DataLoader(subset, batch_size=1, shuffle=False)

    train_imgs = []
    train_labels = []
    for img, label in tqdm(train_loader):
        # breakpoint()
        mean = img.mean().item()
        std = img.std().item()

        # transform_norm = transforms.Compose([transforms.Normalize(mean, std)])
        # img_normalized = transform_norm(img)
        img_normalized = img

        train_imgs.append(img_normalized[0])
        train_labels.append(label.item())


        # Plot img
        # img2 = img.view(3,size,size)
        # plt.imshow(img2.permute(1, 2, 0))
        # plt.show()

    train_labels = np.array(train_labels)
    train_labels = torch.from_numpy(train_labels).long()

    train_images = torch.stack(train_imgs)
    # breakpoint()

    # Save plot of random images
    get_sample(train_images, train_labels, 6, size)

    if not os.path.exists(f"{output_filepath}/{size}"):
        os.makedirs(f"{output_filepath}/{size}")

    # Store data
    torch.save(train_images, f"{output_filepath}/{size}/images.pt")
    torch.save(train_labels, f"{output_filepath}/{size}/labels.pt")
    logger.info("data store successfully")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
