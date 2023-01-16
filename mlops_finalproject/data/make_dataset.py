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


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
@click.option("-n", "--num-images", required=False, type=int)
def main(input_filepath: str, output_filepath: str, num_images: int):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    # Train files -  organized in folders from labels
    transform = transforms.Compose(
        [
            transforms.Resize([32, 32]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_dataset = datasets.ImageFolder(input_filepath, transform=transform)

    if num_images is None:
        num_images = len(train_dataset)

    indices = range(0, num_images)
    subset = Subset(train_dataset, indices)
    train_loader = DataLoader(subset, batch_size=1, shuffle=True)

    train_imgs = []
    train_labels = []
    it = 0
    for img, label in tqdm(train_loader):
        # flatten_tensor = torch.flatten(img, start_dim=0)
        # breakpoint()
        # img = img.view(3, 224, 224)
        train_imgs.append(img)
        train_labels.append(label)
        it +=1
        if it >= 5000:
            break

    train_images = torch.stack(train_imgs)
    train_labels = torch.Tensor(train_dataset.targets)
    # breakpoint()
    train_images = train_images.view(num_images, 3, 32, 32)

    # Store data
    torch.save(train_images, f"{output_filepath}/images.pt")
    torch.save(train_labels, f"{output_filepath}/labels.pt")
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
