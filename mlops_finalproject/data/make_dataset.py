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
from torch.utils.data import Subset, DataLoader, Dataset
from tqdm import tqdm
from PIL import Image

class the_dataset(Dataset):
    def __init__(self, train, trans, input_filepath):
        self.input_filepath = input_filepath
        if train:
            self.test_df = pd.read_csv(input_filepath+"/Train.csv")[["ClassId","Path"]]
        else:
            self.test_df = pd.read_csv(input_filepath+"/Test.csv")[["ClassId","Path"]]
        self.trans = trans

    def __len__(self):
        return len(self.test_df)

    def __getitem__(self, idx):
        label = self.test_df.iloc[idx,0]
        image = Image.open(self.input_filepath + "/" + self.test_df.iloc[idx,1])
        image = self.trans(image)
        return image, label

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

    ## Import test labels

    # Train files -  organized in folders from labels
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize([32, 32]),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    ############ Import training set #################
    train_dataset = the_dataset(True, transform, input_filepath)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    train_imgs = []
    train_labels = []
    for img, label in tqdm(train_loader):
        train_imgs.append(img)
        train_labels.append(label.item())

    train_images = torch.stack(train_imgs)
    train_labels = torch.Tensor(torch.Tensor(train_labels)).long()
    # breakpoint()
    train_images = train_images.view(len(train_dataset), 3, 32, 32)

    # Store data
    if not os.path.exists(f"{output_filepath}train/"):
        os.mkdir(f"{output_filepath}train/")
    torch.save(train_images, f"{output_filepath}train/images.pt")
    torch.save(train_labels, f"{output_filepath}train/labels.pt")
    logger.info("train data stored successfully") 

    ############ Import test set #################
    test_dataset = the_dataset(False, transform, input_filepath)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    test_imgs = []
    test_labels = []
    for img, label in tqdm(test_loader):
        test_imgs.append(img)
        test_labels.append(label.item())

    test_images = torch.stack(test_imgs)
    test_labels = torch.Tensor(torch.Tensor(test_labels)).long()
    # breakpoint()
    test_images = test_images.view(len(test_dataset), 3, 32, 32)

    # Store data
    if not os.path.exists(f"{output_filepath}test/"):
        os.mkdir(f"{output_filepath}test/")
    torch.save(test_images, f"{output_filepath}test/images.pt")
    torch.save(test_labels, f"{output_filepath}test/labels.pt")
    logger.info("test data stored successfully")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
