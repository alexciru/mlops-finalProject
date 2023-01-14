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
from tqdm import tqdm


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath: str, output_filepath:str):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

   # Train files -  organized in folders from labels
    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    train_dataset = datasets.ImageFolder(input_filepath, transform=transform)

    # test files -  labels in csv
    # test_info = pd.read_csv(input_filepath + '/Test.csv')
    # test_labels = test_info["ClassId"].values
    # imgs = test_info["Path"].values

    train_imgs = []
    train_labels = []
    for img, label in tqdm(train_dataset):
        #flatten_tensor = torch.flatten(img, start_dim=0)
        train_imgs.append(img)  #flatten_tensor)
        train_labels.append(label)

    # # Join all of the data
    # final_imgs = []
    # final_imgs = train_dataset.imgs
    # final_imgs = final_imgs.append(test_imgs)

    # final_labels = []
    # final_labels = train_dataset.classes
    # final_labels.append(test_labels)
    train_images = torch.stack(train_imgs)
    train_labels = torch.Tensor(train_dataset.targets)

    # Store data
    torch.save(train_images, f"{output_filepath}/images.pt")
    torch.save(train_labels, f"{output_filepath}/labels.pt")
    logger.info('data store successfully')
    #torch.save(final_imgs, f"{output_filepath}/images.pt")
    #torch.save(final_labels, f"{output_filepath}/labels.pt")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

