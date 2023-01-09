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
    transform = transforms.Compose([transforms.Resize([112, 112]), transforms.ToTensor()])
    train_dataset = datasets.ImageFolder(input_filepath, transform=transform)

    # test files -  labels in csv
    # test_info = pd.read_csv(input_filepath + '/Test.csv')
    # test_labels = test_info["ClassId"].values
    # imgs = test_info["Path"].values

    # test_imgs = []
    # for img in imgs:
    #     path = os.path.join(input_filepath, img)
    #     image = cv2.imread(path)
    #     print(image.shape())
    #     #image_fromarray = Image.fromarray(image, 'RGB') 
    #     #resize_image = image_fromarray.resize((30, 30))
    #     test_imgs.append(np.array(image))


    # # Join all of the data
    # final_imgs = []
    # final_imgs = train_dataset.imgs
    # final_imgs = final_imgs.append(test_imgs)

    # final_labels = []
    # final_labels = train_dataset.classes
    # final_labels.append(test_labels)


    # Store data
    torch.save(train_dataset.imgs, f"{output_filepath}/images.pt")
    torch.save(train_dataset.classes, f"{output_filepath}/labels.pt")

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

