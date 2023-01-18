from torch.utils.data import DataLoader, TensorDataset

# from mlops_finalproject.models import model
import torch
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
from PIL import Image
import numpy as np
from torchvision import transforms
from mlops_finalproject.models import model
from pytorch_lightning import Callback, Trainer
from torchvision import transforms, datasets



def normalize_data(data):
    # Normalize
    for idx, image in enumerate(data):
        mean = image.mean().item()
        std = image.std().item()

        transform_norm = transforms.Compose([transforms.Normalize(mean, std)])
        img_normalized = transform_norm(image)
        data[idx] = img_normalized

    return data

def get_data(path: str) -> list:
    """ Funtion to load the folders with the imgs to predict
    """
    test = pd.read_csv(path + '/Test.csv')
    paths = test["Path"].values
    test_labels = test["ClassId"].values

    test_imgs = []
    transform = transforms.Compose(
        [
        transforms.Resize([32, 32]),
        #transforms.RandomHorizontalFlip(),
        #transforms.RandomRotation(10),
        transforms.ToTensor(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    for img_path in paths:
        fullpath = path + "/" + img_path
        img = Image.open(fullpath)

        tensor = transform(img) # Resize and convert to tensor
        test_imgs.append(tensor)

    output = [
        torch.stack(test_imgs),
        test_labels,
    ]
    # breakpoint()

    return output


@click.command()
@click.argument("model_checkpoint", type=click.Path(exists=True))
# @click.argument("data_path", type=click.Path(exists=True))
@click.option("-n", "--num-images", required=False, type=int)
def main(model_checkpoint, num_images: int):
    """
    Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """

    logger = logging.getLogger(__name__)
    logger.info("Using model in inference")
    print(model_checkpoint)

    mymodel = model.MobileNetV3Lightning()
    mymodel.load_state_dict(torch.load(model_checkpoint))

    images, labels = get_data("data/raw/German")
    # images = normalize_data(images)
    # breakpoint()

    test_dataset = TensorDataset(images, torch.from_numpy(labels))  # create your datset
    testloader = DataLoader(
        test_dataset, batch_size=64, num_workers=8
    )  # create your dataloader


    images = torch.load("data/processed/images.pt")
    labels = torch.load("data/processed/labels.pt")
    traindataset = TensorDataset(images, labels)
    trainloader = DataLoader(traindataset, batch_size=64, shuffle=False, num_workers=8)

    trainer = Trainer()
    trainer.test(mymodel, trainloader)



if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
