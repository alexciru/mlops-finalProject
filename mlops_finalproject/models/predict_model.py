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
from torch.utils.data import Dataset
import os


class MyTestDataset(Dataset):
    def __init__(self, train, path):
        
        self.images_path = os.path.join(path, "test", "images.pt")
        self.labels_path = os.path.join(path, "test", "labels.pt")

        # if train:
        #     self.images_path = os.path.join(path, "train", "images.pt")
        #     self.labels_path = os.path.join(path, "train", "labels.pt")
        
        # elif 

        # else:
        #     self.images_path = os.path.join(path, "test", "images.pt")
        #     self.labels_path = os.path.join(path, "test", "labels.pt")

        self.images = torch.load(self.images_path)
        self.labels = torch.load(self.labels_path)
        self.labels = self.labels.type(torch.LongTensor)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


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
    """Funtion to load the folders with the imgs to predict"""
    test = pd.read_csv(path + "/Test.csv")
    paths = test["Path"].values
    test_labels = test["ClassId"].values

    test_imgs = []
    transform = transforms.Compose(
        [
            transforms.Resize([32, 32]),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(10),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    for img_path in paths:
        fullpath = path + "/" + img_path
        img = Image.open(fullpath)

        tensor = transform(img)  # Resize and convert to tensor
        test_imgs.append(tensor)

    output = [
        torch.stack(test_imgs),
        test_labels,
    ]
    # breakpoint()

    return output


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path, image_label = self.data[index]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image, image_label

@click.command()
@click.argument("model_checkpoint", type=click.Path(exists=True))
@click.argument("data_path", type=click.Path(exists=True))
def main(model_checkpoint, data_path):
    """
    Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """

    logger = logging.getLogger(__name__)
    logger.info("Using model in inference")
    print(model_checkpoint)

    mymodel = model.MobileNetV3Lightning(num_classes=43)
    mymodel.load_state_dict(torch.load(model_checkpoint))

    # test loader
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize([32, 32]),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    max_images = 300
    ir = 0
    images = []
    for element in os.listdir(data_path):
        img = Image.open(f"{data_path}/{element}")
        img = transform(img).unsqueeze(0)
        images.append(img)
        # print(element)
        ir += 1
        if ir >= max_images:
            break

    images_tensor = torch.stack(images)
    images_tensor = images_tensor.view(images_tensor.shape[0],3,32,32)
    fake_labels = [i for i in range(images_tensor.shape[0])]
    # breakpoint()
    dataset = TensorDataset(images_tensor, torch.tensor(fake_labels))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    trainer = Trainer()
    preds = trainer.predict(mymodel, dataloaders=dataloader)
    predictions = [p.item() for p in preds]
    print(f"Prediction: {predictions}")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
