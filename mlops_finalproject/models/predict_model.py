from torch.utils.data import DataLoader, TensorDataset
import torch
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
from PIL import Image
import numpy as np
from torchvision import transforms
from model import ModifiedMobileNetV3


def get_data(path: str) -> list:
    """ Funtion to load the folders with the imgs to predict
    """
    test = pd.read_csv(path + '/Test.csv')
    paths = test["Path"].values
    test_labels = test["ClassId"].values

    test_imgs = []
    transform = transforms.Compose([
        transforms.Resize([32, 32]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    for img_path in paths:
        fullpath = path + "/" + img_path
        img = Image.open(fullpath)
        # normalized
        tensor = transform(img) # Resize and convert to tensor
        test_imgs.append(tensor)

    output = [
        torch.stack(test_imgs),
        torch.tensor(test_labels),
    ]

    return output


def main():
#     """Runs data processing scripts to turn raw data from (../raw) into
#     cleaned data ready to be analyzed (saved in ../processed).
#     """
    images, labels = get_data("data/raw/German")
    test_dataset = TensorDataset(images, labels)
    testloader = DataLoader(test_dataset, batch_size=64)
    print("testloader worked")

    # Init the model.
    model = ModifiedMobileNetV3(num_classes=43)
    # Load previously saved model parameters.
    state_dict = torch.load('models/trained_modelV2.pt')
    model.load_state_dict(state_dict)
    print("Model loaded succesfully")

    
    with torch.no_grad():
         for images, labels in testloader:

             model.eval()
             logps = model.forward(images)
             ps = torch.exp(logps)
             # Take max from the probs
             top_p, top_class = ps.topk(1, dim=1)

             # Compare with labels
             equals = top_class == labels.view(*top_class.shape)

             # mean
             accuracy = torch.mean(equals.type(torch.FloatTensor))


    print("Model evaluation complete.")
    print(f'Accuracy: {accuracy.item()*100}%')
    
if __name__ == "__main__":
     log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
     logging.basicConfig(level=logging.INFO, format=log_fmt)

     # not used in this stub but often useful for finding various files
     project_dir = Path(__file__).resolve().parents[2]

     # find .env automagically by walking up directories until it's found, then
     # load up the .env entries as environment variables
     load_dotenv(find_dotenv())

     main()
