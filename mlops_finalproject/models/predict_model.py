from torch.utils.data import DataLoader, TensorDataset
import torch
import click
from tqdm import tqdm
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
from PIL import Image
import numpy as np
from torchvision import transforms
import torch.nn as nn
from model import Classifier


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
        transforms.ToTensor(),
        ]
    )
    i = 0
    for img_path in paths:
        
        fullpath = path + "/" + img_path
        img = Image.open(fullpath)
        if i==1:
            print(fullpath)
            print(test_labels[i])

        tensor = transform(img) # Resize and convert to tensor
        test_imgs.append(tensor)
        i += 1

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
    model = Classifier(num_classes=43)
    # Load previously saved model parameters.
    state_dict = torch.load('models/trained_modelV9.pt')
    model.load_state_dict(state_dict)
    criterion = nn.CrossEntropyLoss()
    print("Model loaded succesfully")
       
    with torch.no_grad():
        test_correct = 0 # number of correct predictions
        total_test_loss = 0
        for images, labels in testloader:
            # Normalzie the image.
            mean = images.mean().item()
            std = images.std().item()
            transform_norm = transforms.Compose([transforms.Normalize(mean, std)])
            img_normalized = transform_norm(images)

            model.eval()
            logps = model.forward(img_normalized)
            ps = torch.exp(logps)
            # Take max from the probs
            top_p, top_class = ps.topk(1, dim=1)

            # Compare with labels
            equals = top_class == labels.view(*top_class.shape)

            # mean
            accuracy = torch.mean(equals.type(torch.FloatTensor))

            # new shit
            outputs = model(img_normalized)
            loss = criterion(outputs, labels)
            total_test_loss += loss.item()

            ps = torch.exp(outputs)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            #print(labels)
            test_correct += equals.sum().item()

    print("Model evaluation complete - old.")
    print(f'Accuracy: {accuracy.item()*100}%')
    print("Model evaluation complete - new.")
    print("Accuracy: {:.3f}%".format((test_correct / len(testloader.dataset))*100))
    
if __name__ == "__main__":
     log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
     logging.basicConfig(level=logging.INFO, format=log_fmt)

     # not used in this stub but often useful for finding various files
     project_dir = Path(__file__).resolve().parents[2]

     # find .env automagically by walking up directories until it's found, then
     # load up the .env entries as environment variables
     load_dotenv(find_dotenv())

     main()
