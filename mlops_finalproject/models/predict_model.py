<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
from torch.utils.data import DataLoader, TensorDataset
#from mlops_finalproject.models import model
import torch
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
from PIL import Image
import numpy as np
from torchvision import transforms


def get_data(path: str) -> list:
    """ Funtion to load the folders with the imgs to predict
    """
    test = pd.read_csv(path + '/Test.csv')
    paths = test["Path"].values
    test_labels = test["ClassId"].values

    test_imgs = []
    transform = transforms.Compose([transforms.Resize([30, 30]), transforms.ToTensor()])

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




# @click.command()
# @click.argument("model_checkpoint", type=click.Path(exists=True))
# @click.argument("data_path", type=click.Path(exists=True))
# def main(model_checkpoint, data_path):
#     """Runs data processing scripts to turn raw data from (../raw) into
#     cleaned data ready to be analyzed (saved in ../processed).
#     """
#     logger = logging.getLogger(__name__)
#     logger.info("Using model in inference")
#     print(model_checkpoint)

#     mymodel = model.MyAwesomeModel(784, 10)
#     mymodel.load_state_dict(torch.load(model_checkpoint))

#     images, labels = get_data(data_path)
#     images = normalize_data(images)

#     test_dataset = TensorDataset(images, labels)  # create your datset
#     testloader = DataLoader(
#         test_dataset, batch_size=64
#     )  # create your dataloader

#     with torch.no_grad():
#         for images, labels in testloader:
#             # resize images
#             images = images.view(images.shape[0], -1)

#             mymodel.eval()
#             logps = mymodel.forward(images)
#             ps = torch.exp(logps)

#             # Take max from the probs
#             top_p, top_class = ps.topk(1, dim=1)

#             # Compare with labels
#             equals = top_class == labels.view(*top_class.shape)

#             # mean
#             accuracy = torch.mean(equals.type(torch.FloatTensor))

#     print(f'Accuracy: {accuracy.item()*100}%')


    

# if __name__ == "__main__":
#     log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
#     logging.basicConfig(level=logging.INFO, format=log_fmt)

#     # not used in this stub but often useful for finding various files
#     project_dir = Path(__file__).resolve().parents[2]

#     # find .env automagically by walking up directories until it's found, then
#     # load up the .env entries as environment variables
#     load_dotenv(find_dotenv())

#     main()
=======
import torch
>>>>>>> 8218a7e (files for dockerization)
=======
# import torch # TODO: uncomment this
=======
import torch
>>>>>>> d5fbdcc (Finish Initial docker)

print("Predicting")
>>>>>>> 30a8e1f (Dockerfiles done but I commented all installs because Im flying)
