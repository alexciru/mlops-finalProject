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
from google.cloud import storage
import pickle
import io


@click.command()
@click.argument("model_checkpoint")
@click.argument("data_path", type=click.Path(exists=True))
def main(model_checkpoint, data_path):
    """
    Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """

    logger = logging.getLogger(__name__)
    logger.info("Using model in inference")

    BUCKET_NAME = "training-bucket-mlops"
    MODEL_FILE = model_checkpoint
    print(MODEL_FILE)

    client = storage.Client()
    bucket = client.get_bucket(BUCKET_NAME)

    blob = bucket.get_blob(MODEL_FILE)
    # weights_pt = blob.download_to_filename("models/weights_1.pt")
    weights_pt = blob.download_as_bytes()
    # model = MobileNetV3Lightning(num_classes=43)

    # # mymodel = pickle.loads()
    # model = load_state_dict(torch.load(model_checkpoint))
    # print(mymodel)
    buffer = io.BytesIO()
    buffer.write(weights_pt)
    buffer.seek(0)

    mymodel = model.MobileNetV3Lightning(num_classes=43)
    mymodel.load_state_dict(torch.load(buffer))

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
    images_tensor = images_tensor.view(images_tensor.shape[0], 3, 32, 32)
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
