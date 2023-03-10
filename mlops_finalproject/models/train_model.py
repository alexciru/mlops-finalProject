import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import MobileNetV3Lightning
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import wandb
import random
import os
from tqdm import tqdm
from pytorch_lightning import Callback, Trainer
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from datetime import datetime
from google.cloud import storage
import pickle
import requests
import platform
import hydra


class MyDataset(Dataset):
    def __init__(self, train, path):

        if train:
            self.images_path = os.path.join(path, "data", "processed", "train", "images.pt")
            self.labels_path = os.path.join(path, "data", "processed", "train", "labels.pt")

        else:
            self.images_path = os.path.join(path, "data", "processed", "test", "images.pt")
            self.labels_path = os.path.join(path, "data", "processed", "test", "labels.pt")

        self.images = torch.load(self.images_path)
        self.labels = torch.load(self.labels_path)
        self.labels = self.labels.type(torch.LongTensor)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def check_gcp():
    try:
        response = requests.get(
            "http://metadata.google.internal/computeMetadata/v1/project/project-id",
            headers={"Metadata-Flavor": "Google"},
        )
        return response.status_code == 200
    except:
        return False


class MetricTracker(Callback):
    def __init__(self):
        self.training_losses = []
        self.validation_accuracies = []

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        elogs = trainer.logged_metrics
        self.training_losses.append(elogs["train_loss"])

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        elogs = trainer.logged_metrics
        self.validation_accuracies.append(elogs["val_accuracy"])


@hydra.main(config_path=os.path.join(os.getcwd(), "models/"), config_name="config.yaml")
def main(cfg):

    if check_gcp():
        num_workers = 8
        print("Training on cloud")
    else:
        num_workers = 0
        print("Training on premises")

    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    train_dataset = MyDataset(True, root_dir)
    trainloader = DataLoader(
        train_dataset,
        batch_size=cfg.hyperparameters.batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    test_dataset = MyDataset(False, root_dir)
    testloader = DataLoader(
        test_dataset,
        batch_size=cfg.hyperparameters.batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    metrics = MetricTracker()
    model = MobileNetV3Lightning(learn_rate=cfg.hyperparameters.learning_rate, num_classes=43)

    trainer = Trainer(
        max_epochs=9,
        callbacks=[metrics],
        logger=WandbLogger(
            name=hydra.utils.get_original_cwd(),
            project="mlops_finalProject",
            config={
                "architecture": "CNN",
                "dataset": "GTSRB",
                "epochs": cfg.hyperparameters.num_epochs,
                "batch size": cfg.hyperparameters.batch_size,
                "learning_rate": cfg.hyperparameters.learning_rate,
            },
        ),
    )
    wandb.watch(model, log_freq=100)

    trainer.fit(model, trainloader, val_dataloaders=testloader)
    torch.save(model.state_dict(), "models/trained_model_timm_lightning.pt")

    plt.plot(range(len(metrics.training_losses)), metrics.training_losses)
    # Add a title and axis labels
    plt.title("Training Loss vs Training Steps")
    plt.xlabel("Training Steps")
    plt.ylabel("Training Loss")
    plt.savefig("reports/figures/loss_64img.png")
    plt.close()

    plt.plot(range(len(metrics.validation_accuracies)), metrics.validation_accuracies)
    plt.title("validation accuracies vs Epochs Steps")
    plt.xlabel("Epoch")
    plt.ylabel("Validation accuracy")
    plt.savefig("reports/figures/val_acc_64img.png")

    # Save the plot
    # save weights locally
    timestamp = datetime.today().strftime("%Y%m%d_%H%M")
    name = f"inference_model_32img_{timestamp}.pt"

    torch.save(model.state_dict(), os.path.join(root_dir, "models/", name))

    script = model.to_torchscript()
    torch.jit.save(script, os.path.join(root_dir, "models/model_for_inference.pt"))
    script.save(os.path.join("models/model_for_inference2.pt"))

    storage_client = storage.Client()
    bucket = storage_client.get_bucket("training-bucket-mlops")
    blob = bucket.blob(name)
    blob.upload_from_filename(os.path.join(root_dir, "models/", name))
    print(f"Succesfully push the weights {name} into: {bucket}")


if __name__ == "__main__":
    main()
