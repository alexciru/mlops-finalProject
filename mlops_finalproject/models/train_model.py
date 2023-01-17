import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, random_split
import wandb
from pytorch_lightning.loggers import WandbLogger
from mlops_finalproject.models import model
from pytorch_lightning import Callback, Trainer
from torchvision import transforms
import pandas as pd
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, images_path, labels_path):
        self.images = torch.load(images_path)
        self.labels = torch.load(labels_path)
        self.labels = self.labels.type(torch.LongTensor)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class MetricTracker(Callback):
    def __init__(self):
        self.collection = []

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        elogs = trainer.logged_metrics
        self.collection.append(elogs["train_loss"])


def get_val_data(path: str) -> list:
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
        torch.tensor(test_labels),
    ]

    return output

wandb.init(
    # set the wandb project where this run will be logged
    project="test-project",
    entity="mlops_finalproject",

    # track hyperparameters and run metadata
    # config={
    # "learning_rate": learning_rate,
    # "architecture": "CNN",
    # "dataset": "GTSRB",
    # "epochs": num_epochs,
    # }
)

images = torch.load("data/processed/images.pt")
labels = torch.load("data/processed/labels.pt")
breakpoint()
traindataset = TensorDataset(images, labels)
trainloader = DataLoader(traindataset, batch_size=64, shuffle=False, num_workers=8)

#validation set
val_images, val_labels = get_val_data("data/raw/German")
val_dataset = TensorDataset(val_images, val_labels)  # create your datset
val_loader = DataLoader(
    val_dataset, batch_size=64, num_workers=8
)  # create your dataloader

mn_model = model.MobileNetV3Lightning(43, False)

wandb.watch(mn_model, log_freq=100)

data = torch.randn(64, 3, 32, 32)
output = mn_model(data)

cb = MetricTracker()
trainer = Trainer(
    max_epochs=5,
    callbacks=[cb],
    limit_train_batches=0.2,
    logger=WandbLogger(project="mlops_finalProject"),
)

# for img, labels in trainloader:
#     breakpoint()

trainer.fit(mn_model, train_dataloaders=trainloader)

trainer.test(mn_model,  trainloader)

# Create plot for training loss
losses = [i.item() for i in cb.collection]
steps = [i for i in range(len(losses))]

# Use the plot function to draw a line plot
plt.plot(steps, losses)

# Add a title and axis labels
plt.title("Training Loss vs Training Steps with Ligthning")
plt.xlabel("Training Steps")
plt.ylabel("Training Loss")

# # Save the plot
plt.savefig("reports/figures/lossV1.png")

torch.save(mn_model.state_dict(), "models/trained_modelLightning.pt")
