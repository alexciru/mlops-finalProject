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

#Hyperparameters
num_epochs = 4
learning_rate = 0.001        

wandb.init(
    # set the wandb project where this run will be logged
    project="mlops_finalProject",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": learning_rate,
    "architecture": "CNN",
    "dataset": "GTSRB",
    "epochs": num_epochs,
    }
)


class MyDataset(Dataset):
    def __init__(self, train, path):

        if train:
            self.images_path = os.path.join(path, "train", "images.pt")
            self.labels_path = os.path.join(path, "train", "labels.pt")

        else:
            self.images_path = os.path.join(path, "test", "images.pt")
            self.labels_path = os.path.join(path, "test", "labels.pt")

        self.images = torch.load(self.images_path)
        self.labels = torch.load(self.labels_path)
        self.labels = self.labels.type(torch.LongTensor)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# my own version of dataset loading
train_dataset = MyDataset(True, 'data/processed')
print(len(train_dataset))
trainloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# test loader
test_dataset = MyDataset(False, 'data/processed')
print(len(test_dataset))
testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

#Creating a instance of the model
model = MobileNetV3Lightning(num_classes=43)

#init the wandb logging
wandb.watch(model, log_freq=100)

# Train the model
losses = []
steps = 0
accu = 0
for epoch in range(num_epochs):
    print(f"epoch: {epoch+1}/{num_epochs}")
    running_loss = 0
    for (inputs, labels) in tqdm(trainloader):
        loss_, preds_ = model.training_step(inputs, labels)
        running_loss += loss_.item()

    # Optional: Decrease lr
    # model.optimizer.param_groups[0]['lr'] *= 60

    wandb.log({"Train loss": running_loss/len(trainloader)})
    losses.append(running_loss/len(trainloader))
    steps += 1
    print(f"Training loss: {running_loss/len(trainloader)}")   
    # # Print the accuracy
    train_acc = model.test_model(trainloader)
    test_acc = model.test_model(testloader)
    wandb.log({"Train Acc": train_acc})
    wandb.log({"Test Acc": test_acc})
    print('Accuracy of the model on the train images: {} %'.format(train_acc))
    print('Accuracy of the model on the test images: {} %'.format(test_acc))
    if accu < test_acc:
        accu = test_acc
        torch.save(model.state_dict(), 'models/trained_model_64img.pt')

# Use the plot function to draw a line plot
plt.plot(range(steps), losses)

# Add a title and axis labels
plt.title("Training Loss vs Training Steps")
plt.xlabel("Training Steps")
plt.ylabel("Training Loss")

# Save the plot
plt.savefig("reports/figures/loss_64img.png")