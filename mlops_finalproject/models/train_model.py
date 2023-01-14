# Import necessary libraries
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from model import ModifiedMobileNetV3
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import wandb
import random

class MyDataset(Dataset):
    def __init__(self, images_path, labels_path):
        self.images = torch.load(images_path)
        self.labels = torch.load(labels_path)
        self.labels = self.labels.type(torch.LongTensor)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

#Hyperparameters
num_epochs = 2
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

# my own version of dataset loading
dataset = MyDataset('data/processed/images.pt', 'data/processed/labels.pt')
trainloader = DataLoader(dataset, batch_size=128, shuffle=True)

#Creating a instance of the model
model = ModifiedMobileNetV3(num_classes=43)

#init the wandb logging
wandb.watch(model, log_freq=100)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# Train the model
losses = []
steps = 0
model.train
for epoch in range(num_epochs):
    print(f"epoch: {epoch+1}/{num_epochs}")
    running_loss = 0
    for i, (inputs, labels) in enumerate(trainloader, 0):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        wandb.log({"loss": running_loss})
    else:
        losses.append(running_loss/len(trainloader))
        steps += 1
        print(f"Training loss: {running_loss/len(trainloader)}")


# Use the plot function to draw a line plot
plt.plot(range(steps), losses)

# Add a title and axis labels
plt.title("Training Loss vs Training Steps")
plt.xlabel("Training Steps")
plt.ylabel("Training Loss")

# Save the plot
plt.savefig("reports/figures/lossV1.png")

torch.save(model.state_dict(), 'models/trained_modelV1.pt')

# # Test the model
model.eval()
with torch.no_grad():
     correct = 0
     total = 0
     for i, (inputs, labels) in enumerate(trainloader):
         outputs = model(inputs)
         _, predicted = torch.max(outputs.data, 1)
         total += labels.size(0)
         correct += (predicted == labels).sum().item()

# # Print the accuracy
print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))
